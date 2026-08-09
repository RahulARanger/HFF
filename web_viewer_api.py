"""FastAPI backend for the HFF-Net training monitor."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "result"
MONITOR_LOG_SUFFIX = "_resource_usage.jsonl"
MONITOR_SUMMARY_SUFFIX = "_resource_summary.json"
FOLD_PATTERN = re.compile(r"^fold_(\d+)$")


class ViewerServer:
    def __init__(self, results_root: Path) -> None:
        self.results_root = self._resolve_project_path(results_root)

    @staticmethod
    def _resolve_project_path(path: Path) -> Path:
        """Resolve relative CLI paths from the repository, not the shell cwd."""
        expanded = path.expanduser()
        return (PROJECT_ROOT / expanded if not expanded.is_absolute() else expanded).resolve()

def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, object]]:
    """Read only the newest monitor samples so polling stays cheap for long runs."""
    samples: deque[dict[str, object]] = deque(maxlen=limit)
    try:
        with path.open(encoding="utf-8") as monitor_file:
            for line in monitor_file:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(sample, dict):
                    samples.append(sample)
    except FileNotFoundError:
        return []
    return list(samples)


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def monitor_manifest(log_path: Path, results_root: Path) -> Path | None:
    for directory in (log_path.parent, *log_path.parents):
        candidate = directory / "cross_validation_manifest.json"
        if candidate.is_file() and results_root in candidate.parents:
            return candidate
    return None


def monitor_context(log_path: Path, results_root: Path) -> tuple[str, str, Path | None]:
    relative_parts = log_path.relative_to(results_root).parts
    fold_index = next(
        (match.group(1) for part in relative_parts if (match := FOLD_PATTERN.match(part))),
        "unknown",
    )
    fold_position = next(
        (index for index, part in enumerate(relative_parts) if FOLD_PATTERN.match(part)),
        None,
    )
    run_name = relative_parts[fold_position - 1] if fold_position and fold_position > 0 else "standalone"
    return run_name, f"fold_{fold_index}", monitor_manifest(log_path, results_root)


def monitor_value_peak(samples: list[dict[str, object]], field: str) -> int | None:
    values = [sample.get(field) for sample in samples]
    numeric = [int(value) for value in values if isinstance(value, (int, float))]
    return max(numeric) if numeric else None


def training_metrics_file(log_path: Path) -> Path | None:
    """Find the metrics written beside the checkpoint for this fold."""
    candidate = log_path.parent / "training_metrics.json"
    if candidate.is_file():
        return candidate
    return next(
        (path for path in log_path.parents if (path / "training_metrics.json").is_file()),
        None,
    )


def training_progress(
    log_path: Path,
    manifest: dict[str, object],
    *,
    include_history: bool = False,
) -> dict[str, object]:
    metrics_path = training_metrics_file(log_path)
    metrics = read_json(metrics_path) if metrics_path else {}
    history = metrics.get("epochs", [])
    if not isinstance(history, list):
        history = []
    configuration = metrics.get("configuration", {})
    if not isinstance(configuration, dict):
        configuration = {}
    total_epochs = configuration.get("num_epochs") or manifest.get("epochs_per_fold")
    total_epochs = int(total_epochs) if total_epochs is not None else None
    completed_epochs = len(history)
    pending_epochs = max(total_epochs - completed_epochs, 0) if total_epochs is not None else None
    progress_percent = (
        round((completed_epochs / total_epochs) * 100, 1)
        if total_epochs and total_epochs > 0
        else None
    )
    result: dict[str, object] = {
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "pending_epochs": pending_epochs,
        "progress_percent": progress_percent,
        "latest_epoch": history[-1] if history else None,
        "metrics_file": metrics_path.name if metrics_path else None,
    }
    if include_history:
        result["history"] = history
    return result


def monitor_record(
    log_path: Path,
    results_root: Path,
    sample_limit: int = 60,
    *,
    include_training_history: bool = False,
) -> dict[str, object]:
    summary_path = log_path.with_name(log_path.name.replace(MONITOR_LOG_SUFFIX, MONITOR_SUMMARY_SUFFIX))
    samples = read_jsonl_tail(log_path, sample_limit)
    summary = read_json(summary_path)
    latest = samples[-1] if samples else {}
    run_name, fold, manifest_path = monitor_context(log_path, results_root)
    manifest = read_json(manifest_path) if manifest_path else {}
    fold_entries = manifest.get("folds", [])
    fold_entry = (
        next((entry for entry in fold_entries if entry.get("fold") == int(fold[5:])), None)
        if fold != "fold_unknown" and isinstance(fold_entries, list)
        else None
    )
    interval = float(manifest.get("resource_monitor_interval_seconds", 5.0))
    modified_at = datetime.fromtimestamp(log_path.stat().st_mtime, timezone.utc)
    has_summary = bool(summary)
    recent_threshold = max(30.0, interval * 3.0)
    is_recent = (datetime.now(timezone.utc) - modified_at).total_seconds() <= recent_threshold
    root_pid = latest.get("root_pid") or summary.get("root_pid")
    process_visible = bool(root_pid and psutil.pid_exists(int(root_pid)))
    if has_summary:
        status = "completed"
    elif fold_entry and fold_entry.get("status") == "failed":
        status = "failed"
    elif is_recent or process_visible:
        status = "running"
    else:
        status = "stale"

    peak_ram_rss = summary.get("peak_ram_rss_bytes") or monitor_value_peak(samples, "ram_rss_bytes")
    peak_ram_uss = summary.get("peak_ram_uss_bytes") or monitor_value_peak(samples, "ram_uss_bytes")
    peak_gpu = summary.get("peak_gpu_memory_bytes") or monitor_value_peak(samples, "gpu_memory_bytes")
    cpu_values = [
        float(sample["cpu_utilization_percent"])
        for sample in samples
        if isinstance(sample.get("cpu_utilization_percent"), (int, float))
    ]
    peak_cpu = summary.get("peak_cpu_utilization_percent") or (max(cpu_values) if cpu_values else None)
    progress = training_progress(log_path, manifest, include_history=include_training_history)
    fold_started_at = (
        fold_entry.get("started_at_utc") if fold_entry else None
    ) or summary.get("started_at_utc") or (samples[0].get("timestamp_utc") if samples else None)
    fold_completed_at = (
        fold_entry.get("completed_at_utc") if fold_entry else None
    ) or summary.get("completed_at_utc")
    if fold_completed_at is None and has_summary:
        fold_completed_at = datetime.fromtimestamp(summary_path.stat().st_mtime, timezone.utc).isoformat()
    group_id = manifest_path.relative_to(results_root).as_posix() if manifest_path else log_path.relative_to(results_root).as_posix()
    group_started_at = manifest.get("created_at_utc") or fold_started_at
    group_completed_at = manifest.get("completed_at_utc")
    if group_completed_at is None and isinstance(fold_entries, list) and fold_entries and all(
        isinstance(entry, dict) and entry.get("status") == "completed" for entry in fold_entries
    ):
        completed_times = [entry.get("completed_at_utc") for entry in fold_entries if entry.get("completed_at_utc")]
        group_completed_at = max(completed_times) if completed_times else fold_completed_at
    return {
        "id": log_path.relative_to(results_root).as_posix(),
        "label": f"{run_name} · {fold}",
        "run_name": run_name,
        "fold": fold,
        "group_id": group_id,
        "group_label": run_name,
        "group_status": manifest.get("status") if manifest else status,
        "group_started_at": group_started_at,
        "group_completed_at": group_completed_at,
        "started_at": fold_started_at,
        "completed_at": fold_completed_at,
        "backend": latest.get("backend") or summary.get("backend") or "unknown",
        "status": status,
        "updated_at": latest.get("timestamp_utc") or modified_at.isoformat(),
        "sample_count": int(summary.get("sample_count") or len(samples)),
        "interval_seconds": interval,
        "root_pid": root_pid,
        "process_visible": process_visible,
        "resource_log": log_path.relative_to(results_root).as_posix(),
        "summary_file": summary_path.relative_to(results_root).as_posix() if summary_path.is_file() else None,
        "manifest_file": manifest_path.relative_to(results_root).as_posix() if manifest_path else None,
        "training": progress,
        "latest": latest,
        "peak": {
            "ram_rss_bytes": peak_ram_rss,
            "ram_uss_bytes": peak_ram_uss,
            "gpu_memory_bytes": peak_gpu,
            "cpu_utilization_percent": peak_cpu,
        },
    }


def create_app(results_root: Path = DEFAULT_RESULTS_ROOT) -> FastAPI:
    state = ViewerServer(results_root)
    app = FastAPI(title="HFF-Net training monitor API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Shape", "X-Dtype", "X-Spacing", "X-Intensity-Range"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/monitor/runs")
    def monitor_runs() -> dict[str, object]:
        if not state.results_root.exists():
            return {"runs": [], "updated_at": datetime.now(timezone.utc).isoformat()}
        records = [
            monitor_record(path, state.results_root)
            for path in state.results_root.rglob(f"*{MONITOR_LOG_SUFFIX}")
            if path.is_file()
        ]
        records.sort(key=lambda record: (record["status"] != "running", record["updated_at"]),)
        return {"runs": records, "updated_at": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/monitor/runs/{run_id:path}")
    def monitor_run(run_id: str, limit: int = 240) -> dict[str, object]:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 1000.")
        candidate = (state.results_root / run_id).resolve()
        if state.results_root not in candidate.parents or not candidate.name.endswith(MONITOR_LOG_SUFFIX):
            raise HTTPException(status_code=404, detail="Unknown monitor run.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Unknown monitor run.")
        record = monitor_record(
            candidate,
            state.results_root,
            sample_limit=limit,
            include_training_history=True,
        )
        record["samples"] = read_jsonl_tail(candidate, limit)
        return record

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args.results_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
