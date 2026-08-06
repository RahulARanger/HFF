"""FastAPI backend for the HFF-Net VTK.js viewer.

The API returns typed binary volume buffers rather than base64-encoded JSON.
This keeps browser transfers compact and lets VTK.js construct image data
without changing the repository's research preprocessing or model inference.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
import psutil
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from viewer_core import (
    BRATS_SEGMENTATION_COLORS,
    DISPLAY_MODALITIES,
    SEGMENTATION_LABELS,
    canonical_segmentation_labels,
    checkpoint_output_path,
    contrast_limits,
    discover_checkpoints,
    discover_subject_ids,
    find_frequency_file,
    find_scan_path,
    load_frequency_volume,
    load_volume,
    restrict_mask_to_scan_foreground,
    resolve_subject,
    subject_files,
    generate_output_segmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "result"
MONITOR_LOG_SUFFIX = "_resource_usage.jsonl"
MONITOR_SUMMARY_SUFFIX = "_resource_summary.json"
FOLD_PATTERN = re.compile(r"^fold_(\d+)$")


class GenerateRequest(BaseModel):
    checkpoint_id: str


class ViewerServer:
    def __init__(self, dataset_root: Path, results_root: Path) -> None:
        self.dataset_root = dataset_root.expanduser().resolve()
        self.results_root = results_root.expanduser().resolve()

    def subject(self, subject_id: str) -> Path:
        try:
            return resolve_subject(self.dataset_root, subject_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    def checkpoint(self, checkpoint_id: str) -> Path:
        """Resolve a checkpoint from the server-owned results directory."""
        candidate = (self.results_root / checkpoint_id).resolve()
        if self.results_root not in candidate.parents or candidate.suffix != ".pth":
            raise HTTPException(status_code=404, detail="Unknown checkpoint.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Unknown checkpoint.")
        return candidate

    def subject_ids(self) -> tuple[str, ...]:
        if not self.dataset_root.exists():
            return ()
        return discover_subject_ids(str(self.dataset_root))


def binary_volume_response(
    volume: np.ndarray,
    *,
    dtype: Literal["float32", "uint8"],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_range: tuple[float, float] | None = None,
) -> Response:
    if dtype == "float32":
        encoded = np.ascontiguousarray(volume, dtype=np.float32)
    else:
        encoded = np.ascontiguousarray(volume, dtype=np.uint8)

    headers = {
        "X-Shape": ",".join(str(axis) for axis in encoded.shape),
        "X-Dtype": dtype,
        # NumPy volumes are ordered z, y, x; SimpleITK spacing is x, y, z.
        "X-Spacing": ",".join(str(value) for value in reversed(spacing)),
        "Cache-Control": "no-store",
    }
    if intensity_range is not None:
        headers["X-Intensity-Range"] = ",".join(str(value) for value in intensity_range)
    return Response(
        content=encoded.tobytes(order="C"),
        media_type="application/octet-stream",
        headers=headers,
    )


def reference_spacing(path: Path) -> tuple[float, float, float]:
    return tuple(float(value) for value in sitk.ReadImage(str(path)).GetSpacing())


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


def monitor_record(log_path: Path, results_root: Path, sample_limit: int = 60) -> dict[str, object]:
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
    return {
        "id": log_path.relative_to(results_root).as_posix(),
        "label": f"{run_name} · {fold}",
        "run_name": run_name,
        "fold": fold,
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
        "latest": latest,
        "peak": {
            "ram_rss_bytes": peak_ram_rss,
            "ram_uss_bytes": peak_ram_uss,
            "gpu_memory_bytes": peak_gpu,
        },
    }


def create_app(dataset_root: Path = DEFAULT_DATA_ROOT, results_root: Path = DEFAULT_RESULTS_ROOT) -> FastAPI:
    state = ViewerServer(dataset_root, results_root)
    app = FastAPI(title="HFF-Net BraTS viewer API", version="1.0.0")
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

    @app.get("/api/subjects")
    def subjects() -> dict[str, object]:
        records = []
        for subject_id in state.subject_ids():
            subject_dir = state.subject(subject_id)
            scans, segmentation = subject_files(subject_dir)
            modalities = [
                path.name.rsplit("_", 1)[-1].split(".", 1)[0].upper()
                for path in scans
            ]
            records.append(
                {
                    "id": subject_id,
                    "label": subject_dir.name,
                    "modalities": modalities,
                    "has_segmentation": segmentation is not None,
                }
            )
        return {"subjects": records}

    @app.get("/api/checkpoints")
    def checkpoints() -> dict[str, object]:
        return {
            "checkpoints": [
                {
                    "id": checkpoint.relative_to(state.results_root).as_posix(),
                    "label": checkpoint.relative_to(state.results_root).as_posix(),
                    "modified": checkpoint.stat().st_mtime,
                }
                for checkpoint in discover_checkpoints(state.results_root)
            ]
        }

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
        record = monitor_record(candidate, state.results_root, sample_limit=limit)
        record["samples"] = read_jsonl_tail(candidate, limit)
        return record

    @app.get("/api/subjects/{subject_id:path}/metadata")
    def metadata(subject_id: str) -> dict[str, object]:
        subject_dir = state.subject(subject_id)
        scans, segmentation = subject_files(subject_dir)
        scan_metadata: dict[str, object] = {}
        for path in scans:
            modality = path.name.rsplit("_", 1)[-1].split(".", 1)[0].upper()
            actual = load_volume(path)
            available_bands = {
                band: find_frequency_file(path, band) is not None
                for band in ("L", "H1", "H2", "H3", "H4")
            }
            scan_metadata[modality] = {
                "shape": list(actual.shape),
                "spacing": list(reference_spacing(path)),
                "contrast_limits": list(contrast_limits(actual)),
                "frequency": available_bands,
            }

        mask_shape = None
        if segmentation is not None:
            mask_shape = list(load_volume(segmentation).shape)
        return {
            "id": subject_id,
            "label": subject_dir.name,
            "modalities": [modality for modality in DISPLAY_MODALITIES if modality in scan_metadata],
            "scans": scan_metadata,
            "segmentation": {"available": segmentation is not None, "shape": mask_shape},
        }

    @app.get("/api/subjects/{subject_id:path}/volumes/{modality}")
    def volume(subject_id: str, modality: str) -> Response:
        subject_dir = state.subject(subject_id)
        modality = modality.upper()
        if modality not in DISPLAY_MODALITIES:
            raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")
        scan_path = find_scan_path(subject_dir, modality)
        actual = load_volume(scan_path)
        return binary_volume_response(
            actual,
            dtype="float32",
            spacing=reference_spacing(scan_path),
            intensity_range=contrast_limits(actual),
        )

    @app.get("/api/subjects/{subject_id:path}/frequency/{modality}/{band}")
    def frequency(subject_id: str, modality: str, band: str) -> Response:
        subject_dir = state.subject(subject_id)
        modality = modality.upper()
        band = band.upper()
        if modality not in DISPLAY_MODALITIES or band not in {"L", "H1", "H2", "H3", "H4"}:
            raise HTTPException(status_code=400, detail="Unsupported frequency request.")
        scan_path = find_scan_path(subject_dir, modality)
        actual = load_volume(scan_path)
        loaded, available = load_frequency_volume(scan_path, band, actual)
        if not available:
            raise HTTPException(status_code=404, detail=f"Frequency volume {band} is not available.")
        loaded = loaded.astype(np.float32, copy=False)
        return binary_volume_response(
            loaded,
            dtype="float32",
            spacing=reference_spacing(scan_path),
            intensity_range=contrast_limits(loaded),
        )

    @app.get("/api/subjects/{subject_id:path}/masks/{mask_kind}")
    def mask(
        subject_id: str,
        mask_kind: Literal["expected", "output"],
        checkpoint_id: str | None = None,
    ) -> Response:
        subject_dir = state.subject(subject_id)
        flair_path = find_scan_path(subject_dir, "FLAIR")
        flair = load_volume(flair_path)

        if mask_kind == "expected":
            segmentation = subject_files(subject_dir)[1]
            if segmentation is None:
                raise HTTPException(status_code=404, detail="Expected segmentation is not available.")
            labels = canonical_segmentation_labels(load_volume(segmentation), flair.shape)
        else:
            if checkpoint_id is None:
                raise HTTPException(status_code=400, detail="checkpoint_id is required for output masks.")
            checkpoint = state.checkpoint(checkpoint_id)
            output_path = checkpoint_output_path(state.results_root, checkpoint, subject_dir)
            if not output_path.is_file():
                raise HTTPException(status_code=404, detail="Generated output is not available.")
            labels = load_volume(output_path).astype(np.uint8)
            labels = canonical_segmentation_labels(labels, flair.shape)
            labels = np.where(restrict_mask_to_scan_foreground(labels, flair) > 0, labels, 0).astype(np.uint8)

        return binary_volume_response(labels, dtype="uint8", spacing=reference_spacing(flair_path))

    @app.post("/api/subjects/{subject_id:path}/generate")
    def generate(subject_id: str, request: GenerateRequest) -> dict[str, str]:
        subject_dir = state.subject(subject_id)
        checkpoint = state.checkpoint(request.checkpoint_id)
        try:
            output_path = generate_output_segmentation(checkpoint, subject_dir, state.results_root)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "output_id": output_path.relative_to(state.results_root).as_posix(),
            "message": "Output segmentation generated.",
        }

    @app.get("/api/labels")
    def labels() -> dict[str, object]:
        return {
            "labels": [
                {
                    "value": value,
                    "name": name,
                    "color": list(BRATS_SEGMENTATION_COLORS.get(value, (0.0, 0.0, 0.0, 0.0))),
                }
                for value, name in SEGMENTATION_LABELS.items()
            ]
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args.dataset_root, args.results_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
