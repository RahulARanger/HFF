"""FastAPI backend for the HFF-Net training monitor."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import traceback
import uuid
from typing import Literal

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.resource_monitor import ResourceMonitor
from utils.utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("hff.viewer")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "result"
MONITOR_LOG_SUFFIX = "_resource_usage.jsonl"
MONITOR_SUMMARY_SUFFIX = "_resource_summary.json"
FOLD_PATTERN = re.compile(r"^fold_(\d+)$")
EVAL_JOB_LOG_SUFFIX = ".log"
CHECKPOINT_SCORE_PATTERN = re.compile(
    r"^best_(?P<result>Result[12])_et_(?P<et>\d+(?:\.\d+)?)"
    r"_tc_(?P<tc>\d+(?:\.\d+)?)_wt_(?P<wt>\d+(?:\.\d+)?)\.pth$",
    re.IGNORECASE,
)
CHECKPOINT_LAST_SAVE_PATTERN = re.compile(
    r"^best_Jc_(?P<jc>\d+(?:\.\d+)?)\.pth$",
    re.IGNORECASE,
)


class EvaluationRequest(BaseModel):
    """Validated inputs accepted by the web evaluation form."""

    name: str | None = Field(default=None, max_length=120)
    training_run: str | None = Field(default=None, max_length=240)
    fold: str | None = Field(default=None, max_length=64)
    checkpoints: list[str] = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    test_list: str = Field(min_length=1)
    dataset_name: Literal["brats19", "brats20", "brats23men", "msdbts"] = "brats19"
    class_type: Literal["all"] = "all"
    batch_size: int = Field(default=1, ge=1, le=32)
    num_workers: int = Field(default=3, ge=0, le=64)
    resource_monitor_interval: float = Field(default=5.0, gt=0.0, le=300.0)


class EvaluationRenameRequest(BaseModel):
    """Optional human-readable label for an existing evaluation job."""

    name: str | None = Field(default=None, max_length=120)


def normalize_evaluation_name(name: str | None) -> str | None:
    """Treat blank labels as unset while keeping the generated job ID intact."""
    normalized = name.strip() if isinstance(name, str) else ""
    return normalized or None


class ViewerServer:
    def __init__(self, results_root: Path) -> None:
        self.results_root = self._resolve_project_path(results_root)
        self.eval_jobs: dict[str, dict[str, object]] = {}
        self.deleted_validation_runs: set[str] = set()
        self.eval_lock = threading.Lock()
        self.active_eval_job: str | None = None

    @staticmethod
    def _resolve_project_path(path: Path) -> Path:
        """Resolve relative CLI paths from the repository, not the shell cwd."""
        expanded = path.expanduser()
        return (PROJECT_ROOT / expanded if not expanded.is_absolute() else expanded).resolve()

    def evaluation_options(self) -> dict[str, object]:
        """Return discoverable checkpoints and test manifests for the web form."""
        checkpoints = []
        checkpoint_groups: dict[str, dict[str, object]] = {}
        cross_validation_root = self.results_root / "cross_validation"
        if cross_validation_root.exists():
            for path in cross_validation_root.rglob("*.pth"):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(cross_validation_root)
                if len(relative_path.parts) < 2:
                    continue
                run_name = relative_path.parts[0]
                checkpoint = self._checkpoint_option(path, run_name, relative_path)
                checkpoints.append(checkpoint)
                group = checkpoint_groups.setdefault(
                    run_name,
                    {
                        "name": run_name,
                        "label": run_name,
                        "checkpoints": [],
                        "latest_modified_at": 0.0,
                    },
                )
                group["checkpoints"].append(checkpoint)
                group["latest_modified_at"] = max(
                    float(group["latest_modified_at"]),
                    float(checkpoint["modified_at"]),
                )

        checkpoints.sort(key=lambda item: (item["modified_at"], item["path"]), reverse=True)
        groups = sorted(
            checkpoint_groups.values(),
            key=lambda group: (group["latest_modified_at"], group["name"]),
            reverse=True,
        )
        for group in groups:
            group["checkpoints"].sort(
                key=lambda item: (item["modified_at"], item["path"]),
                reverse=True,
            )

        dataset_root = PROJECT_ROOT / "dataset"
        test_lists = sorted(
            (path for path in dataset_root.rglob("*.txt") if path.is_file()),
            key=lambda path: (path.name != "testing.txt", str(path)),
        ) if dataset_root.exists() else []

        def option(path: Path) -> dict[str, str]:
            try:
                label = path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                label = str(path)
            return {"path": str(path), "label": label}

        default_test_list = next(
            (path for path in test_lists if path.name == "testing.txt"),
            test_lists[0] if test_lists else None,
        )
        return {
            "checkpoints": checkpoints,
            "checkpoint_groups": groups,
            "test_lists": [option(path) for path in test_lists],
            "defaults": {
                "output_dir": str(self.results_root / "cross_eval"),
                "test_list": str(default_test_list) if default_test_list else "",
            },
        }

    @staticmethod
    def _checkpoint_option(
        path: Path,
        run_name: str,
        relative_path: Path,
    ) -> dict[str, object]:
        """Describe a checkpoint without changing the path sent to evaluation."""
        modified_at = path.stat().st_mtime
        result_match = CHECKPOINT_SCORE_PATTERN.match(path.name)
        last_save_match = CHECKPOINT_LAST_SAVE_PATTERN.match(path.name)
        is_last_save = last_save_match is not None
        scores: dict[str, float | None] = {"et": None, "tc": None, "wt": None}
        result_name = "Last save" if is_last_save else path.stem
        if result_match:
            result_name = f"best_{result_match.group('result')}"
            scores = {
                "et": float(result_match.group("et")),
                "tc": float(result_match.group("tc")),
                "wt": float(result_match.group("wt")),
            }

        average_dice = None
        available_scores = [score for score in scores.values() if score is not None]
        if len(available_scores) == 3:
            average_dice = sum(available_scores) / 3

        fold_name = next(
            (part for part in relative_path.parts if FOLD_PATTERN.match(part)),
            None,
        )
        return {
            "path": str(path),
            "name": result_name,
            "run_name": run_name,
            "fold_name": fold_name,
            "filename": path.name,
            "is_last_save": is_last_save,
            "scores": scores,
            "average_dice": average_dice,
            "last_save_metric": float(last_save_match.group("jc")) if last_save_match else None,
            "modified_at": modified_at,
        }

    def _job_log_tail(self, job: dict[str, object], limit: int = 6000) -> str:
        log_path = job.get("log_file")
        if not isinstance(log_path, str):
            return ""
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    @staticmethod
    def _write_eval_log_header(
        log_path: Path,
        job_id: str,
        command: list[str],
        request: EvaluationRequest,
    ) -> None:
        """Start a self-contained evaluation log that is useful when sharing failures."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("HFF-Net evaluation backend log\n")
            log_file.write(f"job_id: {job_id}\n")
            log_file.write(f"created_at: {datetime.now(timezone.utc).isoformat()}\n")
            log_file.write(f"command: {shlex.join(command)}\n")
            log_file.write("request:\n")
            log_file.write(json.dumps(request.model_dump(), indent=2, default=str))
            log_file.write("\n\n--- cross_eval.py output ---\n")

    @staticmethod
    def _append_eval_log(log_path: Path, message: str) -> None:
        """Append backend diagnostics without replacing the child-process output."""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n{message.rstrip()}\n")
        except OSError:
            LOGGER.exception("Could not append evaluation diagnostics to %s", log_path)

    def _write_eval_manifest(self, manifest_path: Path, job_id: str) -> None:
        """Persist enough metadata to rediscover validation telemetry after restart."""
        with self.eval_lock:
            payload = dict(self.eval_jobs[job_id])
        payload.pop("log_tail", None)
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def public_eval_job(self, job_id: str) -> dict[str, object]:
        with self.eval_lock:
            job = dict(self.eval_jobs[job_id])
        job["log_tail"] = self._job_log_tail(job)
        progress_path = job.get("progress_file")
        if isinstance(progress_path, str):
            progress = read_json(Path(progress_path))
            if progress:
                job["progress"] = progress
        summary_path = job.get("summary_file")
        if job.get("status") == "completed" and isinstance(summary_path, str) and Path(summary_path).is_file():
            job["summary"] = read_json(Path(summary_path))
        return job

    def rename_eval_job(self, job_id: str, name: str | None) -> dict[str, object]:
        """Update a display label without changing the internal job identifier."""
        with self.eval_lock:
            if job_id not in self.eval_jobs:
                raise KeyError(job_id)
            self.eval_jobs[job_id]["name"] = name
            request = self.eval_jobs[job_id].get("request")
            if isinstance(request, dict):
                request["name"] = name
            manifest_path = self.eval_jobs[job_id].get("manifest_file")

        # A queued job may not have created its manifest yet; the worker will
        # persist the latest label when it creates one.
        if isinstance(manifest_path, str) and Path(manifest_path).parent.exists():
            self._write_eval_manifest(Path(manifest_path), job_id)
        return self.public_eval_job(job_id)

    def _run_evaluation(self, job_id: str, request: EvaluationRequest) -> None:
        output_dir = self._resolve_project_path(Path(request.output_dir))
        checkpoint_list_path = output_dir / f"checkpoint_list_{job_id}.txt"
        log_path = output_dir / f"cross_eval_{job_id}{EVAL_JOB_LOG_SUFFIX}"
        progress_path = output_dir / f"cross_eval_progress_{job_id}.json"
        summary_path = output_dir / "cross_eval_summary.json"
        manifest_path = output_dir / f"validation_job_{job_id}.json"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_list_path.write_text(
                "\n".join(request.checkpoints) + "\n", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "cross_eval.py"),
                "--checkpoint_list", str(checkpoint_list_path),
                "--test_list", str(self._resolve_project_path(Path(request.test_list))),
                "--dataset_name", request.dataset_name,
                "--class_type", request.class_type,
                "--batch_size", str(request.batch_size),
                "--num_workers", str(request.num_workers),
                "--output_dir", str(output_dir),
                "--progress_file", str(progress_path),
            ]
            self._write_eval_log_header(log_path, job_id, command, request)
        except OSError as exc:
            self._append_eval_log(
                log_path,
                "--- backend startup failure ---\n"
                f"exception: {type(exc).__name__}: {exc}\n"
                f"traceback:\n{traceback.format_exc()}",
            )
            LOGGER.exception("Evaluation %s could not be prepared; log=%s", job_id, log_path)
            with self.eval_lock:
                self.eval_jobs[job_id].update({
                    "status": "failed",
                    "return_code": -1,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "log_file": str(log_path),
                    "progress_file": str(progress_path),
                    "summary_file": str(summary_path),
                    "checkpoint_list_file": str(checkpoint_list_path),
                    "manifest_file": str(manifest_path),
                })
                if self.active_eval_job == job_id:
                    self.active_eval_job = None
            return

        with self.eval_lock:
            self.eval_jobs[job_id].update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "pid": None,
                "command": command,
                "log_file": str(log_path),
                "progress_file": str(progress_path),
                "summary_file": str(summary_path),
                "checkpoint_list_file": str(checkpoint_list_path),
                "manifest_file": str(manifest_path),
                "resource_monitor_interval": request.resource_monitor_interval,
            })
        self._write_eval_manifest(manifest_path, job_id)
        LOGGER.info("Evaluation %s started; log=%s", job_id, log_path)

        return_code = -1
        resource_monitor: ResourceMonitor | None = None
        monitor_error: str | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self.eval_lock:
                    self.eval_jobs[job_id]["pid"] = process.pid
                    self.eval_jobs[job_id]["resource_monitoring"] = "starting"
                self._write_eval_manifest(manifest_path, job_id)
                try:
                    resource_monitor = ResourceMonitor(
                        device=get_device(),
                        output_directory=output_dir,
                        interval_seconds=request.resource_monitor_interval,
                        root_pid=process.pid,
                    ).start()
                    with self.eval_lock:
                        self.eval_jobs[job_id]["resource_monitoring"] = "enabled"
                        self.eval_jobs[job_id]["resource_backend"] = resource_monitor.backend
                        self.eval_jobs[job_id]["resource_log"] = str(resource_monitor.output_path)
                        self.eval_jobs[job_id]["resource_summary_file"] = str(resource_monitor.summary_path)
                except Exception as exc:  # Monitoring must never prevent evaluation.
                    monitor_error = str(exc)
                    with self.eval_lock:
                        self.eval_jobs[job_id]["resource_monitoring"] = "unavailable"
                        self.eval_jobs[job_id]["resource_monitor_error"] = monitor_error
                self._write_eval_manifest(manifest_path, job_id)
                return_code = process.wait()
            status = "completed" if return_code == 0 else "failed"
            error = None if return_code == 0 else f"cross_eval.py exited with code {return_code}."
            if return_code != 0:
                self._append_eval_log(
                    log_path,
                    "--- evaluation failure ---\n"
                    f"return_code: {return_code}\n"
                    f"message: {error}",
                )
        except OSError as exc:
            status = "failed"
            error = str(exc)
            self._append_eval_log(
                log_path,
                "--- backend process failure ---\n"
                f"exception: {type(exc).__name__}: {exc}\n"
                f"traceback:\n{traceback.format_exc()}",
            )
            LOGGER.exception("Evaluation %s failed before completion; log=%s", job_id, log_path)
        finally:
            if resource_monitor is not None:
                try:
                    resource_monitor.stop()
                except Exception as exc:  # Persist the job even if final telemetry flush fails.
                    monitor_error = str(exc)

        with self.eval_lock:
            self.eval_jobs[job_id].update({
                "status": status,
                "return_code": return_code,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
                "log_file": str(log_path),
                "progress_file": str(progress_path),
                "summary_file": str(summary_path),
                "checkpoint_list_file": str(checkpoint_list_path),
                "manifest_file": str(manifest_path),
                "resource_monitoring": "enabled" if resource_monitor is not None else "unavailable",
                "resource_monitor_error": monitor_error,
            })
            if resource_monitor is not None:
                self.eval_jobs[job_id]["resource_log"] = str(resource_monitor.output_path)
                self.eval_jobs[job_id]["resource_summary_file"] = str(resource_monitor.summary_path)
            if self.active_eval_job == job_id:
                self.active_eval_job = None
        self._write_eval_manifest(manifest_path, job_id)
        if status == "failed":
            LOGGER.error(
                "Evaluation %s failed (return_code=%s); log=%s",
                job_id,
                return_code,
                log_path,
            )
        else:
            LOGGER.info("Evaluation %s completed; log=%s", job_id, log_path)

    def _validation_record(
        self,
        job_id: str,
        log_path: Path,
        job: dict[str, object],
        *,
        include_samples: bool = False,
    ) -> dict[str, object]:
        """Build a validation-only resource record for the monitor view."""
        resource_log_value = job.get("resource_log")
        if isinstance(resource_log_value, str):
            log_path = Path(resource_log_value)
        summary_value = job.get("resource_summary_file")
        if isinstance(summary_value, str):
            summary_path = Path(summary_value)
        elif log_path.name.endswith(MONITOR_LOG_SUFFIX):
            summary_path = log_path.with_name(
                log_path.name.replace(MONITOR_LOG_SUFFIX, MONITOR_SUMMARY_SUFFIX)
            )
        else:
            summary_path = log_path.with_name("resource_summary_unavailable.json")
        samples = read_jsonl_tail(log_path, 240 if include_samples else 60)
        summary = read_json(summary_path)
        latest = samples[-1] if samples else {}
        request = job.get("request", {}) if isinstance(job.get("request"), dict) else {}
        evaluation_summary = {}
        evaluation_summary_value = job.get("summary_file")
        if isinstance(evaluation_summary_value, str):
            evaluation_summary = read_json(Path(evaluation_summary_value))
        persisted_status = job.get("status")
        status = str(persisted_status or ("completed" if summary else "stale"))
        started_at = job.get("started_at") or summary.get("started_at_utc") or (
            samples[0].get("timestamp_utc") if samples else None
        )
        completed_at = job.get("completed_at") or summary.get("completed_at_utc")
        root_pid = latest.get("root_pid") or summary.get("root_pid") or job.get("pid")
        process_visible = bool(root_pid and psutil.pid_exists(int(root_pid)))
        modified_at = log_path.stat().st_mtime if log_path.is_file() else None
        updated_at = latest.get("timestamp_utc") or (
            datetime.fromtimestamp(modified_at, timezone.utc).isoformat()
            if modified_at else job.get("created_at")
        )
        elapsed = seconds_between(
            started_at if isinstance(started_at, str) else None,
            completed_at if isinstance(completed_at, str) else datetime.now(timezone.utc).isoformat(),
        )
        peak_cpu_values = [
            float(sample["cpu_utilization_percent"])
            for sample in samples
            if isinstance(sample.get("cpu_utilization_percent"), (int, float))
        ]
        record: dict[str, object] = {
            "id": job_id,
            "label": str(job.get("name") or f"Validation · {job_id}"),
            "status": status,
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "completed_at": completed_at,
            "updated_at": updated_at,
            "timing": {
                "started_at": started_at,
                "ended_at": completed_at,
                "elapsed_seconds": elapsed,
                "elapsed_display": timing_display(elapsed),
            },
            "backend": latest.get("backend") or summary.get("backend") or job.get("resource_backend") or "unknown",
            "resource_monitoring": job.get("resource_monitoring") or ("completed" if summary else "unknown"),
            "resource_monitor_error": job.get("resource_monitor_error"),
            "sample_count": int(summary.get("sample_count") or len(samples)),
            "interval_seconds": request.get("resource_monitor_interval") or job.get("resource_monitor_interval") or 5.0,
            "root_pid": root_pid,
            "process_visible": process_visible,
            "resource_log": str(log_path) if log_path.name.endswith(MONITOR_LOG_SUFFIX) else None,
            "summary_file": str(summary_path) if summary_path.is_file() else None,
            "manifest_file": job.get("manifest_file"),
            "request": request,
            "evaluation_summary": evaluation_summary,
            "latest": latest,
            "peak": {
                "ram_rss_bytes": summary.get("peak_ram_rss_bytes") or monitor_value_peak(samples, "ram_rss_bytes"),
                "ram_uss_bytes": summary.get("peak_ram_uss_bytes") or monitor_value_peak(samples, "ram_uss_bytes"),
                "gpu_memory_bytes": summary.get("peak_gpu_memory_bytes") or monitor_value_peak(samples, "gpu_memory_bytes"),
                "cpu_utilization_percent": summary.get("peak_cpu_utilization_percent") or (max(peak_cpu_values) if peak_cpu_values else None),
            },
        }
        if include_samples:
            record["samples"] = samples
        return record

    def validation_sources(self) -> dict[str, tuple[Path, dict[str, object]]]:
        """Collect active jobs and persisted web-validation manifests."""
        sources: dict[str, tuple[Path, dict[str, object]]] = {}
        with self.eval_lock:
            jobs = {job_id: dict(job) for job_id, job in self.eval_jobs.items()}
            deleted_runs = set(self.deleted_validation_runs)
        for job_id, job in jobs.items():
            if job_id in deleted_runs:
                continue
            log_path = job.get("resource_log") or job.get("log_file")
            if isinstance(log_path, str):
                sources[job_id] = (Path(log_path), job)

        if self.results_root.exists():
            for manifest_path in self.results_root.rglob("validation_job_*.json"):
                payload = read_json(manifest_path)
                job_id = payload.get("id")
                log_path = payload.get("resource_log") or payload.get("log_file")
                if (
                    isinstance(job_id, str)
                    and job_id not in deleted_runs
                    and isinstance(log_path, str)
                    and job_id not in sources
                ):
                    payload["manifest_file"] = str(manifest_path)
                    sources[job_id] = (Path(log_path), payload)
        return sources

    def delete_validation_run(self, run_id: str) -> dict[str, object]:
        """Delete monitor telemetry for a finished validation without touching results."""
        source = self.validation_sources().get(run_id)
        if source is None:
            raise KeyError(run_id)
        log_path, job = source
        if job.get("status") in {"queued", "running"}:
            raise RuntimeError("Stop the running evaluation before deleting its validation telemetry.")

        artifact_paths: list[Path] = []
        manifest_value = job.get("manifest_file")
        if isinstance(manifest_value, str):
            artifact_paths.append(Path(manifest_value))
        resource_log_value = job.get("resource_log")
        if isinstance(resource_log_value, str):
            artifact_paths.append(Path(resource_log_value))
        if log_path.name.endswith(MONITOR_LOG_SUFFIX):
            artifact_paths.append(log_path)
            artifact_paths.append(
                log_path.with_name(log_path.name.replace(MONITOR_LOG_SUFFIX, MONITOR_SUMMARY_SUFFIX))
            )
        summary_value = job.get("resource_summary_file")
        if isinstance(summary_value, str):
            artifact_paths.append(Path(summary_value))

        unique_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for path in artifact_paths:
            resolved_path = path.expanduser().resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            if self.results_root not in resolved_path.parents:
                continue
            if not (
                resolved_path.name == f"validation_job_{run_id}.json"
                or resolved_path.name.endswith(MONITOR_LOG_SUFFIX)
                or resolved_path.name.endswith(MONITOR_SUMMARY_SUFFIX)
            ):
                continue
            unique_paths.append(resolved_path)

        for path in unique_paths:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                raise

        with self.eval_lock:
            self.deleted_validation_runs.add(run_id)
        return {"id": run_id, "deleted": True}

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


def is_validation_resource_log(path: Path) -> bool:
    """Keep web-triggered validation telemetry out of the training monitor."""
    return any(path.parent.glob("validation_job_*.json"))


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


def seconds_between(start: str | None, end: str | None) -> float | None:
    """Return an ISO timestamp delta when both timestamps are valid."""
    if not start or not end:
        return None
    try:
        delta = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
    except (TypeError, ValueError):
        return None
    return max(delta, 0.0)


def timing_display(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


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
    now_iso = datetime.now(timezone.utc).isoformat()
    elapsed_seconds = seconds_between(fold_started_at, fold_completed_at or now_iso)
    completed_progress = progress.get("progress_percent")
    estimated_total_seconds: float | None = None
    estimated_remaining_seconds: float | None = None
    history = progress.get("history", [])
    if isinstance(history, list):
        epoch_durations = [
            float(row["epoch_seconds"])
            for row in history
            if isinstance(row, dict) and isinstance(row.get("epoch_seconds"), (int, float)) and float(row["epoch_seconds"]) > 0
        ]
        pending_epochs = progress.get("pending_epochs")
        if epoch_durations and isinstance(pending_epochs, int) and pending_epochs > 0:
            estimated_remaining_seconds = sum(epoch_durations[-5:]) / min(len(epoch_durations), 5) * pending_epochs
            estimated_total_seconds = (elapsed_seconds or 0.0) + estimated_remaining_seconds
    if estimated_remaining_seconds is None and not has_summary and isinstance(completed_progress, (int, float)) and completed_progress > 0 and elapsed_seconds is not None:
        estimated_total_seconds = elapsed_seconds / (float(completed_progress) / 100.0)
        estimated_remaining_seconds = max(estimated_total_seconds - elapsed_seconds, 0.0)
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
        "timing": {
            "started_at": fold_started_at,
            "ended_at": fold_completed_at,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_display": timing_display(elapsed_seconds),
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total_display": timing_display(estimated_total_seconds),
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "estimated_remaining_display": timing_display(estimated_remaining_seconds),
            "estimate_source": "recent epoch durations" if isinstance(history, list) and history and estimated_remaining_seconds is not None and any(isinstance(row, dict) and isinstance(row.get("epoch_seconds"), (int, float)) for row in history) else "progress rate",
        },
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

    @app.get("/api/eval/options")
    def evaluation_options() -> dict[str, object]:
        return state.evaluation_options()

    @app.get("/api/eval/jobs")
    def evaluation_jobs() -> dict[str, object]:
        with state.eval_lock:
            job_ids = sorted(
                state.eval_jobs,
                key=lambda job_id: state.eval_jobs[job_id].get("created_at", ""),
                reverse=True,
            )[:20]
        return {
            "jobs": [state.public_eval_job(job_id) for job_id in job_ids],
            "active_job_id": state.active_eval_job,
        }

    @app.get("/api/eval/jobs/{job_id}")
    def evaluation_job(job_id: str) -> dict[str, object]:
        with state.eval_lock:
            if job_id not in state.eval_jobs:
                raise HTTPException(status_code=404, detail="Unknown evaluation job.")
        return state.public_eval_job(job_id)

    @app.post("/api/eval/jobs", status_code=202)
    def start_evaluation(request: EvaluationRequest) -> dict[str, object]:
        resolved_checkpoints = []
        for checkpoint_string in request.checkpoints:
            checkpoint = state._resolve_project_path(Path(checkpoint_string))
            if not checkpoint.is_file() or checkpoint.suffix != ".pth":
                raise HTTPException(status_code=400, detail=f"Invalid checkpoint: {checkpoint}")
            resolved_checkpoints.append(str(checkpoint))

        test_list = state._resolve_project_path(Path(request.test_list))
        if not test_list.is_file():
            raise HTTPException(status_code=400, detail=f"Test list does not exist: {test_list}")

        output_dir = state._resolve_project_path(Path(request.output_dir))
        job_id = uuid.uuid4().hex[:12]
        with state.eval_lock:
            if state.active_eval_job is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Evaluation job {state.active_eval_job} is already running.",
                )
            state.active_eval_job = job_id
            evaluation_name = normalize_evaluation_name(request.name)
            state.eval_jobs[job_id] = {
                "id": job_id,
                "name": evaluation_name,
                "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pid": None,
                "request": {
                    "name": evaluation_name,
                    "training_run": request.training_run,
                    "fold": request.fold,
                    "checkpoints": resolved_checkpoints,
                    "output_dir": str(output_dir),
                    "test_list": str(test_list),
                    "dataset_name": request.dataset_name,
                    "class_type": request.class_type,
                    "batch_size": request.batch_size,
                    "num_workers": request.num_workers,
                    "resource_monitor_interval": request.resource_monitor_interval,
                },
            }

        normalized_request = request.model_copy(update={
            "name": evaluation_name,
            "checkpoints": resolved_checkpoints,
            "output_dir": str(output_dir),
            "test_list": str(test_list),
        })
        threading.Thread(
            target=state._run_evaluation,
            args=(job_id, normalized_request),
            name=f"hff-eval-{job_id}",
            daemon=True,
        ).start()
        return state.public_eval_job(job_id)

    @app.patch("/api/eval/jobs/{job_id}")
    def rename_evaluation(job_id: str, request: EvaluationRenameRequest) -> dict[str, object]:
        name = normalize_evaluation_name(request.name)
        try:
            return state.rename_eval_job(job_id, name)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown evaluation job.") from None

    @app.get("/api/validation/runs")
    def validation_runs() -> dict[str, object]:
        records = [
            state._validation_record(job_id, log_path, job)
            for job_id, (log_path, job) in state.validation_sources().items()
        ]
        records.sort(key=lambda record: record.get("created_at") or "", reverse=True)
        return {"runs": records, "updated_at": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/validation/runs/{run_id}")
    def validation_run(run_id: str) -> dict[str, object]:
        source = state.validation_sources().get(run_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Unknown validation run.")
        log_path, job = source
        return state._validation_record(run_id, log_path, job, include_samples=True)

    @app.delete("/api/validation/runs/{run_id}")
    def delete_validation_run(run_id: str) -> dict[str, object]:
        try:
            return state.delete_validation_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown validation run.") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not delete validation telemetry: {exc}") from None

    @app.get("/api/monitor/runs")
    def monitor_runs() -> dict[str, object]:
        if not state.results_root.exists():
            return {"runs": [], "updated_at": datetime.now(timezone.utc).isoformat()}
        records = [
            monitor_record(path, state.results_root)
            for path in state.results_root.rglob(f"*{MONITOR_LOG_SUFFIX}")
            if path.is_file() and not is_validation_resource_log(path)
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    uvicorn.run(create_app(args.results_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
