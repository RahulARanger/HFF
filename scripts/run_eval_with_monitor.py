#!/usr/bin/env python3
"""Run one PBS evaluation while publishing monitor-compatible telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.resource_monitor import ResourceMonitor  # noqa: E402
from utils.utils import get_device  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def option_value(arguments: list[str], name: str, default: str | None = None) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = f"{name}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return default


def replace_option(arguments: list[str], name: str, value: str) -> list[str]:
    """Replace an option value while preserving the cross_eval argument order."""
    updated: list[str] = []
    index = 0
    replaced = False
    while index < len(arguments):
        argument = arguments[index]
        if argument == name:
            updated.extend((name, value))
            index += 2
            replaced = True
            continue
        if argument.startswith(f"{name}="):
            updated.append(f"{name}={value}")
            index += 1
            replaced = True
            continue
        updated.append(argument)
        index += 1
    if not replaced:
        updated.extend((name, value))
    return updated


def resolve_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return (base / path if not path.is_absolute() else path).resolve()


def checkpoint_paths(checkpoint_list: Path | None) -> list[str]:
    if checkpoint_list is None or not checkpoint_list.is_file():
        return []
    paths = [
        line.strip()
        for line in checkpoint_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return [
        str((checkpoint_list.parent / path).resolve())
        if not Path(path).expanduser().is_absolute()
        else str(Path(path).expanduser().resolve())
        for path in paths
    ]


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n{message.rstrip()}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--resource-monitor-interval", type=float, default=5.0)
    parser.add_argument("cross_eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cross_eval_args and args.cross_eval_args[0] == "--":
        args.cross_eval_args = args.cross_eval_args[1:]
    if not args.cross_eval_args:
        parser.error("cross_eval.py arguments are required after --")
    if args.resource_monitor_interval <= 0:
        parser.error("--resource-monitor-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    cross_eval_args = list(args.cross_eval_args)
    requested_output_dir = resolve_path(option_value(cross_eval_args, "--output_dir"), repo_root)
    if requested_output_dir is None:
        raise ValueError("PBS evaluation requires --output_dir so monitor telemetry has a stable location.")

    job_id = args.job_id.replace("/", "_")
    job_name = args.job_name.strip() or f"PBS evaluation {job_id}"
    requested_checkpoint_list = resolve_path(option_value(cross_eval_args, "--checkpoint_list"), repo_root)
    test_list = resolve_path(option_value(cross_eval_args, "--test_list"), repo_root)
    output_dir = requested_output_dir / f"pbs_eval_{job_id}"
    requested_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_list = output_dir / f"checkpoint_list_{job_id}.txt"
    checkpoint_list.write_text("\n".join(checkpoint_paths(requested_checkpoint_list)) + "\n", encoding="utf-8")
    progress_file = output_dir / f"cross_eval_progress_{job_id}.json"
    cross_eval_args = replace_option(cross_eval_args, "--checkpoint_list", str(checkpoint_list))
    cross_eval_args = replace_option(cross_eval_args, "--output_dir", str(output_dir))
    cross_eval_args = replace_option(cross_eval_args, "--progress_file", str(progress_file))
    summary_file = output_dir / "cross_eval_summary.json"
    manifest_file = requested_output_dir / f"validation_job_{job_id}.json"
    log_file = output_dir / f"cross_eval_{job_id}.log"
    telemetry_dir = output_dir / f"validation_monitor_{job_id}"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(repo_root / "cross_eval.py"), *cross_eval_args]
    request = {
        "name": job_name,
        "training_run": None,
        "fold": None,
        "checkpoints": checkpoint_paths(checkpoint_list),
        "output_dir": str(output_dir),
        "requested_output_dir": str(requested_output_dir),
        "test_list": str(test_list) if test_list else None,
        "dataset_name": option_value(cross_eval_args, "--dataset_name", "brats19"),
        "class_type": option_value(cross_eval_args, "--class_type", "all"),
        "batch_size": int(option_value(cross_eval_args, "--batch_size", "1") or 1),
        "num_workers": int(option_value(cross_eval_args, "--num_workers", "0") or 0),
        "resource_monitor_interval": args.resource_monitor_interval,
    }
    started_at = utc_now()
    manifest: dict[str, Any] = {
        "id": job_id,
        "name": job_name,
        "status": "queued",
        "created_at": started_at,
        "started_at": None,
        "completed_at": None,
        "pid": None,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "command": command,
        "request": request,
        "log_file": str(log_file),
        "progress_file": str(progress_file) if progress_file else None,
        "summary_file": str(summary_file),
        "checkpoint_list_file": str(checkpoint_list),
        "manifest_file": str(manifest_file),
        "resource_monitor_interval": args.resource_monitor_interval,
        "resource_monitoring": "starting",
    }

    log_file.write_text(
        "HFF-Net PBS evaluation log\n"
        f"job_id: {job_id}\n"
        f"pbs_job_id: {os.environ.get('PBS_JOBID', 'not set')}\n"
        f"created_at: {started_at}\n"
        f"command: {' '.join(command)}\n\n"
        "--- cross_eval.py output ---\n",
        encoding="utf-8",
    )
    write_manifest(manifest_file, manifest)

    resource_monitor: ResourceMonitor | None = None
    return_code = -1
    monitor_error: str | None = None
    process: subprocess.Popen[str] | None = None
    try:
        with log_file.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            manifest.update({
                "status": "running",
                "started_at": utc_now(),
                "pid": process.pid,
            })
            write_manifest(manifest_file, manifest)
            try:
                resource_monitor = ResourceMonitor(
                    device=get_device(),
                    output_directory=telemetry_dir,
                    interval_seconds=args.resource_monitor_interval,
                    root_pid=os.getpid(),
                ).start()
                manifest.update({
                    "resource_monitoring": "enabled",
                    "resource_backend": resource_monitor.backend,
                    "resource_log": str(resource_monitor.output_path),
                    "resource_summary_file": str(resource_monitor.summary_path),
                })
            except Exception as error:  # Monitoring must never prevent evaluation.
                monitor_error = str(error)
                manifest.update({
                    "resource_monitoring": "unavailable",
                    "resource_monitor_error": monitor_error,
                })
            write_manifest(manifest_file, manifest)
            return_code = process.wait()
    except OSError as error:
        monitor_error = str(error)
        append_log(log_file, f"--- PBS evaluation launcher failure ---\n{type(error).__name__}: {error}")
    finally:
        if resource_monitor is not None:
            try:
                resource_monitor.stop()
            except Exception as error:  # Preserve the job record if final telemetry flush fails.
                monitor_error = str(error)

    status = "completed" if return_code == 0 else "failed"
    manifest.update({
        "status": status,
        "return_code": return_code,
        "completed_at": utc_now(),
        "resource_monitoring": "enabled" if resource_monitor is not None else "unavailable",
        "resource_monitor_error": monitor_error,
    })
    if resource_monitor is not None:
        manifest.update({
            "resource_log": str(resource_monitor.output_path),
            "resource_summary_file": str(resource_monitor.summary_path),
        })
    write_manifest(manifest_file, manifest)
    if status == "failed":
        append_log(log_file, f"--- evaluation failure ---\nreturn_code: {return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
