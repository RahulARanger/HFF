"""Run deterministic patient-level k-fold training with ``train.py``.

This script deliberately owns only the cross-validation orchestration.  The
model, preprocessing, optimisation, and validation behaviour remain in
``train.py`` so that a single-fold experiment and a cross-validation fold use
the same training implementation.

The input directory may contain patient directories directly or under
subdirectories such as BraTS ``HGG`` and ``LGG``.  A patient directory is
identified by the presence of a ``*_seg.nii`` or ``*_seg.nii.gz`` file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_FOLDS = 5


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train one independent HFF-Net model for each CV fold.",
        epilog=(
            "Extra arguments after '--' are forwarded unchanged to train.py. "
            "For example: -- --lr 0.1 --batch_size 2"
        ),
    )
    parser.add_argument("dataset_dir", type=Path, help="Root containing BraTS patient directories.")
    parser.add_argument("--epochs", "-e", type=int, default=450, help="Epochs for every fold (default: 450).")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only to allocate patients to folds.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("result/cross_validation"),
        help=(
            "Parent directory for immutable cross-validation experiment runs "
            "(default: result/cross_validation)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional name for this experiment directory. It must not already "
            "exist and may contain only letters, numbers, '.', '_' and '-'."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch train.py (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and validate the splits without launching training.",
    )
    args, train_args = parser.parse_known_args()
    # argparse retains the ``--`` separator in unknown arguments.  It is only
    # meaningful to this wrapper and must not be forwarded to train.py.
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    return args, train_args


def find_patient_directories(dataset_dir: Path) -> list[Path]:
    """Return sorted, unique directories that contain a segmentation volume."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    patients = {
        segmentation.parent.resolve()
        for pattern in ("*_seg.nii", "*_seg.nii.gz")
        for segmentation in dataset_dir.rglob(pattern)
    }
    if not patients:
        raise ValueError(
            f"No patient directories containing '*_seg.nii' or '*_seg.nii.gz' were found below {dataset_dir}."
        )
    return sorted(patients, key=lambda path: str(path))


def split_patients(patients: Sequence[Path], folds: int, seed: int) -> list[list[Path]]:
    """Create deterministic, approximately equal validation folds."""
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    if len(patients) < folds:
        raise ValueError(f"Cannot create {folds} folds from only {len(patients)} patients.")

    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::folds] for index in range(folds)]


def write_patient_list(path: Path, patients: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(patient) for patient in patients) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON artifact so an interrupted job leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary_file:
        json.dump(payload, temporary_file, indent=2, allow_nan=False)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def validate_run_name(run_name: str) -> str:
    """Validate a portable, single-directory experiment name."""
    permitted_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not run_name or any(character not in permitted_characters for character in run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_' and '-'.")
    if run_name in {".", ".."}:
        raise ValueError("--run-name must name a directory, not '.' or '..'.")
    return run_name


def create_run_directory(results_root: Path, requested_name: str | None) -> Path:
    """Create and return a new experiment directory without reusing prior results."""
    results_root.mkdir(parents=True, exist_ok=True)
    if requested_name is not None:
        run_name = validate_run_name(requested_name)
        run_dir = results_root / run_name
        try:
            run_dir.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"Experiment directory already exists: {run_dir}. Choose a new --run-name; "
                "existing results are never overwritten."
            ) from error
        return run_dir

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    # PID keeps simultaneous cluster submissions distinct even if their clocks align.
    run_dir = results_root / f"run_{timestamp}_pid{os.getpid()}"
    run_dir.mkdir()
    return run_dir


def read_fold_metrics(fold_dir: Path) -> dict[str, Any]:
    metrics_files = sorted(fold_dir.rglob("training_metrics.json"))
    if len(metrics_files) != 1:
        raise RuntimeError(
            f"Expected exactly one training_metrics.json below {fold_dir}, found {len(metrics_files)}."
        )
    with metrics_files[0].open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return {"metrics_file": str(metrics_files[0]), **metrics}


def mean_and_std(values: list[float | None]) -> dict[str, float | int | None]:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return {"count": 0, "mean": None, "std": None}
    mean = sum(finite_values) / len(finite_values)
    variance = sum((value - mean) ** 2 for value in finite_values) / len(finite_values)
    return {"count": len(finite_values), "mean": mean, "std": variance**0.5}


def summarise_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    fold_values = [result["metrics"]["best_validation_metrics"] for result in fold_results]
    metric_count = max(len(values) for values in fold_values)
    return {
        "best_checkpoint_metric": [
            mean_and_std([values[index] if index < len(values) else None for values in fold_values])
            for index in range(metric_count)
        ],
        "best_checkpoint_selected_branch": [
            result["metrics"]["best_result"] for result in fold_results
        ],
    }


def main() -> int:
    args, train_args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")

    patients = find_patient_directories(args.dataset_dir)
    validation_folds = split_patients(patients, args.folds, args.seed)
    results_root = args.results_dir.expanduser().resolve()
    results_dir = create_run_directory(results_root, args.run_name)
    split_dir = results_dir / "splits"

    manifest: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
        "patient_count": len(patients),
        "fold_count": args.folds,
        "split_seed": args.seed,
        "epochs_per_fold": args.epochs,
        "forwarded_train_arguments": train_args,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared" if args.dry_run else "running",
        "folds": [],
    }
    fold_jobs: list[tuple[int, Path, Path, Path]] = []
    for fold_index, validation_patients in enumerate(validation_folds, start=1):
        validation_set = set(validation_patients)
        training_patients = [patient for patient in patients if patient not in validation_set]
        train_list = split_dir / f"fold_{fold_index}_train.txt"
        validation_list = split_dir / f"fold_{fold_index}_validation.txt"
        write_patient_list(train_list, training_patients)
        write_patient_list(validation_list, validation_patients)
        fold_dir = results_dir / f"fold_{fold_index}"
        manifest["folds"].append(
            {
                "fold": fold_index,
                "train_patient_count": len(training_patients),
                "validation_patient_count": len(validation_patients),
                "train_list": str(train_list),
                "validation_list": str(validation_list),
                "output_dir": str(fold_dir),
                "status": "pending",
            }
        )
        fold_jobs.append((fold_index, train_list, validation_list, fold_dir))

    manifest_path = results_dir / "cross_validation_manifest.json"
    write_json(manifest_path, manifest)
    LOGGER.info(
        "Created %d deterministic folds for %d patients in new experiment directory %s",
        args.folds,
        len(patients),
        results_dir,
    )

    if args.dry_run:
        return 0

    fold_results: list[dict[str, Any]] = []
    train_script = Path(__file__).with_name("train.py")
    for fold_index, train_list, validation_list, fold_dir in fold_jobs:
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest["folds"][fold_index - 1]["status"] = "running"
        manifest["folds"][fold_index - 1]["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        command = [
            args.python,
            str(train_script),
            "--train_list", str(train_list),
            "--val_list", str(validation_list),
            "--num_epochs", str(args.epochs),
            "--path_trained_models", str(fold_dir),
            *train_args,
        ]
        LOGGER.info("Starting fold %d/%d", fold_index, args.folds)
        try:
            subprocess.run(command, check=True)
            fold_metrics = read_fold_metrics(fold_dir)
        except (subprocess.CalledProcessError, OSError, RuntimeError, json.JSONDecodeError):
            manifest["status"] = "failed"
            manifest["failed_fold"] = fold_index
            manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["folds"][fold_index - 1]["status"] = "failed"
            write_json(manifest_path, manifest)
            raise
        fold_results.append({"fold": fold_index, "metrics": fold_metrics})
        manifest["folds"][fold_index - 1]["status"] = "completed"
        manifest["folds"][fold_index - 1]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)

    summary = {
        "manifest": str(manifest_path),
        "fold_results": fold_results,
        "best_validation_metric_summary": summarise_metrics(fold_results),
    }
    write_json(results_dir / "cross_validation_metrics.json", summary)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    LOGGER.info("Saved cross-validation metric summary to %s", results_dir / "cross_validation_metrics.json")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        LOGGER.error("Cross-validation failed: %s", error)
        raise SystemExit(1) from error
