#!/usr/bin/env python3
"""Create BraTS 2020 train/validation/testing directory splits.

BraTS 2020 keeps the training and validation downloads in separate trees.  This
script therefore makes the train/validation split only from the training tree
and makes the testing split independently from the validation tree.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def subject_directories(root: Path) -> list[Path]:
    """Return sorted, immediate child directories representing subjects."""
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
    )


def copy_subjects(subjects: list[Path], output_root: Path, split_name: str) -> list[Path]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []

    for subject in subjects:
        target = split_root / subject.name
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing subject directory: {target}. "
                "Choose a new --output-dir or remove the previous split."
            )
        shutil.copytree(subject, target)
        copied_paths.append(target)

    return copied_paths


def write_manifest(path: Path, subjects: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for subject in subjects:
            handle.write(f"{subject.resolve()}\n")


def percentage_count(total: int, percentage: int) -> int:
    """Use floor semantics, while allowing the caller to retain all leftovers."""
    return total * percentage // 100


def parse_args() -> argparse.Namespace:
    cwd = Path.cwd()
    parser = argparse.ArgumentParser(
        description=(
            "Split BraTS 2020 training subjects into train/validation and "
            "take a percentage of BraTS 2020 validation subjects as testing."
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=cwd / "dataset" / "brats_2020" / "extracted" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=cwd / "dataset" / "brats_2020" / "extracted" / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cwd / "dataset" / "brats_2020" / "splits",
    )
    parser.add_argument(
        "--training",
        "--train",
        dest="training",
        type=int,
        default=80,
        metavar="PERCENT",
        help="Percentage of selected training subjects assigned to train (default: 80).",
    )
    parser.add_argument("--validation", type=int, default=20, metavar="PERCENT")
    parser.add_argument(
        "--testing",
        type=int,
        default=0,
        metavar="PERCENT",
        help="Percentage of validation-source subjects assigned to testing (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally use only the first N sorted subjects from the training source.",
    )
    parser.add_argument(
        "--training-limit",
        dest="limit",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--testing-limit",
        type=int,
        default=None,
        help="Optionally use only the first N sorted validation subjects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.training < 0 or args.validation < 0 or args.testing < 0:
        raise SystemExit("Error: percentages must be non-negative integers.")
    if args.training + args.validation != 100:
        raise SystemExit("Error: --training + --validation must equal 100.")
    for name, value in (("limit", args.limit), ("testing-limit", args.testing_limit)):
        if value is not None and value <= 0:
            raise SystemExit(f"Error: --{name} must be greater than zero.")

    training_root = args.training_root.resolve()
    validation_root = args.validation_root.resolve()
    output_dir = args.output_dir.resolve()
    for label, root in (("training", training_root), ("validation", validation_root)):
        if not root.is_dir():
            raise SystemExit(f"Error: {label} root not found: {root}")

    training_subjects = subject_directories(training_root)
    testing_subjects = subject_directories(validation_root)
    if not training_subjects:
        raise SystemExit(f"Error: no subject directories found under {training_root}")
    if not testing_subjects:
        raise SystemExit(f"Error: no subject directories found under {validation_root}")

    if args.limit is not None:
        if args.limit > len(training_subjects):
            raise SystemExit(
                f"Error: --limit ({args.limit}) exceeds "
                f"the {len(training_subjects)} training subjects found."
            )
        training_subjects = training_subjects[: args.limit]
    if args.testing_limit is not None:
        if args.testing_limit > len(testing_subjects):
            raise SystemExit(
                f"Error: --testing-limit ({args.testing_limit}) exceeds "
                f"the {len(testing_subjects)} validation subjects found."
            )
        testing_subjects = testing_subjects[: args.testing_limit]

    train_count = percentage_count(len(training_subjects), args.training)
    train_subjects = training_subjects[:train_count]
    validation_subjects = training_subjects[train_count:]
    testing_count = percentage_count(len(testing_subjects), args.testing)
    testing_subjects = testing_subjects[:testing_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_targets = copy_subjects(train_subjects, output_dir, "train")
    validation_targets = copy_subjects(validation_subjects, output_dir, "validation")
    testing_targets = copy_subjects(testing_subjects, output_dir, "testing")
    write_manifest(output_dir / "train.txt", train_targets)
    write_manifest(output_dir / "validation.txt", validation_targets)
    write_manifest(output_dir / "testing.txt", testing_targets)

    print(f"Training source: {training_root} ({len(training_subjects)} subjects)")
    print(f"Testing source:  {validation_root} ({len(testing_subjects)} selected subjects)")
    print(
        f"Created {len(train_subjects)} train, {len(validation_subjects)} validation, "
        f"and {len(testing_subjects)} testing subjects in {output_dir}"
    )


if __name__ == "__main__":
    main()
