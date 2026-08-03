#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset/brats_2019}"
LIMIT="${LIMIT:-}"
TRAIN="${TRAIN:-80}"
VALIDATION="${VALIDATION:-15}"
TESTING="${TESTING:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_ROOT/splits}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/slicer.sh [--limit N] [--train PERCENT] [--validation PERCENT] [--testing PERCENT]

Environment variables:
  DATASET_ROOT   Root folder that contains the downloaded BraTS 2019 dataset tree.
  LIMIT          Number of subject folders to slice from the dataset. Defaults to all records.
  TRAIN          Training percentage, default 80.
  VALIDATION     Validation percentage, default 15.
  TESTING        Testing percentage, default 5.
  OUTPUT_DIR     Where to write split folders plus optional train.txt, validation.txt, and testing.txt manifests.

Notes:
  - The script scans DATASET_ROOT recursively for subject folders containing segmentation files.
  - The selected subject folders are copied into OUTPUT_DIR/train, OUTPUT_DIR/validation, and OUTPUT_DIR/testing.
  - Matching train.txt, validation.txt, and testing.txt manifests are written for compatibility.
  - The split is deterministic after sorting the subject folders by path.
  - TRAIN + VALIDATION + TESTING must equal 100.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --train)
      TRAIN="$2"
      shift 2
      ;;
    --validation)
      VALIDATION="$2"
      shift 2
      ;;
    --testing)
      TESTING="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$LIMIT" ]]; then
  if ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "Error: --limit must be a positive integer." >&2
    exit 1
  fi

  if [[ "$LIMIT" -le 0 ]]; then
    echo "Error: --limit must be greater than zero when provided." >&2
    exit 1
  fi
fi

for value in "$TRAIN" "$VALIDATION" "$TESTING"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "Error: split values must be non-negative integers." >&2
    exit 1
  fi
done

if (( TRAIN + VALIDATION + TESTING != 100 )); then
  echo "Error: TRAIN + VALIDATION + TESTING must equal 100." >&2
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Error: dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 - "$DATASET_ROOT" "$LIMIT" "$TRAIN" "$VALIDATION" "$TESTING" "$OUTPUT_DIR" <<'PY'
import os
import shutil
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1]).resolve()
limit_arg = sys.argv[2]
train_pct = int(sys.argv[3])
validation_pct = int(sys.argv[4])
testing_pct = int(sys.argv[5])
output_dir = Path(sys.argv[6]).resolve()

subject_dirs = []
for seg_file in sorted(dataset_root.rglob('*_seg.nii')) + sorted(dataset_root.rglob('*_seg.nii.gz')):
    subject_dir = seg_file.parent.resolve()
    if subject_dir not in subject_dirs:
        subject_dirs.append(subject_dir)

subject_dirs = sorted(subject_dirs)

if not subject_dirs:
    raise SystemExit(f'No subject folders with segmentation files found under {dataset_root}')

if limit_arg:
    limit = int(limit_arg)
    if limit > len(subject_dirs):
        raise SystemExit(f'Limit {limit} is greater than the number of subject folders found ({len(subject_dirs)})')
else:
    limit = len(subject_dirs)

selected = subject_dirs[:limit]

train_count = limit * train_pct // 100
validation_count = limit * validation_pct // 100
testing_count = limit - train_count - validation_count

train_dirs = selected[:train_count]
validation_dirs = selected[train_count:train_count + validation_count]
testing_dirs = selected[train_count + validation_count:train_count + validation_count + testing_count]

def write_paths(path: Path, paths):
    with path.open('w', encoding='utf-8') as handle:
        for item in paths:
            try:
                rel = item.relative_to(Path.cwd().resolve())
                handle.write(f'./{rel.as_posix()}\n')
            except ValueError:
                handle.write(str(item) + '\n')

def copy_subject_dirs(split_name: str, paths):
    split_root = output_dir / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    for subject_dir in paths:
        target_dir = split_root / subject_dir.name
        shutil.copytree(subject_dir, target_dir, dirs_exist_ok=True)

copy_subject_dirs('train', train_dirs)
copy_subject_dirs('validation', validation_dirs)
copy_subject_dirs('testing', testing_dirs)

write_paths(output_dir / 'train.txt', train_dirs)
write_paths(output_dir / 'validation.txt', validation_dirs)
write_paths(output_dir / 'testing.txt', testing_dirs)

print(f'Found {len(subject_dirs)} subject folders under {dataset_root}')
print(f'Selected first {len(selected)} subject folders')
print(f'Copied {len(train_dirs)} train, {len(validation_dirs)} validation, and {len(testing_dirs)} testing subject folders to {output_dir}')
PY

echo "Done. Split folders written to: $OUTPUT_DIR"
