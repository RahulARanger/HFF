#!/usr/bin/env bash

set -euo pipefail

INPUT_PATH=""
TRAIN_LIST=""
VALIDATION_LIST=""
TESTING_LIST=""
OUTPUT_DIR=""
NLEVELS="${NLEVELS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/generate_low_freq.sh --path PATH [--output-dir PATH] [--nlevels N]
  bash scripts/generate_low_freq.sh --train-list FILE --validation-list FILE --testing-list FILE [--output-dir PATH] [--nlevels N]

Arguments:
  --path        Root folder containing the BraTS subject folders.
  --train-list   Text file containing training subject folders, one per line.
  --validation-list  Text file containing validation subject folders, one per line.
  --testing-list Text file containing testing subject folders, one per line.
  --output-dir  Output folder for low-frequency volumes. Defaults to <path>/low-freq.
  --nlevels     DTCWT decomposition levels. Default: 1.

Notes:
  - The script mirrors the input directory structure under the output folder.
  - It writes one low-frequency file per modality as *_L.nii.gz.
  - In split mode, outputs are written under train/, validation/, and testing/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --train-list)
      TRAIN_LIST="$2"
      shift 2
      ;;
    --validation-list)
      VALIDATION_LIST="$2"
      shift 2
      ;;
    --testing-list)
      TESTING_LIST="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --nlevels)
      NLEVELS="$2"
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

split_mode=false
if [[ -n "$TRAIN_LIST" || -n "$VALIDATION_LIST" || -n "$TESTING_LIST" ]]; then
  split_mode=true
fi

if [[ "$split_mode" == true ]]; then
  if [[ -z "$TRAIN_LIST" || -z "$VALIDATION_LIST" || -z "$TESTING_LIST" ]]; then
    echo "Error: split mode requires --train-list, --validation-list, and --testing-list." >&2
    usage >&2
    exit 1
  fi
else
  if [[ -z "$INPUT_PATH" ]]; then
    echo "Error: --path is required when split lists are not provided." >&2
    usage >&2
    exit 1
  fi
fi

if ! [[ "$NLEVELS" =~ ^[0-9]+$ ]]; then
  echo "Error: --nlevels must be a positive integer." >&2
  exit 1
fi

if [[ "$split_mode" == false ]]; then
  if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: input path not found: $INPUT_PATH" >&2
    exit 1
  fi

  if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$INPUT_PATH/low-freq"
  fi
else
  if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./low-freq"
  fi
fi

mkdir -p "$OUTPUT_DIR"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Error: neither python nor python3 was found on PATH." >&2
  exit 1
fi

"$PYTHON_BIN" - "$INPUT_PATH" "$TRAIN_LIST" "$VALIDATION_LIST" "$TESTING_LIST" "$OUTPUT_DIR" "$NLEVELS" "$split_mode" <<'PY'
from pathlib import Path
import sys
from typing import Iterable

import dtcwt
import nibabel as nib
import numpy as np

input_root = Path(sys.argv[1]).resolve()
train_list = sys.argv[2]
validation_list = sys.argv[3]
testing_list = sys.argv[4]
output_root = Path(sys.argv[5]).resolve()
nlevels = int(sys.argv[6])
split_mode = sys.argv[7].lower() == "true"

transform = dtcwt.Transform2d()
processed = 0

def iter_input_files(root: Path):
    nii_files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    for nii_path in sorted(nii_files):
        if output_root in nii_path.parents:
            continue
        name = nii_path.name.lower()
        if "seg" in name or "_h" in name or "_l" in name:
            continue
        yield nii_path


    def load_subject_dirs(list_path: str) -> list[Path]:
      subject_dirs: list[Path] = []
      with Path(list_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
          line = raw_line.strip()
          if not line or line.startswith("#"):
            continue
          subject_dir = Path(line).expanduser()
          if not subject_dir.is_absolute():
            subject_dir = (Path.cwd() / subject_dir).resolve()
          else:
            subject_dir = subject_dir.resolve()
          if subject_dir.is_dir() and subject_dir not in subject_dirs:
            subject_dirs.append(subject_dir)
      return subject_dirs


    def iter_subject_files(subject_dir: Path):
      for nii_path in sorted(subject_dir.glob("*.nii")) + sorted(subject_dir.glob("*.nii.gz")):
        name = nii_path.name.lower()
        if "seg" in name or "_h" in name or "_l" in name:
          continue
        yield nii_path


    def process_file(nii_path: Path, out_path: Path) -> None:
      global processed

      nii = nib.load(str(nii_path))
      data = nii.get_fdata()

      if data.ndim != 3:
        raise SystemExit(f"Expected 3D NIfTI data, got shape {data.shape} for {nii_path}")

      lowpass_slices = []
      for slice_index in range(data.shape[2]):
        transformed = transform.forward(data[:, :, slice_index], nlevels=nlevels)
        lowpass = transformed.lowpass

        low_min = float(lowpass.min())
        low_max = float(lowpass.max())
        if low_max > low_min:
          normalized = (lowpass - low_min) / (low_max - low_min) * 255.0
        else:
          normalized = np.zeros_like(lowpass)

        lowpass_slices.append(normalized.astype(np.uint8))

      out_path.parent.mkdir(parents=True, exist_ok=True)
      nib.save(nib.Nifti1Image(np.stack(lowpass_slices, axis=2), affine=nii.affine), str(out_path))
      processed += 1
      print(f"Saved low-frequency file: {out_path}")

    if split_mode:
      split_subjects = {
        "train": load_subject_dirs(train_list),
        "validation": load_subject_dirs(validation_list),
        "testing": load_subject_dirs(testing_list),
      }

      for split_name, subject_dirs in split_subjects.items():
        split_output_root = output_root / split_name
        for subject_dir in subject_dirs:
          split_subject_output = split_output_root / subject_dir.name
          for nii_path in iter_subject_files(subject_dir):
            base_name = nii_path.name[:-7] if nii_path.name.endswith(".nii.gz") else nii_path.stem
            process_file(nii_path, split_subject_output / f"{base_name}_L.nii.gz")
    else:
      for nii_path in iter_input_files(input_root):
        nii = nib.load(str(nii_path))
        relative_parent = nii_path.parent.relative_to(input_root)
        base_name = nii_path.name[:-7] if nii_path.name.endswith(".nii.gz") else nii_path.stem
        process_file(nii_path, output_root / relative_parent / f"{base_name}_L.nii.gz")

    print(f"Done. Processed {processed} modality files into {output_root}")
PY