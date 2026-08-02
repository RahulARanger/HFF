#!/usr/bin/env bash

set -euo pipefail

DATASET_SLUG="aryashah2k/brain-tumor-segmentation-brats-2019"
OUTPUT_DIR="${OUTPUT_DIR:-./dataset/brats_2019}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_brats2019.sh [--output-dir PATH]

Environment variables:
  OUTPUT_DIR   Where to store the downloaded archive and extracted dataset.

Notes:
  - This script uses the Kaggle mirror for BraTS 2019.
  - You must have the Kaggle CLI installed and authenticated with ~/.kaggle/kaggle.json.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI not found. Install it with 'pip install kaggle' and configure ~/.kaggle/kaggle.json." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

DOWNLOAD_DIR="$OUTPUT_DIR/download"
EXTRACT_DIR="$OUTPUT_DIR/extracted"

mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

echo "Downloading BraTS 2019 from Kaggle: $DATASET_SLUG"
kaggle datasets download -d "$DATASET_SLUG" -p "$DOWNLOAD_DIR" --force

ZIP_FILE="$(find "$DOWNLOAD_DIR" -maxdepth 1 -name '*.zip' | sort | tail -n 1)"
if [[ -z "${ZIP_FILE:-}" ]]; then
  echo "Error: Kaggle download completed, but no .zip file was found in $DOWNLOAD_DIR." >&2
  exit 1
fi

echo "Extracting archive: $ZIP_FILE"
rm -rf "$EXTRACT_DIR"/*
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"

echo "Full extracted dataset written to: $EXTRACT_DIR"

echo "Done."