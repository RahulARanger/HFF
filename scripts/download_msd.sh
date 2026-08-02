#!/usr/bin/env bash

set -euo pipefail

DATASET_URL="https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2"
OUTPUT_DIR="${OUTPUT_DIR:-./dataset/msd_brain_tumour}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_msd.sh [--output-dir PATH]

Environment variables:
  OUTPUT_DIR   Where to store the downloaded archive and extracted dataset.

Notes:
  - This script uses the MSD Google Drive folder linked in the README.
  - You need the `gdown` CLI installed: pip install gdown
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

if ! command -v gdown >/dev/null 2>&1; then
  echo "Error: gdown CLI not found. Install it with 'pip install gdown'." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

DOWNLOAD_DIR="$OUTPUT_DIR/download"
EXTRACT_DIR="$OUTPUT_DIR/extracted"

mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

echo "Downloading MSD Brain Tumour data from Google Drive"
gdown --folder "$DATASET_URL" -O "$DOWNLOAD_DIR"

echo "Extracting or organizing downloaded files into: $EXTRACT_DIR"
find "$DOWNLOAD_DIR" -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | while IFS= read -r archive; do
  case "$archive" in
    *.zip)
      unzip -q "$archive" -d "$EXTRACT_DIR"
      ;;
    *.tar)
      tar -xf "$archive" -C "$EXTRACT_DIR"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$EXTRACT_DIR"
      ;;
  esac
done

echo "If the Google Drive download produced raw folders instead of archives, they are left under: $DOWNLOAD_DIR"
echo "Full extracted dataset written to: $EXTRACT_DIR"

echo "Done."