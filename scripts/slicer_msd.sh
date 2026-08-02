#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DATASET_ROOT:-./dataset/msd_brain_tumour/extracted}"
DEFAULT_OUTPUT="${OUTPUT_DIR:-./dataset/msd_brain_tumour/splits}"

exec "$SCRIPT_DIR/slicer.sh" --dataset-root "$DEFAULT_ROOT" --output-dir "$DEFAULT_OUTPUT" "$@"