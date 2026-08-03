#!/usr/bin/env bash

set -euo pipefail

INPUT_PATH=""
NSCT_TOOLBOX_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/generate_high_freq.sh --path PATH [--toolbox-dir PATH]

Arguments:
  --path          Root folder containing the subject folders to process.
  --toolbox-dir   Optional path to the NSCT toolbox.
                  Defaults to ./NSCT_BTS/nsct_toolbox.

Notes:
  - The MATLAB entrypoint is NSCT_BTS/nsct_hf.m.
  - Outputs are written beside the input scans as *_H1.nii.gz through *_H4.nii.gz.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --path requires a value." >&2
        usage >&2
        exit 1
      fi
      INPUT_PATH="$2"
      shift 2
      ;;
    --toolbox-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --toolbox-dir requires a value." >&2
        usage >&2
        exit 1
      fi
      NSCT_TOOLBOX_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_PATH" ]]; then
  echo "Error: --path is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$INPUT_PATH" ]]; then
  echo "Error: input path not found: $INPUT_PATH" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="$(cd "$INPUT_PATH" && pwd)"

if [[ -z "$NSCT_TOOLBOX_DIR" ]]; then
  NSCT_TOOLBOX_DIR="$REPO_ROOT/NSCT_BTS/nsct_toolbox"
else
  NSCT_TOOLBOX_DIR="$(cd "$NSCT_TOOLBOX_DIR" && pwd)"
fi

if [[ ! -d "$NSCT_TOOLBOX_DIR" ]]; then
  echo "Error: NSCT toolbox directory not found: $NSCT_TOOLBOX_DIR" >&2
  exit 1
fi

if ! command -v matlab >/dev/null 2>&1; then
  echo "Error: matlab was not found on PATH." >&2
  exit 1
fi

# Rebuild the bundled NSCT MEX helpers on Apple Silicon when needed.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  if [[ ! -f "$NSCT_TOOLBOX_DIR/zconv2.mexmaca64" || \
        ! -f "$NSCT_TOOLBOX_DIR/zconv2S.mexmaca64" || \
        ! -f "$NSCT_TOOLBOX_DIR/atrousc.mexmaca64" ]]; then
    matlab -batch "cd('$NSCT_TOOLBOX_DIR'); mex('zconv2.c'); mex('zconv2S.c'); mex('atrousc.c')"
  fi
fi

export HFF_BASE_DIR="$INPUT_PATH"
export HFF_NSCT_TOOLBOX="$NSCT_TOOLBOX_DIR"

matlab -batch "run('$REPO_ROOT/NSCT_BTS/nsct_hf.m')"
