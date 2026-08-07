#!/usr/bin/env bash
# Prepare one immutable cross-validation run and submit one dependent PBS job
# per fold. The dependency chain keeps the jobs sequential for the one-GPU
# per-user cluster policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${HFF_PYTHON:-python}"
GPU_DEVICE="${HFF_GPU_DEVICE:-}"
FOLDS=5
SEED=42
EPOCHS=450
RESOURCE_MONITOR_INTERVAL=5
RESULTS_DIR="result/cross_validation"
RUN_NAME=""
DATASET_DIR=""
CONDA_BASE="${HFF_CONDA_BASE:-}"
CONDA_SH="${HFF_CONDA_SH:-}"
CONDA_ENV="${HFF_CONDA_ENV:-hffnet}"

CROSS_ARGS=()
TRAIN_ARGS=()
FORWARD_SEPARATOR_SEEN=0

usage() {
  cat <<'USAGE'
Usage:
  HFF_GPU_DEVICE=<device> scripts/submit_train_cv_chain.sh \
    <dataset> --run-name <name> [cross_train.py options] -- [train.py options]

Required:
  <dataset>              Root containing BraTS patient directories.
  --run-name <name>      New experiment name under --results-dir.
  HFF_GPU_DEVICE         One physical GPU index or MIG UUID.

Examples:
  HFF_GPU_DEVICE=6 scripts/submit_train_cv_chain.sh \
    dataset/brats2019/extracted/MICCAI_BraTS_2019_Data_Training \
    --run-name brats19_all_seed42 --epochs 350 -- \
    --dataset_name brats19 --class_type all --batch_size 1

The script first creates the deterministic split lists with --dry-run, then
submits Fold 1 through Fold N with afterok dependencies.
USAGE
}

while [[ $# -gt 0 ]]; do
  if [[ "$FORWARD_SEPARATOR_SEEN" -eq 1 ]]; then
    TRAIN_ARGS+=("$1")
    shift
    continue
  fi

  case "$1" in
    --)
      FORWARD_SEPARATOR_SEEN=1
      shift
      ;;
    --gpu-device|--gpu)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; usage >&2; exit 1; }
      GPU_DEVICE="$2"
      shift 2
      ;;
    --run-name)
      [[ $# -ge 2 ]] || { echo "Error: --run-name requires a value." >&2; usage >&2; exit 1; }
      RUN_NAME="$2"
      shift 2
      ;;
    --results-dir)
      [[ $# -ge 2 ]] || { echo "Error: --results-dir requires a value." >&2; usage >&2; exit 1; }
      RESULTS_DIR="$2"
      shift 2
      ;;
    --folds)
      [[ $# -ge 2 ]] || { echo "Error: --folds requires a value." >&2; usage >&2; exit 1; }
      FOLDS="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Error: --seed requires a value." >&2; usage >&2; exit 1; }
      SEED="$2"
      shift 2
      ;;
    --epochs|-e)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value." >&2; usage >&2; exit 1; }
      EPOCHS="$2"
      shift 2
      ;;
    --resource-monitor-interval)
      [[ $# -ge 2 ]] || { echo "Error: --resource-monitor-interval requires a value." >&2; usage >&2; exit 1; }
      RESOURCE_MONITOR_INTERVAL="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "Error: --python requires a value." >&2; usage >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      CROSS_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -n "$DATASET_DIR" ]]; then
        echo "Error: only one dataset path may be supplied." >&2
        usage >&2
        exit 1
      fi
      DATASET_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$DATASET_DIR" || -z "$RUN_NAME" || -z "$GPU_DEVICE" ]]; then
  echo "Error: dataset, --run-name, and HFF_GPU_DEVICE/--gpu-device are required." >&2
  usage >&2
  exit 1
fi

PREPARE_ARGS=(
  "$DATASET_DIR"
  --epochs "$EPOCHS"
  --folds "$FOLDS"
  --seed "$SEED"
  --resource-monitor-interval "$RESOURCE_MONITOR_INTERVAL"
  --results-dir "$RESULTS_DIR"
  --run-name "$RUN_NAME"
  --dry-run
)
if [[ ${#CROSS_ARGS[@]} -gt 0 ]]; then
  PREPARE_ARGS+=("${CROSS_ARGS[@]}")
fi
PREPARE_ARGS+=(--)
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
  PREPARE_ARGS+=("${TRAIN_ARGS[@]}")
fi

echo "Preparing deterministic cross-validation run: $RUN_NAME"
"$PYTHON_BIN" cross_train.py "${PREPARE_ARGS[@]}"

if [[ "$RESULTS_DIR" = /* ]]; then
  RUN_DIR="$RESULTS_DIR/$RUN_NAME"
else
  RUN_DIR="$REPO_ROOT/$RESULTS_DIR/$RUN_NAME"
fi

QSUB_ENV_ARGS=()
if [[ -n "$CONDA_BASE" ]]; then
  QSUB_ENV_ARGS+=(-v "HFF_CONDA_BASE=$CONDA_BASE")
fi
if [[ -n "$CONDA_SH" ]]; then
  QSUB_ENV_ARGS+=(-v "HFF_CONDA_SH=$CONDA_SH")
fi
if [[ -n "$CONDA_ENV" ]]; then
  QSUB_ENV_ARGS+=(-v "HFF_CONDA_ENV=$CONDA_ENV")
fi
if [[ -n "${HFF_PYTHON:-}" ]]; then
  QSUB_ENV_ARGS+=(-v "HFF_PYTHON=$HFF_PYTHON")
fi

PREVIOUS_JOB_ID=""
for ((fold_index = 1; fold_index <= FOLDS; fold_index++)); do
  QSUB_ARGS=(
    qsub
    -N "hff_${RUN_NAME}_f${fold_index}"
  )
  if [[ ${#QSUB_ENV_ARGS[@]} -gt 0 ]]; then
    QSUB_ARGS=(qsub "${QSUB_ENV_ARGS[@]}" -N "hff_${RUN_NAME}_f${fold_index}")
  fi
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    QSUB_ARGS+=(-W "depend=afterok:$PREVIOUS_JOB_ID")
  fi
  QSUB_ARGS+=(
    -- "$SCRIPT_DIR/submit_train_gpu.pbs"
    --gpu-device "$GPU_DEVICE"
    "$DATASET_DIR"
    --fold-index "$fold_index"
    --run-dir "$RUN_DIR"
    --epochs "$EPOCHS"
    --folds "$FOLDS"
    --seed "$SEED"
    --resource-monitor-interval "$RESOURCE_MONITOR_INTERVAL"
  )
  if [[ ${#CROSS_ARGS[@]} -gt 0 ]]; then
    QSUB_ARGS+=("${CROSS_ARGS[@]}")
  fi
  QSUB_ARGS+=(--)
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    QSUB_ARGS+=("${TRAIN_ARGS[@]}")
  fi

  JOB_ID="$("${QSUB_ARGS[@]}")"
  echo "Submitted fold $fold_index as PBS job $JOB_ID"
  PREVIOUS_JOB_ID="$JOB_ID"
done

echo "Prepared run directory: $RUN_DIR"
echo "Fold jobs were submitted sequentially through afterok dependencies."
