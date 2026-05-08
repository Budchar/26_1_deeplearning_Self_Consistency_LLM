#!/usr/bin/env bash
# Quick smoke for Sweep C: depth=4, width=512, 1K steps.
# Verifies that the patched train.py path, run-name override, and shard data
# loading all work before kicking off the full ~36h sweep.
#
# WARNING: this still uses GPU. Do NOT run while Phase 2 is occupying the GPU.

set -euo pipefail

PHASE2_ROOT="/home/kcai/experiments/dl_team_v2/02_c2_sinks"
SWEEP_ROOT="/home/kcai/experiments/dl_team_v2/05_sweep_c_depth"
PY="/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python"
CODE="$PHASE2_ROOT/code"
LOGDIR="$SWEEP_ROOT/results"
mkdir -p "$LOGDIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEPTH=${DEPTH:-4}
WIDTH=${WIDTH:-512}
N_HEAD=${N_HEAD:-8}
VARIANT=${VARIANT:-softmax}
MAX_STEPS=${MAX_STEPS:-1000}
TAG=${TAG:-smoke}

RUN_NAME="${VARIANT}_d${DEPTH}_w${WIDTH}"
LOG="$LOGDIR/${RUN_NAME}_${TAG}_train.log"

echo "==== [$(date)] SMOKE  depth=$DEPTH  width=$WIDTH  steps=$MAX_STEPS  run=$RUN_NAME ===="
"$PY" "$CODE/train.py" \
  --variant "$VARIANT" \
  --tag "$TAG" \
  --depth "$DEPTH" \
  --width "$WIDTH" \
  --n-head "$N_HEAD" \
  --run-name "$RUN_NAME" \
  --out "$SWEEP_ROOT" \
  --max-steps "$MAX_STEPS" \
  --micro-bs 4 \
  --grad-accum 4 \
  --ckpt-steps "0,100,500,1000" \
  2>&1 | tee -a "$LOG"

echo "[smoke] $(date) done. Inspect $LOG and $SWEEP_ROOT/checkpoints/$RUN_NAME/"
