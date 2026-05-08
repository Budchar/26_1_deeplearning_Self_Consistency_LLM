#!/usr/bin/env bash
# Sweep C: depth-only causal sweep for H3 (peak relative depth ~ 0.68 for
# hallucination-detection probes).
#
# Trains 6 small GPT-2-style transformers from scratch on OpenWebText.
#  - attention variant: softmax (only)
#  - width (n_embd):    512  (fixed)
#  - n_head:            8    (fixed; head_dim = 64)
#  - depth (n_layer):   {4, 8, 12, 16, 24, 32}
#  - all other hyperparams identical to Phase 2 softmax baseline
#    (lr=3e-4, min-lr=3e-5, warmup=1000, wd=0.1, micro-bs=4, grad-accum=4,
#     seq=1024, block-size=1024, grad-checkpoint=on, max-steps=50000,
#     n-train-tokens=1B, seed=42, default DEFAULT_CKPTS).
#
# train.py was patched (backward-compatibly) to accept --depth/--width/
# --n-head/--run-name. Phase 2 callers that omit those flags are unaffected.
#
# Resume-safe: train.py auto-resumes from the latest checkpoint in
# <out>/checkpoints/<run-name>/. Skipping a depth: comment its line in DEPTHS.
#
# Wall-time estimate (RTX 5070, bf16, grad-checkpoint on):
#   d=4  ~1.5h   d=8  ~3h    d=12 ~4.5h
#   d=16 ~6h    d=24 ~9h    d=32 ~12h
#   total ~36h compute (sequential ~36h wall) + I/O ~ ~50-60 GPU-h budget.
#
# Logs: <out>/results/softmax_d{D}_w512_train.log
# TB:   <out>/runs/softmax_d{D}_w512/
# Ckpt: <out>/checkpoints/softmax_d{D}_w512/

set -euo pipefail

PHASE2_ROOT="/home/kcai/experiments/dl_team_v2/02_c2_sinks"
SWEEP_ROOT="/home/kcai/experiments/dl_team_v2/05_sweep_c_depth"
PY="/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python"
CODE="$PHASE2_ROOT/code"        # reuse Phase 2 train.py
LOGDIR="$SWEEP_ROOT/results"
mkdir -p "$LOGDIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEPTHS=(4 8 12 16 24 32)
WIDTH=${WIDTH:-512}
N_HEAD=${N_HEAD:-8}
VARIANT=${VARIANT:-softmax}
MAX_STEPS=${MAX_STEPS:-50000}
MICRO_BS=${MICRO_BS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TAG=${TAG:-full}

echo "[launch] $(date)  depths=${DEPTHS[*]}  width=$WIDTH  n_head=$N_HEAD  variant=$VARIANT  max_steps=$MAX_STEPS"
for D in "${DEPTHS[@]}"; do
  RUN_NAME="${VARIANT}_d${D}_w${WIDTH}"
  LOG="$LOGDIR/${RUN_NAME}_${TAG}_train.log"
  echo "==== [$(date)] training depth=$D  width=$WIDTH  run=$RUN_NAME  log=$LOG ===="
  "$PY" "$CODE/train.py" \
    --variant "$VARIANT" \
    --tag "$TAG" \
    --depth "$D" \
    --width "$WIDTH" \
    --n-head "$N_HEAD" \
    --run-name "$RUN_NAME" \
    --out "$SWEEP_ROOT" \
    --max-steps "$MAX_STEPS" \
    --micro-bs "$MICRO_BS" \
    --grad-accum "$GRAD_ACCUM" \
    2>&1 | tee -a "$LOG"
done

echo "[launch] $(date) all depths done"
