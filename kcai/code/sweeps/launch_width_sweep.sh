#!/usr/bin/env bash
# Sweep D: width-only causal sweep — paired with Sweep C for clean depth/width
# disentangling.
#
# Trains 5 small GPT-2-style transformers from scratch on OpenWebText.
#  - attention variant: softmax (only)
#  - depth (n_layer):   12   (fixed)
#  - width (n_embd):    {256, 384, 512, 768, 1024}
#  - n_head:            scales with width to keep head_dim=64
#  - all other hyperparams identical to Phase 2 softmax baseline
#
# Note: width=512/depth=12 overlaps with Sweep C d=12 — we reuse that run
# instead of retraining (skip if checkpoint already exists).
#
# Wall-time estimate (RTX 5070, bf16, grad-checkpoint on):
#   w=256  ~1.5h   w=384  ~2h    w=512  ~4.5h (reuse from Sweep C)
#   w=768  ~6h    w=1024 ~10h
#   total ~24h compute new + reuse w=512 → ~30 GPU-h budget.
#
# Resume-safe: train.py auto-resumes; comment a width to skip.

set -euo pipefail

PHASE2_ROOT="/home/kcai/experiments/dl_team_v2/02_c2_sinks"
SWEEP_C_ROOT="/home/kcai/experiments/dl_team_v2/05_sweep_c_depth"
SWEEP_ROOT="/home/kcai/experiments/dl_team_v2/07_sweep_d_width"
PY="/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python"
CODE="$PHASE2_ROOT/code"
LOGDIR="$SWEEP_ROOT/results"
mkdir -p "$LOGDIR" "$SWEEP_ROOT/runs" "$SWEEP_ROOT/checkpoints"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# (width, n_head) pairs — head_dim=64 throughout
WIDTHS=(256 384 512 768 1024)
HEADS=(4   6   8   12  16)
DEPTH=${DEPTH:-12}
VARIANT=${VARIANT:-softmax}
MAX_STEPS=${MAX_STEPS:-50000}
MICRO_BS=${MICRO_BS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TAG=${TAG:-full}

echo "[launch-D] $(date) widths=${WIDTHS[*]}  heads=${HEADS[*]}  depth=$DEPTH"
for i in "${!WIDTHS[@]}"; do
  W="${WIDTHS[i]}"
  H="${HEADS[i]}"
  RUN_NAME="${VARIANT}_d${DEPTH}_w${W}"
  LOG="$LOGDIR/${RUN_NAME}_${TAG}_train.log"

  # If the SAME (depth, width) was already trained in Sweep C, reuse it.
  CKPT_C="$SWEEP_C_ROOT/checkpoints/${VARIANT}_d${DEPTH}_w${W}"
  if [[ -d "$CKPT_C" ]] && ls "$CKPT_C"/step050000_full.pt &>/dev/null; then
    echo "==== [$(date)] reuse Sweep C: $CKPT_C (skip retrain) ===="
    # Symlink for downstream eval to find both sweeps in one place
    mkdir -p "$SWEEP_ROOT/checkpoints"
    [[ -e "$SWEEP_ROOT/checkpoints/$RUN_NAME" ]] || ln -s "$CKPT_C" "$SWEEP_ROOT/checkpoints/$RUN_NAME"
    continue
  fi

  echo "==== [$(date)] training depth=$DEPTH width=$W n_head=$H run=$RUN_NAME ===="
  "$PY" "$CODE/train.py" \
    --variant "$VARIANT" \
    --tag "$TAG" \
    --depth "$DEPTH" \
    --width "$W" \
    --n-head "$H" \
    --run-name "$RUN_NAME" \
    --out "$SWEEP_ROOT" \
    --max-steps "$MAX_STEPS" \
    --micro-bs "$MICRO_BS" \
    --grad-accum "$GRAD_ACCUM" \
    2>&1 | tee -a "$LOG"
done

echo "[launch-D] $(date) all widths done"
