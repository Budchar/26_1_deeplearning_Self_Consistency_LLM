#!/usr/bin/env bash
# Full Phase-1 sweep launcher.
#
# Time estimates (5070 12GB, fp16 / bnb-4bit, N=10, 1000 questions, max_new=64):
#   Qwen2.5-1.5B-Instruct  fp16 :  ~25 min / dataset    -> ~75 min total
#   Qwen2.5-3B-Instruct    fp16 :  ~45 min / dataset    -> ~135 min total
#   Qwen2.5-7B-Instruct    4bit :  ~150 min / dataset   -> ~450 min total (~7.5 h)
#   Llama-3.2-1B-Instruct  fp16 :  ~25 min / dataset    -> ~75 min total
#   Llama-3.2-3B-Instruct  fp16 :  ~50 min / dataset    -> ~150 min total
#
# Plus per-dataset NLI clustering (~15 min/1000 Q) and SEPs probes (~3 min).
# Grand total estimate: ~16 h end-to-end on 5070 alone. Run overnight.
#
# Resume-safe: each step skips already-done outputs. Restart at any time.

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
ROOT=~/experiments/dl_team_v2/01_se_seps
LOG=$ROOT/runs/sweep.log
mkdir -p "$ROOT/runs"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$ROOT/code"

MODELS_FP16=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)
MODEL_4BIT=("Qwen/Qwen2.5-7B-Instruct")
DATASETS=("triviaqa" "nq_open" "squad")

echo "=== Phase 1 full sweep starting $(date) ===" | tee -a "$LOG"

# fp16 models first (faster)
python run_phase1.py \
  --models "${MODELS_FP16[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --summary-name sweep_summary_fp16.json 2>&1 | tee -a "$LOG"

# 4-bit Qwen 7B
python run_phase1.py \
  --models "${MODEL_4BIT[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --four-bit "${MODEL_4BIT[@]}" \
  --summary-name sweep_summary_4bit.json 2>&1 | tee -a "$LOG"

echo "=== Phase 1 full sweep done $(date) ===" | tee -a "$LOG"
