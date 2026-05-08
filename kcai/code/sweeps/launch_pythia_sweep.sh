#!/usr/bin/env bash
# Sweep B — Pythia size scan. Reuses 01_se_seps/code/run_phase1.py with
# --out-root pointing here. Same family (EleutherAI/pythia-deduped),
# same training data, same tokenizer (GPT-NeoX); only model size varies.
#
# Sizes: 70m, 160m, 410m, 1b, 1.4b, 2.8b (fp16) + 6.9b (bnb-4bit on RTX 5070 12GB).
# Datasets: triviaqa, nq_open, squad. N=10 samples, 1000 questions each.
#
# Wall-time estimate on 5070 (12GB), max_new=64, N=10:
#   pythia-70m   fp16  :  ~6 min  / dataset  ->  ~18 min  total   (~0.3 h)
#   pythia-160m  fp16  :  ~8 min  / dataset  ->  ~24 min  total   (~0.4 h)
#   pythia-410m  fp16  :  ~14 min / dataset  ->  ~42 min  total   (~0.7 h)
#   pythia-1b    fp16  :  ~22 min / dataset  ->  ~66 min  total   (~1.1 h)
#   pythia-1.4b  fp16  :  ~28 min / dataset  ->  ~84 min  total   (~1.4 h)
#   pythia-2.8b  fp16  :  ~55 min / dataset  ->  ~165 min total   (~2.8 h)
#   pythia-6.9b  4bit  :  ~150 min / dataset ->  ~450 min total   (~7.5 h)
#   NLI clustering (~15 min / 1000Q / dataset) x 7 models x 3 ds  (~5.3 h)
#   SEPs probes + adaptive (~5 min / pair) x 21 pairs             (~1.8 h)
# ----------------------------------------------------------------------
# Grand total estimate: ~50 GPU-hours wall on 5070 alone. Multi-night run.
#
# Resume-safe: each step skips already-done outputs. Re-run this script
# any time to pick up where it left off.

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
SWEEP_ROOT=~/experiments/dl_team_v2/04_sweep_b_pythia
LOG=$SWEEP_ROOT/runs/sweep.log
mkdir -p "$SWEEP_ROOT/runs" "$SWEEP_ROOT/results"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE1_CODE"

MODELS_FP16=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-160m-deduped"
  "EleutherAI/pythia-410m-deduped"
  "EleutherAI/pythia-1b-deduped"
  "EleutherAI/pythia-1.4b-deduped"
  "EleutherAI/pythia-2.8b-deduped"
)
MODEL_4BIT=("EleutherAI/pythia-6.9b-deduped")
DATASETS=("triviaqa" "nq_open" "squad")

echo "=== Sweep B (Pythia) starting $(date) ===" | tee -a "$LOG"

# fp16 group: 70m -> 2.8b
python run_phase1.py \
  --models "${MODELS_FP16[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --out-root "$SWEEP_ROOT" \
  --summary-name sweep_b_summary_fp16.json 2>&1 | tee -a "$LOG"

# 4-bit group: 6.9b
python run_phase1.py \
  --models "${MODEL_4BIT[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --four-bit "${MODEL_4BIT[@]}" \
  --out-root "$SWEEP_ROOT" \
  --summary-name sweep_b_summary_4bit.json 2>&1 | tee -a "$LOG"

echo "=== Sweep B (Pythia) done $(date) ===" | tee -a "$LOG"
