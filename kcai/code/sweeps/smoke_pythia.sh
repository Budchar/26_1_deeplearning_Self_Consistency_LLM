#!/usr/bin/env bash
# Smoke test: pythia-70m x triviaqa, 10 questions, N=10 samples.
# Verifies pipeline works (chat template fallback, GPT-NeoX tokenizer, hidden
# state shapes, NLI clustering, SEPs probes, adaptive SE) before launching
# the full ~50 h sweep.
#
# Outputs land at:
#   04_sweep_b_pythia/runs/EleutherAI__pythia-70m-deduped/triviaqa/
#   04_sweep_b_pythia/results/smoke_pythia.json

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
SWEEP_ROOT=~/experiments/dl_team_v2/04_sweep_b_pythia
LOG=$SWEEP_ROOT/runs/smoke.log
mkdir -p "$SWEEP_ROOT/runs" "$SWEEP_ROOT/results"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE1_CODE"

echo "=== Sweep B smoke (pythia-70m x triviaqa, n=10) starting $(date) ===" | tee -a "$LOG"

python run_phase1.py \
  --models "EleutherAI/pythia-70m-deduped" \
  --datasets triviaqa \
  --n 10 --n-samples 10 --limit 10 \
  --out-root "$SWEEP_ROOT" \
  --summary-name smoke_pythia.json 2>&1 | tee -a "$LOG"

echo "=== Sweep B smoke done $(date) ===" | tee -a "$LOG"
echo
echo "Inspect outputs:"
echo "  ls $SWEEP_ROOT/runs/EleutherAI__pythia-70m-deduped/triviaqa/"
echo "  cat $SWEEP_ROOT/results/smoke_pythia.json"
