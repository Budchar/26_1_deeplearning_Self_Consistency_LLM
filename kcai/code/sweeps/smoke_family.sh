#!/usr/bin/env bash
# Sweep A smoke test: pythia-1.4b × triviaqa × 10 questions.
set -euo pipefail
VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
ROOT=~/experiments/dl_team_v2/06_sweep_a_family
mkdir -p "$ROOT/smoke"
source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export PYTHONUNBUFFERED=1
cd "$PHASE1_CODE"
python run_phase1.py \
  --models "EleutherAI/pythia-1.4b-deduped" \
  --datasets triviaqa \
  --n 10 --n-samples 5 \
  --out-root "$ROOT/smoke" \
  --summary-name smoke.json
