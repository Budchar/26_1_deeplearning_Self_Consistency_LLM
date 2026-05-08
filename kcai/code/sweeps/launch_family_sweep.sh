#!/usr/bin/env bash
# Sweep A — 5개 패밀리 (~1.3B 등급) 비교. 같은 파라미터 등급에서 패밀리만 변경.
#
# 모델:
#   Pythia-1.4B (Pile, GPT-NeoX 계열)
#   Llama-3.2-1B-Instruct (Llama)
#   Qwen2.5-1.5B-Instruct (Qwen)
#   OPT-1.3B (Meta OPT, 2022)
#   GPT-Neo-1.3B (EleutherAI, 2021)
#
# 데이터셋: triviaqa, nq_open, squad (각 1000q)
# 모두 fp16 (~1.3B는 5070 12GB에 무난)
# 예상 시간: ~40 GPU-h (각 모델 25-30 min/dataset × 3 datasets × 5 models + NLI clustering)
#
# 재현성: run_phase1.py가 sample_generator.write_meta() 호출하므로 자동 self-logging.
# Resume-safe: 이미 끝난 (model, dataset) skip.

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
ROOT=~/experiments/dl_team_v2/06_sweep_a_family
LOG=$ROOT/results/family_sweep.log
mkdir -p "$ROOT/runs" "$ROOT/results"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE1_CODE"

MODELS_FP16=(
  "EleutherAI/pythia-1.4b-deduped"
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-1.5B"
  "facebook/opt-1.3b"
  "EleutherAI/gpt-neo-1.3B"
)
DATASETS=("triviaqa" "nq_open" "squad")

echo "=== Sweep A family $(date) ===" | tee -a "$LOG"

python run_phase1.py \
  --models "${MODELS_FP16[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --out-root "$ROOT" \
  --summary-name family_sweep_summary.json 2>&1 | tee -a "$LOG"

echo "=== Sweep A done $(date) ===" | tee -a "$LOG"
