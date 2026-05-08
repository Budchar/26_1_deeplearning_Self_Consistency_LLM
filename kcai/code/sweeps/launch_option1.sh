#!/usr/bin/env bash
# Option 1 — Pile/EleutherAI 효과 분리 검증.
#
# 추가 3 모델 (Sweep A 5 모델과 합쳐 8 패밀리 비교):
#   cerebras/Cerebras-GPT-1.3B                — Pile 학습, 非EleutherAI
#   allenai/OLMo-1B-hf                         — 非Pile (Dolma), 非EleutherAI
#   TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T — SlimPajama, Llama-style
#
# 데이터셋: triviaqa, nq_open, squad (각 1000 질문, N=10)
# 모두 fp16 (~1.1-1.3B 5070 12GB 무난)
# 예상: ~40 GPU-h
#
# Resume-safe: run_phase1.py 가 done id check.

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
ROOT=~/experiments/dl_team_v2/08_option1_pile_check
LOG=$ROOT/results/option1.log
mkdir -p "$ROOT/runs" "$ROOT/results"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE1_CODE"

MODELS_FP16=(
  "cerebras/Cerebras-GPT-1.3B"
  "allenai/OLMo-1B-hf"
  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)
DATASETS=("triviaqa" "nq_open" "squad")

echo "=== Option 1 start $(date) ===" | tee -a "$LOG"

python run_phase1.py \
  --models "${MODELS_FP16[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --out-root "$ROOT" \
  --summary-name option1_summary.json 2>&1 | tee -a "$LOG"

echo "=== Option 1 done $(date) ===" | tee -a "$LOG"

# 자동 이메일
NOTIFY=/home/kcai/experiments/dl_team_v2/scripts/notify_done.sh
SUMMARIZE_PY="$VENV/bin/python /home/kcai/experiments/dl_team_v2/scripts/summarize_step.py"

# summary 우선 단순 텍스트로
BODY=/tmp/orch_option1_$$.md
cat > "$BODY" <<EOF
# 옵션 1 — Pile/EleutherAI 효과 분리 sweep 완료

## 추가된 3 모델
- cerebras/Cerebras-GPT-1.3B (Pile, 非EleutherAI) — "Pile vs EleutherAI" 분리 핵심
- allenai/OLMo-1B-hf (Dolma, 非Pile, 非EleutherAI) — non-Pile baseline
- TinyLlama/TinyLlama-1.1B (SlimPajama, Llama-style) — non-Pile, Llama family

## 검증 가능한 가설

| 시나리오 | 결론 |
|---|---|
| Cerebras gap 큼, Olmo/TinyLlama gap 작음 | **Pile이 원인** 확정 |
| Cerebras gap 작음, 다 비슷 | **EleutherAI 아키텍처/팀이 원인** |
| 전부 작음 | **다른 요인** (instruction 등) |

## 결과 위치
- /home/kcai/experiments/dl_team_v2/08_option1_pile_check/runs/

## 전체 비교 (Sweep A 5 + 옵션 1 3 = 8 패밀리)
SEPs gap 패턴이 명확해지면 KCI paper 핵심 보강됨.
EOF

bash "$NOTIFY" "옵션 1 Pile 분리 검증" "SUCCESS" "Cerebras + Olmo + TinyLlama sweep 완료" "$BODY"
