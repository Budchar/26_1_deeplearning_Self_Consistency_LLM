#!/usr/bin/env bash
# 옵션 C — Instruct 패밀리 안에서 다중 사이즈 sweep.
#
# 옵션 B 결과를 본 후 사용자가 결정해서 launch.
# Phase 1 (Llama 1B/3B Instruct + Qwen 1.5B/3B/7B Instruct) + 본 옵션의 추가 사이즈 →
# Instruct 패밀리 안에서의 깨끗한 스케일링 효과 측정.
#
# 추가 모델 (3종, ~28 GPU-h):
#   Qwen/Qwen2.5-0.5B-Instruct        (fp16, ~1h × 3 datasets = 3h)
#   meta-llama/Llama-3.1-8B-Instruct  (4-bit, ~2.5h × 3 = 7.5h)
#   Qwen/Qwen2.5-14B-Instruct         (4-bit, ~3.5h × 3 = 10.5h)
#   + NLI clustering aggregate ~5h
#
# Phase 1 + 옵션 C → 합쳐 분석:
#   Llama: 1B / 3B / 8B (3점)
#   Qwen:  0.5B / 1.5B / 3B / 7B / 14B (5점)
# → Instruct 패밀리 사이즈 효과 8점 곡선
#
# 사용:
#   nohup bash /home/kcai/experiments/dl_team_v2/scripts/option_c_launcher.sh \
#     > /tmp/option_c.log 2>&1 &
#
# Resume-safe: run_phase1.py가 done id check.

set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
PHASE1_CODE=~/experiments/dl_team_v2/01_se_seps/code
ROOT=~/experiments/dl_team_v2/08_option_c_instruct_scale
LOG=$ROOT/results/option_c.log
mkdir -p "$ROOT/runs" "$ROOT/results"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$PHASE1_CODE"

MODELS_FP16=(
  "Qwen/Qwen2.5-0.5B-Instruct"
)
MODELS_4BIT=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
)
DATASETS=("triviaqa" "nq_open" "squad")

NOTIFY=/home/kcai/experiments/dl_team_v2/scripts/notify_done.sh
SUMMARIZE_PY="$VENV/bin/python /home/kcai/experiments/dl_team_v2/scripts/summarize_step.py"

echo "=== Option C start $(date) ===" | tee -a "$LOG"

# fp16 작은 모델 먼저
python run_phase1.py \
  --models "${MODELS_FP16[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --out-root "$ROOT" \
  --summary-name option_c_fp16.json 2>&1 | tee -a "$LOG"

# 4-bit 큰 모델
python run_phase1.py \
  --models "${MODELS_4BIT[@]}" \
  --datasets "${DATASETS[@]}" \
  --n 1000 --n-samples 10 \
  --four-bit "${MODELS_4BIT[@]}" \
  --out-root "$ROOT" \
  --summary-name option_c_4bit.json 2>&1 | tee -a "$LOG"

# Phase 1 데이터와 합쳐 confident-wrong 재분석
python /home/kcai/experiments/dl_team_v2/scripts/analyze_confident_wrong.py \
  --runs-root "$ROOT/runs" \
  --out "$ROOT/results/confident_wrong" 2>&1 | tee -a "$LOG"

echo "=== Option C done $(date) ===" | tee -a "$LOG"

# 이메일
BODY=/tmp/option_c_body_$$.md
cat > "$BODY" <<EOF
## 옵션 C — Instruct 패밀리 다중 사이즈 sweep 종료

추가된 3개 모델:
- Qwen2.5-0.5B-Instruct
- Llama-3.1-8B-Instruct (4-bit)
- Qwen2.5-14B-Instruct (4-bit)

Phase 1과 합치면:
- Llama 1B / 3B / 8B (3점 곡선)
- Qwen 0.5B / 1.5B / 3B / 7B / 14B (5점 곡선)

새 confident-wrong 분석은 \`$ROOT/results/confident_wrong/\` 에 있습니다.
Instruct 패밀리 안에서의 사이즈 효과를 깨끗하게 비교 가능 — Sweep B(Pythia base) 와의 차이는 "instruction tuning이 환각 검출에 어떤 영향을 주는가"의 답.
EOF

bash "$NOTIFY" "옵션 C Instruct 사이즈" "SUCCESS" "Llama+Qwen Instruct 8점 사이즈 곡선 완료" "$BODY"
