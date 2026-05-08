#!/usr/bin/env bash
# Option 3 — 같은 GPT-2 124M 아키텍처 + 다른 학습 데이터 (Pile vs OpenWebText vs C4).
# 데이터 효과 단독 분리 검증.
#
# Phase 2 train.py 재사용. data_loader 만 dataset별로 다르게.
#
# 흐름:
#   1. 데이터 shard 준비 (Pile, C4 — OpenWebText 이미 있음)
#   2. 3 GPT-2 124M from-scratch 학습 (각 50K step, 동일 하이퍼)
#   3. 학습 후 sample_generator 로 환각 검출 측정
#   4. SE + SEPs probe + metrics
#   5. 메일 발송
#
# 시간 추정 (5070):
#   shard prep: 2-4h
#   training: 3 × 4h = 12h
#   inference + analysis: 4-6h
#   합 ~20h
set -euo pipefail

VENV=~/experiments/dl_team_v2/shared/.venv
ROOT=~/experiments/dl_team_v2/10_option3_data_effect
PHASE2_TRAIN=~/experiments/dl_team_v2/02_c2_sinks/code/train.py
PHASE2_DATA_LOADER_DIR=~/experiments/dl_team_v2/02_c2_sinks/code
SCRIPTS=~/experiments/dl_team_v2/scripts
LOG=$ROOT/results/option3.log
mkdir -p "$ROOT/runs" "$ROOT/results" "$ROOT/checkpoints"

source "$VENV/bin/activate"
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Option 3 start $(date) ===" | tee -a "$LOG"

# ─── Step 1: Data shards 준비 ───
echo "" | tee -a "$LOG"
echo "==== [$(date)] Step 1: prepare Pile / C4 shards ====" | tee -a "$LOG"
"$VENV/bin/python" "$SCRIPTS/prepare_dataset_shards.py" 2>&1 | tee -a "$LOG"

# ─── Step 2: 3 from-scratch 학습 ───
# data_loader.py가 GPT2_SHARD_DIR 환경변수 읽음 (BC 패치됨)
declare -A SHARD_DIRS
SHARD_DIRS[owt]="/home/kcai/experiments/dl_team_v2/_data/owt_shards"
SHARD_DIRS[pile]="/home/kcai/experiments/dl_team_v2/_data/pile_shards"
SHARD_DIRS[c4]="/home/kcai/experiments/dl_team_v2/_data/c4_shards"

DATASETS=("owt" "pile" "c4")
for DS in "${DATASETS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "==== [$(date)] Step 2.${DS}: train GPT-2 124M on $DS ====" | tee -a "$LOG"
  RUN_NAME="softmax_${DS}"
  export GPT2_SHARD_DIR="${SHARD_DIRS[$DS]}"
  echo "  GPT2_SHARD_DIR=$GPT2_SHARD_DIR" | tee -a "$LOG"
  "$VENV/bin/python" "$PHASE2_TRAIN" \
    --variant softmax \
    --tag opt3_${DS} \
    --depth 12 \
    --width 768 \
    --n-head 12 \
    --run-name "$RUN_NAME" \
    --out "$ROOT" \
    --max-steps 50000 \
    --micro-bs 4 \
    --grad-accum 4 \
    2>&1 | tee -a "$LOG"
done

# ─── Step 3: 환각 검출 평가 ───
# (sweep_cd_eval.py 같은 스크립트 재사용 가능. 이미 from-scratch 처리 검증됨)
echo "" | tee -a "$LOG"
echo "==== [$(date)] Step 3: SEPs probe evaluation ====" | tee -a "$LOG"
"$VENV/bin/python" "$SCRIPTS/sweep_cd_eval.py" \
  --device cuda \
  --n 1000 --dataset triviaqa \
  --out "$ROOT/results/option3_h3_eval.json" 2>&1 | tee -a "$LOG"

# ─── Step 4: 결과 요약 + 이메일 ───
echo "" | tee -a "$LOG"
echo "==== [$(date)] Option 3 done, sending email ====" | tee -a "$LOG"
BODY=/tmp/orch_option3_$$.md
cat > "$BODY" <<EOF
옵션 3 — 자체 from-scratch 데이터 효과 검증 완료

같은 GPT-2 124M 아키텍처에 OpenWebText / Pile / C4 각각 from-scratch 학습.
데이터만 다른 controlled experiment. 가장 깨끗한 isolation.

3 모델 × 1 데이터셋 (TriviaQA) = 3 cells 측정.

검증 가설:
- Pile-trained 모델이 OpenWebText/C4 모델보다 SEPs gap 큰가?
- 만약 그렇다면 Option 1의 Pile 발견이 데이터 효과 진짜 확정.
- 만약 비슷하다면 다른 요인 (모델 사이즈 효과 등).

상세 결과는 첨부 PDF 참고.
EOF

bash "$SCRIPTS/notify_done.sh" "옵션 3 데이터 효과 검증" "SUCCESS" "OWT/Pile/C4 from-scratch 비교 완료" "$BODY"

echo "=== Option 3 done $(date) ===" | tee -a "$LOG"
