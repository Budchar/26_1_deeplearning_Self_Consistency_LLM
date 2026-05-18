#!/bin/bash
# 전체 실험 파이프라인 순차 실행
# Usage: bash scripts/run_all.sh [model] [n_samples]
# 예시:  bash scripts/run_all.sh exaone 200

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-exaone}"
N="${2:-200}"
N_GEN=10

echo "========================================"
echo " Hallucination Detection — Full Pipeline"
echo " Model   : $MODEL"
echo " Samples : $N  |  Generations: $N_GEN"
echo "========================================"

echo ""
echo "[Step 0] Download datasets (skip if cached)"
python scripts/download_datasets.py

echo ""
echo "[Exp 01] Token Entropy"
python experiments/01_token_entropy/run.py \
    --model "$MODEL" --dataset triviaqa --n_samples "$N"
python experiments/01_token_entropy/run.py \
    --model "$MODEL" --dataset truthfulqa --n_samples "$N"

echo ""
echo "[Exp 02] Semantic Entropy"
python experiments/02_semantic_entropy/run.py \
    --model "$MODEL" --dataset triviaqa --n_samples $((N/2)) --n_gen "$N_GEN"
python experiments/02_semantic_entropy/run.py \
    --model "$MODEL" --dataset truthfulqa --n_samples $((N/2)) --n_gen "$N_GEN"

echo ""
echo "[Exp 03] Self-Consistency"
python experiments/03_self_consistency/run.py \
    --model "$MODEL" --dataset triviaqa --n_samples "$N" --n_gen "$N_GEN"
python experiments/03_self_consistency/run.py \
    --model "$MODEL" --dataset truthfulqa --n_samples "$N" --n_gen "$N_GEN"

echo ""
echo "[Exp 04] Calibration (ECE)"
python experiments/04_calibration/run.py \
    --model "$MODEL" --dataset triviaqa --n_samples "$N"
python experiments/04_calibration/run.py \
    --model "$MODEL" --dataset truthfulqa --n_samples "$N"

echo ""
echo "[Exp 05] Verbalized Confidence"
python experiments/05_verbalized_confidence/run.py \
    --model "$MODEL" --dataset triviaqa --n_samples "$N"
python experiments/05_verbalized_confidence/run.py \
    --model "$MODEL" --dataset truthfulqa --n_samples "$N"

echo ""
echo "[Analysis] Generating figures..."
python experiments/01_token_entropy/analyze.py
python experiments/02_semantic_entropy/analyze.py

echo ""
echo "========================================"
echo " DONE. Results: results/raw/"
echo "        Figures: results/figures/"
echo "========================================"
