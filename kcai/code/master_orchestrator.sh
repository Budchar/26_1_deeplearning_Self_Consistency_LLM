#!/usr/bin/env bash
# master_orchestrator.sh — Phase 2 종료 후 자동 순차 실행 + 단계별 이메일.
#
# 흐름 (옵션 B):
#   1) Phase 2 종료 대기 (이미 끝났으면 즉시 진행)
#   2) Phase 2 분석 + 이메일
#   3) Sweep B (Pythia 7 사이즈) 실행 + 이메일
#   4) Sweep C (깊이단독 6) 실행 + 이메일
#   5) Sweep D (너비단독 5) 실행 + 이메일
#   6) Sweep C/D 평가 (from-scratch 모델 환각 검출 H3) + 이메일
#   7) Sweep A (5 패밀리) 실행 + 이메일
#   8) Phase 3 v2 실행 + 이메일
#   9) 최종 통합 + 이메일
#
# 옵션 C (Llama/Qwen 다중 사이즈) 는 사용자 결정 후 별도 launcher
#   bash /home/kcai/experiments/dl_team_v2/scripts/option_c_launcher.sh
#
# 사용:
#   nohup bash master_orchestrator.sh > /tmp/master_orchestrator.log 2>&1 &
#
# 중단:
#   kill <pid>
#
# Resume: 같은 명령 재실행. 각 단계가 idempotent (sweep 스크립트들이 done 체크).
#
set -uo pipefail

PY=/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python
SCRIPTS=/home/kcai/experiments/dl_team_v2/scripts
NOTIFY="$SCRIPTS/notify_done.sh"
SUMMARIZE="$PY $SCRIPTS/summarize_step.py"

# ============= helpers =============
phase2_running() {
  pgrep -f "02_c2_sinks/code/train.py" > /dev/null
}

run_step() {
  local NAME="$1"; shift
  local LAUNCHER="$1"; shift
  local STEP_KEY="$1"; shift  # for summarize_step

  echo "========================================="
  echo "[$(date)] STEP: $NAME"
  echo "========================================="

  local STATUS="SUCCESS"
  if [[ -n "$LAUNCHER" ]]; then
    if ! bash "$LAUNCHER"; then
      STATUS="FAIL"
    fi
  fi

  # 본문 생성 (실패해도 빈 파일이라도 만듦)
  local BODY=/tmp/orch_${STEP_KEY}_$$.md
  $SUMMARIZE "$STEP_KEY" --out "$BODY" || echo "(요약 생성 실패)" > "$BODY"

  # 이메일
  bash "$NOTIFY" "$NAME" "$STATUS" "단계 종료 — 본문 첨부" "$BODY" || true

  if [[ "$STATUS" == "FAIL" ]]; then
    echo "[$(date)] $NAME 실패. 중단."
    exit 1
  fi
}

# ============= 1. Phase 2 종료 대기 =============
echo "[$(date)] Phase 2 학습 종료 대기..."
while phase2_running; do
  sleep 120
done
echo "[$(date)] Phase 2 종료 감지 → 진행"

# ============= 2. Phase 2 분석 + 이메일 =============
PHASE2_ANALYZE=/home/kcai/experiments/dl_team_v2/02_c2_sinks/code/run_phase2.py
PHASE2_PLOT=/home/kcai/experiments/dl_team_v2/02_c2_sinks/code/plot_results.py

echo "[$(date)] Phase 2 분석 실행"
$PY "$PHASE2_ANALYZE" --stage analyze --tag full || echo "(analyze 결과 부분/실패)"
$PY "$PHASE2_PLOT" --tag full || echo "(plot 결과 부분/실패)"
BODY=/tmp/orch_phase2_$$.md
$SUMMARIZE phase2 --out "$BODY"
bash "$NOTIFY" "Phase 2 분석" "SUCCESS" "softmax/sigmoid/softpick/softplus 학습+분석 완료" "$BODY"

# ============= 3. Sweep B (Pythia) =============
run_step "Sweep B Pythia" \
  "/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/code/launch_pythia_sweep.sh" \
  "sweep_b"

# ============= 4. Sweep C (depth-only) =============
run_step "Sweep C 깊이단독" \
  "/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/code/launch_depth_sweep.sh" \
  "sweep_c"

# ============= 5. Sweep D (width-only) =============
run_step "Sweep D 너비단독" \
  "/home/kcai/experiments/dl_team_v2/07_sweep_d_width/code/launch_width_sweep.sh" \
  "sweep_d"

# ============= 6. Sweep C/D 평가 (H3 인과 검증) =============
echo "[$(date)] Sweep C/D 평가 (from-scratch 모델 환각 검출)"
$PY /home/kcai/experiments/dl_team_v2/scripts/sweep_cd_eval.py \
  --n 1000 --dataset triviaqa \
  --out /home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results/sweep_cd_h3_eval.json \
  || echo "(sweep_cd_eval 실패/부분)"
BODY=/tmp/orch_sweep_cd_eval_$$.md
$SUMMARIZE sweep_cd_eval --out "$BODY" || echo "(summary fail)" > "$BODY"
bash "$NOTIFY" "Sweep C/D 평가" "SUCCESS" "from-scratch 11개 모델 H3 인과 검증" "$BODY"

# ============= 7. Sweep A (family) =============
run_step "Sweep A 패밀리5종" \
  "/home/kcai/experiments/dl_team_v2/06_sweep_a_family/code/launch_family_sweep.sh" \
  "sweep_a"

# ============= 6. Phase 3 v2 =============
run_step "Phase 3 v2 Grokking" \
  "/home/kcai/experiments/dl_team_v2/03_c3_grokking_v2/code/launch_phase3_v2_gpu.sh" \
  "phase3v2"

# ============= 7. 최종 통합 =============
BODY=/tmp/orch_final_$$.md
$SUMMARIZE final --out "$BODY"
bash "$NOTIFY" "최종 통합 보고서" "SUCCESS" "3주 사전실험 종료" "$BODY"

echo "[$(date)] 모든 단계 종료"
