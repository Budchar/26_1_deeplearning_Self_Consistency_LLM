#!/usr/bin/env bash
# Phase 1 보강 전용 orchestrator (Phase 2 중단 후).
#
# 흐름:
#   1) Sweep B (Pythia 7 사이즈) + 이메일
#   2) Sweep C (깊이단독 6) + 이메일
#   3) Sweep D (너비단독 5) + 이메일
#   4) Sweep C/D 평가 (H3 인과 검증) + 이메일
#   5) Sweep A (5 패밀리 base) + 이메일
#   6) 최종 통합 + 이메일
#
# Phase 2 분석·Phase 3 v2 모두 SKIP (Phase 2 literature 중복으로 중단).
#
# 사용:
#   nohup bash phase1_reinforcement_orchestrator.sh > /tmp/phase1_orch.log 2>&1 &
#
# Resume: 같은 명령 재실행 (각 단계 idempotent).

set -uo pipefail

PY=/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python
SCRIPTS=/home/kcai/experiments/dl_team_v2/scripts
NOTIFY="$SCRIPTS/notify_done.sh"
SUMMARIZE="$PY $SCRIPTS/summarize_step.py"

run_step() {
  local NAME="$1"; shift
  local LAUNCHER="$1"; shift
  local STEP_KEY="$1"; shift

  echo "========================================="
  echo "[$(date)] STEP: $NAME"
  echo "========================================="

  local STATUS="SUCCESS"
  if [[ -n "$LAUNCHER" ]]; then
    if ! bash "$LAUNCHER"; then
      STATUS="FAIL"
    fi
  fi

  local BODY=/tmp/orch_${STEP_KEY}_$$.md
  $SUMMARIZE "$STEP_KEY" --out "$BODY" || echo "(요약 생성 실패)" > "$BODY"

  bash "$NOTIFY" "$NAME" "$STATUS" "단계 종료 — 본문 첨부" "$BODY" || true

  if [[ "$STATUS" == "FAIL" ]]; then
    echo "[$(date)] $NAME 실패. 중단."
    exit 1
  fi
}

echo "[$(date)] Phase 1 보강 orchestrator 시작 (Phase 2 SKIP)"

# ============= 1. Sweep B (Pythia) =============
run_step "Sweep B Pythia" \
  "/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/code/launch_pythia_sweep.sh" \
  "sweep_b"

# ============= 2. Sweep C (depth-only) =============
run_step "Sweep C 깊이단독" \
  "/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/code/launch_depth_sweep.sh" \
  "sweep_c"

# ============= 3. Sweep D (width-only) =============
run_step "Sweep D 너비단독" \
  "/home/kcai/experiments/dl_team_v2/07_sweep_d_width/code/launch_width_sweep.sh" \
  "sweep_d"

# ============= 4. Sweep C/D 평가 (H3 인과 검증) =============
echo "[$(date)] Sweep C/D 평가"
$PY /home/kcai/experiments/dl_team_v2/scripts/sweep_cd_eval.py \
  --n 1000 --dataset triviaqa \
  --out /home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results/sweep_cd_h3_eval.json \
  || echo "(sweep_cd_eval 실패/부분)"
BODY=/tmp/orch_sweep_cd_eval_$$.md
$SUMMARIZE sweep_cd_eval --out "$BODY" || echo "(summary fail)" > "$BODY"
bash "$NOTIFY" "Sweep C/D 평가" "SUCCESS" "from-scratch 11개 모델 H3 인과 검증" "$BODY"

# ============= 5. Sweep A (패밀리 5종 base) =============
run_step "Sweep A 패밀리5종 base" \
  "/home/kcai/experiments/dl_team_v2/06_sweep_a_family/code/launch_family_sweep.sh" \
  "sweep_a"

# ============= 6. 최종 통합 =============
BODY=/tmp/orch_final_$$.md
$SUMMARIZE final --out "$BODY"
bash "$NOTIFY" "최종 통합 보고서" "SUCCESS" "Phase 1 보강 종료" "$BODY"

echo "[$(date)] 모든 단계 종료"
