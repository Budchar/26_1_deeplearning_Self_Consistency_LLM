#!/usr/bin/env bash
# notify_done.sh — 단계 완료 시 파인만식 요약 이메일 발송.
#
# 사용:
#   notify_done.sh "<단계명>" "<상태(SUCCESS|FAIL)>" "<짧은 요약>" [<상세 본문 파일 경로>]
#
# 예:
#   notify_done.sh "Sweep B Pythia" SUCCESS "7개 사이즈 × 3 데이터셋 = 21개 케이스 완료" \
#     /path/to/summary.md
#
# 환경변수로 덮어쓰기 가능:
#   NOTIFY_TO        수신 이메일 (기본: kcai3705@gmail.com)
#   NOTIFY_FROM      발신 이메일 (send_mail.py 기본 사용)
#
# Feynman 요약 가이드: 상세 본문 파일이 있으면 그대로 첨부 + 본문에 인용.
# 없으면 본문 = 단계명 + 상태 + 짧은 요약 + 다음 단계 안내.
set -euo pipefail

STEP="${1:?usage: notify_done.sh <step> <status> <summary> [body_file]}"
STATUS="${2:?missing status}"
SUMMARY="${3:?missing summary}"
BODY_FILE="${4:-}"
NOTIFY_TO="${NOTIFY_TO:-repairer5812@gmail.com}"
SEND_MAIL_PY="/home/kcai/Nextcloud/2. 계속관리/email_send/send_mail.py"
PY="/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python"

TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
SUBJECT="[딥러닝 사전실험] ${STEP} — ${STATUS} (${TS})"

TMP_BODY="$(mktemp /tmp/notify_done_XXXXXX.txt)"
trap 'rm -f "$TMP_BODY"' EXIT

{
  echo "단계: ${STEP}"
  echo "상태: ${STATUS}"
  echo "시각: ${TS}"
  echo "호스트: $(hostname)"
  echo
  echo "요약: ${SUMMARY}"
  echo
  echo "상세 보고서는 첨부 PDF 파일을 참고하세요 (markdown 표가 메일 본문에서 깨지는 문제로 PDF 첨부만 사용)."
  echo
  if [[ "${STATUS}" == "SUCCESS" ]]; then
    echo "단계 정상 종료. 다음 단계가 자동 진행되거나 수동 launch 대기 중."
  else
    echo "단계 실패. 자동 진행이 멈췄을 가능성. 로그 확인 + root cause 점검 필요."
  fi
} > "${TMP_BODY}"

ATTACH=()
if [[ -n "${BODY_FILE}" && -f "${BODY_FILE}" ]]; then
  # MD를 PDF로 변환해 첨부 (스마트폰 가독성).
  PDF_FILE="${BODY_FILE%.md}.pdf"
  if /home/kcai/experiments/dl_team_v2/scripts/md2pdf.sh "${BODY_FILE}" "${PDF_FILE}" 2>/dev/null && [[ -s "${PDF_FILE}" ]]; then
    # PDF + MD 둘 다 첨부 (--attach 반복)
    ATTACH=(--attach "${PDF_FILE}" --attach "${BODY_FILE}")
  else
    ATTACH=(--attach "${BODY_FILE}")
  fi
fi

"${PY}" "${SEND_MAIL_PY}" \
  --to "${NOTIFY_TO}" \
  --subject "${SUBJECT}" \
  --body-file "${TMP_BODY}" \
  "${ATTACH[@]}"

echo "[notify_done] sent to ${NOTIFY_TO}"
