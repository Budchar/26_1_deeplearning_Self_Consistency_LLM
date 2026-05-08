#!/usr/bin/env bash
# md2pdf.sh — Markdown → PDF 변환 (한글 지원, 스마트폰 가독성).
#
# 사용:
#   md2pdf.sh <input.md> <output.pdf>
#
# 흐름: MD → HTML (Python markdown + Korean CSS) → PDF (Chrome headless)
set -euo pipefail

IN="${1:?usage: md2pdf.sh <input.md> <output.pdf>}"
OUT="${2:?missing output.pdf}"
PY=/home/kcai/experiments/dl_team_v2/shared/.venv/bin/python
TMP_HTML="$(mktemp /tmp/md2pdf_XXXXXX.html)"
trap 'rm -f "$TMP_HTML"' EXIT

# MD → HTML with Korean-friendly CSS + GitHub-style tables
"$PY" - "$IN" "$TMP_HTML" <<'PYEOF'
import sys, markdown
md_path, html_path = sys.argv[1], sys.argv[2]
with open(md_path, encoding='utf-8') as f:
    md_text = f.read()
html_body = markdown.markdown(
    md_text,
    extensions=['extra', 'tables', 'fenced_code', 'codehilite'],
)
css = """
@page { size: A4; margin: 18mm 16mm; }
html, body {
  font-family: 'Noto Sans CJK KR', 'NanumSquareRound', 'Apple SD Gothic Neo',
               'Malgun Gothic', sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #111;
  word-break: keep-all;
  word-wrap: break-word;
}
h1 { font-size: 18pt; border-bottom: 2px solid #444; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 14pt; border-bottom: 1px solid #999; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 12pt; margin-top: 14px; }
h4 { font-size: 11pt; margin-top: 10px; }
p, li { font-size: 11pt; }
code { font-family: 'D2Coding', 'Noto Sans Mono CJK KR', monospace;
       background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f4f4f4; padding: 8px 10px; border-radius: 4px; overflow-x: auto;
      font-size: 9.5pt; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 8px 0; width: auto; }
th, td { border: 1px solid #888; padding: 5px 9px; font-size: 10pt; text-align: left; }
th { background: #eee; font-weight: 600; }
hr { border: 0; border-top: 1px solid #ccc; margin: 16px 0; }
blockquote { border-left: 3px solid #888; padding-left: 10px; color: #555; margin: 10px 0; }
strong { font-weight: 600; }
"""
out = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><style>{css}</style></head>
<body>{html_body}</body></html>"""
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(out)
PYEOF

# HTML → PDF via Chrome headless
google-chrome \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --hide-scrollbars \
  --no-pdf-header-footer \
  --print-to-pdf-no-header \
  --print-to-pdf="$OUT" \
  "file://$TMP_HTML" 2>/dev/null

if [[ -s "$OUT" ]]; then
  echo "[md2pdf] $OUT ($(du -h "$OUT" | cut -f1))"
else
  echo "[md2pdf] ERROR: $OUT not created" >&2
  exit 1
fi
