"""REPORT.md → PDF 변환 (markdown → HTML → weasyprint PDF)."""
import sys
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

ROOT = Path(__file__).parent.parent
md_path = ROOT / "REPORT.md"
pdf_path = ROOT / "REPORT.pdf"

md_text = md_path.read_text(encoding="utf-8")

# markdown → HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "nl2br", "toc"],
)

CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

@page {
    size: A4;
    margin: 2.5cm 2.2cm 2.5cm 2.2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: 'Noto Sans KR', 'DejaVu Sans', sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    font-weight: 700;
    text-align: center;
    margin-top: 0;
    margin-bottom: 4pt;
    color: #1a1a2e;
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 8pt;
}

h2 {
    font-size: 13.5pt;
    font-weight: 700;
    color: #16213e;
    margin-top: 24pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3pt;
}

h3 {
    font-size: 11.5pt;
    font-weight: 700;
    color: #0f3460;
    margin-top: 14pt;
    margin-bottom: 4pt;
}

h4 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #333;
    margin-top: 10pt;
}

p { margin: 6pt 0; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead tr {
    background-color: #1a1a2e;
    color: white;
}

th {
    padding: 5pt 7pt;
    text-align: left;
    font-weight: 700;
}

td {
    padding: 4pt 7pt;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) { background-color: #f7f8fc; }
tr:hover { background-color: #eef2ff; }

/* Code blocks */
pre {
    background-color: #f4f4f8;
    border: 1px solid #ddd;
    border-left: 4px solid #0f3460;
    padding: 10pt;
    font-size: 8.5pt;
    font-family: 'Courier New', monospace;
    overflow-x: auto;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-wrap: break-word;
}

code {
    background-color: #f4f4f8;
    padding: 1pt 4pt;
    border-radius: 3pt;
    font-size: 9pt;
    font-family: 'Courier New', monospace;
}

pre code {
    background: none;
    padding: 0;
}

/* Blockquote / key findings */
blockquote {
    border-left: 4px solid #e74c3c;
    margin: 10pt 0;
    padding: 6pt 12pt;
    background: #fff8f8;
    color: #333;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 18pt 0;
}

/* Lists */
ul, ol {
    margin: 6pt 0;
    padding-left: 18pt;
}

li { margin-bottom: 3pt; }

strong { color: #0f3460; }

/* Emoji / special chars */
em { color: #555; }
"""

full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>LLM Hallucination 사전 탐지 및 Overconfidence 분석</title>
</head>
<body>
{html_body}
</body>
</html>"""

font_config = FontConfiguration()
css = CSS(string=CSS_STYLE, font_config=font_config)

print(f"Converting {md_path} → {pdf_path} ...")
HTML(string=full_html, base_url=str(ROOT)).write_pdf(
    pdf_path,
    stylesheets=[css],
    font_config=font_config,
)
print(f"Done: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
