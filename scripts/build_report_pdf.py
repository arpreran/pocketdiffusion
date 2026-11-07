"""Render Solution_Report_P3.md to an A4 PDF (Markdown -> HTML -> Microsoft Edge headless print).

    python scripts/build_report_pdf.py                 # -> build/Solution_Report_P3.pdf
    python scripts/build_report_pdf.py --final         # refuses if TEAM-TODO blocks / placeholders remain

HTML comments (AI markers, notes) are not rendered. Images are embedded from the repository.
Prints the page count at the end.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Solution_Report_P3.md"
BUILD = ROOT / "build"
EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; margin: 0 0 10pt; color: #2b1a66; }
h2 { font-size: 15pt; margin: 18pt 0 6pt; color: #2b1a66; border-bottom: 1.5px solid #c9c1ea; padding-bottom: 3pt;
     break-after: avoid; }
h2.appendix { break-before: page; }
h3 { font-size: 12.5pt; margin: 14pt 0 5pt; color: #3a2a80; break-after: avoid; }
h4 { font-size: 11pt; margin: 12pt 0 4pt; color: #333; break-after: avoid; }
h5 { font-size: 10.5pt; margin: 10pt 0 3pt; color: #444; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.8pt; break-inside: auto; }
th, td { border: 1px solid #c8c8d0; padding: 3pt 5pt; vertical-align: top; }
th { background: #eeebfa; }
tr { break-inside: avoid; }
img { max-width: 100%; max-height: 118mm; display: block; margin: 6pt auto; break-inside: avoid; }
td img { max-height: 45mm; }
p > img:only-child { margin: 8pt auto; }
blockquote { margin: 6pt 0; padding: 6pt 10pt; background: #fff7e0; border-left: 4px solid #f2b705; font-size: 9.5pt; }
pre { background: #f5f5f8; border: 1px solid #ddd; padding: 6pt 8pt; font-size: 8.3pt; white-space: pre-wrap;
      break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; }
hr { border: 0; border-top: 1px solid #ddd; margin: 10pt 0; }
a { color: #3a2a80; text-decoration: none; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    text = SRC.read_text(encoding="utf-8")
    if args.final and ("<!-- TEAM-TODO -->" in text or re.search(r"`\[[^\]]*\]`", text.split("-->", 1)[-1])):
        print("Not final: TEAM-TODO blocks or [placeholders] are still in the report.")
        return 1
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)                       # markers and notes
    html = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    html = html.replace("<h2>Appendices</h2>", "<h2 class='appendix'>Appendices</h2>")
    BUILD.mkdir(exist_ok=True)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><base href='{ROOT.as_uri()}/'>"
            f"<title>PocketDiffusion – Solution Report</title><style>{CSS}</style></head><body>{html}</body></html>")
    html_path = BUILD / "Solution_Report_P3.html"
    html_path.write_text(page, encoding="utf-8")
    edge = next((p for p in EDGE_PATHS if Path(p).exists()), shutil.which("msedge"))
    if not edge:
        print(f"HTML written to {html_path}; open it in a browser and print to PDF (Microsoft Edge not found).")
        return 0
    pdf = BUILD / "Solution_Report_P3.pdf"
    if pdf.exists():
        pdf.unlink()
    # separate profile folder: a normally open Edge window would otherwise swallow the headless run
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={BUILD / 'edge-profile'}", "--no-first-run",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=False, timeout=180,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # the Edge launcher can return before the renderer has finished writing the file
    import time
    last = -1
    for _ in range(120):
        size = pdf.stat().st_size if pdf.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1)
    if not pdf.exists():
        print("PDF was not created; open the HTML in a browser and print to PDF.")
        return 1
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print(f"PDF: {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB, {pages} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
