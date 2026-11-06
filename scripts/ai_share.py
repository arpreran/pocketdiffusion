"""Measure the share of AI-written prose in the report (hackathon rule: at most 30 %).

The report marks text drafted by the AI assistant with HTML comments (invisible when rendered):

    <!-- AI-DRAFTED --> ... <!-- /AI-DRAFTED -->     counted as AI words
    <!-- TEAM-TODO -->  ... <!-- /TEAM-TODO -->      placeholders, not counted (must be replaced)

Every other prose word counts as team-written. Not counted at all: headings, tables (generated
from results/*.json), code blocks, images, HTML comments, the submission checklist and the
template's bold field labels.

Usage: python scripts/ai_share.py Solution_Report_P3.md [--limit 30]
Exit code 0 = PASS (share within the limit and no placeholders left), 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WORD = re.compile(r"[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*")
LABEL = re.compile(r"^\s*[-*]\s+\*\*[^*]+:\*\*")          # "- **Core Technical Challenge:**"
PLACEHOLDER = re.compile(r"`\[[^\]]*\]`")


def count(path: Path) -> dict:
    ai = team = todos = 0
    state = None            # None | "ai" | "todo"
    in_code = in_comment = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line == "<!-- AI-DRAFTED -->":
            state = "ai"
            continue
        if line == "<!-- TEAM-TODO -->":
            state, todos = "todo", todos + 1
            continue
        if line in ("<!-- /AI-DRAFTED -->", "<!-- /TEAM-TODO -->"):
            state = None
            continue
        # other HTML comments (single- or multi-line) are never counted
        line = re.sub(r"<!--.*?-->", "", line)
        if in_comment:
            if "-->" in line:
                in_comment = False
                line = line.split("-->", 1)[1]
            else:
                continue
        if "<!--" in line:
            in_comment = True
            line = line.split("<!--", 1)[0]
        if state == "todo" or not line:
            continue
        if line.startswith(("#", "|", "![", "---", "- [ ]", "- [x]", "- [X]")):   # headings, tables, checklist
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", line):           # template sub-labels, e.g. **Deployment Format**
            continue
        line = LABEL.sub("", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)            # keep link text, drop URLs
        n = len(WORD.findall(line))
        if state == "ai":
            ai += n
        else:
            team += n
    return {"ai": ai, "team": team, "todos": todos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, nargs="?", default=Path("Solution_Report_P3.md"))
    ap.add_argument("--limit", type=float, default=30.0)
    args = ap.parse_args()

    c = count(args.report)
    total = c["ai"] + c["team"]
    share = 100.0 * c["ai"] / total if total else 0.0
    placeholders = PLACEHOLDER.findall(args.report.read_text(encoding="utf-8"))
    print(f"report:               {args.report}")
    print(f"AI-drafted words:     {c['ai']}")
    print(f"team-written words:   {c['team']}")
    print(f"AI share of prose:    {share:.1f} %  (limit {args.limit:.0f} %)")
    print(f"TEAM-TODO blocks left: {c['todos']}")
    print(f"header fields left:   {len(placeholders)}")
    if c["todos"] or placeholders:
        # how much the team still has to write so the limit holds
        need = c["ai"] * (100.0 - args.limit) / args.limit
        print(f"-> NOT READY: replace the placeholders. To stay within {args.limit:.0f} %, the team text "
              f"must total at least {need:.0f} words (currently {c['team']}).")
        return 1
    if share > args.limit:
        print("-> FAIL: rewrite some AI-drafted blocks in your own words (and remove their markers).")
        return 1
    print("-> PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
