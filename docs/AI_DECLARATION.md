# AI-assistance declaration

The declaration that is submitted lives in **section 6 of [`Solution_Report_P3.md`](../Solution_Report_P3.md)**.
Keep it accurate for your team; do not under-state it.

## How the 30 % report rule is checked

* Text drafted by the AI assistant is wrapped in `<!-- AI-DRAFTED -->` … `<!-- /AI-DRAFTED -->`
  (invisible in the rendered report). Placeholders for the team are wrapped in
  `<!-- TEAM-TODO -->` … `<!-- /TEAM-TODO -->`.
* `python scripts/ai_share.py Solution_Report_P3.md` counts prose words inside and outside the
  AI blocks and prints the AI share. It also lists the placeholders that are still open.
* Not counted as prose: headings, tables generated from `results/*.json`, code blocks, images,
  the submission checklist and the template's own field labels. This is our interpretation of
  "report content". If your instructor counts differently, follow the instructor.
* If you rewrite an AI-drafted block in your own words, remove its markers. If you paste AI text
  anywhere else, wrap it in markers so the count stays honest.

## What the AI assistant did (summary for the report)

* System design discussions; most of the source code (`pocketdiff/`, `backend/`, `frontend/`,
  `eval/`, `scripts/`, `app.py`, `serve.py`, `generate.py`).
* The 58 prompts of the compositional benchmark (`prompts/compositional.json`).
* `README.md`, `docs/*`, the report outline, the fact notes in the TEAM-TODO blocks and the
  AI-drafted blocks of the report.

All metrics come from the scripts in `eval/` run on our hardware; nothing is typed in by hand.
