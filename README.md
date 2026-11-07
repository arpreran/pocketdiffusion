# 📱 PocketDiffusion — Resource-Efficient Text-to-Image Generation for Mobile Devices

**Computer Vision Hackathon 2026 · Problem Statement 3 (P3)**

PocketDiffusion turns a natural-language prompt into an image while treating latency and memory
as first-class inputs. An adaptive planner reads the prompt and the device budget and picks
the distilled model, number of sampling steps, native resolution, decoder, weight precision and
number of candidates. A custom diffusion loop executes the plan with deadline-aware early exit
and an out-of-memory fallback chain. Classical computer vision screens and corrects artifacts,
re-ranks candidates and verifies that the requested objects, counts, colours and spatial
relations actually appear.

<!-- RESULTS-HIGHLIGHTS -->
## Results at a glance (RTX 4060 Laptop, 512×512, 200 unseen COCO prompts)

| system | CLIP-L ↑ | FID ↓ | latency ms ↓ | speed-up | peak MB ↓ | weights MB ↓ |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 26.43 | 150.8 | 13307 | 1× | 5065 | 4067 |
| + TAESD decoder | 25.27 | 158.2 | 257 | 52× | 2068 | 1879 |
| + distilled 6-layer text encoder | 25.21 | 157.0 | 360 | 37× | 1162 | 981 |
| SD-Turbo 1 step + TAESD | 26.27 | 150.3 | 85 | 156× | 2489 | 2305 |
| SD-Turbo 1 step + TAESD + INT8 | 26.28 | 150.7 | 104 | 127× | 1671 | 1483 |
| PocketDiffusion (adaptive, fast mode) | 26.29 | 147.3 | 83 | 160× | 2735 | 2305 |
| PocketDiffusion (adaptive, auto mode) | 26.72 | 147.4 | 155 | 86× | 3039 | 2305 |

Compositional benchmark (58 annotated prompts: objects / counts / colours / spatial relations): object F1 **0.958** vs 0.889 for the baseline; all constraints satisfied 59%, rising to **77%** with 8-candidate semantic re-ranking. 
Full tables, ablations and failure cases: `eval_P3_<TeamID>.md`. Every number is produced by `eval/`.
<!-- /RESULTS-HIGHLIGHTS -->

## Live demo

**https://arpreran.github.io/pocketdiffusion/** - permanent address. It checks that the demo server
(the team's RTX 4060 laptop) is running and forwards to it; when the laptop is offline it says so.

## Quick start

```bash
# 1. environment (Python 3.11, NVIDIA GPU recommended; CPU works with fallback plans)
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. pretrained weights (~8 GB, public, no token needed)
python scripts/download_assets.py

# 3. command-line inference on a file of prompts -> outputs/*.png + outputs/log.json
python generate.py --prompts prompts/samples.txt --out outputs/

# 4. run the full-stack app (web frontend + FastAPI + SQLite) -> http://127.0.0.1:8000
python serve.py --share      # --share also prints a public https link for phones / judges
#    (Windows one-click: run.bat --share ; alternative Gradio UI: python app.py)
```

Docker (GPU): `docker build -t pocketdiffusion . && docker run --gpus all -p 8000:8000 -v pdata:/app/data pocketdiffusion`

## Full-stack architecture

| Layer | Tech | What it does |
|---|---|---|
| Frontend | `frontend/` — HTML + CSS + vanilla JS (no build step) | Studio with live denoising frames, live planner preview while you move the sliders, scene-graph chips, before/after slider, artifact heat-map, detection boxes, telemetry (latency waterfall, memory gauge, semantic checklist); Gallery; Lab (SVG charts of the benchmarks + live DB stats); Race (baseline vs optimised) |
| Backend | `backend/main.py` (FastAPI), `backend/jobs.py` | REST API with input validation, one GPU worker thread + job queue, progress streamed with Server-Sent Events, warm-up at start, public tunnel with `--share` |
| Database | `backend/db.py` (SQLite, WAL) | every generation with plan, metrics, timings, candidates, post-processing stages and denoising frames; favourites, search, statistics |

API: `GET /api/system` · `POST /api/analyze` · `POST /api/plan` · `POST /api/generate` · `GET /api/jobs/{id}/events` (SSE) · `GET/DELETE /api/generations[/{id}]` · `POST /api/generations/{id}/favorite` · `GET /api/stats` · `GET /api/benchmarks` · `POST /api/compare`

## Reproduce every number in the report

```bash
python -m eval.run_all          # data prep, NIQE fit, text-encoder distillation, all benchmarks, report
python -m eval.run_all --quick  # reduced N (~25 min) for a sanity check
```

Outputs: `eval_P3_<TeamID>.md` (set your team ID and USNs in `team.json`), `results/*.json`,
`results/plots/*.png`, `results/failures/`. No metric is typed in by hand.

## What is inside (mapped to the problem statement)

| Requirement | Where | How |
|---|---|---|
| **A. Image resizing / resolution optimisation** | `pocketdiff/classical/resize.py` | aspect-preserving latent-grid snapping under a pixel budget; generate-small-then-upscale with Iterative Back-Projection SR (vs nearest/bilinear/bicubic/Lanczos); Full HD / 2K / 4K / **8K outputs** (also picked up from "8K" in the prompt) via IBP to 4K + Lanczos, ~2 s for 7680×4320 – sharper enlargement, no invented detail |
| A. Colour correction & enhancement | `classical/enhance.py` | shades-of-gray WB, adaptive gamma, CLAHE on L\*, saturation soft-knee, vibrance – strengths scale with measured severity |
| A. Edge-aware post-processing | `classical/filters.py`, `enhance.py` | guided filter from scratch (gray + colour guide), base/detail halo-free sharpening with edge mask, guided denoising |
| A. Image quality assessment | `classical/iqa.py` | **NIQE implemented from scratch** (MSCN, GGD/AGGD, MVG) with COCO-fitted pristine model; PocketIQA composite; colourfulness, contrast, noise σ, entropy |
| A. Artifact detection & correction | `classical/artifacts.py`, `postprocess.py` | blank/NaN, clipping, cast factor K, low contrast, over-saturation, noise, **8-px decoder grid via FFT peaks → notch filter**, blur |
| **B. Lightweight diffusion T2I** | `pocketdiff/engine.py` | SD-Turbo (1 step) and SD1.5 + LCM-LoRA (2–8 steps) run in our own denoising loop |
| B. Pretrained model + meaningful adaptation | `engine.py`, `scripts/distill_text_encoder.py` | LCM-LoRA fused into SD1.5 weights; **our 12→6-layer distilled text encoder** (COCO train/val split); planner calibrated on validation data |
| B. Optimised inference pipeline | `engine.py` | FP16, SDPA attention, CFG removal, embedding cache, cuDNN autotune, batched best-of-N, anytime x0 exit |
| B. Prompt encoding & conditional generation | `engine.py`, `prompt_analyzer.py` | CLIP tokenizer/encoder with **long-prompt mode**: prompts beyond CLIP's 77 tokens are split at commas into ≤75-token chunks, encoded separately and concatenated (up to 225 tokens; negative prompt padded to match for CFG); scene-graph parsing |
| **C. ≥2 optimisation strategies** | `optimizations.py`, `config.py` | reduced precision · fewer steps (DPM++/LCM/Turbo) · distillation (LCM, ADD, own KD) · lightweight decoder (TAESD) · **own INT8/INT4 weight quantisation** · magnitude pruning study |
| C. Optimised vs baseline comparison | `eval/bench_efficiency.py`, `eval/eval_quality.py` | identical prompts and seeds; FP32 SD1.5 50-step baseline |
| C. Mobile resource requirements | `planner.py`, report §8 | phone profiles (memory budgets, FLOP-based latency estimates) + real CPU-only measurement |
| Adv. 1 complex prompts | `prompts/compositional.json` | 58 annotated prompts: multi-object, counting, colour binding, spatial, complex scenes |
| Adv. 2 semantic consistency | `pocketdiff/verifier.py`, `eval/eval_compositional.py` | OWLv2 detection + HSV colour check + box geometry → object precision/recall/**F1**, count/colour/spatial accuracy |
| Adv. 3 latency & memory reduction | report §3–5 | measured per stage |
| Adv. 4 degradation from compression / fewer steps | `eval/eval_degradation.py` | steps, resolution, INT8/INT4, pruning 10–50 %, distilled text encoder, decoder, best-of-N |
| Adv. 5 adaptive generation by compute budget | `pocketdiff/planner.py` | utility = CLIP − λ·log latency under latency & memory constraints; λ adapts to prompt complexity |
| Adv. 6 fallback for constrained devices | `planner.py`, `engine.fallback_chain` | infeasible budget → cheapest plan that fits; runtime OOM → N↓ → 384 px → TAESD → INT8 → 256 px → CPU |
| Metrics | `eval/` | CLIP (ViT-L/14), FID + KID, latency, peak memory, model size, quality-vs-cost (+ object F1, NIQE) |

## Project layout

```
serve.py                   one-command start of the full-stack app (--share for a public link)
generate.py                command-line inference: prompts file -> images + log.json
Solution_Report_P3.md      solution report (hackathon template); check AI share: scripts/ai_share.py
frontend/                  index.html, styles.css, app.js  (the web UI)
backend/                   main.py (FastAPI routes), jobs.py (GPU worker + SSE), db.py (SQLite)
app.py                     alternative Gradio interface
pocketdiff/
  config.py                model ids, generation presets, device profiles
  prompt_analyzer.py       scene graph + complexity + safety block-list
  planner.py               budget-aware adaptive planner (cost / quality models)
  engine.py                model cache, own denoising loop, anytime exit, OOM fallback, FLOP counting
  optimizations.py         INT8/INT4 weight-only quantisation, pruning, attention
  pipeline.py              orchestrator: plan -> generate -> screen -> re-rank -> post-process -> verify
  scoring.py               CLIP scoring (re-rank B/32, evaluation L/14) + zero-shot safety check
  verifier.py              OWLv2 + HSV colour + geometry compositional verification
  classical/               filters, enhancement, artifacts, IQA (NIQE), resizing, post-processor
scripts/                   asset download, data preparation, NIQE fitting, text-encoder distillation,
                           report figures, AI-share checker
eval/                      efficiency, quality (CLIP/FID/KID), degradation, compositional F1, ablation, report
prompts/compositional.json annotated complex-prompt benchmark (58 prompts, written with AI assistance)
prompts/samples.txt        example prompts for generate.py
tests/                     unit tests (python -m pytest -q)
docs/                      architecture diagram, design-doc template, AI declaration
```

## Limitations (stated honestly)
- Phone latencies are **estimates** (measured FLOPs ÷ an assumed effective throughput); only the GPU and CPU numbers are measured.
- FID on a few hundred images is biased upwards and only comparable within our table; KID is reported alongside.
- Our NIQE uses a pristine model fitted on COCO photos, so absolute values differ from the MATLAB reference.
- Object-level F1 inherits OWLv2 detection errors; colour naming is heuristic.
- The zero-shot CLIP safety check and keyword block-list are coarse and can be bypassed.
- Distilled diffusion models still struggle with counting, attribute binding and spatial relations (see failure cases).

## Ethics & licences
- Stable Diffusion v1.5 – CreativeML OpenRAIL-M; SD-Turbo – Stability AI Non-Commercial Research Community License (academic use only); LCM-LoRA, TAESD – MIT/OpenRAIL; CLIP – MIT; OWLv2 – Apache-2.0; COCO captions – CC BY 4.0.
- Generative models reproduce biases of their web-scale training data; outputs must not be presented as real photographs.
- AI assistance used in this project is declared in [docs/AI_DECLARATION.md](docs/AI_DECLARATION.md).
