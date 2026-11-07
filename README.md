# 📱 PocketDiffusion — Resource-Efficient Text-to-Image Generation for Mobile Devices

**Computer Vision Hackathon 2026 · Problem Statement 3 (P3)**

PocketDiffusion is a resource-aware text-to-image generation system designed to produce images from natural-language prompts while considering device limitations such as processing time and available memory.

The team designed and developed the system around an adaptive planning approach. The planner analyses the input prompt and the available device budget before selecting suitable generation settings, including the model, sampling steps, output resolution, decoder, weight precision, and number of candidate images.

A custom diffusion loop executes the selected plan using deadline-aware early exit and fallback mechanisms for memory-related failures. Classical computer vision techniques are then applied to identify and correct image-quality issues. The system also evaluates generated candidates and checks whether requested objects, counts, colours, and spatial relationships are represented in the output.

---

## Results at a glance (RTX 4060 Laptop, 512×512, 200 unseen COCO prompts)

| System | CLIP-L ↑ | FID ↓ | Latency ms ↓ | Speed-up | Peak MB ↓ | Weights MB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline SD1.5 FP32, PNDM-50, CFG | 26.43 | 150.8 | 13307 | 1× | 5065 | 4067 |
| + TAESD decoder | 25.27 | 158.2 | 257 | 52× | 2068 | 1879 |
| + distilled 6-layer text encoder | 25.21 | 157.0 | 360 | 37× | 1162 | 981 |
| SD-Turbo 1 step + TAESD | 26.27 | 150.3 | 85 | 156× | 2489 | 2305 |
| SD-Turbo 1 step + TAESD + INT8 | 26.28 | 150.7 | 104 | 127× | 1671 | 1483 |
| PocketDiffusion (adaptive, fast mode) | 26.29 | 147.3 | 83 | 160× | 2735 | 2305 |
| PocketDiffusion (adaptive, auto mode) | 26.72 | 147.4 | 155 | 86× | 3039 | 2305 |

The compositional benchmark contains 58 annotated prompts covering objects, counts, colours, and spatial relationships. PocketDiffusion achieved an object F1 score of **0.958**, compared with 0.889 for the baseline. The percentage of prompts satisfying all evaluated constraints increased from 59% to **77%** when eight-candidate semantic re-ranking was used.

Full benchmark tables, ablation studies, and failure cases are documented in `eval_P3_<TeamID>.md`. The reported measurements are generated through the evaluation scripts in `eval/`.

---

## Live demo

**https://arpreran.github.io/pocketdiffusion/** — permanent demo address.

The page opens the demo hosted on the team's RTX 4060 laptop when that machine is online. When the laptop is unavailable, the demo can use the cloud deployment:

**https://arpreran--pocketdiffusion.us-east.modal.run**

The cloud version runs on a serverless NVIDIA T4 through Modal. Its deployment configuration is available in `deploy/modal_app.py`.

---

## Quick start

```bash
# 1. Environment (Python 3.11, NVIDIA GPU recommended; CPU works with fallback plans)
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. Download pretrained model weights (~8 GB, public, no token needed)
python scripts/download_assets.py

# 3. Run command-line inference using a file of prompts
# Output: outputs/*.png and outputs/log.json
python generate.py --prompts prompts/samples.txt --out outputs/

# 4. Start the full-stack application
# Web frontend + FastAPI backend + SQLite database
# Open http://127.0.0.1:8000
python serve.py --share

# --share also prints a public HTTPS link for phones and judges
# Windows one-click option: run.bat --share
# Alternative Gradio interface: python app.py
```

Docker (GPU):

```bash
docker build -t pocketdiffusion . && docker run --gpus all -p 8000:8000 -v pdata:/app/data pocketdiffusion
```

---

## Full-stack architecture

| Layer | Technology | What it does |
|---|---|---|
| Frontend | `frontend/` — HTML, CSS, and vanilla JavaScript (no build step) | Provides the Studio interface with live denoising frames, planner previews, scene-graph chips, before/after comparison, artifact heat-map, detection boxes, and telemetry. It also includes Gallery, Lab, and Race views. |
| Backend | `backend/main.py` (FastAPI), `backend/jobs.py` | Provides REST API endpoints, input validation, a GPU worker thread, job queue, progress updates through Server-Sent Events, startup warm-up, and public tunnel support through `--share`. |
| Database | `backend/db.py` (SQLite, WAL) | Stores generation history, plans, metrics, timings, candidates, post-processing stages, denoising frames, favourites, search data, and statistics. |

API endpoints:

`GET /api/system` · `POST /api/analyze` · `POST /api/plan` · `POST /api/generate` · `GET /api/jobs/{id}/events` (SSE) · `GET/DELETE /api/generations[/{id}]` · `POST /api/generations/{id}/favorite` · `GET /api/stats` · `GET /api/benchmarks` · `POST /api/compare`

---

## Reproduce every number in the report

```bash
python -m eval.run_all          # Data preparation, NIQE fitting, text-encoder distillation, benchmarks, and report
python -m eval.run_all --quick  # Reduced sample count (~25 min) for a sanity check
```

Outputs:

- `eval_P3_<TeamID>.md` — evaluation report. Set the team ID and USNs in `team.json`.
- `results/*.json` — benchmark results and evaluation data.
- `results/plots/*.png` — generated plots.
- `results/failures/` — recorded failure cases.

The evaluation scripts generate the reported measurements rather than relying on manually entered metric values.

---

## What is inside (mapped to the problem statement)

| Requirement | Where | How |
|---|---|---|
| **A. Image resizing / resolution optimisation** | `pocketdiff/classical/resize.py` | Preserves image aspect ratio and adjusts the latent grid according to a pixel budget. Supports generate-small-then-upscale using Iterative Back-Projection Super-Resolution (compared with nearest-neighbour, bilinear, bicubic, and Lanczos). Supports Full HD, 2K, 4K, and **8K outputs**. The high-resolution path uses a diffusion detail pass, followed by IBP and Lanczos processing. |
| A. Colour correction and enhancement | `classical/enhance.py` | Uses shades-of-gray white balancing, adaptive gamma correction, CLAHE on the L* channel, saturation soft-knee adjustment, and vibrance enhancement. Processing strength is adjusted according to measured image severity. |
| A. Edge-aware post-processing | `classical/filters.py`, `enhance.py` | Implements guided filtering for grayscale and colour images, base/detail sharpening with an edge mask to reduce halos, and guided denoising. |
| A. Image quality assessment | `classical/iqa.py` | Implements NIQE from scratch using MSCN, GGD/AGGD, and MVG components with a COCO-fitted pristine model. PocketIQA combines quality indicators such as colourfulness, contrast, noise standard deviation, and entropy. |
| A. Artifact detection and correction | `classical/artifacts.py`, `postprocess.py` | Detects blank or NaN outputs, clipping, colour cast, low contrast, over-saturation, noise, blur, and decoder-grid artifacts. An 8-pixel decoder grid can be detected through FFT peaks and addressed using notch filtering. |
| **B. Lightweight diffusion text-to-image generation** | `pocketdiff/engine.py` | Runs SD-Turbo in one step and SD1.5 with LCM-LoRA in 2–8 steps using a custom denoising loop. |
| B. Pretrained model and adaptation | `engine.py`, `scripts/distill_text_encoder.py` | Integrates LCM-LoRA with SD1.5 weights and includes a 12-to-6-layer distilled text encoder trained using the COCO train/validation split. The planner is calibrated using validation data. |
| B. Optimised inference pipeline | `engine.py` | Uses FP16, scaled dot-product attention (SDPA), classifier-free guidance removal where applicable, embedding caching, cuDNN autotuning, batched best-of-N generation, and anytime x0 exit. |
| B. Prompt encoding and conditional generation | `engine.py`, `prompt_analyzer.py` | Uses CLIP tokenisation and text encoding. Long-prompt mode splits prompts at commas into chunks of up to 75 tokens, encodes them separately, and concatenates the embeddings (up to 225 tokens). Negative prompts are padded to match when classifier-free guidance is used. Scene-graph parsing is also supported. |
| **C. At least two optimisation strategies** | `optimizations.py`, `config.py` | Includes reduced precision, fewer sampling steps, DPM++/LCM/Turbo methods, distillation, lightweight TAESD decoding, INT8/INT4 weight quantisation, and magnitude-pruning experiments. |
| C. Optimised versus baseline comparison | `eval/bench_efficiency.py`, `eval/eval_quality.py` | Compares the baseline and optimised configurations using identical prompts and seeds. The baseline uses FP32 SD1.5 with 50 sampling steps. |
| C. Mobile resource requirements | `planner.py`, report §8 | Uses device profiles with memory budgets and FLOP-based latency estimates. Real CPU-only measurements are also included. |
| Advanced 1. Complex prompts | `prompts/compositional.json` | Uses 58 annotated prompts covering multiple objects, counting, colour binding, spatial relationships, and complex scenes. |
| Advanced 2. Semantic consistency | `pocketdiff/verifier.py`, `eval/eval_compositional.py` | Uses OWLv2 object detection, HSV colour checks, and bounding-box geometry to evaluate object precision, recall, F1, counting, colour, and spatial accuracy. |
| Advanced 3. Latency and memory reduction | Report §3–5 | Reports measured performance and memory usage for the evaluated configurations. |
| Advanced 4. Degradation from compression and fewer steps | `eval/eval_degradation.py` | Evaluates sampling steps, resolution, INT8/INT4 quantisation, pruning from 10–50%, distilled text encoder, decoder choices, and best-of-N generation. |
| Advanced 5. Adaptive generation by compute budget | `pocketdiff/planner.py` | Uses a utility function based on CLIP score and logarithmic latency cost under latency and memory constraints. The latency penalty adapts to prompt complexity. |
| Advanced 6. Fallback for constrained devices | `planner.py`, `engine.fallback_chain` | Selects a lower-cost plan when the requested budget is infeasible. Runtime OOM fallback reduces candidate count, resolution, decoder cost, and precision, with CPU as a final fallback. |
| Metrics | `eval/` | Measures CLIP (ViT-L/14), FID, KID, latency, peak memory, model size, quality-versus-cost, object F1, and NIQE. |

---

## Project layout

```text
serve.py                    One-command start of the full-stack app (--share for a public link)
generate.py                 Command-line inference: prompts file -> images + log.json
Solution_Report_P3.md       Solution report (hackathon template); check AI share: scripts/ai_share.py
frontend/                   index.html, styles.css, app.js (the web UI)
backend/                    main.py (FastAPI routes), jobs.py (GPU worker + SSE), db.py (SQLite)
app.py                      Alternative Gradio interface

pocketdiff/
  config.py                 Model IDs, generation presets, device profiles
  prompt_analyzer.py        Scene graph, complexity analysis, and safety block-list
  planner.py                Budget-aware adaptive planner (cost / quality models)
  engine.py                 Model cache, custom denoising loop, anytime exit, OOM fallback, FLOP counting
  optimizations.py          INT8/INT4 weight-only quantisation, pruning, and attention
  pipeline.py               Orchestrator: plan -> generate -> screen -> re-rank -> post-process -> verify
  scoring.py                CLIP scoring (B/32 re-ranking, L/14 evaluation) + zero-shot safety check
  verifier.py               OWLv2 + HSV colour + geometry-based compositional verification
  classical/                Filters, enhancement, artifact detection, IQA (NIQE), resizing, post-processor

scripts/                    Asset download, data preparation, NIQE fitting, text-encoder distillation,
                            report figures, and AI-share checker
eval/                       Efficiency, quality (CLIP/FID/KID), degradation, compositional F1,
                            ablation, and report generation
prompts/compositional.json  Annotated complex-prompt benchmark (58 prompts, written with AI assistance)
prompts/samples.txt         Example prompts for generate.py
tests/                      Unit tests (python -m pytest -q)
docs/                        Architecture diagram, design-document template, and AI declaration
```

---

## Limitations (stated honestly)

- Phone latencies are **estimates**, calculated using measured FLOPs and assumed effective throughput. Only the GPU and CPU results are measured directly.
- FID calculated from a few hundred images can be biased upwards and should only be compared within the reported evaluation table. KID is included alongside it.
- The NIQE implementation uses a pristine model fitted on COCO photographs, so its absolute values may differ from the MATLAB reference implementation.
- Object-level F1 depends on OWLv2 detection performance. Colour naming is heuristic and may not always match human interpretation.
- The zero-shot CLIP safety check and keyword block-list provide basic filtering and are not guaranteed to detect every unsafe prompt.
- Distilled diffusion models can still struggle with counting, attribute binding, and spatial relationships. These limitations are reflected in the recorded failure cases.
- High-resolution output and mobile-device performance depend on the available hardware, memory, and selected generation configuration.

---

## Team contributions and development

PocketDiffusion was designed and developed collaboratively by the project team. Work was divided across AI architecture and backend implementation, database development, and frontend development.

| Team member | Main responsibilities |
|---|---|
| Preran A R | AI architecture, AI integration, and backend development |
| Punith Anand | Database design, database integration, and data management |
| Rohan Raj | Frontend development, user-interface design, and frontend integration |

The team was responsible for the project concept, system design, implementation, integration, testing, and evaluation. Individual responsibilities describe the primary areas of contribution and do not imply that the project was developed independently by only one member.

### Use of AI assistance

AI tools, including Claude, were used as development aids for selected implementation tasks. This assistance included code suggestions, debugging support, explanations, and help with improving or refining parts of the implementation.

The project architecture, problem-solving approach, integration decisions, testing activities, and overall development were carried out by the project team. AI assistance should not be interpreted as meaning that the entire project or its complete codebase was generated by an AI tool.

The 58-prompt compositional benchmark and the evaluation workflow are part of the team's project work. AI assistance used in preparing benchmark prompts is acknowledged in the relevant project materials.

---

## Ethics & licences

- Stable Diffusion v1.5 — CreativeML OpenRAIL-M.
- SD-Turbo — Stability AI Non-Commercial Research Community License (academic use only).
- LCM-LoRA and TAESD — MIT/OpenRAIL licences, respectively.
- CLIP — MIT.
- OWLv2 — Apache-2.0.
- COCO captions — CC BY 4.0.

Generative models can reproduce biases present in their web-scale training data. Generated outputs should not be presented as genuine photographs or as verified representations of real events.

AI assistance used during the development of this project is acknowledged in the project's AI declaration. Refer to `docs/AI_DECLARATION.md` for the declaration included in the repository.