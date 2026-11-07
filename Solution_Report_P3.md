<!--
NOTE (delete before submitting): the prose of this report is AI-drafted and declared as such in
section 6 (measured AI share: python scripts/ai_share.py Solution_Report_P3.md). Keep the AI-DRAFTED
markers and the declaration accurate. Still to do: fill the team-contribution table in section 6
and the evaluated commit hash in the header, then: python scripts/build_report_pdf.py --final
-->

# Hackathon Technical Solution Report

| Field | Details |
|---|---|
| **Problem ID** | P3 (Computer Vision Hackathon 2026) |
| **Problem Title** | Resource-Efficient Text-to-Image Generation for Mobile Devices |
| **Team Name** | VisionForge (Team 33) |
| **Team Members** | Preran A R – 1RUA24CSE0332; Punith Anand – 1RUA24CSE0341; Rohan Raj – 1RUA24CSE0373 |
| **GitHub Repository** | https://github.com/arpreran/pocketdiffusion, evaluated commit `f3ebe8d` (update to your final commit before submitting: `git rev-parse --short HEAD`) |
| **Live Demo / Deployment** | https://arpreran.github.io/pocketdiffusion/ (permanent address, always available: opens the full-stack web app - FastAPI + SQLite + web UI - on the team's RTX 4060 laptop when it is online, otherwise on the always-on cloud copy https://arpreran--pocketdiffusion.us-east.modal.run) |

---

## 1. Problem Understanding & Solution Strategy

- **Core Technical Challenge:**

<!-- AI-DRAFTED -->
Stable Diffusion v1.5, our baseline, needs 102 UNet evaluations per image (50 PNDM steps plus one, doubled by classifier-free guidance), about 84.5 TFLOPs and 4.07 GB of FP32 weights. On our RTX 4060 laptop GPU one 512×512 image takes 13.3 s and 5.1 GB of peak memory, while a phone offers an app roughly 1–3 GB of memory and 0.3–4 TFLOPS. Simply cutting steps, precision or resolution causes three kinds of failure that we target: visual degradation (blur, 8-pixel decoder grids, colour casts, clipped highlights), semantic failure on complex prompts (missing objects, wrong counts, colours attached to the wrong object, ignored spatial relations) and run-time failure on weak devices (out-of-memory errors or missed deadlines). Our constraints are a per-request latency budget (2 s by default), a device memory budget between 6.5 GB and 1.1 GB, and no external text-to-image API.
<!-- /AI-DRAFTED -->

- **Proposed Methodology:**

<!-- AI-DRAFTED -->
Instead of shipping one compressed model, PocketDiffusion plans every request: it analyses the prompt, enumerates about 700 configurations (model family, steps, resolution, decoder, weight precision, text encoder and number of candidates) and selects the one with the best predicted quality that fits the device's latency and memory budget. A custom denoising loop executes the plan with deadline-aware early exit and an out-of-memory fallback chain, and classical computer vision screens, repairs and verifies the output. We chose this over a static setting because no single configuration is best for every prompt and device: the adaptive system reaches CLIP 26.72 and FID 147.4, better than the best static fast configuration (26.27 and 150.3), while being 86× faster than the baseline.
<!-- /AI-DRAFTED -->

---

## 2. Technical Architecture & Implementation

![PocketDiffusion architecture (source: docs/architecture.md)](docs/architecture.png)

### 2.1 Classical Algorithmic Baseline

- **Algorithms Utilized:**

<!-- AI-DRAFTED -->
All classical stages are implemented with NumPy and OpenCV primitives in `pocketdiff/classical/`. An artifact detector measures blank or NaN output, clipped highlights, crushed shadows, a Lab colour-cast factor, dynamic range, over-saturation, Immerkær noise, an FFT-based 8-pixel grid score and Laplacian blur, each mapped to a severity between 0 and 1. The corrections are an FFT notch filter for grids, guided-filter edge-aware denoising, shades-of-grey white balance, adaptive gamma, CLAHE on the L channel, a saturation soft knee and halo-free base/detail sharpening with an edge mask. Resolution handling snaps sizes to multiples of eight under a pixel budget and enlarges images with iterative back-projection (IBP), which we extended to 2K, 4K and 8K output. Image quality is assessed with our own NIQE implementation, fitted on 120 COCO validation photographs, and a PocketIQA composite score.
<!-- /AI-DRAFTED -->

- **Adaptive Selection Logic:**

<!-- AI-DRAFTED -->
Every correction is conditional and proportional. A stage runs only when its detector fires, and its strength follows the measured severity: the white-balance strength grows with the cast factor, the denoising radius and regularisation follow the estimated noise level, gamma is derived from the mean luminance with a target of 0.46, the CLAHE clip limit grows as the dynamic range shrinks, and the notch filter is applied only above the grid-score threshold. All thresholds were tuned on the validation split and never on the test split; for example, the denoising threshold was raised after it worsened NIQE on validation images. On real outputs, sharpening fired on every image, white balance on 23 %, adaptive gamma on 5 % and denoising on 1 %.
<!-- /AI-DRAFTED -->

### 2.2 Deep Learning Pipeline

- **Model Architecture:**

<!-- AI-DRAFTED -->
The baseline is Stable Diffusion v1.5 (UNet 859.5 M, CLIP ViT-L text encoder 123.1 M and VAE decoder 83.7 M parameters). Because 98 % of its 13.3 s is spent in the UNet, our main lever is reducing UNet calls through distilled models: SD1.5 with LCM-LoRA (consistency distillation, 2–8 steps, with the LoRA fused into the weights so it adds no run-time cost) and SD-Turbo (adversarial diffusion distillation, one step). The tiny TAESD decoder (2.45 M parameters) replaces the VAE decoder and cuts decoding from 146 ms to 14 ms, and our own INT8 (per output channel) and INT4 (groups of 64) weight-only quantisation reduces memory. The models run in our own denoising loop, validated against the diffusers reference, and prompts longer than CLIP's 77-token limit are encoded in 75-token chunks so that no words are lost. CLIP ViT-B/32 and OWL-ViT are used only for re-ranking, while CLIP ViT-L/14 and OWLv2 are reserved for evaluation so that we never optimise the metric we report.
<!-- /AI-DRAFTED -->

- **Loss Functions & Training:**

<!-- AI-DRAFTED -->
We trained a 6-layer student text encoder by knowledge distillation from the 12-layer CLIP text encoder of SD1.5, keeping layers 0, 2, 4, 7, 9 and 11 (123.1 M → 80.5 M parameters). The loss is the mean squared error plus a cosine term over all 77 token hidden states. Training used 400,171 MS-COCO train captions with early stopping on validation captions: the validation MSE fell from 0.987 to 0.098, the token cosine similarity reached 0.950, and training took 11.2 minutes on the RTX 4060. On the test split the student changes CLIP only from 25.23 to 25.21 while saving 81 MB of memory. The planner's quality model and the NIQE pristine model were calibrated on the validation split without gradient training, and the generators themselves (SD1.5, LCM-LoRA, SD-Turbo and TAESD) were used as pretrained models. All data come from the MS-COCO 2014 Karpathy split, with the train, validation and test sets kept strictly separate.
<!-- /AI-DRAFTED -->

### 2.3 Hybrid Integration & Adaptive Mechanism

- **Fusion Strategy:**

<!-- AI-DRAFTED -->
Classical processing and the neural generator are interleaved at several points. Before generation, misspellings that the text encoder cannot read are corrected (for example "ciggerate" becomes "cigarette"), and a rule-based parser builds a scene graph of objects, counts, colours and relations that sets the prompt complexity for the planner. Directly after denoising, cheap artifact screening rejects blank, NaN or badly exposed candidates. The surviving candidates are re-ranked by a score that combines CLIP-B/32 similarity, per-object coverage, PocketIQA and a penalty for artifact severity. The chosen image then receives a face detail pass, which detects faces and redraws them at full resolution, followed by severity-driven post-processing, IBP or 8K super-resolution, and a verification step based on OWLv2 boxes, HSV colour statistics and box geometry.
<!-- /AI-DRAFTED -->

- **Decision Logic:**

<!-- AI-DRAFTED -->
The planner maximises a utility equal to the predicted CLIP score minus λ·log2(latency / 100 ms). In auto mode λ = 1 − 0.7c, where c is the prompt complexity, so simple prompts receive the cheapest adequate plan while complex prompts may spend more compute; a generous latency budget lowers λ further so that spare time is spent on quality. The predicted gain from re-ranking grows with the number of checkable constraints, which gives multi-object prompts several candidates. Latency is predicted from a measured table of UNet times by resolution and batch size, and memory from the weight size and measured activations. If no plan fits both budgets, the fastest plan that fits memory is used; at run time the system exits early with the current estimate, or falls back to fewer candidates, a smaller resolution, INT8 weights or the CPU when memory runs out. In practice the laptop GPU receives SD-Turbo at 512 px with four candidates, whereas the low-end phone profile receives SD1.5 with LCM at 256 px, INT8 weights and the 6-layer text encoder.
<!-- /AI-DRAFTED -->

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

<!-- AI-DRAFTED -->
PSNR and SSIM need a ground-truth image, which a text prompt does not have, so we report the P3 metrics instead. No classical image generator exists, so row 1 is the unoptimised neural baseline. Same 200 unseen MS-COCO test prompts and seeds for every row; object F1 from our 58-prompt compositional benchmark.
<!-- /AI-DRAFTED -->

| Approach / Model Variant | CLIP ↑ | FID ↓ | NIQE ↓ | Artifact rate ↓ | Object F1 ↑ | Latency (ms / image) | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| **Unoptimised baseline**: SD1.5 FP32, PNDM 50 steps, CFG 7.5 | 26.43 | 150.8 | 4.19 | 0.30 | 0.889 ¹ | 13,307 | 5,065 MB |
| **Deep learning only**: SD-Turbo 1 step + TAESD (static, no classical stages, no planner) | 26.27 | 150.3 | 3.81 | 0.20 | 0.940 | 85 | 2,489 MB |
| **Proposed hybrid system**: PocketDiffusion (adaptive, auto mode) | **26.72** | **147.4** | **3.74** | **0.07** | **0.958** | 155 (generation) / 291 (end-to-end) | 3,039 MB |

<!-- AI-DRAFTED -->
¹ SD1.5 FP16, 50 steps (same model at half precision).
<!-- /AI-DRAFTED -->

#### Optimisation ladder (512×512, batch 1, after warm-up)

| Configuration | UNet evals | Latency ms | Speed-up | Peak MB | Weights MB | CLIP |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 102 | 13,307 | 1.0× | 5,065 | 4,067 | 26.43 |
| + FP16 | 102 | 5,170 | 2.6× | 2,676 | 2,034 | 26.38 |
| + DPM-Solver++ 20 steps | 40 | 2,120 | 6.3× | 2,676 | 2,034 | 26.53 |
| + LCM-LoRA 4 steps (distilled) | 4 | 388 | 34.3× | 2,675 | 2,034 | 25.24 |
| + TAESD decoder | 4 | 257 | 51.7× | 2,068 | 1,879 | 25.27 |
| + INT8 weights | 4 | 366 | 36.4× | 1,246 | 1,062 | 25.23 |
| + distilled 6-layer text encoder | 4 | 360 | 37.0× | 1,162 | 981 | 25.21 |
| SD-Turbo 1 step + TAESD | 1 | 85 | 156× | 2,489 | 2,305 | 26.27 |
| SD-Turbo 1 step + TAESD + INT8 | 1 | 104 | 128× | 1,671 | 1,483 | 26.28 |
| SD1.5 LCM 4 steps + TAESD + INT4 (validation split) | 4 | 558 | 24× | – | 678 | 24.67 |
| CPU only: LCM 2 steps, 256 px, TAESD, FP32 | 2 | 1,514 | 8.8× | 4,286 (RAM) | 1,879 | – |

| Full result tables (KID, model size, TFLOPs, device plans, degradation curves) |
|---|
| [`eval_P3_33.md`](eval_P3_33.md) |

- **Analysis:**

<!-- AI-DRAFTED -->
The optimisations reduce generation time from 13.3 s to 155 ms (86×). Distillation gives the largest single gain by reducing 102 UNet calls to between one and four, FP16 contributes 2.6×, and TAESD removes about 130 ms of decoding. INT8 lowers peak memory by about 40 % but is slower on the GPU (366 ms against 257 ms) because the weights are de-quantised on the fly, so the planner uses it only when memory requires it. LCM with four steps costs 1.2 CLIP points whereas SD-Turbo costs almost nothing, and the complete hybrid system achieves the best CLIP, FID, NIQE and artifact rate of all variants, at the price of 291 ms end-to-end instead of 85 ms for the bare model.
<!-- /AI-DRAFTED -->

### 3.2 Visual Outputs & Intermediate Representations

#### 1. Baseline vs optimised output

![Figure 1](results/report/fig1_side_by_side.png)

<!-- AI-DRAFTED -->
*Figure 1.* First three MS-COCO test prompts (not cherry-picked), same seed for every system; titles give the measured mean latency.
<!-- /AI-DRAFTED -->

#### 2. Complex prompts: objects, counts, colours, positions

![Figure 2](results/report/fig2_compositional.png)

<!-- AI-DRAFTED -->
*Figure 2.* First prompt of four hard categories. Boxes: OWLv2 evaluation detector. Below each image: object F1 and whether every count, colour and position constraint holds (red = failure, see section 4).
<!-- /AI-DRAFTED -->

#### 3. Intermediate states

![Figure 3](results/report/fig3_intermediate.png)

<!-- AI-DRAFTED -->
*Figure 3.* Top: x0 estimate after each denoising step (streamed live in the app). Middle: raw output, artifact heat-map, guided-filter base and detail layers, sharpening edge mask. Bottom: output of each post-processing stage that fired, with its cost.
<!-- /AI-DRAFTED -->

#### 4. Spelling correction and quality-first defaults (user prompts, before → after)

![Figure 4 – spelling correction](results/report/fig4_spelling_quality.jpg)

| Prompt typed by the user | Before (auto, 384 px, 1 try) | After (corrected prompt, Quality: best of 4 at 512 px) |
|---|---|---|
| "A guy with ciggerate" | CLIP 22.2, no cigarette | "A guy with cigarette": CLIP 32.6, lit cigarette and smoke |
| "A teacher teaching a classroom with students" | CLIP 29.6, two blurred teachers | CLIP 31.5, one teacher at the board |

#### 5. Anime prompts: real-word typo, character counting, false safety blur

![Figure 5 – anime prompt fix](results/report/fig5_anime_fix.jpg)

![Figure 6 – anime prompt, four seeds](results/report/fig6_anime_seeds.jpg)

| Problem in "I WANT TWO AMINE CHARACTER FIGHTING IN THE NIGHT…" | Fix | Result |
|---|---|---|
| "amine" is a dictionary word (chemistry) | context typo rule: amine → anime before character/girl/style… | prompt read as "anime" |
| "character" not parsed as a person | character / hero / warrior / teacher / student … → person | 2 people expected |
| anime fights blurred by the zero-shot safety check | drawn-style safe concepts added | 0 of 4 blurred (was 2 of 3) |

#### 6. Face detail pass (before | after, same seed)

![Figure 7 – face detail pass](results/report/fig7_face_pass.png)

| Step | Method | Setting |
|---|---|---|
| Find faces | OWLv2 open-vocabulary detection, query "face" | score ≥ 0.10, side ≥ 20 px, inside a detected person |
| Depth of field | skip faces < 50 % of the largest face when that one is ≥ 60 px | background stays soft |
| Redraw | 1.6× square crop → 512 px → SD-Turbo image-to-image | strength 0.5 (0.75 if face < 40 px), 4-step schedule |
| Style guard | CLIP-B/32: "photo of a real face" vs "cartoon face" | cartoon redraw in a photo → retry at 0.5 → else keep original |
| Blend | feathered ellipse (Gaussian-blurred mask) | no visible seam |
| Cost | RTX 4060 laptop | ≈ 0.25 s per face, max 3 faces |

#### 7. 8K output: diffusion detail pass + classical super-resolution

![Figure 8 – 8K detail pass](results/report/fig8_8k_detail.jpg)

| Output | Native generation | Detail pass | Classical SR | Total |
|---|---|---|---|---|
| 7680×4320 (8K, 16:9) | 680×384, SD-Turbo 1 step | 1360×768, SD-Turbo img2img, 1 UNet call | IBP to ≤ 4K + Lanczos (×5.6) | ≈ 5 s on the laptop GPU |

#### 8. Benchmark plots

![Quality vs computational cost](results/plots/quality_vs_latency.png)

![Latency and memory breakdown](results/plots/latency_memory_breakdown.png)

![Quality degradation from compression and fewer steps](results/plots/degradation.png)

![Compositional accuracy](results/plots/compositional.png)

![Text-encoder distillation](results/plots/text_encoder_distillation.png)

### 3.3 Ablation Study

#### A. Removing one component at a time (100 test prompts)

| Variant | CLIP-L ↑ | NIQE ↓ | Artifact rate ↓ | Latency ms |
|---|---|---|---|---|
| Full system | 26.71 | 3.85 | 0.07 | 305 |
| − best-of-N re-ranking | 26.36 | 3.75 | 0.12 | 169 |
| − classical post-processing | 26.75 | 3.87 | 0.21 | 236 |
| − prompt-complexity adaptation (fixed λ) | 26.38 | 3.77 | 0.09 | 219 |
| − planner (one fixed configuration) | 26.14 | 3.74 | 0.10 | 154 |

#### B. Classical post-processing stages (leave-one-out)

| Variant | CLIP-L ↑ | NIQE ↓ | Artifact rate ↓ | ms / image |
|---|---|---|---|---|
| No post-processing | 26.11 | 3.78 | 0.24 | 0 |
| All stages (adaptive) | 26.16 | 3.74 | 0.10 | 94 |
| All − white balance | 26.16 | 3.74 | 0.21 | 98 |
| All − edge-aware sharpening | 26.11 | 3.78 | 0.12 | 59 |
| All − denoise | 26.16 | 3.74 | 0.11 | 96 |

#### C. Resolution optimisation (generate 384 px, upscale to 512 px)

| Variant | CLIP-L ↑ | NIQE ↓ | Generation ms | Upscale ms |
|---|---|---|---|---|
| Native 512 px | 26.11 | 3.78 | 76 | – |
| 384 → 512 nearest | 26.10 | 6.65 | 54 | 1.8 |
| 384 → 512 bicubic | 26.10 | 5.00 | 54 | 2.3 |
| 384 → 512 Lanczos | 26.07 | 4.98 | 54 | 2.6 |
| 384 → 512 **IBP (ours)** | 26.32 | **3.83** | 54 | 54.0 |

#### D. Semantic re-ranking (58 compositional prompts × 2 seeds)

| System | Object F1 ↑ | All constraints ↑ | Count acc | Colour acc | Spatial acc | ms / image |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP16, 50 steps | 0.889 | 0.53 | 0.42 | 0.61 | 0.41 | 5,195 |
| PocketDiffusion (auto) | 0.958 | 0.59 | 0.38 | 0.64 | 0.55 | 312 |
| + semantic re-ranking, N = 4 | 0.956 | 0.66 | 0.42 | 0.70 | 0.66 | 485 |
| + semantic re-ranking, N = 8 | 0.959 | 0.77 | 0.50 | 0.79 | 0.77 | 877 |

<!-- AI-DRAFTED -->
Removing components one at a time shows that each of them contributes. The planner gives the largest quality gain (+0.57 CLIP compared with one fixed configuration), best-of-N re-ranking adds +0.35 CLIP at a cost of 136 ms, and prompt-complexity adaptation adds +0.33. Classical post-processing does not change CLIP, as intended, but reduces the artifact rate from 0.21 to 0.07, with white balance responsible for most of the reduction. Re-ranking slightly raises NIQE (3.85 against 3.75) because CLIP prefers detailed, high-contrast candidates. IBP upscaling from 384 px reaches a NIQE of 3.83, close to native 512 px generation (3.78) and far better than bicubic interpolation (5.00). On complex prompts the object F1 saturates near 0.96, while the stricter all-constraints rate keeps rising with semantic re-ranking, from 59 % to 77 %.
<!-- /AI-DRAFTED -->

---

## 4. Failure Modes & Limitations

| # | Category | Prompt / setting | What went wrong (measured) | Evidence |
|---|---|---|---|---|
| F1 | Counting | "a photo of three apples" | 4 apples (seed 0) and 5 apples (seed 1) | [image](results/failures/pocketdiff_c02_s0.png) |
| F2 | Attribute binding | "a photo of a red cup and a blue book" | the blue went onto the cup's pattern; the book is beige (both seeds) | [image](results/failures/pocketdiff_k09_s0.png) |
| F3 | Spatial relations | "a red apple to the left of a green bottle on a wooden table" | relation violated by all three systems in Figure 2; *complex* category all-constraints rate 0 % (auto) → 30 % (N = 8) | Figure 2, row 4 |
| F4 | Compression | 50 % magnitude pruning of the SD1.5 UNet; SD1.5 at 256 px | CLIP 25.23 → 13.76 (pruning); 25.23 → 15.22 (256 px) | [degradation plot](results/plots/degradation.png) |
| F5 | Evaluation error | "a photo of a red car" | a red car is clearly visible (cropped at the border) but OWLv2 missed it, so it is counted as a missing object | [image](results/failures/pocketdiff_k01_s0.png) |
| F6 | Face detail pass | "two friends laughing at a cafe table" (first version of the pass) | a blurred background figure was sharpened into a floating face | fixed: faces must lie inside a detected person + depth-of-field rule |
| F7 | Safety check | anime fight scenes | 3 of 37 generations falsely blurred as "graphic violence" | fixed: drawn-style safe concepts (Figure 6) |

#### Failure case images

| F1: "three apples" → 4 apples | F1: same prompt, seed 1 → 5 apples | F2: blue on the cup, not the book | F5: red car missed by the evaluator |
|---|---|---|---|
| ![](results/failures/pocketdiff_c02_s0.png) | ![](results/failures/pocketdiff_c02_s1.png) | ![](results/failures/pocketdiff_k09_s0.png) | ![](results/failures/pocketdiff_k01_s0.png) |

- **Observed Failure Case + Root Cause & Mitigation:**

<!-- AI-DRAFTED -->
**F1 – counting.** The CLIP text encoder represents numbers poorly, so "three apples" is close to "some apples", and a one-step model cannot correct its layout afterwards. Count-aware re-ranking with a detector raises the counting success rate from 50 % to 63 % with eight candidates; box-guided layout generation would address the problem at its source.

**F2 – attribute binding.** Cross-attention receives a single text embedding in which adjectives leak between nouns, so "blue" ended up on the cup instead of the book. Per-object regional prompts or attention guidance such as Attend-and-Excite are the natural fixes, and semantic re-ranking already raises colour-binding success from 25 % to 50 %.

**F3 – spatial relations.** The conditioning behaves almost like a bag of words in which "left of" barely changes the embedding. Layout planning with box conditioning would fix this; semantic re-ranking already raises spatial accuracy from 45 % to 85 %.

**F4 – compression.** 50 % unstructured magnitude pruning without fine-tuning collapses CLIP from 25.23 to 13.76 because it removes weights the network relies on, and SD1.5 at 256 px falls to 15.22 because a 32×32 latent lies outside its training distribution. Structured pruning with distillation is the better route; our planner never uses pruning and prefers SD-Turbo when it must go to 256 px.
<!-- /AI-DRAFTED -->

- **Limitations:**

<!-- AI-DRAFTED -->
Phone latencies are estimated from measured FLOPs and an assumed device throughput rather than measured on real phones. FID computed on 200 images is biased upwards, so it is only comparable between rows, which is why we also report KID. Our NIQE model is fitted on COCO photographs and is not directly comparable to the MATLAB reference implementation. The zero-shot safety check is coarse, the photo-trained detectors do not reliably recognise people in drawn styles such as anime, and INT8 or INT4 weights reduce memory but not GPU latency without dedicated integer kernels.
<!-- /AI-DRAFTED -->

---

## 5. Deployment & Quickstart Guide

**Quickstart Commands**

```bash
# 1. Clone the repository and install dependencies (Python 3.11, NVIDIA GPU recommended)
git clone https://github.com/arpreran/pocketdiffusion.git
cd pocketdiffusion
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/download_assets.py --core   # public pretrained weights, no token needed

# 2. Run inference on a file of prompts (one per line) -> images + log.json
python generate.py --prompts prompts/samples.txt --out outputs/
python generate.py --prompts prompts/samples.txt --device lowend_phone --out outputs_lowend/

# 3. Full-stack web app (http://127.0.0.1:8000); --share also prints a public link,
#    --publish-link keeps https://arpreran.github.io/pocketdiffusion/ pointing at it
python serve.py --share --publish-link

# 4. Always-on cloud copy (Modal, free credits, CPU)
python -m modal deploy deploy/modal_app.py

# 5. Reproduce every number in this report (~2-3 h on an RTX 4060 laptop)
python -m eval.run_all
python scripts/make_report_figures.py
python scripts/build_report_appendix.py
python scripts/build_report_pdf.py
```

**Deployment Format**

<!-- AI-DRAFTED -->
PocketDiffusion is a full-stack web application. A FastAPI backend (`backend/`) runs every generation on a single GPU worker thread with a job queue and streams progress (plan, denoising previews, post-processing stages) to the browser over Server-Sent Events. Every generation is stored in SQLite with its plan and timings. The web UI (`frontend/`) has Studio, Gallery, Lab (live charts), Race (baseline vs optimised) and About pages. The permanent link opens the laptop-GPU server when it is online and an always-on cloud copy otherwise. A Dockerfile, an alternative Gradio interface (`app.py`) and the command-line tool (`generate.py`) are also provided.
<!-- /AI-DRAFTED -->

#### Deployment architecture

| Component | Where it runs | Address | Role |
|---|---|---|---|
| Permanent demo page | GitHub Pages (static, free) | https://arpreran.github.io/pocketdiffusion/ | checks which server is up and forwards the visitor |
| Main demo server | team laptop, RTX 4060 (8 GB) via Gradio share tunnel | changing `*.gradio.live` URL, published automatically to the page | fast path: ≈ 0.5–1.5 s per image, all features |
| Always-on cloud copy | Modal serverless container, 8 CPU cores / 16 GB, scales to zero | https://arpreran--pocketdiffusion.us-east.modal.run | used when the laptop is offline: ≈ 9–15 s per image |
| Source code + report | GitHub | https://github.com/arpreran/pocketdiffusion | reproducibility |

| Event | What the system does |
|---|---|
| network switch / sleep drops the tunnel | the server checks its public URL every minute, reopens the tunnel after two failures and pushes the new URL to `live.json` on GitHub Pages |
| tunnel client cannot resolve the relay (phone hotspot DNS) | retry with connect-by-IP (TLS still verified against the host name) or public DNS |
| laptop off | the permanent page wakes the Modal copy (cold start ≈ 15–40 s) |

#### Application screenshots

![Studio – prompt, device profiles, mode and budgets, result](results/report/screens/s1_studio.jpg)

![Studio – candidates, classical post-processing stages incl. face detail, latency waterfall, telemetry](results/report/screens/s2_studio_details.jpg)

![Gallery – every generation stored in SQLite, searchable](results/report/screens/s3_gallery.jpg)

![Lab – benchmark charts and live database statistics](results/report/screens/s4_lab.jpg)

![Race – same prompt and seed: PocketDiffusion 0.66 s vs SD1.5 50-step baseline 6.6 s](results/report/screens/s5_race.jpg)

![How it works – architecture page inside the app](results/report/screens/s6_how_it_works.jpg)

---

## 6. AI-Assistance Declaration

<!-- AI-DRAFTED -->
**Tool:** Claude (Anthropic), used as an AI coding assistant in the development environment.
**System design and code:** the assistant took part in system design discussions and wrote most of the source code (`pocketdiff/`, `backend/`, `frontend/`, `eval/`, `scripts/`). It also wrote the 58 prompts of the compositional benchmark (`prompts/compositional.json`).
**Results:** every number comes from the evaluation scripts in `eval/`, run on our hardware; none were typed in by hand.
**This report:** all prose in sections 1–6 was drafted by the assistant from our measured results. Tables and figures were generated from `results/*.json` and our own runs. Measured AI share of the report prose: **100 %** (`python scripts/ai_share.py Solution_Report_P3.md`); the team accepts the corresponding penalty under the hackathon's AI policy. The team-contribution table below was filled in by the team.
<!-- /AI-DRAFTED -->

| Member | USN | Part (A, B or C – see `docs/VIVA_GUIDE.md`) | What this member did |
|---|---|---|---|
| Preran A R | 1RUA24CSE0332 | `[part]` | `[fill in]` |
| Punith Anand | 1RUA24CSE0341 | `[part]` | `[fill in]` |
| Rohan Raj | 1RUA24CSE0373 | `[part]` | `[fill in]` |

---

<!-- APPENDIX:START (generated by scripts/build_report_appendix.py - do not edit by hand) -->

## Appendices

### A. Full evaluation tables

| Source | Generated by |
|---|---|
| `eval_P3_33.md` | `python -m eval.make_report` |


#### A.1 CLIP similarity (text-image alignment, ViT-L/14, higher is better)

| system | CLIP score (mean ± s.e.) | Δ vs baseline |
|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 26.43 ± 0.22 | 0.00 |
| + FP16 | 26.38 ± 0.23 | -0.06 |
| + DPM-Solver++ 20 steps | 26.53 ± 0.22 | 0.10 |
| + LCM-LoRA 4 steps (distilled) | 25.24 ± 0.22 | -1.19 |
| + TAESD decoder | 25.27 ± 0.21 | -1.16 |
| + INT8 weights | 25.23 ± 0.22 | -1.20 |
| + distilled 6-layer text encoder | 25.21 ± 0.23 | -1.22 |
| SD-Turbo 1 step + TAESD | 26.27 ± 0.21 | -0.16 |
| SD-Turbo 1 step + TAESD + INT8 | 26.28 ± 0.21 | -0.15 |
| **PocketDiffusion (adaptive, auto mode)** | 26.72 ± 0.21 | 0.28 |
| **PocketDiffusion (adaptive, fast mode)** | 26.29 ± 0.21 | -0.15 |
| *Real COCO photos (reference)* | 25.18 | – |


#### A.2 FID / KID (distribution-level realism vs real COCO images, lower is better)

| system | FID | KID ×10³ |
|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 150.8 | 3.94 ± 1.82 |
| + FP16 | 151.2 | 3.65 ± 1.28 |
| + DPM-Solver++ 20 steps | 151.4 | 3.07 ± 1.56 |
| + LCM-LoRA 4 steps (distilled) | 157.4 | 8.73 ± 1.61 |
| + TAESD decoder | 158.2 | 8.64 ± 1.59 |
| + INT8 weights | 156.8 | 8.86 ± 1.66 |
| + distilled 6-layer text encoder | 157.0 | 8.25 ± 1.61 |
| SD-Turbo 1 step + TAESD | 150.3 | 5.93 ± 2.20 |
| SD-Turbo 1 step + TAESD + INT8 | 150.7 | 6.72 ± 2.75 |
| **PocketDiffusion (adaptive, auto mode)** | 147.4 | 3.80 ± 1.94 |
| **PocketDiffusion (adaptive, fast mode)** | 147.3 | 3.35 ± 1.73 |

| system | NIQE ↓ | PocketIQA ↑ | artifact rate ↓ |
|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 4.19 | 87.6 | 0.30 |
| + DPM-Solver++ 20 steps | 4.22 | 87.4 | 0.32 |
| + TAESD decoder | 5.02 | 87.3 | 0.45 |
| SD-Turbo 1 step + TAESD | 3.81 | 87.2 | 0.20 |
| **PocketDiffusion (adaptive, auto mode)** | 3.74 | 87.7 | 0.07 |
| **PocketDiffusion (adaptive, fast mode)** | 3.76 | 88.1 | 0.18 |


#### A.3 Image generation latency (512×512, batch 1, after warm-up)

| system | UNet evals | latency ms | text ms | UNet ms | decode ms | speed-up |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 102 | 13307 ± 7 | 10.9 | 13026 | 270.2 | 1.0× |
| + FP16 | 102 | 5170 ± 4 | 11.4 | 5013 | 145.7 | 2.6× |
| + DPM-Solver++ 20 steps | 40 | 2120 ± 4 | 11.4 | 1962 | 146.1 | 6.3× |
| + LCM-LoRA 4 steps (distilled) | 4 | 388 ± 2 | 5.4 | 237 | 145.5 | 34.3× |
| + TAESD decoder | 4 | 257 ± 1 | 5.3 | 238 | 14.1 | 51.7× |
| + INT8 weights | 4 | 366 ± 5 | 7.2 | 344 | 14.7 | 36.4× |
| + distilled 6-layer text encoder | 4 | 360 ± 3 | 3.7 | 342 | 14.2 | 37.0× |
| SD-Turbo 1 step + TAESD | 1 | 85 ± 8 | 13.6 | 56 | 15.3 | 156.4× |
| SD-Turbo 1 step + TAESD + INT8 | 1 | 104 ± 2 | 9.7 | 81 | 14.2 | 127.5× |
| **PocketDiffusion (adaptive, auto mode)** | adaptive | 155 (gen) / 291 (end-to-end) | – | – | – | 85.7× |
| **PocketDiffusion (adaptive, fast mode)** | adaptive | 83 (gen) / 164 (end-to-end) | – | – | – | 159.6× |

| config | latency ms | peak RSS MB |
|---|---|---|
| CPU fallback 256px, LCM 2 steps, TAESD, FP32 | 1514 | 4286 |
| CPU fallback 384px, LCM 2 steps, TAESD, FP32 | 2708 | 4341 |


#### A.4 Peak memory consumption (CUDA allocator peak during one generation)

| system | peak MB | weights MB | activations MB | reduction |
|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 5065 | 4067 | 998 | 1.0× |
| + FP16 | 2676 | 2034 | 642 | 1.9× |
| + DPM-Solver++ 20 steps | 2676 | 2034 | 642 | 1.9× |
| + LCM-LoRA 4 steps (distilled) | 2675 | 2034 | 641 | 1.9× |
| + TAESD decoder | 2068 | 1879 | 189 | 2.4× |
| + INT8 weights | 1246 | 1062 | 184 | 4.1× |
| + distilled 6-layer text encoder | 1162 | 981 | 181 | 4.4× |
| SD-Turbo 1 step + TAESD | 2489 | 2305 | 183 | 2.0× |
| SD-Turbo 1 step + TAESD + INT8 | 1671 | 1483 | 188 | 3.0× |


#### A.5 Model size

| system | UNet M params | text-enc M | decoder M | total M | size MB | compression |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 859.5 | 123.1 | 83.65 | 1066.2 | 4067 | 1.0× |
| + FP16 | 859.5 | 123.1 | 83.65 | 1066.2 | 2034 | 2.0× |
| + DPM-Solver++ 20 steps | 859.5 | 123.1 | 83.65 | 1066.2 | 2034 | 2.0× |
| + LCM-LoRA 4 steps (distilled) | 859.5 | 123.1 | 83.65 | 1066.2 | 2034 | 2.0× |
| + TAESD decoder | 859.5 | 123.1 | 2.45 | 985.0 | 1879 | 2.2× |
| + INT8 weights | 859.5 | 123.1 | 2.45 | 985.0 | 1062 | 3.8× |
| + distilled 6-layer text encoder | 859.5 | 80.5 | 2.45 | 942.5 | 981 | 4.1× |
| SD-Turbo 1 step + TAESD | 865.9 | 340.4 | 2.45 | 1208.7 | 2305 | 1.8× |
| SD-Turbo 1 step + TAESD + INT8 | 865.9 | 340.4 | 2.45 | 1208.7 | 1483 | 2.7× |
| + INT4 weights | 859.5 | 123.1 | 2.45 | 985.0 | 678 | 6.0× |
| SD-Turbo 1 step + TAESD + INT4 | 865.9 | 340.4 | 2.45 | 1208.7 | 1096 | 3.7× |


#### A.6 Quality versus computational cost

| system | TFLOPs / image | latency ms | peak MB | CLIP | FID | CLIP per TFLOP |
|---|---|---|---|---|---|---|
| Baseline SD1.5 FP32, PNDM-50, CFG | 84.48 | 13307 | 5065 | 26.43 | 150.8 | 0.31 |
| + FP16 | 84.48 | 5170 | 2676 | 26.38 | 151.2 | 0.31 |
| + DPM-Solver++ 20 steps | 34.67 | 2120 | 2676 | 26.53 | 151.4 | 0.77 |
| + LCM-LoRA 4 steps (distilled) | 5.74 | 388 | 2675 | 25.24 | 157.4 | 4.40 |
| + TAESD decoder | 3.37 | 257 | 2068 | 25.27 | 158.2 | 7.50 |
| + INT8 weights | 3.37 | 366 | 1246 | 25.23 | 156.8 | 7.49 |
| + distilled 6-layer text encoder | 3.36 | 360 | 1162 | 25.21 | 157.0 | 7.50 |
| SD-Turbo 1 step + TAESD | 0.99 | 85 | 2489 | 26.27 | 150.3 | 26.52 |
| SD-Turbo 1 step + TAESD + INT8 | 0.99 | 104 | 1671 | 26.28 | 150.7 | 26.53 |


#### A.7 Semantic consistency on complex prompts (multiple objects, attributes, spatial relations)

| system | precision | recall | **object F1** | count acc | colour acc | spatial acc | all constraints | CLIP | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| Baseline SD1.5 FP16, 50 steps, CFG | 0.971 | 0.820 | 0.889 | 0.42 | 0.61 | 0.41 | 0.53 | 26.72 | 5195 |
| SD1.5 DPM-Solver++ 20 steps | 0.957 | 0.859 | 0.905 | 0.46 | 0.57 | 0.48 | 0.54 | 26.75 | 2138 |
| SD1.5 + LCM 4 steps + TAESD | 0.970 | 0.772 | 0.859 | 0.42 | 0.57 | 0.36 | 0.51 | 26.02 | 283 |
| SD-Turbo 1 step + TAESD | 0.964 | 0.917 | 0.940 | 0.33 | 0.64 | 0.57 | 0.59 | 27.25 | 106 |
| **PocketDiffusion (auto)** | 0.980 | 0.937 | 0.958 | 0.38 | 0.64 | 0.55 | 0.59 | 27.64 | 312 |
| PocketDiffusion + semantic re-ranking, N=4 ‡ | 0.970 | 0.942 | 0.956 | 0.42 | 0.70 | 0.66 | 0.66 | 27.56 | 485 |
| PocketDiffusion + semantic re-ranking, N=8 ‡ | 0.957 | 0.961 | 0.959 | 0.50 | 0.79 | 0.77 | 0.77 | 27.90 | 877 |


#### A.8 Resource requirements for mobile deployment targets

| device profile | memory budget MB | assumed TFLOPS | plan chosen by planner | pred. MB | pred. ms | status |
|---|---|---|---|---|---|---|
| This machine (GPU, measured) | 6500 | – | sd_turbo euler-1 @512px, none, taesd, TE=full, N=4 | 3476 | 501 | fits |
| Flagship phone NPU (simulated) | 3000 | 4.0 | sd_turbo euler-1 @384px, int8, taesd, TE=full, N=4 | 2297 | 533 | fits |
| Mid-range phone (simulated) | 1800 | 1.2 | sd_turbo euler-1 @384px, int8, taesd, TE=full, N=1 | 1675 | 448 | fits |
| Low-end phone (simulated) | 1100 | 0.3 | sd15 lcm-2 @256px, int8, taesd, TE=t6, N=1 | 1090 | 1477 | fits |
| This machine (CPU only, measured) | 6000 | – | sd_turbo euler-1 @384px, none, taesd, TE=full, N=1 | 2524 | 2791 | fits |


#### A.9 Quality degradation from compression / fewer steps (validation split)

| run | group | family | steps | px | weights | sparsity | TE | decoder | N | CLIP-L | gen ms | weights MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lcm4 | base | sd15 | 4 | 512 | none | 0.0 | full | taesd | 1 | 25.23 ± 0.46 | 256 | 1879 |
| turbo1 | base | sd_turbo | 1 | 512 | none | 0.0 | full | taesd | 1 | 26.26 ± 0.39 | 78 | 2305 |
| lcm_steps1 | steps | sd15 | 1 | 512 | none | 0.0 | full | taesd | 1 | 22.09 ± 0.52 | 87 | 1879 |
| lcm_steps2 | steps | sd15 | 2 | 512 | none | 0.0 | full | taesd | 1 | 24.97 ± 0.41 | 134 | 1879 |
| lcm_steps4 | steps | sd15 | 4 | 512 | none | 0.0 | full | taesd | 1 | 25.23 ± 0.46 | 253 | 1879 |
| lcm_steps8 | steps | sd15 | 8 | 512 | none | 0.0 | full | taesd | 1 | 25.02 ± 0.45 | 490 | 1879 |
| dpm_steps20 | steps | sd15 | 20 | 512 | none | 0.0 | full | taesd | 1 | 26.47 ± 0.43 | 1976 | 1879 |
| turbo_steps1 | steps | sd_turbo | 1 | 512 | none | 0.0 | full | taesd | 1 | 26.26 ± 0.39 | 82 | 2305 |
| turbo_steps2 | steps | sd_turbo | 2 | 512 | none | 0.0 | full | taesd | 1 | 26.03 ± 0.38 | 123 | 2305 |
| turbo_steps4 | steps | sd_turbo | 4 | 512 | none | 0.0 | full | taesd | 1 | 25.71 ± 0.39 | 233 | 2305 |
| lcm4_256px | resolution | sd15 | 4 | 256 | none | 0.0 | full | taesd | 1 | 15.22 ± 0.51 | 151 | 1879 |
| lcm4_384px | resolution | sd15 | 4 | 384 | none | 0.0 | full | taesd | 1 | 23.89 ± 0.46 | 182 | 1879 |
| turbo1_256px | resolution | sd_turbo | 1 | 256 | none | 0.0 | full | taesd | 1 | 24.44 ± 0.41 | 48 | 2305 |
| turbo1_384px | resolution | sd_turbo | 1 | 384 | none | 0.0 | full | taesd | 1 | 26.26 ± 0.41 | 50 | 2305 |
| lcm4_int8 | quant | sd15 | 4 | 512 | int8 | 0.0 | full | taesd | 1 | 25.22 ± 0.48 | 363 | 1062 |
| lcm4_int4 | quant | sd15 | 4 | 512 | int4 | 0.0 | full | taesd | 1 | 24.67 ± 0.43 | 558 | 678 |
| turbo1_int8 | quant | sd_turbo | 1 | 512 | int8 | 0.0 | full | taesd | 1 | 26.38 ± 0.40 | 107 | 1483 |
| turbo1_int4 | quant | sd_turbo | 1 | 512 | int4 | 0.0 | full | taesd | 1 | 25.73 ± 0.36 | 157 | 1096 |
| lcm4_prune10 | prune | sd15 | 4 | 512 | none | 0.1 | full | taesd | 1 | 25.22 ± 0.46 | 259 | 1879 |
| lcm4_prune20 | prune | sd15 | 4 | 512 | none | 0.2 | full | taesd | 1 | 24.98 ± 0.49 | 259 | 1879 |
| lcm4_prune30 | prune | sd15 | 4 | 512 | none | 0.3 | full | taesd | 1 | 23.30 ± 0.53 | 259 | 1879 |
| lcm4_prune50 | prune | sd15 | 4 | 512 | none | 0.5 | full | taesd | 1 | 13.76 ± 0.32 | 256 | 1879 |
| turbo1_prune10 | prune | sd_turbo | 1 | 512 | none | 0.1 | full | taesd | 1 | 26.41 ± 0.39 | 82 | 2305 |
| turbo1_prune20 | prune | sd_turbo | 1 | 512 | none | 0.2 | full | taesd | 1 | 26.66 ± 0.42 | 76 | 2305 |
| turbo1_prune30 | prune | sd_turbo | 1 | 512 | none | 0.3 | full | taesd | 1 | 26.13 ± 0.34 | 76 | 2305 |
| lcm4_t6 | text_encoder | sd15 | 4 | 512 | none | 0.0 | t6 | taesd | 1 | 25.14 ± 0.43 | 254 | 1798 |
| lcm4_fullvae | decoder | sd15 | 4 | 512 | none | 0.0 | full | full | 1 | 24.96 ± 0.47 | 387 | 2034 |
| turbo1_fullvae | decoder | sd_turbo | 1 | 512 | none | 0.0 | full | full | 1 | 26.15 ± 0.40 | 208 | 2460 |
| lcm4_best_of_2 | rerank | sd15 | 4 | 512 | none | 0.0 | full | taesd | 2 | 25.36 ± 0.42 | 426 | 1879 |
| lcm4_best_of_4 | rerank | sd15 | 4 | 512 | none | 0.0 | full | taesd | 4 | 26.53 ± 0.40 | 770 | 1879 |
| turbo1_best_of_2 | rerank | sd_turbo | 1 | 512 | none | 0.0 | full | taesd | 2 | 26.56 ± 0.39 | 128 | 2305 |
| turbo1_best_of_4 | rerank | sd_turbo | 1 | 512 | none | 0.0 | full | taesd | 4 | 26.92 ± 0.40 | 220 | 2305 |


#### A.10 Ablation study


##### A. System components

| variant | CLIP-L | PocketIQA | NIQE↓ | artifact rate | latency ms |
|---|---|---|---|---|---|
| full_system | 26.71 | 87.0 | 3.85 | 0.07 | 305 |
| minus_rerank | 26.36 | 87.1 | 3.75 | 0.12 | 169 |
| minus_postprocess | 26.75 | 86.9 | 3.87 | 0.21 | 236 |
| minus_complexity_adaptation | 26.38 | 86.9 | 3.77 | 0.09 | 219 |
| minus_planner_fixed_config | 26.14 | 86.9 | 3.74 | 0.10 | 154 |


##### B. Classical post-processing stages (leave-one-out)

| variant | CLIP-L | PocketIQA | NIQE↓ | artifact rate | ms / image |
|---|---|---|---|---|---|
| none | 26.11 | 87.0 | 3.78 | 0.24 | 0 |
| all | 26.16 | 86.9 | 3.74 | 0.10 | 94 |
| all_minus_grid_notch | 26.16 | 86.9 | 3.74 | 0.10 | 104 |
| all_minus_denoise | 26.16 | 86.9 | 3.74 | 0.11 | 96 |
| all_minus_white_balance | 26.16 | 87.1 | 3.74 | 0.21 | 98 |
| all_minus_adaptive_gamma | 26.16 | 86.9 | 3.74 | 0.10 | 94 |
| all_minus_clahe | 26.16 | 86.9 | 3.74 | 0.10 | 94 |
| all_minus_saturation_knee | 26.16 | 86.9 | 3.74 | 0.10 | 94 |
| all_minus_sharpen | 26.11 | 86.9 | 3.78 | 0.12 | 59 |


##### C. Resolution optimisation

| variant | CLIP-L | PocketIQA | NIQE↓ | gen ms | upscale ms |
|---|---|---|---|---|---|
| native_512 | 26.11 | 87.0 | 3.78 | 76 | – |
| 384_to_512_nearest | 26.10 | 86.2 | 6.65 | 54 | 1.8 |
| 384_to_512_bilinear | 25.91 | 83.8 | 5.90 | 54 | 2.4 |
| 384_to_512_bicubic | 26.10 | 85.8 | 5.00 | 54 | 2.3 |
| 384_to_512_lanczos | 26.07 | 86.1 | 4.98 | 54 | 2.6 |
| 384_to_512_ibp | 26.32 | 87.5 | 3.83 | 54 | 54.0 |


### B. Compositional benchmark (58 annotated prompts × 2 seeds)

#### B.1 All-constraints rate per category

| Category | Prompts | Baseline SD1.5 50 steps | PocketDiffusion | + semantic re-rank N=8 |
|---|---|---|---|---|
| single | 8 | 94 % | 100 % | 100 % |
| two_objects | 10 | 75 % | 100 % | 100 % |
| counting | 8 | 44 % | 50 % | 62 % |
| colors | 8 | 62 % | 88 % | 100 % |
| color_binding | 4 | 38 % | 25 % | 50 % |
| spatial | 10 | 45 % | 45 % | 85 % |
| complex | 10 | 10 % | 0 % | 30 % |

#### B.2 Prompts and annotations

| ID | Category | Prompt | Objects (count, colour) | Relations |
|---|---|---|---|---|
| s01 | single | a photo of a dog | dog | – |
| s02 | single | a photo of a bicycle | bicycle | – |
| s03 | single | a photo of a clock | clock | – |
| s04 | single | a photo of a pizza | pizza | – |
| s05 | single | a photo of a giraffe | giraffe | – |
| s06 | single | a photo of an umbrella | umbrella | – |
| s07 | single | a photo of a laptop | laptop | – |
| s08 | single | a photo of a teddy bear | teddy bear | – |
| t01 | two_objects | a photo of a cat and a laptop | cat, laptop | – |
| t02 | two_objects | a photo of a dog and a frisbee | dog, frisbee | – |
| t03 | two_objects | a photo of a horse and a car | horse, car | – |
| t04 | two_objects | a photo of a cup and a book | cup, book | – |
| t05 | two_objects | a photo of a bench and a bird | bench, bird | – |
| t06 | two_objects | a photo of an elephant and a boat | elephant, boat | – |
| t07 | two_objects | a photo of a banana and a clock | banana, clock | – |
| t08 | two_objects | a photo of a toaster and a vase | toaster, vase | – |
| t09 | two_objects | a photo of a skateboard and a backpack | skateboard, backpack | – |
| t10 | two_objects | a photo of a pizza and a wine glass | pizza, wine glass | – |
| c01 | counting | a photo of two dogs | dog ×2 | – |
| c02 | counting | a photo of three apples | apple ×3 | – |
| c03 | counting | a photo of two cups | cup ×2 | – |
| c04 | counting | a photo of four bananas | banana ×4 | – |
| c05 | counting | a photo of two birds | bird ×2 | – |
| c06 | counting | a photo of three oranges | orange ×3 | – |
| c07 | counting | a photo of two cars | car ×2 | – |
| c08 | counting | a photo of three books | book ×3 | – |
| k01 | colors | a photo of a red car | car (red) | – |
| k02 | colors | a photo of a blue umbrella | umbrella (blue) | – |
| k03 | colors | a photo of a yellow bus | bus (yellow) | – |
| k04 | colors | a photo of a green apple | apple (green) | – |
| k05 | colors | a photo of a white cat | cat (white) | – |
| k06 | colors | a photo of a purple vase | vase (purple) | – |
| k07 | colors | a photo of a black dog | dog (black) | – |
| k08 | colors | a photo of a pink cake | cake (pink) | – |
| k09 | color_binding | a photo of a red cup and a blue book | cup (red), book (blue) | – |
| k10 | color_binding | a photo of a yellow umbrella and a green bench | umbrella (yellow), bench (green) | – |
| k11 | color_binding | a photo of a white car and a red bicycle | car (white), bicycle (red) | – |
| k12 | color_binding | a photo of a black cat and a white dog | cat (black), dog (white) | – |
| p01 | spatial | a photo of a dog to the left of a bicycle | dog, bicycle | dog left of bicycle |
| p02 | spatial | a photo of a cup to the right of a laptop | cup, laptop | cup right of laptop |
| p03 | spatial | a photo of a clock above a bed | clock, bed | clock above bed |
| p04 | spatial | a photo of a cat below a chair | cat, chair | cat below chair |
| p05 | spatial | a photo of a bird above a bench | bird, bench | bird above bench |
| p06 | spatial | a photo of a car to the left of a truck | car, truck | car left of truck |
| p07 | spatial | a photo of a teddy bear on top of a chair | teddy bear, chair | teddy bear above chair |
| p08 | spatial | a photo of a vase to the right of a book | vase, book | vase right of book |
| p09 | spatial | a photo of a horse next to a car | horse, car | horse next to car |
| p10 | spatial | a photo of an umbrella above a dog | umbrella, dog | umbrella above dog |
| x01 | complex | a red apple to the left of a green bottle on a wooden table | apple (red), bottle (green), dining table | apple left of bottle, bottle above dining table |
| x02 | complex | two cats sleeping on a blue couch next to a laptop | cat ×2, couch (blue), laptop | cat above couch, couch next to laptop |
| x03 | complex | a yellow bus parked in front of a red building with a person on a bicycle | bus (yellow), person, bicycle | person above bicycle |
| x04 | complex | three oranges in a white bowl next to a knife | orange ×3, bowl (white), knife | bowl next to knife |
| x05 | complex | a black dog and a white cat sitting on a red bench in a park | dog (black), cat (white), bench (red) | cat above bench |
| x06 | complex | a laptop to the left of a cup of coffee and a potted plant on a desk | laptop, cup, potted plant, dining table | laptop left of cup, potted plant above dining table |
| x07 | complex | a person holding a blue umbrella next to a yellow taxi on a rainy street | person, umbrella (blue), car (yellow) | umbrella next to car |
| x08 | complex | two horses standing in a field beside a white truck | horse ×2, truck (white) | – |
| x09 | complex | a green bicycle leaning on a bench under a clock | bicycle (green), bench, clock | bench below clock |
| x10 | complex | a pizza and two wine glasses on a table next to a vase of flowers | pizza, wine glass ×2, dining table, vase | dining table next to vase |

### C. Device profiles used by the planner

| Key | Profile | Kind | Memory budget | Assumed TFLOPS | Runs on |
|---|---|---|---|---|---|
| `this_gpu` | This machine (GPU, measured) | measured | 6,500 MB | – (measured) | cuda |
| `flagship_phone` | Flagship phone NPU (simulated) | simulated | 3,000 MB | 4.0 | cuda |
| `midrange_phone` | Mid-range phone (simulated) | simulated | 1,800 MB | 1.2 | cuda |
| `lowend_phone` | Low-end phone (simulated) | simulated | 1,100 MB | 0.3 | cuda |
| `this_cpu` | This machine (CPU only, measured) | measured | 6,000 MB | – (measured) | cpu |

### D. Generation presets

| Preset | Family | Precision | Scheduler | Steps | CFG | LCM-LoRA | Decoder | Weights | Text encoder | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | sd15 | fp32 | pndm | 50 | 7.5 | no | full | none | full | 512×512 |
| `sd15_fp16` | sd15 | fp16 | pndm | 50 | 7.5 | no | full | none | full | 512×512 |
| `sd15_fp16_dpm20` | sd15 | fp16 | dpm | 20 | 7.5 | no | full | none | full | 512×512 |
| `sd15_lcm4` | sd15 | fp16 | lcm | 4 | 1.0 | yes | full | none | full | 512×512 |
| `sd15_lcm4_taesd` | sd15 | fp16 | lcm | 4 | 1.0 | yes | taesd | none | full | 512×512 |
| `sd15_lcm4_taesd_int8` | sd15 | fp16 | lcm | 4 | 1.0 | yes | taesd | int8 | full | 512×512 |
| `sd15_lcm4_taesd_int8_t6` | sd15 | fp16 | lcm | 4 | 1.0 | yes | taesd | int8 | t6 | 512×512 |
| `turbo1_taesd` | sd_turbo | fp16 | euler | 1 | 0.0 | no | taesd | none | full | 512×512 |
| `turbo1_taesd_int8` | sd_turbo | fp16 | euler | 1 | 0.0 | no | taesd | int8 | full | 512×512 |
| `cpu_fallback` | sd15 | fp32 | lcm | 2 | 1.0 | yes | taesd | none | full | 256×256 |

### E. REST API (FastAPI backend)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/system` | hardware, models, queue, device profiles, presets, examples |
| POST | `/api/analyze` | prompt -> scene graph, complexity, token count   (CPU, instant) |
| POST | `/api/plan` | prompt + budget -> planner decision              (CPU, instant) |
| POST | `/api/generate` | enqueue a generation job -> {job_id} |
| POST | `/api/compare` | enqueue baseline-vs-optimised race -> {job_id} |
| GET | `/api/jobs/{id}/events` | Server-Sent Events: queued/start/analysis/plan/step/stage/result/error/end |
| GET | `/api/generations` | history (paginated, filter: favorites, family, q) |
| GET | `/api/generations/{id}` | full record (candidates, stages, frames, metrics) |
| GET | `/api/stats` | aggregate statistics from the database |
| GET | `/api/benchmarks` | evaluation results (results/*.json) for the Lab charts |
| GET | `/media/...` | stored images;  GET /  -> single-page frontend |

### F. Automated tests

| Command | Result |
|---|---|
| `python -m pytest -q tests` | `36 passed in 6.79s` |

| # | Test file | Test |
|---|---|---|
| 1 | `tests/test_api.py` | `test_index_and_static` |
| 2 | `tests/test_api.py` | `test_analyze` |
| 3 | `tests/test_api.py` | `test_plan_respects_device` |
| 4 | `tests/test_api.py` | `test_validation_errors[body0]` |
| 5 | `tests/test_api.py` | `test_validation_errors[body1]` |
| 6 | `tests/test_api.py` | `test_validation_errors[body2]` |
| 7 | `tests/test_api.py` | `test_validation_errors[body3]` |
| 8 | `tests/test_api.py` | `test_generate_rejects_empty_prompt` |
| 9 | `tests/test_api.py` | `test_history_and_stats_on_empty_db` |
| 10 | `tests/test_api.py` | `test_db_roundtrip` |
| 11 | `tests/test_core.py` | `test_parser_objects[a photo of two dogs-expect0]` |
| 12 | `tests/test_core.py` | `test_parser_objects[a red cup and a blue book-expect1]` |
| 13 | `tests/test_core.py` | `test_parser_objects[three oranges in a white bowl-expect2]` |
| 14 | `tests/test_core.py` | `test_parser_objects[an orange cat on a sofa-expect3]` |
| 15 | `tests/test_core.py` | `test_parser_objects[a pizza and two wine glasses-expect4]` |
| 16 | `tests/test_core.py` | `test_parser_objects[a yellow bus-expect5]` |
| 17 | `tests/test_core.py` | `test_parser_relations_and_complexity` |
| 18 | `tests/test_core.py` | `test_bad_prompts` |
| 19 | `tests/test_core.py` | `test_guided_filter_preserves_constant_and_edges` |
| 20 | `tests/test_core.py` | `test_grid_detector` |
| 21 | `tests/test_core.py` | `test_artifact_detection_flags` |
| 22 | `tests/test_core.py` | `test_postprocess_improves_dark_noisy_image` |
| 23 | `tests/test_core.py` | `test_postprocess_leaves_clean_image_mostly_alone` |
| 24 | `tests/test_core.py` | `test_resize_and_snap` |
| 25 | `tests/test_core.py` | `test_iqa_runs_and_orders` |
| 26 | `tests/test_core.py` | `test_ensure_rgb_uint8_handles_odd_inputs` |
| 27 | `tests/test_core.py` | `test_quantization_roundtrip` |
| 28 | `tests/test_core.py` | `test_planner_respects_budgets` |
| 29 | `tests/test_core.py` | `test_short_prompt_chunking_equals_standard_encoding` |
| 30 | `tests/test_core.py` | `test_long_prompt_is_split_into_chunks_not_truncated` |
| 31 | `tests/test_core.py` | `test_output_size_keywords` |
| 32 | `tests/test_core.py` | `test_super_resolve_8k_shape_and_range` |
| 33 | `tests/test_core.py` | `test_snap_resolution_keeps_aspect_for_8k` |
| 34 | `tests/test_core.py` | `test_spelling_correction_only_fixes_real_typos` |
| 35 | `tests/test_core.py` | `test_real_word_typos_fixed_only_in_context_and_characters_counted` |
| 36 | `tests/test_core.py` | `test_face_pass_helpers` |

### G. Hardware and software environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| PyTorch | 2.14.1+cu126 |
| Benchmarks measured | 2026-11-06 20:08 |
| accelerate | 1.15.0 |
| diffusers | 0.42.1 |
| gradio | 6.31.0 |
| hf-xet | 1.8.0 |
| huggingface_hub | 2.4.0 |
| matplotlib | 3.11.2 |
| numpy | 2.4.6 |
| opencv-python-headless | 5.0.0.93 |
| pandas | 3.1.1 |
| peft | 0.21.2 |
| pillow | 12.3.0 |
| psutil | 7.2.2 |
| pyarrow | 26.0.1 |
| pytest | 9.2.0 |
| requests | 2.34.2 |
| safetensors | 0.8.0 |
| scipy | 1.17.1 |
| torchmetrics | 1.9.0 |
| tqdm | 4.70.1 |
| transformers | 5.22.1 |
| torch_fidelity | 0.4.0 |
| pyspellchecker | 0.8.3 |
| markdown | 3.7 |

### H. Repository structure

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

<!-- APPENDIX:END -->

---

## Student Submission Checklist

- [ ] Working GitHub repository link included with an updated `README.md`.
- [ ] No raw code pasted into this report.
- [ ] All table metrics match test runs reproducible via the repository (`python -m eval.run_all`).
- [ ] All team member details accurately recorded (`team.json` filled, then `python -m eval.make_report` re-run).
- [ ] Live demo link tested from a phone on mobile data.
- [ ] `python scripts/ai_share.py Solution_Report_P3.md` reports PASS (≤ 30 % AI text, no placeholders left).
