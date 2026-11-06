# Design document – PocketDiffusion (P3)

> **How to use this file.** The headings are the sections a design document needs. Under each
> heading there are *fact notes* (numbers, file names, design decisions) so nothing is missed.
> **Write the actual paragraphs in your own words** — the hackathon caps AI-written report
> content at 30 %. Delete the notes when you are done.

## 1. Problem statement
- P3: lightweight text-to-image generation for resource-constrained mobile devices.
- Must: semantic alignment, visual quality, diversity, while cutting compute, memory, latency.
- Not allowed: just calling a public text-to-image API.

## 2. Goals and non-goals
- Goals: ≥2 optimisation strategies (we have 6), baseline comparison, adaptive budget-aware generation, fallback for constrained devices, classical CV stages, evaluation with CLIP / FID / latency / memory / size.
- Non-goals: training a diffusion model from scratch, real on-phone deployment (we simulate phone budgets; CPU run is the real constrained proxy).

## 3. System architecture
- Diagram: `docs/architecture.md` (Mermaid) – matches modules in `pocketdiff/`.
- Data flow: analyse → plan → generate → screen → re-rank → post-process → verify/safety.
- Explain *why* each stage exists (what failure it prevents).

## 4. Data
- MS-COCO Karpathy split via `yerevann/coco-karpathy` (captions + image URLs).
- test → final metrics; validation → planner calibration + NIQE pristine model; train → text-encoder distillation. No overlap.
- Custom compositional benchmark: `prompts/compositional.json` (58 prompts with annotated objects/counts/colours/relations, written with AI assistance - declare it; categories single, two_objects, counting, colors, color_binding, spatial, complex).

## 5. Deep-learning components
- Baseline: Stable Diffusion v1.5 (UNet 859.5M, CLIP-L text encoder 123M, VAE 83.7M), PNDM 50 steps, CFG 7.5, FP32.
- Optimised families: SD1.5 + LCM-LoRA (consistency distillation adapter, *fused* into weights → zero runtime overhead), SD-Turbo (adversarial diffusion distillation, 1 step).
- Own denoising loop (`pocketdiff/engine.py`): x0 estimate every step → deadline-aware "anytime" output, previews, per-step timing.
- Prompt encoding: CLIP tokenizer (77 tokens, truncation warning), cached embeddings.
- Meaningful adaptation: (1) LCM-LoRA fusion onto SD1.5, (2) our distilled 6-layer text encoder (KD, train/val split), (3) planner calibrated on validation data.

## 6. Model optimisation (each with measured effect – see eval file)
- Reduced precision FP32→FP16.
- Fewer sampling steps: DPM-Solver++ (20), LCM (4), SD-Turbo (1); CFG removed for distilled models (halves UNet batch).
- Knowledge distillation: LCM-LoRA, SD-Turbo, own text-encoder KD (12→6 layers).
- Latent-space / lightweight architecture: TAESD decoder (2.4M vs 83.7M params), lower native latent resolution + classical super-resolution.
- Weight-only quantisation implemented by us: INT8 per-channel, INT4 group-wise (64) with nibble packing.
- Pruning study: layer-wise magnitude pruning 10–50 % (quality degradation curve).

## 7. Classical computer-vision components
- Resolution optimisation: aspect-preserving snapping to multiples of 8 under a pixel budget; IBP super-resolution vs nearest/bilinear/bicubic/Lanczos.
- Colour correction: shades-of-gray white balance (p=6) with severity-scaled strength; adaptive gamma; CLAHE on L*; saturation soft-knee; vibrance.
- Edge-aware post-processing: guided filter (He et al.) implemented with box filters (gray & colour guide); base/detail sharpening with edge mask (no halos, no noise boost).
- Artifact detection: blank/NaN, clipping, crushed shadows, Lab cast factor K=D/M, dynamic range, over-saturation, Immerkær noise σ, 8-px periodic grid via FFT peaks, Laplacian blur index → severities in [0,1]; FFT notch filter for grids.
- Image-quality assessment: NIQE from scratch (MSCN, GGD/AGGD fits, 36-D features, MVG distance) with a COCO-fitted pristine model; PocketIQA composite; colourfulness (Hasler–Süsstrunk), RMS contrast, entropy.

## 8. Adaptive / resource-aware strategy
- Planner enumerates family × steps × resolution × decoder × precision × text encoder × N candidates (~700 configs).
- Utility = predicted CLIP − λ·log2(latency/100 ms); λ shrinks with prompt complexity in auto mode.
- Predicted re-ranking gain scales with the number of checkable constraints (objects + colours + counts + relations), so multi-object prompts get 4 candidates.
- Object-aware re-ranking: whole-prompt CLIP + per-object CLIP coverage (0.5·min + 0.5·mean of "a photo of a red apple"-style phrases).
- Semantic re-ranking uses OWL-ViT v1 (selector), evaluation uses OWLv2 (no metric leakage). Earlier leaky version gave 0.976 F1; honest version 0.956 — report the honest one and explain.
- Constraints: latency budget, memory budget of the device profile. Infeasible → fallback to fastest config that fits memory.
- Runtime fallback chain on OOM: N↓ → 384 px → TAESD → INT8 → 256 px → CPU.
- Device profiles: this GPU (measured), flagship / mid-range / low-end phone (simulated: FLOPs ÷ assumed TFLOPS), this CPU (measured).

## 9. Evaluation methodology
- Metrics in the order of the problem statement: CLIP (ViT-L/14), FID + KID, latency, peak memory, model size, quality-vs-cost.
- Extra: compositional object precision/recall/F1, count/colour/spatial accuracy (OWLv2 + HSV + geometry).
- Fairness: same prompts and seeds for every system; warm-up before timing; CUDA synchronisation; models evicted between configs.
- Re-ranking uses CLIP-B/32, evaluation uses CLIP-L/14 (no metric leakage).

## 10. Results
- Copy the tables from `eval_P3_<TeamID>.md`; discuss trade-offs (which optimisation buys what).

## 11. Ablation study
- System components (re-ranking, post-processing, complexity adaptation, planner), post-processing stage leave-one-out, resolution optimisation.

## 12. Failure analysis (≥3 cases)
- Use `results/failures/` + section 11 of the eval file. Typical causes to discuss: counting (CLIP text encoder has no notion of number), attribute binding (colour leaks to the wrong object), spatial relations (bag-of-words conditioning), detector misses on stylised images, INT4 / pruning artefacts.

## 13. Deployment
- Gradio app (`python app.py --share` → public URL), queue with concurrency 1, loading/progress states, error messages, warm-up at start.
- One-command setup + Dockerfile; no secrets in the repo.

## 14. Limitations
- Phone latencies are estimates; NIQE pristine model is COCO-fitted (not comparable to MATLAB NIQE); FID at small N is biased; zero-shot safety check is coarse; detector-based F1 inherits detector errors.

## 15. Ethics
- Model licences (SD1.5: CreativeML OpenRAIL-M; SD-Turbo: Stability AI non-commercial research licence); dataset biases of LAION/COCO; misuse (deepfakes, NSFW) and our mitigations (block-list + image check); energy savings as a positive.

## 16. Team contributions
- Who built / can explain which module (viva requirement: every member explains their own part).

## 17. AI-assistance declaration
- See `docs/AI_DECLARATION.md` – must be stated explicitly in the report.

## 18. References
- Rombach et al. 2022 (LDM/SD); Luo et al. 2023 (LCM, LCM-LoRA); Sauer et al. 2023 (ADD/SD-Turbo); Lu et al. 2022 (DPM-Solver++); Kim et al. 2023 (BK-SDM); He et al. 2013 (guided filter); Mittal et al. 2013 (NIQE); Irani & Peleg 1991 (IBP); Finlayson & Trezzi 2004 (shades of gray); Hasler & Süsstrunk 2003 (colourfulness); Immerkær 1996 (noise estimation); Hessel et al. 2021 (CLIPScore); Heusel et al. 2017 (FID); Bińkowski et al. 2018 (KID); Minderer et al. 2023 (OWLv2); Ghosh et al. 2023 (GenEval).
