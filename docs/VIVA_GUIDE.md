# Viva & demo preparation (internal – not part of the report)

Demo and viva are worth 25 points; every member must explain *their own part*, the model and
classical choices, the metrics and a failure case. Suggested split for 4 members:

| Member | Owns | Files to know line by line |
|---|---|---|
| 1 | Diffusion engine & optimisation (FP16, LCM-LoRA, SD-Turbo, TAESD, INT8/INT4, pruning) | `engine.py`, `optimizations.py`, `config.py` |
| 2 | Adaptive planner, fallback, device profiles, text-encoder distillation | `planner.py`, `scripts/distill_text_encoder.py` |
| 3 | Classical CV: artifacts, post-processing, guided filter, IBP, NIQE | `classical/*` |
| 4 | Evaluation & app: CLIP/FID/KID, compositional F1 (OWLv2), ablation, Gradio UI | `eval/*`, `verifier.py`, `scoring.py`, `app.py` |

## 3-minute demo script
1. *(20 s)* Problem: SD1.5 needs 50 steps × 2 (CFG) = 102 UNet calls and ~4 GB FP32 weights → impossible on a phone.
2. *(40 s)* **Baseline vs optimised** tab with the judge's prompt: show the speed-up and memory numbers.
3. *(60 s)* **Generate** tab with a complex prompt: open "Why this plan", then switch the device profile to *Low-end phone*
   → the planner switches to INT4 / 256 px / fallback. Show the denoising trajectory and the post-processing stages.
4. *(40 s)* **Semantic check** tab: boxes, colour and relation checks → explain object F1.
5. *(20 s)* **Benchmarks** tab: quality-vs-latency plot, one failure case and its cause.

## Questions you should be able to answer
- **Why is the baseline slow?** 50 PNDM steps (+1) with classifier-free guidance → 102 UNet evaluations of an 860M-param network; FP32 doubles memory traffic.
- **How does LCM-LoRA work?** Consistency distillation trains a LoRA so the UNet maps any noisy latent on the ODE trajectory directly to x0; 2–8 steps suffice and no CFG is needed. We *fuse* the LoRA (W ← W + BA) so it costs nothing at run time.
- **SD-Turbo?** Adversarial Diffusion Distillation: a student trained with a GAN loss + distillation loss from a teacher, producing usable images in 1 step.
- **TAESD?** A 2.4M-param decoder distilled to mimic the 83.7M-param VAE decoder; ~10× cheaper decode, slightly softer details.
- **INT8 / INT4?** Symmetric per-output-channel scale s = max|w|/127 (INT8) or per-64-weight group s = max|w|/7 (INT4, two values per byte). Weights are de-quantised on the fly inside `forward`, so resident memory halves/quarters; GPU latency gets slightly *worse* because there are no fused INT kernels – that is why the planner only chooses quantisation when memory forces it.
- **Why did pruning hurt so much?** Unstructured magnitude pruning without fine-tuning removes weights the network relies on; it also gives no speed-up on dense hardware. Structured pruning + KD (e.g. BK-SDM) is the realistic path – stated as future work.
- **What did you train?** The 6-layer text encoder by knowledge distillation (MSE + cosine on all 77 token states) on COCO *train* captions, early-stopped on *validation* captions.
- **How does the planner decide?** Enumerates ~700 configs, predicts latency (calibrated per-step UNet time × steps × resolution^1.1 × batch^0.85 + decode + post), memory (weights × bytes per precision + measured activations) and CLIP (measured on the validation split); maximises CLIP − λ·log2(latency/100 ms) inside the budgets; λ falls as prompt complexity rises.
- **What is "anytime" generation?** Every step we compute the x0 estimate; if the next step would miss the deadline we stop and decode x0 instead of failing.
- **Guided filter?** Local linear model q = a·I + b in each window; a = cov(I,p)/(var(I)+ε). Large ε → smoothing, small ε → edge-preserving. O(N) via box filters.
- **NIQE?** Natural images have Gaussian-like MSCN statistics; we fit GGD/AGGD parameters per patch (36 features at 2 scales), fit a multivariate Gaussian on pristine photos, and report the Mahalanobis-type distance of the test image's Gaussian. Lower = more natural.
- **Why CLIP-B/32 for re-ranking but CLIP-L/14 for evaluation?** To avoid optimising the exact metric we report (metric leakage).
- **FID with 200 images?** Biased upwards (needs ~10k for absolute values) but comparable across rows with equal N; KID is unbiased at small N.
- **How is object F1 computed?** TP = requested class detected, FN = requested class missing, FP = one of 6 distractor classes detected (hallucination). Micro-averaged over all prompts.
- **Why not 100 % F1?** Object F1 saturates at ~0.96. We inspected every remaining miss: about half are *evaluator* errors (the object is visibly there - a cropped red car, a small cyclist - but OWLv2 misses it), the rest are attribute-binding failures of the diffusion model ("black cat and white dog" became two black-and-white dogs) because the CLIP text encoder mixes adjectives between nouns. Lowering the detector threshold after seeing the results would be tuning the test, so we did not.
- **How did you improve semantic consistency?** (1) Object-aware re-ranking: each requested object is scored separately with CLIP-B/32 ("a photo of a red apple") and the candidate whose *weakest* object scores best wins. (2) The planner gives more candidates to prompts with more checkable constraints. (3) Semantic re-ranking with OWL-ViT v1 - a *different* detector from the OWLv2 evaluator. With 8 candidates, "all constraints satisfied" rose from 53 % (baseline) to 77 %.
- **Why was the earlier 0.976 F1 dropped?** That variant selected images with the same OWLv2 detector that grades them (metric leakage). With an independent selector it scores 0.956 - we report the honest number.
- **Fallback for constrained devices?** Planner: no config fits → fastest config that fits memory. Runtime: OOM → fewer candidates → 384 px → TAESD → INT8 → 256 px → CPU.
