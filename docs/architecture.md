# PocketDiffusion – system architecture

```mermaid
flowchart LR
    subgraph S[0 · Full stack]
        FE[Web frontend SPA<br/>studio · gallery · lab · race<br/>live x0 frames via SSE]
        API[FastAPI backend<br/>REST + validation]
        WK[GPU worker<br/>job queue · event log]
        DB[(SQLite WAL<br/>generations · candidates<br/>stages · frames)]
    end
    FE <--> API --> WK --> DB
    API --> DB
    WK --> U

    U([User prompt + device profile + latency/memory budget]) --> PA

    subgraph A[1 · Understand]
        PA[Prompt analyser<br/>rule-based scene graph:<br/>objects · counts · colours · relations<br/>complexity score · safety block-list]
    end

    subgraph B[2 · Plan - resource-aware]
        PL[Adaptive planner<br/>enumerates ~700 configs<br/>predicts latency · memory · CLIP<br/>calibrated on validation split]
        FB[Fallback chain<br/>N↓ → px↓ → TAESD → INT8 → 256px → CPU]
    end

    subgraph C[3 · Generate - optimised diffusion]
        TE[Text encoder<br/>CLIP-L 12L or our distilled 6L<br/>+ embedding cache]
        UN[UNet<br/>SD-Turbo 1-step or SD1.5+LCM-LoRA fused<br/>FP16 · INT8/INT4 weight-only quant<br/>SDPA attention]
        LOOP[Own denoising loop<br/>anytime x0 · deadline stop<br/>per-step timing · previews]
        DEC[Decoder<br/>TAESD 2.4M or VAE 84M]
    end

    subgraph D[4 · Select & correct - classical CV]
        SC[Artifact screening<br/>blank/NaN · exposure · cast · noise<br/>8-px grid FFT · saturation · blur]
        RR[Best-of-N re-ranking<br/>CLIP ViT-B/32 + PocketIQA]
        PP[Adaptive post-processing<br/>FFT notch · guided-filter denoise<br/>shades-of-gray WB · adaptive gamma<br/>CLAHE · saturation knee · IBP SR<br/>edge-aware sharpening]
    end

    subgraph E[5 · Verify & report]
        V[Semantic check<br/>OWLv2 boxes + HSV colour + geometry<br/>object P/R/F1]
        IQA[IQA<br/>NIQE from scratch · PocketIQA]
        S[Safety: CLIP zero-shot image check]
    end

    PA --> PL --> TE --> LOOP
    UN <--> LOOP
    LOOP --> DEC --> SC --> RR --> PP --> S --> OUT([Image + plan + telemetry])
    PL -.OOM.-> FB -.-> LOOP
    PP --> IQA
    S --> V
```

| Stage | Module | Classical / DL | Key idea |
|---|---|---|---|
| Frontend | `frontend/` (HTML/CSS/JS) | – | framework-free SPA: live denoising stream, planner preview, before/after, detections overlay, gallery, SVG charts, race |
| Backend | `backend/main.py`, `backend/jobs.py` | – | FastAPI REST + Server-Sent Events; one GPU worker thread owns the model |
| Database | `backend/db.py` | – | SQLite (WAL): every generation, candidate, post-processing stage and denoising frame |
| Prompt analysis | `pocketdiff/prompt_analyzer.py` | classical (rules) | scene graph + complexity drives planning and verification |
| Planner | `pocketdiff/planner.py` | data-driven model | maximise predicted CLIP − λ·log(latency) s.t. latency & memory budgets |
| Engine | `pocketdiff/engine.py` | DL | own sampling loop, anytime output, model cache, OOM fallback |
| Compression | `pocketdiff/optimizations.py` | DL | FP16, INT8/INT4 weight-only quantisation, magnitude pruning, LoRA fusion |
| Distillation | `scripts/distill_text_encoder.py` | DL (training) | 12→6-layer text encoder, COCO train/val split |
| Artifacts | `pocketdiff/classical/artifacts.py` | classical | severities, not booleans, so corrections scale with the problem |
| Post-processing | `pocketdiff/classical/postprocess.py` | classical | guided filter, WB, gamma, CLAHE, FFT notch, IBP super-resolution |
| IQA | `pocketdiff/classical/iqa.py` | classical | NIQE (MSCN + AGGD + MVG) from scratch, PocketIQA composite |
| Re-ranking / safety | `pocketdiff/scoring.py` | DL | CLIP-B/32 in the loop, CLIP-L/14 only for evaluation |
| Verification | `pocketdiff/verifier.py` | hybrid | OWLv2 detection + HSV colour naming + box geometry |
| App | `app.py` | – | Gradio UI with loading/error states, trajectory and stage visualisations |
