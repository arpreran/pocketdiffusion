"""End-to-end smoke test + correctness check of our denoising loop against diffusers.

1. Our loop vs. the reference diffusers StableDiffusionPipeline (same weights, scheduler, seed)
   -> mean absolute pixel difference should be tiny (< 0.02).
2. Every preset that is available locally generates one image; timings are printed.
3. The full adaptive pipeline runs once with verification.

Usage: python scripts/smoke_test.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pocketdiff.config import MODEL_IDS, PRESETS, RESULTS_DIR  # noqa: E402
from pocketdiff.engine import DiffusionEngine, make_scheduler  # noqa: E402
from pocketdiff.planner import Budget, family_available  # noqa: E402

PROMPT = "a red apple to the left of a green bottle on a wooden table"


def save(img, name):
    from PIL import Image
    out = RESULTS_DIR / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out / f"{name}.png")


def reference_check(engine: DiffusionEngine) -> float:
    from diffusers import StableDiffusionPipeline
    cfg = PRESETS["sd15_fp16_dpm20"].with_(steps=8)
    ours = engine.generate(PROMPT, cfg, seed=3)
    b = engine.models.get(cfg)
    vae = engine.models.full_vae(b)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=b.text_encoder, tokenizer=b.tokenizer, unet=b.unet,
                                   scheduler=make_scheduler("dpm", b.scheduler_config), safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    gen_lat = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(3)).to("cuda", torch.float16)
    ref = pipe(PROMPT, num_inference_steps=8, guidance_scale=cfg.guidance, latents=gen_lat,
               output_type="np").images[0]
    diff = float(np.abs(ref - ours.images[0]).mean())
    save(ours.images[0], "ours_dpm8")
    save(ref, "diffusers_dpm8")
    return diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    engine = DiffusionEngine()
    if family_available("sd15") and torch.cuda.is_available():
        d = reference_check(engine)
        print(f"[check] our loop vs diffusers pipeline: mean |diff| = {d:.4f} ({'OK' if d < 0.02 else 'MISMATCH'})")
    names = ["sd15_lcm4_taesd", "sd15_lcm4", "turbo1_taesd", "sd15_lcm4_taesd_int8"] if args.quick else list(PRESETS)
    for n in names:
        cfg = PRESETS[n]
        if not family_available(cfg.family):
            print(f"[skip] {n}: weights not downloaded")
            continue
        try:
            engine.generate(PROMPT, cfg, seed=0)          # warm-up / load
            t0 = time.perf_counter()
            r = engine.generate(PROMPT, cfg, seed=1)
            wall = (time.perf_counter() - t0) * 1000
            save(r.images[0], n)
            print(f"[ok] {n:26s} {r.latency_ms:7.0f} ms (wall {wall:6.0f})  peak {r.peak_mem_mb:6.0f} MB  "
                  f"weights {r.weights_mb:6.0f} MB  {r.timings_ms}")
        except Exception as e:
            print(f"[FAIL] {n}: {type(e).__name__}: {e}")
        engine.models.evict_all()
    from pocketdiff.pipeline import PocketDiffusion
    pd = PocketDiffusion(engine=engine)
    for mode in ("fast", "auto", "quality"):
        r = pd.run(PROMPT, budget=Budget(latency_ms=2000, mode=mode), seed=5, verify=not args.quick)
        save(r.image, f"pocketdiff_{mode}")
        print(f"[ok] pocketdiff/{mode}: {r.latency_ms:.0f} ms, plan={r.config.name}, N={len(r.candidates)}, "
              f"events={r.events}, verify={None if r.verification is None else round(r.verification['f1'], 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
