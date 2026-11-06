"""Ablation study of the PocketDiffusion system components.

A) System ablation (test split, N prompts): full system vs. removing one component at a time
   (best-of-N re-ranking, classical post-processing, prompt-adaptive planning, the planner
   itself). Metrics: CLIP-L, PocketIQA, NIQE, artifact-flag rate, latency.
B) Post-processing stage leave-one-out on raw generator outputs (no regeneration needed).
C) Resolution optimisation: generate at 384 px and upscale to 512 with each upscaler vs.
   native 512 generation (CLIP, NIQE, sharpness, latency).

Usage: python -m eval.eval_ablation [--n 100]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from eval.common import RESULTS_DIR, Runner, clip_scores, coco_test, load_set, update_json
from pocketdiff.classical import AdaptivePostProcessor, detect_artifacts, quality_report
from pocketdiff.classical.resize import resize
from pocketdiff.config import PRESETS
from pocketdiff.planner import Budget, family_available


def iqa_stats(imgs) -> dict:
    reps = [quality_report(im) for im in imgs]
    arts = [detect_artifacts(im) for im in imgs]
    niqes = [r["niqe"] for r in reps if r["niqe"] is not None]
    return {"pocket_iqa": float(np.mean([r["pocket_iqa"] for r in reps])),
            "niqe": float(np.mean(niqes)) if niqes else None,
            "artifact_flag_rate": float(np.mean([len(a.flags) > 0 for a in arts])),
            "artifact_severity": float(np.mean([a.total for a in arts]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    rows = coco_test(args.n)
    runner = Runner()
    B = Budget(latency_ms=1500, mode="auto")
    fixed = PRESETS["turbo1_taesd"] if family_available("sd_turbo") else PRESETS["sd15_lcm4_taesd"]
    systems = {
        "full_system": dict(budget=B),
        "minus_rerank": dict(budget=B, rerank=False, n=1),
        "minus_postprocess": dict(budget=B, postprocess="off"),
        "minus_complexity_adaptation": dict(budget=Budget(latency_ms=1500, mode="balanced")),
        "minus_planner_fixed_config": dict(config=fixed, n=1, postprocess="full", rerank=False),
    }
    for name, kw in systems.items():
        kw = dict(kw, verify=False, safety=False)
        if "config" in kw:
            cfg = kw.pop("config")
            runner.run_set(name, rows, "ablation", cfg=cfg, pipeline_kwargs=kw)
        else:
            runner.run_set(name, rows, "ablation", pipeline_kwargs=kw)
    # raw outputs of the fixed config (for B) and 384-px generations (for C)
    runner.run_set("raw_fixed", rows, "ablation", cfg=fixed)
    lowres = fixed.with_(height=384, width=384, name=fixed.name + "_384")
    runner.run_set("raw_fixed_384", rows, "ablation", cfg=lowres)
    runner.pd.engine.models.evict_all()

    # ---------------- A
    out = {}
    for name in systems:
        imgs, prompts, meta = load_set("ablation", name, rows)
        cs = clip_scores(imgs, prompts)
        out[name] = {"clip_mean": float(cs.mean()), "clip_sem": float(cs.std() / np.sqrt(len(cs))),
                     **iqa_stats(imgs),
                     "e2e_latency_ms": float(np.mean([m["latency_ms"] for m in meta.values()])),
                     "gen_ms": float(np.mean([m["gen_ms"] for m in meta.values()]))}
        print(f"A {name:30s} CLIP {out[name]['clip_mean']:.2f}  IQA {out[name]['pocket_iqa']:.1f}  "
              f"NIQE {out[name]['niqe']}  lat {out[name]['e2e_latency_ms']:.0f} ms", flush=True)
    update_json(RESULTS_DIR / "ablation.json", "system", out)

    # ---------------- B: post-processing stages, leave-one-out
    raw, prompts, _ = load_set("ablation", "raw_fixed", rows)
    stages = ["grid_notch", "denoise", "white_balance", "adaptive_gamma", "clahe", "saturation_knee", "sharpen"]
    variants = {"none": None, "all": ()}
    variants.update({f"all_minus_{s}": (s,) for s in stages})
    outb = {}
    for vname, disabled in variants.items():
        t0 = time.perf_counter()
        if disabled is None:
            imgs = raw
        else:
            pp = AdaptivePostProcessor("full", keep_intermediates=False, disabled=disabled)
            imgs = [pp(im).image for im in raw]
        ms = (time.perf_counter() - t0) * 1000 / len(raw)
        cs = clip_scores(imgs, prompts)
        outb[vname] = {"clip_mean": float(cs.mean()), **iqa_stats(imgs), "post_ms_per_image": ms}
        print(f"B {vname:28s} CLIP {cs.mean():.2f}  IQA {outb[vname]['pocket_iqa']:.1f}  "
              f"NIQE {outb[vname]['niqe']}  flags {outb[vname]['artifact_flag_rate']:.2f}", flush=True)
    # how often each stage fires on real generator outputs
    fire = {s: 0 for s in stages}
    pp = AdaptivePostProcessor("full", keep_intermediates=False)
    for im in raw:
        for st in pp(im).stages:
            if st.name in fire:
                fire[st.name] += 1
    outb["_stage_trigger_rate"] = {k: v / len(raw) for k, v in fire.items()}
    update_json(RESULTS_DIR / "ablation.json", "postprocess", outb)

    # ---------------- C: resolution optimisation
    low, prompts_l, meta_l = load_set("ablation", "raw_fixed_384", rows)
    _, _, meta_h = load_set("ablation", "raw_fixed", rows)
    outc = {"native_512": {"clip_mean": float(clip_scores(raw, prompts).mean()), **iqa_stats(raw),
                           "gen_ms": float(np.mean([m["gen_ms"] for m in meta_h.values()]))}}
    for m in ("nearest", "bilinear", "bicubic", "lanczos", "ibp"):
        t0 = time.perf_counter()
        up = [resize(im, (512, 512), m) for im in low]
        ms = (time.perf_counter() - t0) * 1000 / len(low)
        outc[f"384_to_512_{m}"] = {"clip_mean": float(clip_scores(up, prompts_l).mean()), **iqa_stats(up),
                                   "gen_ms": float(np.mean([x["gen_ms"] for x in meta_l.values()])),
                                   "upscale_ms": ms}
        print(f"C 384->512 {m:9s} CLIP {outc[f'384_to_512_{m}']['clip_mean']:.2f}  "
              f"NIQE {outc[f'384_to_512_{m}']['niqe']}", flush=True)
    update_json(RESULTS_DIR / "ablation.json", "resolution", outc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
