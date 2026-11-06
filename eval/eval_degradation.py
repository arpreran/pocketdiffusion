"""Quality-degradation study + planner calibration (on the *validation* split).

Sweeps: sampling steps, resolution, weight precision (FP16/INT8/INT4), magnitude-pruning
sparsity, text encoder (full vs our 6-layer distilled), decoder (VAE vs TAESD) and best-of-N.
Metric: CLIP-L/14 score (+ generation latency). Writes results/degradation.json and the
quality part of artifacts/planner_calibration.json.

Usage: python -m eval.eval_degradation [--n 60]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from eval.common import RESULTS_DIR, Runner, clip_scores, coco_calib, load_set, read_json, update_json
from pocketdiff.config import CALIBRATION_FILE, DISTILLED_TE_DIR, MODEL_IDS, PRESETS
from pocketdiff.optimizations import quantization_error
from pocketdiff.planner import family_available

LCM4 = PRESETS["sd15_lcm4_taesd"]
TURBO1 = PRESETS["turbo1_taesd"]
DPM20 = PRESETS["sd15_fp16_dpm20"].with_(vae="taesd", name="dpm20_taesd")


def sweeps() -> dict[str, list]:
    s: dict[str, list] = {"steps": [], "resolution": [], "quant": [], "prune": [], "text_encoder": [],
                          "decoder": [], "rerank": []}
    if family_available("sd15"):
        for st in (1, 2, 4, 8):
            s["steps"].append((f"lcm_steps{st}", LCM4.with_(steps=st, name=f"lcm_steps{st}"), {}))
        s["steps"].append(("dpm_steps20", DPM20, {}))
        for r in (256, 384):
            s["resolution"].append((f"lcm4_{r}px", LCM4.with_(height=r, width=r, name=f"lcm4_{r}"), {}))
        for q in ("int8", "int4"):
            s["quant"].append((f"lcm4_{q}", LCM4.with_(quant=q, name=f"lcm4_{q}"), {}))
        for p in (0.1, 0.2, 0.3, 0.5):
            s["prune"].append((f"lcm4_prune{int(p * 100)}", LCM4.with_(prune=p, name=f"lcm4_p{p}"), {}))
        if DISTILLED_TE_DIR.exists():
            s["text_encoder"].append(("lcm4_t6", LCM4.with_(text_encoder="t6", name="lcm4_t6"), {}))
        s["decoder"].append(("lcm4_fullvae", LCM4.with_(vae="full", name="lcm4_fullvae"), {}))
        for n in (2, 4):
            s["rerank"].append((f"lcm4_best_of_{n}", LCM4, {"n": n, "rerank": True}))
    if family_available("sd_turbo"):
        for st in (1, 2, 4):
            s["steps"].append((f"turbo_steps{st}", TURBO1.with_(steps=st, name=f"turbo_steps{st}"), {}))
        for r in (256, 384):
            s["resolution"].append((f"turbo1_{r}px", TURBO1.with_(height=r, width=r, name=f"turbo1_{r}"), {}))
        for q in ("int8", "int4"):
            s["quant"].append((f"turbo1_{q}", TURBO1.with_(quant=q, name=f"turbo1_{q}"), {}))
        for p in (0.1, 0.2, 0.3):
            s["prune"].append((f"turbo1_prune{int(p * 100)}", TURBO1.with_(prune=p, name=f"turbo1_p{p}"), {}))
        s["decoder"].append(("turbo1_fullvae", TURBO1.with_(vae="full", name="turbo1_fullvae"), {}))
        for n in (2, 4):
            s["rerank"].append((f"turbo1_best_of_{n}", TURBO1, {"n": n, "rerank": True}))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    rows = coco_calib(args.n)
    runner = Runner()
    plan = sweeps()
    names = []
    for group, items in plan.items():
        for name, cfg, kw in items:
            runner.run_set(name, rows, "calib", cfg=cfg, pipeline_kwargs=kw)
            names.append((group, name, cfg, kw))
            runner.pd.engine.models.evict_all()
    runner.run_set("lcm4", rows, "calib", cfg=LCM4) if family_available("sd15") else None
    runner.run_set("turbo1", rows, "calib", cfg=TURBO1) if family_available("sd_turbo") else None
    runner.pd.engine.models.evict_all()

    out = read_json(RESULTS_DIR / "degradation.json")
    allnames = [("base", "lcm4", LCM4, {}), ("base", "turbo1", TURBO1, {})] + names
    for group, name, cfg, kw in allnames:
        imgs, prompts, meta = load_set("calib", name, rows)
        if not imgs:
            continue
        cs = clip_scores(imgs, prompts)
        out[name] = {"group": group, "family": cfg.family, "steps": cfg.steps, "res": cfg.height,
                     "quant": cfg.quant, "prune": cfg.prune, "te": cfg.text_encoder, "vae": cfg.vae,
                     "n": kw.get("n", 1), "clip_mean": float(cs.mean()), "clip_sem": float(cs.std() / np.sqrt(len(cs))),
                     "gen_ms": float(np.mean([m["gen_ms"] for m in meta.values()])),
                     "peak_mem_mb": float(np.mean([m["peak_mem_mb"] for m in meta.values()])),
                     "weights_mb": float(np.mean([m["weights_mb"] for m in meta.values()]))}
        print(f"{name:22s} CLIP {cs.mean():.2f} ± {cs.std() / np.sqrt(len(cs)):.2f}  {out[name]['gen_ms']:.0f} ms", flush=True)
    (RESULTS_DIR / "degradation.json").write_text(__import__("json").dumps(out, indent=1), encoding="utf-8")

    # weight-space quantisation error on a real UNet layer (explains the quality numbers)
    try:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(MODEL_IDS["sd15"], subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float16)
        errs = {}
        for b in (8, 4):
            e = [quantization_error(m.weight.data, b) for n_, m in unet.named_modules()
                 if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) and m.weight.numel() >= 4096][:200]
            errs[f"int{b}"] = float(np.mean(e))
        update_json(RESULTS_DIR / "degradation.json", "_weight_rel_error", errs)
        del unet
    except Exception as e:
        print("quantisation-error analysis skipped:", e)

    # -------- quality calibration for the planner (validation split only)
    clip = {}
    for k, v in out.items():
        if k.startswith("_") or v.get("res") != 512 or v.get("quant") != "none" or v.get("prune", 0) > 0 \
                or v.get("te") != "full" or v.get("vae") != "taesd" or v.get("n", 1) != 1:
            continue
        fam = "sd_turbo|euler" if v["family"] == "sd_turbo" else ("sd15|dpm" if v["steps"] == 20 else "sd15|lcm")
        clip.setdefault(fam, {})[str(v["steps"])] = v["clip_mean"]
    cal = {"clip": clip}

    def delta(a, b):
        return out[a]["clip_mean"] - out[b]["clip_mean"] if a in out and b in out else None

    def avg(*xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    rd = {256: avg(delta("lcm4_256px", "lcm4"), delta("turbo1_256px", "turbo1")),
          384: avg(delta("lcm4_384px", "lcm4"), delta("turbo1_384px", "turbo1")), 512: 0.0}
    cal["res_delta"] = {str(k): v for k, v in rd.items() if v is not None}
    qd = {"none": 0.0, "int8": avg(delta("lcm4_int8", "lcm4"), delta("turbo1_int8", "turbo1")),
          "int4": avg(delta("lcm4_int4", "lcm4"), delta("turbo1_int4", "turbo1"))}
    cal["quant_delta"] = {k: v for k, v in qd.items() if v is not None}
    if delta("lcm4_t6", "lcm4") is not None:
        cal["te_delta"] = {"full": 0.0, "t6": delta("lcm4_t6", "lcm4")}
    vd = avg(delta("lcm4_fullvae", "lcm4"), delta("turbo1_fullvae", "turbo1"))
    if vd is not None:
        cal["vae_delta"] = {"full": vd, "taesd": 0.0}
    rg = {"1": 0.0}
    for n in (2, 4):
        g = avg(delta(f"lcm4_best_of_{n}", "lcm4"), delta(f"turbo1_best_of_{n}", "turbo1"))
        if g is not None:
            rg[str(n)] = max(g, 0.0)
    cal["rerank_gain"] = rg
    for k, v in cal.items():
        update_json(CALIBRATION_FILE, k, v)
    print("planner calibration ->", CALIBRATION_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
