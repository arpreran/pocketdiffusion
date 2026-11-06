"""Quality on unseen prompts: CLIP score (ViT-L/14) and FID / KID against real COCO images.

Test split = Karpathy *test* captions/images (never used for calibration or training).
Every system gets the same prompts and the same per-prompt seed (paired comparison).

Usage: python -m eval.eval_quality [--n 200] [--systems baseline sd15_lcm4_taesd ... pocketdiff]
"""
from __future__ import annotations

import argparse

import numpy as np

from eval.common import DATA_DIR, RESULTS_DIR, Runner, clip_scores, coco_test, fid_kid, latency_stats, load_set, update_json
from pocketdiff.config import COMPARISON_ORDER, PRESETS
from pocketdiff.planner import Budget, family_available

DEFAULT_SYSTEMS = COMPARISON_ORDER + ["pocketdiff", "pocketdiff_fast"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS)
    ap.add_argument("--skip-fid", action="store_true")
    args = ap.parse_args()
    rows = coco_test(args.n)
    runner = Runner()
    systems = [s for s in args.systems if s.startswith("pocketdiff") or family_available(PRESETS[s].family)]
    for s in systems:
        try:
            if s in ("pocketdiff", "pocketdiff_fast"):
                runner.run_set(s, rows, "coco_test", pipeline_kwargs=dict(
                    budget=Budget(latency_ms=1500, mode="auto" if s == "pocketdiff" else "fast"),
                    verify=False, safety=False))
            else:
                runner.run_set(s, rows, "coco_test")
        except FileNotFoundError as e:
            print(f"skip {s}: {e}")
    runner.pd.engine.models.evict_all()   # free VRAM for the evaluation networks

    for s in systems:
        imgs, prompts, meta = load_set("coco_test", s, rows)
        if not imgs:
            continue
        cs = clip_scores(imgs, prompts)
        res = {"n": len(imgs), "clip_mean": float(cs.mean()), "clip_std": float(cs.std()),
               "clip_sem": float(cs.std() / np.sqrt(len(cs))),
               "gen_latency_ms": latency_stats(meta, "gen_ms"),
               "e2e_latency_ms": latency_stats(meta, "latency_ms"),
               "peak_mem_mb": float(np.mean([m["peak_mem_mb"] for m in meta.values()])),
               "weights_mb": float(np.mean([m["weights_mb"] for m in meta.values()]))}
        if s.startswith("pocketdiff"):
            plans = [m.get("plan") or {} for m in meta.values()]
            fams = {}
            for p in plans:
                k = f"{p.get('family')}/{p.get('steps')}st/{p.get('res')}px/N={p.get('n')}"
                fams[k] = fams.get(k, 0) + 1
            res["plan_histogram"] = fams
        if not args.skip_fid:
            res.update(fid_kid(imgs, DATA_DIR / "coco_test" / "real", ids=[r["id"] for r in rows]))
        update_json(RESULTS_DIR / "quality.json", s, res)
        print(f"{s:26s} CLIP {res['clip_mean']:.2f}  FID {res.get('fid', float('nan')):.1f}  "
              f"KIDx1e3 {res.get('kid_x1000', float('nan')):.2f}  gen {res['gen_latency_ms']['mean']:.0f} ms",
              flush=True)
    # real-image reference CLIP score (upper reference for alignment)
    from eval.common import load_png
    real = [load_png(DATA_DIR / "coco_test" / "real" / f"{r['id']}.jpg") for r in rows]
    cs = clip_scores(real, [r["prompt"] for r in rows])
    update_json(RESULTS_DIR / "quality.json", "_real_images", {"n": len(real), "clip_mean": float(cs.mean()),
                                                               "clip_std": float(cs.std())})
    print(f"{'real COCO images':26s} CLIP {cs.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
