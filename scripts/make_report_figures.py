"""Figures for the solution report (section 3.2 "Visual outputs & intermediate representations").

fig1_side_by_side.png   same prompt, same seed: baseline vs optimised systems (first 3 test prompts,
                        not cherry-picked), with measured latency per system
fig2_compositional.png  complex prompts (first prompt of each hard category): baseline SD1.5 vs
                        PocketDiffusion, with the OWLv2 evaluation boxes and the constraint verdict
fig3_intermediate.png   intermediate representations of one run: x0 estimate after every denoising
                        step, artifact heat-map, guided-filter base / detail layers, edge mask,
                        and the output of every classical post-processing stage that fired

Uses the images already produced by eval/ (figs 1-2) and one live generation (fig 3, GPU).
Usage: python scripts/make_report_figures.py [--skip-gpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.common import GEN_DIR, RESULTS_DIR, coco_test, compositional, load_png  # noqa: E402

OUT = RESULTS_DIR / "report"


def _latency(system: str) -> str:
    eff = json.loads((RESULTS_DIR / "efficiency.json").read_text())
    ms = eff.get(system, {}).get("latency_ms_mean")
    if ms is None:   # adaptive systems: mean end-to-end latency recorded while generating the test set
        meta = json.loads((GEN_DIR / "coco_test" / system / "meta.json").read_text())
        vals = [m.get("latency_ms") for m in meta.values() if isinstance(m, dict) and m.get("latency_ms")]
        ms = float(np.mean(vals)) if vals else None
    return f"{ms / 1000:.1f} s" if ms and ms >= 1000 else (f"{ms:.0f} ms" if ms else "")


def fig_side_by_side() -> None:
    systems = [("baseline", "Baseline SD1.5 FP32\n50 steps + CFG"),
               ("sd15_lcm4_taesd", "SD1.5 + LCM-LoRA 4 steps\n+ TAESD"),
               ("turbo1_taesd", "SD-Turbo 1 step\n+ TAESD"),
               ("pocketdiff", "PocketDiffusion\n(adaptive, auto)")]
    rows = coco_test()[:3]
    fig, axes = plt.subplots(len(rows), len(systems), figsize=(3.2 * len(systems), 3.5 * len(rows)))
    for i, r in enumerate(rows):
        for j, (s, label) in enumerate(systems):
            ax = axes[i, j]
            ax.imshow(load_png(GEN_DIR / "coco_test" / s / f"{r['id']}.png"))
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                lat = _latency(s)
                ax.set_title(label + (f"\n{lat}" if lat else ""), fontsize=10)
        axes[i, 0].set_ylabel("\n".join(_wrap(r["prompt"], 34)), fontsize=8.5)
    fig.suptitle("Same prompt, same seed (first three MS-COCO test prompts, not cherry-picked)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_side_by_side.png", dpi=110)
    plt.close(fig)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def fig_compositional() -> None:
    from eval.eval_compositional import gt_analysis
    from pocketdiff.verifier import CompositionVerifier

    items = compositional()
    picked = []
    for cat in ("counting", "color_binding", "spatial", "complex"):
        picked += [it for it in items if it["category"] == cat][:1]
    systems = [("sd15_fp16", "Baseline SD1.5 FP16, 50 steps"),
               ("pocketdiff", "PocketDiffusion (auto)"),
               ("pocketdiff_semantic8", "PocketDiffusion + semantic\nre-ranking, N=8")]
    ver = CompositionVerifier()
    fig, axes = plt.subplots(len(picked), len(systems), figsize=(3.6 * len(systems), 3.9 * len(picked)))
    for i, it in enumerate(picked):
        ga = gt_analysis(it)
        wanted = {o.name for o in ga.objects}
        for j, (s, label) in enumerate(systems):
            img = load_png(GEN_DIR / "compositional" / s / f"{it['id']}_s0.png")
            res = ver.verify(img, ga, seed=0)
            ax = axes[i, j]
            ax.imshow(ver.draw(img, res, wanted))
            ax.set_xticks([]), ax.set_yticks([])
            verdict = "all constraints OK" if res.all_satisfied else "constraint violated"
            ax.set_xlabel(f"F1 {res.f1:.2f} - {verdict}", fontsize=9,
                          color="#1a7f37" if res.all_satisfied else "#b42318")
            if i == 0:
                ax.set_title(label, fontsize=10)
        axes[i, 0].set_ylabel(f"[{it['category']}]\n" + "\n".join(_wrap(it["prompt"], 30)), fontsize=8.5)
    fig.suptitle("Complex prompts, seed 0 (first prompt of each category). Boxes: OWLv2 evaluation detector",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_compositional.png", dpi=110)
    plt.close(fig)


def fig_intermediate() -> None:
    from pocketdiff.classical.enhance import edge_aware_sharpen
    from pocketdiff.classical.filters import guided_filter_gray, luminance
    from pocketdiff.config import PRESETS
    from pocketdiff.pipeline import PocketDiffusion

    prompt = coco_test()[0]["prompt"]
    frames: list[tuple[int, int, np.ndarray]] = []
    pd = PocketDiffusion()
    res = pd.run(prompt, config=PRESETS["sd15_lcm4_taesd"], seed=0, n=1, postprocess="full",
                 safety=False, on_preview=lambda i, n, img: frames.append((i, n, img)))
    raw = res.raw_image
    gray = luminance(raw)
    base = guided_filter_gray(gray, raw, 2, 1e-3)
    detail = np.clip(0.5 + 4.0 * (raw - base), 0, 1)
    _, edge_w = edge_aware_sharpen(raw)
    stages = [s for s in res.post_stages if getattr(s, "image", None) is not None]

    top = [(f"x0 estimate, step {i}/{n}", im) for i, n, im in frames]
    mid = [("raw decoder output", raw), ("artifact heat-map\nred=clipped, green=HF residual", res.heatmap),
           ("guided-filter base layer", base), ("detail layer (x4)", detail),
           ("edge mask (sharpen weight)", edge_w)]
    bot = [(f"after '{s.name}'\n{s.ms:.0f} ms", s.image) for s in stages] + [("final output", res.image)]
    grid = [top, mid, bot]
    ncol = max(len(r) for r in grid)
    fig, axes = plt.subplots(3, ncol, figsize=(2.8 * ncol, 3.2 * 3))
    for i, row in enumerate(grid):
        for j in range(ncol):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(row):
                title, im = row[j]
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
                ax.set_title(title, fontsize=8.5)
    fig.suptitle(f"Intermediate representations - \"{prompt}\" (SD1.5 + LCM-LoRA 4 steps + TAESD, seed 0)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_intermediate.png", dpi=110)
    plt.close(fig)
    pd.engine.models.evict_all()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true", help="only the figures that need no generation")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_side_by_side()
    print("fig1 done", flush=True)
    fig_compositional()
    print("fig2 done", flush=True)
    if not args.skip_gpu:
        fig_intermediate()
        print("fig3 done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
