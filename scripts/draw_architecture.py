"""Render the system architecture diagram to docs/architecture.png (for the report / slides).

Usage: python scripts/draw_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

COL = {"stack": "#fde2e4", "in": "#eef2f7", "cls": "#dff3e4", "dl": "#e3ecfb", "plan": "#fdf0d5", "out": "#f3e8fd"}
BOXES = {
    # key: (x, y, w, h, colour, title, body)
    "fe": (0.2, 6.2, 2.9, 1.2, "stack", "Web frontend (SPA)", "studio · gallery · lab · race\nlive x0 frames over SSE\nno framework, no build step"),
    "api": (3.5, 6.2, 3.1, 1.2, "stack", "FastAPI backend", "REST: plan · analyze · generate\ncompare · history · stats\nvalidation + error handling"),
    "wk": (7.0, 6.2, 2.6, 1.2, "stack", "GPU worker + job queue", "one job at a time\nevent log -> Server-Sent Events\nwarm-up at start"),
    "db": (10.0, 6.2, 2.3, 1.2, "stack", "SQLite (WAL)", "generations · candidates\nstages · frames\nimages on disk"),
    "in": (0.2, 4.2, 2.2, 1.2, "in", "Input", "prompt\ndevice profile\nlatency / memory budget"),
    "pa": (2.9, 4.2, 2.6, 1.2, "cls", "Prompt analyser", "scene graph: objects, counts,\ncolours, relations\ncomplexity · block-list"),
    "pl": (6.0, 4.2, 3.0, 1.2, "plan", "Adaptive planner", "~700 configs -> predict\nlatency · memory · CLIP\nmax utility s.t. budgets"),
    "fb": (9.5, 4.2, 2.8, 1.2, "plan", "Fallback chain", "N down -> 384px -> TAESD\n-> INT8 -> 256px -> CPU"),
    "te": (0.2, 2.2, 2.6, 1.4, "dl", "Text encoder", "CLIP-L 12 layers\nor our distilled 6 layers\n+ embedding cache"),
    "un": (3.2, 2.2, 3.2, 1.4, "dl", "Distilled UNet", "SD-Turbo 1 step | SD1.5+LCM-LoRA\nFP16 · INT8/INT4 weights\nSDPA attention"),
    "lp": (6.8, 2.2, 2.6, 1.4, "dl", "Own denoising loop", "anytime x0 estimate\ndeadline early-exit\nper-step timing, previews"),
    "de": (9.8, 2.2, 2.5, 1.4, "dl", "Decoder", "TAESD (2.4M)\nor VAE (83.7M)"),
    "sc": (0.2, 0.2, 2.6, 1.4, "cls", "Artifact screening", "blank/NaN · exposure · cast\nnoise · 8-px grid (FFT)\nsaturation · blur"),
    "rr": (3.2, 0.2, 2.6, 1.4, "dl", "Best-of-N re-rank", "CLIP ViT-B/32\n+ PocketIQA\n- artifact severity"),
    "pp": (6.2, 0.2, 3.2, 1.4, "cls", "Adaptive post-processing", "FFT notch · guided denoise · WB\ngamma · CLAHE · sat. knee\nIBP super-res · edge-aware sharpen"),
    "ve": (9.8, 0.2, 2.5, 1.4, "out", "Verify & report", "OWLv2 + HSV + geometry\nobject F1 · NIQE\nsafety · telemetry"),
}
ARROWS = [("fe", "api"), ("api", "wk"), ("wk", "db"), ("wk", "in"), ("in", "pa"), ("pa", "pl"), ("pl", "fb"),
          ("pl", "te"), ("te", "un"), ("un", "lp"), ("lp", "un"), ("lp", "de"), ("de", "sc"), ("sc", "rr"),
          ("rr", "pp"), ("pp", "ve")]


def anchor(b, side):
    x, y, w, h = BOXES[b][:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def main():
    fig, ax = plt.subplots(figsize=(13, 7.9))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.95)
    ax.axis("off")
    for k, (x, y, w, h, c, title, body) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=COL[c], ec="#555", lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + (h - 0.35) / 2, body, ha="center", va="center", fontsize=7.6)
    for a, b in ARROWS:
        xa, ya, *_ = BOXES[a]
        xb, yb, *_ = BOXES[b]
        if abs(ya - yb) < 0.1:
            p, q = (anchor(a, "r"), anchor(b, "l")) if xa < xb else (anchor(a, "l"), anchor(b, "r"))
            if (a, b) == ("lp", "un"):
                p, q = (p[0], p[1] - 0.3), (q[0], q[1] - 0.3)
        else:
            p, q = anchor(a, "b"), anchor(b, "t")
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333",
                                     connectionstyle="arc3,rad=0.0"))
    legend = [("full stack", "stack"), ("classical CV", "cls"), ("deep learning", "dl"),
              ("planning / resource-aware", "plan"), ("evaluation", "out")]
    for i, (t, c) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((0.2 + i * 2.5, 7.62), 0.25, 0.18, boxstyle="round,pad=0.01", fc=COL[c], ec="#555"))
        ax.text(0.55 + i * 2.5, 7.71, t, va="center", fontsize=8)
    ax.set_title("PocketDiffusion – system architecture", fontsize=12, weight="bold", pad=2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
