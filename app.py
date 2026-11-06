"""PocketDiffusion - resource-efficient text-to-image generation (Gradio web app).

Run:   python app.py            (local, http://127.0.0.1:7860)
       python app.py --share    (also creates a public *.gradio.live URL for phones/judges)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

# Once the weights are cached, never touch the network while loading models (a slow or
# flaky connection must not slow down / break the live demo). Pass --online to disable.
_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_HUB / "models--stable-diffusion-v1-5--stable-diffusion-v1-5").exists() and "--online" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr  # noqa: E402
import numpy as np

from pocketdiff.config import DEVICE_PROFILES, PRESETS, RESULTS_DIR
from pocketdiff.planner import Budget
from pocketdiff.pipeline import PocketDiffusion, PromptRejected
from pocketdiff.utils import LOG, fmt_ms, to_uint8

PD: PocketDiffusion | None = None

EXAMPLES = [
    ["a red apple to the left of a green bottle on a wooden table, studio photo"],
    ["two cats sleeping on a blue couch next to a laptop"],
    ["an astronaut riding a horse on the moon, cinematic lighting"],
    ["a yellow bus parked in front of a red brick building, rainy evening"],
    ["three oranges in a white bowl, top view, soft light"],
    ["a cozy wooden cabin in a snowy forest at night, warm light in the windows, watercolor"],
]
SIZE_CHOICES = ["512x512", "768x768", "1024x1024", "768x512 (landscape)", "512x768 (portrait)",
                "384x384", "256x256"]
PROFILE_CHOICES = [(p.label, k) for k, p in DEVICE_PROFILES.items()]
OVERRIDE_CHOICES = ["Planner (adaptive)"] + [k for k in PRESETS if k != "cpu_fallback"]

CSS = """
#title {text-align:center}
.metric-card {font-size: 0.95rem}
footer {visibility: hidden}
"""


def _pd() -> PocketDiffusion:
    global PD
    if PD is None:
        PD = PocketDiffusion()
    return PD


def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.split(" ")[0].split("x")
    return int(w), int(h)


def _u8(img):
    return to_uint8(np.clip(img, 0, 1)) if img is not None else None


def _metrics_md(res, total_ms: float) -> str:
    g = res.gen
    cfg = res.config
    clip_txt = ""
    if res.candidates[res.chosen].clip is not None:
        clip_txt = f" | **CLIP (B/32 re-ranker):** {res.candidates[res.chosen].clip:.1f}"
    iqa_b = res.iqa_before["pocket_iqa"]
    iqa_a = res.iqa_after["pocket_iqa"]
    niqe_b, niqe_a = res.iqa_before.get("niqe"), res.iqa_after.get("niqe")
    niqe_txt = f" | **NIQE:** {niqe_b:.2f} -> {niqe_a:.2f}" if niqe_b is not None and niqe_a is not None else ""
    mem = f"{g.peak_mem_mb:.0f} MB peak GPU" if cfg.device == "cuda" else "CPU run"
    lines = [
        f"**End-to-end latency:** {fmt_ms(res.latency_ms)}  (wall {fmt_ms(total_ms)}) | "
        f"**Memory:** {mem} | **Weights:** {g.weights_mb:.0f} MB",
        f"**Model:** {cfg.family} · {cfg.scheduler} {g.steps_done}/{cfg.steps} steps · "
        f"{cfg.width}x{cfg.height} · {cfg.dtype}/{cfg.quant} · decoder={cfg.vae} · "
        f"text-encoder={cfg.text_encoder} · N={len(res.candidates)}{clip_txt}",
        f"**PocketIQA:** {iqa_b:.1f} -> {iqa_a:.1f}{niqe_txt} | "
        f"**Artifacts found:** {', '.join(res.artifacts_before['flags']) or 'none'}",
    ]
    if res.events:
        lines.append("**Events:** " + " · ".join(res.events))
    return "\n\n".join(lines)


def _plan_md(res) -> str:
    if res.plan is None:
        return "_Manual preset (planner bypassed)._"
    p = res.plan
    rows = "\n".join(f"- {r}" for r in p.reasons)
    alt = "| family | sched | steps | res | dec | quant | TE | N | pred. ms | pred. MB | pred. CLIP |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    for a in p.alternatives[:6]:
        lat = a["device_latency"] if a["device_latency"] is not None else a["latency"]
        alt += (f"| {a['family']} | {a['scheduler']} | {a['steps']} | {a['res']} | {a['vae']} | "
                f"{a['quant']} | {a['te']} | {a['n']} | {lat:.0f} | {a['memory']:.0f} | {a['clip']:.1f} |\n")
    return f"### Why this plan\n{rows}\n\n**Top alternatives considered**\n\n{alt}"


def _verification_md(v: dict | None) -> str:
    if not v:
        return "_Enable **Semantic check** to verify objects, counts, colours and spatial relations._"
    out = [f"**Object precision / recall / F1:** {v['precision']:.2f} / {v['recall']:.2f} / {v['f1']:.2f}"]
    for c in v["count_checks"]:
        out.append(f"- {'✅' if c['ok'] else '❌'} count of **{c['object']}**: expected {c['expected']}, found {c['found']}")
    for c in v["color_checks"]:
        out.append(f"- {'✅' if c['ok'] else '❌'} **{c['object']}** colour: expected {c['expected']}, measured {c['found']}")
    for c in v["relation_checks"]:
        out.append(f"- {'✅' if c['ok'] else '❌'} **{c['subject']}** {c['relation'].replace('_', ' ')} **{c['object']}**")
    out.append(f"\n**All constraints satisfied:** {'yes' if v['all_satisfied'] else 'no'}")
    return "\n".join(out)


def generate(prompt, device_key, mode, latency_ms, memory_mb, size, seed, random_seed, neg,
             post, rerank, verify, show_traj, override, semantic=False, progress=gr.Progress()):
    t_start = time.perf_counter()
    try:
        pd = _pd()
        if random_seed:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        budget = Budget(latency_ms=float(latency_ms), memory_mb=float(memory_mb), mode=mode,
                        device_profile=device_key, target_size=_parse_size(size))
        cfg = None
        if override and override != "Planner (adaptive)":
            cfg = PRESETS[override]
        progress(0.05, desc="Planning")

        def on_step(i, total):
            progress(0.1 + 0.7 * i / total, desc=f"Denoising step {i}/{total}")

        res = pd.run(prompt, budget=budget, seed=seed, negative_prompt=neg or "", config=cfg,
                     postprocess=None if post == "auto" else post, rerank=rerank, verify=verify,
                     preview=show_traj, on_step=on_step, semantic_rerank=bool(semantic),
                     upscale_to=_parse_size(size) if cfg is not None and _parse_size(size) != (cfg.width, cfg.height) else None)
        progress(0.95, desc="Rendering results")
        total = (time.perf_counter() - t_start) * 1000
        for e in res.events:
            gr.Info(e)
        cands = [(_u8(c.image), f"seed {c.seed}" + (f" · CLIP {c.clip:.1f} · IQA {c.iqa:.0f} · score {c.score:.1f}" if c.clip is not None else "")
                  + (" ✔ chosen" if i == res.chosen else "")) for i, c in enumerate(res.candidates)]
        traj = [(_u8(p), f"step {i + 1}: x0 estimate") for i, p in enumerate(res.gen.previews)]
        stages = [(_u8(res.raw_image), "raw generator output")] + [
            (_u8(s.image), f"{s.name} ({s.ms:.0f} ms) {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.params.items()})}")
            for s in res.post_stages if s.image is not None]
        iqa_rows = []
        for k in ("pocket_iqa", "niqe", "sharpness", "contrast", "colorfulness", "noise_sigma", "mean_luma", "clip_high", "clip_low"):
            b, a = res.iqa_before.get(k), res.iqa_after.get(k)
            if b is None:
                continue
            iqa_rows.append([k, round(float(b), 3), round(float(a), 3) if a is not None else None])
        telemetry = {"timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
                     "step_ms": [round(s, 1) for s in res.gen.step_ms],
                     "peak_gpu_mb": round(res.gen.peak_mem_mb, 1),
                     "resident_before_mb": round(res.gen.resident_mb, 1),
                     "config": res.config.to_dict(), "safety": res.safety, "events": res.events,
                     "loaded_models": pd.engine.models.loaded()}
        return (_u8(res.image), _metrics_md(res, total), _plan_md(res), cands, traj, stages,
                iqa_rows, _u8(res.heatmap), res.artifacts_before, _u8(res.verification_image),
                _verification_md(res.verification), res.analysis.to_dict(), telemetry, seed)
    except PromptRejected as e:
        raise gr.Error(str(e))
    except Exception as e:  # never show a stack trace to the user, but log it
        LOG.error("generation failed: %s\n%s", e, traceback.format_exc())
        raise gr.Error(f"Generation failed: {type(e).__name__}: {e}")


def compare(prompt, seed, progress=gr.Progress()):
    try:
        pd = _pd()
        progress(0.05, desc="Optimised pipeline")
        opt = pd.run(prompt, budget=Budget(latency_ms=1500, mode="auto"), seed=int(seed))
        progress(0.35, desc="Baseline: SD1.5, 50 PNDM steps, CFG 7.5 (slow)")
        base_cfg = PRESETS["sd15_fp16"]
        base = pd.run(prompt, config=base_cfg, seed=int(seed), n=1, postprocess="off", rerank=False,
                      on_step=lambda i, t: progress(0.35 + 0.6 * i / t, desc=f"Baseline step {i}/{t}"))
        scorer = pd.scorer
        rows = []
        for name, r in (("Baseline SD1.5 (FP16, 50 steps, CFG 7.5)", base), ("PocketDiffusion (adaptive)", opt)):
            clip = float(scorer.clip_score([r.image], prompt)[0]) if scorer else float("nan")
            rows.append([name, round(r.latency_ms), round(r.gen.peak_mem_mb), round(r.gen.weights_mb),
                         r.config.unet_evals_per_image * len(r.candidates), round(clip, 2),
                         round(r.iqa_after["pocket_iqa"], 1)])
        speedup = base.latency_ms / max(opt.latency_ms, 1)
        md = (f"**Speed-up: {speedup:.1f}x** · memory {rows[0][2]} -> {rows[1][2]} MB · "
              f"UNet evaluations {rows[0][4]} -> {rows[1][4]}. Note: the official baseline in the "
              f"report is FP32; FP16 is used here to keep the live demo responsive.")
        # free the baseline weights so the optimised models stay resident
        return _u8(base.image), _u8(opt.image), rows, md
    except PromptRejected as e:
        raise gr.Error(str(e))
    except Exception as e:
        LOG.error("compare failed: %s\n%s", e, traceback.format_exc())
        raise gr.Error(f"Comparison failed: {type(e).__name__}: {e}")


def _benchmarks_md() -> str:
    p = RESULTS_DIR / "summary.md"
    if p.exists():
        # plots are shown in the gallery below; markdown image links / failure thumbnails would 404
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if not l.startswith("![")]
        return "\n".join(l.replace("![](results/failures/", "(see results/failures/") for l in lines)
    return "_No benchmark results yet. Run `python -m eval.run_all` to produce them._"


def _benchmark_plots() -> list:
    plots = sorted((RESULTS_DIR / "plots").glob("*.png")) if (RESULTS_DIR / "plots").exists() else []
    return [(str(p), p.stem.replace("_", " ")) for p in plots]


def _style_kwargs(fn) -> dict:
    """Gradio 5 takes theme/css in Blocks(), Gradio 6 in launch(): pass them where accepted."""
    import inspect
    params = inspect.signature(fn).parameters
    kw = {}
    if "theme" in params:
        kw["theme"] = gr.themes.Soft()
    if "css" in params:
        kw["css"] = CSS
    return kw


def build_ui() -> gr.Blocks:
    import inspect
    style_in_launch = "css" in inspect.signature(gr.Blocks.launch).parameters
    with gr.Blocks(title="PocketDiffusion", **({} if style_in_launch else _style_kwargs(gr.Blocks.__init__))) as demo:
        gr.Markdown("# 📱 PocketDiffusion\n**Resource-efficient text-to-image generation for mobile "
                    "devices** — adaptive planner · distilled few-step diffusion · INT8/INT4 weights · "
                    "tiny decoder · best-of-N re-ranking · classical artifact correction", elem_id="title")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(label="Prompt", lines=3, max_lines=6,
                                        placeholder="Describe the image (objects, colours, positions, style)...")
                    with gr.Row():
                        device = gr.Dropdown(PROFILE_CHOICES, value="this_gpu", label="Target device profile")
                        mode = gr.Radio(["auto", "fast", "balanced", "quality"], value="auto", label="Mode")
                    with gr.Row():
                        latency = gr.Slider(100, 20000, value=1500, step=50, label="Latency budget (ms)")
                        memory = gr.Slider(500, 8000, value=6500, step=50, label="Memory budget (MB)")
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            size = gr.Dropdown(SIZE_CHOICES, value="512x512", label="Output size")
                            seed = gr.Number(value=42, precision=0, label="Seed")
                            random_seed = gr.Checkbox(value=False, label="Random seed")
                        neg = gr.Textbox(label="Negative prompt (only used by CFG models)", value="")
                        with gr.Row():
                            post = gr.Radio(["auto", "off", "light", "full"], value="auto", label="Classical post-processing")
                            override = gr.Dropdown(OVERRIDE_CHOICES, value="Planner (adaptive)", label="Model / preset")
                        with gr.Row():
                            rerank = gr.Checkbox(value=True, label="Best-of-N CLIP re-ranking")
                            verify = gr.Checkbox(value=True, label="Semantic check (OWLv2)")
                            semantic = gr.Checkbox(value=False, label="Semantic re-ranking (detector picks the candidate that best matches the prompt)")
                            traj = gr.Checkbox(value=True, label="Show denoising trajectory")
                    with gr.Row():
                        btn = gr.Button("Generate", variant="primary", scale=3)
                    gr.Examples(EXAMPLES, inputs=[prompt])
                with gr.Column(scale=5):
                    out_img = gr.Image(label="Output", type="numpy", height=512)
                    metrics = gr.Markdown(elem_classes=["metric-card"])
            plan_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Candidates (best-of-N)"):
                    cand_gal = gr.Gallery(columns=4, height=300)
                with gr.Tab("Denoising trajectory"):
                    traj_gal = gr.Gallery(columns=8, height=240)
                with gr.Tab("Classical post-processing"):
                    stage_gal = gr.Gallery(columns=4, height=300)
                    iqa_tbl = gr.Dataframe(headers=["metric", "before", "after"], interactive=False)
                with gr.Tab("Artifact analysis"):
                    with gr.Row():
                        heat = gr.Image(label="Artifact heat-map (red=clipped, green=HF noise, blue=crushed)", height=320)
                        art_json = gr.JSON(label="Detector output")
                with gr.Tab("Semantic check"):
                    with gr.Row():
                        ver_img = gr.Image(label="Detections (green=requested, red=distractor)", height=320)
                        ver_md = gr.Markdown()
                with gr.Tab("Prompt analysis"):
                    ana_json = gr.JSON()
                with gr.Tab("Telemetry"):
                    tele_json = gr.JSON()
            seed_out = seed
            btn.click(generate,
                      inputs=[prompt, device, mode, latency, memory, size, seed, random_seed, neg,
                              post, rerank, verify, traj, override, semantic],
                      outputs=[out_img, metrics, plan_md, cand_gal, traj_gal, stage_gal, iqa_tbl,
                               heat, art_json, ver_img, ver_md, ana_json, tele_json, seed_out])
            prompt.submit(generate,
                          inputs=[prompt, device, mode, latency, memory, size, seed, random_seed, neg,
                                  post, rerank, verify, traj, override, semantic],
                          outputs=[out_img, metrics, plan_md, cand_gal, traj_gal, stage_gal, iqa_tbl,
                                   heat, art_json, ver_img, ver_md, ana_json, tele_json, seed_out])

        with gr.Tab("Baseline vs optimised"):
            gr.Markdown("Same prompt and seed through the **baseline** (SD1.5, 50 steps, CFG) and through "
                        "**PocketDiffusion**. The baseline takes several seconds - that is the point.")
            with gr.Row():
                c_prompt = gr.Textbox(label="Prompt", value=EXAMPLES[0][0], scale=4)
                c_seed = gr.Number(value=7, precision=0, label="Seed", scale=1)
            c_btn = gr.Button("Run comparison", variant="primary")
            with gr.Row():
                c_base = gr.Image(label="Baseline", height=400)
                c_opt = gr.Image(label="PocketDiffusion", height=400)
            c_tbl = gr.Dataframe(headers=["system", "latency ms", "peak MB", "weights MB",
                                          "UNet evals", "CLIP", "PocketIQA"], interactive=False)
            c_md = gr.Markdown()
            c_btn.click(compare, inputs=[c_prompt, c_seed], outputs=[c_base, c_opt, c_tbl, c_md])

        with gr.Tab("Benchmarks"):
            bench_md = gr.Markdown(_benchmarks_md())
            bench_gal = gr.Gallery(value=_benchmark_plots(), columns=2, height=600)
            refresh = gr.Button("Refresh")
            refresh.click(lambda: (_benchmarks_md(), _benchmark_plots()), outputs=[bench_md, bench_gal])

        with gr.Tab("How it works"):
            arch = Path(__file__).parent / "docs" / "architecture.md"
            gr.Markdown(arch.read_text(encoding="utf-8") if arch.exists() else "See README.md")
    return demo


def warmup():
    pd = _pd()
    t0 = time.perf_counter()
    from pocketdiff.planner import family_available
    for name in ("sd15_lcm4_taesd", "turbo1_taesd"):
        if family_available(PRESETS[name].family):
            try:
                ms = pd.engine.warmup_shapes(PRESETS[name])
                LOG.info("warm-up of %s shapes: %.0f ms", name, ms)
            except Exception as e:
                LOG.warning("shape warm-up failed for %s: %s", name, e)
    for mode in ("fast", "quality"):
        try:
            pd.run("a photo of a cat", budget=Budget(latency_ms=3000, mode=mode), seed=0,
                   verify=False, safety=True)
        except Exception as e:  # warm-up must never block the app
            LOG.warning("warm-up (%s) failed: %s", mode, e)
    try:
        if pd.verifier is not None:
            pd.verifier.detect(np.zeros((64, 64, 3), np.float32), ["cat"])
    except Exception as e:
        LOG.warning("detector warm-up failed: %s", e)
    LOG.info("warm-up done in %.1f s", time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face network access")
    args = ap.parse_args()
    demo = build_ui()
    if not args.no_warmup:
        warmup()
    demo.queue(default_concurrency_limit=1, max_size=20)
    import inspect
    launch_style = _style_kwargs(demo.launch) if "css" in inspect.signature(demo.launch).parameters else {}
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True,
                allowed_paths=[str(RESULTS_DIR)], **launch_style)


if __name__ == "__main__":
    main()
