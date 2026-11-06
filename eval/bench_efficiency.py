"""Efficiency benchmark: latency, peak memory, model size, parameters and FLOPs per configuration.

Each configuration is measured in isolation (all other models evicted first) with 2 warm-up
runs and R timed runs at 512x512, batch 1. Results -> results/efficiency.json and the cost
part of artifacts/planner_calibration.json.

Usage: python -m eval.bench_efficiency [--runs 5] [--cpu]
"""
from __future__ import annotations

import argparse
import gc
import platform
import time

import numpy as np
import torch

from eval.common import RESULTS_DIR, update_json, read_json
from pocketdiff.config import CALIBRATION_FILE, COMPARISON_ORDER, PRESETS
from pocketdiff.engine import DiffusionEngine
from pocketdiff.planner import family_available
from pocketdiff.utils import PeakRSSMonitor, count_params, module_bytes

PROMPT = "a red apple to the left of a green bottle on a wooden table, studio photo"
EXTRA = {
    "sd15_lcm4_taesd_int4": PRESETS["sd15_lcm4_taesd"].with_(name="sd15_lcm4_taesd_int4", quant="int4",
                                                              description="+ INT4 weights"),
    "turbo1_taesd_int4": PRESETS["turbo1_taesd"].with_(name="turbo1_taesd_int4", quant="int4",
                                                        description="SD-Turbo + INT4 weights"),
}


def bench_one(engine: DiffusionEngine, cfg, runs: int) -> dict:
    engine.models.evict_all()
    gc.collect()
    t0 = time.perf_counter()
    bundle = engine.models.get(cfg)
    load_s = time.perf_counter() - t0
    for s in range(2):
        engine.generate(PROMPT, cfg, seed=100 + s)
    lat, text, den, dec, peak, steps = [], [], [], [], [], []
    for s in range(runs):
        engine._emb_cache.clear()  # measure the real text-encoding cost
        r = engine.generate(PROMPT, cfg, seed=s)
        lat.append(r.latency_ms)
        text.append(r.timings_ms["text_encode"])
        den.append(r.timings_ms["denoise"])
        dec.append(r.timings_ms["decode"])
        peak.append(r.peak_mem_mb)
        steps.extend(r.step_ms)
    dev = bundle.device
    decoder = engine.models.taesd(dev, bundle.dtype) if cfg.vae == "taesd" else engine.models.full_vae(bundle)
    flops = engine.count_flops(cfg)
    out = {
        "description": cfg.description,
        "config": cfg.to_dict(),
        "latency_ms_mean": float(np.mean(lat)), "latency_ms_std": float(np.std(lat)),
        "latency_ms_median": float(np.median(lat)),
        "text_ms": float(np.mean(text)), "denoise_ms": float(np.mean(den)), "decode_ms": float(np.mean(dec)),
        "unet_step_ms": float(np.median(steps)),
        "peak_mem_mb": float(np.max(peak)),
        "params_unet_m": count_params(bundle.unet) / 1e6,
        "params_text_encoder_m": count_params(bundle.text_encoder) / 1e6,
        "params_decoder_m": count_params(decoder) / 1e6,
        "size_unet_mb": module_bytes(bundle.unet) / 2**20,
        "size_text_encoder_mb": module_bytes(bundle.text_encoder) / 2**20,
        "size_decoder_mb": module_bytes(decoder) / 2**20,
        "load_s": load_s,
        **flops,
    }
    out["params_total_m"] = out["params_unet_m"] + out["params_text_encoder_m"] + out["params_decoder_m"]
    out["size_total_mb"] = out["size_unet_mb"] + out["size_text_encoder_mb"] + out["size_decoder_mb"]
    out["unet_evals"] = cfg.unet_evals_per_image
    if dev == "cuda":
        out["effective_tflops"] = out["total_gflops"] / out["latency_ms_mean"]  # GFLOP/ms == TFLOP/s
    return out


@torch.inference_mode()
def unet_table(engine: DiffusionEngine, cfg, resolutions=(256, 384, 512), batches=(1, 2, 4, 8)) -> dict:
    """Median UNet ms per call for every (resolution, batch) the planner can choose."""
    engine.models.evict_all()
    b = engine.models.get(cfg)
    emb = engine.encode(b, ["a photo"])
    t = torch.tensor(500, device=b.device)
    tab = {}
    for r in resolutions:
        tab[str(r)] = {}
        for n in batches:
            x = torch.randn((n, 4, r // 8, r // 8), device=b.device, dtype=b.dtype)
            e = emb.expand(n, -1, -1)
            for _ in range(2):
                b.unet(x, t, encoder_hidden_states=e)
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                b.unet(x, t, encoder_hidden_states=e)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            tab[str(r)][str(n)] = float(np.median(ts))
    return tab


def bench_cpu(engine: DiffusionEngine, runs: int = 2) -> dict:
    cfg = PRESETS["cpu_fallback"]
    engine.models.evict_all()
    rows = {}
    for res in (256, 384):
        c = cfg.with_(height=res, width=res, name=f"cpu_fallback_{res}")
        engine.generate(PROMPT, c, seed=99)
        lat = []
        with PeakRSSMonitor() as mon:
            for s in range(runs):
                lat.append(engine.generate(PROMPT, c, seed=s).latency_ms)
        rows[c.name] = {"description": f"CPU fallback {res}px, LCM 2 steps, TAESD, FP32",
                        "latency_ms_mean": float(np.mean(lat)), "peak_rss_mb": mon.peak_mb,
                        "threads": torch.get_num_threads(), "config": c.to_dict()}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="also benchmark the CPU fallback")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    engine = DiffusionEngine()
    names = args.only or (COMPARISON_ORDER + list(EXTRA))
    allcfg = {**PRESETS, **EXTRA}
    results = read_json(RESULTS_DIR / "efficiency.json")
    results["_env"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "torch": torch.__version__, "cpu": platform.processor(),
                       "date": time.strftime("%Y-%m-%d %H:%M")}
    for name in names:
        cfg = allcfg[name]
        if not family_available(cfg.family):
            print(f"skip {name}: {cfg.family} not downloaded")
            continue
        try:
            r = bench_one(engine, cfg, args.runs)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        results[name] = r
        print(f"{name:28s} {r['latency_ms_mean']:8.1f} ms  peak {r['peak_mem_mb']:7.0f} MB  "
              f"size {r['size_total_mb']:7.0f} MB  {r['total_gflops']:9.0f} GFLOPs", flush=True)
        (RESULTS_DIR / "efficiency.json").write_text(__import__("json").dumps(results, indent=1), encoding="utf-8")
    if args.cpu:
        results.update(bench_cpu(engine))
        (RESULTS_DIR / "efficiency.json").write_text(__import__("json").dumps(results, indent=1), encoding="utf-8")

    # -------- cost calibration for the planner
    if torch.cuda.is_available():
        tables = {}
        for fam, preset in (("sd15", "sd15_lcm4_taesd"), ("sd_turbo", "turbo1_taesd")):
            if family_available(PRESETS[preset].family):
                tables[fam] = unet_table(engine, PRESETS[preset])
                print(f"UNet ms table {fam}: {tables[fam]}", flush=True)
        if tables:
            update_json(CALIBRATION_FILE, "unet_ms_table", tables)
            results["_unet_ms_table"] = tables
            (RESULTS_DIR / "efficiency.json").write_text(__import__("json").dumps(results, indent=1), encoding="utf-8")
    cal = read_json(CALIBRATION_FILE)
    unet_ms = cal.get("unet_ms_512", {})
    if "sd15_lcm4_taesd" in results:
        unet_ms["sd15"] = results["sd15_lcm4_taesd"]["unet_step_ms"]
    if "turbo1_taesd" in results:
        unet_ms["sd_turbo"] = results["turbo1_taesd"]["unet_step_ms"]
    if unet_ms:
        update_json(CALIBRATION_FILE, "unet_ms_512", unet_ms)
    if "sd15_lcm4_taesd_int8" in results and "sd15_lcm4_taesd" in results:
        base = results["sd15_lcm4_taesd"]["unet_step_ms"]
        qs = {"none": 1.0, "int8": results["sd15_lcm4_taesd_int8"]["unet_step_ms"] / base}
        if "sd15_lcm4_taesd_int4" in results:
            qs["int4"] = results["sd15_lcm4_taesd_int4"]["unet_step_ms"] / base
        update_json(CALIBRATION_FILE, "quant_slowdown", qs)
    if "sd15_lcm4" in results and "sd15_lcm4_taesd" in results:
        update_json(CALIBRATION_FILE, "decode_ms_512", {"full": results["sd15_lcm4"]["decode_ms"],
                                                        "taesd": results["sd15_lcm4_taesd"]["decode_ms"]})
        update_json(CALIBRATION_FILE, "text_ms", results["sd15_lcm4_taesd"]["text_ms"])
        update_json(CALIBRATION_FILE, "unet_gflops_512", results["sd15_lcm4_taesd"]["unet_gflops_per_call"])
        update_json(CALIBRATION_FILE, "decoder_gflops_512", {"full": results["sd15_lcm4"]["decoder_gflops"],
                                                             "taesd": results["sd15_lcm4_taesd"]["decoder_gflops"]})
        act = results["sd15_lcm4_taesd"]["peak_mem_mb"] - results["sd15_lcm4_taesd"]["size_total_mb"]
        update_json(CALIBRATION_FILE, "act_mb_512", max(act, 100.0))
    if "cpu_fallback_256" in results and "sd15_lcm4_taesd" in results:
        gpu = results["sd15_lcm4_taesd"]["unet_step_ms"] * (256 / 512) ** 2 * 2
        update_json(CALIBRATION_FILE, "cpu_slowdown",
                    max(5.0, results["cpu_fallback_256"]["latency_ms_mean"] / max(gpu, 1.0)))
    print("efficiency results ->", RESULTS_DIR / "efficiency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
