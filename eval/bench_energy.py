"""Energy benchmark: joules per image for every configuration (GPU energy counter via NVML).

NVML's total-energy counter (millijoules since the driver loaded) updates too coarsely to time a
single 85 ms image, so each configuration generates images back-to-back for at least `--seconds`
(after 2 warm-up images) and the energy difference is divided by the number of images. The idle
power of the GPU is measured first (models loaded, nothing running) and reported separately:

  total J/image      = everything the GPU drew while generating
  above-idle J/image = total - idle power x time   (the energy the generation itself added)

Only the GPU is measured (not CPU, RAM or screen). Results -> results/energy.json.
Usage: python -m eval.bench_energy [--seconds 8] [--only turbo1_taesd ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from eval.bench_efficiency import EXTRA, PROMPT
from eval.common import RESULTS_DIR, read_json
from pocketdiff.config import COMPARISON_ORDER, PRESETS
from pocketdiff.engine import DiffusionEngine
from pocketdiff.planner import family_available


class GpuEnergy:
    def __init__(self, index: int = 0):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def joules(self) -> float:
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0

    def window(self, seconds: float, work=None, min_calls: int = 3) -> tuple[float, float, int]:
        """(joules, seconds, n_calls) over at least `seconds` (and `min_calls` calls of `work()`)."""
        torch.cuda.synchronize()
        e0, t0, n = self.joules(), time.perf_counter(), 0
        while True:
            if work is not None:
                work()
                n += 1
                torch.cuda.synchronize()
            else:
                time.sleep(0.25)
            if time.perf_counter() - t0 >= seconds and (work is None or n >= min_calls):
                break
        time.sleep(0.15)                       # let the counter catch up with the last work
        return self.joules() - e0, time.perf_counter() - t0 - 0.15, n


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    meter = GpuEnergy()
    engine = DiffusionEngine()
    allcfg = {**PRESETS, **EXTRA}
    names = args.only or (COMPARISON_ORDER + list(EXTRA) + ["bktiny_dpm12_taesd", "bktiny_dpm12_taesd_int8"])
    out = read_json(RESULTS_DIR / "energy.json")
    out["_env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "date": time.strftime("%Y-%m-%d %H:%M"), "prompt": PROMPT, "size": "512x512, batch 1",
                   "method": "NVML total-energy counter over >= %.0f s of back-to-back images" % args.seconds}
    for name in names:
        cfg = allcfg[name]
        if not family_available(cfg.family):
            print(f"skip {name}: {cfg.family} not downloaded")
            continue
        engine.models.evict_all()
        gc.collect()
        torch.cuda.empty_cache()
        engine.models.get(cfg)
        for s in range(2):
            engine.generate(PROMPT, cfg, seed=100 + s)
        time.sleep(2.0)
        idle_j, idle_s, _ = meter.window(4.0)              # same models resident, GPU idle
        idle_w = idle_j / idle_s
        seed = iter(range(10_000))
        j, s, n = meter.window(args.seconds, lambda: engine.generate(PROMPT, cfg, seed=next(seed)))
        per_img_s = s / n
        r = {"images": n, "seconds": s, "avg_power_w": j / s, "idle_power_w": idle_w,
             "latency_ms": per_img_s * 1000, "j_per_image": j / n,
             "j_above_idle_per_image": max(j / n - idle_w * per_img_s, 0.0),
             "images_per_wh": 3600.0 / (j / n)}
        out[name] = r
        print(f"{name:28s} {r['j_per_image']:8.2f} J/img  ({r['j_above_idle_per_image']:7.2f} above idle)  "
              f"{r['avg_power_w']:5.1f} W avg, idle {idle_w:4.1f} W, {n} images in {s:.1f} s", flush=True)
        (RESULTS_DIR / "energy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    base = out.get("baseline", {}).get("j_per_image")
    if base:
        for k, v in out.items():
            if isinstance(v, dict) and "j_per_image" in v:
                v["energy_saving_x"] = base / v["j_per_image"]
        (RESULTS_DIR / "energy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
