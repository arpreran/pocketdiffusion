"""Adaptive, budget-aware generation planner.

Given (prompt analysis, latency budget, memory budget, device profile, mode) the planner
enumerates every candidate configuration -

    model family x steps x native resolution x decoder x weight precision x text encoder x N

- predicts its cost (latency, peak memory) and its quality (CLIP score), then picks the
configuration with the best utility that fits the budget. If nothing fits, it falls back to
the cheapest configuration that fits memory (resource-aware fallback) and says so.

Cost and quality predictions start from conservative priors and are replaced by *measured*
values once `eval/eval_degradation.py` and `eval/bench_efficiency.py` have written
`artifacts/planner_calibration.json` - i.e. the planner is data-driven, not hand-tuned.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

from .config import DEVICE_PROFILES, DISTILLED_TE_DIR, MODEL_IDS, DeviceProfile, GenConfig, load_calibration
from .prompt_analyzer import PromptAnalysis
from .utils import cuda_available

CPU_SCALE = float(os.environ.get("POCKETDIFF_CPU_SCALE", "1.0"))

# ----------------------------------------------------------------------------- priors
PARAMS_M = {  # millions of parameters (used for memory prediction)
    "sd15": {"unet": 859.5, "te": 123.1, "te_t6": 80.6},
    "sd_turbo": {"unet": 865.9, "te": 340.4},
    "vae": 83.7, "taesd": 2.4, "clip_rerank": 151.3,
}
BYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.02, "int4": 0.56}

PRIOR = {
    # predicted CLIP-L/14 score at 512 px, single sample
    "clip": {
        "sd_turbo|euler": {1: 33.0, 2: 33.2, 4: 33.1},
        "sd15|lcm": {2: 30.0, 4: 31.0, 8: 31.2},
        "sd15|dpm": {20: 31.5},
    },
    "res_delta": {256: -2.5, 384: -0.8, 512: 0.0},
    "quant_delta": {"none": 0.0, "int8": -0.05, "int4": -0.7},
    "te_delta": {"full": 0.0, "t6": -0.6},
    "vae_delta": {"full": 0.0, "taesd": -0.1},
    "rerank_gain": {1: 0.0, 2: 0.6, 4: 1.2},
    # cost model on the local GPU (ms). Measured UNet ms per call: family -> res -> batch.
    # (Overwritten by eval/bench_efficiency.py; fixed overheads dominate at low resolution,
    #  so a simple pixels^k formula is badly wrong - hence a lookup table.)
    "unet_ms_table": {
        "sd15": {256: {1: 35.0, 2: 38.0, 4: 55.0}, 384: {1: 42.0, 2: 63.0, 4: 107.0},
                 512: {1: 60.0, 2: 100.0, 4: 181.0}},
        "sd_turbo": {256: {1: 35.0, 2: 38.0, 4: 55.0}, 384: {1: 42.0, 2: 63.0, 4: 107.0},
                     512: {1: 60.0, 2: 100.0, 4: 181.0}},
    },
    "unet_ms_512": {"sd15": 60.0, "sd_turbo": 60.0},
    "quant_slowdown": {"none": 1.0, "int8": 1.25, "int4": 1.6},
    "decode_ms_512": {"full": 145.0, "taesd": 15.0},
    "text_ms": 8.0, "rerank_ms_per_image": 10.0, "post_ms_512": 140.0,
    "cpu_slowdown": 35.0,
    "act_mb_512": 450.0,                 # activation memory per batch element at 512 px
    "unet_gflops_512": 800.0, "decoder_gflops_512": {"full": 1240.0, "taesd": 190.0},
    "te_gflops": 6.0,
}

MODE_LAMBDA = {"fast": 2.2, "balanced": 0.9, "quality": 0.25}


@dataclass
class Budget:
    latency_ms: float = 2000.0
    memory_mb: float | None = None          # None -> use the device profile's budget
    mode: str = "auto"                      # auto | fast | balanced | quality
    device_profile: str = "this_gpu"
    target_size: tuple[int, int] = (512, 512)
    max_candidates: int = 4


@dataclass
class Plan:
    config: GenConfig
    n_candidates: int
    postprocess: str
    upscale_to: tuple[int, int] | None
    predicted_latency_ms: float
    predicted_device_latency_ms: float | None
    predicted_memory_mb: float
    predicted_clip: float
    utility: float
    deadline_ms: float | None
    fallback: bool = False
    reasons: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


@lru_cache(maxsize=None)
def family_available(family: str) -> bool:
    """True if the checkpoint is in the local Hugging Face cache (no network access)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        need = ["unet/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors"]
        return all(isinstance(try_to_load_from_cache(MODEL_IDS[family], f), str) for f in need)
    except Exception:
        return False


class Planner:
    def __init__(self, calibration: dict | None = None):
        self.cal = calibration if calibration is not None else load_calibration()
        self.p = self._merge(PRIOR, self.cal)

    @staticmethod
    def _merge(prior: dict, cal: dict) -> dict:
        import copy
        p = copy.deepcopy(prior)
        for k, v in (cal or {}).items():
            if k == "clip" and isinstance(v, dict):
                for fam, table in v.items():
                    p["clip"].setdefault(fam, {}).update({int(s): float(c) for s, c in table.items()})
            elif isinstance(v, dict) and isinstance(p.get(k), dict):
                for kk, vv in v.items():
                    key = int(kk) if isinstance(kk, str) and kk.isdigit() else kk
                    p[k][key] = vv
            else:
                p[k] = v
        # Physical priors on measured deltas: compression (lower precision, smaller text encoder,
        # tiny decoder, lower resolution) cannot *add* fidelity - a positive measured delta is
        # sampling noise (s.e. ~0.4 CLIP at N=60) and would make the planner pick e.g. INT8 on a
        # GPU where it is slower. Re-ranking can only help.
        p["quant_delta"] = {k: min(float(v), 0.0) for k, v in p["quant_delta"].items()}
        p["te_delta"] = {k: min(float(v), 0.0) for k, v in p["te_delta"].items()}
        p["res_delta"] = {k: min(float(v), 0.0) for k, v in p["res_delta"].items()}
        vd = p["vae_delta"]
        if vd.get("taesd", 0.0) > vd.get("full", 0.0):
            vd["taesd"] = vd["full"] = 0.0
        p["vae_delta"] = {"full": max(float(vd.get("full", 0.0)) - float(vd.get("taesd", 0.0)), 0.0), "taesd": 0.0}
        p["rerank_gain"] = {k: max(float(v), 0.0) for k, v in p["rerank_gain"].items()}
        return p

    # ------------------------------------------------------------------ predictions
    def predict_clip(self, family: str, sched: str, steps: int, res: int, quant: str, te: str,
                     vae: str, n: int, complexity: float, constraints: int = 1) -> float:
        table = self.p["clip"].get(f"{family}|{sched}", {})
        if steps in table:
            base = table[steps]
        elif table:
            nearest = min(table, key=lambda s: abs(s - steps))
            base = table[nearest]
        else:
            base = 30.0
        res_d = self.p["res_delta"]
        rd = res_d.get(res, res_d[min(res_d, key=lambda r: abs(r - res))])
        # Best-of-N helps more when more things can go wrong: every requested object, colour,
        # count and relation is an independent chance for a single sample to fail, and the
        # re-ranker picks the candidate that satisfies most of them.
        gain = self.p["rerank_gain"].get(n, 0.0) * (0.6 + 0.8 * complexity + 0.35 * max(constraints - 1, 0))
        return (base + rd + self.p["quant_delta"][quant] + self.p["te_delta"][te]
                + self.p["vae_delta"][vae] + gain)

    def unet_ms(self, family: str, res: int, batch: int) -> float:
        """UNet time per call from the measured (res, batch) table, interpolated/extrapolated."""
        table = self.p.get("unet_ms_table", {}).get(family)
        if not table:
            return self.p["unet_ms_512"][family] * (res / 512.0) ** 2.2 * batch ** 0.85
        tab = {int(r): {int(b): float(v) for b, v in row.items()} for r, row in table.items()}
        rs = sorted(tab)

        def at_res(r):
            row = tab[r]
            bs = sorted(row)
            if batch in row:
                return row[batch]
            if batch > bs[-1]:          # linear extrapolation in batch beyond the table
                return row[bs[-1]] * batch / bs[-1]
            lo = max(b for b in bs if b < batch)
            hi = min(b for b in bs if b > batch)
            return row[lo] + (row[hi] - row[lo]) * (batch - lo) / (hi - lo)

        if res in tab:
            return at_res(res)
        if res < rs[0]:
            return at_res(rs[0]) * (res / rs[0]) ** 2
        if res > rs[-1]:
            return at_res(rs[-1]) * (res / rs[-1]) ** 2.2
        lo = max(r for r in rs if r < res)
        hi = min(r for r in rs if r > res)
        return at_res(lo) + (at_res(hi) - at_res(lo)) * (res - lo) / (hi - lo)

    def predict_latency(self, family: str, sched: str, steps: int, cfg_scale: float, res: int,
                        quant: str, vae: str, n: int, post: str, device: str) -> float:
        px = (res / 512.0) ** 2
        evals = steps + (1 if sched == "pndm" else 0)
        batch = n * (2 if cfg_scale > 1 else 1)
        unet = self.unet_ms(family, res, batch) * self.p["quant_slowdown"][quant]
        dec = self.p["decode_ms_512"][vae] * px * n
        rerank = (self.p["rerank_ms_per_image"] + self.p.get("screen_ms_512", 25.0) * px) * n if n > 1 else 0.0
        post_ms = 0.0 if post == "off" else self.p["post_ms_512"] * px * (0.6 if post == "light" else 1.0)
        total = self.p["text_ms"] + evals * unet + dec + rerank + post_ms
        if device == "cpu":
            # POCKETDIFF_CPU_SCALE: how much slower this CPU is than the one the table was measured on
            # (e.g. 4 for a 2-vCPU cloud server vs the 10-thread laptop CPU)
            total = (total - post_ms) * self.p["cpu_slowdown"] * CPU_SCALE + post_ms
        return total

    def predict_memory(self, family: str, res: int, quant: str, te: str, vae: str, n: int,
                       cfg_scale: float, dtype: str = "fp16") -> float:
        pm = PARAMS_M[family]
        unet_mb = pm["unet"] * BYTES[quant if quant != "none" else dtype]
        te_mb = (pm["te_t6"] if te == "t6" else pm["te"]) * BYTES[dtype]
        dec_mb = (PARAMS_M["taesd"] if vae == "taesd" else PARAMS_M["vae"]) * BYTES[dtype]
        rerank_mb = PARAMS_M["clip_rerank"] * BYTES[dtype] if n > 1 else 0.0
        act = self.p["act_mb_512"] * (res / 512.0) ** 2 * n * (2 if cfg_scale > 1 else 1) ** 0.7
        if vae == "full":
            act += 300.0 * (res / 512.0) ** 2   # VAE decoder activations dominate at decode
        return (unet_mb + te_mb + dec_mb + rerank_mb) * 1.0 + act

    def predict_device_latency(self, prof: DeviceProfile, family: str, sched: str, steps: int,
                               cfg_scale: float, res: int, vae: str, n: int) -> float | None:
        if prof.kind != "simulated" or prof.eff_tflops <= 0:
            return None
        px = (res / 512.0) ** 2
        evals = (steps + (1 if sched == "pndm" else 0)) * (2 if cfg_scale > 1 else 1) * n
        gflops = (self.p["te_gflops"] + evals * self.p["unet_gflops_512"] * px
                  + n * self.p["decoder_gflops_512"][vae] * px)
        return gflops / (prof.eff_tflops * 1000.0) * 1000.0  # GFLOP / (TFLOP/s) -> ms

    # ------------------------------------------------------------------ planning
    def candidates(self, prof: DeviceProfile, budget: Budget):
        fams = []
        if family_available("sd_turbo"):
            fams += [("sd_turbo", "euler", s, 0.0, False) for s in (1, 2, 4)]
        if family_available("sd15"):
            fams += [("sd15", "lcm", s, 1.0, True) for s in (2, 4, 8)]
            fams += [("sd15", "dpm", 20, 7.5, False)]
        te_opts = ["full"] + (["t6"] if DISTILLED_TE_DIR.exists() else [])
        for fam, sched, steps, g, lcm in fams:
            for res in (256, 384, 512):
                for vae in ("taesd", "full"):
                    for quant in ("none", "int8", "int4"):
                        for te in te_opts:
                            if te == "t6" and fam != "sd15":
                                continue
                            for n in (1, 2, 4):
                                if n > budget.max_candidates:
                                    continue
                                yield fam, sched, steps, g, lcm, res, vae, quant, te, n

    def plan(self, analysis: PromptAnalysis, budget: Budget) -> Plan:
        prof = DEVICE_PROFILES.get(budget.device_profile, DEVICE_PROFILES["this_gpu"])
        if prof.run_device == "cuda" and not cuda_available():
            # no GPU on this machine (e.g. the cloud backup): plan for what really runs - the CPU
            prof = DEVICE_PROFILES["this_cpu"] if prof.key == "this_gpu" else replace(prof, run_device="cpu")
        mem_budget = budget.memory_mb or prof.memory_budget_mb
        run_device = prof.run_device
        c = analysis.complexity
        mode = budget.mode
        if mode == "auto":
            lam = 1.0 - 0.7 * c          # complex prompts are allowed to spend more compute
            # a generous budget is a request for quality: spend the slack instead of returning in
            # 0.1 s at 384 px (budgets <= 2 s - all evaluation runs - are unchanged)
            lam *= min(1.0, 2000.0 / max(budget.latency_ms, 1.0))
        else:
            lam = MODE_LAMBDA.get(mode, 0.9)
        post = "full" if mode in ("quality", "balanced", "auto") else "light"
        tw, th = budget.target_size
        if not tw or not th:
            tw, th = analysis.requested_size or (512, 512)
        n_constraints = max(1, len(analysis.objects) + sum(o.color is not None for o in analysis.objects)
                            + sum(o.count > 1 for o in analysis.objects) + len(analysis.relations))

        rows = []
        for fam, sched, steps, g, lcm, res, vae, quant, te, n in self.candidates(prof, budget):
            lat = self.predict_latency(fam, sched, steps, g, res, quant, vae, n, post, run_device)
            dev_lat = self.predict_device_latency(prof, fam, sched, steps, g, res, vae, n)
            mem = self.predict_memory(fam, res, quant, te, vae, n, g)
            clip = self.predict_clip(fam, sched, steps, res, quant, te, vae, n, c, n_constraints)
            eff_lat = dev_lat if dev_lat is not None else lat
            util = clip - lam * math.log2(max(eff_lat, 1.0) / 100.0)
            # quantisation is a memory tool: never pick it when fp16 already fits
            rows.append(dict(family=fam, scheduler=sched, steps=steps, guidance=g, lcm=lcm,
                             res=res, vae=vae, quant=quant, te=te, n=n, latency=lat,
                             device_latency=dev_lat, memory=mem, clip=clip, utility=util,
                             fits=(eff_lat <= budget.latency_ms and mem <= mem_budget)))
        if not rows:
            raise RuntimeError("No diffusion checkpoints found locally - run scripts/download_assets.py")

        feasible = [r for r in rows if r["fits"]]
        fallback = False
        reasons = []
        if feasible:
            best = max(feasible, key=lambda r: (round(r["utility"], 3), -r["memory"]))
        else:
            fallback = True
            mem_ok = [r for r in rows if r["memory"] <= mem_budget] or rows
            best = min(mem_ok, key=lambda r: (r["device_latency"] or r["latency"], -r["clip"]))
            reasons.append("No configuration meets both budgets -> resource-aware fallback to the "
                           "fastest configuration that fits in memory.")

        cfg = GenConfig(
            name=f"plan_{best['family']}_{best['scheduler']}{best['steps']}_{best['res']}",
            family=best["family"], dtype="fp16" if run_device == "cuda" else "fp32",
            scheduler=best["scheduler"], steps=best["steps"], guidance=best["guidance"],
            lcm_lora=best["lcm"], vae=best["vae"], quant=best["quant"], text_encoder=best["te"],
            height=best["res"], width=best["res"], device=run_device,
            description="planner-selected")
        # aspect-ratio support: keep the pixel budget of the chosen square resolution
        if (tw, th) != (512, 512):
            from .classical.resize import snap_resolution
            w, h = snap_resolution(tw, th, max_pixels=best["res"] ** 2)
            cfg = cfg.with_(width=w, height=h)
        upscale = (tw, th) if (tw, th) != (cfg.width, cfg.height) else None

        fam_label = "SD-Turbo (1-step distilled)" if best["family"] == "sd_turbo" else (
            "SD1.5 + LCM-LoRA" if best["lcm"] else "SD1.5 DPM-Solver++")
        reasons += [
            f"Prompt complexity {c:.2f} ({analysis.complexity_label}); mode '{mode}' -> "
            f"latency weight {lam:.2f}.",
            f"Model: {fam_label}, {best['steps']} step(s) at {cfg.width}x{cfg.height}.",
            f"Predicted CLIP {best['clip']:.1f}, latency {best['latency']:.0f} ms on "
            f"{'GPU' if run_device == 'cuda' else 'CPU'}, memory {best['memory']:.0f} MB "
            f"(budget {mem_budget:.0f} MB).",
        ]
        if best["device_latency"] is not None:
            reasons.append(f"Estimated on-device latency for {prof.label}: "
                           f"{best['device_latency']:.0f} ms (FLOPs / assumed throughput).")
        if best["quant"] != "none":
            reasons.append(f"Memory budget requires {best['quant'].upper()} weight quantisation.")
        if best["te"] == "t6":
            reasons.append("Using our distilled 6-layer text encoder to save memory.")
        if best["vae"] == "taesd":
            reasons.append("Tiny decoder (TAESD) instead of the 84M-param VAE decoder.")
        if best["n"] > 1:
            reasons.append(f"{best['n']} candidates generated in one batch and re-ranked by "
                           f"CLIP + image quality (best-of-N).")
        if upscale and max(tw, th) > 1536:
            why = f" (prompt asks for {analysis.requested_size_label})" if analysis.requested_size_label else ""
            reasons.append(f"Output {tw}x{th}{why}: generate {cfg.width}x{cfg.height} natively, then classical "
                           f"super-resolution (IBP up to 4K + Lanczos) on the CPU. Diffusion models break "
                           f"down far above their training size, so detail stays that of the native image.")
        elif upscale:
            reasons.append(f"Target {tw}x{th} differs from native size -> classical IBP "
                           f"super-resolution / resize.")

        top = sorted(feasible or rows, key=lambda r: -r["utility"])[:8]
        return Plan(config=cfg, n_candidates=best["n"], postprocess=post, upscale_to=upscale,
                    predicted_latency_ms=best["latency"],
                    predicted_device_latency_ms=best["device_latency"],
                    predicted_memory_mb=best["memory"], predicted_clip=best["clip"],
                    utility=best["utility"],
                    deadline_ms=budget.latency_ms if prof.kind == "measured" else None,
                    fallback=fallback, reasons=reasons, alternatives=top)
