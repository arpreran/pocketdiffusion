"""Diffusion inference engine.

We load the pretrained components ourselves (tokenizer, text encoder, UNet, decoders) and
run our own denoising loop instead of calling a black-box pipeline. That gives us:

* per-stage timing (text encoding / each UNet step / decoding) and peak-memory measurement,
* "anytime" generation: every step produces an x0 estimate, so a deadline can stop the loop
  early and still return a valid image (deadline-aware, resource-aware generation),
* denoising-trajectory previews for the UI (decoded with the 2.4M-param TAESD),
* a model cache that swaps weights to respect a memory budget,
* an OOM fallback chain (fewer candidates -> lower resolution -> tiny decoder -> INT8 -> CPU).
"""
from __future__ import annotations

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .config import DISTILLED_TE_DIR, MODEL_IDS, GenConfig
from .optimizations import magnitude_prune_, quantize_module_, set_fast_attention
from .utils import LOG, count_params, gpu_peak_memory, module_bytes, sync

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def resolve_device(cfg: GenConfig) -> GenConfig:
    """Bad-environment handling: no CUDA -> run the same plan on CPU in FP32."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        return cfg.with_(device="cpu", dtype="fp32")
    if cfg.device == "cpu" and cfg.dtype == "fp16":
        return cfg.with_(dtype="fp32")  # CPU half-precision kernels are slow / incomplete
    return cfg


@dataclass
class ModelBundle:
    key: tuple
    family: str
    device: str
    dtype: torch.dtype
    tokenizer: object
    text_encoder: torch.nn.Module
    unet: torch.nn.Module
    scheduler_config: dict
    vae: torch.nn.Module | None = None
    info: dict = field(default_factory=dict)

    def weights_mb(self) -> float:
        mods = [self.text_encoder, self.unet] + ([self.vae] if self.vae is not None else [])
        return sum(module_bytes(m) for m in mods) / 2**20


@dataclass
class GenResult:
    images: list[np.ndarray]                    # HxWx3 float32 in [0,1]
    config: GenConfig
    seeds: list[int]
    timings_ms: dict
    step_ms: list[float]
    steps_done: int
    early_exit: bool
    peak_mem_mb: float
    weights_mb: float
    previews: list[np.ndarray] = field(default_factory=list)   # x0 per step (first candidate)
    events: list[str] = field(default_factory=list)
    resident_mb: float = 0.0          # CUDA memory already allocated before this call

    @property
    def latency_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


class ModelManager:
    """LRU cache of loaded model bundles with a soft VRAM budget."""

    def __init__(self, max_bundles: int = 2, vram_budget_mb: float = 6500):
        self.cache: OrderedDict[tuple, ModelBundle] = OrderedDict()
        self.max_bundles = max_bundles
        self.vram_budget_mb = vram_budget_mb
        self._taesd: dict[tuple, torch.nn.Module] = {}

    # ------------------------------------------------------------------ public
    def get(self, cfg: GenConfig) -> ModelBundle:
        key = cfg.model_key()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self._make_room(cfg)
        bundle = self._load(cfg)
        self.cache[key] = bundle
        return bundle

    def taesd(self, device: str, dtype: torch.dtype) -> torch.nn.Module:
        k = (device, dtype)
        if k not in self._taesd:
            from diffusers import AutoencoderTiny
            m = AutoencoderTiny.from_pretrained(MODEL_IDS["taesd"], torch_dtype=dtype).to(device)
            m.eval().requires_grad_(False)
            self._taesd[k] = m
        return self._taesd[k]

    def full_vae(self, bundle: ModelBundle) -> torch.nn.Module:
        if bundle.vae is None:
            from diffusers import AutoencoderKL
            repo = MODEL_IDS[bundle.family]
            vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", variant="fp16",
                                                torch_dtype=bundle.dtype)
            bundle.vae = vae.to(bundle.device).eval().requires_grad_(False)
            bundle.info["vae_params"] = count_params(vae)
        return bundle.vae

    def evict_all(self) -> None:
        self.cache.clear()
        self._taesd.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def loaded(self) -> list[str]:
        return [f"{k[0]}/{k[1]}/lcm={k[2]}/q={k[3]}/te={k[4]}/{k[6]}" for k in self.cache]

    # ------------------------------------------------------------------ internals
    def _make_room(self, cfg: GenConfig) -> None:
        need = 4500 if cfg.dtype == "fp32" else 2600
        while self.cache and (len(self.cache) >= self.max_bundles or
                              (cfg.device == "cuda" and self._resident_mb() + need > self.vram_budget_mb)):
            k, _ = self.cache.popitem(last=False)
            LOG.info("evicting model %s", k)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resident_mb(self) -> float:
        return sum(b.weights_mb() for b in self.cache.values() if b.device == "cuda")

    def _load(self, cfg: GenConfig) -> ModelBundle:
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        t0 = time.perf_counter()
        repo = MODEL_IDS[cfg.family]
        dtype = DTYPES[cfg.dtype]
        dev = cfg.device
        LOG.info("loading %s (%s, %s, lcm=%s, quant=%s, te=%s)", repo, cfg.dtype, dev,
                 cfg.lcm_lora, cfg.quant, cfg.text_encoder)
        tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        if cfg.text_encoder == "t6":
            if cfg.family != "sd15" or not DISTILLED_TE_DIR.exists():
                raise FileNotFoundError("Distilled text encoder not found - run "
                                        "scripts/distill_text_encoder.py first.")
            te = CLIPTextModel.from_pretrained(DISTILLED_TE_DIR, torch_dtype=dtype)
        else:
            te = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", variant="fp16",
                                               torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", variant="fp16",
                                                    torch_dtype=dtype)
        te.to(dev).eval().requires_grad_(False)
        unet.to(dev).eval().requires_grad_(False)
        info = {"attention": set_fast_attention(unet)}

        if cfg.lcm_lora:
            info["lcm_lora"] = self._fuse_lcm_lora(unet, te, tok, repo)
        if cfg.prune > 0:
            info["pruning"] = magnitude_prune_(unet, cfg.prune)
        info["unet_params"] = count_params(unet)
        info["text_encoder_params"] = count_params(te)
        if cfg.quant in ("int8", "int4"):
            bits = 8 if cfg.quant == "int8" else 4
            info["quant_unet"] = quantize_module_(unet, bits=bits)
        info["unet_mb"] = module_bytes(unet) / 2**20
        info["text_encoder_mb"] = module_bytes(te) / 2**20

        from diffusers import DDPMScheduler  # only to read the scheduler config
        sched_cfg = dict(DDPMScheduler.load_config(repo, subfolder="scheduler"))
        info["load_ms"] = (time.perf_counter() - t0) * 1000
        if dev == "cuda":
            torch.backends.cudnn.benchmark = True
        return ModelBundle(cfg.model_key(), cfg.family, dev, dtype, tok, te, unet, sched_cfg,
                           info=info)

    @staticmethod
    def _fuse_lcm_lora(unet, te, tok, repo) -> dict:
        """Load the LCM-LoRA consistency-distillation adapter and fuse it into the UNet weights
        (no runtime LoRA overhead afterwards)."""
        from diffusers import LCMScheduler, StableDiffusionPipeline
        sched = LCMScheduler.from_pretrained(repo, subfolder="scheduler")
        pipe = StableDiffusionPipeline(vae=None, text_encoder=te, tokenizer=tok, unet=unet,
                                       scheduler=sched, safety_checker=None,
                                       feature_extractor=None, requires_safety_checker=False)
        pipe.load_lora_weights(MODEL_IDS["lcm_lora"], weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="lcm")
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()
        del pipe
        return {"adapter": MODEL_IDS["lcm_lora"], "fused": True, "scale": 1.0}


# ============================================================================== schedulers
def make_scheduler(name: str, base_config: dict):
    from diffusers import (DPMSolverMultistepScheduler, EulerDiscreteScheduler, LCMScheduler,
                           PNDMScheduler)
    cfg = {k: v for k, v in base_config.items() if not k.startswith("_")}
    if name != "pndm":
        cfg.pop("skip_prk_steps", None)
        cfg.pop("set_alpha_to_one", None)
    if name == "pndm":
        return PNDMScheduler.from_config(cfg, skip_prk_steps=True)
    if name == "dpm":
        return DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++",
                                                       use_karras_sigmas=True)
    if name == "lcm":
        return LCMScheduler.from_config(cfg)
    if name == "euler":
        return EulerDiscreteScheduler.from_config(cfg, timestep_spacing="trailing")
    raise ValueError(f"unknown scheduler {name}")


# ============================================================================== engine
class DiffusionEngine:
    def __init__(self, manager: ModelManager | None = None):
        self.models = manager or ModelManager()
        self._emb_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    # ------------------------------------------------------------------ text
    @torch.inference_mode()
    def encode(self, bundle: ModelBundle, prompts: list[str]) -> torch.Tensor:
        """Prompt encoding with an LRU cache (repeated prompts cost nothing)."""
        out = []
        for p in prompts:
            k = (bundle.key, p)
            if k in self._emb_cache:
                self._emb_cache.move_to_end(k)
                out.append(self._emb_cache[k])
                continue
            ids = bundle.tokenizer([p], padding="max_length", truncation=True,
                                   max_length=bundle.tokenizer.model_max_length,
                                   return_tensors="pt").input_ids.to(bundle.device)
            emb = bundle.text_encoder(ids)[0]
            self._emb_cache[k] = emb
            if len(self._emb_cache) > 64:
                self._emb_cache.popitem(last=False)
            out.append(emb)
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------ decode
    @torch.inference_mode()
    def decode(self, bundle: ModelBundle, latents: torch.Tensor, vae: str) -> torch.Tensor:
        if vae == "taesd":
            dec = self.models.taesd(bundle.device, bundle.dtype)
            x = dec.decode(latents.to(bundle.dtype)).sample
        else:
            dec = self.models.full_vae(bundle)
            x = dec.decode(latents.to(bundle.dtype) / dec.config.scaling_factor).sample
        return (x.float() / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> list[np.ndarray]:
        x = torch.nan_to_num(x, nan=0.0)
        return [im for im in x.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)]

    # ------------------------------------------------------------------ main loop
    @torch.inference_mode()
    def generate(self, prompt: str, cfg: GenConfig, seed: int = 0, n: int = 1,
                 negative_prompt: str = "", deadline_ms: float | None = None,
                 preview: bool = False,
                 on_step: Callable[[int, int], None] | None = None,
                 on_preview: Callable[[int, int, np.ndarray], None] | None = None) -> GenResult:
        cfg = resolve_device(cfg)
        if cfg.height % 8 or cfg.width % 8:
            raise ValueError("height and width must be multiples of 8")
        timings: dict[str, float] = {}
        events: list[str] = []

        t0 = time.perf_counter()
        bundle = self.models.get(cfg)
        if cfg.vae == "full":
            self.models.full_vae(bundle)
        else:
            self.models.taesd(bundle.device, bundle.dtype)
        load_ms = (time.perf_counter() - t0) * 1000
        if load_ms > 50:
            events.append(f"model load/swap {load_ms:.0f} ms (not counted in latency)")

        dev, dtype = bundle.device, bundle.dtype
        sched = make_scheduler(cfg.scheduler, bundle.scheduler_config)
        seeds = [int(seed) + i for i in range(n)]
        torch.manual_seed(int(seed))  # LCM injects fresh noise between steps -> make it reproducible
        resident = torch.cuda.memory_allocated() / 2**20 if dev == "cuda" else 0.0
        t_start = time.perf_counter()   # deadline clock starts after (possible) model loading

        with gpu_peak_memory() as mem:
            sync(dev)
            t0 = time.perf_counter()
            cond = self.encode(bundle, [prompt])
            if cfg.cfg:
                uncond = self.encode(bundle, [negative_prompt or ""])
                emb = torch.cat([uncond.expand(n, -1, -1), cond.expand(n, -1, -1)])
            else:
                emb = cond.expand(n, -1, -1)
            sync(dev)
            timings["text_encode"] = (time.perf_counter() - t0) * 1000

            # initial noise from per-candidate CPU generators -> identical across devices
            shape = (1, bundle.unet.config.in_channels, cfg.height // 8, cfg.width // 8)
            lat = torch.cat([torch.randn(shape, generator=torch.Generator("cpu").manual_seed(s))
                             for s in seeds]).to(dev, dtype)
            sched.set_timesteps(cfg.steps, device=dev)
            lat = lat * sched.init_noise_sigma
            timesteps = sched.timesteps

            step_ms: list[float] = []
            previews: list[np.ndarray] = []
            x0 = None
            early = False
            decode_est = 15.0 if cfg.vae == "taesd" else 120.0
            for i, t in enumerate(timesteps):
                sync(dev)
                ts = time.perf_counter()
                inp = torch.cat([lat] * 2) if cfg.cfg else lat
                inp = sched.scale_model_input(inp, t)
                eps = bundle.unet(inp, t, encoder_hidden_states=emb).sample
                if cfg.cfg:
                    e_u, e_c = eps.chunk(2)
                    eps = e_u + cfg.guidance * (e_c - e_u)
                x_t = lat
                out = sched.step(eps, t, lat, return_dict=True)
                lat = out.prev_sample
                x0 = self._x0(out, sched, eps, x_t, t)
                sync(dev)
                step_ms.append((time.perf_counter() - ts) * 1000)
                if (preview or on_preview) and x0 is not None:
                    pv = self._to_numpy(self.decode(bundle, x0[:1], "taesd"))[0]
                    if preview:
                        previews.append(pv)
                    if on_preview:
                        on_preview(i + 1, len(timesteps), pv)
                if on_step:
                    on_step(i + 1, len(timesteps))
                if deadline_ms is not None and i < len(timesteps) - 1:
                    elapsed = (time.perf_counter() - t_start) * 1000
                    # fastest observed step: a one-off slow first step (lazy kernel loading)
                    # must not abort the whole generation
                    if elapsed + float(np.min(step_ms)) + decode_est * n > deadline_ms:
                        early = True
                        events.append(f"deadline {deadline_ms:.0f} ms reached -> stopped after "
                                      f"{i + 1}/{len(timesteps)} steps (anytime x0 output)")
                        break
            timings["denoise"] = float(sum(step_ms))
            final = x0 if (early and x0 is not None) else lat
            if cfg.scheduler == "lcm" and x0 is not None:
                final = x0  # consistency models output the denoised estimate
            if not torch.isfinite(final).all():
                events.append("non-finite latents detected (numerical overflow)")

            sync(dev)
            t0 = time.perf_counter()
            imgs = self._to_numpy(self.decode(bundle, final, cfg.vae))
            sync(dev)
            timings["decode"] = (time.perf_counter() - t0) * 1000

        peak = mem["peak_allocated_mb"] if dev == "cuda" else 0.0
        decoder_mb = sum(p.numel() * p.element_size() for p in (
            self.models.taesd(dev, dtype) if cfg.vae == "taesd" else self.models.full_vae(bundle)
        ).parameters()) / 2**20
        weights = (module_bytes(bundle.text_encoder) + module_bytes(bundle.unet)) / 2**20 + decoder_mb
        return GenResult(images=imgs, config=cfg, seeds=seeds, timings_ms=timings,
                         step_ms=step_ms, steps_done=len(step_ms), early_exit=early,
                         peak_mem_mb=peak, weights_mb=weights, previews=previews,
                         events=events, resident_mb=resident)

    @staticmethod
    def _x0(out, sched, eps, x_t, t):
        """Clean-image estimate at the current step (works for every scheduler we use)."""
        for attr in ("denoised", "pred_original_sample"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        if hasattr(sched, "alphas_cumprod"):
            a = sched.alphas_cumprod.to(x_t.device)[int(t)].to(x_t.dtype)
            return (x_t - (1 - a).sqrt() * eps) / a.sqrt()
        return None

    @torch.inference_mode()
    def warmup_shapes(self, cfg: GenConfig, resolutions=(256, 384, 512), batches=(1, 2, 4)) -> float:
        """Run one UNet + decoder call per (resolution, batch) so CUDA kernels for every shape
        the planner may pick are loaded before the first real request."""
        cfg = resolve_device(cfg)
        t0 = time.perf_counter()
        b = self.models.get(cfg)
        emb = self.encode(b, ["warm-up"])
        dec = self.models.taesd(b.device, b.dtype)
        t = torch.tensor(500, device=b.device)
        for r in resolutions:
            for n in batches:
                x = torch.zeros((n, 4, r // 8, r // 8), device=b.device, dtype=b.dtype)
                b.unet(x, t, encoder_hidden_states=emb.expand(n, -1, -1))
                dec.decode(x)
        sync(b.device)
        return (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ fallback
    def generate_with_fallback(self, prompt: str, cfg: GenConfig, **kw) -> GenResult:
        """Resource-aware execution: on out-of-memory, degrade the plan step by step."""
        events = []
        n = kw.pop("n", 1)
        chain = fallback_chain(cfg, n)
        last_err = None
        for c, nn_ in chain:
            try:
                res = self.generate(prompt, c, n=nn_, **kw)
                res.events = events + res.events
                return res
            except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
                last_err = e
                events.append(f"OOM with {c.name} ({c.height}px, n={nn_}, {c.quant}) -> degrading")
                LOG.warning(events[-1])
                self.models.evict_all()
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                last_err = e
                events.append(f"OOM ({c.name}) -> degrading")
                self.models.evict_all()
        raise RuntimeError(f"All fallback levels failed: {last_err}")

    # ------------------------------------------------------------------ analysis
    @torch.inference_mode()
    def count_flops(self, cfg: GenConfig) -> dict:
        """FLOPs of one image under `cfg` via torch's FlopCounterMode (conv, matmul, sdpa)."""
        from torch.utils.flop_counter import FlopCounterMode

        cfg = resolve_device(cfg)
        bundle = self.models.get(cfg)
        dev, dtype = bundle.device, bundle.dtype
        ids = torch.zeros((1, bundle.tokenizer.model_max_length), dtype=torch.long, device=dev)
        lat = torch.zeros((2 if cfg.cfg else 1, 4, cfg.height // 8, cfg.width // 8), device=dev, dtype=dtype)
        t = torch.tensor(500, device=dev)
        res = {}
        with FlopCounterMode(display=False) as fc:
            emb = bundle.text_encoder(ids)[0]
        res["text_encoder_gflops"] = fc.get_total_flops() / 1e9
        emb = emb.expand(lat.shape[0], -1, -1)
        with FlopCounterMode(display=False) as fc:
            bundle.unet(lat, t, encoder_hidden_states=emb)
        per_call = fc.get_total_flops() / 1e9
        evals = cfg.steps + (1 if cfg.scheduler == "pndm" else 0)
        res["unet_gflops_per_call"] = per_call
        res["unet_gflops_total"] = per_call * evals
        dec = self.models.taesd(dev, dtype) if cfg.vae == "taesd" else self.models.full_vae(bundle)
        with FlopCounterMode(display=False) as fc:
            dec.decode(lat[:1])
        res["decoder_gflops"] = fc.get_total_flops() / 1e9
        res["total_gflops"] = res["text_encoder_gflops"] * (2 if cfg.cfg else 1) + \
            res["unet_gflops_total"] + res["decoder_gflops"]
        return res


def fallback_chain(cfg: GenConfig, n: int) -> list[tuple[GenConfig, int]]:
    chain = [(cfg, n)]
    if n > 1:
        chain.append((cfg, 1))
    c = cfg
    if c.height * c.width > 384 * 384:
        c = c.with_(height=384, width=384, name=c.name + "+384px")
        chain.append((c, 1))
    if c.vae != "taesd":
        c = c.with_(vae="taesd", name=c.name + "+taesd")
        chain.append((c, 1))
    if c.quant == "none":
        c = c.with_(quant="int8", name=c.name + "+int8")
        chain.append((c, 1))
    c = c.with_(height=256, width=256, name=c.name + "+256px")
    chain.append((c, 1))
    if c.device == "cuda":
        chain.append((c.with_(device="cpu", dtype="fp32", quant="none", name=c.name + "+cpu"), 1))
    return chain
