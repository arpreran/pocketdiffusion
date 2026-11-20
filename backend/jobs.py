"""Background GPU worker + job registry.

A single worker thread owns the GPU (one generation at a time, like a phone would). The API
enqueues jobs and clients subscribe to a job's event log over Server-Sent Events. Events are
appended to a list, so a client that connects late (or reconnects) replays everything.
"""
from __future__ import annotations

import base64
import io
import json
import math
import queue
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pocketdiff.config import PRESETS, GenConfig
from pocketdiff.planner import Budget
from pocketdiff.prompt_analyzer import resolve_output_size
from pocketdiff.pipeline import PocketDiffusion, PromptRejected
from pocketdiff.utils import LOG

from .db import Database


def to_pil(img01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(np.nan_to_num(img01), 0, 1) * 255).round().astype(np.uint8))


def b64_jpeg(img01: np.ndarray, size: int = 256, quality: int = 70) -> str:
    im = to_pil(img01)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def config_from_dict(d: dict) -> GenConfig:
    """GenConfig stored as JSON (to_dict adds derived keys such as 'cfg') -> GenConfig."""
    names = {f.name for f in dc_fields(GenConfig)}
    return GenConfig(**{k: v for k, v in (d or {}).items() if k in names})


MAX_UPLOAD_PIXELS = 40_000_000      # 40 MP: bigger uploads are rejected (decompression bombs)


def decode_upload(data: str) -> Image.Image:
    """'data:image/...;base64,...' (or bare base64) -> RGB PIL image, EXIF rotation applied."""
    b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        raise PromptRejected("The uploaded file is not an image PocketDiffusion can read (use JPEG, PNG or WebP).")
    if im.width * im.height > MAX_UPLOAD_PIXELS:
        raise PromptRejected("The uploaded image is too large (over 40 megapixels).")
    return ImageOps.exif_transpose(im).convert("RGB")


def edit_size(w: int, h: int, long_side: int = 512) -> tuple[int, int]:
    """Working size for image-to-image: long side `long_side`, aspect kept, multiples of 8."""
    s = long_side / max(w, h)
    return max(128, int(round(w * s / 8)) * 8), max(128, int(round(h * s / 8)) * 8)


def _image_saver(d: Path, gid: str):
    def save(img, name, fmt="JPEG", size=None, quality=88):
        if img is None:
            return None
        im = img if isinstance(img, Image.Image) else to_pil(img)
        if size:
            im = im.copy()
            im.thumbnail((size, size))
        im.save(d / name, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
        return f"{gid}/{name}"
    return save


def _jsonable(x):
    """numpy scalars / arrays / tuples -> plain JSON types (recursively)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, np.ndarray) or v.ndim == 0}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    created: float = field(default_factory=time.time)
    status: str = "queued"              # queued | running | done | error
    events: list[dict] = field(default_factory=list)
    result_id: str | None = None
    cancelled: bool = False             # live previews: superseded by a newer keystroke
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: str, data: dict | None = None) -> None:
        with self.lock:
            self.events.append({"event": event, "data": _jsonable(data or {}), "t": time.time()})


class JobManager:
    def __init__(self, db: Database, store_dir: Path):
        self.db = db
        self.store = Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[Job] = queue.Queue(maxsize=32)
        self.pd: PocketDiffusion | None = None
        self.ready = threading.Event()
        self.warmup_error: str | None = None
        self.busy_job: str | None = None
        self.worker = threading.Thread(target=self._loop, daemon=True, name="gpu-worker")
        self.worker.start()

    # ------------------------------------------------------------------ public
    def submit(self, kind: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        self.jobs[job.id] = job
        try:
            self.q.put_nowait(job)
        except queue.Full:
            job.status = "error"
            job.emit("error", {"message": "Server is busy (queue full). Please try again in a moment."})
            return job
        job.emit("queued", {"position": self.q.qsize(), "busy": self.busy_job is not None})
        self._gc()
        return job

    def pending_for(self, owner: str) -> int:
        """Jobs of one visitor that are still queued or running."""
        return sum(1 for j in list(self.jobs.values()) if j.kind != "preview"
                   and j.params.get("owner") == owner and j.status in ("queued", "running"))

    def busy_with_full_job(self) -> bool:
        """A generation / edit / race is running (live previews then wait for the next keystroke)."""
        j = self.jobs.get(self.busy_job or "")
        return bool(j and j.kind != "preview")

    def queue_length(self) -> int:
        return self.q.qsize() + (1 if self.busy_job else 0)

    def pipeline(self) -> PocketDiffusion:
        if self.pd is None:
            self.pd = PocketDiffusion()
        return self.pd

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            job = self.q.get()
            if job.cancelled:                       # a newer live preview replaced this one
                job.status = "done"
                job.emit("result", {"superseded": True})
                job.emit("end", {"status": "done"})
                continue
            self.busy_job = job.id
            job.status = "running"
            job.emit("start", {"kind": job.kind})
            try:
                if job.kind == "generate":
                    self._generate(job)
                elif job.kind == "compare":
                    self._compare(job)
                elif job.kind == "warmup":
                    self._warmup(job)
                elif job.kind == "attention":
                    self._attention(job)
                elif job.kind == "edit":
                    self._edit(job)
                elif job.kind == "preview":
                    self._preview(job)
                job.status = "done"
            except PromptRejected as e:
                job.status = "error"
                job.emit("error", {"message": str(e)})
            except Exception as e:
                LOG.error("job %s failed: %s\n%s", job.id, e, traceback.format_exc())
                job.status = "error"
                job.emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                job.emit("end", {"status": job.status})
                self.busy_job = None

    def _warmup(self, job: Job, attempts: int = 3) -> None:
        """Load and warm the models; retried, and a final failure is shown in the UI (not 'warming up' forever)."""
        for i in range(1, attempts + 1):
            try:
                self._warmup_once(job)
                self.warmup_error = None
                return
            except Exception as e:
                self.warmup_error = f"{type(e).__name__}: {e}"
                LOG.error("warm-up attempt %d/%d failed: %s\n%s", i, attempts, e, traceback.format_exc())
                if i < attempts:
                    time.sleep(5 * i)
        raise RuntimeError(f"model warm-up failed {attempts} times: {self.warmup_error}")

    def _warmup_once(self, job: Job) -> None:
        from pocketdiff.planner import family_available
        from pocketdiff.preload import import_ml_libraries
        from pocketdiff.utils import cuda_available
        import_ml_libraries()
        pd = self.pipeline()
        if cuda_available():   # per-shape kernel autotuning only matters on the GPU
            for name in ("turbo1_taesd", "sd15_lcm4_taesd"):
                if family_available(PRESETS[name].family):
                    ms = pd.engine.warmup_shapes(PRESETS[name])
                    job.emit("stage", {"name": f"warm-up {name}", "ms": ms})
            pd.run("a photo of a cat", budget=Budget(latency_ms=3000, mode="quality"), seed=0)
        else:                  # CPU server: just load the planner's usual model once
            pd.run("a photo of a cat", budget=Budget(latency_ms=20000, mode="fast", device_profile="this_cpu"), seed=0)
        if pd.verifier is not None:
            pd.verifier.detect(np.zeros((64, 64, 3), np.float32), ["cat"])
        self.ready.set()

    def _budget(self, p: dict) -> Budget:
        w, h = p.get("width", 512), p.get("height", 512)
        return Budget(latency_ms=float(p.get("latency_ms", 1500)), memory_mb=float(p.get("memory_mb") or 0) or None,
                      mode=p.get("mode", "auto"), device_profile=p.get("device_profile", "this_gpu"),
                      target_size=(int(w), int(h)))

    def _generate(self, job: Job) -> None:
        p = job.params
        pd = self.pipeline()
        seed = int(p["seed"]) if p.get("seed") is not None else random.randint(0, 2**31 - 1)
        cfg = PRESETS.get(p.get("preset") or "")
        t0 = time.perf_counter()

        def on_preview(i, total, img):
            job.emit("step", {"i": i, "total": total, "preview": b64_jpeg(img)})

        def on_step(i, total):
            if not p.get("stream_previews", True):
                job.emit("step", {"i": i, "total": total})

        upscale = None
        if cfg is not None:
            tw, th = resolve_output_size(p["prompt"], p.get("width", 512), p.get("height", 512))
            if (tw, th) != (cfg.width, cfg.height):
                upscale = (tw, th)
        res = pd.run(p["prompt"], budget=self._budget(p), seed=seed, negative_prompt=p.get("negative_prompt") or "",
                     config=cfg, postprocess=None if p.get("postprocess", "auto") == "auto" else p["postprocess"],
                     rerank=bool(p.get("rerank", True)), verify=bool(p.get("verify", True)),
                     semantic_rerank=bool(p.get("semantic_rerank", False)), preview=True,
                     on_preview=on_preview if p.get("stream_previews", True) else None, on_step=on_step,
                     on_event=lambda name, data=None: job.emit(name, data), upscale_to=upscale)
        wall = (time.perf_counter() - t0) * 1000
        chosen = res.candidates[res.chosen]
        if chosen.clip is None and pd.scorer is not None:   # single candidate: score it for display
            chosen.clip = float(pd.scorer.clip_score([res.image], res.analysis.clean_prompt)[0])
        job.emit("stage", {"name": "saving"})
        gid = self._save(job, res, seed, wall)
        job.result_id = gid
        job.emit("result", self.db.get(gid))

    def _compare(self, job: Job) -> None:
        p = job.params
        pd = self.pipeline()
        seed = int(p.get("seed") or 7)
        job.emit("lane", {"lane": "optimised", "state": "start"})
        t0 = time.perf_counter()
        opt = pd.run(p["prompt"], budget=Budget(latency_ms=1500, mode="auto"), seed=seed,
                     on_step=lambda i, t: job.emit("lane", {"lane": "optimised", "i": i, "total": t,
                                                            "elapsed_ms": (time.perf_counter() - t0) * 1000}))
        opt_id = self._save(job, opt, seed, (time.perf_counter() - t0) * 1000, kind="compare")
        job.emit("lane", {"lane": "optimised", "state": "done", "generation": self.db.get(opt_id)})
        job.emit("lane", {"lane": "baseline", "state": "start"})
        t1 = time.perf_counter()
        base = pd.run(p["prompt"], config=PRESETS["sd15_fp16"], seed=seed, n=1, postprocess="off", rerank=False,
                      verify=False, on_step=lambda i, t: job.emit("lane", {
                          "lane": "baseline", "i": i, "total": t, "elapsed_ms": (time.perf_counter() - t1) * 1000}))
        base_id = self._save(job, base, seed, (time.perf_counter() - t1) * 1000, kind="compare")
        scorer = pd.scorer
        clip_b = clip_o = None
        if scorer is not None:
            clip_b = float(scorer.clip_score([base.image], p["prompt"])[0])
            clip_o = float(scorer.clip_score([opt.image], p["prompt"])[0])
        job.emit("lane", {"lane": "baseline", "state": "done", "generation": self.db.get(base_id)})
        job.emit("result", {"baseline": self.db.get(base_id), "optimised": self.db.get(opt_id),
                            "clip_baseline": clip_b, "clip_optimised": clip_o,
                            "speedup": base.latency_ms / max(opt.latency_ms, 1e-3),
                            "unet_evals": [base.config.unet_evals_per_image,
                                           opt.config.unet_evals_per_image * len(opt.candidates)]})

    # ------------------------------------------------------------------ explain: attention maps
    def _attention(self, job: Job) -> None:
        """Cross-attention heat map per prompt word for a stored generation (cached on disk)."""
        from pocketdiff.attention import attention_maps, overlay, segmentation

        gid = job.params["gid"]
        g = self.db.get(gid)
        if g is None:
            raise PromptRejected("This generation no longer exists.")
        if g.get("kind") == "edit":
            raise PromptRejected("Explain works for text-to-image results; this image was edited from a picture.")
        cfg = config_from_dict(g.get("config") or {})
        seed = next((c["seed"] for c in g.get("candidates") or [] if c.get("chosen")), g.get("seed") or 0)
        job.emit("stage", {"name": "attention", "config": cfg.name})
        res = attention_maps(self.pipeline().engine, g["prompt"], cfg, int(seed),
                             negative_prompt=g.get("negative_prompt") or "")
        d = self.store / gid / "attn"
        d.mkdir(parents=True, exist_ok=True)
        base = Image.open(self.store / g["image_path"]).convert("RGB")
        base.thumbnail((768, 768))
        save = _image_saver(d, f"{gid}/attn")
        words = [{"word": w.word, "color": w.color, "share": round(w.share, 4), "tokens": len(w.positions),
                  "image": save(overlay(base, w.heat), f"w{i}.jpg", quality=86)}
                 for i, w in enumerate(res.words)]
        out = {"id": gid, "words": words, "segmentation": save(segmentation(base, res.words), "seg.jpg", quality=86),
               "layers": res.layers, "steps": res.steps, "ms": round(res.ms, 1), "latent": list(res.size),
               "model": cfg.name, "seed": int(seed)}
        (d / "attention.json").write_text(json.dumps(out), encoding="utf-8")
        job.emit("result", out)

    # ------------------------------------------------------------------ live preview while typing
    def _preview(self, job: Job) -> None:
        """One SD-Turbo step at 512 px, no candidates / post-processing / saving (~0.1 s on the GPU).
        Not stored: the full pipeline runs when the user presses Generate."""
        from pocketdiff.planner import family_available

        p = job.params
        pd = self.pipeline()
        t0 = time.perf_counter()
        analysis = pd.analyze(p["prompt"])
        if not analysis.clean_prompt:
            raise PromptRejected("empty prompt")
        if analysis.blocked_terms:
            raise PromptRejected(f"This prompt contains blocked terms ({', '.join(analysis.blocked_terms)}).")
        cfg = (PRESETS["turbo1_taesd"] if family_available("sd_turbo")
               else PRESETS["sd15_lcm4_taesd"].with_(steps=2)).with_(width=512, height=512)
        res = pd.engine.generate(analysis.clean_prompt, cfg, seed=int(p.get("seed") or 42), n=1)
        img = res.images[0]
        if pd.scorer is not None and pd.scorer.safety([img])[0]["unsafe"]:
            import cv2
            img = cv2.GaussianBlur(img, (0, 0), 25)
        job.emit("result", {"image": b64_jpeg(img, size=512, quality=82), "gen_ms": res.latency_ms,
                            "total_ms": (time.perf_counter() - t0) * 1000, "model": cfg.name,
                            "prompt": analysis.clean_prompt})

    # ------------------------------------------------------------------ image-to-image editing
    def _edit_source(self, p: dict) -> Image.Image:
        if p.get("source_id"):
            g = self.db.get(p["source_id"])
            if g is None:
                raise PromptRejected("The source image no longer exists.")
            return Image.open(self.store / (g.get("full_path") or g["image_path"])).convert("RGB")
        data = p.pop("image", None)                     # do not keep megabytes in the job registry
        if not data:
            raise PromptRejected("Upload an image (or pick one from the Gallery) to edit.")
        return decode_upload(data)

    def _edit(self, job: Job) -> None:
        """Image-to-image (SDEdit): the picture is encoded, noised to `strength` and redrawn to match
        the prompt. Low strength keeps composition and colours; high strength follows the prompt."""
        from pocketdiff.planner import family_available
        from pocketdiff.utils import cuda_available

        p = job.params
        pd = self.pipeline()
        t0 = time.perf_counter()
        analysis = pd.analyze(p["prompt"])
        if not analysis.clean_prompt:
            raise PromptRejected("Describe what the edited image should show.")
        if analysis.blocked_terms:
            raise PromptRejected(f"This prompt contains blocked terms ({', '.join(analysis.blocked_terms)}). "
                                 "Please rephrase.")
        job.emit("analysis", analysis.to_dict())
        src = self._edit_source(p)
        seed = int(p["seed"]) if p.get("seed") is not None else random.randint(0, 2**31 - 1)
        strength = float(min(max(float(p.get("strength", 0.55)), 0.1), 1.0))
        gpu = cuda_available()
        base = PRESETS["turbo1_taesd"] if family_available("sd_turbo") else PRESETS["sd15_lcm4_taesd"]
        w, h = edit_size(*src.size)
        cfg = base.with_(name=f"{base.name}_img2img", steps=4, width=w, height=h,
                         vae="full" if gpu else "taesd", device="cuda" if gpu else "cpu",
                         dtype="fp16" if gpu else "fp32")
        job.emit("stage", {"name": "denoise", "config": cfg.to_dict(), "n": 1})
        src01 = np.asarray(src.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        img, ms = pd.engine.refine(src01, analysis.clean_prompt, cfg, (w, h), strength=strength,
                                   steps=cfg.steps, seed=seed, exact=True)
        timings = {"gen.denoise": ms}
        events = [f"image-to-image: {src.size[0]}x{src.size[1]} source -> {w}x{h}, strength {strength:.2f} "
                  f"(denoising from t={int(strength * 999)}, {math.ceil(strength * cfg.steps)} steps)"]
        safety = None
        if pd.scorer is not None:
            ts = time.perf_counter()
            safety = pd.scorer.safety([img])[0]
            timings["safety"] = (time.perf_counter() - ts) * 1000
            if safety["unsafe"]:
                import cv2
                img = cv2.GaussianBlur(img, (0, 0), 25)
                events.append("Image-level safety check flagged the output -> blurred.")
        clip = float(pd.scorer.clip_score([img], analysis.clean_prompt)[0]) if pd.scorer is not None else None
        job.emit("stage", {"name": "saving"})
        wall = (time.perf_counter() - t0) * 1000
        gid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.store / gid
        d.mkdir(parents=True, exist_ok=True)
        save = _image_saver(d, gid)
        src.thumbnail((1024, 1024))
        row = {
            "id": gid, "created_at": time.time(), "kind": "edit", "prompt": analysis.clean_prompt,
            "owner": p.get("owner"),
            "negative_prompt": None, "seed": seed, "device_profile": "this_gpu" if gpu else "this_cpu",
            "mode": "edit", "family": cfg.family, "scheduler": cfg.scheduler, "steps": cfg.steps,
            "steps_done": math.ceil(strength * cfg.steps), "width": w, "height": h, "quant": cfg.quant,
            "decoder": cfg.vae, "text_encoder": cfg.text_encoder, "n_candidates": 1, "latency_ms": wall,
            "gen_ms": ms, "clip": clip, "image_path": save(img, "final.png", "PNG"),
            "raw_path": save(src, "source.jpg", quality=90),
            "analysis_json": json.dumps(_jsonable(analysis.to_dict())),
            "config_json": json.dumps(_jsonable(cfg.to_dict())),
            "timings_json": json.dumps(_jsonable({**timings, "wall": wall})),
            "metrics_json": json.dumps(_jsonable({"edit": {"strength": strength, "source_id": p.get("source_id")},
                                                  "safety": safety})),
            "events_json": json.dumps(events),
        }
        cands = [{"generation_id": gid, "idx": 0, "seed": seed, "clip": clip, "iqa": None, "score": None,
                  "chosen": 1, "image_path": save(img, "cand_0.jpg", size=384)}]
        self.db.insert_generation(row, cands, [], [])
        job.result_id = gid
        job.emit("result", self.db.get(gid))

    # ------------------------------------------------------------------ persistence
    def _save(self, job: Job, res, seed: int, wall_ms: float, kind: str = "generate") -> str:
        gid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.store / gid
        d.mkdir(parents=True, exist_ok=True)

        save = _image_saver(d, gid)

        cfg = res.config
        v = res.verification or {}
        big = max(res.image.shape[:2]) > 2048
        row = {
            "id": gid, "created_at": time.time(), "kind": kind, "prompt": res.analysis.clean_prompt,
            "owner": job.params.get("owner"),
            "negative_prompt": job.params.get("negative_prompt"), "seed": seed,
            "device_profile": job.params.get("device_profile", "this_gpu"), "mode": job.params.get("mode", "auto"),
            "latency_budget": job.params.get("latency_ms"), "memory_budget": job.params.get("memory_mb"),
            "family": cfg.family, "scheduler": cfg.scheduler, "steps": cfg.steps, "steps_done": res.gen.steps_done,
            "width": res.image.shape[1], "height": res.image.shape[0], "quant": cfg.quant, "decoder": cfg.vae,
            "text_encoder": cfg.text_encoder, "n_candidates": len(res.candidates), "latency_ms": res.latency_ms,
            "gen_ms": res.gen.latency_ms, "peak_mem_mb": res.gen.peak_mem_mb, "weights_mb": res.gen.weights_mb,
            "clip": res.candidates[res.chosen].clip, "iqa_before": res.iqa_before["pocket_iqa"],
            "iqa_after": res.iqa_after["pocket_iqa"], "niqe_before": res.iqa_before.get("niqe"),
            "niqe_after": res.iqa_after.get("niqe"), "object_f1": v.get("f1"),
            "all_ok": None if not v else int(bool(v.get("all_satisfied"))), "complexity": res.analysis.complexity,
            "early_exit": int(res.gen.early_exit), "fallback": int(bool(res.plan and res.plan.fallback)),
            # 2K/4K/8K: a 2048-px preview for the UI and the full-resolution JPEG for download
            "image_path": (save(res.image, "final.jpg", size=2048, quality=90) if big
                           else save(res.image, "final.png", "PNG")),
            "full_path": save(res.image, "full.jpg", quality=92) if big else None,
            "raw_path": save(res.raw_image, "raw.jpg"),
            "heatmap_path": save(res.heatmap, "heatmap.jpg"),
            "verify_path": save(res.verification_image, "verify.jpg"),
            "plan_json": json.dumps(_jsonable(res.plan.to_dict())) if res.plan else None,
            "analysis_json": json.dumps(_jsonable(res.analysis.to_dict())),
            "config_json": json.dumps(_jsonable(cfg.to_dict())),
            "timings_json": json.dumps(_jsonable({**res.timings_ms, "wall": wall_ms})),
            "metrics_json": json.dumps(_jsonable({
                "iqa_before": res.iqa_before, "iqa_after": res.iqa_after,
                "artifacts_before": res.artifacts_before, "artifacts_after": res.artifacts_after,
                "verification": v or None, "safety": res.safety, "step_ms": res.gen.step_ms,
                "resident_mb": res.gen.resident_mb, "seeds": res.gen.seeds})),
            "events_json": json.dumps(res.events),
        }
        cands = [{"generation_id": gid, "idx": i, "seed": c.seed, "clip": c.clip, "iqa": c.iqa, "score": c.score,
                  "chosen": int(i == res.chosen), "image_path": save(c.image, f"cand_{i}.jpg", size=384)}
                 for i, c in enumerate(res.candidates)]
        stages = [{"generation_id": gid, "idx": i, "name": s.name, "ms": s.ms,
                   "params_json": json.dumps(_jsonable(s.params)),
                   "image_path": save(s.image, f"stage_{i}_{s.name}.jpg", size=384) if s.image is not None else None}
                  for i, s in enumerate(res.post_stages)]
        frames = [{"generation_id": gid, "step": i + 1, "image_path": save(f, f"step_{i + 1}.jpg", size=256, quality=80)}
                  for i, f in enumerate(res.gen.previews)]
        self.db.insert_generation(row, cands, stages, frames)
        return gid

    def _gc(self, keep_s: float = 3600) -> None:
        now = time.time()
        for k in [k for k, j in self.jobs.items() if j.status in ("done", "error") and now - j.created > keep_s]:
            self.jobs.pop(k, None)
