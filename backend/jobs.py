"""Background GPU worker + job registry.

A single worker thread owns the GPU (one generation at a time, like a phone would). The API
enqueues jobs and clients subscribe to a job's event log over Server-Sent Events. Events are
appended to a list, so a client that connects late (or reconnects) replays everything.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pocketdiff.config import PRESETS
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

    def _warmup(self, job: Job) -> None:
        from pocketdiff.planner import family_available
        pd = self.pipeline()
        for name in ("turbo1_taesd", "sd15_lcm4_taesd"):
            if family_available(PRESETS[name].family):
                ms = pd.engine.warmup_shapes(PRESETS[name])
                job.emit("stage", {"name": f"warm-up {name}", "ms": ms})
        pd.run("a photo of a cat", budget=Budget(latency_ms=3000, mode="quality"), seed=0)
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

    # ------------------------------------------------------------------ persistence
    def _save(self, job: Job, res, seed: int, wall_ms: float, kind: str = "generate") -> str:
        gid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.store / gid
        d.mkdir(parents=True, exist_ok=True)

        def save(img, name, fmt="JPEG", size=None, quality=88):
            if img is None:
                return None
            im = to_pil(img)
            if size:
                im.thumbnail((size, size))
            path = d / name
            im.save(path, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
            return f"{gid}/{name}"

        cfg = res.config
        v = res.verification or {}
        big = max(res.image.shape[:2]) > 2048
        row = {
            "id": gid, "created_at": time.time(), "kind": kind, "prompt": res.analysis.clean_prompt,
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
