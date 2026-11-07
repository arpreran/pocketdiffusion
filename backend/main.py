"""PocketDiffusion REST + streaming API (FastAPI).

Routes
------
GET  /api/system                 hardware, models, queue, device profiles, presets, examples
POST /api/analyze                prompt -> scene graph, complexity, token count   (CPU, instant)
POST /api/plan                   prompt + budget -> planner decision              (CPU, instant)
POST /api/generate               enqueue a generation job -> {job_id}
POST /api/compare                enqueue baseline-vs-optimised race -> {job_id}
GET  /api/jobs/{id}/events       Server-Sent Events: queued/start/analysis/plan/step/stage/result/error/end
GET  /api/generations            history (paginated, filter: favorites, family, q)
GET  /api/generations/{id}       full record (candidates, stages, frames, metrics)
POST /api/generations/{id}/favorite
DELETE /api/generations/{id}
GET  /api/stats                  aggregate statistics from the database
GET  /api/benchmarks             evaluation results (results/*.json) for the Lab charts
GET  /media/...                  stored images;  GET /  -> single-page frontend
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from pocketdiff.config import DATA_DIR, DEVICE_PROFILES, MODEL_IDS, PRESETS, RESULTS_DIR, ROOT
from pocketdiff.planner import Budget, Planner, family_available
from pocketdiff.prompt_analyzer import analyze_prompt
from pocketdiff.utils import LOG

from .db import Database
from .jobs import JobManager

FRONTEND = ROOT / "frontend"
STORE = DATA_DIR / "store"
EXAMPLES = [
    "a red apple to the left of a green bottle on a wooden table, studio photo",
    "two cats sleeping on a blue couch next to a laptop",
    "an astronaut riding a horse on the moon, cinematic lighting",
    "a yellow bus parked in front of a red brick building, rainy evening",
    "three oranges in a white bowl, top view, soft light",
    "a cozy wooden cabin in a snowy forest at night, warm light in the windows, watercolor",
    "a corgi wearing sunglasses on a skateboard, golden hour",
    "a futuristic city skyline at dusk, neon reflections on wet streets",
]

app = FastAPI(title="PocketDiffusion API", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
db = Database(DATA_DIR / "pocketdiffusion.db")
jobs = JobManager(db, STORE)
STORE.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=STORE), name="media")
if (ROOT / "docs").exists():
    app.mount("/docs-static", StaticFiles(directory=ROOT / "docs"), name="docs")
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


# ------------------------------------------------------------------ request models
class BudgetIn(BaseModel):
    prompt: str = Field("", max_length=2000)
    device_profile: str = "this_gpu"
    mode: str = "auto"
    latency_ms: float = Field(1500, ge=50, le=120000)
    memory_mb: float | None = Field(None, ge=100, le=64000)
    width: int = Field(512, ge=0, le=7680)      # 0 x 0 = auto: size named in the prompt ("8K"), else 512
    height: int = Field(512, ge=0, le=7680)

    @field_validator("width", "height")
    @classmethod
    def _size(cls, v):
        if v and v < 128:
            raise ValueError("width/height must be 0 (auto) or at least 128")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v):
        if v not in ("auto", "fast", "balanced", "quality"):
            raise ValueError("mode must be auto, fast, balanced or quality")
        return v

    @field_validator("device_profile")
    @classmethod
    def _dev(cls, v):
        if v not in DEVICE_PROFILES:
            raise ValueError(f"unknown device profile '{v}'")
        return v


class GenerateIn(BudgetIn):
    negative_prompt: str = Field("", max_length=1000)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    postprocess: str = "auto"
    rerank: bool = True
    verify: bool = True
    semantic_rerank: bool = False
    preset: str | None = None
    stream_previews: bool = True

    @field_validator("postprocess")
    @classmethod
    def _pp(cls, v):
        if v not in ("auto", "off", "light", "full"):
            raise ValueError("postprocess must be auto, off, light or full")
        return v

    @field_validator("preset")
    @classmethod
    def _preset(cls, v):
        if v in (None, "", "planner"):
            return None
        if v not in PRESETS:
            raise ValueError(f"unknown preset '{v}'")
        return v


class CompareIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int = Field(7, ge=0, le=2**31 - 1)


# ------------------------------------------------------------------ helpers
_tok = None
_tok_lock = threading.Lock()


def tokenizer():
    """Thread-safe lazy load; a failure is NOT cached, so a later request retries."""
    global _tok
    if _tok is None:
        with _tok_lock:
            if _tok is None:
                try:
                    from transformers import CLIPTokenizer
                    _tok = CLIPTokenizer.from_pretrained(MODEL_IDS["sd15"], subfolder="tokenizer")
                except Exception as e:
                    LOG.warning("tokenizer unavailable: %s", e)
                    return None
    return _tok


_planner: Planner | None = None
_planner_lock = threading.Lock()


def planner() -> Planner:
    global _planner
    with _planner_lock:
        if _planner is None:
            _planner = Planner()
    return _planner


# warm the CPU-side helpers in the background so the first request is instant
threading.Thread(target=lambda: (tokenizer(), planner()), daemon=True).start()


def read_json(name: str) -> dict:
    p = RESULTS_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def baseline_ms() -> float | None:
    e = read_json("efficiency.json")
    return e.get("baseline", {}).get("latency_ms_mean")


# ------------------------------------------------------------------ routes
@app.get("/api/system")
def system():
    import torch
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "vram_mb": props.total_memory / 2**20}
    pd = jobs.pd
    return {
        "app": "PocketDiffusion", "version": "1.0.0", "gpu": gpu, "torch": torch.__version__,
        "ready": jobs.ready.is_set(), "queue": jobs.queue_length(),
        "loaded_models": pd.engine.models.loaded() if pd else [],
        "families": {f: family_available(f) for f in ("sd15", "sd_turbo")},
        "device_profiles": [{"key": k, "label": p.label, "kind": p.kind, "memory_budget_mb": p.memory_budget_mb,
                             "eff_tflops": p.eff_tflops, "notes": p.notes} for k, p in DEVICE_PROFILES.items()],
        "presets": [{"key": k, "description": c.description} for k, c in PRESETS.items() if k != "cpu_fallback"],
        "examples": EXAMPLES, "baseline_ms": baseline_ms(),
        # cloud backup (Hugging Face Space, CPU only): the UI says so and points to the GPU version
        "cloud": bool(os.environ.get("SPACE_ID")), "gpu_link": os.environ.get("POCKETDIFF_GPU_LINK", ""),
        "default_device": "this_gpu" if gpu else "this_cpu",
    }


@app.post("/api/analyze")
def analyze(body: BudgetIn):
    return analyze_prompt(body.prompt, tokenizer=tokenizer()).to_dict()


@app.post("/api/plan")
def plan(body: BudgetIn):
    pa = analyze_prompt(body.prompt or "a photo", tokenizer=tokenizer())
    try:
        p = planner().plan(pa, Budget(latency_ms=body.latency_ms, memory_mb=body.memory_mb, mode=body.mode,
                                      device_profile=body.device_profile, target_size=(body.width, body.height)))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    d = p.to_dict()
    d["analysis"] = pa.to_dict()
    return d


@app.post("/api/generate")
def generate(body: GenerateIn):
    if not body.prompt.strip():
        raise HTTPException(422, "Please enter a prompt describing the image.")
    job = jobs.submit("generate", body.model_dump())
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.post("/api/compare")
def compare(body: CompareIn):
    if not family_available("sd15"):
        raise HTTPException(503, "The race needs the SD1.5 baseline (50 steps, FP32), which is not installed on "
                                 "this server - it would take minutes per image without a GPU. "
                                 "Use the GPU version of the app for the race.")
    job = jobs.submit("compare", body.model_dump())
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def stream():
        i, idle = 0, 0
        yield "retry: 2000\n\n"
        while True:
            if await request.is_disconnected():
                return
            with job.lock:
                new = job.events[i:]
            for ev in new:
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
            i += len(new)
            if job.status in ("done", "error") and i >= len(job.events):
                return
            idle = 0 if new else idle + 1
            if idle and idle % 300 == 0:          # ~15 s without events: keep proxies/tunnels open
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/generations")
def list_generations(limit: int = 24, offset: int = 0, favorites: bool = False, family: str | None = None,
                     q: str | None = None):
    rows, total = db.list(limit=max(1, min(limit, 100)), offset=max(0, offset), favorites=favorites,
                          family=family or None, q=q or None)
    return {"items": rows, "total": total}


@app.get("/api/generations/{gid}")
def get_generation(gid: str):
    g = db.get(gid)
    if g is None:
        raise HTTPException(404, "not found")
    return g


@app.post("/api/generations/{gid}/favorite")
def favorite(gid: str):
    v = db.toggle_favorite(gid)
    if v is None:
        raise HTTPException(404, "not found")
    return {"favorite": v}


@app.delete("/api/generations/{gid}")
def delete_generation(gid: str):
    if "/" in gid or "\\" in gid or ".." in gid:
        raise HTTPException(400, "bad id")
    if not db.delete(gid):
        raise HTTPException(404, "not found")
    shutil.rmtree(STORE / gid, ignore_errors=True)
    return {"deleted": gid}


@app.get("/api/stats")
def stats():
    return db.stats(baseline_ms())


@app.get("/api/benchmarks")
def benchmarks():
    q, e = read_json("quality.json"), read_json("efficiency.json")
    comp, deg, abl = read_json("compositional.json"), read_json("degradation.json"), read_json("ablation.json")
    for v in comp.values():
        if isinstance(v, dict):
            v.pop("per_prompt", None)
    return {"quality": q, "efficiency": {k: v for k, v in e.items() if not k.startswith("_unet")},
            "compositional": comp, "degradation": deg, "ablation": abl}


@app.get("/")
def index():
    f = FRONTEND / "index.html"
    if not f.exists():
        return JSONResponse({"error": "frontend missing"}, status_code=500)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=422)
