"""PocketDiffusion REST + streaming API (FastAPI).

Routes
------
GET  /api/system                 hardware, models, queue, device profiles, presets, examples
POST /api/analyze                prompt -> scene graph, complexity, token count   (CPU, instant)
POST /api/plan                   prompt + budget -> planner decision              (CPU, instant)
POST /api/generate               enqueue a generation job -> {job_id}
POST /api/compare                enqueue baseline-vs-optimised race -> {job_id}
POST /api/edit                   image-to-image edit of an upload or a stored image -> {job_id}
POST /api/dream                  morphing animation (seed walk or prompt morph) -> {job_id}
POST /api/preview                live preview while typing (1 step, not stored)
GET  /api/jobs/{id}/events       Server-Sent Events: queued/start/analysis/plan/step/stage/result/error/end
GET  /api/generations            history (paginated, filter: favorites, family, q)
GET  /api/generations/{id}       full record (candidates, stages, frames, metrics)
POST /api/generations/{id}/favorite
POST /api/generations/{id}/attention   cross-attention heat map per prompt word ("Explain")
DELETE /api/generations/{id}
GET  /api/stats                  aggregate statistics from the database
GET  /api/benchmarks             evaluation results (results/*.json) for the Lab charts
GET  /share/{id}                 public share page for one image (Open Graph preview)
GET  /manifest.webmanifest, /sw.js   installable app (PWA)
GET  /media/...                  stored images;  GET /  -> single-page frontend
"""
from __future__ import annotations

import asyncio
import collections
import hmac
import html
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from pocketdiff.config import DATA_DIR, DEVICE_PROFILES, MODEL_IDS, PRESETS, RESULTS_DIR, ROOT
from pocketdiff.planner import Budget, Planner, family_available
from pocketdiff.preload import import_ml_libraries
from pocketdiff.prompt_analyzer import analyze_prompt
from pocketdiff.utils import LOG

from .db import Database
from .jobs import JobManager

# before the GPU worker and the tokenizer-preload thread exist: their concurrent lazy imports of
# transformers/diffusers could otherwise fail ("cannot import name 'AutoModel' from 'transformers'")
import_ml_libraries()

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
# read-only cross-origin GETs, so the permanent status page (GitHub Pages) can check that this
# server is up before forwarding visitors to it; writes (POST/DELETE) stay same-origin only
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=[])

# ------------------------------------------------------------------ visitors, team access, rate limits
# The public link is open to anyone, so every browser gets an anonymous visitor id (cookie). A
# visitor's gallery shows only their own images, and only they can favourite or delete them. The
# team is recognised as a request made on the laptop itself (http://127.0.0.1:8000 - through the
# tunnel the Host header is the public *.gradio.live name), or with POCKETDIFF_ADMIN_KEY.
VID_COOKIE = "pd_vid"
_VID = re.compile(r"[0-9a-f]{32}")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def is_admin(request: Request) -> bool:
    key = os.environ.get("POCKETDIFF_ADMIN_KEY", "")
    if key and hmac.compare_digest(request.headers.get("x-admin-key", ""), key):
        return True
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
    client = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for") or request.headers.get("x-forwarded-host")
    return client in LOCAL_HOSTS and host in LOCAL_HOSTS and not forwarded


@app.middleware("http")
async def visitor_identity(request: Request, call_next):
    vid = request.cookies.get(VID_COOKIE, "")
    new = not _VID.fullmatch(vid)
    if new:
        vid = secrets.token_hex(16)
    request.state.vid = vid
    request.state.admin = is_admin(request)
    response = await call_next(request)
    if new:
        host = request.headers.get("host", "")
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https" \
            or host.endswith(".live")
        response.set_cookie(VID_COOKIE, vid, max_age=365 * 86400, httponly=True, samesite="lax", secure=secure)
    return response


class RateLimiter:
    """Sliding-window limit per visitor (the team is exempt)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 5000:                       # forget idle visitors
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 120][:2500]:
                    self.hits.pop(k, None)
            return True


JOB_LIMIT = RateLimiter(int(os.environ.get("POCKETDIFF_JOBS_PER_MIN", "12")))
PREVIEW_LIMIT = RateLimiter(90)                 # live previews: about one per keystroke pause
_last_preview: dict[str, object] = {}
MAX_PENDING_PER_VISITOR = 2


def admit_job(request: Request) -> str:
    """Owner id for a new GPU job; 429 when a visitor sends too many."""
    vid = request.state.vid
    if request.state.admin:
        return vid
    if jobs.pending_for(vid) >= MAX_PENDING_PER_VISITOR:
        raise HTTPException(429, "You already have images in the queue - wait for them to finish.")
    if not JOB_LIMIT.allow(vid):
        raise HTTPException(429, "Too many images in a minute from this browser - please wait a moment.")
    return vid


def can_modify(request: Request, owner: str | None) -> bool:
    return bool(request.state.admin or (owner and owner == request.state.vid))


def public_view(g: dict, request: Request) -> dict:
    """A generation as the API returns it: the owner id is never sent, only whether it is yours."""
    g = dict(g)
    g["mine"] = can_modify(request, g.pop("owner", None))
    return g
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


class EditIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    strength: float = Field(0.55, ge=0.1, le=1.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    source_id: str | None = Field(None, max_length=64)
    image: str | None = Field(None, max_length=14_000_000)      # data URL, ~10 MB file

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.source_id) == bool(self.image):
            raise ValueError("send exactly one of source_id (a stored image) or image (an upload)")
        if self.source_id and not GID.fullmatch(self.source_id):
            raise ValueError("bad source_id")
        return self


class DreamIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    prompt_b: str = Field("", max_length=2000)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    frames: int = Field(24, ge=8, le=48)


class PreviewIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class CompareIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int = Field(7, ge=0, le=2**31 - 1)


# ------------------------------------------------------------------ helpers
GID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")


def check_gid(gid: str) -> str:
    if not GID.fullmatch(gid or ""):
        raise HTTPException(400, "bad id")
    return gid


def public_page() -> str:
    """Permanent address that forwards to the current live link (set by serve.py --publish-link)."""
    return os.environ.get("POCKETDIFF_PUBLIC_PAGE", "")


async def wait_job(job, timeout_s: float):
    """Await a worker job from an async route; returns its 'result' event data."""
    t_end = time.time() + timeout_s
    while job.status not in ("done", "error"):
        if time.time() > t_end:
            raise HTTPException(504, "The GPU worker is busy - please try again in a moment.")
        await asyncio.sleep(0.1)
    with job.lock:
        evs = list(job.events)
    if job.status == "error":
        msg = next((e["data"].get("message") for e in reversed(evs) if e["event"] == "error"), "failed")
        raise HTTPException(422, msg)
    return next((e["data"] for e in reversed(evs) if e["event"] == "result"), None)


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
def system(request: Request):
    import torch
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "vram_mb": props.total_memory / 2**20}
    pd = jobs.pd
    return {
        "app": "PocketDiffusion", "version": "1.0.0", "gpu": gpu, "torch": torch.__version__,
        "ready": jobs.ready.is_set(), "queue": jobs.queue_length(), "warmup_error": jobs.warmup_error,
        "loaded_models": pd.engine.models.loaded() if pd else [],
        "families": {f: family_available(f) for f in ("sd15", "sd_turbo")},
        "device_profiles": [{"key": k, "label": p.label, "kind": p.kind, "memory_budget_mb": p.memory_budget_mb,
                             "eff_tflops": p.eff_tflops, "notes": p.notes} for k, p in DEVICE_PROFILES.items()],
        "presets": [{"key": k, "description": c.description} for k, c in PRESETS.items() if k != "cpu_fallback"],
        "examples": EXAMPLES, "baseline_ms": baseline_ms(),
        # cloud backup (Hugging Face Space, CPU only): the UI says so and points to the GPU version
        "cloud": bool(os.environ.get("SPACE_ID")), "gpu_link": os.environ.get("POCKETDIFF_GPU_LINK", ""),
        "default_device": "this_gpu" if gpu else "this_cpu",
        "public_page": public_page(),
        "admin": bool(request.state.admin),
        "features": ["generate", "edit", "attention", "share", "pwa", "compare"] + (["preview", "dream"] if gpu else []),
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
def generate(body: GenerateIn, request: Request):
    if not body.prompt.strip():
        raise HTTPException(422, "Please enter a prompt describing the image.")
    job = jobs.submit("generate", {**body.model_dump(), "owner": admit_job(request)})
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.post("/api/compare")
def compare(body: CompareIn, request: Request):
    from pocketdiff.utils import cuda_available
    if not family_available("sd15") or not cuda_available():
        raise HTTPException(503, "The race runs the SD1.5 baseline (50 steps, FP32), which needs a GPU - "
                                 "this server has none and it would take minutes per image. "
                                 "Use the GPU version of the app for the race.")
    job = jobs.submit("compare", {**body.model_dump(), "owner": admit_job(request)})
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.post("/api/edit")
def edit(body: EditIn, request: Request):
    if body.source_id and not db.owner_of(body.source_id)[0]:
        raise HTTPException(404, "source image not found")
    job = jobs.submit("edit", {**body.model_dump(), "owner": admit_job(request)})
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.post("/api/dream")
def dream(body: DreamIn, request: Request):
    """Dream mode: a morphing animation (seed walk, or prompt A -> prompt B) from 1-step frames."""
    from pocketdiff.utils import cuda_available
    if not cuda_available():
        raise HTTPException(503, "Dream mode needs the GPU server (one generation per frame).")
    job = jobs.submit("dream", {**body.model_dump(), "owner": admit_job(request)})
    return {"job_id": job.id, "queue": jobs.queue_length()}


@app.post("/api/preview")
async def preview(body: PreviewIn, request: Request):
    """Live preview while typing: one fast SD-Turbo step, not stored. A newer request from the same
    browser replaces an older one that has not started; while a full job runs the answer is busy."""
    from pocketdiff.utils import cuda_available
    if not cuda_available():
        raise HTTPException(503, "Live preview needs the GPU server.")
    if not jobs.ready.is_set():
        raise HTTPException(503, "The models are still loading.")
    vid = request.state.vid
    if not request.state.admin and not PREVIEW_LIMIT.allow(vid):
        raise HTTPException(429, "Live preview is updating too often - type a little slower.")
    if jobs.busy_with_full_job() or jobs.q.qsize() > 2:
        return {"busy": True}
    old = _last_preview.get(vid)
    if old is not None and old.status == "queued":
        old.cancelled = True
    job = jobs.submit("preview", {"prompt": body.prompt, "seed": body.seed, "owner": vid})
    _last_preview[vid] = job
    try:
        return await wait_job(job, timeout_s=20) or {"superseded": True}
    finally:
        jobs.jobs.pop(job.id, None)              # previews carry an image: do not keep them around
        if _last_preview.get(vid) is job:
            _last_preview.pop(vid, None)


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
def list_generations(request: Request, limit: int = 24, offset: int = 0, favorites: bool = False,
                     family: str | None = None, q: str | None = None, mine: bool = False):
    everyone = bool(request.state.admin) and not mine
    rows, total = db.list(limit=max(1, min(limit, 100)), offset=max(0, offset), favorites=favorites,
                          family=family or None, q=q or None, owner=request.state.vid, everyone=everyone)
    return {"items": [public_view(r, request) for r in rows], "total": total, "everyone": everyone}


@app.get("/api/generations/{gid}")
def get_generation(gid: str, request: Request):
    g = db.get(gid)                       # readable by id (share links); not listed for others
    if g is None:
        raise HTTPException(404, "not found")
    return public_view(g, request)


@app.post("/api/generations/{gid}/attention")
async def attention(gid: str, request: Request):
    """Explain: which image regions each prompt word influenced (cross-attention, cached)."""
    check_gid(gid)
    cached = STORE / gid / "attn" / "attention.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    g = db.get(gid)
    if g is None:
        raise HTTPException(404, "not found")
    if g.get("kind") == "edit":
        raise HTTPException(422, "Explain works for text-to-image results; this image was edited from a picture.")
    if not jobs.ready.is_set():
        raise HTTPException(503, "The models are still loading - try again in a few seconds.")
    return await wait_job(jobs.submit("attention", {"gid": gid, "owner": admit_job(request)}), timeout_s=240)


@app.post("/api/generations/{gid}/favorite")
def favorite(gid: str, request: Request):
    exists, owner = db.owner_of(gid)
    if not exists:
        raise HTTPException(404, "not found")
    if not can_modify(request, owner):
        raise HTTPException(403, "Only the person who made this image can change it.")
    v = db.toggle_favorite(gid)
    if v is None:
        raise HTTPException(404, "not found")
    return {"favorite": v}


@app.delete("/api/generations/{gid}")
def delete_generation(gid: str, request: Request):
    if "/" in gid or "\\" in gid or ".." in gid:
        raise HTTPException(400, "bad id")
    exists, owner = db.owner_of(gid)
    if not exists:
        raise HTTPException(404, "not found")
    if not can_modify(request, owner):
        raise HTTPException(403, "Only the person who made this image can delete it.")
    if not db.delete(gid):
        raise HTTPException(404, "not found")
    shutil.rmtree(STORE / gid, ignore_errors=True)
    return {"deleted": gid}


@app.get("/api/stats")
def stats():
    return db.stats(baseline_ms())


@app.get("/api/benchmarks")
def benchmarks():
    q, e, en = read_json("quality.json"), read_json("efficiency.json"), read_json("energy.json")
    comp, deg, abl = read_json("compositional.json"), read_json("degradation.json"), read_json("ablation.json")
    for v in comp.values():
        if isinstance(v, dict):
            v.pop("per_prompt", None)
    return {"quality": q, "efficiency": {k: v for k, v in e.items() if not k.startswith("_unet")},
            "compositional": comp, "degradation": deg, "ablation": abl,
            "energy": {k: v for k, v in en.items() if not k.startswith("_")}}


SHARE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - PocketDiffusion</title>
<meta name="description" content="{desc}">
<meta property="og:type" content="website"><meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}"><meta property="og:image" content="{og_image}">
<meta name="twitter:card" content="summary_large_image"><meta name="theme-color" content="#0b0b14">
<link rel="icon" href="/static/icons/icon-192.png">
<style>
:root{{--bg:#0b0b14;--card:#151524;--text:#ecebf6;--muted:#a3a2b8;--a1:#8b5cf6;--a2:#06b6d4;--line:#2a2a40}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f5fb;--card:#fff;--text:#16151f;--muted:#5d5c70;--line:#e3e1ee}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:24px 16px 48px}}
.brand{{font-weight:700;letter-spacing:-.01em;color:var(--text);text-decoration:none}}.brand b{{color:var(--a1)}}
figure{{margin:16px 0;background:var(--card);border:1px solid var(--line);border-radius:18px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto}}
figcaption{{padding:14px 16px}}figcaption p{{margin:0 0 10px;font-size:17px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px}}.chips span{{font:12px ui-monospace,monospace;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:3px 9px}}
.actions{{display:flex;flex-wrap:wrap;gap:10px}}
.btn{{display:inline-block;padding:10px 16px;border-radius:12px;border:1px solid var(--line);color:var(--text);text-decoration:none;font-weight:600}}
.btn.primary{{background:linear-gradient(135deg,var(--a1),var(--a2));border:0;color:#fff}}
</style></head><body><main>
<a class="brand" href="/">Pocket<b>Diffusion</b></a>
<figure><img src="{image}" alt="{alt}" width="{w}" height="{h}">
<figcaption><p>{prompt}</p><div class="chips">{chips}</div></figcaption></figure>
<div class="actions"><a class="btn primary" href="/?remix={gid}#studio">Remix this prompt</a>
<a class="btn" href="/#studio">Make your own</a><a class="btn" href="{download}" download>Download {w}&times;{h}</a></div>
</main></body></html>"""


def _origin(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or ("https" if host.endswith(".live") else request.url.scheme)
    return f"{proto}://{host}" if host else str(request.base_url).rstrip("/")


@app.get("/share/{gid}", response_class=HTMLResponse)
def share_page(gid: str, request: Request):
    check_gid(gid)
    g = db.get(gid)
    if g is None:
        return HTMLResponse("<!doctype html><meta charset=utf-8><title>Not found</title>"
                            "<p style='font-family:system-ui;padding:24px'>This image was deleted. "
                            "<a href='/'>Open PocketDiffusion</a></p>", status_code=404)
    e = html.escape
    fam = {"sd_turbo": "SD-Turbo", "sd15": "SD1.5"}.get(g["family"], g["family"])
    chips = [f"{fam} &middot; {g['steps_done']}/{g['steps']} steps", f"{g['width']}&times;{g['height']}",
             f"{g['latency_ms'] / 1000:.2f} s" if g.get("latency_ms") else "",
             f"CLIP {g['clip']:.1f}" if g.get("clip") is not None else "", "edited" if g.get("kind") == "edit" else ""]
    prompt = g["prompt"]
    title = prompt if len(prompt) <= 70 else prompt[:67].rsplit(" ", 1)[0] + "..."
    return SHARE_HTML.format(
        title=e(title), desc=e(f"Generated in {g['latency_ms'] / 1000:.1f} s on a laptop GPU with PocketDiffusion"),
        og_image=e(f"{_origin(request)}/media/{g['image_path']}"), image=e(f"/media/{g['image_path']}"),
        download=e(f"/media/{g.get('full_path') or g['image_path']}"), alt=e(prompt), prompt=e(prompt),
        chips="".join(f"<span>{c}</span>" for c in chips if c), gid=e(gid), w=g["width"], h=g["height"])


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(FRONTEND / "manifest.webmanifest", media_type="application/manifest+json",
                        headers={"Cache-Control": "no-cache"})


@app.get("/sw.js")
def service_worker():
    # served from the root so the worker's scope is the whole app
    return FileResponse(FRONTEND / "sw.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/")
def index():
    f = FRONTEND / "index.html"
    if not f.exists():
        return JSONResponse({"error": "frontend missing"}, status_code=500)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=422)
