"""Shared evaluation utilities: prompt sets, cached generation, CLIP / FID / KID."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pocketdiff.config import DATA_DIR, PRESETS, RESULTS_DIR, GenConfig  # noqa: E402

GEN_DIR = RESULTS_DIR / "gen"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def coco_test(n: int | None = None) -> list[dict]:
    rows = load_jsonl(DATA_DIR / "coco_test" / "prompts.jsonl")
    return rows[:n] if n else rows


def coco_calib(n: int | None = None) -> list[dict]:
    rows = load_jsonl(DATA_DIR / "coco_calib" / "prompts.jsonl")
    return rows[:n] if n else rows


def compositional() -> list[dict]:
    return json.loads((ROOT / "prompts" / "compositional.json").read_text(encoding="utf-8"))


def save_png(img01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img01, 0, 1) * 255).round().astype(np.uint8)).save(path)


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def update_json(path: Path, key: str, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    d[key] = value
    path.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ------------------------------------------------------------------------------ generation
class Runner:
    """Generates (and caches on disk) one image per prompt for a named system."""

    def __init__(self):
        from pocketdiff.pipeline import PocketDiffusion
        self.pd = PocketDiffusion()

    def run_set(self, system: str, rows: list[dict], subdir: str, seed_base: int = 0,
                cfg: GenConfig | None = None, pipeline_kwargs: dict | None = None,
                overwrite: bool = False) -> dict:
        """system: a preset name, 'pocketdiff' (full adaptive system) or a custom name with cfg."""
        out_dir = GEN_DIR / subdir / system
        meta_path = out_dir / "meta.json"
        meta = read_json(meta_path) if not overwrite else {}
        kw = dict(pipeline_kwargs or {})
        if cfg is None and system in PRESETS:
            cfg = PRESETS[system]
        todo = [r for r in rows if overwrite or not (out_dir / f"{r['id']}.png").exists()
                or r["id"] not in meta]
        if todo:
            print(f"[{system}] generating {len(todo)}/{len(rows)} images -> {out_dir}", flush=True)
        # warm-up so the first timed image does not include cuDNN autotuning / model loading
        if todo:
            self._call(todo[0]["prompt"], cfg, 0, kw)
        t_all = time.time()
        for i, r in enumerate(todo):
            seed = seed_base + int(abs(hash_id(r["id"])) % 100000)
            res = self._call(r["prompt"], cfg, seed, kw)
            save_png(res.image, out_dir / f"{r['id']}.png")
            meta[r["id"]] = {"latency_ms": res.latency_ms, "gen_ms": res.gen.latency_ms,
                             "peak_mem_mb": res.gen.peak_mem_mb, "weights_mb": res.gen.weights_mb,
                             "steps_done": res.gen.steps_done, "n": len(res.candidates),
                             "config": res.config.name, "seed": seed,
                             "plan": None if res.plan is None else {
                                 "family": res.config.family, "steps": res.config.steps,
                                 "res": res.config.height, "n": len(res.candidates),
                                 "quant": res.config.quant, "vae": res.config.vae}}
            if (i + 1) % 25 == 0:
                el = time.time() - t_all
                print(f"  {i + 1}/{len(todo)}  ({el / (i + 1):.2f} s/img)", flush=True)
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta

    def _call(self, prompt, cfg, seed, kw):
        if cfg is None:   # full adaptive system
            return self.pd.run(prompt, seed=seed, **kw)
        base = dict(n=1, postprocess="off", rerank=False, safety=False)
        base.update(kw)
        return self.pd.run(prompt, config=cfg, seed=seed, **base)


def hash_id(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


def load_set(subdir: str, system: str, rows: list[dict]) -> tuple[list[np.ndarray], list[str], dict]:
    out_dir = GEN_DIR / subdir / system
    meta = read_json(out_dir / "meta.json")
    imgs, prompts = [], []
    for r in rows:
        p = out_dir / f"{r['id']}.png"
        if p.exists():
            imgs.append(load_png(p))
            prompts.append(r["prompt"])
    return imgs, prompts, meta


# ------------------------------------------------------------------------------ metrics
_EVAL_CLIP = None


def eval_clip():
    global _EVAL_CLIP
    if _EVAL_CLIP is None:
        from pocketdiff.scoring import ClipScorer
        from pocketdiff.config import MODEL_IDS
        _EVAL_CLIP = ClipScorer(MODEL_IDS["clip_eval"])
    return _EVAL_CLIP


def clip_scores(imgs, prompts) -> np.ndarray:
    return eval_clip().clip_score(imgs, prompts)


def fid_kid(gen_imgs: list[np.ndarray], real_dir: Path, kid_subset: int = 100, ids: list[str] | None = None) -> dict:
    """FID and KID (Inception-v3 pool features, torchmetrics / torch-fidelity weights)."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    kid = KernelInceptionDistance(subset_size=min(kid_subset, len(gen_imgs)), normalize=True).to(dev)
    real_files = [real_dir / f"{i}.jpg" for i in ids] if ids else sorted(real_dir.glob("*.jpg"))
    real_files = [f for f in real_files if f.exists()]

    def _batches(arrs):
        for i in range(0, len(arrs), 32):
            x = torch.from_numpy(np.stack(arrs[i:i + 32])).permute(0, 3, 1, 2).float().to(dev)
            yield torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", antialias=True)

    reals = [load_png(f) for f in real_files]
    for x in _batches(reals):
        fid.update(x, real=True)
        kid.update(x, real=True)
    for x in _batches(gen_imgs):
        fid.update(x, real=False)
        kid.update(x, real=False)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_x1000": float(kid_mean) * 1000,
            "kid_std_x1000": float(kid_std) * 1000, "n_real": len(reals), "n_gen": len(gen_imgs)}


def latency_stats(meta: dict, key: str = "latency_ms") -> dict:
    v = np.array([m[key] for m in meta.values()]) if meta else np.zeros(1)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "std": float(v.std())}
