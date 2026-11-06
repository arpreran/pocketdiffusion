"""Build the evaluation / calibration / training splits from MS-COCO (Karpathy split).

Splits (no overlap - the planner is calibrated on *validation*, evaluated on *test*):
    data/coco_test/      prompts.jsonl + real/*.jpg   -> CLIP score, FID/KID (test split)
    data/coco_calib/     prompts.jsonl                -> planner calibration, degradation study
    data/coco_niqe/      real/*.jpg                    -> NIQE pristine model (validation split)
    data/coco_train_captions.txt                      -> text-encoder distillation (train split)

Usage: python scripts/prepare_data.py --n-test 300 --n-calib 100 --n-niqe 120
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_split(name: str) -> pd.DataFrame:
    path = Path(snapshot_download("yerevann/coco-karpathy", repo_type="dataset",
                                  allow_patterns=[f"data/{name}-*.parquet"]))
    files = sorted((path / "data").glob(f"{name}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _captions(row) -> list[str]:
    s = row["sentences"]
    if isinstance(s, str):
        return [s]
    return [str(x).strip() for x in list(s)]


def _center_square(img: Image.Image, size: int = 512) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.BICUBIC)


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            _center_square(img).save(dest, quality=95)
            return True
        except Exception:
            continue
    return False


def build_image_split(df: pd.DataFrame, n: int, out: Path, seed: int, with_images: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    rows = []
    for i in idx[: int(n * 1.15) + 5]:  # oversample: some downloads may fail
        r = df.iloc[i]
        caps = _captions(r)
        rows.append({"id": f"{int(r.get('cocoid', i)):012d}", "prompt": caps[0], "captions": caps,
                     "url": r.get("url", "")})
    ok_rows = rows
    if with_images:
        (out / "real").mkdir(exist_ok=True)
        with ThreadPoolExecutor(8) as ex:
            oks = list(ex.map(lambda r: _download(r["url"], out / "real" / f"{r['id']}.jpg"), rows))
        ok_rows = [r for r, ok in zip(rows, oks) if ok]
    ok_rows = ok_rows[:n]
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for r in ok_rows:
            f.write(json.dumps(r) + "\n")
    return len(ok_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--n-niqe", type=int, default=120)
    args = ap.parse_args()

    test = load_split("test")
    val = load_split("validation")
    print("columns:", list(test.columns), "| test rows:", len(test), "| val rows:", len(val))

    n = build_image_split(test, args.n_test, DATA / "coco_test", seed=0, with_images=True)
    print(f"coco_test: {n} prompt/image pairs")
    half = len(val) // 2
    n = build_image_split(val.iloc[:half], args.n_calib, DATA / "coco_calib", seed=1, with_images=False)
    print(f"coco_calib: {n} prompts")
    n = build_image_split(val.iloc[half:], args.n_niqe, DATA / "coco_niqe", seed=2, with_images=True)
    print(f"coco_niqe: {n} real images")

    train = load_split("train")
    caps = [c for _, r in train.iterrows() for c in _captions(r)]
    caps = sorted(set(c.replace("\n", " ") for c in caps if 3 <= len(c.split()) <= 60))
    (DATA / "coco_train_captions.txt").write_text("\n".join(caps), encoding="utf-8")
    val_caps = sorted(set(c for _, r in val.iloc[:half].iterrows() for c in _captions(r)))
    (DATA / "coco_val_captions.txt").write_text("\n".join(val_caps), encoding="utf-8")
    print(f"train captions: {len(caps)} | val captions: {len(val_caps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
