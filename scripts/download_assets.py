"""Download every pretrained asset PocketDiffusion needs (idempotent).

Only the files that are actually used are fetched (fp16 variants, no TF/Flax
weights, no safety-checker), which keeps the total download around 8 GB.

Usage:
    python scripts/download_assets.py            # everything
    python scripts/download_assets.py --core     # only what the app needs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SD_FP16 = [
    "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
]
CLIP_FILES = ["config.json", "model.safetensors", "*.txt", "*.json"]

# (repo_id, repo_type, allow_patterns, needed_by_app)
ASSETS = [
    ("madebyollin/taesd", "model", ["config.json", "diffusion_pytorch_model.safetensors"], True),
    ("latent-consistency/lcm-lora-sdv1-5", "model", ["pytorch_lora_weights.safetensors"], True),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", "model", SD_FP16, True),
    # this repo only ships pytorch_model.bin on main (no safetensors)
    ("openai/clip-vit-base-patch32", "model", ["config.json", "pytorch_model.bin", "*.txt", "*.json"], True),
    ("yerevann/coco-karpathy", "dataset", ["data/test-*.parquet", "data/validation-*.parquet",
                                           "data/train-*.parquet"], False),
    ("openai/clip-vit-large-patch14", "model", CLIP_FILES, False),
    ("google/owlv2-base-patch16-ensemble", "model", CLIP_FILES, False),
    # selection detector for semantic re-ranking (deliberately different from the OWLv2 evaluator)
    ("google/owlvit-base-patch32", "model", CLIP_FILES, True),
    ("stabilityai/sd-turbo", "model", SD_FP16, True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true", help="only assets used by the app")
    args = ap.parse_args()

    failed = []
    for repo, rtype, patterns, core in ASSETS:
        if args.core and not core:
            continue
        t0 = time.time()
        print(f"[download] {repo} ...", flush=True)
        try:
            path = snapshot_download(repo, repo_type=rtype, allow_patterns=patterns)
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            print(f"[done] {repo}: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s -> {path}", flush=True)
        except Exception as exc:  # keep going; report at the end
            print(f"[FAILED] {repo}: {exc}", flush=True)
            failed.append(repo)

    # FID Inception weights are fetched lazily by torch-fidelity on first use.
    if failed:
        print("Failed:", failed)
        return 1
    print("All assets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
