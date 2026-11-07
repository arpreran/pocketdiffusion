"""Upload the app to a Hugging Face Space (always-on CPU backup of the live demo).

Needs a one-time login on this computer:  hf auth login   (the token stays in the HF cache;
this script never reads or prints it - huggingface_hub uses it internally).

    python scripts/deploy_space.py                  # -> https://huggingface.co/spaces/<you>/pocketdiffusion
    python scripts/deploy_space.py --name my-space

The Space builds deploy/space/Dockerfile (CPU PyTorch, models baked in) and serves the same
FastAPI + web UI on port 7860. Data, generated evaluation images and local weights are not uploaded.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ROOT / "deploy" / "space"
SKIP_DIRS = {".git", ".claude", "__pycache__", "data", "outputs", ".pytest_cache", "deploy"}
SKIP_PREFIXES = ("results/gen/", "results/smoke/")
SKIP_GLOBS = ("*.pyc", "*.safetensors", "*.bin", "*.ckpt", "*.log", ".env")


def files_to_upload() -> list[tuple[str, Path]]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if set(p.relative_to(ROOT).parts[:-1]) & SKIP_DIRS or rel.startswith(SKIP_PREFIXES):
            continue
        if any(fnmatch.fnmatch(p.name, g) for g in SKIP_GLOBS) or rel in ("README.md", "Dockerfile"):
            continue
        out.append((rel, p))
    # the Space's own README (front matter: sdk docker, port 7860) and CPU Dockerfile
    out += [("README.md", SPACE_FILES / "README.md"), ("Dockerfile", SPACE_FILES / "Dockerfile")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pocketdiffusion")
    args = ap.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in to Hugging Face. Run:  hf auth login")
        return 1
    repo_id = f"{user}/{args.name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    files = files_to_upload()
    size = sum(p.stat().st_size for _, p in files) / 1e6
    print(f"uploading {len(files)} files ({size:.1f} MB) to spaces/{repo_id} ...", flush=True)
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)) for rel, p in files]
    api.create_commit(repo_id, operations=ops, repo_type="space",
                      commit_message="Deploy PocketDiffusion (CPU backup)")
    print(f"done. Build log and app: https://huggingface.co/spaces/{repo_id}")
    print(f"direct app URL:          https://{user.lower()}-{args.name.lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
