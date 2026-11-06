"""Fit the NIQE 'pristine' multivariate-Gaussian model on real COCO photographs.

Usage: python scripts/fit_niqe.py   (needs data/coco_niqe/real from prepare_data.py)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pocketdiff.classical.iqa import fit_pristine  # noqa: E402
from pocketdiff.config import DATA_DIR, NIQE_PARAMS_FILE  # noqa: E402


def main() -> int:
    files = sorted((DATA_DIR / "coco_niqe" / "real").glob("*.jpg"))
    if not files:
        print("No images in data/coco_niqe/real - run scripts/prepare_data.py first")
        return 1
    t0 = time.time()
    imgs = [np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0 for f in files]
    model = fit_pristine(imgs)
    NIQE_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NIQE_PARAMS_FILE, mu=model["mu"], cov=model["cov"])
    print(f"NIQE pristine model from {len(files)} images / {model['n_patches']} sharp patches "
          f"-> {NIQE_PARAMS_FILE} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
