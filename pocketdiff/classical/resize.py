"""Resolution optimisation.

Diffusion cost grows ~linearly with the number of latent pixels (and attention grows
quadratically), so the cheapest way to save compute on a phone is to *generate small and
upscale smart*. This module provides:

* `snap_resolution`   - aspect-ratio-preserving size whose sides are multiples of 8 (latent
                        grid) under a pixel budget.
* `upscale`           - classical upscalers: nearest / bilinear / bicubic / lanczos / IBP.
* `ibp_upscale`       - Iterative Back-Projection super-resolution (Irani & Peleg 1991):
                        refine an interpolated image so that, when re-downsampled, it matches the
                        low-resolution input, followed by edge-aware sharpening.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .enhance import edge_aware_sharpen

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


def snap_resolution(width: int, height: int, max_pixels: int = 512 * 512, multiple: int = 8,
                    min_side: int = 128, max_side: int = 1024) -> tuple[int, int]:
    """Closest (w, h) with the same aspect ratio, sides multiple of `multiple`, w*h <= max_pixels."""
    # no per-side clipping before the aspect ratio is taken: 7680x4320 (8K) must stay 16:9
    width = float(max(width or 512, 1))
    height = float(max(height or 512, 1))
    aspect = width / height
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    w = width * scale
    h = height * scale
    w = int(np.clip(round(w / multiple) * multiple, min_side, max_side))
    h = int(np.clip(round(h / multiple) * multiple, min_side, max_side))
    while w * h > max_pixels and (w > min_side or h > min_side):  # rounding may overshoot
        if aspect >= 1:
            w -= multiple
        else:
            h -= multiple
    return w, h


def resize(img01: np.ndarray, size: tuple[int, int], method: str = "lanczos") -> np.ndarray:
    w, h = size
    if (img01.shape[1], img01.shape[0]) == (w, h):
        return img01
    if method == "ibp":
        return ibp_upscale(img01, size)
    down = w < img01.shape[1]
    interp = cv2.INTER_AREA if down else INTERP.get(method, cv2.INTER_LANCZOS4)
    return np.clip(cv2.resize(img01, (w, h), interpolation=interp), 0, 1).astype(np.float32)


def ibp_upscale(lr01: np.ndarray, size: tuple[int, int], iters: int = 4, step: float = 1.0,
                sharpen: float = 0.35) -> np.ndarray:
    """Iterative back-projection. x_{k+1} = x_k + step * U(y - D(x_k))."""
    w, h = size
    lr = lr01.astype(np.float32)
    x = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(iters):
        sim = cv2.resize(cv2.GaussianBlur(x, (0, 0), 0.5), (lr.shape[1], lr.shape[0]),
                         interpolation=cv2.INTER_AREA)
        err = lr - sim
        x = x + step * cv2.resize(err, (w, h), interpolation=cv2.INTER_CUBIC)
    x = np.clip(x, 0, 1)
    if sharpen > 0:
        x, _ = edge_aware_sharpen(x, amount=sharpen)
    return x.astype(np.float32)


def super_resolve(img01: np.ndarray, size: tuple[int, int],
                  max_ibp_pixels: int = 3840 * 2160) -> np.ndarray:
    """Very large upscales (e.g. 680x384 -> 8K 7680x4320, x11).

    IBP super-resolution with edge-aware sharpening up to ~4K, then Lanczos for the remaining
    factor. Keeps memory bounded: an 8K float image is ~400 MB and the guided filter inside the
    sharpening step needs about ten such buffers. No new detail is invented - the output is a
    sharper enlargement of the generated image.
    """
    w, h = size
    if w * h <= max_ibp_pixels:
        return ibp_upscale(img01, size)
    s = math.sqrt(max_ibp_pixels / float(w * h))
    mid = (max(8, int(round(w * s / 8)) * 8), max(8, int(round(h * s / 8)) * 8))
    x = ibp_upscale(img01, mid)
    return np.clip(cv2.resize(x, (w, h), interpolation=cv2.INTER_LANCZOS4), 0, 1).astype(np.float32)


def upscale_comparison(lr01: np.ndarray, size: tuple[int, int]) -> dict[str, np.ndarray]:
    """All upscalers side by side (used in the UI and in the resolution ablation)."""
    return {m: resize(lr01, size, m) for m in ("nearest", "bilinear", "bicubic", "lanczos", "ibp")}
