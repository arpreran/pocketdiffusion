"""Edge-aware filters implemented from first principles.

* Guided filter (He, Sun, Tang - TPAMI 2013): O(N) edge-preserving smoothing built from box
  filters. Used for denoising, base/detail decomposition and halo-free sharpening.
* Bilateral filter wrapper (OpenCV) for comparison.
* Frequency-domain helpers for periodic (grid / checkerboard) artefact analysis.
"""
from __future__ import annotations

import cv2
import numpy as np


def box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window with reflected borders."""
    return cv2.boxFilter(x, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter_gray(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter with a single-channel guide. guide, src: float32 HxW (or HxWxC src)."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    if src.ndim == 3:
        return np.stack([guided_filter_gray(guide, src[..., c], r, eps)
                         for c in range(src.shape[2])], axis=-1)
    mean_i = box(guide, r)
    mean_p = box(src, r)
    corr_ip = box(guide * src, r)
    corr_ii = box(guide * guide, r)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)           # local linear model  q = a*I + b
    b = mean_p - a * mean_i
    return box(a, r) * guide + box(b, r)


def guided_filter_color(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter with an RGB guide (better edge preservation than a gray guide).

    guide: HxWx3 float32 in [0,1]; src: HxW or HxWxC float32.
    """
    if src.ndim == 3:
        return np.stack([guided_filter_color(guide, src[..., c], r, eps)
                         for c in range(src.shape[2])], axis=-1)
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    h, w, _ = g.shape
    mean_g = [box(g[..., k], r) for k in range(3)]
    mean_p = box(p, r)
    cov_gp = np.stack([box(g[..., k] * p, r) - mean_g[k] * mean_p for k in range(3)], axis=-1)
    # 3x3 covariance of the guide in each window
    sigma = np.empty((h, w, 3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            v = box(g[..., a] * g[..., b], r) - mean_g[a] * mean_g[b]
            sigma[..., a, b] = v
            sigma[..., b, a] = v
    sigma += eps * np.eye(3, dtype=np.float32)
    coef = np.linalg.solve(sigma, cov_gp[..., None])[..., 0]        # HxWx3
    bias = mean_p - sum(coef[..., k] * mean_g[k] for k in range(3))
    q = sum(box(coef[..., k], r) * g[..., k] for k in range(3)) + box(bias, r)
    return q


def bilateral(img01: np.ndarray, d: int = 7, sigma_color: float = 0.08,
              sigma_space: float = 5.0) -> np.ndarray:
    return cv2.bilateralFilter(img01.astype(np.float32), d, sigma_color, sigma_space)


def luminance(img01: np.ndarray) -> np.ndarray:
    return (0.299 * img01[..., 0] + 0.587 * img01[..., 1] + 0.114 * img01[..., 2]).astype(np.float32)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def log_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centered log-magnitude spectrum, normalised to [0,1] for visualisation."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.log1p(np.abs(f))
    return (mag - mag.min()) / (np.ptp(mag) + 1e-8)


def periodic_peak_score(gray: np.ndarray, period: int = 8) -> tuple[float, np.ndarray]:
    """Energy at the harmonics of `period` relative to their neighbourhood.

    Latent-diffusion decoders upsample 8x; a faulty decode leaves a faint 8-px grid, which
    shows up as isolated spectral peaks at k*N/8. Returns (score, peak mask in fft-shifted
    coordinates). A score around 1 means "no grid"; >~3 indicates visible periodic artefacts.
    """
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f = np.abs(np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win)))
    cy, cx = h // 2, w // 2
    mask = np.zeros_like(f, dtype=bool)
    ratios = []
    for ky in range(-(period // 2), period // 2 + 1):
        for kx in range(-(period // 2), period // 2 + 1):
            if ky == 0 and kx == 0:
                continue
            y = cy + int(round(ky * h / period))
            x = cx + int(round(kx * w / period))
            if not (2 <= y < h - 2 and 2 <= x < w - 2):
                continue
            peak = f[y - 1:y + 2, x - 1:x + 2].max()
            ring = np.concatenate([f[y - 4:y - 2, x - 4:x + 5].ravel(), f[y + 3:y + 5, x - 4:x + 5].ravel()])
            ratios.append(peak / (np.median(ring) + 1e-6))
            mask[y - 1:y + 2, x - 1:x + 2] = True
    score = float(np.median(ratios)) if ratios else 1.0
    return score, mask


def notch_filter(gray_or_rgb: np.ndarray, mask_shifted: np.ndarray) -> np.ndarray:
    """Suppress the given spectral peaks (smooth Gaussian notches) channel by channel."""
    soft = cv2.GaussianBlur(mask_shifted.astype(np.float32), (0, 0), 1.5)
    soft = soft / (soft.max() + 1e-8)
    keep = 1.0 - 0.9 * soft
    keep = np.fft.ifftshift(keep)

    def _one(ch):
        f = np.fft.fft2(ch)
        f[0, 0] = f[0, 0]  # keep DC
        return np.real(np.fft.ifft2(f * keep)).astype(np.float32)

    if gray_or_rgb.ndim == 2:
        return _one(gray_or_rgb)
    return np.stack([_one(gray_or_rgb[..., c]) for c in range(gray_or_rgb.shape[2])], axis=-1)
