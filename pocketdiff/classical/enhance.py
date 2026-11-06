"""Colour correction and enhancement operators (all take/return float32 RGB in [0,1])."""
from __future__ import annotations

import cv2
import numpy as np

from .filters import gradient_magnitude, guided_filter_gray, luminance


def shades_of_gray_wb(img: np.ndarray, p: float = 6.0, strength: float = 1.0) -> np.ndarray:
    """Shades-of-Gray white balance (Finlayson & Trezzi 2004); p=1 is Gray-World, p=inf is Max-RGB.

    `strength` in [0,1] blends between the input and the fully corrected image, so a mild cast
    only gets a mild correction (adaptive).
    """
    x = np.clip(img, 1e-4, 1.0)
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    gain = illum.mean() / (illum + 1e-6)
    gain = 1.0 + strength * (gain - 1.0)
    return np.clip(img * gain[None, None, :], 0.0, 1.0).astype(np.float32)


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel only (no hue shift)."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def adaptive_gamma(img: np.ndarray, target_mean: float = 0.46, max_strength: float = 0.6) -> np.ndarray:
    """Gamma chosen so the mean luminance moves towards `target_mean` (bounded correction)."""
    m = float(np.clip(luminance(img).mean(), 1e-3, 0.999))
    gamma = np.log(target_mean) / np.log(m)
    gamma = float(np.clip(gamma, 1.0 - max_strength, 1.0 + max_strength))
    return np.power(np.clip(img, 0, 1), gamma).astype(np.float32)


def saturation_soft_knee(img: np.ndarray, knee: float = 0.85, target_max: float = 0.95) -> np.ndarray:
    """Compress only the most saturated pixels (fixes 'neon' over-saturation of distilled models)."""
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    over = s > knee
    s_out = s.copy()
    s_out[over] = knee + (target_max - knee) * np.tanh((s[over] - knee) / (1.0 - knee + 1e-6))
    hsv[..., 1] = s_out
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)


def vibrance(img: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """Boost saturation of *muted* colours more than already-saturated ones."""
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    hsv[..., 1] = np.clip(s + amount * (1.0 - s) * s * 2.0, 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)


def edge_aware_denoise(img: np.ndarray, sigma_noise: float, r: int = 2) -> np.ndarray:
    """Guided-filter denoising whose eps is tied to the estimated noise level."""
    eps = float(np.clip((sigma_noise / 255.0) ** 2 * 4.0, 1e-5, 4e-3))
    guide = luminance(img)
    return np.clip(guided_filter_gray(guide, img, r, eps), 0, 1).astype(np.float32)


def edge_aware_sharpen(img: np.ndarray, amount: float = 0.6, r: int = 2, eps: float = 1e-3,
                       noise_floor: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Halo-free sharpening by base/detail decomposition.

    base   = guided_filter(img)          (edge-preserving, so no halo at strong edges)
    detail = img - base
    out    = img + amount * w * detail,  w = edge mask that is ~0 in flat (noisy) regions.

    Returns (sharpened image, edge weight map for visualisation).
    """
    gray = luminance(img)
    base = guided_filter_gray(gray, img, r, eps)
    detail = img - base
    g = gradient_magnitude(gray)
    g = g / (np.percentile(g, 99) + 1e-6)
    w = np.clip((g - noise_floor) / (0.25 - noise_floor), 0.0, 1.0)
    w = cv2.GaussianBlur(w, (0, 0), 1.0)
    out = img + amount * w[..., None] * detail
    return np.clip(out, 0, 1).astype(np.float32), w.astype(np.float32)


def local_contrast(img: np.ndarray, amount: float = 0.3, r: int = 16, eps: float = 1e-2) -> np.ndarray:
    """Large-radius detail boost (clarity). The guided filter keeps edges from haloing."""
    gray = luminance(img)
    base = guided_filter_gray(gray, gray, r, eps)
    detail = gray - base
    gain = (gray + amount * detail + 1e-4) / (gray + 1e-4)
    return np.clip(img * gain[..., None], 0, 1).astype(np.float32)
