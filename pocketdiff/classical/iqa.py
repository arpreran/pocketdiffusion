"""No-reference image-quality assessment (classical).

* Simple, interpretable cues: sharpness, RMS contrast, colourfulness, exposure, noise, entropy.
* NIQE (Mittal, Soundararajan, Bovik - IEEE SPL 2013) implemented from scratch: natural-scene
  statistics of MSCN coefficients -> multivariate Gaussian distance to a "pristine" model.
  Our pristine model is fitted on real COCO photographs by `scripts/fit_niqe.py`, so absolute
  values are not comparable with MATLAB NIQE, but relative comparisons between systems are.
* PocketIQA: a 0-100 composite used for re-ranking and for before/after post-processing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.special import gamma as G

from .filters import luminance

# ------------------------------------------------------------------ simple cues


def sharpness(gray01: np.ndarray) -> float:
    """Variance of the Laplacian (on 0-255 scale)."""
    return float(cv2.Laplacian((gray01 * 255).astype(np.float32), cv2.CV_32F).var())


def rms_contrast(gray01: np.ndarray) -> float:
    return float(gray01.std())


def colorfulness(img01: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric (0 = gray, >80 = extremely colourful)."""
    r, g, b = (img01[..., c] * 255 for c in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def noise_sigma(gray01: np.ndarray) -> float:
    """Immerkaer (1996) fast noise-variance estimate, sigma on the 0-255 scale."""
    g = (gray01 * 255).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, cv2.CV_32F, k)
    h, w = g.shape
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


def entropy(gray01: np.ndarray) -> float:
    hist = np.bincount((gray01 * 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def exposure_stats(gray01: np.ndarray) -> dict:
    return {
        "mean_luma": float(gray01.mean()),
        "clip_high": float((gray01 > 0.98).mean()),
        "clip_low": float((gray01 < 0.02).mean()),
    }


# ------------------------------------------------------------------ NIQE from scratch
_GAM = np.arange(0.2, 10.0, 0.001)
_R_GGD = G(1.0 / _GAM) * G(3.0 / _GAM) / G(2.0 / _GAM) ** 2
_R_AGGD = G(2.0 / _GAM) ** 2 / (G(1.0 / _GAM) * G(3.0 / _GAM))


def _gauss_window(size: int = 7, sigma: float = 7.0 / 6.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


_WIN = _gauss_window()


def mscn(gray255: np.ndarray) -> np.ndarray:
    """Mean-subtracted contrast-normalised coefficients."""
    mu = cv2.filter2D(gray255, cv2.CV_32F, _WIN, borderType=cv2.BORDER_REPLICATE)
    var = cv2.filter2D(gray255 * gray255, cv2.CV_32F, _WIN, borderType=cv2.BORDER_REPLICATE) - mu * mu
    sigma = np.sqrt(np.abs(var))
    return (gray255 - mu) / (sigma + 1.0), sigma


def _ggd_fit(x: np.ndarray) -> tuple[float, float]:
    sig2 = np.mean(x * x)
    e = np.mean(np.abs(x))
    rho = sig2 / (e * e + 1e-12)
    alpha = _GAM[np.argmin(np.abs(rho - _R_GGD))]
    return float(alpha), float(sig2)


def _aggd_fit(x: np.ndarray) -> tuple[float, float, float, float]:
    left = x[x < 0]
    right = x[x >= 0]
    lstd = np.sqrt(np.mean(left * left)) if left.size else 1e-6
    rstd = np.sqrt(np.mean(right * right)) if right.size else 1e-6
    ghat = lstd / (rstd + 1e-12)
    rhat = (np.mean(np.abs(x)) ** 2) / (np.mean(x * x) + 1e-12)
    rhatn = rhat * (ghat ** 3 + 1) * (ghat + 1) / ((ghat ** 2 + 1) ** 2)
    alpha = _GAM[np.argmin((_R_AGGD - rhatn) ** 2)]
    bl = lstd * np.sqrt(G(1 / alpha) / G(3 / alpha))
    br = rstd * np.sqrt(G(1 / alpha) / G(3 / alpha))
    mean = (br - bl) * G(2 / alpha) / G(1 / alpha)
    return float(alpha), float(mean), float(lstd ** 2), float(rstd ** 2)


def _patch_features(m: np.ndarray) -> list[float]:
    feats = list(_ggd_fit(m.ravel()))
    pairs = [
        (m[:, :-1], m[:, 1:]),        # horizontal neighbour
        (m[:-1, :], m[1:, :]),        # vertical
        (m[:-1, :-1], m[1:, 1:]),     # main diagonal
        (m[:-1, 1:], m[1:, :-1]),     # anti-diagonal
    ]
    for a, b in pairs:
        feats.extend(_aggd_fit((a * b).ravel()))
    return feats


def niqe_features(gray01: np.ndarray, patch: int = 96, select_sharp: float | None = None) -> np.ndarray:
    """36-D NSS features per patch at two scales. If select_sharp is given (e.g. 0.75), only
    patches whose sharpness exceeds that fraction of the max are kept (pristine-model fitting)."""
    img = (gray01 * 255.0).astype(np.float32)
    h, w = img.shape
    h, w = (h // patch) * patch, (w // patch) * patch
    if h == 0 or w == 0:
        raise ValueError("Image smaller than one NIQE patch.")
    img = img[:h, :w]
    feats_scales, sharp = [], None
    for scale in (1, 2):
        if scale == 2:
            img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        m, sig = mscn(img)
        p = patch // scale
        rows = []
        sh = []
        for y in range(0, img.shape[0] - p + 1, p):
            for x in range(0, img.shape[1] - p + 1, p):
                rows.append(_patch_features(m[y:y + p, x:x + p]))
                sh.append(sig[y:y + p, x:x + p].mean())
        feats_scales.append(np.asarray(rows, np.float64))
        if scale == 1:
            sharp = np.asarray(sh)
    feats = np.concatenate(feats_scales, axis=1)
    if select_sharp is not None and sharp is not None and sharp.max() > 0:
        feats = feats[sharp > select_sharp * sharp.max()]
    return feats


def fit_pristine(images01: list[np.ndarray]) -> dict:
    allf = [niqe_features(luminance(im), select_sharp=0.75) for im in images01]
    f = np.concatenate([a for a in allf if len(a)], axis=0)
    return {"mu": f.mean(axis=0), "cov": np.cov(f, rowvar=False), "n_patches": len(f)}


@lru_cache(maxsize=1)
def _load_pristine(path: str):
    p = Path(path)
    if not p.exists():
        return None
    d = np.load(p)
    return d["mu"], d["cov"]


def niqe(img01: np.ndarray, params_path: str | None = None) -> float | None:
    """Lower is better. Returns None if no pristine model has been fitted yet."""
    if params_path is None:
        from ..config import NIQE_PARAMS_FILE
        params_path = str(NIQE_PARAMS_FILE)
    pr = _load_pristine(params_path)
    if pr is None:
        return None
    mu_p, cov_p = pr
    gray = luminance(img01)
    if min(gray.shape) < 192:
        gray = cv2.resize(gray, (max(192, gray.shape[1] * 2), max(192, gray.shape[0] * 2)),
                          interpolation=cv2.INTER_CUBIC)
    f = niqe_features(gray)
    mu_d = f.mean(axis=0)
    cov_d = np.cov(f, rowvar=False) if len(f) > 1 else np.zeros_like(cov_p)
    d = mu_p - mu_d
    inv = np.linalg.pinv((cov_p + cov_d) / 2.0)
    return float(np.sqrt(max(d @ inv @ d, 0.0)))


# ------------------------------------------------------------------ composite score


def quality_report(img01: np.ndarray, with_niqe: bool = True) -> dict:
    gray = luminance(img01)
    rep = {
        "sharpness": sharpness(gray),
        "contrast": rms_contrast(gray),
        "colorfulness": colorfulness(img01),
        "noise_sigma": noise_sigma(gray),
        "entropy": entropy(gray),
        **exposure_stats(gray),
    }
    rep["niqe"] = niqe(img01) if with_niqe else None
    rep["pocket_iqa"] = pocket_iqa(rep)
    return rep


def pocket_iqa(rep: dict) -> float:
    """0-100 composite. Each cue is mapped to [0,1] with a documented, monotone transfer curve."""
    s_sharp = np.clip(np.log10(rep["sharpness"] + 1.0) / 3.0, 0, 1)        # 1000 -> 1.0
    s_contrast = np.clip(rep["contrast"] / 0.22, 0, 1)
    s_color = np.clip(rep["colorfulness"] / 70.0, 0, 1)
    s_noise = 1.0 - np.clip((rep["noise_sigma"] - 2.0) / 10.0, 0, 1)
    s_expo = 1.0 - np.clip(abs(rep["mean_luma"] - 0.47) / 0.4, 0, 1)
    s_clip = 1.0 - np.clip((rep["clip_high"] + rep["clip_low"]) / 0.15, 0, 1)
    s_entropy = np.clip((rep["entropy"] - 4.0) / 3.5, 0, 1)
    parts = [(0.22, s_sharp), (0.14, s_contrast), (0.12, s_color), (0.14, s_noise),
             (0.12, s_expo), (0.12, s_clip), (0.06, s_entropy)]
    score = sum(w * s for w, s in parts)
    wsum = sum(w for w, _ in parts)
    if rep.get("niqe") is not None:
        s_niqe = 1.0 - np.clip((rep["niqe"] - 3.0) / 12.0, 0, 1)
        score += 0.08 * s_niqe
        wsum += 0.08
    return float(100.0 * score / wsum)
