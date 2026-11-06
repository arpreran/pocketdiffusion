"""Artifact detection for generated images.

Each detector returns a continuous severity in [0,1] plus the raw measurement, so the
post-processor can scale its corrections (adaptive) instead of applying fixed filters.

Detected artifact types
-----------------------
blank / NaN output     - catastrophic decode failure (triggers regeneration, not correction)
over-exposure          - clipped highlights
under-exposure         - crushed shadows / overall dark image
colour cast            - Lab chroma-offset "cast factor" (equivalent-circle method)
low contrast           - narrow luminance dynamic range
over-saturation        - "neon" colours typical for few-step distilled models
noise / grain          - Immerkaer noise sigma
periodic grid          - 8-px spectral peaks left by the latent decoder
blur                   - low Laplacian variance relative to image content
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .filters import luminance, periodic_peak_score
from .iqa import noise_sigma, sharpness


@dataclass
class ArtifactReport:
    severities: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    catastrophic: bool = False
    grid_mask: np.ndarray | None = None      # spectral peak mask (for notch filtering)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("grid_mask", None)
        return d

    @property
    def total(self) -> float:
        return float(sum(self.severities.values()))


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def detect_artifacts(img01: np.ndarray, full: bool = True) -> ArtifactReport:
    """full=False is the cheap screening mode used for every best-of-N candidate (exposure,
    cast, contrast, saturation on a 2x-downsampled copy); full=True adds noise, periodic grid
    and blur at full resolution and is run once on the selected image."""
    rep = ArtifactReport()
    if not full:
        img01 = cv2.resize(img01, (img01.shape[1] // 2, img01.shape[0] // 2), interpolation=cv2.INTER_AREA)
    if not np.isfinite(img01).all():
        rep.catastrophic = True
        rep.flags.append("nan_output")
        img01 = np.nan_to_num(img01, nan=0.0)
    gray = luminance(img01)
    m = rep.measurements
    m["mean_luma"] = float(gray.mean())
    m["std_luma"] = float(gray.std())
    if m["std_luma"] < 0.01:
        rep.catastrophic = True
        rep.flags.append("blank_output")

    # exposure
    m["clip_high"] = float((img01.max(axis=2) > 0.985).mean())
    m["clip_low"] = float((gray < 0.015).mean())
    rep.severities["overexposure"] = max(_ramp(m["clip_high"], 0.02, 0.15), _ramp(m["mean_luma"], 0.72, 0.9))
    rep.severities["underexposure"] = max(_ramp(m["clip_low"], 0.05, 0.3), _ramp(-m["mean_luma"], -0.22, -0.08))

    # colour cast: K = D / M in Lab (D = distance of mean chroma from neutral, M = chroma spread)
    lab = cv2.cvtColor(np.clip(img01, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    a, b = lab[..., 1], lab[..., 2]
    d = float(np.hypot(a.mean(), b.mean()))
    mm = float(np.hypot(a.std(), b.std()))
    m["cast_D"] = d
    m["cast_K"] = d / (mm + 1e-6)
    m["cast_angle_deg"] = float(np.degrees(np.arctan2(b.mean(), a.mean())))
    # K > 1 flags casts in low-chroma scenes; very large offsets are casts even in colourful ones
    rep.severities["color_cast"] = max(_ramp(m["cast_K"], 1.0, 2.5) * _ramp(d, 4.0, 12.0),
                                       _ramp(d, 20.0, 40.0) * _ramp(m["cast_K"], 0.5, 1.0))

    # contrast
    p1, p99 = np.percentile(gray, [1, 99])
    m["dynamic_range"] = float(p99 - p1)
    rep.severities["low_contrast"] = _ramp(-m["dynamic_range"], -0.6, -0.3)

    # saturation
    hsv = cv2.cvtColor(np.clip(img01, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    m["oversat_frac"] = float(((hsv[..., 1] > 0.9) & (hsv[..., 2] > 0.6)).mean())
    rep.severities["oversaturation"] = _ramp(m["oversat_frac"], 0.08, 0.3)
    if not full:
        rep.flags.extend(k for k, v in rep.severities.items() if v >= 0.3)
        return rep

    # noise
    m["noise_sigma"] = noise_sigma(gray)
    rep.severities["noise"] = _ramp(m["noise_sigma"], 7.0, 14.0)  # tuned on validation images

    # periodic grid
    score, mask = periodic_peak_score(gray, period=8)
    m["grid_score"] = score
    rep.grid_mask = mask
    rep.severities["periodic_grid"] = _ramp(score, 2.5, 6.0)

    # blur (normalised by local contrast so dark/flat scenes aren't falsely flagged)
    s = sharpness(gray)
    m["laplacian_var"] = s
    m["blur_index"] = float(s / (1e4 * m["std_luma"] ** 2 + 1e-6))
    rep.severities["blur"] = _ramp(-m["blur_index"], -0.6, -0.15)

    rep.flags.extend(k for k, v in rep.severities.items() if v >= 0.3)
    return rep


def artifact_heatmap(img01: np.ndarray) -> np.ndarray:
    """Per-pixel visualisation: red = clipped highlights, blue = crushed shadows,
    green = high-frequency residual energy (noise / grid), as an RGB overlay in [0,1]."""
    gray = luminance(img01)
    hi = (img01.max(axis=2) > 0.985).astype(np.float32)
    lo = (gray < 0.015).astype(np.float32)
    resid = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.2))
    resid = np.clip(resid / (np.percentile(resid, 99.5) + 1e-6), 0, 1)
    overlay = 0.45 * img01 + 0.55 * np.stack([hi, resid * 0.9, lo], axis=-1)
    return np.clip(overlay, 0, 1).astype(np.float32)
