"""Adaptive, artifact-driven post-processing.

detect -> decide -> correct, where every correction strength is a function of the measured
artifact severity (so a clean image passes through almost unchanged). Each stage is recorded
so the UI can show the intermediate images and the report can ablate individual stages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import enhance
from .artifacts import ArtifactReport, detect_artifacts
from .filters import notch_filter
from .resize import resize

LEVELS = ("off", "light", "full")


@dataclass
class Stage:
    name: str
    params: dict
    ms: float
    image: np.ndarray | None = None


@dataclass
class PostResult:
    image: np.ndarray
    before: ArtifactReport
    after: ArtifactReport | None
    stages: list[Stage] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)


class AdaptivePostProcessor:
    def __init__(self, level: str = "full", keep_intermediates: bool = True,
                 disabled: tuple[str, ...] = ()):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self.level = level
        self.keep = keep_intermediates
        self.disabled = set(disabled)   # for ablations, e.g. {"sharpen"}

    def _run(self, name, fn, img, params, stages):
        if name in self.disabled:
            return img
        t0 = time.perf_counter()
        out = fn(img)
        stages.append(Stage(name, params, (time.perf_counter() - t0) * 1000.0,
                            out.copy() if self.keep else None))
        return out

    def __call__(self, img01: np.ndarray, target_size: tuple[int, int] | None = None,
                 upscaler: str = "ibp", measure_after: bool = True) -> PostResult:
        img = np.clip(np.nan_to_num(img01.astype(np.float32)), 0, 1)
        t0 = time.perf_counter()
        rep = detect_artifacts(img)
        stages: list[Stage] = [Stage("detect", {"flags": rep.flags},
                                     (time.perf_counter() - t0) * 1000.0, None)]
        if self.level == "off" or rep.catastrophic:
            if target_size:
                img = self._run("resize", lambda x: resize(x, target_size, upscaler), img,
                                {"size": target_size, "method": upscaler}, stages)
            return PostResult(img, rep, None, stages)

        sev = rep.severities
        light = self.level == "light"
        k = 0.5 if light else 1.0

        # 1) periodic grid -> spectral notch filter
        if sev.get("periodic_grid", 0) > 0.2 and rep.grid_mask is not None:
            img = self._run("grid_notch", lambda x: np.clip(notch_filter(x, rep.grid_mask), 0, 1),
                            img, {"grid_score": rep.measurements["grid_score"]}, stages)

        # 2) noise -> guided-filter denoise (eps tied to sigma)
        if sev.get("noise", 0) > 0.3:
            s = rep.measurements["noise_sigma"]
            img = self._run("denoise", lambda x: enhance.edge_aware_denoise(x, s * k), img,
                            {"noise_sigma": s}, stages)

        # 3) colour cast -> shades-of-gray white balance with severity-scaled strength
        if sev.get("color_cast", 0) > 0.1:
            st = float(np.clip(sev["color_cast"] * 0.8 * k, 0, 0.8))
            img = self._run("white_balance", lambda x: enhance.shades_of_gray_wb(x, 6.0, st), img,
                            {"strength": st, "cast_K": rep.measurements["cast_K"]}, stages)

        # 4) exposure -> bounded adaptive gamma
        if sev.get("underexposure", 0) > 0.15 or sev.get("overexposure", 0) > 0.15:
            ms = 0.15 + 0.35 * max(sev["underexposure"], sev["overexposure"]) * k
            img = self._run("adaptive_gamma", lambda x: enhance.adaptive_gamma(x, 0.46, ms), img,
                            {"max_strength": ms}, stages)

        # 5) low contrast -> CLAHE on L (clip limit grows with severity)
        if sev.get("low_contrast", 0) > 0.1:
            cl = 1.2 + 1.8 * sev["low_contrast"] * k
            img = self._run("clahe", lambda x: enhance.clahe_lab(x, cl, 8), img,
                            {"clip_limit": cl}, stages)

        # 6) over-saturation -> soft-knee saturation compression
        if sev.get("oversaturation", 0) > 0.1:
            img = self._run("saturation_knee", lambda x: enhance.saturation_soft_knee(x, 0.85, 0.95),
                            img, {"knee": 0.85}, stages)

        # 7) resolution change (IBP super-resolution or area downscale)
        if target_size and (target_size[0], target_size[1]) != (img.shape[1], img.shape[0]):
            img = self._run("resize", lambda x: resize(x, target_size, upscaler), img,
                            {"size": target_size, "method": upscaler}, stages)

        # 8) edge-aware sharpening: stronger when the image is soft, never on noisy flats
        if not light or sev.get("blur", 0) > 0.3:
            amt = float(0.25 + 0.5 * sev.get("blur", 0)) * k
            img = self._run("sharpen", lambda x: enhance.edge_aware_sharpen(x, amt)[0], img,
                            {"amount": amt}, stages)

        after = detect_artifacts(img) if measure_after else None
        return PostResult(img, rep, after, stages)
