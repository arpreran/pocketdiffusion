"""Dream mode: a smooth morphing animation from single-step generations.

SD-Turbo turns noise into an image in one UNet call (~85 ms), so an animation is just many
generations whose inputs change a little from frame to frame:

* seed walk   - the initial noise moves along a great circle (spherical interpolation) between the
                noise of seed A and seed B: the picture flows from one composition to the next;
* prompt morph - the prompt embedding moves linearly from prompt A to prompt B with the noise held
                fixed: the same scene changes its content ("summer" -> "winter").

Frames are generated in batches of 4 and eased (smoothstep) so the motion starts and ends gently.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation of two Gaussian noise tensors (keeps the noise statistics; a plain
    linear mix would shrink its norm and wash the frames out half-way)."""
    af, bf = a.flatten().double(), b.flatten().double()
    dot = torch.clamp(torch.dot(af, bf) / (af.norm() * bf.norm()), -1.0, 1.0)
    omega = torch.arccos(dot)
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    so = torch.sin(omega)
    return ((torch.sin((1 - t) * omega) / so) * a.double() + (torch.sin(t * omega) / so) * b.double()).to(a.dtype)


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def dream_frames(engine, cfg, prompt_a: str, prompt_b: str, seed_a: int, seed_b: int, frames: int = 24,
                 batch: int = 4, on_frame: Callable[[int, int, np.ndarray], None] | None = None) -> tuple[list[np.ndarray], float]:
    """Returns (frames, milliseconds of generation)."""
    import time

    from .engine import resolve_device

    cfg = resolve_device(cfg)
    bundle = engine.models.get(cfg)
    nck = max(engine.n_prompt_chunks(bundle, prompt_a), engine.n_prompt_chunks(bundle, prompt_b))
    ea = engine.encode(bundle, [prompt_a], nck)
    eb = ea if prompt_b == prompt_a else engine.encode(bundle, [prompt_b], nck)
    shape = (1, bundle.unet.config.in_channels, cfg.height // 8, cfg.width // 8)
    na = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(seed_a)))
    nb = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(seed_b)))
    ts = [ease(i / max(frames - 1, 1)) for i in range(frames)]
    out: list[np.ndarray] = []
    t0 = time.perf_counter()
    for k in range(0, frames, batch):
        chunk = ts[k:k + batch]
        noise = torch.cat([slerp(na, nb, t) for t in chunk]) if seed_a != seed_b else na.expand(len(chunk), -1, -1, -1)
        cond = torch.cat([ea + (eb - ea) * t for t in chunk])
        res = engine.generate(prompt_a, cfg, seed=seed_a, n=len(chunk), noise=noise.contiguous(), cond=cond)
        for img in res.images:
            out.append(img)
            if on_frame:
                on_frame(len(out), frames, img)
    return out, (time.perf_counter() - t0) * 1000


def ping_pong(frames: list) -> list:
    """A loop that plays forward then backward (no jump at the seam)."""
    return list(frames) + list(frames[-2:0:-1])


def frame_count(seconds: float, fps: int = 12) -> int:
    return int(min(max(math.ceil(seconds * fps / 2), 8), 48))
