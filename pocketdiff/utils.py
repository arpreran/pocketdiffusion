"""Small shared helpers: timing, memory measurement, image conversion."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

import numpy as np
from PIL import Image

LOG = logging.getLogger("pocketdiff")
if not LOG.handlers:
    logging.basicConfig(level=os.environ.get("POCKETDIFF_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # torch missing or broken driver -> treat as CPU-only machine
        return False


def sync(device: str = "cuda") -> None:
    if device.startswith("cuda") and cuda_available():
        import torch
        torch.cuda.synchronize()


class Timer:
    """Wall-clock timer that synchronises CUDA so GPU work is actually counted."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.records: dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.records[name] = self.records.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    @property
    def total_ms(self) -> float:
        return sum(self.records.values())


class PeakRSSMonitor:
    """Polls process RSS in a background thread (CPU peak-memory measurement)."""

    def __init__(self, interval_s: float = 0.01):
        import psutil
        self._proc = psutil.Process()
        self.interval = interval_s
        self.peak = 0
        self.start_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.start_rss = self._proc.memory_info().rss
        self.peak = self.start_rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20


@contextmanager
def gpu_peak_memory():
    """Yields a dict that receives peak allocated / reserved CUDA memory in MB."""
    out = {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    if not cuda_available():
        yield out
        return
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield out
    finally:
        torch.cuda.synchronize()
        out["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        out["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20


def module_bytes(module) -> int:
    """Bytes held by parameters + buffers (true storage size, handles packed int4)."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


def count_params(module) -> int:
    """Logical parameter count (packed int4 buffers are counted as two weights per byte)."""
    n = sum(p.numel() for p in module.parameters())
    for name, b in module.named_buffers():
        if name.endswith("qweight"):
            n += b.numel() * (2 if b.dtype.__str__().endswith("uint8") else 1)
    return n


# ---------------------------------------------------------------- image conversion helpers
def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def ensure_rgb_uint8(img) -> np.ndarray:
    """Accepts PIL / float / uint8 / gray / RGBA input and returns HxWx3 uint8 (bad-input safe)."""
    if img is None:
        raise ValueError("No image given.")
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Expected an HxWxC image, got shape {img.shape}")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 colour channels, got {img.shape[2]}")
    if img.dtype != np.uint8:
        img = to_uint8(np.nan_to_num(img.astype(np.float32), nan=0.0))
    return np.ascontiguousarray(img)


def pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(ensure_rgb_uint8(img))


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"
