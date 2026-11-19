"""Import the heavy ML libraries once, in the calling (main) thread, before any worker thread starts.

transformers / diffusers / peft resolve most of their names lazily on first access. That lazy-import
machinery is not thread-safe: when the GPU worker (warm-up: diffusers -> peft -> transformers.AutoModel)
and the tokenizer-preload thread (transformers.CLIPTokenizer) import at the same moment, one of them can
see a half-initialised module and fail with

    cannot import name 'AutoModel' from 'transformers'

which made the server start without working models. Importing everything up front removes the race.
"""
from __future__ import annotations

import threading
import time

from .utils import LOG

_done = False
_lock = threading.Lock()


def import_ml_libraries() -> float:
    """Idempotent; returns the seconds spent (0 when already done)."""
    global _done
    with _lock:
        if _done:
            return 0.0
        t0 = time.perf_counter()
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import (AutoModel, CLIPModel, CLIPProcessor, CLIPTextModel,  # noqa: F401
                                  CLIPTokenizer, Owlv2ForObjectDetection, Owlv2Processor,
                                  OwlViTForObjectDetection, OwlViTProcessor)
        import diffusers  # noqa: F401
        import diffusers.loaders.peft  # noqa: F401
        from diffusers import (AutoencoderKL, AutoencoderTiny, DPMSolverMultistepScheduler,  # noqa: F401
                               EulerDiscreteScheduler, LCMScheduler, PNDMScheduler,
                               UNet2DConditionModel)
        _done = True
        dt = time.perf_counter() - t0
        LOG.info("ML libraries imported in %.1f s", dt)
        return dt
