"""CLIP-based text-image alignment scoring and an image-level safety check.

Two different CLIP models are used on purpose:
* ViT-B/32 (151M params) inside the system for best-of-N re-ranking - small enough for a phone.
* ViT-L/14 only for *evaluation* (the standard CLIPScore backbone), so the system never
  optimises the exact metric it is evaluated on.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .config import MODEL_IDS

UNSAFE_CONCEPTS = ["explicit nudity", "pornographic image", "graphic violence with blood and gore",
                   "a naked person"]
SAFE_CONCEPTS = ["a safe for work photo", "an ordinary everyday picture", "a landscape",
                 "a picture of an object", "a photo of a person wearing clothes", "an animal"]


class ClipScorer:
    def __init__(self, model_id: str, device: str | None = None, dtype: torch.dtype | None = None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self._safety_text = None

    @torch.inference_mode()
    def image_features(self, images: list[np.ndarray]) -> torch.Tensor:
        u8 = [(np.clip(im, 0, 1) * 255).astype(np.uint8) if im.dtype != np.uint8 else im for im in images]
        px = self.processor(images=u8, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        f = self.model.get_image_features(pixel_values=px)
        if not isinstance(f, torch.Tensor):  # newer transformers may return a model output
            f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
        return torch.nn.functional.normalize(f.float(), dim=-1)

    @torch.inference_mode()
    def text_features(self, texts: list[str]) -> torch.Tensor:
        tok = self.processor.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                       return_tensors="pt").to(self.device)
        f = self.model.get_text_features(**tok)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
        return torch.nn.functional.normalize(f.float(), dim=-1)

    def clip_score(self, images: list[np.ndarray], prompts: list[str] | str) -> np.ndarray:
        """CLIPScore as in torchmetrics: 100 * max(cos(image, text), 0). One prompt per image,
        or a single prompt shared by all images."""
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        scores = []
        bs = 16
        for i in range(0, len(images), bs):
            fi = self.image_features(images[i:i + bs])
            ft = self.text_features(prompts[i:i + bs])
            scores.append((100.0 * (fi * ft).sum(-1)).clamp(min=0).cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    def score_with_parts(self, images: list[np.ndarray], prompt: str, phrases: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Whole-prompt CLIPScore per image plus a [n_images, n_phrases] matrix of per-object scores
        (e.g. "a photo of a red apple"). Image features are computed once and reused."""
        fi = torch.cat([self.image_features(images[i:i + 16]) for i in range(0, len(images), 16)])
        ft = self.text_features([prompt] + list(phrases))
        sims = (100.0 * fi @ ft.T).clamp(min=0).cpu().numpy()
        return sims[:, 0], sims[:, 1:]

    def safety(self, images: list[np.ndarray], margin: float = 0.02) -> list[dict]:
        """Zero-shot NSFW check: flag an image if it is closer to any unsafe concept than to
        every safe concept by more than `margin` (cosine). Coarse, documented as a limitation."""
        if self._safety_text is None:
            self._safety_text = (self.text_features(UNSAFE_CONCEPTS), self.text_features(SAFE_CONCEPTS))
        unsafe_t, safe_t = self._safety_text
        fi = self.image_features(images)
        su = (fi @ unsafe_t.T).max(dim=1).values
        ss = (fi @ safe_t.T).max(dim=1).values
        return [{"unsafe": bool(u - s > margin), "unsafe_sim": float(u), "safe_sim": float(s)}
                for u, s in zip(su.cpu().numpy(), ss.cpu().numpy())]

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=2)
def get_scorer(kind: str = "rerank", device: str | None = None) -> ClipScorer:
    model_id = MODEL_IDS["clip_rerank" if kind == "rerank" else "clip_eval"]
    return ClipScorer(model_id, device=device)
