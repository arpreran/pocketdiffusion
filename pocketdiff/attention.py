"""Explainability: cross-attention heat maps ("which words drew which pixels").

Stable Diffusion's UNet reads the prompt through cross-attention layers (`attn2` in every
transformer block). In each of them every image position (query) distributes a softmax over the
prompt tokens (keys). Summed over layers, heads and denoising steps, the weight a token receives
at a position says how strongly that word shaped that part of the image (DAAM, Tang et al. 2023).

`attention_maps` re-runs the stored generation with the same seed and configuration - the result
is the same image - while a recording processor captures the attention probabilities of every
cross-attention layer. Self-attention layers keep their fast fused kernel; the UNet's original
processors are always restored, also after an error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "and", "or", "with", "by", "for", "from", "is",
    "are", "his", "her", "its", "their", "into", "onto", "over", "under", "as", "very", "it", "that",
    "this", "be", "while", "near", "next", "up", "down", "out", "off", "some", "has", "have",
}
# categorical colours for the word segmentation (readable on light and dark images)
PALETTE = ["#ff5a5f", "#2ec4b6", "#ffbe0b", "#3a86ff", "#8338ec", "#fb5607", "#06d6a0", "#ff006e",
           "#90be6d", "#4cc9f0", "#f15bb5", "#c77dff", "#e9c46a", "#00bbf9", "#f4a261", "#43aa8b"]
# perceptual "inferno"-like colormap stops for the per-word overlays
_CMAP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                  [245, 125, 21], [250, 193, 39], [252, 255, 164]], np.float32) / 255.0


@dataclass
class WordMap:
    word: str
    positions: list[int]               # token positions in the (chunked) text sequence
    heat: np.ndarray                    # (H, W) in [0, 1]
    share: float = 0.0                  # fraction of the image this word dominates
    color: str = "#ffffff"


@dataclass
class AttentionResult:
    words: list[WordMap]
    layers: int
    steps: int
    ms: float
    size: tuple[int, int]               # latent (h, w)
    notes: list[str] = field(default_factory=list)


class _Accumulator:
    """Sums cross-attention probabilities (cond branch, mean over heads) at latent resolution."""

    def __init__(self, lat_h: int, lat_w: int):
        self.h, self.w = lat_h, lat_w
        self.sum = None
        self.calls = 0

    def add(self, probs, heads: int) -> None:
        import torch.nn.functional as F

        bh, q, k = probs.shape
        b = max(bh // heads, 1)
        p = probs.reshape(b, heads, q, k)[-1].float().mean(0)      # CFG batch is [uncond, cond]
        for f in (1, 2, 4, 8, 16):
            hh, ww = -(-self.h // f), -(-self.w // f)              # UNet down-sampling rounds up
            if hh * ww == q:
                break
        else:
            return
        m = p.t().reshape(1, k, hh, ww)
        if (hh, ww) != (self.h, self.w):
            m = F.interpolate(m, size=(self.h, self.w), mode="bilinear", align_corners=False)
        m = m[0]
        self.sum = m if self.sum is None else self.sum + m
        self.calls += 1


def _recording_processor(acc: _Accumulator):
    from diffusers.models.attention_processor import AttnProcessor

    class _Recorder(AttnProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     temb=None, *args, **kwargs):
            orig = attn.get_attention_scores

            def scores(query, key, mask=None):
                p = orig(query, key, mask)
                acc.add(p, attn.heads)
                return p

            attn.get_attention_scores = scores          # instance attribute shadows the method
            try:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            finally:
                del attn.get_attention_scores

    return _Recorder()


def word_positions(tokenizer, prompt: str) -> list[tuple[str, list[int]]]:
    """Prompt words -> their token positions in the chunked encoding (BOS at c*77, then the chunk).

    CLIP's BPE marks the last piece of a word with '</w>', so pieces are merged until one ends
    with it ("astronaut" -> 'astron', 'aut</w>'). Stop-words and punctuation are left out;
    repeated words are merged.
    """
    from .utils import prompt_token_chunks

    L = tokenizer.model_max_length
    words: dict[str, list[int]] = {}
    order: list[str] = []
    for c, ids in enumerate(prompt_token_chunks(tokenizer, prompt)):
        pieces = tokenizer.convert_ids_to_tokens(ids)
        cur, pos = "", []
        for j, piece in enumerate(pieces):
            cur += piece.replace("</w>", "")
            pos.append(c * L + 1 + j)
            if piece.endswith("</w>") or j == len(pieces) - 1:
                w = cur.strip().lower()
                if w and any(ch.isalnum() for ch in w) and w not in STOPWORDS:
                    if w not in words:
                        words[w] = []
                        order.append(w)
                    words[w].extend(pos)
                cur, pos = "", []
    return [(w, words[w]) for w in order]


def attention_maps(engine, prompt: str, cfg, seed: int, negative_prompt: str = "",
                   max_words: int = 16) -> AttentionResult:
    """Re-run one generation with attention recording; returns one normalised heat map per word."""
    import torch

    from .engine import resolve_device

    cfg = resolve_device(cfg)
    bundle = engine.models.get(cfg)
    unet = bundle.unet
    acc = _Accumulator(cfg.height // 8, cfg.width // 8)
    original = dict(unet.attn_processors)
    rec = _recording_processor(acc)
    t0 = time.perf_counter()
    try:
        unet.set_attn_processor({k: (rec if ".attn2." in k else v) for k, v in original.items()})
        res = engine.generate(prompt, cfg, seed=seed, n=1, negative_prompt=negative_prompt)
    finally:
        unet.set_attn_processor(original)
    ms = (time.perf_counter() - t0) * 1000
    if acc.sum is None:
        raise RuntimeError("no cross-attention layers were recorded")
    total = (acc.sum / acc.calls).cpu().numpy()                    # (K, h, w)

    out: list[WordMap] = []
    for w, pos in word_positions(bundle.tokenizer, prompt)[:max_words]:
        pos = [p for p in pos if p < total.shape[0]]
        if not pos:
            continue
        m = total[pos].sum(0)
        m = m - m.min()
        m = m / max(float(m.max()), 1e-8)
        out.append(WordMap(w, pos, m.astype(np.float32)))
    # segmentation: each pixel belongs to the word that attends to it most (if any clearly does)
    if out:
        stack = np.stack([wm.heat for wm in out])
        win = stack.argmax(0)
        strong = stack.max(0) > 0.35
        for i, wm in enumerate(out):
            wm.share = float(((win == i) & strong).mean())
            wm.color = PALETTE[i % len(PALETTE)]
    del res
    if bundle.device == "cuda":
        torch.cuda.empty_cache()
    return AttentionResult(out, layers=acc.calls // max(cfg.steps, 1), steps=cfg.steps, ms=ms,
                           size=(acc.h, acc.w))


# ------------------------------------------------------------------ rendering
def colormap(m: np.ndarray) -> np.ndarray:
    x = np.clip(m, 0, 1) * (len(_CMAP) - 1)
    i = np.minimum(x.astype(int), len(_CMAP) - 2)
    f = (x - i)[..., None]
    return _CMAP[i] * (1 - f) + _CMAP[i + 1] * f


def _resize(m: np.ndarray, w: int, h: int) -> np.ndarray:
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) / 255.0


def overlay(image: Image.Image, heat: np.ndarray) -> Image.Image:
    """Word heat map on top of the image: strong attention glows, the rest is dimmed."""
    base = np.asarray(image.convert("RGB"), np.float32) / 255.0
    h, w = base.shape[:2]
    m = _resize(heat, w, h)
    a = (0.15 + 0.6 * m)[..., None]
    out = base * (0.45 + 0.55 * m[..., None]) * (1 - a) + colormap(m) * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def segmentation(image: Image.Image, words: list[WordMap]) -> Image.Image:
    """Every pixel tinted with the colour of the word that attends to it most."""
    base = np.asarray(image.convert("RGB"), np.float32) / 255.0
    h, w = base.shape[:2]
    if not words:
        return image.convert("RGB")
    stack = np.stack([_resize(wm.heat, w, h) for wm in words])
    win, peak = stack.argmax(0), stack.max(0)
    cols = np.array([[int(wm.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for wm in words], np.float32)
    tint = cols[win]
    a = np.where(peak > 0.35, 0.55, 0.0)[..., None]
    out = base * (1 - a) + tint * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# ------------------------------------------------------------------ repair: boost a missing object's word
def _boost_processor(positions: list[int], factor: float, cond_only: bool):
    """Cross-attention processor that multiplies the attention every image position gives to the
    `positions` tokens by `factor` (rows renormalised). With classifier-free guidance only the
    conditional half of the batch is changed."""
    from diffusers.models.attention_processor import AttnProcessor

    class _Boost(AttnProcessor):
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                     temb=None, *args, **kwargs):
            orig = attn.get_attention_scores

            def scores(query, key, mask=None):
                p = orig(query, key, mask)
                pos = [i for i in positions if i < p.shape[-1]]
                if pos:
                    b = max(p.shape[0] // attn.heads, 1)
                    v = p.view(b, attn.heads, p.shape[1], p.shape[2])
                    sel = v[b // 2:] if cond_only and b >= 2 else v
                    sel[..., pos] *= factor
                    sel /= sel.sum(-1, keepdim=True)
                return p

            attn.get_attention_scores = scores
            try:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            finally:
                del attn.get_attention_scores

    return _Boost()


def boosted_generate(engine, prompt: str, cfg, seed: int, words: list[str], factor: float = 2.0,
                     negative_prompt: str = ""):
    """Generate with extra cross-attention on `words` (all their BPE tokens); same seed and config.
    Returns (GenResult, positions) - positions is empty when none of the words is in the prompt."""
    from .engine import resolve_device

    cfg = resolve_device(cfg)
    bundle = engine.models.get(cfg)
    want = {w.lower() for w in words}
    positions = [p for w, pos in word_positions(bundle.tokenizer, prompt) if w in want for p in pos]
    if not positions:
        return None, []
    unet = bundle.unet
    original = dict(unet.attn_processors)
    proc = _boost_processor(positions, factor, cond_only=cfg.cfg)
    try:
        unet.set_attn_processor({k: (proc if ".attn2." in k else v) for k, v in original.items()})
        res = engine.generate(prompt, cfg, seed=seed, n=1, negative_prompt=negative_prompt)
    finally:
        unet.set_attn_processor(original)
    return res, positions
