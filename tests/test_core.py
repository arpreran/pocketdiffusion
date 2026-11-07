"""Unit tests for the parts that do not need model weights (run: python -m pytest -q)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketdiff.classical import AdaptivePostProcessor, detect_artifacts, quality_report, snap_resolution  # noqa: E402
from pocketdiff.classical.filters import guided_filter_color, guided_filter_gray, periodic_peak_score  # noqa: E402
from pocketdiff.classical.resize import ibp_upscale, resize  # noqa: E402
from pocketdiff.prompt_analyzer import analyze_prompt, sanitize_prompt  # noqa: E402
from pocketdiff.utils import ensure_rgb_uint8  # noqa: E402

RNG = np.random.default_rng(0)


def natural_like(h=256, w=256):
    """Smooth random image with edges (stand-in for a photo)."""
    import cv2
    x = RNG.random((h // 8, w // 8, 3)).astype(np.float32)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    x[h // 3: h // 2, w // 4: 3 * w // 4] = [0.9, 0.2, 0.1]
    return np.clip(x, 0, 1)


# ------------------------------------------------------------------ prompt analysis
@pytest.mark.parametrize("prompt,expect", [
    ("a photo of two dogs", {"dog": (2, None)}),
    ("a red cup and a blue book", {"cup": (1, "red"), "book": (1, "blue")}),
    ("three oranges in a white bowl", {"orange": (3, None), "bowl": (1, "white")}),
    ("an orange cat on a sofa", {"cat": (1, "orange"), "couch": (1, None)}),
    ("a pizza and two wine glasses", {"pizza": (1, None), "wine glass": (2, None)}),
    ("a yellow bus", {"bus": (1, "yellow")}),
])
def test_parser_objects(prompt, expect):
    pa = analyze_prompt(prompt)
    got = {o.name: (o.count, o.color) for o in pa.objects}
    assert got == expect


def test_parser_relations_and_complexity():
    pa = analyze_prompt("a red apple to the left of a green bottle on a wooden table")
    rels = {(r.subject, r.relation, r.object) for r in pa.relations}
    assert ("apple", "left_of", "bottle") in rels
    assert ("bottle", "above", "dining table") in rels
    assert pa.complexity > analyze_prompt("a photo of a dog").complexity


def test_bad_prompts():
    assert analyze_prompt("").warnings
    assert analyze_prompt(None).clean_prompt == ""
    assert sanitize_prompt("a\x00b\n\n  c") == "ab c"
    assert analyze_prompt("x" * 5000).clean_prompt == "x" * 2000   # cap fits long-prompt mode (225 tokens)
    assert analyze_prompt("a nude person").blocked_terms == ["nude"]


# ------------------------------------------------------------------ classical CV
def test_guided_filter_preserves_constant_and_edges():
    img = np.zeros((64, 64), np.float32)
    img[:, 32:] = 1.0
    out = guided_filter_gray(img, img, 4, 1e-4)
    assert abs(out[:, :28].mean()) < 0.02 and abs(out[:, 36:].mean() - 1) < 0.02
    rgb = natural_like(64, 64)
    out_c = guided_filter_color(rgb, rgb[..., 0], 2, 1e-3)
    assert out_c.shape == (64, 64)


def test_grid_detector():
    base = natural_like()
    yy, xx = np.mgrid[:256, :256]
    grid = 0.08 * ((xx % 8 < 1) | (yy % 8 < 1))
    s_clean, _ = periodic_peak_score(base.mean(2))
    s_grid, _ = periodic_peak_score((base.mean(2) + grid).astype(np.float32))
    assert s_grid > s_clean * 1.5


def test_artifact_detection_flags():
    img = natural_like()
    dark = img * 0.08
    rep = detect_artifacts(dark)
    assert rep.severities["underexposure"] > 0.3
    cast = np.clip(img * np.array([1.0, 0.75, 0.45], np.float32), 0, 1)
    assert detect_artifacts(cast).severities["color_cast"] > detect_artifacts(img).severities["color_cast"]
    noisy = np.clip(img + RNG.normal(0, 0.06, img.shape).astype(np.float32), 0, 1)
    assert detect_artifacts(noisy).severities["noise"] > 0.3
    assert detect_artifacts(np.zeros((64, 64, 3), np.float32)).catastrophic
    nan = img.copy()
    nan[0, 0, 0] = np.nan
    assert "nan_output" in detect_artifacts(nan).flags


def test_postprocess_improves_dark_noisy_image():
    img = natural_like()
    bad = np.clip(img * 0.15 + RNG.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)
    res = AdaptivePostProcessor("full")(bad)
    assert res.image.shape == bad.shape
    assert res.after.severities["underexposure"] < res.before.severities["underexposure"]
    assert np.isfinite(res.image).all()


def test_postprocess_leaves_clean_image_mostly_alone():
    img = natural_like()
    res = AdaptivePostProcessor("full")(img)
    assert np.abs(res.image - img).mean() < 0.06


def test_resize_and_snap():
    assert snap_resolution(1920, 1080, 512 * 512) == (680, 384)
    w, h = snap_resolution(512, 512)
    assert (w, h) == (512, 512)
    w, h = snap_resolution(100, 5000)
    assert w % 8 == 0 and h % 8 == 0 and w * h <= 512 * 512
    lr = natural_like(128, 128)
    up = ibp_upscale(lr, (256, 256))
    assert up.shape == (256, 256, 3)
    down = resize(up, (128, 128), "area")
    assert np.abs(down - lr).mean() < 0.05     # back-projection consistency


def test_iqa_runs_and_orders():
    img = natural_like()
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    q1 = quality_report(img, with_niqe=False)
    q2 = quality_report(blurred, with_niqe=False)
    assert q1["sharpness"] > q2["sharpness"]
    assert 0 <= q1["pocket_iqa"] <= 100


def test_ensure_rgb_uint8_handles_odd_inputs():
    assert ensure_rgb_uint8(np.zeros((8, 8))).shape == (8, 8, 3)
    assert ensure_rgb_uint8(np.zeros((8, 8, 4), np.uint8)).shape == (8, 8, 3)
    assert ensure_rgb_uint8(np.full((8, 8, 3), np.nan, np.float32)).max() == 0
    with pytest.raises(ValueError):
        ensure_rgb_uint8(None)


# ------------------------------------------------------------------ quantisation (CPU, tiny layers)
def test_quantization_roundtrip():
    torch = pytest.importorskip("torch")
    from pocketdiff.optimizations import QConv2d, QLinear, quantization_error
    lin = torch.nn.Linear(128, 64)
    x = torch.randn(4, 128)
    for bits, tol in ((8, 0.02), (4, 0.25)):
        q = QLinear(lin, bits, 64)
        rel = (q(x) - lin(x)).norm() / lin(x).norm()
        assert rel < tol, (bits, float(rel))
    conv = torch.nn.Conv2d(16, 32, 3, padding=1)
    xc = torch.randn(1, 16, 8, 8)
    qc = QConv2d(conv, 8, 64)
    assert ((qc(xc) - conv(xc)).norm() / conv(xc).norm()) < 0.02
    assert quantization_error(lin.weight.data, 8) < quantization_error(lin.weight.data, 4)


def test_planner_respects_budgets(monkeypatch):
    pytest.importorskip("torch")
    import pocketdiff.planner as P
    monkeypatch.setattr(P, "family_available", lambda fam: True)
    pl = P.Planner(calibration={})
    simple = analyze_prompt("a photo of a dog")
    complex_ = analyze_prompt("two cats on a blue couch next to a red laptop to the left of a lamp")
    p1 = pl.plan(simple, P.Budget(latency_ms=400, mode="fast"))
    assert p1.predicted_latency_ms <= 400 and not p1.fallback
    p2 = pl.plan(complex_, P.Budget(latency_ms=5000, mode="auto"))
    assert p2.n_candidates >= p1.n_candidates
    low = pl.plan(simple, P.Budget(latency_ms=3000, device_profile="lowend_phone"))
    assert low.predicted_memory_mb <= P.DEVICE_PROFILES["lowend_phone"].memory_budget_mb or low.fallback
    assert low.config.quant in ("int8", "int4")
    imp = pl.plan(simple, P.Budget(latency_ms=1, memory_mb=10))
    assert imp.fallback


# ------------------------------------------------------------------ long prompts and 2K/4K/8K outputs
def _clip_tokenizer():
    import pytest
    try:
        from transformers import CLIPTokenizer

        from pocketdiff.config import MODEL_IDS
        return CLIPTokenizer.from_pretrained(MODEL_IDS["sd15"], subfolder="tokenizer", local_files_only=True)
    except Exception:
        pytest.skip("SD1.5 tokenizer not cached")


def test_short_prompt_chunking_equals_standard_encoding():
    import torch

    from pocketdiff.utils import chunk_ids_tensor, prompt_token_chunks
    tok = _clip_tokenizer()
    for p in ["", "a photo of a cat", "a red apple to the left of a green bottle on a wooden table"]:
        ref = tok([p], padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids
        assert torch.equal(chunk_ids_tensor(tok, prompt_token_chunks(tok, p)), ref)


def test_long_prompt_is_split_into_chunks_not_truncated():
    from pocketdiff.prompt_analyzer import analyze_prompt
    from pocketdiff.utils import chunk_ids_tensor, prompt_token_chunks
    tok = _clip_tokenizer()
    long = ", ".join(["a red umbrella on a wet street"] * 14)          # ~100 tokens
    chunks = prompt_token_chunks(tok, long)
    assert len(chunks) == 2 and all(len(c) <= 75 for c in chunks)
    assert sum(len(c) for c in chunks) == len(tok.tokenize(long))     # nothing dropped
    assert chunk_ids_tensor(tok, chunks, 3).shape == (3, 77)          # padded to match a negative prompt
    pa = analyze_prompt(long, tokenizer=tok)
    assert pa.n_chunks == 2 and not pa.truncated
    assert analyze_prompt(", ".join(["a red umbrella"] * 80), tokenizer=tok).truncated   # > 225 tokens


def test_output_size_keywords():
    from pocketdiff.prompt_analyzer import analyze_prompt, resolve_output_size
    p = "neon city at midnight, photorealistic, IMAX, ultra-detailed 8K."
    assert analyze_prompt(p).requested_size == (7680, 4320)
    assert resolve_output_size(p, 0, 0) == (7680, 4320)
    assert resolve_output_size(p, 1024, 1024) == (1024, 1024)       # explicit size wins
    assert resolve_output_size("a cat in 4k uhd", None, None) == (3840, 2160)
    assert resolve_output_size("a cat", 0, 0) == (512, 512)
    assert analyze_prompt("a photo of 8 kittens").requested_size is None


def test_super_resolve_8k_shape_and_range():
    from pocketdiff.classical import super_resolve
    rng = np.random.default_rng(0)
    img = rng.random((384, 680, 3)).astype(np.float32)
    out = super_resolve(img, (7680, 4320))
    assert out.shape == (4320, 7680, 3) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_snap_resolution_keeps_aspect_for_8k():
    assert snap_resolution(7680, 4320, 512 * 512) == (680, 384)    # 16:9 stays 16:9
    assert snap_resolution(3840, 2160, 384 * 384) == (512, 288)


def test_spelling_correction_only_fixes_real_typos():
    from pocketdiff.prompt_analyzer import analyze_prompt
    pytest.importorskip("spellchecker")
    tok = _clip_tokenizer()
    pa = analyze_prompt("A boy lighting a ciggerate in a MODREN city", tokenizer=tok)
    assert pa.corrections == [["ciggerate", "cigarette"], ["MODREN", "modern"]]
    assert pa.clean_prompt == "A boy lighting a cigarette in a MODERN city"
    keep = "Preran on a skyscraper rooftop, photorealistic, IMAX, ultra-detailed 8K, 35mm"
    assert analyze_prompt(keep, tokenizer=tok).corrections == []      # names / style words untouched
