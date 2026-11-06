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
    assert analyze_prompt("x" * 5000).clean_prompt == "x" * 600
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
