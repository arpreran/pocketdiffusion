"""Tests for the v1.6 features that need no GPU: explain (attention maps), image-to-image editing,
share pages and the installable web app.      python -m pytest -q tests/test_features.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("fastapi")
pytest.importorskip("torch")

GID = "20260101-120000-abc123"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    import backend.db as dbmod
    import backend.main as main
    from fastapi.testclient import TestClient
    main.db = dbmod.Database(tmp_path_factory.mktemp("db") / "test.db")
    main.jobs.db = main.db
    store = tmp_path_factory.mktemp("store")
    main.STORE = store
    main.jobs.store = store
    (store / GID).mkdir()
    Image.new("RGB", (64, 48), (200, 40, 40)).save(store / GID / "final.png")
    row = {"id": GID, "created_at": time.time(), "kind": "generate", "prompt": 'a red apple <b>"bold"</b>',
           "seed": 7, "family": "sd_turbo", "scheduler": "euler", "steps": 1, "steps_done": 1, "width": 64,
           "height": 48, "latency_ms": 640.0, "clip": 31.2, "image_path": f"{GID}/final.png",
           "config_json": json.dumps({"name": "turbo1_taesd", "family": "sd_turbo", "steps": 1, "cfg": False})}
    main.db.insert_generation(row, [{"generation_id": GID, "idx": 0, "seed": 9, "clip": 31.2, "iqa": None,
                                     "score": None, "chosen": 1, "image_path": None}], [], [])
    return main, TestClient(main.app), store


# ------------------------------------------------------------------ installable app
def test_manifest_and_service_worker(ctx):
    _, client, _ = ctx
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and "manifest" in r.headers["content-type"]
    m = r.json()
    assert m["display"] == "standalone" and m["start_url"].startswith("/")
    for icon in m["icons"]:
        assert client.get(icon["src"]).status_code == 200, icon["src"]
    assert any(i.get("purpose") == "maskable" for i in m["icons"])
    sw = client.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"]
    assert sw.headers.get("service-worker-allowed") == "/"
    assert "/api/" in sw.text                       # API calls are never served from the cache
    page = client.get("/").text
    assert 'rel="manifest"' in page and "apple-touch-icon" in page


def test_system_lists_features(ctx):
    _, client, _ = ctx
    s = client.get("/api/system").json()
    assert {"edit", "attention", "share", "pwa"} <= set(s["features"])
    assert "public_page" in s


# ------------------------------------------------------------------ share page
def test_share_page_escapes_and_previews(ctx):
    _, client, _ = ctx
    r = client.get(f"/share/{GID}")
    assert r.status_code == 200
    assert '<b>"bold"' not in r.text and "&lt;b&gt;&quot;bold&quot;" in r.text   # prompt is HTML-escaped
    assert 'property="og:image"' in r.text and f"/media/{GID}/final.png" in r.text
    assert f"/?remix={GID}" in r.text


def test_share_page_unknown_and_bad_ids(ctx):
    _, client, _ = ctx
    assert client.get("/share/20990101-000000-ffffff").status_code == 404
    assert client.get("/share/..%2f..%2fsecret").status_code in (400, 404)
    assert client.get("/share/not-an-id").status_code == 400


# ------------------------------------------------------------------ explain
def test_attention_rejects_bad_and_unknown_ids(ctx):
    _, client, _ = ctx
    assert client.post("/api/generations/bad/attention").status_code == 400
    assert client.post("/api/generations/20990101-000000-ffffff/attention").status_code == 404


def test_attention_served_from_cache(ctx):
    _, client, store = ctx
    d = store / GID / "attn"
    d.mkdir(exist_ok=True)
    payload = {"id": GID, "words": [{"word": "apple", "image": f"{GID}/attn/w0.jpg", "share": 0.4}],
               "segmentation": f"{GID}/attn/seg.jpg", "layers": 16, "steps": 1}
    (d / "attention.json").write_text(json.dumps(payload))
    r = client.post(f"/api/generations/{GID}/attention")
    assert r.status_code == 200 and r.json()["words"][0]["word"] == "apple"


def test_attention_rendering_helpers():
    from pocketdiff.attention import PALETTE, WordMap, colormap, overlay, segmentation
    c = colormap(np.linspace(0, 1, 11))
    assert c.shape == (11, 3) and c.min() >= 0 and c.max() <= 1
    assert np.all(np.diff(c.sum(1)) > 0)                      # brightness rises with attention
    img = Image.new("RGB", (80, 60), (90, 90, 90))
    left = np.zeros((6, 8), np.float32); left[:, :4] = 1
    right = 1 - left
    ov = overlay(img, left)
    assert ov.size == (80, 60)
    a = np.asarray(ov, np.float32)
    assert a[:, :30].mean() > a[:, 50:].mean()                 # attended half glows
    words = [WordMap("cat", [1], left, color=PALETTE[0]), WordMap("dog", [2], right, color=PALETTE[1])]
    seg = np.asarray(segmentation(img, words), np.float32)
    assert seg[30, 5, 0] > seg[30, 5, 2]                       # #ff5a5f tint on the left
    assert seg[30, 75, 1] > seg[30, 75, 0]                     # #2ec4b6 tint on the right


def test_attention_accumulator_resolutions():
    import torch
    from pocketdiff.attention import _Accumulator
    acc = _Accumulator(48, 64)
    heads, k = 8, 77
    for f in (1, 2, 4, 8):                                       # every UNet resolution level
        q = (48 // f) * (64 // f)
        p = torch.zeros(2 * heads, q, k)
        p[heads:, :, 5] = 1.0                                     # cond half attends to token 5 only
        acc.add(p, heads)
    m = acc.sum / acc.calls
    assert acc.calls == 4 and tuple(m.shape) == (k, 48, 64)
    assert torch.allclose(m[5], torch.ones(48, 64)) and float(m[0].abs().max()) == 0.0


def test_word_positions_merge_bpe_and_skip_stopwords():
    from pocketdiff.attention import word_positions

    class Tok:                                                    # tiny stand-in for the CLIP tokenizer
        model_max_length = 77
        vocab = {"a</w>": 1, "astron": 2, "aut</w>": 3, "on</w>": 4, "the</w>": 5, "moon</w>": 6, ",</w>": 7}

        def __call__(self, text, **kw):
            ids = {"a astronaut on the moon ,": [1, 2, 3, 4, 5, 6, 7], ",": [7]}.get(text, [])
            return {"input_ids": ids}

        def convert_ids_to_tokens(self, ids):
            inv = {v: k for k, v in self.vocab.items()}
            return [inv[i] for i in ids]

    words = word_positions(Tok(), "a astronaut on the moon ,")
    assert words == [("astronaut", [2, 3]), ("moon", [6])]       # positions are 1-based after BOS


# ------------------------------------------------------------------ image-to-image editing
def test_edit_validation(ctx):
    _, client, _ = ctx
    assert client.post("/api/edit", json={"prompt": "make it snowy"}).status_code == 422
    both = {"prompt": "x", "source_id": GID, "image": "data:image/png;base64,AAAA"}
    assert client.post("/api/edit", json=both).status_code == 422
    assert client.post("/api/edit", json={"prompt": "x", "source_id": "../etc"}).status_code == 422
    assert client.post("/api/edit", json={"prompt": "x", "source_id": "20990101-000000-ffffff"}).status_code == 404
    assert client.post("/api/edit", json={"prompt": "x", "source_id": GID, "strength": 3}).status_code == 422


def test_edit_helpers():
    from backend.jobs import PromptRejected, config_from_dict, decode_upload, edit_size
    assert edit_size(1024, 768) == (512, 384)
    assert edit_size(4000, 3000) == (512, 384)
    w, h = edit_size(3000, 400)
    assert w == 512 and h >= 128 and h % 8 == 0
    buf = io.BytesIO()
    Image.new("RGB", (30, 20), (1, 2, 3)).save(buf, format="PNG")
    im = decode_upload("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert im.size == (30, 20) and im.mode == "RGB"
    with pytest.raises(PromptRejected):
        decode_upload("data:image/png;base64," + base64.b64encode(b"not an image").decode())
    cfg = config_from_dict({"name": "x", "family": "sd_turbo", "steps": 1, "cfg": False, "unet_evals_per_image": 1})
    assert cfg.family == "sd_turbo" and cfg.steps == 1


def test_pages_forwarder_supports_share_links():
    page = (Path(__file__).resolve().parents[1] / "deploy" / "pages" / "index.html").read_text(encoding="utf-8")
    assert "share" in page and "/share/" in page


def test_offline_showcase_files_exist():
    pages = Path(__file__).resolve().parents[1] / "deploy" / "pages"
    sc = pages / "showcase" / "showcase.json"
    if not sc.exists():
        pytest.skip("showcase not built (python scripts/build_showcase.py)")
    d = json.loads(sc.read_text(encoding="utf-8"))
    paths = [d["video"]] if d.get("video") else []
    for it in d["items"]:
        assert it["prompt"] and it["latency_ms"] > 0
        paths += [it["image"], it["thumb"]]
        if it.get("explain"):
            paths += [it["explain"]["segmentation"]] + [w["image"] for w in it["explain"]["words"]]
    if d.get("edit"):
        paths += [d["edit"]["source"], d["edit"]["result"]]
    missing = [p for p in paths if not (pages / p).exists()]
    assert not missing, missing
    page = (pages / "index.html").read_text(encoding="utf-8")
    assert "showcase/showcase.json" in page and "?showcase" in page


# ------------------------------------------------------------------ public link: owners and limits
def test_visitors_see_and_change_only_their_own_images(ctx):
    main, _, _ = ctx
    from fastapi.testclient import TestClient
    alice, bob = TestClient(main.app), TestClient(main.app)
    alice.get("/api/system")
    bob.get("/api/system")
    va, vb = alice.cookies.get("pd_vid"), bob.cookies.get("pd_vid")
    assert va and vb and va != vb
    for gid, owner in (("20260102-120000-aaaaaa", va), ("20260102-120001-bbbbbb", vb)):
        main.db.insert_generation({"id": gid, "created_at": time.time(), "kind": "generate", "prompt": f"by {owner[:4]}",
                                   "owner": owner, "image_path": f"{GID}/final.png"}, [], [], [])
    mine = alice.get("/api/generations").json()
    assert [g["id"] for g in mine["items"]] == ["20260102-120000-aaaaaa"] and mine["everyone"] is False
    assert all(g["mine"] and "owner" not in g for g in mine["items"])
    assert alice.post("/api/generations/20260102-120001-bbbbbb/favorite").status_code == 403
    assert alice.delete("/api/generations/20260102-120001-bbbbbb").status_code == 403
    assert alice.get("/api/generations/20260102-120001-bbbbbb").json()["mine"] is False   # share links still work
    assert alice.post("/api/generations/20260102-120000-aaaaaa/favorite").json()["favorite"] is True
    assert alice.delete("/api/generations/20260102-120000-aaaaaa").status_code == 200
    assert alice.get("/api/system").json()["admin"] is False


def test_team_recognised_only_on_the_laptop_itself():
    from types import SimpleNamespace
    import backend.main as main

    def req(host, client="127.0.0.1", **headers):
        return SimpleNamespace(headers={"host": host, **headers}, client=SimpleNamespace(host=client))
    assert main.is_admin(req("127.0.0.1:8000"))
    assert main.is_admin(req("localhost:8000"))
    assert not main.is_admin(req("abc123.gradio.live"))                       # through the tunnel
    assert not main.is_admin(req("192.168.1.5:8000", client="192.168.1.9"))  # phone on the same Wi-Fi
    assert not main.is_admin(req("127.0.0.1:8000", **{"x-forwarded-for": "1.2.3.4"}))


def test_rate_limiter_window():
    import backend.main as main
    lim = main.RateLimiter(3)
    assert [lim.allow("v") for _ in range(4)] == [True, True, True, False]
    assert lim.allow("other")


# ------------------------------------------------------------------ object repair: attention boost
def test_attention_boost_processor_shifts_attention_to_the_word():
    import torch
    from diffusers.models.attention_processor import Attention, AttnProcessor
    from pocketdiff.attention import _boost_processor
    torch.manual_seed(0)
    attn = Attention(query_dim=16, cross_attention_dim=8, heads=2, dim_head=8)
    x, ctx = torch.randn(1, 6, 16), torch.randn(1, 5, 8)
    with torch.no_grad():
        ref = AttnProcessor()(attn, x, ctx)
        same = _boost_processor([3], 1.0, cond_only=False)(attn, x, ctx)
        strong = _boost_processor([3], 1e4, cond_only=False)(attn, x, ctx)
        only_3 = AttnProcessor()(attn, x, ctx[:, 3:4].expand(1, 5, 8))   # every key = token 3
    assert torch.allclose(ref, same, atol=1e-5)                      # factor 1 changes nothing
    assert torch.allclose(strong, only_3, atol=1e-3)                 # huge factor -> attends to token 3 only
    assert not hasattr(attn, "get_attention_scores") or callable(attn.get_attention_scores)


def test_boost_cond_only_leaves_unconditional_half():
    import torch
    from diffusers.models.attention_processor import Attention, AttnProcessor
    from pocketdiff.attention import _boost_processor
    torch.manual_seed(1)
    attn = Attention(query_dim=16, cross_attention_dim=8, heads=2, dim_head=8)
    x, ctx = torch.randn(2, 6, 16), torch.randn(2, 5, 8)             # CFG batch: [uncond, cond]
    with torch.no_grad():
        ref = AttnProcessor()(attn, x, ctx)
        out = _boost_processor([2], 50.0, cond_only=True)(attn, x, ctx)
    assert torch.allclose(out[0], ref[0], atol=1e-5) and not torch.allclose(out[1], ref[1], atol=1e-3)


def test_planner_offers_the_tiny_model_only_when_downloaded(monkeypatch):
    import pocketdiff.planner as P
    from pocketdiff.prompt_analyzer import analyze_prompt
    monkeypatch.setattr(P, "family_available", lambda f: f in ("sd15", "bk_tiny"))
    pl = P.Planner()
    fams = {c[0] for c in pl.candidates(P.DEVICE_PROFILES["lowend_phone"], P.Budget())}
    assert fams == {"sd15", "bk_tiny"}
    plan = pl.plan(analyze_prompt("a cat on a sofa"), P.Budget(device_profile="lowend_phone", latency_ms=20000))
    assert plan.predicted_memory_mb <= P.DEVICE_PROFILES["lowend_phone"].memory_budget_mb
