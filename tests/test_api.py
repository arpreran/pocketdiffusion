"""Backend API tests that need no GPU (python -m pytest -q tests/test_api.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("fastapi")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import backend.db as dbmod
    import backend.main as main
    from fastapi.testclient import TestClient
    main.db = dbmod.Database(tmp_path_factory.mktemp("db") / "test.db")   # isolated database
    main.jobs.db = main.db
    return TestClient(main.app)


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "PocketDiffusion" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_analyze(client):
    r = client.post("/api/analyze", json={"prompt": "two cats on a blue couch"})
    assert r.status_code == 200
    objs = {o["name"]: o for o in r.json()["objects"]}
    assert objs["cat"]["count"] == 2 and objs["couch"]["color"] == "blue"


def test_plan_respects_device(client, monkeypatch):
    import pocketdiff.planner as P
    monkeypatch.setattr(P, "family_available", lambda f: True)
    import backend.main as main
    main._planner = None
    low = client.post("/api/plan", json={"prompt": "a dog", "device_profile": "lowend_phone", "latency_ms": 3000}).json()
    assert low["config"]["quant"] in ("int8", "int4")
    assert low["predicted_memory_mb"] <= 1100 or low["fallback"]


@pytest.mark.parametrize("body", [
    {"prompt": "a dog", "mode": "turbo"},
    {"prompt": "a dog", "device_profile": "toaster"},
    {"prompt": "a dog", "latency_ms": 1},
    {"prompt": "a dog", "width": 99999},
])
def test_validation_errors(client, body):
    assert client.post("/api/plan", json=body).status_code == 422


def test_generate_rejects_empty_prompt(client):
    assert client.post("/api/generate", json={"prompt": "   "}).status_code == 422
    assert client.post("/api/generate", json={"prompt": "x", "preset": "nope"}).status_code == 422


def test_history_and_stats_on_empty_db(client):
    r = client.get("/api/generations").json()
    assert r == {"items": [], "total": 0}
    assert client.get("/api/generations/does-not-exist").status_code == 404
    assert client.post("/api/generations/nope/favorite").status_code == 404
    assert client.delete("/api/generations/..%2F..").status_code in (400, 404)
    s = client.get("/api/stats").json()
    assert s["n"] == 0


def test_db_roundtrip(client):
    import backend.main as main
    row = {"id": "t1", "created_at": 1.0, "kind": "generate", "prompt": "a dog", "family": "sd_turbo",
           "latency_ms": 100.0, "clip": 30.0, "plan_json": '{"reasons": ["x"]}'}
    main.db.insert_generation(row, [{"generation_id": "t1", "idx": 0, "seed": 1, "clip": 30.0, "iqa": 80.0,
                                     "score": 31.0, "chosen": 1, "image_path": None}], [], [])
    g = client.get("/api/generations/t1").json()
    assert g["plan"]["reasons"] == ["x"] and g["candidates"][0]["chosen"] == 1
    assert client.post("/api/generations/t1/favorite").json()["favorite"] is True
    assert client.get("/api/generations?favorites=true").json()["total"] == 1
    assert client.get("/api/stats").json()["n"] == 1
    assert client.delete("/api/generations/t1").status_code == 200
