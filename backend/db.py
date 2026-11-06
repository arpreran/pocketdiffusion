"""SQLite persistence for generations, candidates and post-processing stages.

Standard-library sqlite3 only (no ORM, no server). WAL mode allows the API threads to read
while the GPU worker writes. Every public function opens its own short-lived connection,
which is the simplest thread-safe pattern for SQLite.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'generate',      -- generate | compare
    prompt          TEXT NOT NULL,
    negative_prompt TEXT,
    seed            INTEGER,
    device_profile  TEXT,
    mode            TEXT,
    latency_budget  REAL,
    memory_budget   REAL,
    family          TEXT,
    scheduler       TEXT,
    steps           INTEGER,
    steps_done      INTEGER,
    width           INTEGER,
    height          INTEGER,
    quant           TEXT,
    decoder         TEXT,
    text_encoder    TEXT,
    n_candidates    INTEGER,
    latency_ms      REAL,
    gen_ms          REAL,
    peak_mem_mb     REAL,
    weights_mb      REAL,
    clip            REAL,
    iqa_before      REAL,
    iqa_after       REAL,
    niqe_before     REAL,
    niqe_after      REAL,
    object_f1       REAL,
    all_ok          INTEGER,
    complexity      REAL,
    early_exit      INTEGER,
    fallback        INTEGER,
    favorite        INTEGER NOT NULL DEFAULT 0,
    image_path      TEXT,
    raw_path        TEXT,
    heatmap_path    TEXT,
    verify_path     TEXT,
    plan_json       TEXT,
    analysis_json   TEXT,
    config_json     TEXT,
    timings_json    TEXT,
    metrics_json    TEXT,
    events_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_gen_created ON generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gen_family ON generations(family);

CREATE TABLE IF NOT EXISTS candidates (
    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    seed          INTEGER,
    clip          REAL,
    iqa           REAL,
    score         REAL,
    chosen        INTEGER,
    image_path    TEXT,
    PRIMARY KEY (generation_id, idx)
);

CREATE TABLE IF NOT EXISTS stages (
    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    name          TEXT NOT NULL,
    ms            REAL,
    params_json   TEXT,
    image_path    TEXT,
    PRIMARY KEY (generation_id, idx)
);

CREATE TABLE IF NOT EXISTS frames (
    generation_id TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    image_path    TEXT,
    PRIMARY KEY (generation_id, step)
);
"""

LIST_COLUMNS = ("id, created_at, kind, prompt, seed, device_profile, mode, family, scheduler, steps, "
                "width, height, quant, n_candidates, latency_ms, peak_mem_mb, clip, iqa_after, "
                "object_f1, all_ok, favorite, image_path")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ writes
    def insert_generation(self, row: dict, candidates: list[dict], stages: list[dict],
                          frames: list[dict]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.connect() as c:
            c.execute(f"INSERT INTO generations ({cols}) VALUES ({qs})", list(row.values()))
            c.executemany("INSERT INTO candidates (generation_id, idx, seed, clip, iqa, score, chosen, image_path) "
                          "VALUES (:generation_id, :idx, :seed, :clip, :iqa, :score, :chosen, :image_path)", candidates)
            c.executemany("INSERT INTO stages (generation_id, idx, name, ms, params_json, image_path) "
                          "VALUES (:generation_id, :idx, :name, :ms, :params_json, :image_path)", stages)
            c.executemany("INSERT INTO frames (generation_id, step, image_path) "
                          "VALUES (:generation_id, :step, :image_path)", frames)

    def toggle_favorite(self, gid: str) -> bool | None:
        with self.connect() as c:
            r = c.execute("SELECT favorite FROM generations WHERE id=?", (gid,)).fetchone()
            if r is None:
                return None
            new = 0 if r["favorite"] else 1
            c.execute("UPDATE generations SET favorite=? WHERE id=?", (new, gid))
            return bool(new)

    def delete(self, gid: str) -> bool:
        with self.connect() as c:
            cur = c.execute("DELETE FROM generations WHERE id=?", (gid,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------ reads
    def list(self, limit: int = 24, offset: int = 0, favorites: bool = False, family: str | None = None,
             q: str | None = None) -> tuple[list[dict], int]:
        where, args = ["kind = 'generate'"], []
        if favorites:
            where.append("favorite = 1")
        if family:
            where.append("family = ?")
            args.append(family)
        if q:
            where.append("prompt LIKE ?")
            args.append(f"%{q}%")
        w = " AND ".join(where)
        with self.connect() as c:
            total = c.execute(f"SELECT COUNT(*) FROM generations WHERE {w}", args).fetchone()[0]
            rows = c.execute(f"SELECT {LIST_COLUMNS} FROM generations WHERE {w} "
                             "ORDER BY created_at DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return [dict(r) for r in rows], total

    def get(self, gid: str) -> dict | None:
        with self.connect() as c:
            r = c.execute("SELECT * FROM generations WHERE id=?", (gid,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["candidates"] = [dict(x) for x in c.execute(
                "SELECT * FROM candidates WHERE generation_id=? ORDER BY idx", (gid,))]
            d["stages"] = [dict(x) for x in c.execute(
                "SELECT * FROM stages WHERE generation_id=? ORDER BY idx", (gid,))]
            d["frames"] = [dict(x) for x in c.execute(
                "SELECT * FROM frames WHERE generation_id=? ORDER BY step", (gid,))]
        for k in list(d):
            if k.endswith("_json") and d[k]:
                d[k[:-5]] = json.loads(d.pop(k))
            elif k.endswith("_json"):
                d[k[:-5]] = None
                d.pop(k)
        for s in d["stages"]:
            s["params"] = json.loads(s.pop("params_json") or "{}")
        return d

    def stats(self, baseline_ms: float | None) -> dict:
        with self.connect() as c:
            g = c.execute("""SELECT COUNT(*) n, AVG(latency_ms) lat, AVG(clip) clip, AVG(iqa_after) iqa,
                                    AVG(peak_mem_mb) mem, SUM(latency_ms) total_ms, AVG(object_f1) f1,
                                    SUM(favorite) favs, MIN(latency_ms) best
                             FROM generations WHERE kind='generate'""").fetchone()
            fam = [dict(r) for r in c.execute(
                """SELECT family, COUNT(*) n, AVG(latency_ms) lat, AVG(clip) clip
                   FROM generations WHERE kind='generate' GROUP BY family ORDER BY n DESC""")]
            dev = [dict(r) for r in c.execute(
                """SELECT device_profile, COUNT(*) n FROM generations WHERE kind='generate'
                   GROUP BY device_profile ORDER BY n DESC""")]
            recent = [dict(r) for r in c.execute(
                """SELECT created_at, latency_ms, clip FROM generations WHERE kind='generate'
                   ORDER BY created_at DESC LIMIT 40""")]
        out = dict(g)
        out["by_family"] = fam
        out["by_device"] = dev
        out["recent"] = list(reversed(recent))
        if baseline_ms and out["n"]:
            out["baseline_ms"] = baseline_ms
            out["time_saved_s"] = (baseline_ms * out["n"] - (out["total_ms"] or 0)) / 1000.0
        out["generated_at"] = time.time()
        return out
