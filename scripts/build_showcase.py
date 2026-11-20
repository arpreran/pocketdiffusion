"""Build the offline showcase shown on the permanent GitHub Pages page (deploy/pages/showcase/).

When the team laptop is asleep or offline, the permanent page cannot open the live demo; it then
shows this gallery of real results instead (prompt, model, steps, latency, CLIP score, Explain maps
and an image-to-image edit), plus a short recording of the app.

  python scripts/build_showcase.py --generate   # run the curated prompts on the local server first
  python scripts/build_showcase.py              # (re-)export from the ids in showcase/selection.json

Needs the local server (python serve.py) for --generate and Explain maps; the export itself only
reads data/store and the database.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "store"
DB = ROOT / "data" / "pocketdiffusion.db"
OUT = ROOT / "deploy" / "pages" / "showcase"
SELECTION = OUT / "selection.json"
API = "http://127.0.0.1:8000"

# (style label, prompt, seed) - each prompt shows one ability; seeds picked by looking at results
CURATED = [
    ("Composition", "a red apple to the left of a green bottle on a wooden table, studio photo", 2),
    ("Photo", "a corgi wearing sunglasses on a skateboard, golden hour", 7),
    ("Photo", "a cozy wooden cabin in a snowy forest at night, warm light in the windows", 21),
    ("Cinematic", "a futuristic city skyline at dusk, neon reflections on wet streets, cinematic film still", 5),
    ("Portrait", "portrait photo of an elderly fisherman with a white beard, natural light, 85mm", 3),
    ("Counting", "two cats sleeping on a blue couch next to a laptop", 5),
    ("Anime", "a girl with a red umbrella at a rainy train station, anime style, cel shading, vibrant colors, clean line art", 8),
    ("Watercolor", "a lighthouse on a cliff above a stormy sea, watercolor painting, soft washes, paper texture", 4),
    ("Pixel art", "a tiny fox exploring a mushroom forest, pixel art, 16-bit, crisp pixels", 2),
    ("3D render", "a cute robot watering plants in a greenhouse, 3d render, soft studio lighting, high detail", 9),
    ("Oil painting", "a bowl of sunflowers on a windowsill, oil painting, visible brush strokes, canvas texture", 12),
    ("Cinematic", "an astronaut riding a horse on the moon, cinematic film still, dramatic lighting", 3),
]
# indices whose cross-attention maps are shown -> the words to show (the objects tell the story)
EXPLAIN = {0: ["apple", "bottle", "table"], 1: ["corgi", "sunglasses", "skateboard"],
           11: ["astronaut", "horse", "moon"]}
EDIT = (11, "an astronaut riding a horse on the moon, watercolor painting, soft washes, paper texture", 0.9)
FAMILY = {"sd_turbo": "SD-Turbo", "sd15": "SD1.5 + LCM"}


def run_job(client: httpx.Client, path: str, body: dict) -> dict:
    job = client.post(f"{API}{path}", json=body).raise_for_status().json()
    event = None
    with client.stream("GET", f"{API}/api/jobs/{job['job_id']}/events", timeout=300) as r:
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event in ("result", "error"):
                data = json.loads(line[5:])
                if event == "error":
                    raise RuntimeError(data.get("message"))
                return data
    raise RuntimeError("job ended without a result")


def generate() -> dict:
    sel = {"items": [], "edit": None}
    with httpx.Client(timeout=60) as c:
        for i, (style, prompt, seed) in enumerate(CURATED):
            g = run_job(c, "/api/generate", {"prompt": prompt, "seed": seed, "mode": "quality", "width": 512,
                                             "height": 512, "latency_ms": 1500})
            print(f"{i:2d} {style:12s} {g['latency_ms']:6.0f} ms  CLIP {g['clip'] or 0:5.1f}  {g['id']}")
            sel["items"].append({"style": style, "id": g["id"]})
            if i in EXPLAIN:
                c.post(f"{API}/api/generations/{g['id']}/attention", timeout=300).raise_for_status()
        src = sel["items"][EDIT[0]]["id"]
        e = run_job(c, "/api/edit", {"prompt": EDIT[1], "strength": EDIT[2], "seed": 3, "source_id": src})
        sel["edit"] = {"id": e["id"], "source_id": src}
        print("edit", e["id"], f"{e['latency_ms']:.0f} ms")
    OUT.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    return sel


def webp(src: Path, dst: Path, size: int, q: int = 80) -> str:
    im = Image.open(src).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "WEBP", quality=q, method=6)
    return dst.relative_to(OUT.parent).as_posix()


def export(sel: dict) -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    img = OUT / "img"
    items = []
    for k, it in enumerate(sel["items"]):
        g = con.execute("SELECT * FROM generations WHERE id=?", (it["id"],)).fetchone()
        if g is None:
            print("missing", it["id"]); continue
        d = {"style": it["style"], "prompt": g["prompt"], "model": FAMILY.get(g["family"], g["family"]),
             "steps": g["steps_done"], "candidates": g["n_candidates"], "latency_ms": round(g["latency_ms"]),
             "clip": round(g["clip"], 1) if g["clip"] is not None else None, "size": f"{g['width']}×{g['height']}",
             "image": webp(STORE / g["image_path"], img / f"{k:02d}.webp", 768),
             "thumb": webp(STORE / g["image_path"], img / f"{k:02d}_t.webp", 384, 76)}
        attn = STORE / it["id"] / "attn" / "attention.json"
        if attn.exists():
            a = json.loads(attn.read_text(encoding="utf-8"))
            by = {w["word"]: w for w in a["words"]}
            words = [by[w] for w in EXPLAIN.get(k, []) if w in by] or                 sorted(a["words"], key=lambda w: -w["share"])[:3]
            d["explain"] = {
                "segmentation": webp(STORE / a["segmentation"], img / f"{k:02d}_seg.webp", 640),
                "words": [{"word": w["word"], "color": w["color"], "share": w["share"],
                           "image": webp(STORE / w["image"], img / f"{k:02d}_w{j}.webp", 640)}
                          for j, w in enumerate(words)]}
        items.append(d)
    edit = None
    if sel.get("edit"):
        e = con.execute("SELECT * FROM generations WHERE id=?", (sel["edit"]["id"],)).fetchone()
        if e is not None:
            m = json.loads(e["metrics_json"] or "{}").get("edit", {})
            edit = {"prompt": e["prompt"], "strength": m.get("strength"), "latency_ms": round(e["latency_ms"]),
                    "source": webp(STORE / e["raw_path"], img / "edit_source.webp", 640),
                    "result": webp(STORE / e["image_path"], img / "edit_result.webp", 640)}
    data = {"items": items, "edit": edit, "video": "showcase/walkthrough.webp"
            if (OUT / "walkthrough.webp").exists() else None}
    # link-preview image (Open Graph): 4 x 2 collage, 1200 x 630 JPEG
    og = Image.new("RGB", (1200, 630), (11, 13, 23))
    for k, it in enumerate(items[:8]):
        im = Image.open(OUT.parent / it["thumb"]).convert("RGB").resize((300, 300), Image.LANCZOS)
        og.paste(im, ((k % 4) * 300, 15 + (k // 4) * 300))
    og.save(OUT / "og.jpg", quality=84)
    (OUT / "showcase.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"showcase: {len(items)} images, explain on {sum('explain' in i for i in items)}, "
          f"edit {'yes' if edit else 'no'} -> {OUT}  ({total / 2**20:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="run the curated prompts on the local server")
    args = ap.parse_args()
    if args.generate:
        sel = generate()
    elif SELECTION.exists():
        sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    else:
        sys.exit("no selection yet - run with --generate (local server must be running)")
    export(sel)


if __name__ == "__main__":
    main()
