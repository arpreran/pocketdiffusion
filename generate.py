"""Command-line inference: prompts in, images + a JSON log (plan, timings, scores) out.

    python generate.py --prompt "a red apple to the left of a green bottle"
    python generate.py --prompts prompts/samples.txt --out outputs/
    python generate.py --prompts prompts/samples.txt --device lowend_phone --latency 1500

The first prompt also pays the one-off model loading time (~10 s); later prompts show the real latency.

Every prompt goes through the same adaptive pipeline as the web app: prompt analysis ->
budget-aware plan -> generation -> screening / re-ranking -> classical post-processing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_HUB / "models--stable-diffusion-v1-5--stable-diffusion-v1-5").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # weights cached -> never touch the network


def main() -> int:
    from pocketdiff.config import DEVICE_PROFILES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="a single prompt")
    src.add_argument("--prompts", type=Path, help="text file, one prompt per line")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--device", default="this_gpu", choices=sorted(DEVICE_PROFILES))
    ap.add_argument("--latency", type=float, default=2000.0, help="latency budget in ms")
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "balanced", "quality"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="also run the OWLv2 semantic check")
    args = ap.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [ln.strip() for ln in args.prompts.read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from pocketdiff.pipeline import PocketDiffusion
    from pocketdiff.planner import Budget

    pd = PocketDiffusion()
    budget = Budget(latency_ms=args.latency, mode=args.mode, device_profile=args.device)
    log = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        res = pd.run(prompt, budget=budget, seed=args.seed + i, verify=args.verify)
        ms = (time.perf_counter() - t0) * 1000
        name = f"{i:03d}.png"
        Image.fromarray((res.image.clip(0, 1) * 255).astype("uint8")).save(args.out / name)
        cfg = res.config
        entry = {"file": name, "prompt": prompt, "end_to_end_ms": round(ms, 1),
                 "plan": res.plan.to_dict() if res.plan else cfg.to_dict(),
                 "candidates": len(res.candidates),
                 "post_stages": [s.name for s in res.post_stages],
                 "events": res.events}
        if res.verification:
            entry["verification"] = {k: res.verification.get(k) for k in ("f1", "all_satisfied")}
        log.append(entry)
        print(f"[{i + 1}/{len(prompts)}] {ms:7.0f} ms  {cfg.family} {cfg.steps} step(s) @{cfg.width}px "
              f"N={len(res.candidates)}  -> {args.out / name}", flush=True)
    (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"wrote {len(log)} image(s) and log.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
