"""Reproduce every number in the report with one command.

    python -m eval.run_all            # full run (~2-3 h on an RTX 4060 Laptop GPU)
    python -m eval.run_all --quick    # smaller N for a fast sanity check (~25 min)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("prepare data", [sys.executable, "scripts/prepare_data.py"]),
    ("fit NIQE", [sys.executable, "scripts/fit_niqe.py"]),
    ("distil text encoder", [sys.executable, "scripts/distill_text_encoder.py"]),
    ("efficiency benchmark", [sys.executable, "-m", "eval.bench_efficiency", "--cpu"]),
    ("degradation + calibration", [sys.executable, "-m", "eval.eval_degradation"]),
    ("quality (CLIP/FID/KID)", [sys.executable, "-m", "eval.eval_quality"]),
    ("compositional (F1)", [sys.executable, "-m", "eval.eval_compositional"]),
    ("ablation", [sys.executable, "-m", "eval.eval_ablation"]),
    ("no-reference IQA on test images", [sys.executable, "-m", "eval.eval_iqa_test"]),
    ("report", [sys.executable, "-m", "eval.make_report"]),
    ("report figures", [sys.executable, "scripts/make_report_figures.py"]),
]
QUICK = {"degradation + calibration": ["--n", "30"], "quality (CLIP/FID/KID)": ["--n", "80"],
         "compositional (F1)": ["--seeds", "1"], "ablation": ["--n", "40"],
         "distil text encoder": ["--steps", "1500"], "efficiency benchmark": ["--runs", "3"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-step", type=int, default=0)
    args = ap.parse_args()
    for i, (name, cmd) in enumerate(STEPS):
        if i < args.from_step:
            continue
        if args.quick:
            cmd = cmd + QUICK.get(name, [])
        print(f"\n=== [{i}] {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"=== [{i}] {name} finished in {(time.time() - t0) / 60:.1f} min (exit {rc})", flush=True)
        if rc != 0:
            print("stopping; fix the error and resume with --from-step", i)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
