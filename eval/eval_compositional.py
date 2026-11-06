"""Semantic consistency on complex prompts (multiple objects, counts, colours, spatial relations).

For every prompt in prompts/compositional.json (annotated ground truth) and every seed,
OWLv2 detects the requested objects plus distractor classes, colours are checked with HSV
statistics inside the boxes and relations geometrically. Reported per system:

* object-level precision / recall / F1 (micro-averaged; FP = detected distractor classes)
* counting, colour, spatial accuracy and the share of prompts with *all* constraints satisfied
* CLIP-L/14 score
* parser accuracy: does our rule-based prompt analyser recover the annotated scene graph?

Usage: python -m eval.eval_compositional [--seeds 2] [--systems sd15_fp16 sd15_lcm4_taesd turbo1_taesd pocketdiff]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from eval.common import RESULTS_DIR, Runner, clip_scores, compositional, load_set, update_json
from pocketdiff.config import PRESETS
from pocketdiff.planner import Budget, family_available
from pocketdiff.prompt_analyzer import PromptAnalysis, Relation, SceneObject, analyze_prompt

DEFAULT = ["sd15_fp16", "sd15_fp16_dpm20", "sd15_lcm4_taesd", "turbo1_taesd", "pocketdiff", "pocketdiff_semantic"]
# pocketdiff_semantic re-ranks its candidates with the same OWLv2 detector that is used for scoring:
# its F1 is an optimistic upper bound (selection and evaluation share a model) and is reported as such.


def gt_analysis(item: dict) -> PromptAnalysis:
    pa = analyze_prompt(item["prompt"])
    pa.objects = [SceneObject(o["name"], o.get("count", 1), o.get("color")) for o in item["objects"]]
    pa.relations = [Relation(r["subject"], r["relation"], r["object"]) for r in item["relations"]]
    return pa


def parser_accuracy(items) -> dict:
    obj_ok = rel_ok = cnt_ok = col_ok = 0
    n_obj = n_rel = 0
    for it in items:
        pa = analyze_prompt(it["prompt"])
        pred = {(o.name, o.count, o.color) for o in pa.objects}
        for o in it["objects"]:
            n_obj += 1
            names = {p[0] for p in pred}
            obj_ok += o["name"] in names
            cnt_ok += any(p[0] == o["name"] and p[1] == o.get("count", 1) for p in pred)
            col_ok += any(p[0] == o["name"] and p[2] == o.get("color") for p in pred)
        prel = {(r.subject, r.relation, r.object) for r in pa.relations}
        for r in it["relations"]:
            n_rel += 1
            rel_ok += (r["subject"], r["relation"], r["object"]) in prel
    return {"object_recall": obj_ok / n_obj, "count_acc": cnt_ok / n_obj, "color_acc": col_ok / n_obj,
            "relation_recall": rel_ok / n_rel if n_rel else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--systems", nargs="*", default=DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (dry runs)")
    args = ap.parse_args()
    items = compositional()[: args.limit]
    rows = [{"id": f"{it['id']}_s{s}", "prompt": it["prompt"], "item": it, "seed": s}
            for it in items for s in range(args.seeds)]
    runner = Runner()
    systems = [s for s in args.systems if s.startswith("pocketdiff") or family_available(PRESETS[s].family)]
    for s in systems:
        if s == "pocketdiff":
            runner.run_set(s, rows, "compositional", pipeline_kwargs=dict(
                budget=Budget(latency_ms=1500, mode="auto"), verify=False, safety=False))
        elif s == "pocketdiff_semantic":
            runner.run_set(s, rows, "compositional", pipeline_kwargs=dict(
                budget=Budget(latency_ms=3000, mode="quality"), verify=False, safety=False,
                semantic_rerank=True))
        elif s == "pocketdiff_semantic8":   # slower operating point: 8 candidates
            runner.run_set(s, rows, "compositional", pipeline_kwargs=dict(
                budget=Budget(latency_ms=3000, mode="quality"), verify=False, safety=False,
                semantic_rerank=True, n=8))
        else:
            runner.run_set(s, rows, "compositional")
    runner.pd.engine.models.evict_all()

    from pocketdiff.verifier import CompositionVerifier
    ver = CompositionVerifier()
    update_json(RESULTS_DIR / "compositional.json", "_parser", parser_accuracy(items))
    for s in systems:
        imgs, prompts, meta = load_set("compositional", s, rows)
        if len(imgs) != len(rows):
            print(f"{s}: incomplete ({len(imgs)}/{len(rows)})")
            continue
        tp = fp = fn = 0
        checks = defaultdict(list)
        per_cat = defaultdict(list)
        per_prompt = []
        for img, r in zip(imgs, rows):
            vr = ver.verify(img, gt_analysis(r["item"]), seed=r["seed"])
            tp, fp, fn = tp + vr.tp, fp + vr.fp, fn + vr.fn
            for c in vr.count_checks:
                checks["count"].append(c["ok"])
            for c in vr.color_checks:
                checks["color"].append(c["ok"])
            for c in vr.relation_checks:
                checks["spatial"].append(c["ok"])
            per_cat[r["item"]["category"]].append(vr.all_satisfied)
            per_prompt.append({"id": r["id"], "prompt": r["prompt"], "category": r["item"]["category"],
                               "f1": vr.f1, "all_ok": vr.all_satisfied, "fn": vr.fn, "fp": vr.fp,
                               "counts": vr.count_checks, "colors": vr.color_checks,
                               "relations": vr.relation_checks})
        p = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 1.0
        cs = clip_scores(imgs, prompts)
        res = {"n_images": len(imgs), "object_precision": p, "object_recall": rc,
               "object_f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
               "count_acc": float(np.mean(checks["count"])) if checks["count"] else None,
               "color_acc": float(np.mean(checks["color"])) if checks["color"] else None,
               "spatial_acc": float(np.mean(checks["spatial"])) if checks["spatial"] else None,
               "all_constraints_rate": float(np.mean([x["all_ok"] for x in per_prompt])),
               "per_category_all_ok": {k: float(np.mean(v)) for k, v in per_cat.items()},
               "clip_mean": float(cs.mean()),
               "gen_ms_mean": float(np.mean([m["gen_ms"] for m in meta.values()])),
               "per_prompt": per_prompt}
        update_json(RESULTS_DIR / "compositional.json", s, res)
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
        print(f"{s:22s} F1 {res['object_f1']:.3f} (P {p:.3f} R {rc:.3f})  count {fmt(res['count_acc'])}  "
              f"color {fmt(res['color_acc'])}  spatial {fmt(res['spatial_acc'])}  all-ok {res['all_constraints_rate']:.2f}  "
              f"CLIP {res['clip_mean']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
