"""Compositional verification: does the image contain what the prompt asked for?

Hybrid deep + classical:
* OWLv2 open-vocabulary detector finds the requested objects (and distractor classes, so
  false positives can be counted -> precision / recall / F1).
* Colour attributes are checked *classically*: HSV hue/saturation/value statistics inside the
  detected box (centre crop) are mapped to a basic colour name.
* Spatial relations are checked geometrically from box centres.

Used for (a) the compositional benchmark (object-level F1, count / colour / spatial accuracy)
and (b) the UI's semantic-consistency panel.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

from .config import MODEL_IDS
from .prompt_analyzer import COCO_CLASSES, PromptAnalysis

QUERY_ALIASES = {"tv": "television", "couch": "sofa", "dining table": "table",
                 "sports ball": "ball", "cell phone": "mobile phone", "hair drier": "hair dryer"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in pixels

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


@dataclass
class VerificationResult:
    detections: list[Detection]
    tp: int = 0
    fp: int = 0
    fn: int = 0
    count_checks: list[dict] = field(default_factory=list)
    color_checks: list[dict] = field(default_factory=list)
    relation_checks: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def all_satisfied(self) -> bool:
        checks = self.count_checks + self.color_checks + self.relation_checks
        return self.fn == 0 and all(c["ok"] for c in checks)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1,
                 all_satisfied=self.all_satisfied)
        return d


# ------------------------------------------------------------------ classical colour naming
def dominant_color(img01: np.ndarray, box) -> tuple[str, dict]:
    """Name the dominant colour inside the central 60% of the box via HSV statistics."""
    h, w = img01.shape[:2]
    x0, y0, x1, y1 = box
    cx0, cx1 = int(x0 + 0.2 * (x1 - x0)), int(x1 - 0.2 * (x1 - x0))
    cy0, cy1 = int(y0 + 0.2 * (y1 - y0)), int(y1 - 0.2 * (y1 - y0))
    cx0, cy0 = max(cx0, 0), max(cy0, 0)
    cx1, cy1 = min(max(cx1, cx0 + 1), w), min(max(cy1, cy0 + 1), h)
    crop = img01[cy0:cy1, cx0:cx1].astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    votes: dict[str, float] = {}

    def vote(name, mask):
        votes[name] = votes.get(name, 0.0) + float(mask.sum())

    achrom = S < 0.18
    vote("black", achrom & (V < 0.22))
    vote("white", achrom & (V > 0.78))
    vote("gray", achrom & (V >= 0.22) & (V <= 0.78))
    chrom = ~achrom & (V >= 0.15)
    hue_bins = [("red", 0, 15), ("orange", 15, 40), ("yellow", 40, 68), ("green", 68, 165),
                ("cyan", 165, 195), ("blue", 195, 255), ("purple", 255, 290), ("pink", 290, 340),
                ("red", 340, 361)]
    for name, lo, hi in hue_bins:
        m = chrom & (H >= lo) & (H < hi)
        if name == "orange":   # dark orange reads as brown
            vote("brown", m & (V < 0.55))
            m = m & (V >= 0.55)
        if name == "red":
            vote("pink", m & (S < 0.45) & (V > 0.6))
            m = m & ~((S < 0.45) & (V > 0.6))
        vote(name, m)
    total = sum(votes.values()) or 1.0
    dist = {k: v / total for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
    return next(iter(dist)), dist


COLOR_COMPAT = {"gray": {"gray", "white", "black"}, "pink": {"pink", "red", "purple"},
                "brown": {"brown", "orange", "red"}, "purple": {"purple", "blue", "pink"},
                "cyan": {"cyan", "blue", "green"}, "orange": {"orange", "red", "brown", "yellow"}}


def color_matches(expected: str, dist: dict) -> bool:
    ok = COLOR_COMPAT.get(expected, {expected})
    share = sum(v for k, v in dist.items() if k in ok)
    return share >= 0.25


# ------------------------------------------------------------------ geometry
def relation_holds(rel: str, a: Detection, b: Detection, img_w: int, img_h: int) -> bool:
    (ax, ay), (bx, by) = a.center, b.center
    tol_x, tol_y = 0.03 * img_w, 0.03 * img_h
    if rel == "left_of":
        return ax + tol_x < bx
    if rel == "right_of":
        return ax > bx + tol_x
    if rel == "above":
        return ay + tol_y < by
    if rel == "below":
        return ay > by + tol_y
    if rel == "next_to":
        gap = max(0.0, max(a.box[0], b.box[0]) - min(a.box[2], b.box[2]))
        return gap < 0.25 * img_w and abs(ay - by) < 0.35 * img_h
    if rel == "in_front_of":   # nearer objects sit lower in the frame / appear larger
        return a.box[3] >= b.box[3] - tol_y
    if rel == "behind":
        return a.box[3] <= b.box[3] + tol_y
    return False


def _nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    dets = sorted(dets, key=lambda d: -d.score)
    keep: list[Detection] = []
    for d in dets:
        if all(_iou(d.box, k.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class CompositionVerifier:
    def __init__(self, device: str | None = None, threshold: float | None = None,
                 n_distractors: int = 6, model_id: str | None = None):
        """model_id: OWLv2 (default, used for evaluation) or OWL-ViT v1 (used for in-loop
        selection, so that selection and evaluation never share a model)."""
        self.model_id = model_id or MODEL_IDS["detector"]
        self.is_v2 = "owlv2" in self.model_id
        if self.is_v2:
            from transformers import Owlv2ForObjectDetection as Det, Owlv2Processor as Proc
        else:
            from transformers import OwlViTForObjectDetection as Det, OwlViTProcessor as Proc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Proc.from_pretrained(self.model_id)
        self.model = Det.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        # OWL-ViT v1 produces lower-calibrated scores than v2 -> lower default threshold
        self.threshold = threshold if threshold is not None else (0.25 if self.is_v2 else 0.12)
        self.n_distractors = n_distractors

    def distractors(self, analysis: PromptAnalysis, seed: int = 0) -> list[str]:
        wanted = {o.name for o in analysis.objects}
        pool = [c for c in COCO_CLASSES if c not in wanted and c != "person"]
        rng = np.random.default_rng(zlib.crc32(analysis.clean_prompt.encode()) + seed)
        return list(rng.choice(pool, size=min(self.n_distractors, len(pool)), replace=False))

    @torch.inference_mode()
    def detect(self, img01: np.ndarray, labels: list[str]) -> list[Detection]:
        u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        queries = [f"a photo of a {QUERY_ALIASES.get(l, l)}" for l in labels]
        inputs = self.processor(text=[queries], images=u8, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model(**inputs)
        # OWLv2 pads the image to a square (boxes relative to the padded square); v1 resizes
        side = max(u8.shape[:2])
        hw = [side, side] if self.is_v2 else [u8.shape[0], u8.shape[1]]
        target = torch.tensor([hw], device=self.device)
        post = getattr(self.processor, "post_process_grounded_object_detection", None)
        if post is not None:
            res = post(outputs=out, target_sizes=target, threshold=self.threshold)[0]
        else:
            res = self.processor.post_process_object_detection(
                outputs=out, target_sizes=target, threshold=self.threshold)[0]
        dets = []
        for score, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box.float().cpu()]
            x0, x1 = np.clip([x0, x1], 0, u8.shape[1])
            y0, y1 = np.clip([y0, y1], 0, u8.shape[0])
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            dets.append(Detection(labels[int(lab)], float(score), (x0, y0, x1, y1)))
        by_label: dict[str, list[Detection]] = {}
        for d in dets:
            by_label.setdefault(d.label, []).append(d)
        return [d for ds in by_label.values() for d in _nms(ds)]

    def verify(self, img01: np.ndarray, analysis: PromptAnalysis, seed: int = 0) -> VerificationResult:
        wanted = []
        for o in analysis.objects:
            if o.name not in wanted:
                wanted.append(o.name)
        distract = self.distractors(analysis, seed)
        dets = self.detect(img01, wanted + distract) if wanted else self.detect(img01, distract)
        found = {}
        for d in dets:
            found.setdefault(d.label, []).append(d)
        res = VerificationResult(detections=dets)
        res.tp = sum(1 for w in wanted if w in found)
        res.fn = len(wanted) - res.tp
        res.fp = sum(1 for x in distract if x in found)
        h, w = img01.shape[:2]
        for o in analysis.objects:
            ds = sorted(found.get(o.name, []), key=lambda d: -d.score)
            if o.count > 1:
                res.count_checks.append({"object": o.name, "expected": o.count, "found": len(ds),
                                         "ok": len(ds) == o.count})
            if o.color:
                if ds:
                    name, dist = dominant_color(img01, ds[0].box)
                    ok = color_matches(o.color, dist)
                else:
                    name, ok = None, False
                res.color_checks.append({"object": o.name, "expected": o.color, "found": name,
                                         "ok": ok})
        for r in analysis.relations:
            a = max(found.get(r.subject, []), key=lambda d: d.score, default=None)
            b = max(found.get(r.object, []), key=lambda d: d.score, default=None)
            ok = bool(a and b and a is not b and relation_holds(r.relation, a, b, w, h))
            res.relation_checks.append({"subject": r.subject, "relation": r.relation,
                                        "object": r.object, "ok": ok})
        return res

    @staticmethod
    def draw(img01: np.ndarray, res: VerificationResult, wanted: set[str]) -> np.ndarray:
        out = (np.clip(img01, 0, 1) * 255).astype(np.uint8).copy()
        for d in res.detections:
            color = (40, 200, 60) if d.label in wanted else (230, 60, 50)
            x0, y0, x1, y1 = map(int, d.box)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cv2.putText(out, f"{d.label} {d.score:.2f}", (x0 + 3, max(y0 + 14, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out.astype(np.float32) / 255.0
