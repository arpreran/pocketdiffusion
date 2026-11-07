"""PocketDiffusion orchestrator.

    prompt -> analyse -> plan (budget-aware) -> generate N candidates (anytime, OOM-safe)
           -> artifact screening -> CLIP + IQA re-ranking -> adaptive classical post-processing
           -> (optional) compositional verification -> safety check -> result + full telemetry
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .classical import AdaptivePostProcessor, artifact_heatmap, detect_artifacts, quality_report, super_resolve
from .classical.postprocess import Stage
from .config import MODEL_IDS, GenConfig
from .engine import DiffusionEngine, GenResult
from .planner import Budget, Plan, Planner
from .prompt_analyzer import PromptAnalysis, analyze_prompt
from .utils import LOG


class PromptRejected(ValueError):
    pass


LARGE_OUTPUT_PX = 1536   # longer side above this: post-process at native size, super-resolve last
DETAIL_PASS_MAX_SIDE = 1536   # diffusion detail pass works at <= 2x native and at most this long side
OBJECT_WEIGHT = 1.0   # weight of per-object coverage relative to the whole-prompt CLIP score
_NUM = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
_PLURAL = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
           "bus": "buses", "bench": "benches", "sandwich": "sandwiches", "wine glass": "wine glasses",
           "couch": "couches", "dining table": "tables", "toothbrush": "toothbrushes"}


def object_phrases(analysis: PromptAnalysis) -> list[str]:
    """One short caption per requested object, keeping its count and colour."""
    out = []
    for o in analysis.objects:
        name = o.name if o.name != "dining table" else "table"
        if o.count > 1:
            name = _PLURAL.get(o.name, name if name.endswith("s") else name + "s")
        art = _NUM.get(o.count, str(o.count))
        if o.count == 1 and (o.color or name)[0] in "aeiou":
            art = "an"
        out.append(f"a photo of {art} {o.color + ' ' if o.color else ''}{name}")
    return out


@dataclass
class Candidate:
    image: np.ndarray
    seed: int
    clip: float | None = None
    iqa: float | None = None
    score: float | None = None
    artifacts: dict | None = None


@dataclass
class RunResult:
    image: np.ndarray
    raw_image: np.ndarray
    analysis: PromptAnalysis
    plan: Plan | None
    config: GenConfig
    gen: GenResult
    candidates: list[Candidate]
    chosen: int
    post_stages: list = field(default_factory=list)
    artifacts_before: dict | None = None
    artifacts_after: dict | None = None
    iqa_before: dict | None = None
    iqa_after: dict | None = None
    heatmap: np.ndarray | None = None
    verification: dict | None = None
    verification_image: np.ndarray | None = None
    safety: dict | None = None
    timings_ms: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return float(sum(v for k, v in self.timings_ms.items() if k != "verify"))


class PocketDiffusion:
    def __init__(self, engine: DiffusionEngine | None = None, planner: Planner | None = None):
        self.engine = engine or DiffusionEngine()
        self.planner = planner or Planner()
        self._scorer = None
        self._verifier = None
        self._selector = None

    # lazily created helpers (so the app starts fast and works without optional models)
    @property
    def scorer(self):
        if self._scorer is None:
            from .scoring import ClipScorer
            from .config import MODEL_IDS
            try:
                self._scorer = ClipScorer(MODEL_IDS["clip_rerank"])
            except Exception as e:  # missing download -> re-ranking disabled, not a crash
                LOG.warning("CLIP re-ranker unavailable: %s", e)
                self._scorer = False
        return self._scorer or None

    @property
    def verifier(self):
        if self._verifier is None:
            from .verifier import CompositionVerifier
            try:
                self._verifier = CompositionVerifier()
            except Exception as e:
                LOG.warning("Detector unavailable: %s", e)
                self._verifier = False
        return self._verifier or None

    @property
    def selector(self):
        """Detector used to *choose* among candidates (OWL-ViT v1). Deliberately a different
        model from `verifier` (OWLv2), which scores the F1 benchmark. Falls back to the
        verifier only if the selector weights are not installed."""
        if self._selector is None:
            from .config import MODEL_IDS
            from .verifier import CompositionVerifier
            try:
                self._selector = CompositionVerifier(model_id=MODEL_IDS["selector"])
            except Exception as e:
                LOG.warning("Selector detector unavailable (%s) - falling back to the verifier", e)
                self._selector = self.verifier or False
        return self._selector or None

    def analyze(self, prompt: str) -> PromptAnalysis:
        tok = None
        for b in self.engine.models.cache.values():
            tok = b.tokenizer
            break
        if tok is None:   # no model loaded yet: the SD tokenizers share the CLIP BPE vocabulary
            if getattr(self, "_tok", None) is None:
                try:
                    from transformers import CLIPTokenizer
                    self._tok = CLIPTokenizer.from_pretrained(MODEL_IDS["sd15"], subfolder="tokenizer")
                except Exception as e:
                    LOG.warning("tokenizer unavailable for prompt analysis: %s", e)
                    self._tok = False
            tok = self._tok or None
        return analyze_prompt(prompt, tokenizer=tok)

    # ------------------------------------------------------------------ main entry
    def run(self, prompt: str, budget: Budget | None = None, seed: int = 0,
            negative_prompt: str = "", config: GenConfig | None = None, n: int | None = None,
            postprocess: str | None = None, rerank: bool = True, verify: bool = False,
            preview: bool = False, safety: bool = True, upscale_to: tuple[int, int] | None = None,
            on_step=None, semantic_rerank: bool = False, on_preview=None,
            on_event=None, detail_pass: bool = True) -> RunResult:
        emit = on_event or (lambda name, data=None: None)
        timings: dict[str, float] = {}
        events: list[str] = []
        budget = budget or Budget()

        t0 = time.perf_counter()
        analysis = self.analyze(prompt)
        if not analysis.clean_prompt:
            raise PromptRejected("Please enter a prompt describing the image.")
        if analysis.blocked_terms and safety:
            raise PromptRejected("This prompt contains blocked terms "
                                 f"({', '.join(analysis.blocked_terms)}). Please rephrase.")
        events.extend(analysis.warnings)

        plan = None
        if config is None:
            plan = self.planner.plan(analysis, budget)
            config = plan.config
            n = n or plan.n_candidates
            postprocess = postprocess or plan.postprocess
            upscale_to = upscale_to or plan.upscale_to
            if plan.fallback:
                events.append("Budget infeasible -> fallback plan used.")
        n = max(1, int(n or 1))
        postprocess = postprocess or "full"
        timings["analyse+plan"] = (time.perf_counter() - t0) * 1000
        emit("analysis", analysis.to_dict())
        if plan is not None:
            emit("plan", plan.to_dict())
        emit("stage", {"name": "denoise", "config": config.to_dict(), "n": n})

        deadline = None
        if plan and plan.deadline_ms:
            # leave head-room for screening, re-ranking and post-processing after generation
            deadline = max(plan.deadline_ms - 250.0, 0.6 * plan.deadline_ms)
        gen = self.engine.generate_with_fallback(
            analysis.clean_prompt, config, seed=seed, n=n, negative_prompt=negative_prompt,
            deadline_ms=deadline, preview=preview, on_step=on_step, on_preview=on_preview)
        events.extend(gen.events)
        timings.update({f"gen.{k}": v for k, v in gen.timings_ms.items()})

        emit("stage", {"name": "screen"})
        # ---- cheap screening of every candidate (blank / NaN / exposure / cast / saturation)
        cands = [Candidate(im, s) for im, s in zip(gen.images, gen.seeds)]
        t0 = time.perf_counter()
        reps = [detect_artifacts(c.image, full=False) for c in cands]
        if all(r.catastrophic for r in reps):
            events.append("All candidates blank/NaN -> regenerating with a new seed.")
            gen = self.engine.generate_with_fallback(analysis.clean_prompt, gen.config,
                                                     seed=seed + 1000, n=1)
            cands = [Candidate(im, s) for im, s in zip(gen.images, gen.seeds)]
            reps = [detect_artifacts(c.image, full=False) for c in cands]
        for c, r in zip(cands, reps):
            c.artifacts = r.to_dict()
        timings["screen"] = (time.perf_counter() - t0) * 1000

        # ---- best-of-N re-ranking: CLIP (alignment) + PocketIQA (quality) - artifacts
        chosen = 0
        cand_ver: dict[int, object] = {}
        if len(cands) > 1 and rerank and self.scorer is not None:
            t0 = time.perf_counter()
            # Object-aware re-ranking: besides the whole prompt, every requested object is scored
            # on its own ("a photo of a red apple", "a photo of two cats"). A candidate that drops
            # an object gets a low score for that phrase, so we reward the *weakest* object
            # (min) as well as the average. CLIP ViT-B/32 only - never the evaluation models.
            phrases = object_phrases(analysis)
            clips, parts = self.scorer.score_with_parts([c.image for c in cands], analysis.clean_prompt, phrases)
            for i, (c, cs, r) in enumerate(zip(cands, clips, reps)):
                q = quality_report(c.image, with_niqe=False)["pocket_iqa"]
                c.clip, c.iqa = float(cs), float(q)
                coverage = 0.5 * float(parts[i].min()) + 0.5 * float(parts[i].mean()) if phrases else 0.0
                c.score = (float(cs) + OBJECT_WEIGHT * coverage + 0.05 * (q - 50.0) - 2.0 * r.total
                           - (50.0 if r.catastrophic else 0.0))
            timings["rerank"] = (time.perf_counter() - t0) * 1000
            # optional semantic re-ranking: a detector checks objects / counts / colours /
            # relations of every candidate and the constraint-satisfaction rate is added.
            # It uses the *selector* detector (OWL-ViT v1), never the OWLv2 evaluation detector.
            if semantic_rerank and analysis.objects and self.selector is not None:
                t0 = time.perf_counter()
                for i, c in enumerate(cands):
                    vr = self.selector.verify(c.image, analysis, seed=seed)
                    cand_ver[i] = vr
                    checks = vr.count_checks + vr.color_checks + vr.relation_checks
                    sat = (vr.tp + sum(ch["ok"] for ch in checks)) / max(vr.tp + vr.fn + len(checks), 1)
                    c.score += 6.0 * sat - 2.0 * vr.fp
                timings["semantic_rerank"] = (time.perf_counter() - t0) * 1000
            chosen = int(np.argmax([c.score for c in cands]))
        raw = cands[chosen].image

        emit("stage", {"name": "postprocess", "chosen": chosen})
        # ---- adaptive classical post-processing (full artifact analysis on the chosen image)
        t0 = time.perf_counter()
        pp = AdaptivePostProcessor(level=postprocess, keep_intermediates=True)
        big_target = upscale_to if upscale_to and max(upscale_to) > LARGE_OUTPUT_PX else None
        post = pp(raw, target_size=None if big_target else upscale_to, measure_after=False)
        timings["postprocess"] = (time.perf_counter() - t0) * 1000
        final = post.image

        # ---- quality telemetry (not counted in latency: diagnostics only)
        after = detect_artifacts(final) if postprocess != "off" else None
        iqa_before = quality_report(raw)
        iqa_after = quality_report(final) if postprocess != "off" else iqa_before
        heat = artifact_heatmap(raw)

        # ---- safety (image level)
        safety_res = None
        if safety and self.scorer is not None:
            t0 = time.perf_counter()
            safety_res = self.scorer.safety([final])[0]
            timings["safety"] = (time.perf_counter() - t0) * 1000
            if safety_res["unsafe"]:
                final = cv2.GaussianBlur(final, (0, 0), 25)
                events.append("Image-level safety check flagged the output -> blurred.")

        emit("stage", {"name": "verify"})
        # ---- optional semantic verification (OWLv2 + HSV colour + geometry)
        ver, ver_img = None, None
        if (verify or semantic_rerank) and analysis.objects and self.verifier is not None:
            t0 = time.perf_counter()
            vr = self.verifier.verify(final, analysis, seed=seed)
            ver = vr.to_dict()
            ver_img = self.verifier.draw(final, vr, {o.name for o in analysis.objects})
            timings["verify"] = (time.perf_counter() - t0) * 1000

        # ---- very large outputs (2K/4K/8K): classical super-resolution as the last step, so every
        # metric above ran on the native image and no stage processes a 33-megapixel buffer
        if big_target is not None:
            h0, w0 = final.shape[:2]
            # 1) diffusion detail pass at 2x native (GPU, single-step SD-Turbo img2img): real new detail
            gc = gen.config
            if detail_pass and gc.family == "sd_turbo" and not gc.cfg and gc.device == "cuda":
                emit("stage", {"name": "detail_pass"})
                from .classical.resize import snap_resolution
                rw, rh = snap_resolution(big_target[0], big_target[1], max_pixels=(2 * w0) * (2 * h0),
                                         max_side=DETAIL_PASS_MAX_SIDE)
                try:
                    final, ms = self.engine.refine(final, analysis.clean_prompt, gc, (rw, rh), seed=seed)
                    timings["detail_pass"] = ms
                    post.stages.append(Stage("detail_pass", {"from": [w0, h0], "to": [rw, rh],
                                                             "method": "SD-Turbo img2img, strength 0.5, 1 UNet call"},
                                             ms, final.copy()))
                    events.append(f"Detail pass: SD-Turbo image-to-image at {rw}x{rh} redrew fine detail "
                                  f"({ms:.0f} ms).")
                    h0, w0 = final.shape[:2]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    events.append("Detail pass skipped (out of GPU memory) - classical super-resolution only.")
            # 2) classical super-resolution for the remaining factor
            emit("stage", {"name": "super_resolution", "size": list(big_target)})
            t0 = time.perf_counter()
            final = super_resolve(final, big_target)
            ms = (time.perf_counter() - t0) * 1000
            timings["super_resolution"] = ms
            post.stages.append(Stage("super_resolution", {"from": [w0, h0], "to": list(big_target),
                                                          "method": "IBP (<=4K) + Lanczos"}, ms, None))
            events.append(f"Output {big_target[0]}x{big_target[1]}: classical super-resolution from "
                          f"{w0}x{h0} (x{big_target[0] / w0:.1f}) in {ms:.0f} ms.")

        return RunResult(image=final, raw_image=raw, analysis=analysis, plan=plan, config=gen.config,
                         gen=gen, candidates=cands, chosen=chosen, post_stages=post.stages,
                         artifacts_before=post.before.to_dict(),
                         artifacts_after=after.to_dict() if after else None,
                         iqa_before=iqa_before, iqa_after=iqa_after, heatmap=heat,
                         verification=ver, verification_image=ver_img, safety=safety_res,
                         timings_ms=timings, events=events)
