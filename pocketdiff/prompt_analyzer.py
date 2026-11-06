"""Rule-based prompt understanding.

Turns a free-form prompt into a small *scene graph* (objects, counts, colour attributes,
spatial relations) and a complexity score in [0, 1]. The scene graph drives

* the adaptive planner (complex prompts get more candidates / steps),
* the compositional verifier (are the requested objects, counts, colours, relations present?),
* the compositional F1 benchmark.

No neural network is used here on purpose: it is deterministic, explainable, and costs <1 ms.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# 80 COCO categories (+ common synonyms) -> canonical name. The open-vocabulary detector can
# handle any noun, but restricting the vocabulary keeps the verification reliable.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "child": "person",
    "kid": "person", "people": "person", "men": "person", "women": "person", "children": "person",
    "lady": "person", "guy": "person", "astronaut": "person", "player": "person",
    "bike": "bicycle", "motorbike": "motorcycle", "plane": "airplane", "aeroplane": "airplane",
    "jet": "airplane", "ship": "boat", "sofa": "couch", "table": "dining table",
    "desk": "dining table", "television": "tv", "monitor": "tv", "phone": "cell phone",
    "smartphone": "cell phone", "mobile phone": "cell phone", "fridge": "refrigerator",
    "puppy": "dog", "kitten": "cat", "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "basketball": "sports ball", "plant": "potted plant",
    "doughnut": "donut", "glass": "wine glass", "mug": "cup", "teddy": "teddy bear",
    "racket": "tennis racket", "bag": "handbag", "purse": "handbag", "lorry": "truck",
    "taxi": "car", "hotdog": "hot dog",
}
IRREGULAR_PLURALS = {"people": "person", "men": "person", "women": "person",
                     "children": "person", "mice": "mouse", "knives": "knife",
                     "sheep": "sheep", "skis": "skis", "buses": "bus", "benches": "bench",
                     "sandwiches": "sandwich", "boxes": "box", "glasses": "glass",
                     "couches": "couch", "brushes": "toothbrush"}

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black",
          "white", "gray", "grey", "silver", "gold", "cyan", "teal", "violet", "beige"]
COLOR_CANON = {"grey": "gray", "silver": "gray", "gold": "yellow", "teal": "cyan",
               "violet": "purple", "beige": "white"}

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair of": 2, "three": 3,
                "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "couple of": 2}

# relation phrase -> canonical relation (ordered longest first when matching)
RELATIONS = {
    "to the left of": "left_of", "on the left of": "left_of", "left of": "left_of",
    "to the right of": "right_of", "on the right of": "right_of", "right of": "right_of",
    "on top of": "above", "above": "above", "over": "above", "on": "on",
    "below": "below", "under": "below", "underneath": "below", "beneath": "below",
    "next to": "next_to", "beside": "next_to", "near": "next_to", "by": "next_to",
    "in front of": "in_front_of", "behind": "behind",
}
STYLE_WORDS = ["photo", "photograph", "painting", "oil painting", "watercolor", "sketch",
               "anime", "cartoon", "3d render", "render", "cinematic", "portrait",
               "illustration", "pixel art", "digital art", "realistic", "macro"]

# Resolution words in a prompt ("8K", "4K UHD") are only style tokens for the diffusion model:
# they never change the output size. The app maps them to an output size that is reached with
# classical super-resolution (see pipeline.py), longest match first.
RESOLUTION_WORDS = [
    (r"\b(8k|4320p)\b", "8K", (7680, 4320)),
    (r"\b(4k|uhd|2160p)\b", "4K", (3840, 2160)),
    (r"\b(2k|qhd|1440p)\b", "2K", (2560, 1440)),
    (r"\b(full\s*hd|fhd|1080p)\b", "Full HD", (1920, 1080)),
]


def requested_output(text: str) -> tuple[str, tuple[int, int]] | None:
    low = text.lower()
    for pat, label, size in RESOLUTION_WORDS:
        if re.search(pat, low):
            return label, size
    return None


def resolve_output_size(prompt: str, width: int | None, height: int | None) -> tuple[int, int]:
    """Explicit width/height win; 0/None means 'auto': the size named in the prompt, else 512x512."""
    if width and height:
        return int(width), int(height)
    req = requested_output(prompt)
    return req[1] if req else (512, 512)


# Small, deliberately conservative block-list (the image-level CLIP check is the second line).
BLOCKLIST = ["nude", "naked", "nsfw", "porn", "sex", "explicit", "gore", "beheading",
             "child abuse", "rape"]


@dataclass
class SceneObject:
    name: str
    count: int = 1
    color: str | None = None
    span: tuple[int, int] = (0, 0)   # word index span in the prompt


@dataclass
class Relation:
    subject: str
    relation: str
    object: str


@dataclass
class PromptAnalysis:
    prompt: str
    clean_prompt: str
    objects: list[SceneObject] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    n_words: int = 0
    n_tokens: int | None = None
    truncated: bool = False
    n_chunks: int = 1
    requested_size: tuple[int, int] | None = None
    requested_size_label: str | None = None
    complexity: float = 0.0
    complexity_label: str = "simple"
    blocked_terms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_instances(self) -> int:
        return sum(o.count for o in self.objects)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_instances"] = self.total_instances
        return d

    def summary(self) -> str:
        if not self.objects:
            objs = "no countable COCO objects recognised"
        else:
            objs = ", ".join(
                f"{o.count}x {o.color + ' ' if o.color else ''}{o.name}" for o in self.objects)
        rels = "; ".join(f"{r.subject} {r.relation.replace('_', ' ')} {r.object}"
                         for r in self.relations) or "none"
        return (f"objects: {objs} | relations: {rels} | complexity: {self.complexity:.2f} "
                f"({self.complexity_label})")


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _singular(word: str) -> str:
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[:-2] in ("bus", "box", "glass", "bench"):
        return word[:-2]
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _canonical(phrase: str) -> str | None:
    if phrase in SYNONYMS:
        return SYNONYMS[phrase]
    if phrase in COCO_CLASSES:
        return phrase
    return None


def sanitize_prompt(prompt: str, max_chars: int = 2000) -> str:   # long-prompt mode: ~225 tokens
    """Strip control characters / excessive whitespace; cap length (bad-input handling)."""
    if prompt is None:
        return ""
    prompt = "".join(ch for ch in str(prompt) if ch.isprintable() or ch == " ")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt[:max_chars]


def analyze_prompt(prompt: str, tokenizer=None) -> PromptAnalysis:
    clean = sanitize_prompt(prompt)
    pa = PromptAnalysis(prompt=prompt or "", clean_prompt=clean)
    low = clean.lower()
    words = _WORD_RE.findall(low)
    pa.n_words = len(words)

    if not clean:
        pa.warnings.append("Prompt is empty.")
        return pa

    pa.blocked_terms = sorted({b for b in BLOCKLIST if re.search(rf"\b{re.escape(b)}\b", low)})

    if tokenizer is not None:
        from .utils import MAX_PROMPT_CHUNKS, _body_ids, prompt_token_chunks
        n_body = len(_body_ids(tokenizer, clean))
        pa.n_tokens = n_body + 2
        chunks = prompt_token_chunks(tokenizer, clean)
        pa.n_chunks = len(chunks)
        used = sum(len(c) for c in chunks)
        if used < n_body:
            pa.truncated = True
            pa.warnings.append(
                f"Prompt has {n_body} tokens; long-prompt mode reads up to {MAX_PROMPT_CHUNKS} chunks "
                f"of 75 ({used} tokens). The rest is ignored - shorten it or put key objects first.")
        elif pa.n_chunks > 1:
            pa.warnings.append(
                f"Long prompt ({n_body} tokens): encoded in {pa.n_chunks} chunks of up to 75 tokens "
                f"(CLIP reads 77 at a time).")

    req = requested_output(clean)
    if req:
        pa.requested_size_label, pa.requested_size = req[0], req[1]

    pa.styles = [s for s in STYLE_WORDS if re.search(rf"\b{re.escape(s)}\b", low)]

    # ---- object detection over word n-grams (longest match first)
    i = 0
    prev_end = 0
    objects: list[SceneObject] = []
    while i < len(words):
        matched = None
        for n in (3, 2, 1):
            if i + n > len(words):
                continue
            gram = " ".join(words[i:i + n])
            last = words[i + n - 1]
            cand = _canonical(gram)
            plural = last in IRREGULAR_PLURALS and IRREGULAR_PLURALS[last] != last
            if cand is None:
                sing = " ".join(words[i:i + n - 1] + [_singular(last)])
                cand = _canonical(sing)
                plural = cand is not None
            elif last in ("people", "men", "women", "children"):
                plural = True
            if cand and n == 1 and last in COLORS and i + 1 < len(words) and (
                    _canonical(words[i + 1]) or _canonical(_singular(words[i + 1]))):
                cand = None  # "orange cat": colour adjective, not the fruit
            if cand:
                matched = (cand, n, plural)
                break
        if matched is None:
            i += 1
            continue
        name, n, plural = matched
        # look back (max 4 words, never past the previous object) for count / colour modifiers
        count, color, explicit_count = 1, None, False
        for j in range(max(prev_end, i - 4), i):
            w = words[j]
            two = " ".join(words[j:j + 2])
            if two in NUMBER_WORDS:
                count, explicit_count = NUMBER_WORDS[two], True
            elif w in NUMBER_WORDS:
                count, explicit_count = NUMBER_WORDS[w], True
            elif w.isdigit() and 0 < int(w) < 20:
                count, explicit_count = int(w), True
            if w in COLORS:
                color = COLOR_CANON.get(w, w)
        if plural and (not explicit_count or count == 1):
            count = max(count, 2)  # unspecified plural ("dogs") -> at least two
        objects.append(SceneObject(name=name, count=count, color=color, span=(i, i + n)))
        i += n
        prev_end = i
    pa.objects = objects

    # ---- spatial relations: phrase between two objects
    rel_phrases = sorted(RELATIONS, key=lambda s: -len(s.split()))
    for k in range(len(objects) - 1):
        a, b = objects[k], objects[k + 1]
        between = " ".join(words[a.span[1]:b.span[0]])
        for phrase in rel_phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", between):
                rel = RELATIONS[phrase]
                if rel == "on":
                    rel = "above"
                pa.relations.append(Relation(a.name, rel, b.name))
                break

    # ---- complexity score (weights chosen so a 2-object + colour + relation prompt ~ 0.6)
    n_obj = len(objects)
    n_attr = sum(o.color is not None for o in objects)
    n_count = sum(o.count > 1 for o in objects)
    n_rel = len(pa.relations)
    length_term = min(pa.n_words / 40.0, 1.0)
    score = (0.18 * max(n_obj - 1, 0) + 0.12 * n_attr + 0.15 * n_count + 0.2 * n_rel
             + 0.15 * length_term)
    pa.complexity = float(min(score, 1.0))
    pa.complexity_label = ("simple" if pa.complexity < 0.25 else
                           "moderate" if pa.complexity < 0.55 else "complex")
    return pa
