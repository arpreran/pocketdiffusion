"""Knowledge distillation: 12-layer CLIP ViT-L/14 text encoder -> 6-layer student.

Teacher : the frozen Stable Diffusion v1.5 text encoder (123M params, 12 transformer layers)
Student : same width, 6 layers, initialised from teacher layers KEEP (depth pruning)
Loss    : MSE + (1 - cosine) between student and teacher *last hidden states over all 77
          positions* - exactly the tensor the UNet cross-attends to (padding positions included).
Data    : MS-COCO train captions (train) / validation-split captions (validation, early stopping).

Usage: python scripts/distill_text_encoder.py --steps 4000 --batch 128
Output: artifacts/text_encoder_t6_sd15/ (+ training curve json/png)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pocketdiff.config import DATA_DIR, DISTILLED_TE_DIR, MODEL_IDS  # noqa: E402

KEEP = [0, 2, 4, 7, 9, 11]


def build_student(teacher, keep):
    from transformers import CLIPTextConfig, CLIPTextModel
    cfg = CLIPTextConfig.from_dict(teacher.config.to_dict())
    cfg.num_hidden_layers = len(keep)
    student = CLIPTextModel(cfg)
    sd_t = teacher.state_dict()
    sd_s = student.state_dict()
    for k in sd_s:
        if ".layers." in k:
            pre, rest = k.split(".layers.", 1)
            i, tail = rest.split(".", 1)
            src = f"{pre}.layers.{keep[int(i)]}.{tail}"
        else:
            src = k
        sd_s[k] = sd_t[src].float().clone()
    student.load_state_dict(sd_s)
    return student


def batches(lines, bs, rng):
    while True:
        rng.shuffle(lines)
        for i in range(0, len(lines) - bs + 1, bs):
            yield lines[i:i + bs]


@torch.no_grad()
def evaluate(student, teacher, tok, lines, device, bs=256):
    student.eval()
    mse_sum, cos_sum, n = 0.0, 0.0, 0
    for i in range(0, len(lines), bs):
        ids = tok(lines[i:i + bs], padding="max_length", truncation=True, max_length=77,
                  return_tensors="pt").input_ids.to(device)
        t = teacher(ids)[0].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = student(ids)[0]
        s = s.float()
        mse_sum += F.mse_loss(s, t, reduction="sum").item() / (t.shape[1] * t.shape[2])
        cos_sum += F.cosine_similarity(s[:, 1:], t[:, 1:], dim=-1).mean(dim=1).sum().item()
        n += ids.shape[0]
    student.train()
    return mse_sum / n, cos_sum / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=2000)
    args = ap.parse_args()

    from transformers import CLIPTextModel, CLIPTokenizer
    if not torch.cuda.is_available():
        print("CUDA required for distillation")
        return 1
    device = "cuda"
    torch.manual_seed(0)
    rng = random.Random(0)
    train_lines = (DATA_DIR / "coco_train_captions.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (DATA_DIR / "coco_val_captions.txt").read_text(encoding="utf-8").splitlines()
    rng.shuffle(val_lines)
    val_lines = val_lines[: args.n_val]
    print(f"train captions {len(train_lines)} | val captions {len(val_lines)}")

    repo = MODEL_IDS["sd15"]
    tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
    teacher = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", variant="fp16",
                                            torch_dtype=torch.float16).to(device).eval()
    teacher.requires_grad_(False)
    student = build_student(teacher, KEEP).to(device).train()
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"teacher {n_t / 1e6:.1f}M params -> student {n_s / 1e6:.1f}M params")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    log = {"keep_layers": KEEP, "teacher_params": n_t, "student_params": n_s, "train": [], "val": []}
    mse0, cos0 = evaluate(student, teacher, tok, val_lines, device)
    log["val"].append({"step": 0, "mse": mse0, "cos": cos0})
    print(f"step 0: val mse {mse0:.4f} cos {cos0:.4f}")
    best = (mse0, 0)
    DISTILLED_TE_DIR.mkdir(parents=True, exist_ok=True)
    gen = batches(train_lines, args.batch, rng)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        texts = next(gen)
        ids = tok(texts, padding="max_length", truncation=True, max_length=77,
                  return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            t = teacher(ids)[0].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = student(ids)[0]
        s = s.float()
        loss_mse = F.mse_loss(s, t)
        loss_cos = 1 - F.cosine_similarity(s[:, 1:], t[:, 1:], dim=-1).mean()
        loss = loss_mse + 0.5 * loss_cos
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            log["train"].append({"step": step, "loss": loss.item(), "mse": loss_mse.item(),
                                 "cos_loss": loss_cos.item(), "lr": sched.get_last_lr()[0]})
        if step % args.eval_every == 0 or step == args.steps:
            mse, cos = evaluate(student, teacher, tok, val_lines, device)
            log["val"].append({"step": step, "mse": mse, "cos": cos})
            print(f"step {step}: train loss {loss.item():.4f} | val mse {mse:.4f} cos {cos:.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            if mse < best[0]:
                best = (mse, step)
                sd16 = {k: v.detach().to(torch.float16).cpu() for k, v in student.state_dict().items()}
                student.config.torch_dtype = "float16"
                student.save_pretrained(DISTILLED_TE_DIR, state_dict=sd16)
    log["best"] = {"val_mse": best[0], "step": best[1]}
    log["train_seconds"] = time.time() - t0
    (DISTILLED_TE_DIR / "distill_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([r["step"] for r in log["train"]], [r["mse"] for r in log["train"]], label="train MSE")
        ax[0].plot([r["step"] for r in log["val"]], [r["mse"] for r in log["val"]], "o-", label="val MSE")
        ax[0].set_xlabel("step"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Distillation loss")
        ax[1].plot([r["step"] for r in log["val"]], [r["cos"] for r in log["val"]], "o-")
        ax[1].set_xlabel("step"); ax[1].set_title("Val token cosine (student vs teacher)")
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "results" / "plots"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "text_encoder_distillation.png", dpi=120)
    except Exception as e:
        print("plot failed:", e)
    print(f"best val mse {best[0]:.4f} at step {best[1]} -> {DISTILLED_TE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
