"""Model-compression techniques implemented in this project.

1. Reduced precision     - FP32 -> FP16 casting (handled in the engine).
2. Weight-only quantisation (our implementation, no external quantisation library):
     INT8: symmetric, per-output-channel scales            (2x smaller than FP16)
     INT4: symmetric, group-wise (group=64) scales, 2 weights packed per byte (~3.6x smaller)
   Weights are stored quantised and de-quantised on the fly inside forward(), so the
   resident model size and the steady-state memory both shrink; only one layer's FP16
   weight exists transiently at any time.
3. Unstructured magnitude pruning - used to *measure* quality degradation vs sparsity.
4. Knowledge distillation - LCM-LoRA (consistency distillation, fused into the UNet) and our
   own distilled 6-layer text encoder (see scripts/distill_text_encoder.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SKIP_DEFAULT = ("conv_in", "conv_out", "time_embedding", "time_proj")


# ----------------------------------------------------------------------------- quantisation
def _quantize_int8(w: torch.Tensor):
    w2 = w.reshape(w.shape[0], -1).float()
    scale = w2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w2 / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def _quantize_int4(w: torch.Tensor, group: int):
    out = w.shape[0]
    w2 = w.reshape(out, -1).float()
    n = w2.shape[1]
    pad = (-n) % group
    if pad:
        w2 = F.pad(w2, (0, pad))
    g = w2.reshape(out, -1, group)
    scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(g / scale).clamp(-8, 7).to(torch.int16) + 8           # 0..15
    q = q.reshape(out, -1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)            # 2 nibbles / byte
    return packed, scale, n


class QuantizedWeight(nn.Module):
    """Holds a quantised weight and reconstructs it in the activation dtype on demand."""

    def __init__(self, weight: torch.Tensor, bits: int, group: int = 64,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.bits = bits
        self.group = group
        self.shape = tuple(weight.shape)
        self.dtype = dtype
        if bits == 8:
            q, s = _quantize_int8(weight)
            self.register_buffer("qweight", q)
            self.register_buffer("scale", s.to(dtype))
            self.n = q.shape[1]
        elif bits == 4:
            q, s, n = _quantize_int4(weight, group)
            self.register_buffer("qweight", q)
            self.register_buffer("scale", s.to(dtype))
            self.n = n
        else:
            raise ValueError("bits must be 8 or 4")

    def dequantize(self) -> torch.Tensor:
        if self.bits == 8:
            w = self.qweight.to(self.scale.dtype) * self.scale
        else:
            lo = (self.qweight & 0x0F).to(torch.int16) - 8
            hi = (self.qweight >> 4).to(torch.int16) - 8
            q = torch.stack([lo, hi], dim=-1).reshape(self.qweight.shape[0], -1)
            g = q.reshape(q.shape[0], -1, self.group).to(self.scale.dtype) * self.scale
            w = g.reshape(q.shape[0], -1)[:, : self.n]
        return w.reshape(self.shape)


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int, group: int):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.qw = QuantizedWeight(lin.weight.data, bits, group, lin.weight.dtype)
        self.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False) if lin.bias is not None else None

    @property
    def weight(self):  # some library code peeks at .weight.dtype / .device
        return self.qw.dequantize()

    def forward(self, x):
        return F.linear(x, self.qw.dequantize().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class QConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, bits: int, group: int):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding_mode = conv.padding_mode
        self.qw = QuantizedWeight(conv.weight.data, bits, group, conv.weight.dtype)
        self.bias = nn.Parameter(conv.bias.data.clone(), requires_grad=False) if conv.bias is not None else None

    @property
    def weight(self):
        return self.qw.dequantize()

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.conv2d(x, self.qw.dequantize().to(x.dtype), b, self.stride, self.padding,
                        self.dilation, self.groups)


def quantize_module_(model: nn.Module, bits: int = 8, group: int = 64,
                     skip: tuple[str, ...] = SKIP_DEFAULT, min_numel: int = 4096) -> dict:
    """Replace Linear/Conv2d layers in-place with weight-only quantised versions."""
    replaced, kept = 0, 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(full.startswith(s) or f".{s}" in full for s in skip):
                kept += 1
                continue
            if isinstance(child, nn.Linear) and child.weight.numel() >= min_numel:
                setattr(parent, child_name, QLinear(child, bits, group))
                replaced += 1
            elif isinstance(child, nn.Conv2d) and child.weight.numel() >= min_numel:
                setattr(parent, child_name, QConv2d(child, bits, group))
                replaced += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"bits": bits, "replaced_layers": replaced, "kept_layers": kept}


@torch.no_grad()
def quantization_error(weight: torch.Tensor, bits: int, group: int = 64) -> float:
    """Relative Frobenius error ||W - Q(W)|| / ||W|| (reported in the degradation study)."""
    qw = QuantizedWeight(weight, bits, group, torch.float32)
    return float((qw.dequantize() - weight.float()).norm() / weight.float().norm())


# ----------------------------------------------------------------------------- pruning
@torch.no_grad()
def magnitude_prune_(model: nn.Module, sparsity: float, skip: tuple[str, ...] = SKIP_DEFAULT) -> dict:
    """Layer-wise unstructured L1 pruning: zero the `sparsity` fraction of smallest weights."""
    if sparsity <= 0:
        return {"sparsity": 0.0, "pruned_params": 0}
    total, pruned = 0, 0
    for name, mod in model.named_modules():
        if any(name.startswith(s) or f".{s}" in name for s in skip):
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            w = mod.weight.data
            k = int(sparsity * w.numel())
            if k < 1:
                continue
            thr = w.abs().float().flatten().kthvalue(k).values
            mask = w.abs().float() > thr
            w.mul_(mask.to(w.dtype))
            total += w.numel()
            pruned += int((~mask).sum())
    return {"sparsity": sparsity, "pruned_params": pruned, "prunable_params": total}


# ----------------------------------------------------------------------------- misc
def set_fast_attention(unet) -> str:
    """Use PyTorch scaled-dot-product attention (flash / mem-efficient kernels) when present."""
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet.set_attn_processor(AttnProcessor2_0())
        return "sdpa"
    except Exception:
        return "default"
