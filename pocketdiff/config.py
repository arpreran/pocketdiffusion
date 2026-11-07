"""Central configuration: paths, model registry, generation presets, device profiles."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("POCKETDIFF_DATA", ROOT / "data"))   # cloud: a writable location
# POCKETDIFF_RESULTS lets a dry run write somewhere else without touching the real results
RESULTS_DIR = Path(os.environ.get("POCKETDIFF_RESULTS", ROOT / "results"))
ARTIFACTS_DIR = ROOT / "artifacts"          # things *we* trained / fitted
CALIBRATION_FILE = Path(os.environ.get("POCKETDIFF_CALIBRATION", ARTIFACTS_DIR / "planner_calibration.json"))
NIQE_PARAMS_FILE = ARTIFACTS_DIR / "niqe_coco_pristine.npz"
DISTILLED_TE_DIR = ARTIFACTS_DIR / "text_encoder_t6_sd15"

# --------------------------------------------------------------------------------------
# Pretrained checkpoints (Hugging Face Hub ids). All are public, none are gated.
# --------------------------------------------------------------------------------------
MODEL_IDS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",   # baseline + LCM student base
    "sd_turbo": "stabilityai/sd-turbo",                       # adversarially distilled, 1-step
    "lcm_lora": "latent-consistency/lcm-lora-sdv1-5",         # consistency-distillation adapter
    "taesd": "madebyollin/taesd",                             # tiny latent decoder (2.4M params)
    "clip_rerank": "openai/clip-vit-base-patch32",            # on-device scorer for best-of-N
    "clip_eval": "openai/clip-vit-large-patch14",             # evaluation only (standard CLIPScore)
    "detector": "google/owlv2-base-patch16-ensemble",         # open-vocab detector (verification/F1)
    # a *different* detector for in-loop candidate selection, so selection never uses the model
    # that scores the F1 benchmark (no metric leakage)
    "selector": "google/owlvit-base-patch32",
}

FAMILIES = {
    # family -> (hub id, scheduler used by default, latent scaling factor)
    "sd15": MODEL_IDS["sd15"],
    "sd_turbo": MODEL_IDS["sd_turbo"],
}


@dataclass(frozen=True)
class GenConfig:
    """Everything that defines *how* an image is generated (one row of the comparison table)."""

    name: str
    family: str = "sd15"            # "sd15" | "sd_turbo"
    dtype: str = "fp16"             # "fp32" | "fp16"
    scheduler: str = "pndm"         # "pndm" | "dpm" | "lcm" | "euler"
    steps: int = 50
    guidance: float = 7.5           # <=1 disables classifier-free guidance (halves UNet batch)
    lcm_lora: bool = False          # fuse the LCM-LoRA consistency adapter into the UNet
    vae: str = "full"               # "full" (KL-f8 VAE, 83.7M) | "taesd" (tiny decoder, 2.4M)
    quant: str = "none"             # "none" | "int8" | "int4" (weight-only, our implementation)
    text_encoder: str = "full"      # "full" | "t6" (our 6-layer distilled text encoder, sd15 only)
    prune: float = 0.0              # unstructured magnitude-pruning sparsity (degradation study)
    height: int = 512
    width: int = 512
    device: str = "cuda"            # "cuda" | "cpu"
    description: str = ""

    @property
    def cfg(self) -> bool:
        return self.guidance > 1.0

    @property
    def unet_evals_per_image(self) -> int:
        # PNDM (skip_prk_steps) performs one extra UNet call; CFG doubles the batch.
        evals = self.steps + (1 if self.scheduler == "pndm" else 0)
        return evals * (2 if self.cfg else 1)

    def with_(self, **kw) -> "GenConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cfg"] = self.cfg
        d["unet_evals_per_image"] = self.unet_evals_per_image
        return d

    def model_key(self) -> tuple:
        """Configs that share a key share the same loaded weights."""
        return (self.family, self.dtype, self.lcm_lora, self.quant, self.text_encoder,
                self.prune, self.device)


PRESETS: dict[str, GenConfig] = {
    "baseline": GenConfig(
        name="baseline", family="sd15", dtype="fp32", scheduler="pndm", steps=50, guidance=7.5,
        description="Stable Diffusion v1.5, FP32, PNDM 50 steps, CFG 7.5, full VAE (reference)"),
    "sd15_fp16": GenConfig(
        name="sd15_fp16", dtype="fp16", scheduler="pndm", steps=50, guidance=7.5,
        description="+ reduced precision (FP16)"),
    "sd15_fp16_dpm20": GenConfig(
        name="sd15_fp16_dpm20", dtype="fp16", scheduler="dpm", steps=20, guidance=7.5,
        description="+ fast ODE solver (DPM-Solver++ 20 steps)"),
    "sd15_lcm4": GenConfig(
        name="sd15_lcm4", dtype="fp16", scheduler="lcm", steps=4, guidance=1.0, lcm_lora=True,
        description="+ consistency distillation (LCM-LoRA, 4 steps, no CFG)"),
    "sd15_lcm4_taesd": GenConfig(
        name="sd15_lcm4_taesd", dtype="fp16", scheduler="lcm", steps=4, guidance=1.0,
        lcm_lora=True, vae="taesd",
        description="+ tiny latent decoder (TAESD)"),
    "sd15_lcm4_taesd_int8": GenConfig(
        name="sd15_lcm4_taesd_int8", dtype="fp16", scheduler="lcm", steps=4, guidance=1.0,
        lcm_lora=True, vae="taesd", quant="int8",
        description="+ INT8 weight-only quantization of UNet"),
    "sd15_lcm4_taesd_int8_t6": GenConfig(
        name="sd15_lcm4_taesd_int8_t6", dtype="fp16", scheduler="lcm", steps=4, guidance=1.0,
        lcm_lora=True, vae="taesd", quant="int8", text_encoder="t6",
        description="+ our distilled 6-layer text encoder"),
    "turbo1_taesd": GenConfig(
        name="turbo1_taesd", family="sd_turbo", dtype="fp16", scheduler="euler", steps=1,
        guidance=0.0, vae="taesd",
        description="Lightweight architecture selection: SD-Turbo (ADD-distilled), 1 step, TAESD"),
    "turbo1_taesd_int8": GenConfig(
        name="turbo1_taesd_int8", family="sd_turbo", dtype="fp16", scheduler="euler", steps=1,
        guidance=0.0, vae="taesd", quant="int8",
        description="SD-Turbo 1 step + TAESD + INT8 weights"),
    "cpu_fallback": GenConfig(
        name="cpu_fallback", family="sd15", dtype="fp32", scheduler="lcm", steps=2, guidance=1.0,
        lcm_lora=True, vae="taesd", height=256, width=256, device="cpu",
        description="Resource-aware fallback: CPU, 256 px, LCM 2 steps, TAESD"),
}

# Order used in the comparison tables of the report.
COMPARISON_ORDER = [
    "baseline", "sd15_fp16", "sd15_fp16_dpm20", "sd15_lcm4", "sd15_lcm4_taesd",
    "sd15_lcm4_taesd_int8", "sd15_lcm4_taesd_int8_t6", "turbo1_taesd", "turbo1_taesd_int8",
]


# --------------------------------------------------------------------------------------
# Device profiles for the resource-aware planner.
#   "measured"  -> latencies come from calibration on this machine
#   "simulated" -> latency = FLOPs / assumed effective throughput (clearly an *estimate*)
# Effective-throughput numbers are deliberately conservative assumptions, not vendor specs.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceProfile:
    key: str
    label: str
    kind: str                     # "measured" | "simulated"
    memory_budget_mb: float       # memory the generator may use
    eff_tflops: float = 0.0       # assumed effective dense TFLOPS (simulated profiles)
    run_device: str = "cuda"      # where the demo actually executes the plan
    allow_fp32: bool = True
    notes: str = ""


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "this_gpu": DeviceProfile(
        "this_gpu", "This machine (GPU, measured)", "measured", memory_budget_mb=6500,
        notes="Latency and memory measured live on the local CUDA GPU."),
    "flagship_phone": DeviceProfile(
        "flagship_phone", "Flagship phone NPU (simulated)", "simulated", memory_budget_mb=3000,
        eff_tflops=4.0, allow_fp32=False,
        notes="~12 GB RAM phone, ~3 GB usable by one app; assumed 4 effective TFLOPS (FP16/INT8)."),
    "midrange_phone": DeviceProfile(
        "midrange_phone", "Mid-range phone (simulated)", "simulated", memory_budget_mb=1800,
        eff_tflops=1.2, allow_fp32=False,
        notes="~8 GB RAM phone, ~1.8 GB usable; assumed 1.2 effective TFLOPS."),
    "lowend_phone": DeviceProfile(
        "lowend_phone", "Low-end phone (simulated)", "simulated", memory_budget_mb=1100,
        eff_tflops=0.3, allow_fp32=False,
        notes="~4-6 GB RAM phone, ~1.1 GB usable; assumed 0.3 effective TFLOPS (CPU/GPU)."),
    "this_cpu": DeviceProfile(
        "this_cpu", "This machine (CPU only, measured)", "measured", memory_budget_mb=6000,
        run_device="cpu", notes="Real constrained-device proxy: generation runs on CPU."),
}


def load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


@dataclass
class TeamInfo:
    team_id: str = "TXX"
    usns: list[str] = field(default_factory=lambda: ["USN1", "USN2", "USN3", "USN4"])

    @staticmethod
    def load() -> "TeamInfo":
        p = ROOT / "team.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return TeamInfo(team_id=d.get("team_id", "TXX"), usns=d.get("usns", []))
        return TeamInfo()
