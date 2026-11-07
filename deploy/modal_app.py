"""PocketDiffusion on Modal (serverless GPU) - always-on public URL, laptop not needed.

    python -m modal deploy deploy/modal_app.py      # run from the repository root

Same FastAPI backend + web UI as serve.py, on an NVIDIA T4. The container image has the CUDA
build of PyTorch and every model baked in (no download at start-up). Modal scales to zero when
nobody uses the app (costs nothing) and starts a container on the next visit (~1-2 min cold
start, then ~1-2 s per image). The Starter plan's $30/month free credits cover ~50 T4-hours.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"HF_HOME": "/models", "PYTHONUNBUFFERED": "1", "POCKETDIFF_DATA": "/tmp/pocketdiff-data"})
    # bake the weights into the image: SD1.5 (+LCM-LoRA), SD-Turbo, TAESD, CLIP-B/32, OWL-ViT, OWLv2
    .add_local_file(str(ROOT / "scripts" / "download_assets.py"), f"{REMOTE}/scripts/download_assets.py", copy=True)
    .run_commands(f"cd {REMOTE} && python scripts/download_assets.py --core",
                  f"cd {REMOTE} && python scripts/download_assets.py --cloud")
    # the application code (small; added last so code changes redeploy in seconds)
    # ("**/" so files in sub-folders match too - e.g. the 161 MB distilled text encoder is not uploaded)
    .add_local_dir(str(ROOT), REMOTE, ignore=[
        ".git", ".claude", "data", "outputs", "deploy", "**/__pycache__", "**/*.pyc", "**/*.safetensors",
        "**/*.bin", "**/*.ckpt", "results/gen", "results/smoke", "results/report", "results/failures",
        ".pytest_cache", "**/*.log",
    ])
)

app = modal.App("pocketdiffusion", image=image)


# CPU container: Modal requires a card on file for GPU functions, CPU runs on the free credits alone.
# 8 physical cores + 16 GB RAM: SD-Turbo 1 step in ~5-15 s per image; ~$0.5/h only while in use.
@app.function(cpu=8.0, memory=16384, timeout=3600, scaledown_window=600, max_containers=1,
              env={"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "POCKETDIFF_CPU_SCALE": "1.0"})
@modal.concurrent(max_inputs=64)     # SSE streams + API calls share one container (one GPU worker)
@modal.asgi_app(label="pocketdiffusion")
def web():
    import os
    import sys

    os.chdir(REMOTE)
    sys.path.insert(0, REMOTE)
    from backend.main import app as fastapi_app
    from backend.main import jobs

    jobs.submit("warmup", {})        # load + warm the models in the background
    return fastapi_app
