# Hugging Face Space (Docker SDK, free "CPU basic" hardware: 2 vCPU, 16 GB RAM, no GPU).
# Always-on backup of the live demo: same app, CPU-only, SD-Turbo + TAESD (~10-30 s per image).
# Deployed with: python scripts/deploy_space.py   (this file becomes the Space's Dockerfile)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    POCKETDIFF_CPU_SCALE=4
WORKDIR /home/user/app

# CPU-only PyTorch first (torchvision too, so nothing pulls the large CUDA build from PyPI)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# bake the weights into the image: SD-Turbo, TAESD, CLIP-B/32 (re-ranking + safety), OWLv2 (semantic check)
RUN python scripts/download_assets.py --cloud

EXPOSE 7860
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "7860"]
