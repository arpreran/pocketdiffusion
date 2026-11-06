# GPU image (needs the NVIDIA Container Toolkit):  docker build -t pocketdiffusion .
#                                                docker run --gpus all -p 8000:8000 -v pdata:/app/data pocketdiffusion
# (the volume keeps the SQLite database and generated images across container restarts)
# Without a GPU the app still runs (CPU fallback plans), just slower.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HOME=/models GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY pocketdiff ./pocketdiff
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY prompts ./prompts
COPY docs ./docs
COPY app.py serve.py ./
COPY artifacts ./artifacts
COPY results/*.json results/summary.md ./results/
COPY results/plots ./results/plots

# bake the weights into the image so the container starts without network access
RUN python scripts/download_assets.py --core

EXPOSE 8000
CMD ["python", "serve.py", "--port", "8000"]
