FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-dev

# MODEL_VARIANT bake (rebuild to switch models, env vars at runtime won't redownload)
ARG ACESTEP_DIT_MODEL=acestep-v15-xl-turbo
ARG ACESTEP_LM_MODEL=acestep-5Hz-lm-1.7B
ENV ACESTEP_DIT_MODEL=${ACESTEP_DIT_MODEL}
ENV ACESTEP_LM_MODEL=${ACESTEP_LM_MODEL}
ENV ACESTEP_CHECKPOINTS_DIR=/app/checkpoints

COPY download_models.py .
RUN uv run python download_models.py

# build-essential ajouté après le bake modèles : triton/torch._inductor compile
# des kernels CUDA à la volée et exige gcc. Sans ça vLLM crash → fallback PyTorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server.py .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
