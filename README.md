# ACE-Step Server

A self-hosted, GPU-accelerated music generation REST API powered by [ACE-Step v1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) — a 4 GB diffusion model for text-to-music generation with optional Chain-of-Thought reasoning via a 5Hz semantic LM.

Packaged as a Docker image with a lightweight FastAPI server. Synchronous endpoint that returns base64-encoded audio in a single response — no polling, no task queue. Model weights are baked into the image at build time — no volume or HuggingFace token needed at runtime.

Pre-built image is available on Docker Hub:

```bash
docker pull naturelbenton/acestep-server:latest
```

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Running the Container](#running-the-container)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Performance & VRAM Tuning](#performance--vram-tuning)
- [Why This Image](#why-this-image)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- GPU-accelerated text-to-music generation (CUDA, ACE-Step v1.5 turbo / xl-turbo)
- Optional Chain-of-Thought metadata + audio code generation via 5Hz LM (1.7B / 0.6B)
- Synchronous API — `POST /generate` returns base64 audio in the response, no polling
- INT8 weight-only quantization (torchao) for reduced VRAM
- Optional CPU offload for very tight VRAM budgets
- Full song generation with or without lyrics, multilingual vocals
- Weights baked into the image — no HuggingFace token or volume needed at runtime

## Requirements

| Requirement | Details |
|---|---|
| Docker | 20.10+ with BuildKit |
| NVIDIA GPU | Compute Capability 8.0+ (Ampere or newer recommended for INT8 tensor cores) |
| nvidia-container-toolkit | [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| VRAM | ~6 GB (turbo + LM 1.7B PT) / ~10 GB (xl-turbo + LM 1.7B vLLM) |
| RAM | ~14 GB host RAM during inference |
| Disk | ~22 GB (image with all checkpoints baked) |

## Quick Start

```bash
# Pull the image
docker pull naturelbenton/acestep-server:latest

# Run
docker run --gpus all -p 8000:8000 naturelbenton/acestep-server:latest

# Health check (wait ~60s for models to load on first boot)
curl http://localhost:8000/health

# Generate (sync, returns base64 in response)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "uplifting indie pop with female vocals",
    "lyrics": "[Instrumental]",
    "instrumental": true,
    "duration": 30,
    "audio_format": "wav",
    "thinking": true
  }' | jq -r '.audios[0]' | base64 -d > output.wav
```

## Building the Image

```bash
git clone git@github.com:laurentf/acestep-server.git
cd acestep-server

# Default build (xl-turbo + LM 1.7B baked, ~22 GB image)
docker build -t naturelbenton/acestep-server:latest .

# Smaller variant (turbo only, ~10 GB image)
docker build --build-arg ACESTEP_DIT_MODEL=acestep-v15-turbo \
             -t naturelbenton/acestep-server:turbo .

# Smaller LM variant
docker build --build-arg ACESTEP_LM_MODEL=acestep-5Hz-lm-0.6B \
             -t naturelbenton/acestep-server:lm-0.6b .
```

The build downloads model weights from HuggingFace via `ace-step.model_downloader` and bakes them into `/app/checkpoints`. This is a one-time step — no download happens at runtime.

> **Note:** the image uses `uv` to install `ace-step` directly from the official git repo (not PyPI — the published package is outdated as of 2025-05). This requires Python 3.12 and a CUDA 12.8 base image.

To push:

```bash
docker push naturelbenton/acestep-server:latest
```

## Running the Container

### Basic

```bash
docker run --gpus all -p 8000:8000 naturelbenton/acestep-server:latest
```

### With docker-compose (standalone)

```yaml
services:
  acestep:
    image: naturelbenton/acestep-server:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - ACESTEP_DIT_MODEL=acestep-v15-turbo
      - ACESTEP_LM_BACKEND=pt
      - ACESTEP_QUANTIZATION=int8_weight_only
      - ACESTEP_OFFLOAD_TO_CPU=false
```

### Integrated into an existing project

```yaml
services:
  my-app:
    image: my-app:latest
    environment:
      - ACESTEP_URL=http://acestep:8000

  acestep:
    image: naturelbenton/acestep-server:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

From `my-app`, call the API at `http://acestep:8000/generate`.

## API Reference

The server exposes two endpoints on port **8000**.

---

### GET /health

Liveness check + active model info.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "dit_model": "acestep-v15-turbo",
  "dit_loaded": true,
  "lm_model": "acestep-5Hz-lm-1.7B",
  "quantization": "int8_weight_only",
  "offload_to_cpu": false,
  "device": "auto"
}
```

`lm_model` is `null` if `ACESTEP_INIT_LM=false` or LM init failed.

---

### POST /generate

Synchronous text-to-music generation. Blocks until generation completes, returns base64-encoded audio.

**Content-Type:** `application/json`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `caption` | string | *required* | Music description (style, mood, instruments). Max 512 chars. |
| `lyrics` | string | `"[Instrumental]"` | Lyrics text. Use `"[Instrumental]"` for instrumental. Max 4096 chars. Supports structure tags like `[verse]`, `[chorus]`. |
| `instrumental` | bool | `true` | Force instrumental even if `lyrics` are provided. |
| `duration` | float | `30.0` | Audio duration in seconds (10.0–240.0). |
| `audio_format` | string | `"wav"` | Output format: `wav`, `flac`, `mp3`, `opus`, `aac`. |
| `seed` | int | `-1` | Reproducibility seed. `-1` for random. |
| `inference_steps` | int | `8` | Diffusion steps. `8` for turbo (default), 25–50 for non-turbo models. |
| `guidance_scale` | float | `7.0` | CFG strength. Ignored for turbo (forced to 1.0). |
| `thinking` | bool | `true` | Use 5Hz LM for Chain-of-Thought metadata + audio codes. Adds ~10–20s but improves long-form coherence and lyric alignment. Requires `ACESTEP_INIT_LM=true`. |

**Response:**

```json
{
  "audios": ["UklGRiQAAABXQVZFZm10IBAAAAA..."],
  "metadata": {
    "caption": "...",
    "duration": 30.0,
    "inference_steps": 8,
    "audio_format": "wav",
    "thinking": true,
    "lm_used": true
  }
}
```

`audios` is an array of base64-encoded audio strings (`batch_size=1` always returns one element).

**Example — instrumental track:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "tense ambient cinematic score, dark atmosphere",
    "lyrics": "[Instrumental]",
    "instrumental": true,
    "duration": 60,
    "audio_format": "flac",
    "seed": 42
  }' | jq -r '.audios[0]' | base64 -d > track.flac
```

**Example — song with lyrics (Python):**

```python
import base64, requests

response = requests.post("http://localhost:8000/generate", json={
    "caption": "energetic pop rock, female vocals, 90s alternative vibe",
    "lyrics": "[verse]\nWalking down the street tonight\n[chorus]\nFeels alright, feels alright",
    "instrumental": False,
    "duration": 120,
    "audio_format": "wav",
    "thinking": True,
})
data = response.json()
with open("song.wav", "wb") as f:
    f.write(base64.b64decode(data["audios"][0]))
```

## Configuration

All configuration is via environment variables. Most can be changed at runtime without rebuilding.

| Variable | Default | Description |
|---|---|---|
| `ACESTEP_DIT_MODEL` | `acestep-v15-xl-turbo` | DiT checkpoint name. Available: `acestep-v15-turbo` (2B, ~3 GB VRAM), `acestep-v15-xl-turbo` (4B, ~5 GB VRAM with int8). Build-time arg. |
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-1.7B` | LM checkpoint name. Available: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`. Build-time arg. |
| `ACESTEP_INIT_LM` | `true` | Load the LM at startup. Set to `false` to free ~3-4 GB VRAM (no thinking mode). Runtime. |
| `ACESTEP_LM_BACKEND` | `vllm` | LM inference backend: `vllm` (faster, more VRAM) or `pt` (slower, less VRAM, more reliable). Runtime. |
| `ACESTEP_QUANTIZATION` | `int8_weight_only` | DiT quantization: `int8_weight_only`, `fp8_weight_only` (Ada/Hopper only), `w8a8_dynamic`, or empty string for none. Runtime. |
| `ACESTEP_OFFLOAD_TO_CPU` | `true` | CPU offload for low-VRAM. Significantly slower (~90s per model load with int8 due to torchao param-by-param transfer). Runtime. |
| `ACESTEP_DEVICE` | `auto` | `auto`, `cuda`, `cpu`. Runtime. |
| `ACESTEP_CHECKPOINTS_DIR` | `/app/checkpoints` | Where the baked weights live. Runtime. |

## Performance & VRAM Tuning

The default image bakes both `xl-turbo` (4B) and the unified ACE-Step bundle (which includes turbo + VAE + Qwen3 embedding + LM 1.7B). You can pick the active DiT and LM at runtime via env vars.

### Recommended VRAM-tier configs

| Available VRAM | Config | Quality | Speed |
|---|---|---|---|
| **6 GB** | turbo + `INIT_LM=false` | OK for instrumentals | RTF ~1.5× |
| **10 GB** | turbo + LM 1.7B + PT backend + no offload | Excellent for songs | RTF ~1.0× |
| **12 GB** | turbo + LM 1.7B + vLLM + no offload | Excellent + faster LM | RTF ~0.6× |
| **16 GB+ (alone)** | xl-turbo + LM 1.7B + vLLM + no offload | Best quality | RTF ~0.5× |

RTF = Real-Time Factor (wall-clock time / audio duration). Lower is better; 1.0× = generation matches audio length.

### Known constraints

- **`ACESTEP_LM_BACKEND=vllm` over-allocates VRAM** in some configurations because ace-step's `gpu_memory_utilization` calculation sums DiT + LM target as a fraction of total GPU memory, but vLLM interprets it as its own allocation budget. Symptom: VRAM pre-flight check fails with `Insufficient free VRAM`. Workaround: switch to `ACESTEP_LM_BACKEND=pt`.
- **`ACESTEP_OFFLOAD_TO_CPU=true` is very slow with INT8 quantization** because torch 2.10 + torchao 0.16 cannot perform `model.to('cuda')` in bulk on `AffineQuantizedTensor` — it falls back to per-parameter transfer (~90 seconds per model load). Use `false` whenever VRAM allows.
- **First request after startup is slower** by 15–30s due to torch.compile / triton / vLLM warmup. The cache is in `/root/.cache/torch/inductor` inside the container and is lost on `docker compose down`. Mount a volume to persist.

## Why This Image

The official ACE-Step repo provides a Docker image (`valyriantech/ace-step-1.5`) but it ships both a Gradio UI and a FastAPI server in the same container, each loading the models independently. On a GPU under 24 GB this guarantees an OOM at startup (LM is loaded twice).

This image:

- Runs a single FastAPI process — no Gradio, no double model loading
- Exposes a sync `POST /generate` endpoint that returns base64 audio (vs. the official `release_task` + polling + download flow)
- Uses `uv` to install `ace-step` from the official git repo (the PyPI package `ace-step` is stuck at v0.1.0 from May 2025 and lacks the v1.5 architecture)
- Bakes model checkpoints at build time for fast cold starts

The architecture is inspired by [kortexa-ai/music-gen.server](https://github.com/kortexa-ai/music-gen.server) (used by [writ-fm](https://github.com/keltokhy/writ-fm) in production), but Dockerized and Linux-CUDA-only (the upstream targets macOS MLX as well).

## Project Structure

```
acestep-server/
├── server.py           # FastAPI server — /generate, /health
├── Dockerfile          # Builds the image, bakes model checkpoints
├── download_models.py  # Downloads checkpoints from HuggingFace at build time
├── pyproject.toml      # Python deps managed by uv (ace-step from git)
└── .dockerignore
```

## License

The ACE-Step model is released by ACE Studio and StepFun under the [MIT License](https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE). The 5Hz LM checkpoints are MIT-licensed. This server wrapper is MIT-licensed.
