"""FastAPI server exposing ACE-Step v1.5 music generation.

Endpoints:
    GET  /health     — liveness + active model info
    POST /generate   — text → music, returns base64 audio (sync)

Runtime configuration (env):
    ACESTEP_DIT_MODEL          DiT checkpoint name        (default: acestep-v15-xl-turbo)
    ACESTEP_LM_MODEL           LM checkpoint name         (default: acestep-5Hz-lm-1.7B)
    ACESTEP_CHECKPOINTS_DIR    checkpoints directory      (default: /app/checkpoints)
    ACESTEP_DEVICE             auto / cuda / cpu          (default: auto)
    ACESTEP_QUANTIZATION       int8_weight_only / fp8_weight_only / w8a8_dynamic / empty
                               (default: int8_weight_only)
    ACESTEP_OFFLOAD_TO_CPU     true/false                 (default: true)
    ACESTEP_LM_BACKEND         vllm or pt                 (default: vllm)
    ACESTEP_INIT_LM            true/false — load LM at startup (default: true)
                               set to false to free ~3-4 GB VRAM (no thinking mode)
"""

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


PROJECT_ROOT = "/app"
CHECKPOINTS_DIR = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "/app/checkpoints")
DIT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-xl-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
DEVICE = os.environ.get("ACESTEP_DEVICE", "auto")
QUANTIZATION = os.environ.get("ACESTEP_QUANTIZATION", "int8_weight_only") or None
OFFLOAD_TO_CPU = os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "true").lower() in ("1", "true", "yes")
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "vllm")
INIT_LM = os.environ.get("ACESTEP_INIT_LM", "true").lower() in ("1", "true", "yes")


class Pipeline:
    def __init__(self) -> None:
        self.dit: AceStepHandler | None = None
        self.llm: LLMHandler | None = None

    def load(self) -> None:
        self.dit = AceStepHandler()
        msg, ok = self.dit.initialize_service(
            project_root=PROJECT_ROOT,
            config_path=DIT_MODEL,
            device=DEVICE,
            quantization=QUANTIZATION,
            offload_to_cpu=OFFLOAD_TO_CPU,
            offload_dit_to_cpu=OFFLOAD_TO_CPU,
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {msg}")

        if not INIT_LM:
            return

        self.llm = LLMHandler()
        msg, ok = self.llm.initialize(
            checkpoint_dir=CHECKPOINTS_DIR,
            lm_model_path=LM_MODEL,
            backend=LM_BACKEND,
            device=DEVICE,
        )
        if not ok:
            self.llm = None


pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    pipeline.load()
    yield


app = FastAPI(title="ACE-Step server", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    caption: str = Field(..., max_length=512)
    lyrics: str = Field(default="[Instrumental]", max_length=4096)
    instrumental: bool = True
    duration: float = Field(default=30.0, ge=10.0, le=240.0)
    audio_format: str = Field(default="wav", pattern="^(wav|flac|mp3|opus|aac)$")
    seed: int = -1
    inference_steps: int = Field(default=8, ge=1, le=200)
    guidance_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    thinking: bool = True


class GenerateResponse(BaseModel):
    audios: list[str]
    metadata: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dit_model": DIT_MODEL,
        "dit_loaded": pipeline.dit is not None,
        "lm_model": LM_MODEL if pipeline.llm else None,
        "quantization": QUANTIZATION,
        "offload_to_cpu": OFFLOAD_TO_CPU,
        "device": DEVICE,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if pipeline.dit is None:
        raise HTTPException(503, detail="DiT model not loaded")

    params = GenerationParams(
        task_type="text2music",
        caption=req.caption,
        lyrics=req.lyrics,
        instrumental=req.instrumental,
        duration=req.duration,
        inference_steps=req.inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        thinking=req.thinking and pipeline.llm is not None,
        shift=3.0,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format=req.audio_format,
        use_random_seed=(req.seed == -1),
        seeds=None if req.seed == -1 else [req.seed],
    )

    with tempfile.TemporaryDirectory(prefix="acestep_") as save_dir:
        result = await asyncio.to_thread(
            generate_music,
            dit_handler=pipeline.dit,
            llm_handler=pipeline.llm,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        if not result.success:
            raise HTTPException(500, detail=result.error or "Generation failed")

        encoded: list[str] = []
        for fname in sorted(os.listdir(save_dir)):
            fpath = os.path.join(save_dir, fname)
            if os.path.isfile(fpath) and fname.endswith((".wav", ".flac", ".mp3", ".opus", ".aac")):
                with open(fpath, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode())

    return GenerateResponse(
        audios=encoded,
        metadata={
            "caption": req.caption,
            "duration": req.duration,
            "inference_steps": req.inference_steps,
            "audio_format": req.audio_format,
            "thinking": params.thinking,
            "lm_used": pipeline.llm is not None and req.thinking,
        },
    )
