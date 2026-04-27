"""Pre-download ACE-Step checkpoints at Docker build time.

Reads ACESTEP_DIT_MODEL, ACESTEP_LM_MODEL, ACESTEP_CHECKPOINTS_DIR env vars
set by the Dockerfile. Weights are baked into the image.
"""

import os
import sys
from pathlib import Path

from acestep.model_downloader import ensure_dit_model, ensure_lm_model, ensure_main_model


def main() -> None:
    checkpoints_dir = Path(os.environ["ACESTEP_CHECKPOINTS_DIR"])
    dit_model = os.environ["ACESTEP_DIT_MODEL"]
    lm_model = os.environ["ACESTEP_LM_MODEL"]

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> Downloading main model bundle (VAE, embeddings, default LM) to {checkpoints_dir}")
    ok, msg = ensure_main_model(checkpoints_dir=checkpoints_dir, prefer_source="huggingface")
    print(msg)
    if not ok:
        sys.exit(f"Failed to download main model: {msg}")

    print(f"==> Downloading DiT model: {dit_model}")
    ok, msg = ensure_dit_model(dit_model, checkpoints_dir=checkpoints_dir, prefer_source="huggingface")
    print(msg)
    if not ok:
        sys.exit(f"Failed to download DiT model: {msg}")

    print(f"==> Downloading LM model: {lm_model}")
    ok, msg = ensure_lm_model(lm_model, checkpoints_dir=checkpoints_dir, prefer_source="huggingface")
    print(msg)
    if not ok:
        sys.exit(f"Failed to download LM model: {msg}")

    print("==> All checkpoints baked.")


if __name__ == "__main__":
    main()
