"""
SWE-Pruner on Modal -- L4 GPU, scale-to-zero.

Deploy:  modal deploy deploy/modal_app.py
Serve (dev): modal serve deploy/modal_app.py
"""

import modal
from pathlib import Path

MODEL_REPO = "ayanami-kitasan/code-pruner"
MODEL_DIR = "/model"
LOCAL_PKG = Path(__file__).parent.parent / "swe-pruner" / "src" / "swe_pruner"

# PyTorch devel image: has CUDA toolkit + gcc, flash-attn compiles fast
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    )
    .pip_install("packaging", "setuptools>=69", "wheel", "ninja")
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "transformers==4.57.6",
        "huggingface-hub>=0.36.0",
        "fastapi>=0.123.0",
        "uvicorn>=0.38.0",
        "socksio>=1.0.0",
        "typer>=0.19.2",
    )
    .run_commands(
        f"python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO}', local_dir='{MODEL_DIR}')"
        f"\""
    )
    .add_local_dir(LOCAL_PKG, "/root/swe_pruner")
)

app = modal.App("swe-pruner", image=image)


@app.function(
    gpu="L4",
    scaledown_window=300,
    max_containers=3,
)
@modal.asgi_app()
def serve():
    import asyncio
    import os
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, "/root")

    from fastapi import FastAPI, HTTPException
    from swe_pruner.prune_wrapper import (
        PruneRequest,
        PruneResponse,
        SwePrunerForCodePruning,
    )

    os.environ["SWEPRUNER_MODEL_PATH"] = MODEL_DIR
    model = SwePrunerForCodePruning.from_pretrained(MODEL_DIR)
    executor = ThreadPoolExecutor(max_workers=1)

    web_app = FastAPI(title="SWE-Pruner")

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model_loaded": model is not None}

    @web_app.post("/prune", response_model=PruneResponse)
    async def prune(request: PruneRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        response = await loop.run_in_executor(executor, model.prune, request)
        latency_ms = int((time.monotonic() - t0) * 1000)
        response_dict = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else response.dict()
        )
        response_dict["latency_ms"] = latency_ms
        return response

    return web_app
