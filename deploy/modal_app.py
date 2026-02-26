"""
Winnow on Modal -- L4 GPU, scale-to-zero, API key auth + rate limiting.

Deploy:  modal deploy deploy/modal_app.py
Serve (dev): modal serve deploy/modal_app.py
"""

import json
import modal
import os
import time
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

app = modal.App("winnow", image=image)

# Rate limit tiers: rate = tokens added per second, burst = max bucket capacity
RATE_TIERS: dict[str, dict[str, float]] = {
    "trial": {"rate": 2.0, "burst": 5},
    "pro": {"rate": 5.0, "burst": 10},
    "team": {"rate": 10.0, "burst": 20},
}


class TokenBucket:
    """Per-key token bucket rate limiter. Resets when the container scales down."""

    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def load_api_keys() -> dict[str, dict]:
    """Load API keys from WINNOW_API_KEYS env var (set via Modal Secret)."""
    raw = os.environ.get("WINNOW_API_KEYS", "")
    if not raw:
        return {}
    return json.loads(raw)


@app.function(
    gpu="T4",
    scaledown_window=300,
    max_containers=3,
    secrets=[modal.Secret.from_name("winnow-keys")],
)
@modal.asgi_app()
def serve():
    import asyncio
    import sys
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, "/root")

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from swe_pruner.prune_wrapper import (
        PruneRequest,
        PruneResponse,
        SwePrunerForCodePruning,
    )

    os.environ["SWEPRUNER_MODEL_PATH"] = MODEL_DIR
    model = SwePrunerForCodePruning.from_pretrained(MODEL_DIR)
    executor = ThreadPoolExecutor(max_workers=1)

    api_keys = load_api_keys()
    buckets: dict[str, TokenBucket] = {}

    def get_bucket(key: str) -> TokenBucket:
        if key not in buckets:
            tier = api_keys[key].get("tier", "trial")
            limits = RATE_TIERS.get(tier, RATE_TIERS["trial"])
            buckets[key] = TokenBucket(limits["rate"], limits["burst"])
        return buckets[key]

    def verify_api_key(request: Request) -> str:
        """Validate Bearer token, return the API key. Raises HTTPException on failure."""
        auth = request.headers.get("Authorization", "")
        if not auth:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Pass Authorization: Bearer sk_winnow_...",
            )
        parts = auth.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer":
            raise HTTPException(status_code=401, detail="Invalid API key")
        token = parts[1]
        if token not in api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return token

    web_app = FastAPI(title="Winnow")

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model_loaded": model is not None}

    @web_app.post("/prune", response_model=PruneResponse)
    async def prune(request: PruneRequest, raw_request: Request):
        api_key = verify_api_key(raw_request)

        bucket = get_bucket(api_key)
        if not bucket.allow():
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "1"},
            )

        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        response = await loop.run_in_executor(executor, model.prune, request)
        latency_ms = int((time.monotonic() - t0) * 1000)

        agent = request.client_info.agent if request.client_info else "unknown"
        model_id = request.client_info.model if request.client_info else ""
        print(f"prune | agent={agent} model={model_id} latency={latency_ms}ms")

        response_dict = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else response.dict()
        )
        response_dict["latency_ms"] = latency_ms
        return response

    return web_app
