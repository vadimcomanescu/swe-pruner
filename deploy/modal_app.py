"""
Winnow on Modal -- dynamic batching with separate GPU and CPU containers.

Architecture:
  HTTP -> FastAPI (CPU, @modal.concurrent) -> engine.infer.remote.aio(req)
                                                     |
                                               Modal batching layer
                                               (accumulates N reqs over 50ms)
                                                     |
                                               InferenceEngine (GPU, @modal.cls)
                                               -> prune_batch(N reqs) -> GPU [B=N]

Deploy:  modal deploy deploy/modal_app.py
Serve (dev): modal serve deploy/modal_app.py
"""

import logging
import modal
import os
import time
from pathlib import Path

log = logging.getLogger("winnow.modal")

MODEL_REPO = "ayanami-kitasan/code-pruner"
MODEL_DIR = "/model"
LOCAL_PKG = Path(__file__).parent.parent / "swe-pruner" / "src" / "swe_pruner"
DB_MODULE = Path(__file__).parent / "db.py"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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
        "asyncpg>=0.30.0",
    )
    .run_commands(
        f"python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO}', local_dir='{MODEL_DIR}')"
        f"\""
    )
    .add_local_dir(LOCAL_PKG, "/root/swe_pruner")
    .add_local_file(DB_MODULE, "/root/db.py")
)

app = modal.App("winnow", image=image)

RATE_TIERS: dict[str, dict[str, float]] = {
    "trial": {"rate": 2.0, "burst": 5},
    "pro": {"rate": 5.0, "burst": 10},
    "team": {"rate": 10.0, "burst": 20},
}

INFERENCE_TIMEOUT_S = 60


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


# ---------------------------------------------------------------------------
# GPU container: loads model, batched inference
# ---------------------------------------------------------------------------


@app.cls(gpu="T4", scaledown_window=300, max_containers=3)
class InferenceEngine:
    @modal.enter()
    def load_model(self):
        import sys
        import torch

        sys.path.insert(0, "/root")
        from swe_pruner.prune_wrapper import SwePrunerForCodePruning

        os.environ["SWEPRUNER_MODEL_PATH"] = MODEL_DIR
        self.model = SwePrunerForCodePruning.from_pretrained(MODEL_DIR)
        log.info(
            "Model loaded on %s, VRAM: %.1f MB allocated",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        )

    @modal.batched(max_batch_size=8, wait_ms=50)
    async def infer(self, requests: list[dict]) -> list[dict]:
        from swe_pruner.prune_wrapper import PruneRequest

        prune_requests = [PruneRequest(**r) for r in requests]
        log.info("Batched inference: %d requests", len(prune_requests))
        results = self._safe_batch_infer(prune_requests)
        return [r.model_dump() for r in results]

    def _safe_batch_infer(self, prune_requests: list) -> list:
        """Run prune_batch with OOM recovery: split batch in half recursively."""
        import torch
        from swe_pruner.prune_wrapper import PruneResponse

        try:
            return self.model.prune_batch(prune_requests)
        except torch.cuda.OutOfMemoryError:
            log.warning(
                "CUDA OOM on batch of %d, splitting", len(prune_requests)
            )
            torch.cuda.empty_cache()
            if len(prune_requests) == 1:
                req = prune_requests[0]
                code_lines = req.code.splitlines()
                return [
                    PruneResponse(
                        score=0.0,
                        pruned_code=req.code,
                        token_scores=[],
                        kept_frags=list(range(1, len(code_lines) + 1)),
                        origin_token_cnt=0,
                        left_token_cnt=0,
                        model_input_token_cnt=0,
                        error_msg="CUDA out of memory, returned unpruned",
                    )
                ]
            mid = len(prune_requests) // 2
            left = self._safe_batch_infer(prune_requests[:mid])
            right = self._safe_batch_infer(prune_requests[mid:])
            return left + right
        finally:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CPU container: FastAPI with auth, rate limiting, usage tracking
# ---------------------------------------------------------------------------


@app.function(secrets=[modal.Secret.from_name("winnow-supabase")])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    import asyncio
    import sys
    from contextlib import asynccontextmanager

    sys.path.insert(0, "/root")

    import db
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from swe_pruner.prune_wrapper import PruneRequest, PruneResponse

    engine = InferenceEngine()

    buckets: dict[str, TokenBucket] = {}

    def get_bucket(key_hash: str, tier: str) -> TokenBucket:
        if key_hash not in buckets:
            limits = RATE_TIERS.get(tier, RATE_TIERS["trial"])
            buckets[key_hash] = TokenBucket(limits["rate"], limits["burst"])
        return buckets[key_hash]

    async def verify_api_key(request: Request) -> tuple[str, str]:
        """Validate Bearer token. Returns (key_hash, tier)."""
        auth = request.headers.get("Authorization", "")
        if not auth:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Pass Authorization: Bearer sk_winnow_...",
            )
        parts = auth.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer":
            raise HTTPException(status_code=401, detail="Invalid API key")
        result = await db.validate_key(parts[1])
        if result is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return result

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dsn = os.environ["DATABASE_URL"]
        await db.init(dsn)
        await db.load_keys()
        db.start_background_tasks()
        yield
        await db.shutdown()

    web_app = FastAPI(title="Winnow", lifespan=lifespan)

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "architecture": "batched"}

    @web_app.post("/prune", response_model=PruneResponse)
    async def prune(request: PruneRequest, raw_request: Request):
        key_hash, tier = await verify_api_key(raw_request)

        bucket = get_bucket(key_hash, tier)
        if not bucket.allow():
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "1"},
            )

        t0 = time.monotonic()
        try:
            response_dict = await asyncio.wait_for(
                engine.infer.remote.aio(request.model_dump()),
                timeout=INFERENCE_TIMEOUT_S,
            )
            response = PruneResponse(**response_dict)
        except asyncio.TimeoutError:
            log.error("Inference timed out after %ds", INFERENCE_TIMEOUT_S)
            raise HTTPException(status_code=504, detail="Inference timed out")
        except Exception:
            log.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed")

        latency_ms = int((time.monotonic() - t0) * 1000)

        agent = request.client_info.agent if request.client_info else "unknown"
        model_id = request.client_info.model if request.client_info else ""

        db.record_usage(
            key_hash=key_hash,
            tokens_in=response.origin_token_cnt,
            tokens_out=response.left_token_cnt,
            latency_ms=latency_ms,
            agent=agent,
            model=model_id,
            threshold=request.threshold,
            score=response.score,
            error_msg=response.error_msg,
        )

        return response

    return web_app
