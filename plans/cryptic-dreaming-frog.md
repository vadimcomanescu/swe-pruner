# Deploying Winnow as a Public Service

## Context

Winnow runs as a local FastAPI server with GPU inference. Goal: deploy publicly for thousands of programmers at low cost.

**Model reality**: Despite being "0.6B parameters", the model uses ~9GB VRAM because:
- Qwen3-Reranker-0.6B backbone loaded with all hidden states (`output_hidden_states=True`)
- Multi-layer fusion concatenates 3 hidden state layers (early/middle/final), tripling the fused dimension
- Float32 upcasts in `TokenScorer.forward()` for fusion and compression heads
- 8192 max sequence length with KV cache + Flash Attention 2 overhead
- CUDA context overhead

## Recommendation: Modal

Modal is the best fit: keeps FastAPI native, scale-to-zero, minimal code changes.

| Factor | Modal | RunPod Serverless | SaladCloud |
|--------|-------|-------------------|------------|
| Code changes | Minimal (add decorators) | Rewrite to RunPod SDK | Docker + custom orchestration |
| FastAPI native | Yes, `@modal.asgi_app()` | No | No |
| Scale to zero | Yes | Yes | No |
| Cold start | 2-4s | 2s (FlashBoot) | Slow, unreliable |
| Reliability | High (datacenter) | High | Low (consumer GPUs) |
| Free tier | $30/mo credits | None | None |

### GPU selection

| GPU | VRAM | Modal $/hr | Fit? |
|-----|------|-----------|------|
| T4 | 16GB | ~$0.59 | Tight. 7GB headroom for 8K-token activations. Single request only, risk of OOM on long inputs. |
| **L4** | **24GB** | **~$1.10** | **Comfortable. 15GB headroom. Can handle a few concurrent requests.** |
| A10G | 24GB | ~$1.10 | Same as L4, wider availability. |

**Use L4 or A10G**, not T4. The T4's 7GB headroom is too tight when processing 8K-token sequences with all hidden states materialized in float32.

### Cost estimate (1000 DAU)

- ~10 prune requests/user/day = 10,000 requests/day
- Each request: ~200-500ms on L4 (8K tokens, single forward pass)
- Active GPU time: ~3000-5000s/day = ~1-1.4 GPU-hours/day
- Monthly: ~1.2 * 30 * $1.10 = **~$40/mo** on L4
- At 10x traffic (10K DAU): ~$400/mo
- Scale-to-zero means zero cost when idle (nights, weekends)

## Implementation Plan

### 1. Create Modal deployment script

**New file**: `winnow/deploy/modal_app.py`

```python
import modal

app = modal.App("winnow")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "fastapi", "uvicorn",
        "flash-attn", "typer", "huggingface-hub", "pydantic",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("winnow")
)

# Bake model weights into a Modal volume (persistent, shared across containers)
model_volume = modal.Volume.from_name("winnow-model", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4",                    # 24GB VRAM, comfortable for 9GB model
    volumes={"/model": model_volume},
    container_idle_timeout=300,  # keep warm 5min after last request
    allow_concurrent_inputs=4,   # limited concurrency per container
    secrets=[modal.Secret.from_name("winnow-keys")],
)
@modal.asgi_app()
def serve():
    import os
    os.environ["WINNOW_MODEL_PATH"] = "/model"
    from winnow.online_serving import app as fastapi_app
    return fastapi_app
```

### 2. Modify `online_serving.py`

**File**: `winnow/src/winnow/online_serving.py`

Changes:
- **Configurable concurrency**: Change `ThreadPoolExecutor(max_workers=1)` to read from `WINNOW_WORKERS` env var (default 1, set to 4 on Modal with L4)
- **Bearer token auth middleware**: Check `Authorization: Bearer <key>` against `WINNOW_API_KEYS` env var (comma-separated list)
- **CORS middleware**: Allow browser-based MCP clients
- **Request size limit**: Cap `code` field length to prevent abuse (e.g., 500KB)

### 3. Model weights setup

One-time upload to Modal volume:
```bash
modal volume put winnow-model ./model/
```

Or bake into image via HuggingFace download at build time:
```python
image = image.run_commands(
    "huggingface-cli download ayanami-kitasan/code-pruner --local-dir /model"
)
```

### 4. Deployment

```bash
modal deploy deploy/modal_app.py           # production
modal serve deploy/modal_app.py            # dev (temporary URL)
```

Modal provides a stable `*.modal.run` URL. Custom domain can be configured.

## Files to create/modify

| File | Action |
|------|--------|
| `winnow/deploy/modal_app.py` | Create - Modal deployment config |
| `winnow/src/winnow/online_serving.py` | Modify - auth, CORS, configurable workers, request limits |
| `winnow/deploy/README.md` | Create - deployment instructions |

## Alternatives to revisit later

- **RunPod**: ~25% cheaper per GPU-hour but requires rewriting handler away from FastAPI
- **SaladCloud**: Absurdly cheap ($0.50/hr A100) for batch workloads, but unreliable for real-time API
- **Self-hosted Hetzner/OVH**: Dedicated GPU server ~$80-120/mo, no scale-to-zero but predictable for steady traffic

## Verification

1. `modal serve deploy/modal_app.py` - temporary dev endpoint
2. `curl -X POST https://<endpoint>/prune -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d '{"query":"error handling","code":"def foo():\n  pass","threshold":0.5}'`
3. `curl https://<endpoint>/health`
4. Load test with `hey -n 1000 -c 10` to verify concurrency and no OOMs
