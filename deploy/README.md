# Modal Deployment

Serverless deployment of SWE-Pruner on Modal (L4 GPU, scale-to-zero).

## Live Endpoint

```
https://vadimcomanescu--swe-pruner-serve.modal.run
```

## Prerequisites

```bash
uv tool install modal
modal setup   # authenticates to your Modal workspace
```

## Deploy

```bash
# Production deploy (persistent URL, auto-restarts on update)
modal deploy deploy/modal_app.py

# Dev mode (live reload on file changes, tears down on Ctrl-C)
modal serve deploy/modal_app.py
```

After `modal deploy`, Modal prints the endpoint URL. It stays up until you redeploy or run `modal app stop swe-pruner`.

## Redeploy after code changes

The local `swe-pruner/src/swe_pruner/` package is mounted into the image via `add_local_dir`. Rerunning `modal deploy` picks up any Python changes without rebuilding the image layers (flash-attn, model weights, etc. are all cached).

## Test the endpoint

```bash
# Health check
curl https://vadimcomanescu--swe-pruner-serve.modal.run/health

# Prune request
curl -s -X POST https://vadimcomanescu--swe-pruner-serve.modal.run/prune \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling",
    "code": "def foo():\n    try:\n        pass\n    except Exception as e:\n        raise\n\ndef bar():\n    x = 1\n    return x\n",
    "threshold": 0.5
  }'
```

## How it works

`deploy/modal_app.py` builds a single image layer stack:

| Layer | What it does | Cached? |
|-------|-------------|---------|
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | Base: PyTorch + CUDA + gcc/nvcc | registry cache |
| `pip_install packaging setuptools wheel ninja` | Build tools for flash-attn | yes, after first build |
| `pip_install flash-attn --no-build-isolation` | Flash Attention 2 (~26s compile) | yes, after first build |
| `pip_install transformers fastapi ...` | Runtime deps | yes |
| `snapshot_download ayanami-kitasan/code-pruner` | Model weights baked in | yes, ~1.35 GB |
| `add_local_dir swe-pruner/src/swe_pruner` | Local package (re-synced on every deploy) | no |

The app function runs on an L4 GPU with a 5-minute scaledown window. Cold start (including model load) takes ~30 seconds; warm requests are fast.

## Configuration

Edit `modal_app.py` to change:
- `gpu="L4"` -- switch to `"A10G"`, `"A100"`, etc.
- `scaledown_window=300` -- seconds idle before scale-to-zero
- `MODEL_REPO` -- swap in a different HuggingFace model
