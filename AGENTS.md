# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Winnow is a neural code pruner that selectively removes irrelevant lines from source code before feeding it to LLM agents, reducing token costs 23-54%. It uses a 0.6B parameter model (Qwen3-Reranker-0.6B backbone) with multi-layer fusion and CRF-based compression head.

## Project Layout

- `winnow/` - Core package (published to PyPI as `winnow`). Contains the model, inference code, and FastAPI server.
- `downstream_eval/` - Benchmark evaluation scripts (SWE-bench, LongCodeQA, LCC). Expects Slurm with 4+ GPUs.
- `examples/` - Integration demos for Claude Agent SDK and OpenHands.
- `utils/` - Analysis scripts: threshold optimization, score distribution, token visualization.
- `data/` - Experiment trace archives and hyperparameter configs.

## Build & Run (core package)

```bash
cd winnow
uv sync                                      # install deps (Python 3.13, CUDA/cu124)
winnow --model-path ./model --port 8000   # start FastAPI server
```

Model weights are tracked via git-lfs. If `model.safetensors` shows a pointer file, run `git lfs pull`. Alternatively download from HuggingFace:
```bash
huggingface-cli download ayanami-kitasan/code-pruner --local-dir ./model
```

Flash Attention 2 is required. Install the pre-built wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) or build from source with `pip install flash-attn --no-build-isolation`.

## Testing

```bash
# Smoke-test the running server (requires httpie + jq)
cd winnow
./test-prune.sh "error handling" some_file.py
```

No pytest suite exists for the core package yet.

## API Endpoints

- `GET /health` - health check
- `POST /prune` - accepts `PruneRequest` JSON (`query`, `code`, `threshold`, `always_keep_first_frags`, `chunk_overlap_tokens`), returns `PruneResponse`

## Architecture (winnow/src/winnow/)

The inference pipeline flows through four files:

1. **configuration.py** - `WinnowConfig(PretrainedConfig)`: model hyperparameters (backbone path, bottleneck dim, CRF vs FFN head type, fusion layer count, dropout).
2. **model_structure.py** - `TokenScorer`: the neural network. Runs a Qwen3-Reranker-0.6B backbone, extracts hidden states from early/middle/final layers, concatenates them (multi-layer fusion), passes through self-attention fusion layers, then a CRF compression head that emits per-token keep/drop logits. Also produces a document-level relevance score via yes/no logit comparison on the last token.
3. **winnow.py** - `WinnowForCodeCompression(PreTrainedModel)`: thin wrapper that calls `TokenScorer.forward()` and returns `WinnowOutput(token_logits, score_logits)`.
4. **prune_wrapper.py** - `WinnowForCodePruning`: the user-facing class. Handles tokenization, chunking for long code (with configurable overlap), maps token-level scores to line-level decisions, applies threshold filtering, and reassembles pruned output. Entry point: `model.prune(PruneRequest) -> PruneResponse`.
5. **online_serving.py** - FastAPI app + Typer CLI. Wraps `WinnowForCodePruning` behind `/prune` endpoint. Inference runs in a single-worker `ThreadPoolExecutor` to avoid GIL contention with async.

Key data flow: `PruneRequest` -> tokenize -> chunk if needed -> `WinnowForCodeCompression.forward()` -> token logits -> line-level score aggregation -> threshold filter -> `PruneResponse` with pruned code.

## Environment

- Python 3.13, PyTorch (cu124 index), Transformers 4.57.6
- Uses `uv` for dependency management with `pyproject.toml` + `uv.lock`
- CUDA GPU required for inference
- `WINNOW_MODEL_PATH` env var overrides default model location (`./model`)
