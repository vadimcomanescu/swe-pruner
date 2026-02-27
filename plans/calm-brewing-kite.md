# Dynamic Batching for Winnow Modal Deployment

## Context

Winnow's Modal deployment processes each `/prune` request as a separate GPU forward pass (batch_size=1). Under load (30 concurrent requests, 3 T4 containers), requests queue behind a single-worker ThreadPoolExecutor per container, causing p90=58s, 5% timeout rate, 0.86 req/s total. Unacceptable for a SaaS product.

The model already supports batch dimension `[B, L, ...]` throughout (confirmed in `model_structure.py` and `swepruner.py`). The fix is architectural: batch multiple requests into a single forward pass.

### What about torch.compile / INT8?

Investigated with citations. Neither is the right move right now:

- **torch.compile on T4**: 10-20% speedup realistically ([HuggingFace benchmarks](https://huggingface.co/docs/transformers/en/perf_torch_compile)), but 20-60s cold-start compilation overhead kills scale-to-zero. Flash Attention 2 causes graph breaks ([Benjamin Warner benchmark](https://benjaminwarner.dev/2023/08/16/flash-attention-compile)).

- **INT8 quantization**: Works via bitsandbytes, but [Qwen3-0.6B shows ~10% accuracy drop at INT4](https://arxiv.org/html/2505.02214v1) and even INT8 degrades small models measurably. For a reranker where score quality = pruning quality, this needs careful calibration testing before production. Not a quick win.

**Batching is the main lever.** Pure architectural improvement, zero quality impact, 3-5x throughput gain.

## Architecture Change

```
BEFORE:  HTTP -> FastAPI (GPU container) -> ThreadPool(1) -> model.prune(1 req) -> GPU [B=1]

AFTER:   HTTP -> FastAPI (CPU, @modal.concurrent) -> engine.infer.remote.aio(req)
                                                           |
                                                     Modal batching layer
                                                     (accumulates N reqs over 50ms)
                                                           |
                                                     InferenceEngine (GPU, @modal.cls)
                                                     -> prune_batch(N reqs) -> GPU [B=N]
```

Two Modal entities on the same app:
- **`InferenceEngine`** (`@app.cls(gpu="T4")`) - loads model, `@modal.batched` inference
- **`serve()`** (`@app.function()`, no GPU) - FastAPI with auth/rate-limiting/usage-tracking

GPU containers only run inference. CPU containers handle HTTP concurrency cheaply.

## Changes

### 1. `swe-pruner/src/swe_pruner/prune_wrapper.py` (additions only, existing code untouched)

**`build_input_for_llm_unpadded(query, code, tokenizer, max_length, instruction)`**
- Same tokenization as `build_input_for_llm` but returns raw `list[int]` + metadata, no padding/tensors
- Reuses existing prefix/suffix/truncation logic

**`batch_inputs(unpadded_inputs, pad_token_id) -> (input_ids [B, L], attention_mask [B, L], metadata_list)`**
- Pads to longest-in-batch (not global 8192), stacks into tensors
- Key optimization: batch of 8 requests averaging 2K tokens = `[8, 2048]` not `[8, 8192]`, ~4x less memory

**`SwePrunerForCodePruning._process_chunk_batch(queries, code_chunks, max_length)`**
- Calls `build_input_for_llm_unpadded` for each, `batch_inputs`, single `self.forward()`, slices results
- Returns `list[(chunk_score, token_scores, offsets)]`

**`SwePrunerForCodePruning.prune_batch(requests: list[PruneRequest]) -> list[PruneResponse]`**
- Partitions into single-chunk vs multi-chunk requests
- Single-chunk: batched forward via `_process_chunk_batch`
- Multi-chunk: sequential `self.prune(req)` (rare, already saturates GPU)
- Post-processes each through existing `aggregate_token_scores_to_lines` and `prune_code_lines`

### 2. `deploy/modal_app.py` (restructure)

**`InferenceEngine` class:**
```python
@app.cls(gpu="T4", scaledown_window=300, max_containers=3,
         secrets=[modal.Secret.from_name("winnow-supabase")])
class InferenceEngine:
    @modal.enter()
    def load_model(self):
        self.model = SwePrunerForCodePruning.from_pretrained(MODEL_DIR)

    @modal.batched(max_batch_size=8, wait_ms=50)
    async def infer(self, requests: list) -> list:
        # OOM-safe batched inference with recursive split-retry
```

- `max_batch_size=8`: conservative for T4 at worst-case 8192-token inputs. [Memory analysis: ~1.1GB/seq at 8192, 8 seqs = ~10GB of 14.8GB available](estimated from KV cache + activation calculations).
- `wait_ms=50`: at most 50ms added latency, enough to accumulate 1-5 requests under load
- OOM recovery: catch OOM -> `empty_cache()` -> split batch in half -> retry recursively -> single-request fallback returns unpruned
- `empty_cache()` in `finally` after every batch

**`serve()` function:**
```python
@app.function(secrets=[modal.Secret.from_name("winnow-supabase")])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    engine = InferenceEngine()
    # FastAPI with same auth/rate-limiting/usage-tracking
    # Endpoint: await engine.infer.remote.aio(request)
```

- No GPU, no ThreadPoolExecutor
- `@modal.concurrent(max_inputs=100)` for cheap HTTP concurrency
- Each request calls `engine.infer.remote.aio(request)`, Modal batches on GPU side

### 3. No changes
- `model_structure.py` - already batch-ready (all ops `[B, L, ...]`)
- `swepruner.py` - already batch-ready
- `online_serving.py` - local server, unaffected
- `deploy/stress_test.py` - reuse as-is for verification

## Expected Performance

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| Throughput/container | ~0.3 req/s | ~0.9-1.5 req/s |
| Total throughput (3 containers) | ~0.9 req/s | ~3-5 req/s |
| p90 latency @ 30 concurrent | 58s | 15-25s |
| Timeout rate | 5% | ~0% |
| GPU cost efficiency | 1x | 3-5x |

## Verification

1. Deploy: `modal deploy deploy/modal_app.py`
2. Health check: `curl https://vadimcomanescu--winnow-serve.modal.run/health`
3. Stress test: `uv run --with httpx python3 deploy/stress_test.py --url https://vadimcomanescu--winnow-serve.modal.run --token sk_winnow_test123 -n 30 --max-files 60`
4. Compare against baseline: p90=58s, 0.86 req/s, 5% failures
5. Send batch of very large files to verify OOM recovery works (split-retry, graceful fallback)
