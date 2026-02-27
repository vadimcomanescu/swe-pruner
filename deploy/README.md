# Modal Deployment

Serverless deployment of SWE-Pruner on Modal (L4 GPU, scale-to-zero).

## Live Endpoint

```
https://vadimcomanescu--winnow-serve.modal.run
```

## Prerequisites

```bash
uv tool install modal
modal setup   # authenticates to your Modal workspace
```

## Database Setup (Supabase)

API keys and usage logs live in a `winnow` schema inside the existing Supabase project. The pruner connects directly via Postgres (through PgBouncer pooler).

### 1. Create the schema

Paste the contents of `deploy/schema.sql` into the Supabase SQL editor and run it. This creates:

- `winnow.api_keys` -- SaaS writes keys, pruner reads them
- `winnow.usage_log` -- pruner batch-writes usage events, SaaS queries for dashboards

### 2. Create the Modal secret

Get the **Session mode** connection string from Supabase dashboard (Settings > Database, port 6543):

```bash
modal secret create winnow-supabase \
  DATABASE_URL="postgresql://postgres.[project-ref]:[password]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"
```

### 3. Insert a test key

```sql
INSERT INTO winnow.api_keys (key_hash, key_prefix, tier)
VALUES (
  encode(sha256(convert_to('sk_winnow_test123', 'UTF8')), 'hex'),
  'sk_winnow_te...',
  'pro'
);
```

The pruner hashes incoming bearer tokens with SHA-256 and looks up the hash. The SaaS app is responsible for generating keys and storing their hashes.

### Key management

Keys are managed by the SaaS app (separate repo), not by redeploying. To add/revoke keys:

- **Add**: SaaS inserts a row into `winnow.api_keys` with the SHA-256 hash. The pruner picks it up within 60s (or immediately on first use via cache-miss query).
- **Revoke**: SaaS sets `revoked_at = now()`. The pruner stops accepting the key within 60s.

### Rate limit tiers

Per-key, per-container token bucket. State resets when containers scale down.

| Tier | Rate | Burst |
|------|------|-------|
| trial | 2 req/s | 5 |
| pro | 5 req/s | 10 |
| team | 10 req/s | 20 |

### Response codes

| Scenario | Status | Body |
|----------|--------|------|
| No `Authorization` header | 401 | `{"detail": "Missing API key. Pass Authorization: Bearer sk_winnow_..."}` |
| Invalid key | 401 | `{"detail": "Invalid API key"}` |
| Rate limited | 429 | `{"detail": "Rate limit exceeded"}` + `Retry-After: 1` header |
| Valid | 200 | Normal `PruneResponse` |

### Usage logging

The pruner buffers usage events in memory and bulk-inserts them every 10 seconds (or every 100 rows). This keeps DB round-trips to ~6/min/container even under heavy agent traffic. Events are also flushed on container shutdown.

Query usage from the SaaS side:

```sql
SELECT key_hash, count(*), sum(tokens_saved), avg(latency_ms)
FROM winnow.usage_log
WHERE ts > now() - interval '24 hours'
GROUP BY key_hash;
```

## Deploy

```bash
# Production deploy (persistent URL, auto-restarts on update)
modal deploy deploy/modal_app.py

# Dev mode (live reload on file changes, tears down on Ctrl-C)
modal serve deploy/modal_app.py
```

After `modal deploy`, Modal prints the endpoint URL. It stays up until you redeploy or run `modal app stop winnow`.

## Redeploy after code changes

The local `swe-pruner/src/swe_pruner/` package is mounted into the image via `add_local_dir`. Rerunning `modal deploy` picks up any Python changes without rebuilding the image layers (flash-attn, model weights, etc. are all cached).

## Test the endpoint

```bash
# Health check (no auth required)
curl https://vadimcomanescu--winnow-serve.modal.run/health

# Prune request (requires API key)
curl -s -X POST https://vadimcomanescu--winnow-serve.modal.run/prune \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_winnow_test123" \
  -d '{
    "query": "error handling",
    "code": "def foo():\n    try:\n        pass\n    except Exception as e:\n        raise\n\ndef bar():\n    x = 1\n    return x\n",
    "threshold": 0.5
  }'

# Verify usage was logged (wait 10s for buffer flush)
# In Supabase SQL editor:
#   SELECT * FROM winnow.usage_log ORDER BY ts DESC LIMIT 5;
```

## How it works

`deploy/modal_app.py` builds a single image layer stack:

| Layer | What it does | Cached? |
|-------|-------------|---------|
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | Base: PyTorch + CUDA + gcc/nvcc | registry cache |
| `pip_install packaging setuptools wheel ninja` | Build tools for flash-attn | yes, after first build |
| `pip_install flash-attn --no-build-isolation` | Flash Attention 2 (~26s compile) | yes, after first build |
| `pip_install transformers fastapi asyncpg ...` | Runtime deps | yes |
| `snapshot_download ayanami-kitasan/code-pruner` | Model weights baked in | yes, ~1.35 GB |
| `add_local_dir swe-pruner/src/swe_pruner` | Local package (re-synced on every deploy) | no |
| `add_local_file deploy/db.py` | Database module (re-synced on every deploy) | no |

The app function runs on a T4 GPU with a 5-minute scaledown window. Cold start (including model load + DB connect) takes ~30 seconds; warm requests are fast.

## Configuration

Edit `modal_app.py` to change:
- `gpu="T4"` -- switch to `"A10G"`, `"A100"`, etc.
- `scaledown_window=300` -- seconds idle before scale-to-zero
- `MODEL_REPO` -- swap in a different HuggingFace model
