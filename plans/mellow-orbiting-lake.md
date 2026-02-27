# Supabase-backed API Keys + Per-Key Usage Logging

## Context

The pruner stores API keys in a Modal env var (JSON blob). Adding a key requires redeploying. There's no usage tracking and no way for a SaaS frontend to manage keys or show users their stats. Users are managed in a separate SaaS that uses Supabase (Postgres). Each user can have multiple API keys. Autonomous AI agents will pound `/prune` on every file read/grep, so traffic scales fast: 1,000+ req/day per active user, 100K-1M+ req/day at scale.

## Architecture

```
SaaS (separate repo)                Pruner (this repo, Modal)
┌──────────────────┐                ┌──────────────────────┐
│ auth.users        │                │ NO user concept       │
│ winnow.api_keys ──── reads ──────│ validates key_hash    │
│ billing/Stripe    │                │ enforces tier/rate    │
│                  │◄── reads ──────│ winnow.usage_log      │
│ dashboard UI      │                │ (batched writes)      │
└──────────────────┘                └──────────────────────┘
```

- Pruner sees keys and tiers, never users
- SaaS creates/revokes keys, pruner validates them
- Pruner batch-writes usage events, SaaS queries them for dashboards
- No `/usage` endpoint on the pruner; SaaS queries Supabase directly

## Database: `winnow` schema (new, in existing Supabase project)

Existing travel/altrad tables stay in `public` untouched. Winnow gets its own schema.

### `deploy/schema.sql` (new file)

```sql
CREATE SCHEMA IF NOT EXISTS winnow;

-- API keys (SaaS writes, pruner reads)
CREATE TABLE winnow.api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash    TEXT NOT NULL UNIQUE,       -- SHA-256 hex of raw key
    key_prefix  TEXT NOT NULL,              -- "sk_winnow_a1b2..." for display
    tier        TEXT NOT NULL DEFAULT 'trial'
                    CHECK (tier IN ('trial', 'pro', 'team')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ,               -- NULL = active
    user_id     UUID,                      -- FK to auth.users
    label       TEXT DEFAULT ''
);

CREATE INDEX idx_api_keys_hash ON winnow.api_keys (key_hash)
    WHERE revoked_at IS NULL;

-- Usage log (pruner writes, SaaS reads)
CREATE TABLE winnow.usage_log (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    key_hash        TEXT NOT NULL,          -- not FK, for resilience
    tokens_in       INT NOT NULL,
    tokens_out      INT NOT NULL,
    tokens_saved    INT GENERATED ALWAYS AS (tokens_in - tokens_out) STORED,
    latency_ms      INT NOT NULL,
    agent           TEXT NOT NULL DEFAULT 'unknown',
    model           TEXT NOT NULL DEFAULT '',
    threshold       REAL NOT NULL,
    score           REAL,
    error_msg       TEXT
);

CREATE INDEX idx_usage_log_key_ts ON winnow.usage_log (key_hash, ts DESC);
CREATE INDEX idx_usage_log_ts ON winnow.usage_log (ts DESC);
```

Key design decisions:
- `usage_log.key_hash` is TEXT not FK. If a key is deleted, usage history survives.
- `tokens_saved` is a generated column. SaaS queries don't need app-level math.
- SaaS joins `usage_log.key_hash → api_keys.key_hash → api_keys.user_id → auth.users` for per-user dashboards.

## Key validation

Standard pattern (GitHub, Stripe): SaaS stores `SHA-256(raw_key)`, pruner hashes incoming bearer token and looks up the hash.

1. **Container startup**: fetch all active keys via `SELECT key_hash, tier FROM winnow.api_keys WHERE revoked_at IS NULL`. Populate in-memory cache.
2. **Background refresh**: every 60s, re-run the query, atomic dict swap.
3. **Per-request**: hash incoming key, check cache (O(1) dict lookup), return tier.
4. **Cache miss**: single-row query for that hash (handles keys created between refreshes). If found, add to cache. If not, 401.
5. **Revoked keys**: stale for max 60s until next refresh. Acceptable.

## Usage logging: batched writes

Agents hit `/prune` on every file read. At scale this is 10-20+ req/sec sustained with higher bursts. Per-request INSERT is wasteful.

**Strategy: buffer in memory, bulk INSERT periodically.**

- Append each usage event to an in-memory list (~200 bytes/row)
- Flush every 10 seconds OR every 100 rows (whichever first)
- One bulk `INSERT INTO winnow.usage_log VALUES (...), (...), (...)` per flush
- Flush on SIGTERM (Modal sends this before killing container on scale-down)
- If flush fails, log warning, discard batch. This is analytics, not billing.
- Memory: even 10,000 buffered rows = ~2MB. The model uses 9GB. Irrelevant.

Result: ~6 DB calls/min/container instead of potentially thousands.

## Connection: direct Postgres via asyncpg

Not the Supabase REST API (too much HTTP overhead per call at scale). Direct Postgres connection through Supabase's built-in PgBouncer pooler (port 6543).

- One persistent connection per Modal container, reused for key refresh + usage flushes
- No per-query cost. Included in every Supabase tier.
- Connection limit: ~15 on free tier. With max 3 containers = 3 connections. Fine.

## Files to change

### 1. `deploy/schema.sql` -- NEW (~30 lines)
SQL above. Run in Supabase SQL editor. Version-controlled here as source of truth.

### 2. `deploy/db.py` -- NEW (~130 lines)
All database interaction:

```
KeyCache class
  - _keys: dict[str, str]  (hash → tier)
  - replace(keys): atomic swap
  - get(hash) → tier | None

UsageBuffer class
  - _buffer: list[dict]
  - append(event): add to list
  - flush() → list[dict]: swap buffer, return old contents

Module-level functions:
  - init(dsn: str): create asyncpg connection
  - load_keys(): SELECT → populate KeyCache
  - start_background_tasks(): key refresh (60s) + usage flush (10s/100 rows)
  - validate_key(raw_key) → (key_hash, tier) | None
  - record_usage(key_hash, tokens_in, tokens_out, ...): append to buffer
  - shutdown(): final flush + close connection
```

### 3. `deploy/modal_app.py` -- MODIFY

**Remove:**
- `import json`
- `load_api_keys()` function
- `api_keys` dict and direct key lookup
- `verify_api_key()` sync function
- `winnow-keys` secret reference
- The `print()` log line

**Add:**
- `asyncpg` to pip_install list
- Copy `deploy/db.py` into the image (`.add_local_file`)
- `winnow-supabase` secret (contains `DATABASE_URL`)
- FastAPI `startup` event: `await db.init(dsn)`, `await db.load_keys()`, `db.start_background_tasks()`
- FastAPI `shutdown` event: `await db.shutdown()` (final flush)
- Auth becomes async: extract bearer token, call `await db.validate_key(token)`, returns `(key_hash, tier)` or raises 401
- `get_bucket()` keyed on `key_hash` instead of raw key
- After inference: `db.record_usage(key_hash=..., tokens_in=response.origin_token_cnt, tokens_out=response.left_token_cnt, ...)` (appends to buffer, no await needed)

**Unchanged:**
- `TokenBucket` class and rate limiting logic (stays in-memory, per-container)
- Image build layers (flash-attn, model weights, etc.)
- GPU config, scaledown_window, max_containers
- `/health` endpoint
- Inference executor pattern

### 4. `deploy/README.md` -- MODIFY
- Replace "API Key Setup" section (remove WINNOW_API_KEYS JSON blob)
- Add Supabase setup: schema creation, secret creation
- Document `DATABASE_URL` secret format

## What does NOT change

- `swe-pruner/src/swe_pruner/*` (core package, models, inference) -- untouched
- `swe-pruner/src/swe_pruner/online_serving.py` (local server, no auth) -- untouched
- `swe-pruner/src/swe_pruner/prune_wrapper.py` (PruneRequest/PruneResponse) -- untouched
- MCP server -- untouched
- Existing Supabase tables (travel app, altrad) -- untouched

## Modal secret

```bash
modal secret create winnow-supabase \
  DATABASE_URL="postgresql://postgres.[project-ref]:[password]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"
```

Use the "Session mode" (port 6543) connection string from Supabase dashboard > Settings > Database.

## Verification

```bash
# 1. Create schema in Supabase SQL editor (paste deploy/schema.sql)

# 2. Insert a test key:
#    INSERT INTO winnow.api_keys (key_hash, key_prefix, tier)
#    VALUES (encode(sha256(convert_to('sk_winnow_test123', 'UTF8')), 'hex'),
#            'sk_winnow_te...', 'pro');

# 3. Create Modal secret
modal secret create winnow-supabase \
  DATABASE_URL="postgresql://postgres.[ref]:[pw]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"

# 4. Dev-serve
modal serve deploy/modal_app.py

# 5. Test auth works (should return pruned code)
curl -s -X POST https://...modal.run/prune \
  -H "Authorization: Bearer sk_winnow_test123" \
  -H "Content-Type: application/json" \
  -d '{"query":"error handling","code":"def foo():\n    pass\n","threshold":0.5}'

# 6. Test invalid key returns 401
curl -s -X POST https://...modal.run/prune \
  -H "Authorization: Bearer sk_winnow_bogus" \
  -H "Content-Type: application/json" \
  -d '{"query":"test","code":"x=1","threshold":0.5}'

# 7. Wait 10s for buffer flush, verify usage in Supabase:
#    SELECT * FROM winnow.usage_log ORDER BY ts DESC LIMIT 5;

# 8. Deploy for real
modal deploy deploy/modal_app.py
```
