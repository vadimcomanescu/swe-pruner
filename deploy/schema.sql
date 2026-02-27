-- Winnow schema: API keys + usage logging for the pruner service.
-- Run this in the Supabase SQL editor. Existing tables in `public` are untouched.

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
