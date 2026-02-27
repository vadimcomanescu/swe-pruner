"""
Supabase/Postgres backend for API key validation and usage logging.

Uses asyncpg for direct Postgres connections (through Supabase PgBouncer pooler).
Key cache refreshes every 60s. Usage events are buffered and flushed in bulk
every 10s or 100 rows to minimize DB round-trips under high agent traffic.
"""

import asyncio
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from urllib.parse import unquote

import asyncpg

log = logging.getLogger("winnow.db")

# ---------------------------------------------------------------------------
# Key cache
# ---------------------------------------------------------------------------

@dataclass
class KeyCache:
    """Thread-safe in-memory cache of active API key hashes -> tiers."""

    _keys: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def replace(self, keys: dict[str, str]) -> None:
        with self._lock:
            self._keys = keys

    def get(self, key_hash: str) -> str | None:
        with self._lock:
            return self._keys.get(key_hash)

    def add(self, key_hash: str, tier: str) -> None:
        with self._lock:
            self._keys[key_hash] = tier

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


# ---------------------------------------------------------------------------
# Usage buffer
# ---------------------------------------------------------------------------

@dataclass
class UsageBuffer:
    """Accumulates usage events in memory for periodic bulk INSERT."""

    _buffer: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, event: dict) -> int:
        with self._lock:
            self._buffer.append(event)
            return len(self._buffer)

    def flush(self) -> list[dict]:
        """Atomically swap the buffer and return the old contents."""
        with self._lock:
            old = self._buffer
            self._buffer = []
            return old


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_conn: asyncpg.Connection | None = None
_cache = KeyCache()
_usage = UsageBuffer()
_tasks: list[asyncio.Task] = []

FLUSH_INTERVAL_S = 10
FLUSH_BATCH_SIZE = 100
KEY_REFRESH_INTERVAL_S = 60


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _parse_dsn(dsn: str) -> dict:
    """Parse DSN into asyncpg keyword args, handling passwords with special chars."""
    # Strip scheme
    rest = dsn.split("://", 1)[1] if "://" in dsn else dsn
    # Split on last @ to separate userinfo from host (password may contain @)
    at_idx = rest.rfind("@")
    userinfo = rest[:at_idx]
    hostpart = rest[at_idx + 1:]
    # User:password (URL-decode both in case Supabase percent-encoded the password)
    user, password = userinfo.split(":", 1)
    user = unquote(user)
    password = unquote(password)
    # Host:port/database
    host_and_db = hostpart.split("/", 1)
    host_port = host_and_db[0]
    database = host_and_db[1] if len(host_and_db) > 1 else "postgres"
    host, port_str = host_port.rsplit(":", 1)
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": int(port_str),
        "database": database,
    }


async def init(dsn: str) -> None:
    global _conn
    params = _parse_dsn(dsn)
    import ssl as _ssl
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    log.warning("Connecting: user=%s host=%s port=%s db=%s", params["user"], params["host"], params["port"], params["database"])
    _conn = await asyncpg.connect(**params, ssl=ctx)
    log.info("Connected to Supabase Postgres")


async def load_keys() -> int:
    """Fetch all active keys into the cache. Returns count loaded."""
    assert _conn is not None
    rows = await _conn.fetch(
        "SELECT key_hash, tier FROM winnow.api_keys WHERE revoked_at IS NULL"
    )
    keys = {row["key_hash"]: row["tier"] for row in rows}
    _cache.replace(keys)
    log.info("Loaded %d active API keys", len(keys))
    return len(keys)


def start_background_tasks() -> None:
    loop = asyncio.get_event_loop()
    _tasks.append(loop.create_task(_key_refresh_loop()))
    _tasks.append(loop.create_task(_usage_flush_loop()))


async def shutdown() -> None:
    for t in _tasks:
        t.cancel()
    for t in _tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    _tasks.clear()
    await _flush_usage()
    if _conn is not None:
        await _conn.close()
    log.info("DB shutdown complete")


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def validate_key(raw_key: str) -> tuple[str, str] | None:
    """
    Hash the raw key, check cache, fallback to single-row query.
    Returns (key_hash, tier) or None if invalid.
    """
    key_hash = hash_key(raw_key)

    tier = _cache.get(key_hash)
    if tier is not None:
        return key_hash, tier

    # Cache miss: key might have been created between refreshes
    assert _conn is not None
    row = await _conn.fetchrow(
        "SELECT tier FROM winnow.api_keys WHERE key_hash = $1 AND revoked_at IS NULL",
        key_hash,
    )
    if row is not None:
        _cache.add(key_hash, row["tier"])
        return key_hash, row["tier"]

    return None


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------

def record_usage(
    *,
    key_hash: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    agent: str = "unknown",
    model: str = "",
    threshold: float = 0.5,
    score: float | None = None,
    error_msg: str | None = None,
) -> None:
    """Append a usage event to the buffer (non-blocking, no await)."""
    _usage.append({
        "key_hash": key_hash,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "agent": agent,
        "model": model,
        "threshold": threshold,
        "score": score,
        "error_msg": error_msg,
    })


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------

async def _key_refresh_loop() -> None:
    while True:
        await asyncio.sleep(KEY_REFRESH_INTERVAL_S)
        try:
            await load_keys()
        except Exception:
            log.exception("Key refresh failed")


async def _usage_flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_S)
        try:
            await _flush_usage()
        except Exception:
            log.exception("Usage flush failed")


async def _flush_usage() -> None:
    batch = _usage.flush()
    if not batch:
        return
    assert _conn is not None
    try:
        await _conn.executemany(
            """
            INSERT INTO winnow.usage_log
                (key_hash, tokens_in, tokens_out, latency_ms, agent, model, threshold, score, error_msg)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            [
                (
                    e["key_hash"],
                    e["tokens_in"],
                    e["tokens_out"],
                    e["latency_ms"],
                    e["agent"],
                    e["model"],
                    e["threshold"],
                    e["score"],
                    e["error_msg"],
                )
                for e in batch
            ],
        )
        log.info("Flushed %d usage events", len(batch))
    except Exception:
        log.exception("Failed to flush %d usage events (discarded)", len(batch))
