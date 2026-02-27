# Add Agent/Model Identification + Cost Savings Estimation

## Context

Winnow reports token savings using Qwen3 tokenizer counts, but customers use Claude, GPT, Gemini, etc. Each has a different tokenizer and different pricing. To show customers actual dollar savings, we need to:
1. Know which model they're using (sent from the MCP client)
2. Estimate tokens in that model's economy (character-ratio approximation)
3. Return estimated savings in the response

## Approach: Character-ratio estimation

Installing every tokenizer (tiktoken, SentencePiece, etc.) would bloat the GPU image and add fragile deps. Instead, use a chars-per-token ratio per model family. For code, all major BPE tokenizers land in the 3.0-3.5 range. Good enough for cost estimation (within ~10%), not billing.

## Changes

### 1. `swe-pruner/src/swe_pruner/prune_wrapper.py` - Add models, wire savings

Add `ClientInfo` and `SavingsEstimate` Pydantic models. Add optional `client_info` field to `PruneRequest`, optional `savings` field to `PruneResponse`.

```python
class ClientInfo(BaseModel):
    agent: str = "unknown"          # "claude-code", "codex", "cursor", etc.
    model: str = ""                 # "claude-sonnet-4-20250514", "gpt-4o", etc.
    mcp_server_version: str = ""

class SavingsEstimate(BaseModel):
    target_model: str
    estimated_original_tokens: int
    estimated_pruned_tokens: int
    estimated_tokens_saved: int
    estimated_cost_saved_usd: float
    input_price_per_mtok: float
    chars_per_token: float
```

Wire `estimate_savings()` call before both `return PruneResponse(...)` sites (lines 553 and 631). Pass `savings=None` on the early-return path (line 553), compute it on the main path (line 631).

### 2. `swe-pruner/src/swe_pruner/pricing.py` - New file, model pricing registry

Pure-data module. Dict mapping model ID prefixes to `(price_per_mtok, chars_per_token)`. One function: `estimate_savings(original_code, pruned_code, model_id) -> SavingsEstimate | None`.

Pricing table (subset):

| Model prefix | $/MTok input | chars/token |
|---|---|---|
| claude-opus-4 | 15.00 | 3.2 |
| claude-sonnet-4 | 3.00 | 3.2 |
| claude-haiku-4.5 | 1.00 | 3.2 |
| gpt-4o | 2.50 | 3.3 |
| gpt-4.1 | 2.00 | 3.3 |
| gpt-4.1-mini | 0.40 | 3.3 |
| gemini-2.5-pro | 1.25 | 3.4 |
| gemini-2.5-flash | 0.30 | 3.4 |
| unknown (fallback) | 3.00 | 3.3 |

Resolution: exact match first, then longest prefix match (handles date suffixes like `-20250514`).

### 3. `winnow-mcp/src/index.ts` - Send client metadata

**Agent auto-detection** from env vars:
- `CLAUDECODE` set -> `"claude-code"`
- `CODEX_HOME` / `CODEX_SANDBOX_ID` -> `"codex"`
- `CURSOR_SESSION_ID` -> `"cursor"`
- `WINNOW_AGENT` override -> whatever user sets
- Fallback: `"unknown"`

**Model**: must be user-configured via `WINNOW_CLIENT_MODEL` env var. The MCP server cannot introspect which model the host agent uses.

**callPruner** body changes from `{ code, query, threshold }` to `{ code, query, threshold, client_info: { agent, model, mcp_server_version } }`.

**PrunerResponse** interface gets optional `savings` field. Surface it in `prune_code` tool output as a footer line.

### 4. `deploy/modal_app.py` - Optional logging

No structural changes needed (FastAPI auto-deserializes the updated Pydantic models). Add one `print()` line after inference to log agent/model/savings to Modal logs for analytics.

## File list

| File | Action |
|---|---|
| `swe-pruner/src/swe_pruner/prune_wrapper.py` | Add `ClientInfo`, `SavingsEstimate`, wire into `PruneRequest`/`PruneResponse`, compute savings at line 631 |
| `swe-pruner/src/swe_pruner/pricing.py` | **New file**. Model pricing registry + `estimate_savings()` |
| `winnow-mcp/src/index.ts` | Add `detectAgent()`, `WINNOW_CLIENT_MODEL` env var, `buildClientInfo()`, update `callPruner` body and response types |
| `deploy/modal_app.py` | Add analytics `print()` after prune call |

## Backward compatibility

- `PruneRequest.client_info` defaults to `None`. Old clients omit it, server returns `savings: null`.
- `PruneResponse.savings` defaults to `None`. Old clients ignore it.
- No changes to `/health` or auth flow.

## Verification

```bash
# 1. Build swe-pruner core (no tests exist, just import check)
cd swe-pruner && python -c "from swe_pruner.pricing import estimate_savings; print('ok')"

# 2. Build winnow-mcp
cd winnow-mcp && npm run build

# 3. Redeploy Modal
modal deploy deploy/modal_app.py

# 4. Test with client_info
curl -s -X POST https://winnow.nadicode.com/prune \
  -H "Authorization: Bearer sk_winnow_test" \
  -H "Content-Type: application/json" \
  -d '{"code":"def hello():\n    print(\"hello\")\n    return 42\n","query":"return value","threshold":0.5,"client_info":{"agent":"claude-code","model":"claude-sonnet-4-20250514","mcp_server_version":"0.2.0"}}' \
  | python -m json.tool
# Should include "savings" field with estimated_tokens_saved and estimated_cost_saved_usd

# 5. Test without client_info (backward compat)
curl -s -X POST https://winnow.nadicode.com/prune \
  -H "Authorization: Bearer sk_winnow_test" \
  -H "Content-Type: application/json" \
  -d '{"code":"x=1","query":"test","threshold":0.5}' \
  | python -m json.tool
# Should have "savings": null
```
