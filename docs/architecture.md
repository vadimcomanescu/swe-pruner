# Winnow Architecture

## What it is

Winnow is a hosted neural code pruning service. It sits between a coding agent and an LLM, reducing the token count of source code by 23-54% before the agent sends it to the model. It keeps only the lines relevant to what the agent is trying to do.

The pruning model is Qwen3-Reranker-0.6B with multi-layer fusion and a CRF compression head. It scores every token in a file against a query, aggregates scores to line level, and drops lines below a threshold while inserting `(filtered N lines)` placeholders so the LLM still understands code structure.

---

## System overview

```
┌─────────────────────────────────────────────────────┐
│  Coding agent (Claude Code, Cursor, Windsurf, etc.) │
│                                                     │
│  Reads a 600-line file                              │
│  → calls winnow MCP tool instead of built-in Read  │
└───────────────────┬─────────────────────────────────┘
                    │ MCP stdio (tool call)
                    ▼
┌─────────────────────────────────────────────────────┐
│  winnow-mcp  (TypeScript, npx winnow-mcp)           │
│                                                     │
│  - Reads file from disk                             │
│  - Enforces focus question on large files           │
│  - Sends code + query to Winnow API                 │
│  - Returns pruned code to agent                     │
└───────────────────┬─────────────────────────────────┘
                    │ HTTPS POST /prune
                    │ Authorization: Bearer sk_winnow_...
                    ▼
┌─────────────────────────────────────────────────────┐
│  Winnow API  (Modal, https://winnow.nadicode.com)   │
│                                                     │
│  - FastAPI endpoint                                 │
│  - API key auth + per-key rate limiting             │
│  - Runs inference on L4 GPU (scale-to-zero)         │
│  - Logs agent, model, latency, tokens saved         │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Qwen3-Reranker-0.6B (prune_wrapper.py)             │
│                                                     │
│  - Multi-layer fusion (early/mid/final hidden states│
│  - CRF compression head: per-token keep/drop logits │
│  - Aggregates token → line scores                   │
│  - Applies threshold, returns pruned code           │
└─────────────────────────────────────────────────────┘
```

The agent gets back 300 lines instead of 600. It sends those to the LLM. The LLM never sees the pruned lines.

---

## Repositories

| Repo | Language | Purpose |
|---|---|---|
| `swe-pruner` (this repo) | Python | Model, inference server, Modal deployment |
| `winnow-mcp` | TypeScript | MCP server published to npm (cloud API client) |
| `swe-pruner-mcp` | Python | MCP server for local/self-hosted use |

---

## Components

### 1. Inference server (`swe-pruner/`)

The GPU backend. Everything in `swe-pruner/src/swe_pruner/`:

| File | Role |
|---|---|
| `configuration.py` | `WinnowConfig`: model hyperparameters |
| `model_structure.py` | `TokenScorer`: backbone + fusion layers + CRF head |
| `winnow.py` | `WinnowForCodeCompression`: thin wrapper, returns `token_logits` + `score_logits` |
| `prune_wrapper.py` | `SwePrunerForCodePruning`: tokenize → chunk → infer → aggregate → prune. User-facing entry point. |
| `online_serving.py` | FastAPI app + Typer CLI. Wraps the pruner behind `/prune` and `/health`. |

`PruneRequest` fields:

```python
class PruneRequest(BaseModel):
    query: str               # what the agent is looking for
    code: str                # source code to prune
    threshold: float = 0.5  # 0.7 aggressive | 0.5 balanced | 0.3 conservative
    always_keep_first_frags: bool = False
    chunk_overlap_tokens: int = 50
    client_info: ClientInfo = ClientInfo()  # agent name + model from MCP client
```

`PruneResponse` fields:

```python
class PruneResponse(BaseModel):
    score: float              # document-level relevance score (0-1)
    pruned_code: str          # code with irrelevant lines replaced by placeholders
    kept_frags: List[int]     # line numbers kept
    origin_token_cnt: int     # token count before pruning
    left_token_cnt: int       # token count after pruning
    model_input_token_cnt: int
    error_msg: Optional[str]
```

### 2. Modal deployment (`deploy/modal_app.py`)

Wraps the inference server in a Modal app. On every request:
1. Verifies Bearer token against `WINNOW_API_KEYS` secret
2. Checks per-key rate limit (token bucket)
3. Runs inference in a `ThreadPoolExecutor` to avoid blocking the async loop
4. Logs `agent | model | latency | tokens_saved` to Modal logs

GPU: L4 (24GB VRAM). The model needs ~9GB VRAM due to multi-layer fusion keeping 3x hidden states. Scale-to-zero after 5 minutes idle. Cold start ~30s.

See `deploy/README.md` for ops: deploying, managing API keys, rate limit tiers.

### 3. Cloud MCP server (`winnow-mcp`)

TypeScript MCP server published to npm as `winnow-mcp`. Runs as a child process of the coding agent via stdio transport.

What it does:
- `read_file`: reads a file from disk, prunes it if a `context_focus_question` is set and the file is source code
- `grep`: runs ripgrep (or a built-in fallback), prunes the output
- `prune_code`: prunes code already in context

Agent identification: reads `clientInfo.name` from the MCP `initialize` handshake. Every MCP client sends this automatically (`"claude-code"`, `"cursor-vscode"`, `"Windsurf"`, `"Zed"`, etc.). No configuration needed.

Model identification: passed as a `model` parameter on each tool call (e.g. `"claude-sonnet-4-20250514"`). Dynamic because agents can switch models mid-session.

Both are sent to the API as `client_info` for analytics.

### 4. Local MCP server (`swe-pruner-mcp`)

Python MCP server for local use. Talks to a local pruner instance running on `http://127.0.0.1:17845` instead of the cloud API. Used when running the model locally on your own GPU.

Setup: the pruner backend runs as a systemd user service (`~/.config/systemd/user/swe-pruner.service`). See `swe-pruner-mcp/CLAUDE.md` for local setup.

---

## Request flow (detailed)

```
agent calls: read_file(path="/src/auth.py", context_focus_question="error handling")
    │
    ▼
winnow-mcp
    reads /src/auth.py from disk (600 lines, 18,400 chars)
    checks: is it a code file? yes
    checks: >200 lines and has focus question? yes → prune
    POST https://winnow.nadicode.com/prune
        body: {
            code: "...",
            query: "error handling",
            threshold: 0.5,
            client_info: { agent: "claude-code", model: "claude-sonnet-4-20250514", mcp_server_version: "0.2.0" }
        }
    │
    ▼
Modal (API key verified, rate limit checked)
    runs SwePrunerForCodePruning.prune(request)
        tokenizes code (1,847 tokens)
        fits in one chunk? yes
        runs Qwen3-Reranker forward pass
        gets per-token logits
        aggregates to line scores
        keeps lines with score >= 0.5
        builds pruned_code with "(filtered N lines)" placeholders
    logs: prune | agent=claude-code model=claude-sonnet-4-20250514 latency=2341ms
    returns PruneResponse
    │
    ▼
winnow-mcp
    returns pruned_code (312 lines, 9,100 chars) as tool result
    │
    ▼
agent → LLM
    sees 312 lines instead of 600
    sends 9,100 chars instead of 18,400 to the LLM
    ~50% token savings on this file
```

---

## Analytics

Each request logs to Modal's output stream:

```
prune | agent=claude-code model=claude-sonnet-4-20250514 latency=2341ms
```

Fields: agent name (from MCP clientInfo), model ID (from tool parameter), latency in ms. These flow into Modal's log infrastructure. A future dashboard will aggregate them per API key to show customers their usage and savings.

---

## What's not built yet

- **SaaS dashboard**: API key management, usage graphs, tokens saved per day, estimated cost savings. Pricing and cost calculations live there, not in the inference server.
- **Stripe integration**: trial/pro/team subscription billing.
- **Request batching**: currently single-threaded per container. Batching 4-8 requests per forward pass would cut GPU cost 3-4x.
- **Token cap**: should reject requests over 100K tokens before hitting the GPU.

See `docs/business_model.md` for pricing analysis and implementation roadmap.

---

## Running locally

### Inference server

```bash
cd swe-pruner
uv sync
# model weights via git-lfs or:
huggingface-cli download ayanami-kitasan/code-pruner --local-dir ./model
winnow --model-path ./model --port 17845
```

### Cloud MCP (against local server)

```bash
WINNOW_API_URL=http://127.0.0.1:17845 WINNOW_API_KEY=any npx winnow-mcp
```

### Production deploy

```bash
modal deploy deploy/modal_app.py
```
