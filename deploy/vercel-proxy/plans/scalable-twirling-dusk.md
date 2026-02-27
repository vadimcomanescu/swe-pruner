# Create `winnow-mcp` TypeScript MCP Server

## Context

Winnow (SWE-Pruner) needs a distributable MCP server so customers can plug neural code pruning into Claude Code, Codex, OpenCode, or any MCP-compatible tool. The existing Python MCP server at `/home/vadim/Code/swe-pruner-mcp/` works but requires `uvx` and Python 3.13. Rewriting as a TypeScript npm package gives us `npx winnow-mcp` which works everywhere Node is installed.

The MCP server is a thin HTTP client. It reads files / runs grep locally, then optionally sends code through `https://winnow.nadicode.com/prune` for neural pruning. No GPU, no model, just HTTP.

## Deliverables

1. New repo at `/home/vadim/Code/winnow-mcp/`
2. Pushed to GitHub as a private repo
3. Published to npm as `winnow-mcp`

## Project Structure

```
winnow-mcp/
  src/index.ts       # Single file, ~300 lines. All logic.
  package.json
  tsconfig.json
  .gitignore
  AGENTS.md
  README.md
```

## Implementation

### Step 1: Scaffold

Create `/home/vadim/Code/winnow-mcp/` with:

**package.json**
- `"name": "winnow-mcp"`, `"type": "module"`, `"bin": { "winnow-mcp": "./dist/index.js" }`
- `"files": ["dist"]` (only publish compiled output)
- Dependencies: `@modelcontextprotocol/sdk`, `zod`
- Dev: `typescript`, `@types/node`
- Scripts: `"build": "tsc"`, `"prepublishOnly": "npm run build"`
- `"engines": { "node": ">=18" }`

**tsconfig.json** - ES2022, Node16 module/resolution, strict, outDir `dist`

**.gitignore** - `node_modules/`, `dist/`

### Step 2: Write `src/index.ts`

Port from `/home/vadim/Code/swe-pruner-mcp/server.py`. Single file, top to bottom:

1. **Shebang + imports**: `#!/usr/bin/env node`, MCP SDK, zod, node:fs, node:path, node:child_process
2. **Config from env vars**:
   - `WINNOW_API_KEY` (required, warn if missing)
   - `WINNOW_API_URL` (default `https://winnow.nadicode.com`)
   - `WINNOW_MAX_LINES` (default 200), `WINNOW_MAX_CHARS` (default 20000)
   - `WINNOW_DEFAULT_THRESHOLD` (default 0.5), `WINNOW_ENFORCE_FOCUS` (default true)
3. **`CODE_EXTENSIONS` Set** - port the 40+ extensions from Python
4. **Helpers** (direct ports from `server.py`):
   - `isCodeFile(path)` - extension check
   - `safePath(path)` - canonicalize, check root boundary
   - `enforceLargeOutput(lines, chars, focus, forceFull, desc)` - throw if too large
   - `callPruner(code, query, threshold)` - `fetch()` POST to API with Bearer auth, 60s timeout, fallback to original on error
   - `maybePrune(text, focus, threshold, path)` - skip non-code, skip empty focus, call pruner
   - `rgAvailable()` / `runRg(pattern, path, recursive, regex)` - ripgrep via `execSync`
5. **3 tools** registered on `McpServer`:
   - `read_file(path, context_focus_question?, prune_threshold?, force_full?)` - readFileSync + enforceLargeOutput + maybePrune
   - `grep(pattern, path?, context_focus_question?, prune_threshold?, recursive?, regex?, force_full?)` - rg or manual walk + enforceLargeOutput + maybePrune
   - `prune_code(code, query, threshold?)` - direct callPruner, return JSON
6. **Server startup** - stdio transport, connect

**Key differences from Python:**
- HTTP auth: `Authorization: Bearer` header (not `client_id` in body)
- Default API URL: `https://winnow.nadicode.com` (not localhost)
- No `get_stats` tool, no `client_id` tracking
- Uses Node built-in `fetch` (no httpx equivalent needed)

### Step 3: Build and smoke test

```bash
cd /home/vadim/Code/winnow-mcp
npm install && npm run build
# Verify dist/index.js exists and has shebang
# Quick test: WINNOW_API_KEY=sk_winnow_test node dist/index.js (should start, wait for stdio)
```

### Step 4: Write README.md

- What it is (one paragraph)
- Setup per tool:
  ```bash
  # Claude Code
  claude mcp add winnow -e WINNOW_API_KEY=sk_winnow_... -- npx winnow-mcp

  # Any MCP client (manual config)
  {"mcpServers": {"winnow": {"command": "npx", "args": ["-y", "winnow-mcp"], "env": {"WINNOW_API_KEY": "..."}}}}
  ```
- Environment variables table
- Tools table (3 tools, params, descriptions)

### Step 5: Write AGENTS.md

Single-file TypeScript MCP server. Build with `npm run build`. Port of the Python server at `swe-pruner-mcp/server.py`.

### Step 6: Git init + push to GitHub private repo

```bash
cd /home/vadim/Code/winnow-mcp
git init && git add -A && git commit -m "Initial commit"
gh repo create winnow-mcp --private --source . --push
```

### Step 7: Publish to npm

```bash
npm publish
```

Note: requires npm login (`npm whoami` to check). If not logged in, run `npm login` first.

## What's NOT included

- No test suite (add later)
- No `get_stats` tool (server-side concern)
- No client_id tracking
- No retry logic (fallback-to-original handles failures)
- No connection pooling (single fetch calls are fine)
- No Windows-specific handling

## Verification

```bash
# 1. Build succeeds
npm run build

# 2. Binary runs
WINNOW_API_KEY=sk_winnow_test npx .
# Should start MCP server on stdio (no crash)

# 3. Add to Claude Code and test
claude mcp add winnow -e WINNOW_API_KEY=sk_winnow_test -- npx /home/vadim/Code/winnow-mcp
# Then in a Claude Code session, use read_file on a .py file with context_focus_question
```
