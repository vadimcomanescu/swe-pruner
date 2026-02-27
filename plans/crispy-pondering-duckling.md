# Move winnow MCP config to project-local .mcp.json

## Context

The winnow MCP server (the npm `winnow-mcp` package) is configured in `~/.claude-adriana/.claude.json` under `projects[/home/vadim/Code/swe-pruner].mcpServers.winnow`. This is a project-scoped entry buried in a global config file. It should live in `.mcp.json` at the project root per standard practice.

Current config:
```json
{
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "winnow-mcp"],
  "env": {
    "WINNOW_API_KEY": "sk_winnow_test123"
  }
}
```

## Changes

### 1. Create `/home/vadim/Code/swe-pruner/.mcp.json`

```json
{
  "mcpServers": {
    "winnow": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "winnow-mcp"],
      "env": {
        "WINNOW_API_KEY": "sk_winnow_test123"
      }
    }
  }
}
```

### 2. Remove winnow entry from `~/.claude-adriana/.claude.json`

Delete the `winnow` key from `projects["/home/vadim/Code/swe-pruner"].mcpServers`. If `mcpServers` becomes empty, remove it too.

## Verification

- Run `/mcp` in Claude Code to confirm winnow still shows up as a project server
- Verify winnow tools are accessible
