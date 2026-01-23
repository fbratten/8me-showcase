---
layout: page
title: Setting Up MCP
permalink: /tutorials/mcp-setup/
parent: Tutorials
---

# Setting Up MCP

Configure MCP servers for Claude Desktop.

---

## What is MCP?

MCP (Model Context Protocol) lets Claude access external tools and resources. 8me's Tier 3 is an MCP server.

---

## Claude Desktop Config

Find your config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

---

## Adding ralph-workflow

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "node",
      "args": ["path/to/8me/src/tier3-mcp-server-ts/dist/index.js"]
    }
  }
}
```

---

## Verify Installation

1. Restart Claude Desktop
2. Look for "ralph-workflow" in the MCP servers list
3. Try: "Show me the task queue" - Claude should access `task://queue`

---

## Available Resources

| Resource | Description |
|----------|-------------|
| `task://queue` | All tasks |
| `task://current` | Current in-progress task |
| `task://stats` | Queue statistics |

---

## Available Tools

| Tool | Description |
|------|-------------|
| `add_task` | Add new task |
| `start_task` | Begin working on task |
| `complete_task` | Mark task done |
| `fail_task` | Mark task failed |

---

## Next Steps

- [Tier 3 Install Guide](../tier3-install/)
- [Lab 13: MCP Server](../../labs/13-mcp-server)

---

<div style="text-align: center;">
  <a href="../">← Back to Tutorials</a>
</div>
