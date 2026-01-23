---
layout: page
title: Tier 3 Install
permalink: /tutorials/tier3-install/
parent: Tutorials
---

# Tier 3: Installing the MCP Server

Deploy the Ralph MCP server.

---

## TypeScript Version (Recommended)

### Build

```bash
cd src/tier3-mcp-server-ts
npm install
npm run build
```

### Test

```bash
node dist/index.js
# Server starts, outputs MCP protocol messages
```

---

## Python Version

### Install

```bash
cd src/tier3-mcp-server
pip install -e .
```

### Run

```bash
ralph-mcp
# Or: python -m ralph_mcp_server
```

---

## Add to Claude Desktop

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "node",
      "args": ["/path/to/8me/src/tier3-mcp-server-ts/dist/index.js"]
    }
  }
}
```

Restart Claude Desktop.

---

## Verify

In Claude Desktop, ask:
> "Show me the current task queue"

Claude should access `task://queue` resource.

---

## Next Steps

- [Configure the Server](./tier3-config)
- [API Reference](./tier3-api)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
