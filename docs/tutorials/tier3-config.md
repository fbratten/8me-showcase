---
layout: page
title: Tier 3 Config
permalink: /tutorials/tier3-config/
parent: Tutorials
---

# Tier 3: Server Configuration

Configure the Ralph MCP server.

---

## Claude Desktop Config

Location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

---

## Basic Config

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "node",
      "args": ["/path/to/dist/index.js"]
    }
  }
}
```

---

## With Environment Variables

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": {
        "RALPH_TASK_FILE": "/path/to/tasks.json",
        "RALPH_MAX_ATTEMPTS": "5"
      }
    }
  }
}
```

---

## Multiple MCP Servers

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "node",
      "args": ["/path/to/ralph/dist/index.js"]
    },
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["/allowed/path"]
    }
  }
}
```

---

## Troubleshooting

### Server Not Showing

1. Check config JSON syntax
2. Verify file paths exist
3. Restart Claude Desktop

### Permission Errors

Ensure the command is executable:
```bash
chmod +x /path/to/dist/index.js
```

---

## Next Steps

- [API Reference](./tier3-api)
- [Lab 13: MCP Server](../../labs/13-mcp-server)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
