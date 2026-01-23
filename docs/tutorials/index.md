---
layout: default
title: Tutorials
nav_order: 4
has_children: true
permalink: /tutorials/
---

# Tutorials

Quick-start guides to get you up and running.

---

## Getting Started

### [Your First Loop](../labs/01-first-loop)
15-minute introduction to AI loops. Build a working loop from scratch.

### [Understanding Verification](./verification-basics)
Why verification matters and how to implement it.

### [Setting Up MCP](./mcp-setup)
Configure MCP servers for Claude Desktop.

---

## By Tier

### Tier 0: Hello World
- [Run the Hello World example](./tier0-quickstart)
- [Understand the code](./tier0-explained)

### Tier 1: Full Loop
- [CLI usage guide](./tier1-cli)
- [Task management](./tier1-tasks)
- [Circuit breakers](./tier1-safety)

### Tier 2: Claude Code Skill
- [Install the skill](./tier2-install)
- [Using /ralph commands](./tier2-commands)

### Tier 3: MCP Server
- [Install the server](./tier3-install)
- [Configure in Claude Desktop](./tier3-config)
- [Available resources & tools](./tier3-api)

---

## Quick Reference

### Installation

```bash
# Clone the repository
git clone https://github.com/fbratten/8me.git
cd 8me

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key"
```

### Running Each Tier

```bash
# Tier 0
cd src/tier0-hello-world
python ralph_hello_world.py

# Tier 1
cd src/tier1-ralph-loop
python ralph_loop.py --help

# Tier 2 (Claude Code)
/ralph add "Your task here"
/ralph

# Tier 3 (MCP Server)
cd src/tier3-mcp-server
pip install -e .
ralph-mcp
```

---

<a href="../labs/01-first-loop" class="button primary">Start with Your First Loop →</a>
