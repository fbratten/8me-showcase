# 8me Showcase

> Educational platform for learning AI agent orchestration

**Live Site:** [https://fbratten.github.io/8me-showcase](https://fbratten.github.io/8me-showcase)
**Code Repository:** [https://github.com/fbratten/8me](https://github.com/fbratten/8me)

## Overview

8me teaches AI agent orchestration from fundamentals to production systems. Starting with simple loops and progressing to sophisticated multi-agent patterns.

## Architecture

This repository contains **educational content** with links to the **code repository**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Bidirectional Linking                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   8me-showcase (this repo)      8me (code repo)            │
│   ┌─────────────────────┐      ┌─────────────────────┐     │
│   │ Tutorials           │◄────►│ src/tier0-hello...  │     │
│   │ Labs                │      │ src/tier1-ralph...  │     │
│   │ Concept docs        │      │ src/tier2-ralph...  │     │
│   │ Architecture guides │      │ src/tier3-mcp-...   │     │
│   └─────────────────────┘      └─────────────────────┘     │
│           │                              │                  │
│           │   "Get the code" ──────────► │                  │
│           │ ◄──────── "Learn more"       │                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Tier Structure

| Tier | Name | Type | Description |
|------|------|------|-------------|
| 0 | Hello World | Code | Simplest loop (~90 lines) |
| 1 | Ralph Loop | Code | Full CLI with verification |
| 2 | Ralph Skill | Code | Claude Code skill |
| 3 | MCP Server | Code | FastMCP server |
| **3.5** | **Orchestration** | **Concepts** | Patterns & providers |
| 4 | Orchestrator | Code | Optional client |
| **5** | **Documentation** | **This site** | Tutorials & guides |
| **5.5** | **Labs** | **Hands-on** | 15 progressive labs |

## Quick Start

1. **Brand new to loops?** → Start with [Your First Loop](./labs/01-first-loop.md)
2. **Want working code?** → Visit [8me repository](https://github.com/fbratten/8me)
3. **Learning patterns?** → Read [Orchestration Concepts](./concepts/)
4. **Hands-on practice?** → Try [Lab 01](./labs/01-first-loop.md)

## Labs Curriculum

### Beginner (Labs 01-03)
- Lab 01: Your First Loop
- Lab 02: External State
- Lab 03: Simple Verification

### Intermediate (Labs 04-07)
- Lab 04: JSON Task Management
- Lab 05: Tool Calling Patterns
- Lab 06: Circuit Breakers
- Lab 07: Confidence-Based Retry

### Advanced (Labs 08-12)
- Lab 08: Building an Orchestrator
- Lab 09: Multi-Agent Patterns
- Lab 10: Gating and Drift Prevention
- Lab 11: Self-Play Oscillation
- Lab 12: Memory Integration

### Integration (Labs 13-15)
- Lab 13: MCP Server Deployment
- Lab 14: Claude Code Skill Development
- Lab 15: End-to-End Workflow

## Philosophy

> "Loop persistently until the job is done."

The "Ralph Wiggum" methodology teaches that success comes from:
- **Persistence**: Keep trying until verified complete
- **External verification**: Don't trust AI self-assessment alone
- **Safety limits**: Circuit breakers prevent infinite loops

*For the AI nerds:* This loop-until-done pattern was the common approach before sophisticated orchestration frameworks. Sometimes the old ways persist because they work.

> **Note:** Labs contain *pedagogical variations* with richer abstractions (interfaces, hooks, builders) optimized for teaching concepts layer-by-layer. The [8me source repo](https://github.com/fbratten/8me) contains the "mundane" working implementations optimized for actual use.

## Local Development

```bash
# Clone this showcase repo
git clone https://github.com/fbratten/8me-showcase.git
cd 8me-showcase

# Serve locally (requires Jekyll)
bundle install
bundle exec jekyll serve

# View at http://localhost:4000
```

## Contributing

This is a personal educational project. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

MIT License - See [LICENSE](./LICENSE) file for details.

---

*Part of the [8me](https://github.com/fbratten/8me) autonomous loop toolkit & Adaptivearts.ai™ initiative*
