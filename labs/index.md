---
layout: page
title: Labs Curriculum
permalink: /labs/
---

# Labs Curriculum

Hands-on labs for learning AI agent orchestration from scratch.

## Learning Path

```
BEGINNER (01-03) → INTERMEDIATE (04-07) → ADVANCED (08-12) → INTEGRATION (13-15)
```

---

## Beginner Labs

Master the fundamentals.

### [Lab 01: Your First Loop](./01-first-loop.md)
Build a basic loop that reads tasks, calls Claude, and saves results.
- **Time:** 30 minutes
- **Prerequisites:** Python basics, API key

### [Lab 02: External State](./02-external-state.md)
Track progress between runs with external state management.
- **Time:** 45 minutes
- **Prerequisites:** Lab 01

### [Lab 03: Simple Verification](./03-verification.md)
Verify AI outputs before accepting them.
- **Time:** 45 minutes
- **Prerequisites:** Lab 02

---

## Intermediate Labs

Build production-quality loops.

### [Lab 04: JSON Task Management](./04-json-tasks.md)
Structure tasks with JSON and metadata.
- **Time:** 1 hour
- **Prerequisites:** Labs 01-03

### [Lab 05: Tool Calling](./05-tool-calling.md)
Use Claude's tool calling for structured output.
- **Time:** 1 hour
- **Prerequisites:** Lab 04

### [Lab 06: Circuit Breakers](./06-circuit-breakers.md)
Prevent infinite loops with safety limits.
- **Time:** 45 minutes
- **Prerequisites:** Lab 05

### [Lab 07: Confidence Retry](./07-confidence-retry.md)
Use AI confidence for smart retry decisions.
- **Time:** 1 hour
- **Prerequisites:** Lab 06

---

## Advanced Labs

Master sophisticated patterns.

### [Lab 08: Orchestrator](./08-orchestrator.md)
Build a complete orchestrator class.
- **Time:** 2 hours
- **Prerequisites:** Labs 01-07

### [Lab 09: Multi-Agent](./09-multi-agent.md)
Coordinate multiple specialized agents.
- **Time:** 2 hours
- **Prerequisites:** Lab 08

### [Lab 10: Gating](./10-gating.md)
Implement pre/post-condition enforcement.
- **Time:** 1.5 hours
- **Prerequisites:** Lab 09

### [Lab 11: Self-Play](./11-self-play.md)
Internal debate for quality assurance.
- **Time:** 2 hours
- **Prerequisites:** Lab 10

### [Lab 12: Memory](./12-memory.md)
Integrate Minna Memory for persistence.
- **Time:** 1.5 hours
- **Prerequisites:** Lab 11

---

## Integration Labs

Connect everything together.

### [Lab 13: MCP Server](./13-mcp-server.md)
Deploy your orchestrator as an MCP server.
- **Time:** 2 hours
- **Prerequisites:** Labs 01-12

### [Lab 14: Claude Code Skill](./14-skill.md)
Create a custom Claude Code skill.
- **Time:** 1.5 hours
- **Prerequisites:** Lab 13

### [Lab 15: End-to-End](./15-end-to-end.md)
Build a complete production system.
- **Time:** 3 hours
- **Prerequisites:** All previous labs

---

## Get the Code

All lab code is available in the [8me repository](https://github.com/fbratten/8me):

- **Tier 0:** `src/tier0-hello-world/` - Basic loop
- **Tier 1:** `src/tier1-ralph-loop/` - Full CLI
- **Tier 2:** `src/tier2-ralph-skill/` - Claude Code skill
- **Tier 3:** `src/tier3-mcp-server/` - MCP server

---

<a href="./01-first-loop" class="button primary">Start Lab 01 →</a>
