---
layout: page
title: Concepts
permalink: /concepts/
---

# Orchestration Concepts

Understand the theory behind AI agent orchestration.

---

## Core Concepts

### [The Loop Pattern](./loop-pattern.md)
The foundation of all orchestration: read, execute, verify, repeat.

### [Verification Strategies](./verification.md)
External vs internal verification, confidence scoring, hybrid approaches.

### [Circuit Breakers](./circuit-breakers.md)
Safety mechanisms to prevent infinite loops and runaway costs.

### [State Management](./state-management.md)
Persisting state for reliability and debugging.

---

## Advanced Patterns

### [Self-Play & Oscillation](./self-play.md)
Internal debate patterns for quality assurance (DIALECTIC, proposer/critic).

### [Gating Mechanisms](./gating.md)
Pre and post-condition enforcement with rollback.

### [Drift Prevention](./drift-prevention.md)
Keeping agents aligned with original intent over multiple iterations.

### [Multi-Agent Coordination](./multi-agent.md)
Pipelines, swarms, and role-based teams.

---

## Framework Comparison

### [Provider Overview](./providers.md)
Compare LangChain, CrewAI, AutoGen, Semantic Kernel, Haystack, DSPy.

### [SPINE Integration](./spine.md)
Context engineering and MCP-native orchestration.

### [Minna Memory](./minna-memory.md)
Cross-session persistence for AI agents.

---

## The 8me Philosophy

### When to Use What

| Complexity | Solution |
|------------|----------|
| Simple tasks | Basic loop (Tier 0-1) |
| Structured output | Tool calling (Tier 1) |
| Claude Code integration | Skills (Tier 2) |
| External state/tools | MCP Server (Tier 3) |
| Complex orchestration | Frameworks or SPINE |

### Build vs Buy

Before reaching for a framework:

1. **Do you need it?** Simple loops solve many problems
2. **Complexity cost**: Frameworks add abstraction overhead
3. **Debugging difficulty**: More layers = harder debugging
4. **Lock-in risk**: Framework APIs change frequently

---

## Reading Order

If you're new to orchestration:

1. Start with [The Loop Pattern](./loop-pattern.md)
2. Learn [Verification Strategies](./verification.md)
3. Understand [Circuit Breakers](./circuit-breakers.md)
4. Then explore advanced patterns as needed

---

## Full Conceptual Documentation

Detailed pseudo-code and explanations live in the 8me repository:

**[src/tier3.5-orchestration-concepts/](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)**

- `01-fundamentals.md` - Core patterns with pseudo-code
- `02-patterns.md` - Advanced patterns (self-play, gating, drift)
- `03-providers.md` - Framework comparison
- `04-spine-integration.md` - SPINE concepts
- `05-minna-memory.md` - Memory showcase
