---
layout: default
title: SPINE Integration
nav_order: 10
parent: Concepts
permalink: /concepts/spine/
---

# SPINE Integration

Context engineering and MCP-native orchestration.

---

## Overview

SPINE (Software Project Intelligence & Navigation Engine) focuses on **context engineering**, **self-play validation**, and **intelligent multi-provider routing**.

---

## Key Differentiators

| Feature | What It Does |
|---------|--------------|
| **Multi-Provider Routing** | Routes tasks to optimal LLM |
| **6-Layer Context Stacks** | Constitution-based prompts |
| **DIALECTIC Self-Play** | Thesis/antithesis/synthesis |
| **Atomic Instrumentation** | ToolEnvelope wraps every call |
| **Oscillation Detection** | 3-pattern stuck detection |
| **Conflict Resolution** | 6-type contradiction handling |

---

## Context Stacks

SPINE manages context as layered stacks:

```mermaid
graph TB
    L4["L4: History Context<br/>(recent actions, state)"]:::accent
    L3["L3: Task Context<br/>(current goal, constraints)"]:::tertiary
    L2["L2: Project Context<br/>(CLAUDE.md, structure)"]:::secondary
    L1["L1: System Context<br/>(base personality/rules)"]:::primary

    L4 --> L3 --> L2 --> L1

    classDef primary fill:#2563eb,color:#fff
    classDef secondary fill:#7c3aed,color:#fff
    classDef tertiary fill:#0d9488,color:#fff
    classDef accent fill:#f59e0b,color:#000
```

---

## DIALECTIC Methodology

```mermaid
graph LR
    T["THESIS<br/>Propose"]:::primary --> A["ANTITHESIS<br/>Critique"]:::secondary --> S["SYNTHESIS<br/>Merge"]:::tertiary

    classDef primary fill:#2563eb,color:#fff
    classDef secondary fill:#7c3aed,color:#fff
    classDef tertiary fill:#0d9488,color:#fff
```

Three phases per round:
1. **THESIS**: Generate proposal
2. **ANTITHESIS**: Challenge proposal
3. **SYNTHESIS**: Resolve conflicts

---

## Multi-Provider Routing

Route tasks to optimal LLM provider:

| Task Type | Provider |
|-----------|----------|
| PLANNING | Anthropic Claude |
| EXECUTION | OpenAI GPT |
| MULTIMODAL | Google Gemini |
| DISCOVERY | xAI Grok |

---

## Tiered Enforcement

| Tier | Description | Requirements |
|------|-------------|--------------|
| **1** | Simple, single-file | Direct execution OK |
| **2** | Multi-file, features | SHOULD use subagents |
| **3** | Architecture, research | MUST use subagents + MCP |

---

## Integration with 8me

| 8me Tier | SPINE Usage |
|----------|-------------|
| Tier 0 | None (pure learning) |
| Tier 1 | Circuit breaker patterns |
| Tier 2 | Skill integration, context loading |
| Tier 3 | MCP resources and tools |
| Tier 3.5 | DIALECTIC, oscillation detection |
| Tier 4 | Full Adaptive MCP Orchestrator |

---

## Next Steps

- Learn about [Minna Memory](../minna-memory/)
- See full documentation in [8me repository](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)

---

<div style="text-align: center;">
  <a href="../">← Back to Concepts</a>
</div>
