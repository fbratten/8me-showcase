---
layout: default
title: Self-Play & Oscillation
nav_order: 5
parent: Concepts
permalink: /concepts/self-play/
---

# Self-Play & Oscillation

Internal debate patterns for quality assurance.

---

## Self-Play Pattern

Two "perspectives" debate until convergence:

```mermaid
graph TB
    PROPOSER["PROPOSER<br/>Build X"]:::primary
    CRITIC["CRITIC<br/>Check X"]:::secondary
    PROPOSAL["Proposal"]:::tertiary
    CRITIQUE["Critique"]:::tertiary
    COMMIT["COMMIT"]:::primary
    REFINE["REFINE"]:::accent

    PROPOSER <--> CRITIC
    PROPOSER --> PROPOSAL
    CRITIC --> CRITIQUE
    PROPOSAL --> CRITIQUE
    CRITIQUE -- APPROVED --> COMMIT
    CRITIQUE -- REJECTED --> REFINE
    REFINE --> PROPOSER

    classDef primary fill:#2563eb,color:#fff
    classDef secondary fill:#7c3aed,color:#fff
    classDef tertiary fill:#0d9488,color:#fff
    classDef accent fill:#f59e0b,color:#000
```

---

## Implementation

```python
def self_play_orchestration(task, max_rounds=5):
    """
    Internal debate between proposer and critic.
    Converges when critic approves or max rounds reached.
    """
    proposal = None
    history = []

    for round in range(max_rounds):
        # Phase 1: Proposer generates/refines
        if proposal is None:
            proposal = proposer_generate(task)
        else:
            proposal = proposer_refine(task, proposal, critique)

        # Phase 2: Critic evaluates
        critique = critic_evaluate(task, proposal)

        history.append({
            "round": round,
            "proposal": proposal,
            "critique": critique
        })

        # Check for convergence
        if critique.approved:
            return {
                "status": "converged",
                "result": proposal,
                "rounds": round + 1
            }

    return {
        "status": "max_rounds",
        "result": proposal,
        "rounds": max_rounds
    }
```

---

## DIALECTIC Methodology

SPINE's self-play pattern uses thesis/antithesis/synthesis:

```mermaid
graph LR
    T["THESIS<br/>Propose"]:::primary
    A["ANTITHESIS<br/>Critique"]:::secondary
    S["SYNTHESIS<br/>Merge"]:::tertiary

    T --> A --> S

    T -.- T1["Generate Solution"]:::primary
    A -.- A1["Challenge Solution"]:::secondary
    S -.- S1["Resolve Conflict"]:::tertiary

    classDef primary fill:#2563eb,color:#fff
    classDef secondary fill:#7c3aed,color:#fff
    classDef tertiary fill:#0d9488,color:#fff
```

---

## Oscillation Pattern

Alternating between perspectives to refine understanding:

```mermaid
graph LR
    R1["Round 1: A<br/>Expand / Explore / Diverge"]:::primary
    R2["Round 2: B<br/>Contract / Focus / Converge"]:::secondary
    R3["Round 3: A<br/>Expand / Explore / Diverge"]:::primary
    R4["Round 4: B<br/>Contract / Focus / Converge"]:::secondary
    MORE["..."]:::dark

    R1 --> R2 --> R3 --> R4 --> MORE

    classDef primary fill:#2563eb,color:#fff
    classDef secondary fill:#7c3aed,color:#fff
    classDef dark fill:#1e293b,color:#fff
```

---

## Oscillation Detection

Detect when execution is going in circles:

```python
class OscillationDetector:
    def __init__(self, window_size=5):
        self.history = []
        self.window_size = window_size

    def record(self, state):
        state_hash = hash(str(state))
        self.history.append(state_hash)

    def is_oscillating(self) -> bool:
        if len(self.history) < self.window_size:
            return False

        recent = self.history[-self.window_size:]
        unique = set(recent)

        # Only 2 unique states = A → B → A → B pattern
        if len(unique) <= 2:
            return True

        return False
```

---

## When to Use

| Scenario | Recommended |
|----------|-------------|
| Code review | Self-play (proposer/critic) |
| Design decisions | Oscillation (expand/contract) |
| Complex reasoning | DIALECTIC |
| Quality assurance | Self-play |

---

## Next Steps

- Learn about [Gating Mechanisms](../gating/)
- See implementation: [Lab 11: Self-Play](../../labs/11-self-play)

---

<div style="text-align: center;">
  <a href="../">← Back to Concepts</a>
</div>
