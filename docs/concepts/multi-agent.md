---
layout: page
title: Multi-Agent Coordination
parent: Concepts
---

# Multi-Agent Coordination

Pipelines, swarms, and role-based teams.

---

## Multi-Agent Patterns

### Pipeline Pattern

Specialized agents in sequence:

```python
def multi_agent_pipeline(task):
    """
    Pipeline of specialized agents.
    Each agent has a specific role.
    """
    # Agent 1: Planner
    plan = planner_agent(task)

    # Agent 2: Implementer
    implementation = implementer_agent(task, plan)

    # Agent 3: Reviewer
    review = reviewer_agent(task, implementation)

    if review.approved:
        return implementation
    else:
        return multi_agent_pipeline_with_feedback(
            task, implementation, review.feedback
        )
```

---

### Swarm Pattern

Multiple agents work in parallel:

```python
def multi_agent_swarm(task, agent_pool):
    """
    Swarm approach: multiple agents work in parallel,
    results are merged/voted on.
    """
    # Dispatch to all agents
    futures = [
        agent.execute_async(task)
        for agent in agent_pool
    ]

    # Gather results
    results = wait_all(futures)

    # Merge strategy: vote, merge, or select best
    final_result = merge_results(results, strategy="vote")

    return final_result
```

---

### Role-Based Teams

Agents with defined roles collaborate:

```
┌─────────────────────────────────────────────────────────┐
│                   Role-Based Team                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │ PLANNER  │───►│IMPLEMENTER│───►│ REVIEWER │        │
│   │          │    │          │    │          │        │
│   │ Designs  │    │ Executes │    │ Validates│        │
│   │ approach │    │ plan     │    │ quality  │        │
│   └──────────┘    └──────────┘    └──────────┘        │
│        │               │               │               │
│        └───────────────┴───────────────┘               │
│                    Feedback Loop                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Coordination Strategies

### 1. Sequential (Pipeline)

```python
result = agent_a(task)
result = agent_b(result)
result = agent_c(result)
```

### 2. Parallel (Swarm)

```python
results = parallel_execute([agent_a, agent_b, agent_c], task)
final = merge(results)
```

### 3. Hierarchical

```python
def hierarchical(task, manager, workers):
    subtasks = manager.decompose(task)
    results = [workers[i].execute(st) for i, st in enumerate(subtasks)]
    return manager.synthesize(results)
```

---

## Conflict Resolution

When agents disagree:

```python
class ConflictResolver:
    CONFLICT_TYPES = [
        "factual",       # Mutually exclusive claims
        "definitional",  # Same term used differently
        "temporal",      # Different time windows
        "scope",         # Different contexts
        "metric",        # Different evaluation criteria
        "interpretive"   # Different judgments
    ]

    def resolve(self, conflicts):
        for conflict in conflicts:
            if conflict["type"] in ["scope", "temporal"]:
                resolution = self.reconcile_context(conflict)
            elif conflict["type"] == "factual":
                resolution = self.prefer_evidence(conflict)
            else:
                resolution = self.generate_verification(conflict)
        return resolutions
```

---

## When to Use Multi-Agent

| Scenario | Pattern |
|----------|---------|
| Complex tasks needing specialization | Pipeline |
| Need diverse perspectives | Swarm |
| Team simulation | Role-based |
| Quality assurance | Proposer/Critic |

---

## Next Steps

- Learn about [Provider Comparison](./providers)
- See implementation: [Lab 09: Multi-Agent](../labs/09-multi-agent)

---

<div style="text-align: center;">
  <a href="./">← Back to Concepts</a>
</div>
