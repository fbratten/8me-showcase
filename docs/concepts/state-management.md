---
layout: page
title: State Management
permalink: /concepts/state-management/
parent: Concepts
---

# State Management

Persisting state for reliability and debugging.

---

## Why State Management?

Without external state:
- Crashes lose all progress
- Can't resume interrupted work
- No audit trail
- Difficult debugging

---

## State Machine Pattern

For tasks with defined transitions:

```python
class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"

def state_machine_loop(task_store):
    """
    Tasks transition through states.
    Different handlers for each state.
    """
    while has_work(task_store):
        task = get_next_task(task_store)

        if task.state == TaskState.PENDING:
            start_task(task)
            task.state = TaskState.IN_PROGRESS

        elif task.state == TaskState.IN_PROGRESS:
            result = execute(task)
            task.result = result
            task.state = TaskState.REVIEW

        elif task.state == TaskState.REVIEW:
            if verify(task.result):
                task.state = TaskState.COMPLETED
            else:
                task.state = TaskState.PENDING  # Back to queue
                task.attempts += 1

        save_task(task)

    return summarize(task_store)
```

**Use when:** Complex workflows, audit trails needed, multiple handlers

---

## Storage Options

### 1. File-Based (JSON)

Simple and portable:

```python
import json

def save_state(tasks, path="tasks.json"):
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2)

def load_state(path="tasks.json"):
    with open(path) as f:
        data = json.load(f)
    return [Task.from_dict(d) for d in data]
```

### 2. SQLite (Minna Memory)

Structured with relationships:

```python
from minna_memory import MinnaMemory

memory = MinnaMemory(".spine/minna.db")

# Store task state
memory.store(
    entity=task.id,
    attribute="status",
    value="in_progress"
)

# Recall state
state = memory.recall(entity=task.id, attribute="status")
```

### 3. MCP Resources

External state via MCP:

```python
# task://queue resource
tasks = await mcp_client.read_resource("task://queue")

# task://current resource
current = await mcp_client.read_resource("task://current")
```

---

## State Transitions

```
┌─────────────────────────────────────────────────────────┐
│                    State Transitions                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   PENDING ──────► IN_PROGRESS ──────► REVIEW           │
│      ▲                                   │              │
│      │                          ┌────────┴────────┐     │
│      │                          ▼                 ▼     │
│      │                     COMPLETED           FAILED   │
│      │                                           │      │
│      └───────────── (retry) ─────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Best Practices

1. **Always persist before processing** - Save state before attempting work
2. **Atomic updates** - Don't leave state in inconsistent condition
3. **Include timestamps** - Track when states changed
4. **Log transitions** - Audit trail for debugging
5. **Handle recovery** - What happens on restart?

---

## Next Steps

- Learn about [Self-Play Patterns](../self-play/)
- See implementation: [Lab 02: External State](../labs/02-external-state)

---

<div style="text-align: center;">
  <a href="./">← Back to Concepts</a>
</div>
