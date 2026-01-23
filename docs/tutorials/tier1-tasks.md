---
layout: page
title: Tier 1 Task Management
permalink: /tutorials/tier1-tasks/
parent: Tutorials
---

# Tier 1: Task Management

JSON-based task state management.

---

## Task Structure

```json
{
  "id": "1",
  "description": "Write a haiku about loops",
  "status": "pending",
  "attempts": 0,
  "max_attempts": 3,
  "result": null,
  "error": null,
  "created_at": "2026-01-19T00:00:00",
  "completed_at": null
}
```

---

## Task States

| State | Meaning |
|-------|---------|
| `pending` | Ready to execute |
| `in_progress` | Currently being worked on |
| `completed` | Successfully finished |
| `failed` | Gave up after max attempts |

---

## State Transitions

```
pending ──► in_progress ──► completed
                │
                ▼
            (retry) ──► pending
                │
                ▼
              failed
```

---

## TaskManager API

```python
from task_manager import TaskManager

tm = TaskManager("tasks.json")

# Add task
task = tm.add_task("Write a haiku")

# Get next pending
task = tm.get_next_pending()

# Update status
tm.mark_in_progress(task.id)
tm.mark_completed(task.id, "The result...")
tm.mark_failed(task.id, "Reason for failure")

# Query
all_tasks = tm.get_all_tasks()
pending = tm.get_tasks_by_status("pending")
```

---

## File Format

```json
{
  "tasks": [...],
  "metadata": {
    "created": "2026-01-19T00:00:00",
    "last_updated": "2026-01-19T00:00:00"
  }
}
```

---

## Next Steps

- [Safety Features](../tier1-safety/)
- [Lab 04: JSON Tasks](../../labs/04-json-tasks)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
