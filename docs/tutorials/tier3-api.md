---
layout: page
title: Tier 3 API
permalink: /tutorials/tier3-api/
parent: Tutorials
---

# Tier 3: API Reference

Resources and tools available in the Ralph MCP server.

---

## Resources

### task://queue

All tasks in the queue.

```json
{
  "tasks": [
    {
      "id": "1",
      "description": "Write tests",
      "status": "pending",
      "attempts": 0
    }
  ]
}
```

---

### task://current

Currently in-progress task (if any).

```json
{
  "id": "2",
  "description": "Refactor auth",
  "status": "in_progress",
  "attempts": 1
}
```

---

### task://stats

Queue statistics.

```json
{
  "total": 5,
  "pending": 2,
  "in_progress": 1,
  "completed": 2,
  "failed": 0
}
```

---

## Tools

### add_task

Add a new task to the queue.

**Input:**
```json
{
  "description": "Write unit tests",
  "priority": "high"
}
```

**Output:**
```json
{
  "id": "3",
  "status": "pending"
}
```

---

### start_task

Begin working on a task.

**Input:**
```json
{
  "task_id": "1"
}
```

---

### complete_task

Mark a task as completed.

**Input:**
```json
{
  "task_id": "1",
  "result": "Created 5 test files"
}
```

---

### fail_task

Mark a task as failed.

**Input:**
```json
{
  "task_id": "1",
  "reason": "Could not parse input",
  "retryable": true
}
```

---

### reset_task

Reset a task to pending.

**Input:**
```json
{
  "task_id": "1"
}
```

---

### get_next_task

Get the next pending task.

**Output:**
```json
{
  "id": "2",
  "description": "Next task to work on"
}
```

---

### clear_completed

Remove all completed tasks.

---

## Next Steps

- [Lab 13: Build Your Own MCP Server](../../labs/13-mcp-server)
- [MCP Setup Guide](../mcp-setup/)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
