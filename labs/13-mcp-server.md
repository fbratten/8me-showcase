---
layout: lab
title: "Lab 13: MCP Server Deployment"
lab_number: 13
difficulty: integration
time: 2 hours
prerequisites: Labs 01-12 completed
---

# Lab 13: MCP Server Deployment

Deploy your orchestrator as an MCP server accessible to AI tools.

## Objectives

By the end of this lab, you will:
- Understand MCP (Model Context Protocol) architecture
- Deploy your orchestrator as an MCP server
- Expose resources and tools via MCP
- Test with Claude Desktop

## Prerequisites

- Labs 01-12 completed
- Python 3.10+
- Understanding of APIs and servers

## What is MCP?

**Model Context Protocol (MCP)** is a standard for AI tools to interact with external services:

```
┌─────────────────┐         ┌─────────────────┐
│   AI Client     │         │   MCP Server    │
│ (Claude Desktop)│◄───────►│ (Your Service)  │
└─────────────────┘   MCP   └─────────────────┘
                   Protocol
```

MCP provides:
- **Resources**: Read-only data (like GET endpoints)
- **Tools**: Actions that change state (like POST endpoints)
- **Standard protocol**: Works with any MCP-compatible client

---

## MCP Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      MCP SERVER                             │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐ │
│  │    Resources     │    │          Tools               │ │
│  │                  │    │                              │ │
│  │ • task://queue   │    │ • add_task(description)     │ │
│  │ • task://current │    │ • start_task(id)            │ │
│  │ • task://stats   │    │ • complete_task(id, result) │ │
│  │                  │    │ • fail_task(id, reason)     │ │
│  └──────────────────┘    └──────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                 Your Orchestrator                     │ │
│  │                                                       │ │
│  │  TaskStore ◄─► Executor ◄─► Verifier ◄─► Safety     │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## Step 1: Install FastMCP

```bash
pip install fastmcp
```

FastMCP is a lightweight framework for building MCP servers in Python.

---

## Step 2: Create the MCP Server

Create `mcp_server.py`:

```python
"""
MCP Server - Lab 13

Exposes the orchestrator via Model Context Protocol.
"""

from fastmcp import FastMCP
from typing import Optional, List
from dataclasses import asdict
import json

# Import your orchestrator components
from task_manager import TaskManager, TaskStatus


# Initialize FastMCP server
mcp = FastMCP("ralph-workflow")

# Initialize task manager (shared state)
task_manager = TaskManager("mcp_tasks.json")


# ==================== Resources ====================
# Resources are read-only views of data

@mcp.resource("task://queue")
def get_task_queue() -> str:
    """
    Get all tasks in the queue.

    Returns JSON array of all tasks with their status.
    """
    tasks = task_manager.get_all()
    return json.dumps([
        {
            "id": t.id,
            "description": t.description,
            "status": t.status.value if hasattr(t.status, 'value') else t.status,
            "attempts": t.attempts,
            "priority": getattr(t, 'priority', 5)
        }
        for t in tasks
    ], indent=2)


@mcp.resource("task://current")
def get_current_task() -> str:
    """
    Get the current task being processed.

    Returns the next pending task, or null if none.
    """
    task = task_manager.get_next()
    if task:
        return json.dumps({
            "id": task.id,
            "description": task.description,
            "status": task.status.value if hasattr(task.status, 'value') else task.status,
            "attempts": task.attempts
        }, indent=2)
    return json.dumps(None)


@mcp.resource("task://stats")
def get_stats() -> str:
    """
    Get task processing statistics.

    Returns counts and rates for task completion.
    """
    stats = task_manager.get_stats()
    return json.dumps(stats, indent=2)


@mcp.resource("task://pending")
def get_pending_tasks() -> str:
    """
    Get only pending tasks.

    Returns tasks that are waiting to be processed.
    """
    pending = task_manager.get_pending()
    return json.dumps([
        {
            "id": t.id,
            "description": t.description,
            "priority": getattr(t, 'priority', 5)
        }
        for t in pending
    ], indent=2)


@mcp.resource("task://completed")
def get_completed_tasks() -> str:
    """
    Get completed tasks with results.

    Returns tasks that have been successfully processed.
    """
    completed = task_manager.get_completed()
    return json.dumps([
        {
            "id": t.id,
            "description": t.description,
            "result": t.result[:200] if t.result else None  # Truncate long results
        }
        for t in completed
    ], indent=2)


# ==================== Tools ====================
# Tools are actions that modify state

@mcp.tool()
def add_task(
    description: str,
    priority: int = 5,
    max_attempts: int = 3
) -> str:
    """
    Add a new task to the queue.

    Args:
        description: What the task should accomplish
        priority: Priority level (1=highest, 10=lowest)
        max_attempts: Maximum retry attempts

    Returns:
        The created task ID
    """
    task = task_manager.create(
        description=description,
        priority=priority,
        max_attempts=max_attempts
    )
    return f"Created task: {task.id}"


@mcp.tool()
def start_task(task_id: str) -> str:
    """
    Mark a task as in-progress.

    Args:
        task_id: The task ID to start

    Returns:
        Confirmation message
    """
    task = task_manager.start(task_id)
    if task:
        return f"Started task: {task_id} (attempt {task.attempts})"
    return f"Task not found: {task_id}"


@mcp.tool()
def complete_task(task_id: str, result: str) -> str:
    """
    Mark a task as completed with result.

    Args:
        task_id: The task ID to complete
        result: The result/output of the task

    Returns:
        Confirmation message
    """
    task = task_manager.complete(task_id, result)
    if task:
        return f"Completed task: {task_id}"
    return f"Task not found: {task_id}"


@mcp.tool()
def fail_task(task_id: str, reason: str) -> str:
    """
    Mark a task as failed.

    Args:
        task_id: The task ID to fail
        reason: Why the task failed

    Returns:
        Confirmation message
    """
    task = task_manager.fail(task_id, reason)
    if task:
        return f"Failed task: {task_id} - {reason}"
    return f"Task not found: {task_id}"


@mcp.tool()
def retry_task(task_id: str, feedback: str = "") -> str:
    """
    Reset a task for retry.

    Args:
        task_id: The task ID to retry
        feedback: Optional feedback for the retry

    Returns:
        Confirmation message or error if max attempts reached
    """
    task = task_manager.retry(task_id, feedback)
    if task:
        if task.status == TaskStatus.FAILED:
            return f"Task {task_id} exceeded max attempts"
        return f"Task {task_id} queued for retry"
    return f"Task not found: {task_id}"


@mcp.tool()
def get_next_task() -> str:
    """
    Get the next task to process.

    Returns the highest priority pending task.
    """
    task = task_manager.get_next()
    if task:
        return json.dumps({
            "id": task.id,
            "description": task.description,
            "attempts": task.attempts,
            "max_attempts": task.max_attempts
        }, indent=2)
    return "No pending tasks"


@mcp.tool()
def clear_completed() -> str:
    """
    Remove all completed tasks from the queue.

    Returns:
        Number of tasks removed
    """
    completed = task_manager.get_completed()
    count = len(completed)
    for task in completed:
        task_manager.delete(task.id)
    return f"Cleared {count} completed tasks"


@mcp.tool()
def reset_all_tasks() -> str:
    """
    Reset all tasks to pending status.

    Use for testing or reprocessing.
    """
    task_manager.reset_all()
    return "All tasks reset to pending"


# ==================== Entry Point ====================

if __name__ == "__main__":
    mcp.run()
```

---

## Step 3: Create Package Structure

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ralph-mcp-server"
version = "1.0.0"
description = "MCP server for Ralph workflow orchestration"
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=0.1.0",
    "anthropic>=0.18.0",
]

[project.scripts]
ralph-mcp = "mcp_server:mcp.run"

[tool.hatch.build.targets.wheel]
packages = ["."]
```

---

## Step 4: Install and Test

```bash
# Install in development mode
pip install -e .

# Run the server directly
python mcp_server.py

# Or use the installed command
ralph-mcp
```

---

## Step 5: Configure Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json` on Linux/Mac or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "ralph-mcp",
      "args": []
    }
  }
}
```

Or if running from source:

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop to load the server.

---

## Step 6: Test with Claude

In Claude Desktop, you can now:

```
User: Add a task to write a haiku about MCP

Claude: [Uses add_task tool]
Created task: task-001

User: What tasks are in the queue?

Claude: [Reads task://queue resource]
Here are the current tasks:
- task-001: Write a haiku about MCP (pending)

User: Process the next task

Claude: [Uses get_next_task, then complete_task]
Task completed! Here's the haiku:
...
```

---

## Understanding MCP Concepts

### Resources vs Tools

| Aspect | Resources | Tools |
|--------|-----------|-------|
| Purpose | Read data | Perform actions |
| HTTP analogy | GET | POST/PUT/DELETE |
| Side effects | None | Yes |
| Caching | Can be cached | Not cached |
| Use case | View state | Change state |

### Resource URIs

```python
@mcp.resource("task://queue")      # All tasks
@mcp.resource("task://current")    # Current task
@mcp.resource("task://stats")      # Statistics

# Pattern: protocol://path
# - protocol: category of resource
# - path: specific resource
```

### Tool Definitions

```python
@mcp.tool()
def add_task(
    description: str,      # Required parameter
    priority: int = 5,     # Optional with default
    max_attempts: int = 3  # Optional with default
) -> str:
    """
    Add a new task to the queue.  # Description shown to AI

    Args:
        description: What the task should accomplish  # Param docs
        priority: Priority level (1=highest, 10=lowest)

    Returns:
        The created task ID  # Return description
    """
```

---

## Advanced: Adding the Orchestrator

Extend the server to include full orchestration:

```python
from orchestrator import Orchestrator, OrchestratorBuilder
from components import ClaudeExecutor, ClaudeVerifier, SimpleCircuitBreaker

# Global orchestrator instance
orchestrator = None

@mcp.tool()
def run_orchestrator(max_tasks: int = 10) -> str:
    """
    Run the orchestrator to process pending tasks.

    Args:
        max_tasks: Maximum tasks to process in this run

    Returns:
        Summary of tasks processed
    """
    global orchestrator

    if orchestrator is None:
        orchestrator = (OrchestratorBuilder()
            .with_task_store(task_manager)
            .with_executor(ClaudeExecutor())
            .with_verifier(ClaudeVerifier())
            .with_safety(SimpleCircuitBreaker(max_iterations=max_tasks))
            .build())

    stats = orchestrator.run()

    return json.dumps({
        "processed": stats.tasks_processed,
        "completed": stats.tasks_completed,
        "failed": stats.tasks_failed,
        "elapsed_seconds": stats.elapsed_seconds
    }, indent=2)
```

---

## Exercises

### Exercise 1: Add Batch Operations

Add tools for batch task operations:

```python
@mcp.tool()
def add_tasks_batch(descriptions: List[str]) -> str:
    """Add multiple tasks at once."""
    pass

@mcp.tool()
def complete_tasks_batch(task_ids: List[str], result: str) -> str:
    """Complete multiple tasks with the same result."""
    pass
```

### Exercise 2: Add Filtering Resource

Create a resource that accepts parameters:

```python
@mcp.resource("task://filter/{status}")
def get_tasks_by_status(status: str) -> str:
    """Get tasks filtered by status."""
    pass
```

### Exercise 3: Add WebSocket Notifications

Implement real-time task notifications (advanced):

```python
# When task status changes, notify connected clients
async def notify_task_change(task_id: str, new_status: str):
    pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] MCP server starts without errors
- [ ] Resources return correct JSON
- [ ] Tools modify state correctly
- [ ] Claude Desktop can connect to your server
- [ ] You can add and complete tasks via Claude

---

## Key Takeaway

> MCP makes your orchestrator accessible to AI tools.

With MCP, your orchestrator becomes:
- **Accessible**: Any MCP client can use it
- **Discoverable**: Tools and resources are self-documenting
- **Standardized**: Follows the MCP protocol
- **Composable**: Works with other MCP servers

---

## Get the Code

Full implementation: [8me/src/tier3-mcp-server/](https://github.com/fbratten/8me/tree/main/src/tier3-mcp-server)

---

<div class="lab-navigation">
  <a href="./12-memory.md" class="prev">← Previous: Lab 12 - Memory Integration</a>
  <a href="./14-skill.md" class="next">Next: Lab 14 - Claude Code Skill →</a>
</div>
