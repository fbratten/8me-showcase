---
layout: lab
title: "Lab 04: JSON Task Management"
lab_number: 4
difficulty: intermediate
time: 1 hour
prerequisites: Labs 01-03 completed
---

# Lab 04: JSON Task Management

Build a robust TaskManager class with full CRUD operations and metadata tracking.

## Objectives

By the end of this lab, you will:
- Structure tasks with rich JSON schemas
- Implement full CRUD operations
- Track metadata (attempts, timestamps, history)
- Query tasks by status, age, and other criteria

## Prerequisites

- Labs 01-03 completed
- Understanding of JSON and Python classes

## Why Structured Task Management?

In earlier labs, we used simple dictionaries. That works for demos, but production systems need:

| Ad-hoc Approach | Structured Approach |
|-----------------|---------------------|
| Scattered logic | Centralized TaskManager |
| Implicit state | Explicit lifecycle |
| No history | Full audit trail |
| Hard to query | Rich query methods |
| Fragile | Validated schemas |

---

## The Task Schema

A production-ready task schema:

```json
{
  "id": "task-001",
  "description": "Write a haiku about loops",
  "status": "pending",
  "priority": 1,
  "created_at": "2026-01-19T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "attempts": 0,
  "max_attempts": 3,
  "result": null,
  "criteria": {
    "type": "haiku",
    "requirements": ["3 lines", "5-7-5 syllables"]
  },
  "metadata": {
    "source": "user-input",
    "tags": ["creative", "poetry"]
  },
  "history": []
}
```

### Field Descriptions

| Field | Type | Purpose |
|-------|------|---------|
| `id` | string | Unique identifier |
| `description` | string | What to do |
| `status` | enum | pending/in_progress/completed/failed |
| `priority` | int | Lower = higher priority |
| `created_at` | ISO date | When task was created |
| `started_at` | ISO date | When work began |
| `completed_at` | ISO date | When task finished |
| `attempts` | int | Number of tries so far |
| `max_attempts` | int | Maximum allowed attempts |
| `result` | string | The final output |
| `criteria` | object | Verification requirements |
| `metadata` | object | Extensible custom data |
| `history` | array | Audit trail of events |

---

## Step 1: Create the TaskManager Class

Create `task_manager.py`:

```python
"""
Task Manager - Lab 04

Production-ready task management with JSON persistence,
full CRUD operations, and rich querying capabilities.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HistoryEntry:
    """A single history event."""
    timestamp: str
    event: str
    details: Optional[str] = None


@dataclass
class Task:
    """A single task with full metadata."""
    id: str
    description: str
    status: str = "pending"
    priority: int = 5
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[str] = None
    criteria: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    failure_reason: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.criteria is None:
            self.criteria = {}
        if self.metadata is None:
            self.metadata = {}
        if self.history is None:
            self.history = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create Task from dictionary."""
        return cls(**data)


class TaskManager:
    """
    Manages tasks with JSON persistence and rich querying.

    Usage:
        manager = TaskManager("tasks.json")
        task = manager.create("Write a haiku")
        manager.start(task.id)
        manager.complete(task.id, "Result here")
    """

    def __init__(self, filepath: str = "tasks.json"):
        self.filepath = Path(filepath)
        self.tasks: Dict[str, Task] = {}
        self._load()

    # ==================== Persistence ====================

    def _load(self):
        """Load tasks from JSON file."""
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                data = json.load(f)
                for task_data in data.get("tasks", []):
                    task = Task.from_dict(task_data)
                    self.tasks[task.id] = task
            print(f"Loaded {len(self.tasks)} tasks from {self.filepath}")
        else:
            print(f"No existing tasks file, starting fresh")

    def _save(self):
        """Persist all tasks to JSON file."""
        data = {
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    # ==================== CRUD Operations ====================

    def create(
        self,
        description: str,
        priority: int = 5,
        criteria: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        max_attempts: int = 3
    ) -> Task:
        """Create a new task."""
        task = Task(
            id=f"task-{uuid.uuid4().hex[:8]}",
            description=description,
            priority=priority,
            criteria=criteria,
            metadata=metadata,
            max_attempts=max_attempts
        )

        self._add_history(task, "created", f"Priority: {priority}")
        self.tasks[task.id] = task
        self._save()

        return task

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> Optional[Task]:
        """Update task fields."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self._add_history(task, "updated", f"Fields: {list(kwargs.keys())}")
        self._save()
        return task

    def delete(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save()
            return True
        return False

    # ==================== Lifecycle Operations ====================

    def start(self, task_id: str) -> Optional[Task]:
        """Mark a task as in progress."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        task.status = TaskStatus.IN_PROGRESS.value
        task.started_at = datetime.now(timezone.utc).isoformat()
        task.attempts += 1

        self._add_history(task, "started", f"Attempt {task.attempts}")
        self._save()
        return task

    def complete(self, task_id: str, result: str) -> Optional[Task]:
        """Mark a task as completed with result."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        task.status = TaskStatus.COMPLETED.value
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result = result

        self._add_history(task, "completed", f"Result length: {len(result)}")
        self._save()
        return task

    def fail(self, task_id: str, reason: str) -> Optional[Task]:
        """Mark a task as permanently failed."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        task.status = TaskStatus.FAILED.value
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.failure_reason = reason

        self._add_history(task, "failed", reason)
        self._save()
        return task

    def retry(self, task_id: str, feedback: str = "") -> Optional[Task]:
        """Reset task to pending for retry."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        if task.attempts >= task.max_attempts:
            return self.fail(task_id, f"Max attempts ({task.max_attempts}) reached")

        task.status = TaskStatus.PENDING.value
        task.started_at = None

        self._add_history(task, "retry_scheduled", feedback)
        self._save()
        return task

    # ==================== Query Methods ====================

    def get_all(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())

    def get_by_status(self, status: str) -> List[Task]:
        """Get tasks with specific status."""
        return [t for t in self.tasks.values() if t.status == status]

    def get_pending(self) -> List[Task]:
        """Get all pending tasks, sorted by priority."""
        pending = self.get_by_status(TaskStatus.PENDING.value)
        return sorted(pending, key=lambda t: t.priority)

    def get_next(self) -> Optional[Task]:
        """Get the next task to process (highest priority pending)."""
        pending = self.get_pending()
        return pending[0] if pending else None

    def has_pending(self) -> bool:
        """Check if there are pending tasks."""
        return len(self.get_pending()) > 0

    def get_failed(self) -> List[Task]:
        """Get all failed tasks."""
        return self.get_by_status(TaskStatus.FAILED.value)

    def get_completed(self) -> List[Task]:
        """Get all completed tasks."""
        return self.get_by_status(TaskStatus.COMPLETED.value)

    def get_by_tag(self, tag: str) -> List[Task]:
        """Get tasks with a specific tag."""
        return [
            t for t in self.tasks.values()
            if tag in t.metadata.get("tags", [])
        ]

    def get_stale(self, hours: int = 24) -> List[Task]:
        """Get in-progress tasks older than specified hours."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        stale = []

        for task in self.get_by_status(TaskStatus.IN_PROGRESS.value):
            if task.started_at:
                started = datetime.fromisoformat(task.started_at).timestamp()
                if started < cutoff:
                    stale.append(task)

        return stale

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        all_tasks = list(self.tasks.values())
        total = len(all_tasks)

        if total == 0:
            return {"total": 0, "message": "No tasks"}

        completed = len(self.get_completed())
        failed = len(self.get_failed())
        pending = len(self.get_pending())
        in_progress = len(self.get_by_status(TaskStatus.IN_PROGRESS.value))

        # Calculate average attempts for completed tasks
        completed_tasks = self.get_completed()
        avg_attempts = (
            sum(t.attempts for t in completed_tasks) / len(completed_tasks)
            if completed_tasks else 0
        )

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "completion_rate": completed / total * 100,
            "failure_rate": failed / total * 100 if total > 0 else 0,
            "average_attempts": round(avg_attempts, 2)
        }

    # ==================== Utility ====================

    def _add_history(self, task: Task, event: str, details: str = ""):
        """Add an entry to task history."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": details
        }
        task.history.append(entry)

    def reset_all(self):
        """Reset all tasks to pending (for testing)."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING.value
            task.attempts = 0
            task.result = None
            task.started_at = None
            task.completed_at = None
            self._add_history(task, "reset", "Manual reset")
        self._save()

    def import_from_text(self, filepath: str, default_priority: int = 5):
        """Import tasks from a text file (one per line)."""
        with open(filepath, "r") as f:
            for line in f:
                description = line.strip()
                if description and not description.startswith("#"):
                    self.create(description, priority=default_priority)

    def export_results(self, filepath: str):
        """Export completed task results to a file."""
        with open(filepath, "w") as f:
            for task in self.get_completed():
                f.write(f"## Task: {task.description}\n\n")
                f.write(f"{task.result}\n\n")
                f.write(f"---\n\n")
```

---

## Step 2: Create a CLI Interface

Create `cli.py`:

```python
"""
Task Manager CLI - Lab 04

Command-line interface for task management.
"""

import argparse
from task_manager import TaskManager


def main():
    parser = argparse.ArgumentParser(description="Task Manager CLI")
    parser.add_argument("--file", default="tasks.json", help="Tasks file")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add task
    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("description", help="Task description")
    add_parser.add_argument("--priority", type=int, default=5)

    # List tasks
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", help="Filter by status")

    # Show task
    show_parser = subparsers.add_parser("show", help="Show task details")
    show_parser.add_argument("task_id", help="Task ID")

    # Stats
    subparsers.add_parser("stats", help="Show statistics")

    # Import
    import_parser = subparsers.add_parser("import", help="Import from text file")
    import_parser.add_argument("filepath", help="Text file path")

    # Reset
    subparsers.add_parser("reset", help="Reset all tasks to pending")

    args = parser.parse_args()
    manager = TaskManager(args.file)

    if args.command == "add":
        task = manager.create(args.description, priority=args.priority)
        print(f"Created: {task.id}")

    elif args.command == "list":
        if args.status:
            tasks = manager.get_by_status(args.status)
        else:
            tasks = manager.get_all()

        for task in tasks:
            icon = {"pending": "○", "in_progress": "◐", "completed": "✓", "failed": "✗"}
            print(f"{icon.get(task.status, '?')} [{task.id}] {task.description[:50]}")

    elif args.command == "show":
        task = manager.get(args.task_id)
        if task:
            print(f"ID: {task.id}")
            print(f"Description: {task.description}")
            print(f"Status: {task.status}")
            print(f"Priority: {task.priority}")
            print(f"Attempts: {task.attempts}/{task.max_attempts}")
            print(f"Created: {task.created_at}")
            if task.history:
                print(f"\nHistory ({len(task.history)} events):")
                for entry in task.history[-5:]:  # Last 5 events
                    print(f"  {entry['timestamp'][:19]} - {entry['event']}")
        else:
            print(f"Task not found: {args.task_id}")

    elif args.command == "stats":
        stats = manager.get_stats()
        print(f"Total: {stats['total']}")
        print(f"Completed: {stats['completed']} ({stats['completion_rate']:.1f}%)")
        print(f"Failed: {stats['failed']} ({stats['failure_rate']:.1f}%)")
        print(f"Pending: {stats['pending']}")
        print(f"In Progress: {stats['in_progress']}")
        print(f"Avg Attempts: {stats['average_attempts']}")

    elif args.command == "import":
        manager.import_from_text(args.filepath)
        print(f"Imported tasks from {args.filepath}")

    elif args.command == "reset":
        manager.reset_all()
        print("All tasks reset to pending")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

## Step 3: Use in Your Loop

Update your main loop to use the TaskManager:

```python
"""
Loop with TaskManager - Lab 04
"""

import anthropic
from task_manager import TaskManager


client = anthropic.Anthropic()
manager = TaskManager("tasks.json")


def process_task(task):
    """Process a single task."""
    manager.start(task.id)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": task.description}]
    )
    result = response.content[0].text

    # Simplified verification (see Lab 03 for full version)
    manager.complete(task.id, result)
    return True


def main():
    # Import tasks if empty
    if not manager.tasks:
        manager.import_from_text("tasks.txt")

    stats = manager.get_stats()
    print(f"Tasks: {stats['pending']} pending, {stats['completed']} done\n")

    while manager.has_pending():
        task = manager.get_next()
        print(f"Processing [{task.id}] {task.description[:40]}...")

        if process_task(task):
            print(f"  ✓ Completed\n")
        else:
            print(f"  ✗ Failed\n")

    # Final stats
    stats = manager.get_stats()
    print(f"\nDone! {stats['completion_rate']:.0f}% completion rate")
    print(f"Average attempts: {stats['average_attempts']}")


if __name__ == "__main__":
    main()
```

---

## Step 4: Test the CLI

```bash
# Add tasks
python cli.py add "Write a haiku about Python" --priority 1
python cli.py add "Explain recursion simply" --priority 2
python cli.py add "List 3 benefits of testing" --priority 3

# List all tasks
python cli.py list

# Show details
python cli.py show task-abc12345

# Check stats
python cli.py stats

# Import from file
python cli.py import tasks.txt
```

---

## Understanding the Design

### Task Lifecycle

```
    ┌──────────┐
    │  CREATE  │
    └────┬─────┘
         │
         ▼
    ┌──────────┐
    │ PENDING  │◄─────────────────┐
    └────┬─────┘                  │
         │ start()                │ retry()
         ▼                        │
    ┌──────────────┐              │
    │ IN_PROGRESS  │──────────────┤
    └────┬─────────┘              │
         │                        │
    ┌────┴────┐                   │
    │         │                   │
    ▼         ▼                   │
┌────────┐  ┌────────┐            │
│COMPLETED│  │ FAILED │───────────┘
└────────┘  └────────┘   (if retries left)
```

### Why History Matters

```python
task.history = [
    {"timestamp": "...", "event": "created", "details": "Priority: 1"},
    {"timestamp": "...", "event": "started", "details": "Attempt 1"},
    {"timestamp": "...", "event": "retry_scheduled", "details": "Wrong format"},
    {"timestamp": "...", "event": "started", "details": "Attempt 2"},
    {"timestamp": "...", "event": "completed", "details": "Result length: 47"}
]
```

History enables:
- **Debugging**: See exactly what happened
- **Auditing**: Track all state changes
- **Analytics**: Measure time between events
- **Replay**: Understand failure patterns

### Priority Scheduling

```python
def get_next(self) -> Optional[Task]:
    pending = self.get_pending()  # Already sorted by priority
    return pending[0] if pending else None
```

Lower priority number = processed first. This enables:
- Urgent tasks (priority 1) processed before routine tasks (priority 5)
- Consistent ordering across restarts

---

## Exercises

### Exercise 1: Add Dependencies

Allow tasks to depend on other tasks:

```python
{
    "id": "task-002",
    "depends_on": ["task-001"],  # Must complete task-001 first
    ...
}
```

Update `get_pending()` to respect dependencies.

### Exercise 2: Add Deadlines

Add deadline tracking:

```python
{
    "deadline": "2026-01-20T18:00:00Z",
    ...
}

def get_overdue(self) -> List[Task]:
    """Get tasks past their deadline."""
    pass
```

### Exercise 3: Task Templates

Create reusable task templates:

```python
manager.create_from_template("code-review", variables={"file": "main.py"})
```

---

## Checkpoint

Before moving on, verify:
- [ ] Tasks persist correctly to JSON
- [ ] History tracks all state changes
- [ ] CLI commands work as expected
- [ ] Priority scheduling works correctly
- [ ] Stats accurately reflect task states

---

## Key Takeaway

> Structured task management scales better than ad-hoc.

A proper TaskManager gives you:
- **Single source of truth** for all task state
- **Full audit trail** via history
- **Rich querying** for monitoring and debugging
- **Consistent lifecycle** management
- **Easy testing** via reset and import

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/task_manager.py](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./03-verification" class="prev">← Previous: Lab 03 - Simple Verification</a>
  <a href="./05-tool-calling" class="next">Next: Lab 05 - Tool Calling →</a>
</div>
