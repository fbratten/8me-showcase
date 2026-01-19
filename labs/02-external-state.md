---
layout: lab
title: "Lab 02: External State"
lab_number: 2
difficulty: beginner
time: 45 minutes
prerequisites: Lab 01 completed
---

# Lab 02: External State

Add persistent state that survives script restarts and enables resume capability.

## Objectives

By the end of this lab, you will:
- Understand external state management
- Read and write task status to files
- Track progress between runs
- Resume work after interruptions

## Prerequisites

- Lab 01 completed
- Understanding of JSON basics

## The Core Concept

In Lab 01, all state lived in memory. If the script crashed, we'd lose all progress:

```python
# Lab 01: Memory-only state (lost on crash)
tasks = read_tasks("tasks.txt")
for task in tasks:
    process(task)  # If crash here, no record of what completed
```

**External state** survives restarts:

```python
# Lab 02: External state (survives crash)
state = load_state("state.json")

while state.has_pending():
    task = state.get_next()
    result = process(task)
    state.mark_complete(task)
    state.save()  # Persist after every task!
```

---

## Step 1: Design the State File

We'll use JSON to track task status:

```json
{
  "tasks": [
    {
      "id": 1,
      "description": "Write a haiku about loops",
      "status": "completed",
      "result": "Endless turning wheel..."
    },
    {
      "id": 2,
      "description": "Explain recursion",
      "status": "pending",
      "result": null
    }
  ]
}
```

Each task has:
- **id**: Unique identifier
- **description**: What to do
- **status**: `pending`, `in_progress`, or `completed`
- **result**: The AI's response (null until completed)

## Step 2: Create the State Manager

Create `state_manager.py`:

```python
"""
State Manager - Lab 02

Manages persistent state for task tracking.
Enables resume capability after script restarts.
"""

import json
from pathlib import Path
from typing import Optional


class StateManager:
    """Manages task state with JSON persistence."""

    def __init__(self, filepath: str = "state.json"):
        self.filepath = Path(filepath)
        self.tasks: list[dict] = []
        self._load()

    def _load(self):
        """Load state from file, or initialize empty."""
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                data = json.load(f)
                self.tasks = data.get("tasks", [])
            print(f"Loaded {len(self.tasks)} tasks from {self.filepath}")
        else:
            self.tasks = []
            print(f"No existing state found, starting fresh")

    def save(self):
        """Persist current state to file."""
        with open(self.filepath, "w") as f:
            json.dump({"tasks": self.tasks}, f, indent=2)

    def add_task(self, description: str) -> dict:
        """Add a new task to the queue."""
        task = {
            "id": len(self.tasks) + 1,
            "description": description,
            "status": "pending",
            "result": None
        }
        self.tasks.append(task)
        self.save()
        return task

    def load_from_file(self, filepath: str):
        """Import tasks from a text file (one per line)."""
        with open(filepath, "r") as f:
            for line in f:
                description = line.strip()
                if description:
                    self.add_task(description)

    def get_pending(self) -> list[dict]:
        """Return all pending tasks."""
        return [t for t in self.tasks if t["status"] == "pending"]

    def get_next(self) -> Optional[dict]:
        """Get the next pending task."""
        pending = self.get_pending()
        return pending[0] if pending else None

    def has_pending(self) -> bool:
        """Check if there are pending tasks."""
        return len(self.get_pending()) > 0

    def mark_in_progress(self, task_id: int):
        """Mark a task as in progress."""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "in_progress"
                self.save()
                return

    def mark_complete(self, task_id: int, result: str):
        """Mark a task as completed with its result."""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["result"] = result
                self.save()
                return

    def get_stats(self) -> dict:
        """Return statistics about task progress."""
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t["status"] == "completed"])
        pending = len([t for t in self.tasks if t["status"] == "pending"])
        in_progress = len([t for t in self.tasks if t["status"] == "in_progress"])

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "in_progress": in_progress,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }
```

## Step 3: Update the Main Loop

Create `loop_with_state.py`:

```python
"""
Loop with External State - Lab 02

Demonstrates resumable task processing with JSON state persistence.
"""

import anthropic
from state_manager import StateManager


client = anthropic.Anthropic()


def complete_task(task: str) -> str:
    """Send task to Claude and get response."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {"role": "user", "content": task}
        ]
    )
    return response.content[0].text


def main():
    # Initialize state manager
    state = StateManager("state.json")

    # If no tasks exist, load from file
    if len(state.tasks) == 0:
        print("No tasks found. Loading from tasks.txt...")
        state.load_from_file("tasks.txt")

    # Show current progress
    stats = state.get_stats()
    print(f"\nProgress: {stats['completed']}/{stats['total']} completed "
          f"({stats['progress_percent']:.0f}%)")

    if not state.has_pending():
        print("\nAll tasks completed! Nothing to do.")
        return

    print(f"\n{stats['pending']} tasks remaining. Starting loop...\n")

    # Main loop - process pending tasks
    while state.has_pending():
        task = state.get_next()

        print(f"[Task {task['id']}] {task['description'][:40]}...")

        # Mark as in progress
        state.mark_in_progress(task["id"])

        # Call Claude
        result = complete_task(task["description"])

        # Mark complete and save
        state.mark_complete(task["id"], result)

        stats = state.get_stats()
        print(f"  ✓ Completed ({stats['completed']}/{stats['total']})\n")

    print("All tasks completed!")
    print(f"State saved to: {state.filepath}")


if __name__ == "__main__":
    main()
```

## Step 4: Test Resume Capability

Create `tasks.txt`:
```
Write a haiku about persistence
Explain why state machines are useful
Give a fun fact about JSON
```

### Test 1: Fresh Run

```bash
python loop_with_state.py
```

Output:
```
No existing state found, starting fresh
No tasks found. Loading from tasks.txt...

Progress: 0/3 completed (0%)

3 tasks remaining. Starting loop...

[Task 1] Write a haiku about persistence...
  ✓ Completed (1/3)

[Task 2] Explain why state machines are useful...
  ✓ Completed (2/3)

[Task 3] Give a fun fact about JSON...
  ✓ Completed (3/3)

All tasks completed!
```

### Test 2: Simulate Crash (Ctrl+C mid-run)

Reset and run again, interrupt after first task:

```bash
rm state.json  # Reset
python loop_with_state.py
# Press Ctrl+C after "Task 1 completed"
```

### Test 3: Resume

Run again - it picks up where it left off:

```bash
python loop_with_state.py
```

Output:
```
Loaded 3 tasks from state.json

Progress: 1/3 completed (33%)

2 tasks remaining. Starting loop...

[Task 2] Explain why state machines are useful...
  ✓ Completed (2/3)
...
```

**The loop resumed from task 2!**

---

## Understanding the Code

### Why External State Matters

| Without External State | With External State |
|------------------------|---------------------|
| Crash = lost progress | Crash = resume from last save |
| No visibility | Can inspect state.json |
| No debugging | Can manually fix stuck tasks |
| Single run only | Reliable multi-run workflows |

### The Save-After-Every-Task Pattern

```python
state.mark_complete(task["id"], result)  # This calls save()
```

We save after **every** task completion. This costs a tiny bit of I/O but ensures:
- Maximum 1 task of lost work on crash
- Always-consistent state file
- Real-time progress visibility

### State File as Debug Tool

You can manually inspect and edit `state.json`:

```bash
# Check progress
cat state.json | jq '.tasks[] | select(.status == "pending")'

# Reset a stuck task
cat state.json | jq '.tasks[1].status = "pending"' > temp.json && mv temp.json state.json
```

---

## Exercises

### Exercise 1: Add Timestamps

Extend the task schema to track when tasks were started and completed:

```python
{
    "id": 1,
    "description": "...",
    "status": "completed",
    "result": "...",
    "started_at": "2026-01-19T10:30:00Z",
    "completed_at": "2026-01-19T10:30:05Z"
}
```

### Exercise 2: Add a Reset Command

Add a method to reset all tasks to pending:

```python
def reset_all(self):
    """Reset all tasks to pending status."""
    # Your code here
```

### Exercise 3: Progress Report

Create a script that reads `state.json` and prints a formatted progress report without running any tasks.

---

## Checkpoint

Before moving on, verify:
- [ ] Your script resumes correctly after interruption
- [ ] `state.json` correctly reflects task status
- [ ] You understand why external state enables reliability
- [ ] You can manually inspect the state file

---

## Key Takeaway

> External state enables reliability and debugging.

In-memory state is fast but fragile. External state trades a tiny bit of speed for:
- **Reliability**: Resume after any crash
- **Visibility**: Inspect progress anytime
- **Debugging**: Manually fix issues
- **Auditability**: Track what happened

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

The Tier 1 implementation extends this pattern with verification, retries, and circuit breakers.

---

<div class="lab-navigation">
  <a href="./01-first-loop.md" class="prev">← Previous: Lab 01 - Your First Loop</a>
  <a href="./03-verification.md" class="next">Next: Lab 03 - Simple Verification →</a>
</div>
