---
layout: default
title: Tier 1 CLI Usage
nav_order: 5
parent: Tutorials
permalink: /tutorials/tier1-cli/
---

# Tier 1: CLI Usage Guide

Command-line interface for the full Ralph Loop.

---

## Basic Usage

```bash
cd src/tier1-ralph-loop
python ralph_loop.py
```

---

## Command Options

```bash
# Custom task file
python ralph_loop.py --tasks my_tasks.json

# Add a task and run
python ralph_loop.py --add "Write a function to calculate factorial"

# Set confidence threshold (default: 0.7)
python ralph_loop.py --confidence 0.8

# Set max iterations (default: 50)
python ralph_loop.py --max-iterations 100

# Quiet mode
python ralph_loop.py --quiet

# Show stats only
python ralph_loop.py --stats-only
```

---

## Adding Tasks

### Via CLI

```bash
python ralph_loop.py --add "Explain quicksort"
```

### Via Python

```python
from task_manager import TaskManager

tm = TaskManager("tasks.json")
tm.add_task("Write a haiku about coding")
```

---

## Viewing Status

```bash
python ralph_loop.py --stats-only
```

Output:
```
Task Statistics:
  Total: 5
  Pending: 2
  In Progress: 1
  Completed: 2
  Failed: 0
```

---

## Example Session

```bash
$ python ralph_loop.py --add "Write a limerick about Python"
Added task: Write a limerick about Python

$ python ralph_loop.py
=== Starting Ralph Loop ===
Processing: Write a limerick about Python
...
Confidence: 0.95 - Accepted!
=== Complete: 1 done, 0 failed ===
```

---

## Next Steps

- [Task Management](../tier1-tasks/)
- [Safety Features](../tier1-safety/)

---

<div style="text-align: center;">
  <a href="../">← Back to Tutorials</a>
</div>
