---
layout: lab
title: "Lab 01: Your First Loop"
lab_number: 1
difficulty: beginner
time: 30 minutes
prerequisites: Python basics, Anthropic API key
---

# Lab 01: Your First Loop

Build a basic loop that reads tasks, calls Claude, and saves results.

## Objectives

By the end of this lab, you will:
- Understand the basic loop concept
- Run a simple AI-powered loop
- See how loops enable persistence

## Prerequisites

- Python 3.8+
- Anthropic API key (`ANTHROPIC_API_KEY` environment variable)

## The Core Concept

Every AI orchestration system starts here:

```python
while tasks_remain:
    task = get_next_task()
    result = ai.complete(task)
    save_result(result)
```

This simple pattern is **surprisingly powerful**. Let's build it.

---

## Step 1: Set Up Your Environment

```bash
# Create a new directory
mkdir first-loop && cd first-loop

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install anthropic
```

## Step 2: Create Your Tasks File

Create `tasks.txt`:

```
Write a haiku about loops
Explain why recursion is like a mirror
Tell me a fun fact about persistence
```

## Step 3: Write the Loop

Create `loop.py`:

```python
"""
Your First Loop - Lab 01

This script demonstrates the core loop pattern:
1. Read tasks from a file
2. Call Claude to complete each task
3. Save results to an output file
4. Loop until all tasks are done
"""

import anthropic

# Initialize the Claude client
client = anthropic.Anthropic()


def read_tasks(filepath: str) -> list[str]:
    """Read tasks from a file, one per line."""
    with open(filepath, "r") as f:
        tasks = [line.strip() for line in f if line.strip()]
    return tasks


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


def save_result(task: str, result: str, filepath: str):
    """Append result to output file."""
    with open(filepath, "a") as f:
        f.write(f"## Task: {task}\n\n")
        f.write(f"{result}\n\n")
        f.write("-" * 50 + "\n\n")


def main():
    """The main loop."""
    tasks = read_tasks("tasks.txt")
    total = len(tasks)
    completed = 0

    print(f"Starting loop with {total} tasks...\n")

    for task in tasks:
        print(f"Processing: {task[:50]}...")

        # Call Claude
        result = complete_task(task)

        # Save result
        save_result(task, result, "results.txt")

        completed += 1
        print(f"  ✓ Completed ({completed}/{total})\n")

    print(f"Done! {completed} tasks completed.")
    print("Results saved to results.txt")


if __name__ == "__main__":
    main()
```

## Step 4: Run the Loop

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
python loop.py
```

Expected output:
```
Starting loop with 3 tasks...

Processing: Write a haiku about loops...
  ✓ Completed (1/3)

Processing: Explain why recursion is like a mirror...
  ✓ Completed (2/3)

Processing: Tell me a fun fact about persistence...
  ✓ Completed (3/3)

Done! 3 tasks completed.
Results saved to results.txt
```

---

## Understanding the Code

### The Loop Pattern

```python
for task in tasks:          # Loop through all tasks
    result = complete(task)  # Execute each one
    save_result(result)      # Persist the result
```

This is the foundation. Every sophisticated orchestration system builds on this.

### Why This Works

1. **Persistence**: Even if one task fails, others can complete
2. **Visibility**: We can see progress as it happens
3. **Simplicity**: Easy to understand and debug

### What's Missing?

This basic loop doesn't handle:
- Task failures (what if Claude returns an error?)
- Verification (is the result actually good?)
- State persistence (what if the script crashes?)

We'll add these in future labs.

---

## Exercises

### Exercise 1: Add Task Numbers
Modify the output to include task numbers:
```
## Task 1 of 3: Write a haiku...
```

### Exercise 2: Error Handling
Wrap `complete_task()` in a try/except and handle errors gracefully.

### Exercise 3: Custom Model
Add a command-line argument to specify the Claude model.

---

## Checkpoint

Before moving on, verify:
- [ ] Your script runs without errors
- [ ] `results.txt` contains all task results
- [ ] You understand the basic loop pattern

---

## Key Takeaway

> Loops let imperfect AI succeed through persistence.

A single Claude call might not give the perfect answer. But a loop that retries, verifies, and adapts **will eventually succeed**.

---

## Get the Code

Full implementation: [8me/src/tier0-hello-world/](https://github.com/fbratten/8me/tree/main/src/tier0-hello-world)

---

<div class="lab-navigation">
  <span class="prev">← Previous: None (this is Lab 01)</span>
  <a href="./02-external-state.md" class="next">Next: Lab 02 - External State →</a>
</div>
