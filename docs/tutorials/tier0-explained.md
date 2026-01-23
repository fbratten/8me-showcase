---
layout: page
title: Tier 0 Explained
parent: Tutorials
---

# Tier 0: Understanding the Code

Line-by-line explanation of the Hello World loop.

---

## The Core Loop

```python
while True:
    task = find_next_task()
    if task is None:
        break  # All done!

    response = ask_claude(task)
    print(response)
    mark_task_done(task)
```

That's the essence. Everything else is details.

---

## Key Functions

### `read_tasks()`

```python
def read_tasks(filepath):
    with open(filepath, 'r') as f:
        return f.read().splitlines()
```

Loads the task file into memory.

---

### `find_next_task()`

```python
def find_next_task(lines):
    for i, line in enumerate(lines):
        if line.startswith('#') or line.startswith('[DONE]'):
            continue
        if line.strip():
            return i, line
    return None, None
```

Finds first line that's:
- Not a comment (`#`)
- Not already done (`[DONE]`)
- Not empty

---

### `ask_claude()`

```python
def ask_claude(task):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": task}]
    )
    return message.content[0].text
```

Simple API call - send task, get response.

---

### `mark_task_done()`

```python
def mark_task_done(lines, index, filepath):
    lines[index] = f"[DONE] {lines[index]}"
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
```

Prepends `[DONE]` and saves the file.

---

## What's Missing (Intentionally)

- ❌ Error handling
- ❌ Verification
- ❌ Retry logic
- ❌ Circuit breakers

These are covered in Tier 1.

---

## The Philosophy

```
Success = Imperfect AI × Persistent Loop × External Verification
```

Tier 0 has the loop. Tier 1 adds verification.

---

## Next Steps

- [Tier 1: Full MVP](./tier1-cli)
- [Lab 01: Build Your Own](../../labs/01-first-loop)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
