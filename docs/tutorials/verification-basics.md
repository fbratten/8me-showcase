---
layout: default
title: Understanding Verification
nav_order: 1
parent: Tutorials
permalink: /tutorials/verification-basics/
---

# Understanding Verification

Why verification matters and how to implement it.

---

## The Problem

AI models are probabilistic. They might:
- Claim success when they failed
- Produce plausible-looking but wrong output
- Miss edge cases

**Never trust AI self-assessment alone.**

---

## Verification Types

### 1. Structural Verification

Check the format:

```python
def verify_json(output):
    try:
        data = json.loads(output)
        return "required_field" in data
    except:
        return False
```

### 2. Confidence-Based

Use AI's confidence score:

```python
if outcome.confidence >= 0.7:
    accept(result)
else:
    retry()
```

### 3. External Verification

Use tools to verify:

```python
def verify_code(code):
    result = subprocess.run(['python', '-c', code])
    return result.returncode == 0
```

---

## In 8me

Tier 1 uses confidence-based verification:

```python
outcome = executor.execute_task(task)

if isinstance(outcome, TaskCompletion):
    if outcome.confidence >= min_confidence:
        manager.mark_completed(task.id, outcome.result)
    else:
        # Retry - confidence too low
        task.attempts += 1
```

---

## Next Steps

- [Lab 03: Simple Verification](../../labs/03-verification)
- [Verification Concepts](../concepts/verification)

---

<div style="text-align: center;">
  <a href="../">← Back to Tutorials</a>
</div>
