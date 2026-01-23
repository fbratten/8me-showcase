---
layout: page
title: The Loop Pattern
permalink: /concepts/loop-pattern/
parent: Concepts
---

# The Loop Pattern

The foundation of all AI agent orchestration.

---

## Core Concept

```
while task not complete:
    1. Read current state
    2. Execute action (call AI)
    3. Verify result
    4. Update state
```

This simple pattern is the backbone of reliable AI automation. It's not sophisticated, but it works.

---

## Why Loops Matter

AI models are **probabilistic**, not deterministic. They:
- Sometimes fail to follow instructions
- Occasionally hallucinate
- May produce incomplete output
- Can misunderstand context

A loop compensates for this by **trying again** until verified success.

---

## The Basic Loop

```python
def basic_loop(task):
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        # 1. Execute
        result = call_ai(task)

        # 2. Verify
        if verify(result):
            return result  # Success!

        # 3. Log and retry
        print(f"Attempt {attempt} failed, retrying...")

    raise Exception("Max attempts exceeded")
```

---

## Key Components

### 1. State Management
Track what's been done and what remains:
```python
state = {
    "task": "Write a haiku",
    "status": "pending",
    "attempts": 0,
    "result": None
}
```

### 2. Execution
Call the AI model with the task:
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": task}]
)
```

### 3. Verification
Confirm the output meets requirements:
```python
def verify(result):
    # Check length, format, content, etc.
    return len(result.split('\n')) == 3  # Haiku has 3 lines
```

### 4. Safety Limits
Prevent infinite loops:
```python
if attempt >= max_attempts:
    raise CircuitBreakerTripped("Too many attempts")
```

---

## Loop Variations

### Sequential Loop
Process tasks one at a time:
```python
for task in task_queue:
    result = loop_until_done(task)
    save_result(result)
```

### Retry with Backoff
Add delays between attempts:
```python
import time

delay = 1  # seconds
for attempt in range(max_attempts):
    result = try_task()
    if verify(result):
        return result
    time.sleep(delay)
    delay *= 2  # Exponential backoff
```

### Conditional Retry
Only retry certain types of failures:
```python
result = try_task()
if result.error_type == "rate_limit":
    time.sleep(60)
    retry()
elif result.error_type == "invalid_output":
    retry_with_clearer_prompt()
else:
    fail_permanently()
```

---

## Anti-Patterns

### No Exit Condition
```python
# BAD: Can loop forever
while True:
    result = try_task()
    if verify(result):
        break
```

### No State Persistence
```python
# BAD: Loses progress on crash
while not done:
    result = try_task()  # If this crashes, start over
```

### Trusting AI Self-Assessment
```python
# BAD: AI says it's done, but is it?
result = ai.complete_task()
if result.says_complete:  # Don't trust this alone
    return result
```

---

## Best Practices

1. **Always have a max attempts limit**
2. **Persist state externally** (file, database)
3. **Use external verification** when possible
4. **Log every attempt** for debugging
5. **Implement graceful degradation**

---

## Next Steps

- Learn about [Verification Strategies](../verification/)
- Understand [Circuit Breakers](../circuit-breakers/)
- See it in action: [Lab 01: Your First Loop](../../labs/01-first-loop)

---

<div style="text-align: center;">
  <a href="../">← Back to Concepts</a>
</div>
