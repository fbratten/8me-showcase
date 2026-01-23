---
layout: page
title: Tier 1 Safety Features
permalink: /tutorials/tier1-safety/
parent: Tutorials
---

# Tier 1: Circuit Breakers

Safety mechanisms to prevent infinite loops.

---

## Why Circuit Breakers?

Without limits, loops can:
- Run forever
- Burn API credits
- Mask bugs

---

## Configuration

```python
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    max_iterations=50,      # Hard stop at 50 loops
    max_task_attempts=3,    # Max 3 tries per task
    similarity_threshold=3  # Stop if same error 3x
)
breaker = CircuitBreaker(config)
```

---

## Trip Conditions

### 1. Max Iterations

```python
if breaker.iterations >= config.max_iterations:
    raise CircuitBreakerTripped("Max iterations reached")
```

### 2. Repeated Errors

```python
if same_error_count >= config.similarity_threshold:
    raise CircuitBreakerTripped("Repeated error detected")
```

### 3. Repeated Output

```python
if output_hash in recent_outputs:
    raise CircuitBreakerTripped("Identical output detected")
```

---

## Usage in Loop

```python
while not breaker.tripped():
    breaker.record_iteration()

    task = manager.get_next_pending()
    if not task:
        break

    result = executor.execute(task)
    breaker.record_output(result)

    if result.error:
        breaker.record_error(result.error)
```

---

## Best Practices

1. **Always set limits** - Never run unlimited
2. **Log breaker events** - Debug why it tripped
3. **Tune thresholds** - Adjust based on task type

---

## Next Steps

- [Lab 06: Circuit Breakers](../../labs/06-circuit-breakers)
- [Circuit Breaker Concepts](../concepts/circuit-breakers)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
