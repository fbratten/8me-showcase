---
layout: page
title: Circuit Breakers
permalink: /concepts/circuit-breakers/
parent: Concepts
---

# Circuit Breakers

Safety mechanisms to prevent runaway loops.

---

## Why Circuit Breakers?

Without limits, a loop can:
- Run forever (infinite loop)
- Burn through API credits
- Overwhelm external services
- Mask underlying bugs

Circuit breakers **fail fast** when something is wrong.

---

## Types of Circuit Breakers

### 1. Attempt Limits

```python
MAX_ATTEMPTS = 5

for attempt in range(MAX_ATTEMPTS):
    result = try_task()
    if verify(result):
        return result

raise CircuitBreakerTripped("Max attempts exceeded")
```

### 2. Time Limits

```python
import time

TIMEOUT_SECONDS = 300  # 5 minutes
start_time = time.time()

while True:
    if time.time() - start_time > TIMEOUT_SECONDS:
        raise CircuitBreakerTripped("Timeout exceeded")

    result = try_task()
    if verify(result):
        return result
```

### 3. Cost Limits

```python
MAX_TOKENS = 100_000
total_tokens = 0

while True:
    result = try_task()
    total_tokens += result.usage.total_tokens

    if total_tokens > MAX_TOKENS:
        raise CircuitBreakerTripped("Token budget exceeded")

    if verify(result):
        return result
```

### 4. Error Rate Limits

```python
from collections import deque

ERROR_THRESHOLD = 0.8
WINDOW_SIZE = 10

recent_results = deque(maxlen=WINDOW_SIZE)

while True:
    result = try_task()
    recent_results.append(result.success)

    error_rate = 1 - (sum(recent_results) / len(recent_results))
    if error_rate > ERROR_THRESHOLD:
        raise CircuitBreakerTripped("Error rate too high")
```

---

## Circuit Breaker States

```
     ┌─────────┐
     │  CLOSED │ ←── Normal operation
     └────┬────┘
          │ failures exceed threshold
          ▼
     ┌─────────┐
     │  OPEN   │ ←── Failing fast, not trying
     └────┬────┘
          │ after cooldown period
          ▼
     ┌─────────┐
     │HALF-OPEN│ ←── Testing if recovered
     └────┬────┘
          │ success → CLOSED
          │ failure → OPEN
```

### Implementation

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown=60):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.cooldown:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Circuit is open")

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
```

---

## Combining Circuit Breakers

```python
class CompositeCircuitBreaker:
    def __init__(self):
        self.breakers = [
            AttemptLimitBreaker(max_attempts=10),
            TimeLimitBreaker(timeout_seconds=300),
            TokenBudgetBreaker(max_tokens=50_000),
            ErrorRateBreaker(threshold=0.8, window=10),
        ]

    def check(self, context):
        for breaker in self.breakers:
            breaker.check(context)  # Raises if tripped

    def record(self, result):
        for breaker in self.breakers:
            breaker.record(result)
```

---

## Graceful Degradation

When a circuit breaker trips, don't just crash:

```python
try:
    result = loop_with_breakers(task)
except CircuitBreakerTripped as e:
    # Option 1: Return partial result
    if partial_result:
        return PartialResult(partial_result, reason=str(e))

    # Option 2: Fall back to simpler approach
    return fallback_method(task)

    # Option 3: Queue for later
    retry_queue.add(task, delay=3600)

    # Option 4: Alert and fail
    alert_on_call(f"Circuit breaker tripped: {e}")
    raise
```

---

## Configuration Best Practices

| Parameter | Suggested Start | Adjust Based On |
|-----------|----------------|-----------------|
| Max attempts | 5-10 | Task complexity |
| Timeout | 5-10 minutes | Expected task duration |
| Token budget | 50k-100k | Task verbosity |
| Error threshold | 80% | Acceptable failure rate |

---

## Logging and Monitoring

Always log circuit breaker events:

```python
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def on_trip(self, reason):
        logger.warning(f"Circuit breaker tripped: {reason}")
        metrics.increment("circuit_breaker.trips", tags={"reason": reason})

    def on_reset(self):
        logger.info("Circuit breaker reset")
        metrics.increment("circuit_breaker.resets")
```

---

## Next Steps

- Learn about [State Management](../state-management/)
- See implementation: [Lab 06: Circuit Breakers](../../labs/06-circuit-breakers)

---

<div style="text-align: center;">
  <a href="../">← Back to Concepts</a>
</div>
