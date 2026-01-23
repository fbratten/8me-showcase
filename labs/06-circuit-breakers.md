---
layout: lab
title: "Lab 06: Circuit Breakers"
lab_number: 6
difficulty: intermediate
time: 45 minutes
prerequisites: Lab 05 completed
---

# Lab 06: Circuit Breakers

Prevent infinite loops and runaway costs with safety limits.

## Objectives

By the end of this lab, you will:
- Understand why circuit breakers are essential
- Implement iteration and time limits
- Detect repeated failures and stuck loops
- Track costs and enforce budgets

## Prerequisites

- Lab 05 completed (tool calling)
- Understanding of the loop pattern

## The Problem: Infinite Loops

Without safety limits, loops can run forever:

```python
# Dangerous: No exit condition for edge cases
while has_pending_tasks():
    task = get_next()
    result = execute(task)

    if not verify(result):
        retry(task)  # What if it NEVER passes?
```

Real failure modes:
- Task is **impossible** (AI keeps trying forever)
- Task is **ambiguous** (different wrong answers each time)
- Verification is **too strict** (nothing ever passes)
- API is **failing** (same error repeated)
- Output is **looping** (same response over and over)

**Circuit breakers detect these patterns and stop the loop.**

---

## The Circuit Breaker Pattern

```python
class CircuitBreaker:
    def check(self) -> Tuple[bool, str]:
        """
        Check if the loop should continue.

        Returns:
            (True, "OK") if safe to continue
            (False, "reason") if loop should stop
        """
        if self.iterations > self.max_iterations:
            return False, "Max iterations reached"

        if self.consecutive_failures > self.max_consecutive_failures:
            return False, "Too many consecutive failures"

        if self.is_output_repeating():
            return False, "Output repetition detected"

        if self.estimated_cost > self.max_cost:
            return False, "Cost limit exceeded"

        return True, "OK"
```

---

## Step 1: Create the Circuit Breaker

Create `circuit_breaker.py`:

```python
"""
Circuit Breaker - Lab 06

Safety mechanisms to prevent infinite loops and runaway costs.
"""

import time
import hashlib
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker limits."""
    max_iterations: int = 100
    max_consecutive_failures: int = 5
    max_time_seconds: int = 300  # 5 minutes
    max_cost_dollars: float = 1.0
    repetition_window: int = 5
    repetition_threshold: int = 3  # Same output 3 times = stuck


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    iterations: int = 0
    consecutive_failures: int = 0
    total_failures: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_cost: float = 0.0
    recent_outputs: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=10))
    trip_reason: Optional[str] = None
    is_tripped: bool = False


class CircuitBreaker:
    """
    Safety mechanism to prevent infinite loops.

    Usage:
        breaker = CircuitBreaker()

        while breaker.allow_continue():
            result = process_task()

            if success:
                breaker.record_success(result, tokens_used=500)
            else:
                breaker.record_failure(error_message)

        if breaker.is_tripped:
            print(f"Circuit breaker tripped: {breaker.trip_reason}")
    """

    # Approximate cost per 1K tokens (adjust based on model)
    COST_PER_1K_INPUT = 0.003   # Claude Sonnet input
    COST_PER_1K_OUTPUT = 0.015  # Claude Sonnet output

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()

    def allow_continue(self) -> bool:
        """Check if the loop should continue."""
        ok, reason = self.check()
        if not ok:
            self.state.is_tripped = True
            self.state.trip_reason = reason
        return ok

    def check(self) -> Tuple[bool, str]:
        """
        Run all safety checks.

        Returns:
            (True, "OK") if safe to continue
            (False, "reason") if should stop
        """
        # Check iteration limit
        if self.state.iterations >= self.config.max_iterations:
            return False, f"Max iterations ({self.config.max_iterations}) reached"

        # Check consecutive failures
        if self.state.consecutive_failures >= self.config.max_consecutive_failures:
            return False, f"Too many consecutive failures ({self.state.consecutive_failures})"

        # Check time limit
        elapsed = time.time() - self.state.start_time
        if elapsed >= self.config.max_time_seconds:
            return False, f"Time limit ({self.config.max_time_seconds}s) exceeded"

        # Check cost limit
        if self.state.estimated_cost >= self.config.max_cost_dollars:
            return False, f"Cost limit (${self.config.max_cost_dollars:.2f}) exceeded"

        # Check for output repetition (stuck loop)
        if self._is_output_repeating():
            return False, "Output repetition detected (loop appears stuck)"

        # Check for error repetition (same error repeatedly)
        if self._is_error_repeating():
            return False, "Same error repeating (underlying issue not resolving)"

        return True, "OK"

    def record_success(self, output: str, input_tokens: int = 0, output_tokens: int = 0):
        """Record a successful iteration."""
        self.state.iterations += 1
        self.state.consecutive_failures = 0

        # Track output for repetition detection
        output_hash = self._hash_output(output)
        self.state.recent_outputs.append(output_hash)

        # Track cost
        self._add_cost(input_tokens, output_tokens)

    def record_failure(self, error: str, input_tokens: int = 0, output_tokens: int = 0):
        """Record a failed iteration."""
        self.state.iterations += 1
        self.state.consecutive_failures += 1
        self.state.total_failures += 1

        # Track error for repetition detection
        error_hash = self._hash_output(error)
        self.state.recent_errors.append(error_hash)

        # Track cost (failures still cost money!)
        self._add_cost(input_tokens, output_tokens)

    def record_skip(self):
        """Record a skipped iteration (no API call)."""
        self.state.iterations += 1

    def reset(self):
        """Reset the circuit breaker for a new run."""
        self.state = CircuitBreakerState()

    def _hash_output(self, text: str) -> str:
        """Create a hash of output for comparison."""
        # Normalize whitespace and case for comparison
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _is_output_repeating(self) -> bool:
        """Check if recent outputs are repeating."""
        if len(self.state.recent_outputs) < self.config.repetition_threshold:
            return False

        recent = list(self.state.recent_outputs)[-self.config.repetition_window:]

        # Check if any hash appears too many times
        for output_hash in set(recent):
            if recent.count(output_hash) >= self.config.repetition_threshold:
                return True

        return False

    def _is_error_repeating(self) -> bool:
        """Check if same error is repeating."""
        if len(self.state.recent_errors) < self.config.repetition_threshold:
            return False

        recent = list(self.state.recent_errors)[-self.config.repetition_window:]

        for error_hash in set(recent):
            if recent.count(error_hash) >= self.config.repetition_threshold:
                return True

        return False

    def _add_cost(self, input_tokens: int, output_tokens: int):
        """Add to estimated cost."""
        input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT
        output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT
        self.state.estimated_cost += input_cost + output_cost

    # ==================== Properties ====================

    @property
    def is_tripped(self) -> bool:
        """Check if breaker has tripped."""
        return self.state.is_tripped

    @property
    def trip_reason(self) -> Optional[str]:
        """Get the reason the breaker tripped."""
        return self.state.trip_reason

    @property
    def iterations(self) -> int:
        """Get current iteration count."""
        return self.state.iterations

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.state.start_time

    @property
    def estimated_cost(self) -> float:
        """Get estimated cost so far."""
        return self.state.estimated_cost

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "iterations": self.state.iterations,
            "consecutive_failures": self.state.consecutive_failures,
            "total_failures": self.state.total_failures,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "estimated_cost": round(self.state.estimated_cost, 4),
            "is_tripped": self.state.is_tripped,
            "trip_reason": self.state.trip_reason
        }
```

---

## Step 2: Create Specialized Breakers

Add these to `circuit_breaker.py`:

```python
class TaskCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker specialized for per-task limits.

    Tracks limits per task, not just globally.
    """

    def __init__(
        self,
        max_attempts_per_task: int = 3,
        config: Optional[CircuitBreakerConfig] = None
    ):
        super().__init__(config)
        self.max_attempts_per_task = max_attempts_per_task
        self.task_attempts: dict = {}

    def check_task(self, task_id: str) -> Tuple[bool, str]:
        """Check if a specific task should be retried."""
        attempts = self.task_attempts.get(task_id, 0)

        if attempts >= self.max_attempts_per_task:
            return False, f"Task {task_id} exceeded max attempts ({self.max_attempts_per_task})"

        return True, "OK"

    def record_task_attempt(self, task_id: str):
        """Record an attempt for a specific task."""
        self.task_attempts[task_id] = self.task_attempts.get(task_id, 0) + 1

    def get_task_attempts(self, task_id: str) -> int:
        """Get attempt count for a task."""
        return self.task_attempts.get(task_id, 0)


class CostAwareCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker with detailed cost tracking and alerts.
    """

    def __init__(
        self,
        budget_dollars: float = 1.0,
        alert_threshold: float = 0.8,  # Alert at 80% of budget
        config: Optional[CircuitBreakerConfig] = None
    ):
        config = config or CircuitBreakerConfig()
        config.max_cost_dollars = budget_dollars
        super().__init__(config)

        self.alert_threshold = alert_threshold
        self.alert_triggered = False

    def check_budget_alert(self) -> Optional[str]:
        """Check if we're approaching budget limit."""
        if self.state.estimated_cost >= self.config.max_cost_dollars * self.alert_threshold:
            if not self.alert_triggered:
                self.alert_triggered = True
                percent = (self.state.estimated_cost / self.config.max_cost_dollars) * 100
                return f"WARNING: {percent:.0f}% of budget used (${self.state.estimated_cost:.4f})"
        return None

    def get_remaining_budget(self) -> float:
        """Get remaining budget in dollars."""
        return max(0, self.config.max_cost_dollars - self.state.estimated_cost)
```

---

## Step 3: Integrate with the Loop

Create `loop_with_breaker.py`:

```python
"""
Loop with Circuit Breaker - Lab 06

Demonstrates safe loop execution with circuit breakers.
"""

from task_manager import TaskManager
from executor import execute_task, verify_result
from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    TaskCircuitBreaker
)


def main():
    manager = TaskManager("tasks.json")

    # Configure circuit breaker
    config = CircuitBreakerConfig(
        max_iterations=50,
        max_consecutive_failures=3,
        max_time_seconds=120,  # 2 minutes
        max_cost_dollars=0.50,
        repetition_threshold=3
    )

    breaker = TaskCircuitBreaker(
        max_attempts_per_task=3,
        config=config
    )

    # Create sample tasks if needed
    if not manager.tasks:
        manager.create("Write a haiku about safety")
        manager.create("Explain circuit breakers in one sentence")
        manager.create("This task is impossible: divide by zero mentally")

    print(f"Starting loop with circuit breaker...")
    print(f"Limits: {config.max_iterations} iterations, "
          f"{config.max_time_seconds}s, ${config.max_cost_dollars}\n")

    # Main loop with circuit breaker
    while manager.has_pending() and breaker.allow_continue():
        task = manager.get_next()

        # Check per-task limit
        task_ok, task_reason = breaker.check_task(task.id)
        if not task_ok:
            print(f"[{task.id}] {task_reason}")
            manager.fail(task.id, task_reason)
            continue

        breaker.record_task_attempt(task.id)

        print(f"[{task.id}] Attempt {breaker.get_task_attempts(task.id)}")
        print(f"  Task: {task.description[:40]}...")

        manager.start(task.id)

        # Execute
        result = execute_task(task.to_dict())

        # Estimate tokens (simplified)
        input_tokens = len(task.description.split()) * 2
        output_tokens = len(result.get("result", "").split()) * 2

        if result["status"] == "completed":
            breaker.record_success(
                result["result"],
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            print(f"  ✓ Completed (confidence: {result['confidence']:.0%})")
            manager.complete(task.id, result["result"])

        elif result["status"] == "failed":
            breaker.record_failure(
                result["reason"],
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            print(f"  ✗ Failed: {result['reason']}")

            # Check if we should retry or give up on this task
            if breaker.get_task_attempts(task.id) < 3:
                manager.retry(task.id, result["reason"])
            else:
                manager.fail(task.id, result["reason"])

        else:
            breaker.record_failure("Unknown status")
            manager.fail(task.id, "Unknown execution status")

        # Progress report
        stats = breaker.get_stats()
        print(f"  [Breaker: {stats['iterations']} iter, "
              f"${stats['estimated_cost']:.4f}, "
              f"{stats['elapsed_seconds']}s]\n")

    # Final report
    print("=" * 60)

    if breaker.is_tripped:
        print(f"⚠️  CIRCUIT BREAKER TRIPPED: {breaker.trip_reason}")
    else:
        print("✓ Loop completed normally")

    print("\nCircuit Breaker Stats:")
    for key, value in breaker.get_stats().items():
        print(f"  {key}: {value}")

    print("\nTask Stats:")
    task_stats = manager.get_stats()
    print(f"  Completed: {task_stats['completed']}")
    print(f"  Failed: {task_stats['failed']}")
    print(f"  Pending: {task_stats['pending']}")


if __name__ == "__main__":
    main()
```

---

## Step 4: Test Circuit Breaker Scenarios

Create `test_breaker.py`:

```python
"""
Test Circuit Breaker Scenarios - Lab 06
"""

from circuit_breaker import CircuitBreaker, CircuitBreakerConfig


def test_iteration_limit():
    """Test that iteration limit trips the breaker."""
    config = CircuitBreakerConfig(max_iterations=5)
    breaker = CircuitBreaker(config)

    for i in range(10):
        if not breaker.allow_continue():
            print(f"Tripped at iteration {i}: {breaker.trip_reason}")
            break
        breaker.record_success(f"output {i}")

    assert breaker.is_tripped
    assert "iterations" in breaker.trip_reason.lower()
    print("✓ Iteration limit test passed")


def test_consecutive_failures():
    """Test that consecutive failures trip the breaker."""
    config = CircuitBreakerConfig(max_consecutive_failures=3, max_iterations=100)
    breaker = CircuitBreaker(config)

    for i in range(10):
        if not breaker.allow_continue():
            print(f"Tripped at iteration {i}: {breaker.trip_reason}")
            break
        breaker.record_failure("same error")

    assert breaker.is_tripped
    assert "consecutive" in breaker.trip_reason.lower()
    print("✓ Consecutive failures test passed")


def test_output_repetition():
    """Test that repeated outputs trip the breaker."""
    config = CircuitBreakerConfig(
        repetition_threshold=3,
        max_iterations=100
    )
    breaker = CircuitBreaker(config)

    for i in range(10):
        if not breaker.allow_continue():
            print(f"Tripped at iteration {i}: {breaker.trip_reason}")
            break
        breaker.record_success("same output every time")

    assert breaker.is_tripped
    assert "repetition" in breaker.trip_reason.lower()
    print("✓ Output repetition test passed")


def test_cost_limit():
    """Test that cost limit trips the breaker."""
    config = CircuitBreakerConfig(max_cost_dollars=0.01, max_iterations=1000)
    breaker = CircuitBreaker(config)

    for i in range(100):
        if not breaker.allow_continue():
            print(f"Tripped at iteration {i}: {breaker.trip_reason}")
            print(f"Estimated cost: ${breaker.estimated_cost:.4f}")
            break
        # Simulate expensive calls
        breaker.record_success(f"output {i}", input_tokens=500, output_tokens=500)

    assert breaker.is_tripped
    assert "cost" in breaker.trip_reason.lower()
    print("✓ Cost limit test passed")


def test_success_resets_consecutive():
    """Test that success resets consecutive failure count."""
    config = CircuitBreakerConfig(max_consecutive_failures=3, max_iterations=100)
    breaker = CircuitBreaker(config)

    breaker.record_failure("error 1")
    breaker.record_failure("error 2")
    assert breaker.state.consecutive_failures == 2

    breaker.record_success("success!")
    assert breaker.state.consecutive_failures == 0

    print("✓ Consecutive reset test passed")


if __name__ == "__main__":
    test_iteration_limit()
    test_consecutive_failures()
    test_output_repetition()
    test_cost_limit()
    test_success_resets_consecutive()
    print("\n✓ All tests passed!")
```

---

## Understanding Circuit Breakers

### Why Multiple Checks?

| Check | Prevents |
|-------|----------|
| Iteration limit | Loops that run forever |
| Consecutive failures | Stuck on impossible task |
| Time limit | Long-running stuck loops |
| Cost limit | Runaway API spending |
| Output repetition | Loops producing same wrong answer |
| Error repetition | Same underlying error not resolving |

### The Trip Pattern

```
Normal Operation          Approaching Limit         Tripped
      ↓                         ↓                     ↓
┌─────────┐              ┌─────────┐            ┌─────────┐
│ ✓ ✓ ✓ ✓ │   ───────▶   │ ✓ ✓ ✗ ✗ │  ───────▶  │ ⛔ STOP │
│ ✓ ✓ ✓ ✓ │              │ ✗ ✗ ✗   │            │         │
└─────────┘              └─────────┘            └─────────┘
```

### Cost Estimation

```python
# Approximate costs (Claude Sonnet, January 2026)
INPUT_COST = $0.003 / 1K tokens
OUTPUT_COST = $0.015 / 1K tokens

# Example: 500 input + 500 output tokens
cost = (500/1000 * 0.003) + (500/1000 * 0.015)
cost = $0.009 per iteration

# 100 iterations = $0.90
```

---

## Best Practices

### 1. Start Conservative

```python
# Start with tight limits, loosen as needed
config = CircuitBreakerConfig(
    max_iterations=20,      # Start low
    max_cost_dollars=0.10,  # Start cheap
)
```

### 2. Log Trips

```python
if breaker.is_tripped:
    logging.warning(f"Circuit breaker tripped: {breaker.trip_reason}")
    logging.info(f"Stats: {breaker.get_stats()}")
```

### 3. Different Limits for Different Tasks

```python
# Simple tasks: tight limits
simple_config = CircuitBreakerConfig(max_iterations=10)

# Complex tasks: looser limits
complex_config = CircuitBreakerConfig(max_iterations=50, max_cost_dollars=1.0)
```

### 4. Alert Before Trip

```python
if breaker.state.iterations > breaker.config.max_iterations * 0.8:
    print("Warning: Approaching iteration limit")
```

---

## Exercises

### Exercise 1: Add Rate Limiting

Add requests-per-minute limiting:

```python
def check_rate_limit(self) -> Tuple[bool, str]:
    """Ensure we don't exceed API rate limits."""
    # Track requests per minute
    # Return False if too many recent requests
    pass
```

### Exercise 2: Gradual Backoff

Implement exponential backoff on failures:

```python
def get_backoff_seconds(self) -> float:
    """Calculate backoff based on consecutive failures."""
    return min(2 ** self.state.consecutive_failures, 60)
```

### Exercise 3: Circuit Breaker Dashboard

Create a simple dashboard showing real-time breaker stats:

```python
def print_dashboard(breaker: CircuitBreaker):
    """Print a visual dashboard of breaker status."""
    pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Circuit breaker trips on iteration limit
- [ ] Circuit breaker trips on consecutive failures
- [ ] Output repetition is detected
- [ ] Cost tracking works correctly
- [ ] You understand when each check is useful

---

## Key Takeaway

> Safety limits prevent runaway costs and infinite loops.

Circuit breakers are **non-negotiable** in production:
- **Protect your budget** from runaway API costs
- **Prevent infinite loops** from stuck tasks
- **Detect stuck states** via repetition detection
- **Provide visibility** into loop health
- **Enable debugging** with detailed stats

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/circuit_breaker.py](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./05-tool-calling" class="prev">← Previous: Lab 05 - Tool Calling</a>
  <a href="./07-confidence-retry" class="next">Next: Lab 07 - Confidence Retry →</a>
</div>
