---
layout: default
title: "Lab 07: Confidence Retry"
nav_order: 7
parent: Labs
lab_number: 7
difficulty: intermediate
time: 1 hour
prerequisites: Lab 06 completed
---

# Lab 07: Confidence Retry

Use AI confidence scores for smart retry decisions that balance quality and cost.

## Objectives

By the end of this lab, you will:
- Understand confidence-based decision making
- Implement adaptive retry thresholds
- Balance quality vs cost tradeoffs
- Track and optimize retry behavior

## Prerequisites

- Lab 06 completed (circuit breakers)
- Understanding of tool calling (Lab 05)

## The Problem: Fixed Thresholds

Simple retry logic uses fixed thresholds:

```python
# Fixed threshold - too rigid
if confidence >= 0.9:
    accept()
else:
    retry()  # Even at 0.89? Even after 5 attempts?
```

Problems with this approach:
- **0.89 vs 0.90**: Arbitrary cutoff rejects good results
- **Diminishing returns**: 5th retry rarely beats 4th
- **Cost blindness**: Retries cost money regardless of improvement chance
- **No context**: Same threshold for easy and hard tasks

## The Solution: Adaptive Thresholds

```python
# Adaptive - smarter decisions
if confidence >= 0.95:
    accept()  # High confidence = accept immediately
elif confidence >= 0.80 and attempts >= 2:
    accept()  # Good enough after retries
elif confidence >= 0.70 and attempts >= 3:
    accept()  # Acceptable after many retries
elif attempts >= max_attempts:
    accept_or_fail()  # Decide based on minimum threshold
else:
    retry()  # Try to improve
```

---

## Step 1: Create the Confidence Manager

Create `confidence_manager.py`:

```python
"""
Confidence Manager - Lab 07

Adaptive confidence-based retry decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum


class Decision(Enum):
    """Possible decisions for a result."""
    ACCEPT = "accept"
    RETRY = "retry"
    FAIL = "fail"
    ACCEPT_WITH_WARNING = "accept_with_warning"


@dataclass
class ConfidenceConfig:
    """Configuration for confidence-based decisions."""
    # Immediate accept threshold
    high_confidence: float = 0.95

    # Accept after retries thresholds
    medium_confidence: float = 0.80
    medium_confidence_min_attempts: int = 2

    low_confidence: float = 0.70
    low_confidence_min_attempts: int = 3

    # Minimum acceptable (below = fail)
    minimum_confidence: float = 0.50

    # Maximum attempts before forced decision
    max_attempts: int = 5

    # Cost-aware settings
    cost_per_attempt: float = 0.01  # Estimated $ per retry
    max_cost_per_task: float = 0.10


@dataclass
class AttemptRecord:
    """Record of a single attempt."""
    attempt_number: int
    confidence: float
    result_preview: str
    cost: float = 0.0


@dataclass
class TaskConfidenceState:
    """Tracks confidence history for a task."""
    task_id: str
    attempts: List[AttemptRecord] = field(default_factory=list)
    total_cost: float = 0.0
    final_decision: Optional[Decision] = None
    final_confidence: Optional[float] = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def best_confidence(self) -> float:
        if not self.attempts:
            return 0.0
        return max(a.confidence for a in self.attempts)

    @property
    def latest_confidence(self) -> float:
        if not self.attempts:
            return 0.0
        return self.attempts[-1].confidence

    @property
    def is_improving(self) -> bool:
        """Check if confidence is trending upward."""
        if len(self.attempts) < 2:
            return True  # Assume can improve
        return self.attempts[-1].confidence > self.attempts[-2].confidence


class ConfidenceManager:
    """
    Manages confidence-based retry decisions.

    Usage:
        manager = ConfidenceManager()

        while True:
            result, confidence = execute_task(task)
            decision = manager.evaluate(task_id, confidence, result)

            if decision == Decision.ACCEPT:
                complete_task(result)
                break
            elif decision == Decision.RETRY:
                continue
            else:  # FAIL
                fail_task()
                break
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()
        self.task_states: Dict[str, TaskConfidenceState] = {}

    def evaluate(
        self,
        task_id: str,
        confidence: float,
        result_preview: str = "",
        cost: float = 0.0
    ) -> Tuple[Decision, str]:
        """
        Evaluate whether to accept, retry, or fail.

        Args:
            task_id: Unique task identifier
            confidence: AI's confidence in the result (0-1)
            result_preview: First ~50 chars of result (for logging)
            cost: Cost of this attempt in dollars

        Returns:
            Tuple of (Decision, reason_string)
        """
        # Get or create task state
        if task_id not in self.task_states:
            self.task_states[task_id] = TaskConfidenceState(task_id=task_id)

        state = self.task_states[task_id]

        # Record this attempt
        state.attempts.append(AttemptRecord(
            attempt_number=state.attempt_count + 1,
            confidence=confidence,
            result_preview=result_preview[:50],
            cost=cost
        ))
        state.total_cost += cost

        # Make decision
        decision, reason = self._make_decision(state, confidence)

        # Record final decision if terminal
        if decision in (Decision.ACCEPT, Decision.ACCEPT_WITH_WARNING, Decision.FAIL):
            state.final_decision = decision
            state.final_confidence = confidence

        return decision, reason

    def _make_decision(
        self,
        state: TaskConfidenceState,
        confidence: float
    ) -> Tuple[Decision, str]:
        """Core decision logic."""
        attempts = state.attempt_count
        config = self.config

        # Check 1: High confidence = immediate accept
        if confidence >= config.high_confidence:
            return Decision.ACCEPT, f"High confidence ({confidence:.0%})"

        # Check 2: Medium confidence after some attempts
        if confidence >= config.medium_confidence and attempts >= config.medium_confidence_min_attempts:
            return Decision.ACCEPT, f"Good confidence ({confidence:.0%}) after {attempts} attempts"

        # Check 3: Low confidence after many attempts
        if confidence >= config.low_confidence and attempts >= config.low_confidence_min_attempts:
            return Decision.ACCEPT_WITH_WARNING, f"Acceptable ({confidence:.0%}) after {attempts} attempts"

        # Check 4: Max attempts reached
        if attempts >= config.max_attempts:
            if confidence >= config.minimum_confidence:
                return Decision.ACCEPT_WITH_WARNING, f"Max attempts reached, accepting ({confidence:.0%})"
            else:
                return Decision.FAIL, f"Max attempts reached, confidence too low ({confidence:.0%})"

        # Check 5: Cost limit reached
        if state.total_cost >= config.max_cost_per_task:
            if confidence >= config.minimum_confidence:
                return Decision.ACCEPT_WITH_WARNING, f"Cost limit reached, accepting ({confidence:.0%})"
            else:
                return Decision.FAIL, f"Cost limit reached, confidence too low ({confidence:.0%})"

        # Check 6: Not improving after multiple attempts
        if attempts >= 3 and not state.is_improving:
            if confidence >= config.low_confidence:
                return Decision.ACCEPT_WITH_WARNING, f"Not improving, accepting best ({confidence:.0%})"
            # Continue trying if still below acceptable

        # Default: retry
        return Decision.RETRY, f"Confidence ({confidence:.0%}) below threshold, retrying"

    def get_state(self, task_id: str) -> Optional[TaskConfidenceState]:
        """Get the confidence state for a task."""
        return self.task_states.get(task_id)

    def get_stats(self) -> dict:
        """Get overall statistics."""
        if not self.task_states:
            return {"tasks": 0}

        total_attempts = sum(s.attempt_count for s in self.task_states.values())
        total_cost = sum(s.total_cost for s in self.task_states.values())

        decisions = [s.final_decision for s in self.task_states.values() if s.final_decision]
        accepts = sum(1 for d in decisions if d in (Decision.ACCEPT, Decision.ACCEPT_WITH_WARNING))
        fails = sum(1 for d in decisions if d == Decision.FAIL)

        return {
            "tasks": len(self.task_states),
            "total_attempts": total_attempts,
            "avg_attempts": round(total_attempts / len(self.task_states), 2),
            "total_cost": round(total_cost, 4),
            "accepts": accepts,
            "fails": fails,
            "acceptance_rate": round(accepts / len(decisions) * 100, 1) if decisions else 0
        }

    def reset(self):
        """Reset all state."""
        self.task_states.clear()
```

---

## Step 2: Create Confidence Strategies

Add different strategies to `confidence_manager.py`:

```python
class QualityFirstStrategy(ConfidenceConfig):
    """Prioritize quality over speed/cost."""

    def __init__(self):
        super().__init__(
            high_confidence=0.98,
            medium_confidence=0.90,
            medium_confidence_min_attempts=3,
            low_confidence=0.85,
            low_confidence_min_attempts=4,
            minimum_confidence=0.75,
            max_attempts=7,
            max_cost_per_task=0.25
        )


class CostFirstStrategy(ConfidenceConfig):
    """Prioritize cost over perfect quality."""

    def __init__(self):
        super().__init__(
            high_confidence=0.85,
            medium_confidence=0.70,
            medium_confidence_min_attempts=1,
            low_confidence=0.60,
            low_confidence_min_attempts=2,
            minimum_confidence=0.50,
            max_attempts=3,
            max_cost_per_task=0.05
        )


class BalancedStrategy(ConfidenceConfig):
    """Balance quality and cost (default)."""

    def __init__(self):
        super().__init__(
            high_confidence=0.95,
            medium_confidence=0.80,
            medium_confidence_min_attempts=2,
            low_confidence=0.70,
            low_confidence_min_attempts=3,
            minimum_confidence=0.50,
            max_attempts=5,
            max_cost_per_task=0.10
        )


class TaskTypeStrategy:
    """Select strategy based on task type."""

    STRATEGIES = {
        "code": QualityFirstStrategy(),      # Code needs high accuracy
        "creative": CostFirstStrategy(),     # Creative is subjective
        "factual": QualityFirstStrategy(),   # Facts must be correct
        "summary": BalancedStrategy(),       # Summaries can vary
        "default": BalancedStrategy()
    }

    @classmethod
    def get_config(cls, task_type: str) -> ConfidenceConfig:
        return cls.STRATEGIES.get(task_type, cls.STRATEGIES["default"])
```

---

## Step 3: Integrate with the Loop

Create `loop_with_confidence.py`:

```python
"""
Loop with Confidence Retry - Lab 07

Demonstrates adaptive confidence-based retry decisions.
"""

from task_manager import TaskManager
from executor import execute_task
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from confidence_manager import (
    ConfidenceManager,
    Decision,
    TaskTypeStrategy
)


def get_task_type(task) -> str:
    """Determine task type from criteria or description."""
    criteria_type = task.criteria.get("type", "") if task.criteria else ""

    if criteria_type in ["code", "function", "script"]:
        return "code"
    elif criteria_type in ["haiku", "poem", "story", "creative"]:
        return "creative"
    elif criteria_type in ["fact", "factual", "definition"]:
        return "factual"
    elif criteria_type in ["summary", "overview"]:
        return "summary"

    # Infer from description
    desc_lower = task.description.lower()
    if any(kw in desc_lower for kw in ["write code", "function", "implement"]):
        return "code"
    elif any(kw in desc_lower for kw in ["haiku", "poem", "story", "creative"]):
        return "creative"

    return "default"


def process_task(
    manager: TaskManager,
    task,
    confidence_mgr: ConfidenceManager,
    breaker: CircuitBreaker
) -> bool:
    """Process a task with confidence-based retry."""
    task_id = task.id

    print(f"\n[{task_id}] {task.description[:50]}...")

    # Determine strategy based on task type
    task_type = get_task_type(task)
    print(f"  Task type: {task_type}")

    while True:
        # Check circuit breaker
        if not breaker.allow_continue():
            print(f"  ⚠️ Circuit breaker tripped: {breaker.trip_reason}")
            manager.fail(task_id, f"Circuit breaker: {breaker.trip_reason}")
            return False

        manager.start(task_id)

        # Execute task
        result = execute_task(task.to_dict())

        if result["status"] != "completed":
            print(f"  ✗ Execution failed: {result.get('reason', 'Unknown')}")
            breaker.record_failure(result.get("reason", ""))
            manager.fail(task_id, result.get("reason", "Execution failed"))
            return False

        confidence = result["confidence"]
        output = result["result"]

        # Estimate cost
        cost = 0.01  # Simplified estimate

        # Evaluate with confidence manager
        decision, reason = confidence_mgr.evaluate(
            task_id=task_id,
            confidence=confidence,
            result_preview=output[:50],
            cost=cost
        )

        state = confidence_mgr.get_state(task_id)
        print(f"  Attempt {state.attempt_count}: {confidence:.0%} confidence")
        print(f"  Decision: {decision.value} - {reason}")

        if decision == Decision.ACCEPT:
            print(f"  ✓ Accepted!")
            breaker.record_success(output)
            manager.complete(task_id, output)
            return True

        elif decision == Decision.ACCEPT_WITH_WARNING:
            print(f"  ⚠️ Accepted with warning")
            breaker.record_success(output)
            manager.complete(task_id, output)
            return True

        elif decision == Decision.FAIL:
            print(f"  ✗ Failed")
            breaker.record_failure("Confidence too low")
            manager.fail(task_id, reason)
            return False

        else:  # RETRY
            print(f"  ↻ Retrying...")
            breaker.record_failure("Low confidence")
            manager.retry(task_id, reason)
            # Loop continues


def main():
    manager = TaskManager("tasks.json")

    # Circuit breaker
    breaker = CircuitBreaker(CircuitBreakerConfig(
        max_iterations=50,
        max_consecutive_failures=5
    ))

    # Confidence manager with balanced strategy
    confidence_mgr = ConfidenceManager()

    # Create sample tasks
    if not manager.tasks:
        manager.create(
            "Write a haiku about Python programming",
            criteria={"type": "creative"}
        )
        manager.create(
            "Write a Python function that calculates fibonacci numbers",
            criteria={"type": "code"}
        )
        manager.create(
            "What is the capital of France?",
            criteria={"type": "factual"}
        )
        manager.create(
            "Summarize the benefits of version control in 2 sentences",
            criteria={"type": "summary"}
        )

    print("=" * 60)
    print("CONFIDENCE-BASED RETRY DEMO")
    print("=" * 60)

    while manager.has_pending() and breaker.allow_continue():
        task = manager.get_next()
        process_task(manager, task, confidence_mgr, breaker)

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    conf_stats = confidence_mgr.get_stats()
    print(f"\nConfidence Stats:")
    print(f"  Tasks processed: {conf_stats['tasks']}")
    print(f"  Total attempts: {conf_stats['total_attempts']}")
    print(f"  Avg attempts/task: {conf_stats['avg_attempts']}")
    print(f"  Acceptance rate: {conf_stats['acceptance_rate']}%")
    print(f"  Estimated cost: ${conf_stats['total_cost']:.4f}")

    print(f"\nTask Results:")
    for task in manager.get_all():
        icon = {"completed": "✓", "failed": "✗"}.get(task.status, "?")
        state = confidence_mgr.get_state(task.id)
        attempts = state.attempt_count if state else 0
        conf = f"{state.final_confidence:.0%}" if state and state.final_confidence else "N/A"
        print(f"  {icon} {task.description[:40]}... ({attempts} attempts, {conf})")


if __name__ == "__main__":
    main()
```

---

## Understanding Adaptive Confidence

### The Decision Tree

```
                    ┌─────────────────┐
                    │  New Result     │
                    │  confidence=X   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         X >= 0.95      X >= 0.80      X >= 0.70
              │         attempts≥2     attempts≥3
              │              │              │
              ▼              ▼              ▼
           ACCEPT         ACCEPT        ACCEPT*
                                       (* warning)
              │              │              │
              └──────────────┴──────┬───────┘
                                    │
                                    ▼
                            attempts >= max?
                              /         \
                             /           \
                            ▼             ▼
                     X >= minimum?     RETRY
                       /      \
                      ▼        ▼
                   ACCEPT*   FAIL
```

### Why Adaptive Works

| Fixed Threshold | Adaptive Threshold |
|-----------------|-------------------|
| Rejects 89% confidence | Accepts 89% after 2 attempts |
| Same threshold for all tasks | Different thresholds per task type |
| Ignores attempt count | Lowers bar after retries |
| Ignores cost | Stops when cost limit reached |
| Binary accept/reject | Graduated accept/warn/retry/fail |

### Cost-Quality Tradeoff

```
Quality ▲
        │     ┌─────────────────────┐
   100% │     │  Quality First      │  $$$
        │     │  (code, facts)      │
        │     └─────────────────────┘
        │           ┌───────────────┐
    85% │           │  Balanced     │  $$
        │           │  (default)    │
        │           └───────────────┘
        │                 ┌─────────┐
    70% │                 │  Cost   │  $
        │                 │  First  │
        └─────────────────┴─────────┴──────▶ Cost
```

---

## Exercises

### Exercise 1: Confidence Decay

Implement confidence that decays with time since generation:

```python
def adjust_for_staleness(confidence: float, age_seconds: float) -> float:
    """Reduce confidence for old results."""
    decay_rate = 0.01 per hour
    # Implement decay
    pass
```

### Exercise 2: Ensemble Confidence

Use multiple models and aggregate confidence:

```python
def ensemble_confidence(results: List[Tuple[str, float]]) -> float:
    """Combine multiple model results into aggregate confidence."""
    # If models agree, higher confidence
    # If models disagree, lower confidence
    pass
```

### Exercise 3: Confidence Calibration

Track actual success rate vs reported confidence to calibrate:

```python
class CalibratedConfidenceManager:
    """Adjust AI confidence based on historical accuracy."""

    def calibrate(self, reported: float, actual_success: bool):
        """Update calibration based on outcome."""
        pass

    def adjusted_confidence(self, reported: float) -> float:
        """Get calibrated confidence."""
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Different strategies produce different accept thresholds
- [ ] Adaptive thresholds lower with more attempts
- [ ] Cost tracking works correctly
- [ ] Task type affects strategy selection
- [ ] You understand the quality-cost tradeoff

---

## Key Takeaway

> Confidence scores enable smart retry decisions.

With adaptive confidence:
- **Accept early** when confidence is high
- **Accept eventually** when good enough after retries
- **Fail fast** when confidence never reaches minimum
- **Control costs** by limiting retry spend
- **Match strategy to task** (strict for code, lenient for creative)

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./06-circuit-breakers" class="prev">← Previous: Lab 06 - Circuit Breakers</a>
  <a href="./08-orchestrator" class="next">Next: Lab 08 - Building an Orchestrator →</a>
</div>
