---
layout: default
title: "Lab 08: Building an Orchestrator"
nav_order: 8
parent: Labs
lab_number: 8
difficulty: advanced
time: 2 hours
prerequisites: Labs 01-07 completed
---

# Lab 08: Building an Orchestrator

Build a **production-shaped orchestrator skeleton** - minimal but extensible, with pluggable adapters and a hardening checklist for production use.

## Objectives

By the end of this lab, you will:
- Build a modular orchestrator from the ground up, layer by layer
- Understand how each component (store, executor, verifier, safety) fits together
- Have a working orchestrator you can run without API keys
- Know what's needed to harden it for production

## Prerequisites

- Labs 01-07 completed
- Understanding of all intermediate concepts

## Code Map

Before diving in, here's what we'll build:

| File | Purpose |
|------|---------|
| `interfaces.py` | Contracts + shared types (no vendor code) |
| `orchestrator.py` | Core loop logic (vendor-agnostic) |
| `components.py` | Adapters (JSON store, Claude executor, circuit breaker) |
| `main.py` | Assembly + demo |

---

## The Layered Approach

We'll build the orchestrator in layers, each adding one concept:

| Layer | Adds | What You Get |
|-------|------|--------------|
| 0 | Basic loop | `get_next() → execute() → save_result()` |
| 1 | State + attempts | `IN_PROGRESS / COMPLETED / FAILED` tracking |
| 2 | Retries | `attempts < max_attempts ? requeue : fail` |
| 3 | Verification | Execution success ≠ task done |
| 4 | Safety | Circuit breaker stops runaway loops |
| 5 | Hooks | Observability without coupling |
| 6 | Builder | Ergonomic composition |

Each layer produces a runnable system.

---

## Layer 0: The Smallest Orchestrator Loop

The core pattern is just three lines:

```python
while has_pending():
    task = get_next()
    result = execute(task)
    save_result(task, result)
```

That's it. Everything else is refinement.

---

## Step 1: Define the Interfaces

Create `interfaces.py` - the contracts that all components must follow:

```python
"""
Orchestrator Interfaces - Lab 08

Abstract interfaces for pluggable orchestrator components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Universal task representation."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from executing a task."""
    success: bool
    output: str
    confidence: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result from verifying an execution."""
    passed: bool
    confidence: float
    feedback: str
    issues: List[str] = field(default_factory=list)


class TaskStore(ABC):
    """Abstract interface for task storage."""

    @abstractmethod
    def get_next(self) -> Optional[Task]:
        """Get the next pending task."""
        pass

    @abstractmethod
    def has_pending(self) -> bool:
        """Check if there are pending tasks."""
        pass

    @abstractmethod
    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        pass

    @abstractmethod
    def save_result(self, task_id: str, result: str) -> None:
        """Save task result."""
        pass

    @abstractmethod
    def increment_attempts(self, task_id: str) -> None:
        """Increment attempt counter."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        pass


class Executor(ABC):
    """Abstract interface for task execution."""

    @abstractmethod
    def execute(self, task: Task) -> ExecutionResult:
        """Execute a task and return result."""
        pass


class Verifier(ABC):
    """Abstract interface for result verification."""

    @abstractmethod
    def verify(self, task: Task, result: ExecutionResult) -> VerificationResult:
        """Verify an execution result."""
        pass


class SafetyCheck(ABC):
    """Abstract interface for safety checks."""

    @abstractmethod
    def allow_continue(self) -> bool:
        """Check if it's safe to continue."""
        pass

    @abstractmethod
    def record_success(self) -> None:
        """Record a successful iteration."""
        pass

    @abstractmethod
    def record_failure(self, error: str) -> None:
        """Record a failed iteration."""
        pass

    @abstractmethod
    def get_stop_reason(self) -> Optional[str]:
        """Get reason for stopping, if any."""
        pass
```

**Key design decisions:**
- `Task` has no `max_attempts` - that's a config concern, not a data concern
- All mutable defaults use `field(default_factory=...)` - avoids shared state bugs
- `increment_attempts` takes `task_id: str` - consistent with other methods

---

## Step 2: Create the Core Orchestrator

Create `orchestrator.py` - the engine that coordinates all components:

```python
"""
Core Orchestrator - Lab 08

The main orchestration engine that coordinates all components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, List
import time

from interfaces import (
    Task, TaskStatus, TaskStore, Executor, Verifier, SafetyCheck,
    ExecutionResult, VerificationResult
)


class OrchestratorState(Enum):
    """Orchestrator lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    verify_results: bool = True
    retry_on_failure: bool = True
    max_attempts_per_task: int = 3   # Total tries, including the first
    pause_between_tasks: float = 0.0


@dataclass
class OrchestratorStats:
    """Statistics for the orchestrator run."""
    tasks_started: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    executions: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        return {
            "tasks_started": self.tasks_started,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "executions": self.executions,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# Type alias for hooks
Hook = Callable[["Orchestrator", Optional[Task], Optional[ExecutionResult]], None]


class Orchestrator:
    """
    The main orchestration engine.

    Coordinates task storage, execution, verification, and safety
    to process tasks reliably.
    """

    def __init__(
        self,
        task_store: TaskStore,
        executor: Executor,
        verifier: Optional[Verifier] = None,
        safety: Optional[SafetyCheck] = None,
        config: Optional[OrchestratorConfig] = None
    ):
        self.task_store = task_store
        self.executor = executor
        self.verifier = verifier
        self.safety = safety
        self.config = config or OrchestratorConfig()

        self.state = OrchestratorState.IDLE
        self.stats = OrchestratorStats()

        # Lifecycle hooks
        self._hooks: dict[str, List[Hook]] = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_task_fail": [],
            "on_task_retry": [],
            "on_run_start": [],
            "on_run_end": [],
        }

    # ==================== Lifecycle ====================

    def run(self) -> OrchestratorStats:
        """Run the orchestrator until all tasks are processed or stopped."""
        self.state = OrchestratorState.RUNNING
        self.stats = OrchestratorStats()
        self._trigger_hook("on_run_start", None, None)

        try:
            while True:
                # Handle pause state - sleep and continue checking
                if self.state == OrchestratorState.PAUSED:
                    time.sleep(0.1)
                    continue

                if not self._should_continue():
                    break

                task = self.task_store.get_next()
                if not task:
                    break

                self._process_task(task)

                if self.config.pause_between_tasks > 0:
                    time.sleep(self.config.pause_between_tasks)

        except Exception:
            self.state = OrchestratorState.ERROR
            raise
        finally:
            self.stats.end_time = time.time()
            self._trigger_hook("on_run_end", None, None)

        if self.state != OrchestratorState.ERROR:
            self.state = OrchestratorState.STOPPED
        return self.stats

    def pause(self) -> None:
        """Pause the orchestrator (loop continues but waits)."""
        if self.state == OrchestratorState.RUNNING:
            self.state = OrchestratorState.PAUSED

    def resume(self) -> None:
        """Resume a paused orchestrator."""
        if self.state == OrchestratorState.PAUSED:
            self.state = OrchestratorState.RUNNING

    def stop(self) -> None:
        """Stop the orchestrator."""
        self.state = OrchestratorState.STOPPED

    # ==================== Core Logic ====================

    def _should_continue(self) -> bool:
        """Check if the orchestrator should continue running."""
        if self.state not in (OrchestratorState.RUNNING, OrchestratorState.PAUSED):
            return False

        if not self.task_store.has_pending():
            return False

        if self.safety and not self.safety.allow_continue():
            return False

        return True

    def _process_task(self, task: Task) -> None:
        """Process a single task through the full lifecycle."""
        self.stats.tasks_started += 1
        self._trigger_hook("on_task_start", task, None)

        # Mark task as in progress and increment attempts
        self.task_store.update_status(task.id, TaskStatus.IN_PROGRESS)
        self.task_store.increment_attempts(task.id)

        # Get refreshed task with updated attempt count
        current_task = self.task_store.get_task(task.id) or task

        # Execute
        result = self.executor.execute(current_task)
        self.stats.executions += 1

        if not result.success:
            self._handle_failure(current_task, result.error or "Execution failed")
            return

        # Verify (if enabled and verifier provided)
        if self.config.verify_results and self.verifier:
            verification = self.verifier.verify(current_task, result)
            if not verification.passed:
                self._handle_failure(current_task, verification.feedback)
                return

        # Success!
        self._complete_task(current_task, result)

    def _complete_task(self, task: Task, result: ExecutionResult) -> None:
        """Mark task as completed."""
        self.task_store.update_status(task.id, TaskStatus.COMPLETED)
        self.task_store.save_result(task.id, result.output)
        self.stats.tasks_completed += 1

        if self.safety:
            self.safety.record_success()

        self._trigger_hook("on_task_complete", task, result)

    def _handle_failure(self, task: Task, reason: str) -> None:
        """Handle a failed execution or verification."""
        if self.safety:
            self.safety.record_failure(reason)

        # Get current attempt count
        refreshed = self.task_store.get_task(task.id) or task

        # Retry if: retries enabled AND attempts < max
        can_retry = (
            self.config.retry_on_failure and
            refreshed.attempts < self.config.max_attempts_per_task
        )

        if can_retry:
            self.task_store.update_status(task.id, TaskStatus.PENDING)
            self._trigger_hook("on_task_retry", task, None)
        else:
            self.task_store.update_status(task.id, TaskStatus.FAILED)
            self.stats.tasks_failed += 1
            self._trigger_hook("on_task_fail", task, None)

    # ==================== Hooks ====================

    def add_hook(self, event: str, hook: Hook) -> None:
        """Add a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(hook)

    def _trigger_hook(
        self,
        event: str,
        task: Optional[Task],
        result: Optional[ExecutionResult]
    ) -> None:
        """Trigger all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(self, task, result)
            except Exception as e:
                # Don't let hooks crash the orchestrator
                print(f"Hook error ({event}): {e}")


class OrchestratorBuilder:
    """Builder for creating orchestrators with fluent API."""

    def __init__(self):
        self._task_store = None
        self._executor = None
        self._verifier = None
        self._safety = None
        self._config = None
        self._hooks: dict[str, List[Hook]] = {}

    def with_task_store(self, store: TaskStore) -> "OrchestratorBuilder":
        self._task_store = store
        return self

    def with_executor(self, executor: Executor) -> "OrchestratorBuilder":
        self._executor = executor
        return self

    def with_verifier(self, verifier: Verifier) -> "OrchestratorBuilder":
        self._verifier = verifier
        return self

    def with_safety(self, safety: SafetyCheck) -> "OrchestratorBuilder":
        self._safety = safety
        return self

    def with_config(self, config: OrchestratorConfig) -> "OrchestratorBuilder":
        self._config = config
        return self

    def on_task_start(self, hook: Hook) -> "OrchestratorBuilder":
        self._hooks.setdefault("on_task_start", []).append(hook)
        return self

    def on_task_complete(self, hook: Hook) -> "OrchestratorBuilder":
        self._hooks.setdefault("on_task_complete", []).append(hook)
        return self

    def on_task_fail(self, hook: Hook) -> "OrchestratorBuilder":
        self._hooks.setdefault("on_task_fail", []).append(hook)
        return self

    def on_task_retry(self, hook: Hook) -> "OrchestratorBuilder":
        self._hooks.setdefault("on_task_retry", []).append(hook)
        return self

    def build(self) -> Orchestrator:
        if not self._task_store:
            raise ValueError("TaskStore is required")
        if not self._executor:
            raise ValueError("Executor is required")

        orchestrator = Orchestrator(
            task_store=self._task_store,
            executor=self._executor,
            verifier=self._verifier,
            safety=self._safety,
            config=self._config
        )

        for event, hooks in self._hooks.items():
            for hook in hooks:
                orchestrator.add_hook(event, hook)

        return orchestrator
```

---

## Walkthrough Trace

Here's what happens when a task needs one retry:

```
Task-001 picked up
  → status = IN_PROGRESS
  → attempts = 1
  → executor.execute() returns success
  → verifier.verify() FAILS (low confidence)
  → attempts (1) < max_attempts (3) → requeue
  → status = PENDING
  → on_task_retry hook fires

Task-001 picked up again
  → status = IN_PROGRESS
  → attempts = 2
  → executor.execute() returns success
  → verifier.verify() PASSES
  → status = COMPLETED
  → on_task_complete hook fires
```

This trace is worth a page of explanation.

---

## Step 3: Implement Concrete Components

Create `components.py` - the pluggable adapters:

```python
"""
Concrete Orchestrator Components - Lab 08

Ready-to-use implementations of orchestrator interfaces.
"""

import json
from pathlib import Path
from typing import Optional, List
import anthropic

from interfaces import (
    Task, TaskStatus, TaskStore, Executor, Verifier, SafetyCheck,
    ExecutionResult, VerificationResult
)


# ==================== Task Stores ====================

class InMemoryTaskStore(TaskStore):
    """In-memory task store - great for testing and learning."""

    def __init__(self, tasks: Optional[List[Task]] = None):
        self.tasks: dict[str, Task] = {}
        if tasks:
            for task in tasks:
                self.tasks[task.id] = task

    def add(self, task_id: str, description: str) -> Task:
        task = Task(id=task_id, description=description)
        self.tasks[task_id] = task
        return task

    def get_next(self) -> Optional[Task]:
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def has_pending(self) -> bool:
        return any(t.status == TaskStatus.PENDING for t in self.tasks.values())

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = status

    def save_result(self, task_id: str, result: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].result = result

    def increment_attempts(self, task_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].attempts += 1

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)


class JSONTaskStore(TaskStore):
    """Task store backed by a JSON file."""

    def __init__(self, filepath: str = "tasks.json"):
        self.filepath = Path(filepath)
        self.tasks: dict[str, Task] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                data = json.load(f)
                for task_data in data.get("tasks", []):
                    task = Task(
                        id=task_data["id"],
                        description=task_data["description"],
                        status=TaskStatus(task_data.get("status", "pending")),
                        result=task_data.get("result"),
                        attempts=task_data.get("attempts", 0),
                        metadata=task_data.get("metadata", {})
                    )
                    self.tasks[task.id] = task

    def _save(self):
        data = {
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "result": t.result,
                    "attempts": t.attempts,
                    "metadata": t.metadata
                }
                for t in self.tasks.values()
            ]
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, description: str, **kwargs) -> Task:
        task_id = f"task-{len(self.tasks) + 1:03d}"
        task = Task(id=task_id, description=description, **kwargs)
        self.tasks[task_id] = task
        self._save()
        return task

    def get_next(self) -> Optional[Task]:
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def has_pending(self) -> bool:
        return any(t.status == TaskStatus.PENDING for t in self.tasks.values())

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self._save()

    def save_result(self, task_id: str, result: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].result = result
            self._save()

    def increment_attempts(self, task_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].attempts += 1
            self._save()

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)


# ==================== Executors ====================

class FakeExecutor(Executor):
    """Fake executor for testing - no API keys needed."""

    def __init__(self, success: bool = True, output: str = "Fake result"):
        self.success = success
        self.output = output
        self.call_count = 0

    def execute(self, task: Task) -> ExecutionResult:
        self.call_count += 1
        return ExecutionResult(
            success=self.success,
            output=f"{self.output} for: {task.description}",
            confidence=0.9 if self.success else 0.0,
            error=None if self.success else "Fake failure"
        )


class ClaudeExecutor(Executor):
    """Executor that uses Claude for task completion."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def execute(self, task: Task) -> ExecutionResult:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": task.description}]
            )

            output = response.content[0].text

            return ExecutionResult(
                success=True,
                output=output,
                confidence=0.85,
                metadata={
                    "model": self.model,
                    "tokens": response.usage.output_tokens
                }
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                confidence=0.0,
                error=str(e)
            )


# ==================== Verifiers ====================

class AlwaysPassVerifier(Verifier):
    """Verifier that always passes - for testing."""

    def verify(self, task: Task, result: ExecutionResult) -> VerificationResult:
        return VerificationResult(
            passed=True,
            confidence=1.0,
            feedback="Auto-passed"
        )


class ClaudeVerifier(Verifier):
    """Verifier that uses Claude to check results."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def verify(self, task: Task, result: ExecutionResult) -> VerificationResult:
        prompt = f"""Verify if this result adequately completes the task.

TASK: {task.description}

RESULT:
{result.output}

Respond with:
PASSED: yes or no
CONFIDENCE: 0.0 to 1.0
FEEDBACK: brief explanation
ISSUES: comma-separated list of issues (if any)
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_response(response.content[0].text)

        except Exception as e:
            return VerificationResult(
                passed=False,
                confidence=0.0,
                feedback=f"Verification error: {e}",
                issues=["Verification failed"]
            )

    def _parse_response(self, text: str) -> VerificationResult:
        passed = False
        confidence = 0.5
        feedback = ""
        issues = []

        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("PASSED:"):
                passed = "yes" in line.lower()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.upper().startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() != "none":
                    issues = [i.strip() for i in issues_str.split(",")]

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            feedback=feedback,
            issues=issues
        )


# ==================== Safety ====================

class SimpleCircuitBreaker(SafetyCheck):
    """Simple circuit breaker implementation."""

    def __init__(
        self,
        max_iterations: int = 100,
        max_consecutive_failures: int = 5
    ):
        self.max_iterations = max_iterations
        self.max_consecutive_failures = max_consecutive_failures
        self.iterations = 0
        self.consecutive_failures = 0
        self.stop_reason = None

    def allow_continue(self) -> bool:
        if self.iterations >= self.max_iterations:
            self.stop_reason = f"Max iterations ({self.max_iterations}) reached"
            return False

        if self.consecutive_failures >= self.max_consecutive_failures:
            self.stop_reason = f"Too many consecutive failures ({self.consecutive_failures})"
            return False

        return True

    def record_success(self) -> None:
        self.iterations += 1
        self.consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        self.iterations += 1
        self.consecutive_failures += 1

    def get_stop_reason(self) -> Optional[str]:
        return self.stop_reason
```

---

## Step 4: Put It All Together

Create `main.py` - first without API keys, then with Claude:

```python
"""
Orchestrator Demo - Lab 08

Run without API keys first, then upgrade to Claude.
"""

from orchestrator import Orchestrator, OrchestratorBuilder, OrchestratorConfig
from components import (
    InMemoryTaskStore, FakeExecutor, AlwaysPassVerifier,
    JSONTaskStore, ClaudeExecutor, ClaudeVerifier,
    SimpleCircuitBreaker
)
from interfaces import Task


def on_task_start(orch, task, result):
    print(f"\n  [{task.attempts + 1}] Starting: {task.description[:40]}...")


def on_task_complete(orch, task, result):
    print(f"      ✓ Completed!")


def on_task_fail(orch, task, result):
    print(f"      ✗ Failed after {task.attempts} attempts")


def on_task_retry(orch, task, result):
    print(f"      ↻ Will retry...")


def demo_without_api():
    """Run the orchestrator without needing API keys."""
    print("=" * 60)
    print("DEMO: Orchestrator with Fake Components")
    print("=" * 60)

    # Create in-memory store with test tasks
    store = InMemoryTaskStore()
    store.add("task-001", "Write a haiku")
    store.add("task-002", "Explain loops")
    store.add("task-003", "List 3 benefits")

    # Build orchestrator with fake components
    orchestrator = (OrchestratorBuilder()
        .with_task_store(store)
        .with_executor(FakeExecutor(success=True))
        .with_verifier(AlwaysPassVerifier())
        .with_safety(SimpleCircuitBreaker(max_iterations=10))
        .with_config(OrchestratorConfig(max_attempts_per_task=3))
        .on_task_start(on_task_start)
        .on_task_complete(on_task_complete)
        .on_task_fail(on_task_fail)
        .on_task_retry(on_task_retry)
        .build())

    stats = orchestrator.run()
    print(f"\nStats: {stats.to_dict()}")


def demo_with_claude():
    """Run the orchestrator with Claude (requires API key)."""
    print("\n" + "=" * 60)
    print("DEMO: Orchestrator with Claude")
    print("=" * 60)

    store = JSONTaskStore("tasks.json")

    # Add sample tasks if empty
    if not store.has_pending() and len(store.tasks) == 0:
        store.add("Write a haiku about orchestration")
        store.add("Explain the conductor pattern in one sentence")

    orchestrator = (OrchestratorBuilder()
        .with_task_store(store)
        .with_executor(ClaudeExecutor())
        .with_verifier(ClaudeVerifier())
        .with_safety(SimpleCircuitBreaker(max_iterations=50))
        .with_config(OrchestratorConfig(
            verify_results=True,
            max_attempts_per_task=3
        ))
        .on_task_start(on_task_start)
        .on_task_complete(on_task_complete)
        .on_task_fail(on_task_fail)
        .on_task_retry(on_task_retry)
        .build())

    stats = orchestrator.run()
    print(f"\nStats: {stats.to_dict()}")


if __name__ == "__main__":
    # Always works - no API key needed
    demo_without_api()

    # Uncomment to test with Claude (requires ANTHROPIC_API_KEY)
    # demo_with_claude()
```

---

## Understanding the Architecture

### Why Interfaces?

```python
# Without interfaces: tightly coupled
class Orchestrator:
    def __init__(self):
        self.store = JSONTaskStore()  # Hardcoded!

# With interfaces: loosely coupled
class Orchestrator:
    def __init__(self, store: TaskStore):
        self.store = store  # Any TaskStore works
```

Benefits:
- **Testability**: Inject fakes for testing
- **Flexibility**: Swap implementations easily
- **Extensibility**: Add new implementations without changing core

---

## Hardening Checklist

This lab gives you a **skeleton**. For production, add:

| Category | What to Add |
|----------|-------------|
| **Timeouts** | Execution timeout, verification timeout |
| **Idempotency** | Dedup by task ID, at-least-once → exactly-once |
| **Persistence** | Durable task store (Postgres, Redis) |
| **Concurrency** | Thread safety, parallel execution |
| **Observability** | Structured logs, metrics, tracing |
| **Dead Letter Queue** | Store permanently failed tasks for review |
| **Graceful Shutdown** | Handle SIGTERM, drain in-flight tasks |

---

## Exercises

### Exercise 1: Add Parallel Execution

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelOrchestrator(Orchestrator):
    def __init__(self, *args, max_workers: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
```

### Exercise 2: Add Priority Queue

```python
class PriorityTaskStore(TaskStore):
    def get_next(self) -> Optional[Task]:
        # Return highest priority pending task
        pass
```

### Exercise 3: Add Checkpointing

```python
class CheckpointingOrchestrator(Orchestrator):
    def save_checkpoint(self):
        pass

    def restore_checkpoint(self):
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Orchestrator runs with fake components (no API key)
- [ ] Tasks go through: PENDING → IN_PROGRESS → COMPLETED/FAILED
- [ ] Retries work (task requeued when verification fails)
- [ ] Circuit breaker stops runaway loops
- [ ] Hooks fire at correct lifecycle points

---

## Key Takeaway

> Orchestrators compose smaller pieces into reliable workflows.

Start with the smallest loop, add one feature at a time, and keep the hardening checklist handy for when you go to production.

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./07-confidence-retry" class="prev">← Previous: Lab 07 - Confidence Retry</a>
  <a href="./09-multi-agent" class="next">Next: Lab 09 - Multi-Agent Patterns →</a>
</div>
