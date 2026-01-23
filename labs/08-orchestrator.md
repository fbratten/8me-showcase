---
layout: lab
title: "Lab 08: Building an Orchestrator"
lab_number: 8
difficulty: advanced
time: 2 hours
prerequisites: Labs 01-07 completed
---

# Lab 08: Building an Orchestrator

Compose all previous concepts into a complete, production-ready orchestrator.

## Objectives

By the end of this lab, you will:
- Understand orchestration architecture
- Build a modular orchestrator with pluggable components
- Implement the orchestrator lifecycle
- Create reusable execution strategies

## Prerequisites

- Labs 01-07 completed
- Understanding of all intermediate concepts

## What is an Orchestrator?

An orchestrator is the **conductor** that coordinates all the pieces:

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Task    │   │ Executor │   │ Verifier │   │ Circuit  │ │
│  │  Store   │   │          │   │          │   │ Breaker  │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘ │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │                                  │
│                    ┌─────┴─────┐                           │
│                    │   Loop    │                           │
│                    └───────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

The orchestrator:
1. **Gets tasks** from the store
2. **Executes** them via the executor
3. **Verifies** results via the verifier
4. **Enforces safety** via the circuit breaker
5. **Manages state** throughout the lifecycle

---

## Step 1: Define the Interfaces

Create `interfaces.py`:

```python
"""
Orchestrator Interfaces - Lab 08

Abstract interfaces for pluggable orchestrator components.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any
from dataclasses import dataclass
from enum import Enum


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
    max_attempts: int = 3
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result from executing a task."""
    success: bool
    output: str
    confidence: float
    error: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VerificationResult:
    """Result from verifying an execution."""
    passed: bool
    confidence: float
    feedback: str
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


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
    def update_status(self, task_id: str, status: TaskStatus):
        """Update task status."""
        pass

    @abstractmethod
    def save_result(self, task_id: str, result: str):
        """Save task result."""
        pass

    @abstractmethod
    def increment_attempts(self, task_id: str):
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
    def record_success(self):
        """Record a successful iteration."""
        pass

    @abstractmethod
    def record_failure(self, error: str):
        """Record a failed iteration."""
        pass

    @abstractmethod
    def get_stop_reason(self) -> Optional[str]:
        """Get reason for stopping, if any."""
        pass
```

---

## Step 2: Create the Core Orchestrator

Create `orchestrator.py`:

```python
"""
Core Orchestrator - Lab 08

The main orchestration engine that coordinates all components.
"""

from typing import Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
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
    max_retries_per_task: int = 3
    pause_between_tasks: float = 0.0  # Seconds
    log_level: str = "info"


@dataclass
class OrchestratorStats:
    """Statistics for the orchestrator run."""
    tasks_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_completed / self.tasks_processed

    def to_dict(self) -> dict:
        return {
            "tasks_processed": self.tasks_processed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_attempts": self.total_attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "success_rate": round(self.success_rate * 100, 1)
        }


# Type alias for hooks
Hook = Callable[["Orchestrator", Task, Optional[ExecutionResult]], None]


class Orchestrator:
    """
    The main orchestration engine.

    Coordinates task storage, execution, verification, and safety
    to process tasks reliably.

    Usage:
        orchestrator = Orchestrator(
            task_store=my_store,
            executor=my_executor,
            verifier=my_verifier,
            safety=my_circuit_breaker
        )

        orchestrator.run()

        print(orchestrator.stats.to_dict())
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
        """
        Run the orchestrator until all tasks are processed or stopped.

        Returns:
            OrchestratorStats with run statistics
        """
        self.state = OrchestratorState.RUNNING
        self.stats = OrchestratorStats()
        self._trigger_hook("on_run_start", None, None)

        try:
            while self._should_continue():
                task = self.task_store.get_next()
                if not task:
                    break

                self._process_task(task)

                if self.config.pause_between_tasks > 0:
                    time.sleep(self.config.pause_between_tasks)

        except Exception as e:
            self.state = OrchestratorState.ERROR
            raise
        finally:
            self.stats.end_time = time.time()
            self._trigger_hook("on_run_end", None, None)

        self.state = OrchestratorState.STOPPED
        return self.stats

    def pause(self):
        """Pause the orchestrator."""
        if self.state == OrchestratorState.RUNNING:
            self.state = OrchestratorState.PAUSED

    def resume(self):
        """Resume a paused orchestrator."""
        if self.state == OrchestratorState.PAUSED:
            self.state = OrchestratorState.RUNNING

    def stop(self):
        """Stop the orchestrator."""
        self.state = OrchestratorState.STOPPED

    # ==================== Core Logic ====================

    def _should_continue(self) -> bool:
        """Check if the orchestrator should continue running."""
        if self.state != OrchestratorState.RUNNING:
            return False

        if not self.task_store.has_pending():
            return False

        if self.safety and not self.safety.allow_continue():
            return False

        return True

    def _process_task(self, task: Task):
        """Process a single task through the full lifecycle."""
        self.stats.tasks_processed += 1
        self._trigger_hook("on_task_start", task, None)

        # Mark task as in progress
        self.task_store.update_status(task.id, TaskStatus.IN_PROGRESS)
        self.task_store.increment_attempts(task)

        # Execute
        result = self.executor.execute(task)
        self.stats.total_attempts += 1

        if not result.success:
            self._handle_execution_failure(task, result)
            return

        # Verify (if enabled and verifier provided)
        if self.config.verify_results and self.verifier:
            verification = self.verifier.verify(task, result)

            if not verification.passed:
                self._handle_verification_failure(task, result, verification)
                return

        # Success!
        self._complete_task(task, result)

    def _complete_task(self, task: Task, result: ExecutionResult):
        """Mark task as completed."""
        self.task_store.update_status(task.id, TaskStatus.COMPLETED)
        self.task_store.save_result(task.id, result.output)
        self.stats.tasks_completed += 1

        if self.safety:
            self.safety.record_success()

        self._trigger_hook("on_task_complete", task, result)

    def _handle_execution_failure(self, task: Task, result: ExecutionResult):
        """Handle a failed execution."""
        if self.safety:
            self.safety.record_failure(result.error or "Execution failed")

        if self._should_retry(task):
            self._retry_task(task, result.error or "Execution failed")
        else:
            self._fail_task(task, result.error or "Execution failed")

    def _handle_verification_failure(
        self,
        task: Task,
        result: ExecutionResult,
        verification: VerificationResult
    ):
        """Handle a failed verification."""
        if self.safety:
            self.safety.record_failure(verification.feedback)

        if self._should_retry(task):
            self._retry_task(task, verification.feedback)
        else:
            self._fail_task(task, verification.feedback)

    def _should_retry(self, task: Task) -> bool:
        """Determine if a task should be retried."""
        if not self.config.retry_on_failure:
            return False

        # Get current task state for attempt count
        current_task = self.task_store.get_task(task.id)
        if current_task and current_task.attempts >= self.config.max_retries_per_task:
            return False

        return True

    def _retry_task(self, task: Task, reason: str):
        """Queue task for retry."""
        self.task_store.update_status(task.id, TaskStatus.PENDING)
        self._trigger_hook("on_task_retry", task, None)

    def _fail_task(self, task: Task, reason: str):
        """Mark task as permanently failed."""
        self.task_store.update_status(task.id, TaskStatus.FAILED)
        self.stats.tasks_failed += 1
        self._trigger_hook("on_task_fail", task, None)

    # ==================== Hooks ====================

    def add_hook(self, event: str, hook: Hook):
        """Add a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(hook)

    def _trigger_hook(
        self,
        event: str,
        task: Optional[Task],
        result: Optional[ExecutionResult]
    ):
        """Trigger all hooks for an event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(self, task, result)
            except Exception as e:
                # Don't let hooks crash the orchestrator
                print(f"Hook error ({event}): {e}")


class OrchestratorBuilder:
    """
    Builder for creating orchestrators with fluent API.

    Usage:
        orchestrator = (OrchestratorBuilder()
            .with_task_store(my_store)
            .with_executor(my_executor)
            .with_verifier(my_verifier)
            .with_safety(my_breaker)
            .with_config(my_config)
            .on_task_complete(my_callback)
            .build())
    """

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

## Step 3: Implement Concrete Components

Create `components.py`:

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
                        max_attempts=task_data.get("max_attempts", 3),
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
                    "max_attempts": t.max_attempts,
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

    def update_status(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self._save()

    def save_result(self, task_id: str, result: str):
        if task_id in self.tasks:
            self.tasks[task_id].result = result
            self._save()

    def increment_attempts(self, task: Task):
        if task.id in self.tasks:
            self.tasks[task.id].attempts += 1
            self._save()

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)


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
                confidence=0.85,  # Default confidence
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

    def record_success(self):
        self.iterations += 1
        self.consecutive_failures = 0

    def record_failure(self, error: str):
        self.iterations += 1
        self.consecutive_failures += 1

    def get_stop_reason(self) -> Optional[str]:
        return self.stop_reason
```

---

## Step 4: Put It All Together

Create `main.py`:

```python
"""
Orchestrator Demo - Lab 08

Complete example of the orchestrator in action.
"""

from orchestrator import Orchestrator, OrchestratorBuilder, OrchestratorConfig
from components import JSONTaskStore, ClaudeExecutor, ClaudeVerifier, SimpleCircuitBreaker


def on_task_start(orch, task, result):
    print(f"\n▶ Starting: {task.description[:50]}...")


def on_task_complete(orch, task, result):
    print(f"  ✓ Completed!")


def on_task_fail(orch, task, result):
    print(f"  ✗ Failed")


def on_task_retry(orch, task, result):
    print(f"  ↻ Retrying...")


def main():
    # Create components
    store = JSONTaskStore("tasks.json")
    executor = ClaudeExecutor()
    verifier = ClaudeVerifier()
    safety = SimpleCircuitBreaker(max_iterations=50, max_consecutive_failures=3)

    # Add sample tasks if empty
    if not store.has_pending() and len(store.tasks) == 0:
        store.add("Write a haiku about orchestration")
        store.add("Explain the conductor pattern in one sentence")
        store.add("List 3 benefits of modular architecture")

    # Build orchestrator
    config = OrchestratorConfig(
        verify_results=True,
        retry_on_failure=True,
        max_retries_per_task=3
    )

    orchestrator = (OrchestratorBuilder()
        .with_task_store(store)
        .with_executor(executor)
        .with_verifier(verifier)
        .with_safety(safety)
        .with_config(config)
        .on_task_start(on_task_start)
        .on_task_complete(on_task_complete)
        .on_task_fail(on_task_fail)
        .on_task_retry(on_task_retry)
        .build())

    # Run!
    print("=" * 60)
    print("ORCHESTRATOR DEMO")
    print("=" * 60)

    stats = orchestrator.run()

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nStats: {stats.to_dict()}")

    if safety.get_stop_reason():
        print(f"\n⚠️ Stopped: {safety.get_stop_reason()}")

    print("\nTasks:")
    for task in store.tasks.values():
        icon = {"completed": "✓", "failed": "✗", "pending": "○"}.get(task.status.value, "?")
        print(f"  {icon} [{task.id}] {task.description[:40]}...")
        if task.result:
            print(f"      Result: {task.result[:60]}...")


if __name__ == "__main__":
    main()
```

---

## Understanding the Architecture

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **TaskStore** | Persistence, querying, state management |
| **Executor** | AI interaction, prompt construction |
| **Verifier** | Quality checks, validation |
| **SafetyCheck** | Limits, circuit breaking |
| **Orchestrator** | Coordination, lifecycle, hooks |

### The Orchestrator Loop

```
┌────────────────────────────────────────────────────┐
│                 orchestrator.run()                  │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ should_continue │◄───────────────────┐
              └───────┬────────┘                     │
                      │ yes                          │
                      ▼                              │
              ┌────────────────┐                     │
              │   get_next()   │                     │
              └───────┬────────┘                     │
                      │                              │
                      ▼                              │
              ┌────────────────┐                     │
              │   execute()    │                     │
              └───────┬────────┘                     │
                      │                              │
                      ▼                              │
              ┌────────────────┐                     │
              │   verify()     │                     │
              └───────┬────────┘                     │
                      │                              │
            ┌─────────┴─────────┐                    │
            │                   │                    │
            ▼                   ▼                    │
       ┌────────┐         ┌────────┐                │
       │ PASS   │         │ FAIL   │                │
       └───┬────┘         └───┬────┘                │
           │                  │                      │
           ▼                  ▼                      │
       complete()        retry/fail()               │
           │                  │                      │
           └──────────────────┴──────────────────────┘
```

### Why Interfaces?

```python
# Without interfaces: tightly coupled
class Orchestrator:
    def __init__(self):
        self.store = JSONTaskStore()  # Hardcoded!
        self.executor = ClaudeExecutor()  # Hardcoded!

# With interfaces: loosely coupled
class Orchestrator:
    def __init__(self, store: TaskStore, executor: Executor):
        self.store = store  # Any TaskStore implementation
        self.executor = executor  # Any Executor implementation
```

Benefits:
- **Testability**: Inject mocks for testing
- **Flexibility**: Swap implementations easily
- **Extensibility**: Add new implementations without changing core

---

## Exercises

### Exercise 1: Add Parallel Execution

Extend the orchestrator to process multiple tasks in parallel:

```python
class ParallelOrchestrator(Orchestrator):
    def __init__(self, *args, max_workers: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers

    def run(self):
        # Use ThreadPoolExecutor or asyncio
        pass
```

### Exercise 2: Add Priority Queue

Implement priority-based task selection:

```python
class PriorityTaskStore(TaskStore):
    def get_next(self) -> Optional[Task]:
        # Return highest priority pending task
        pass
```

### Exercise 3: Add Checkpointing

Save orchestrator state for resume after crash:

```python
class CheckpointingOrchestrator(Orchestrator):
    def save_checkpoint(self):
        # Save current state to disk
        pass

    def restore_checkpoint(self):
        # Restore state from disk
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Orchestrator processes tasks end-to-end
- [ ] Verification gates task completion
- [ ] Circuit breaker stops runaway loops
- [ ] Hooks fire at correct lifecycle points
- [ ] Builder pattern creates valid orchestrators

---

## Key Takeaway

> Orchestrators compose smaller pieces into reliable workflows.

A well-designed orchestrator:
- **Separates concerns** with clear interfaces
- **Enables testing** via dependency injection
- **Provides visibility** through hooks and stats
- **Ensures reliability** with verification and safety checks
- **Scales cleanly** through pluggable components

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./07-confidence-retry" class="prev">← Previous: Lab 07 - Confidence Retry</a>
  <a href="./09-multi-agent" class="next">Next: Lab 09 - Multi-Agent Patterns →</a>
</div>
