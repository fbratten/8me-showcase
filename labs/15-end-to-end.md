---
layout: lab
title: "Lab 15: End-to-End Workflow"
lab_number: 15
difficulty: integration
time: 3 hours
prerequisites: All previous labs completed
---

# Lab 15: End-to-End Workflow

Connect all components into a complete, production-ready system.

## Objectives

By the end of this lab, you will:
- Integrate all previous components
- Build a complete end-to-end pipeline
- Handle real-world scenarios
- Deploy a production-ready system

## Prerequisites

- All previous labs (01-14) completed
- Understanding of all components

## The Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                                   │
│                    "Process my code review tasks"                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE + SKILL                             │
│                                                                         │
│  SKILL.md: "When user asks to process tasks, use /ralph command..."    │
│                                                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER                                     │
│                                                                         │
│  Resources:                    Tools:                                   │
│  • task://queue                • add_task()                            │
│  • task://stats                • complete_task()                       │
│                                • run_orchestrator()                    │
│                                                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                                    │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   Task   │  │ Executor │  │ Verifier │  │  Gates   │  │  Memory  │ │
│  │  Store   │  │          │  │          │  │          │  │          │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │             │             │        │
│       └─────────────┴─────────────┴─────────────┴─────────────┘        │
│                                   │                                     │
│                          ┌────────┴────────┐                           │
│                          │  Circuit Breaker │                           │
│                          └─────────────────┘                           │
│                                                                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       RESULTS + PERSISTENCE                             │
│                                                                         │
│  • Task results saved to JSON                                          │
│  • Memories stored for future sessions                                 │
│  • Logs written for debugging                                          │
│  • Metrics exported for monitoring                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create the Unified System

Create `system.py`:

```python
"""
End-to-End System - Lab 15

Complete integration of all orchestration components.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

# Core components
from task_manager import TaskManager
from orchestrator import Orchestrator, OrchestratorBuilder, OrchestratorConfig
from components import ClaudeExecutor, ClaudeVerifier, SimpleCircuitBreaker

# Advanced components
from confidence_manager import ConfidenceManager, BalancedStrategy
from gates import GateRunner, TaskDefinedGate, BudgetGate, OutputLengthGate, ConfidenceGate
from drift_detection import KeywordDriftGate
from memory_interface import InMemoryStore
from memory_orchestrator import MemoryOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ralph-system")


@dataclass
class SystemConfig:
    """Configuration for the complete system."""
    # Task settings
    task_file: str = "tasks.json"
    max_attempts_per_task: int = 3

    # Orchestrator settings
    verify_results: bool = True
    use_confidence_manager: bool = True

    # Safety settings
    max_iterations: int = 100
    max_consecutive_failures: int = 5
    max_cost_dollars: float = 1.0

    # Gate settings
    use_gates: bool = True
    min_task_length: int = 10
    min_confidence: float = 0.7

    # Memory settings
    use_memory: bool = True
    memory_file: str = "memory.json"

    # Logging
    log_file: Optional[str] = "ralph.log"
    verbose: bool = True


@dataclass
class SystemStats:
    """Statistics from a system run."""
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    tasks_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    gates_checked: int = 0
    gates_failed: int = 0
    total_cost: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds() if self.ended_at else None,
            "tasks_processed": self.tasks_processed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.tasks_completed / self.tasks_processed * 100 if self.tasks_processed else 0,
            "gates_checked": self.gates_checked,
            "gates_failed": self.gates_failed,
            "total_cost": round(self.total_cost, 4),
            "errors": self.errors
        }


class RalphSystem:
    """
    Complete end-to-end orchestration system.

    Integrates:
    - Task management
    - Execution with Claude
    - Verification
    - Confidence-based retry
    - Pre/post gates
    - Circuit breaker safety
    - Memory persistence
    - Logging and metrics

    Usage:
        system = RalphSystem()
        system.add_task("Write a hello world function")
        system.add_task("Create unit tests")
        stats = system.run()
        print(stats.to_dict())
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.stats: Optional[SystemStats] = None

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all system components."""
        # Task store
        self.task_manager = TaskManager(self.config.task_file)

        # Executor and verifier
        self.executor = ClaudeExecutor()
        self.verifier = ClaudeVerifier() if self.config.verify_results else None

        # Circuit breaker
        self.safety = SimpleCircuitBreaker(
            max_iterations=self.config.max_iterations,
            max_consecutive_failures=self.config.max_consecutive_failures
        )

        # Confidence manager
        self.confidence_manager = None
        if self.config.use_confidence_manager:
            self.confidence_manager = ConfidenceManager(BalancedStrategy())

        # Gates
        self.gate_runner = None
        if self.config.use_gates:
            self.gate_runner = GateRunner()
            self.gate_runner.add_pre_gate(TaskDefinedGate(self.config.min_task_length))
            self.gate_runner.add_pre_gate(BudgetGate(self.config.max_cost_dollars))
            self.gate_runner.add_post_gate(OutputLengthGate(min_length=10))
            self.gate_runner.add_post_gate(ConfidenceGate(self.config.min_confidence))
            self.gate_runner.add_post_gate(KeywordDriftGate(min_overlap=0.2))

        # Memory
        self.memory = None
        if self.config.use_memory:
            self.memory = InMemoryStore()

        logger.info("System initialized with config: %s", self.config)

    def add_task(self, description: str, **kwargs) -> str:
        """Add a task to the queue."""
        task = self.task_manager.create(description, **kwargs)
        logger.info("Added task: %s", task.id)
        return task.id

    def add_tasks(self, descriptions: list) -> list:
        """Add multiple tasks."""
        return [self.add_task(d) for d in descriptions]

    def run(self, project: str = "default") -> SystemStats:
        """
        Run the complete orchestration pipeline.

        Args:
            project: Project name for memory context

        Returns:
            SystemStats with run results
        """
        import uuid

        # Initialize stats
        self.stats = SystemStats(
            session_id=f"session-{uuid.uuid4().hex[:8]}",
            started_at=datetime.now()
        )

        logger.info("Starting session: %s", self.stats.session_id)

        # Load memory context
        if self.memory:
            prior_context = self.memory.recall(entity=project, limit=10)
            if prior_context:
                logger.info("Loaded %d memories from previous sessions", len(prior_context))

        try:
            # Main processing loop
            while self.task_manager.has_pending() and self.safety.allow_continue():
                task = self.task_manager.get_next()
                if not task:
                    break

                self._process_task(task, project)

        except Exception as e:
            logger.error("System error: %s", e)
            self.stats.errors.append(str(e))

        finally:
            # Finalize
            self.stats.ended_at = datetime.now()

            # Save memory
            if self.memory:
                self.memory.summarize_session(
                    session_id=self.stats.session_id,
                    summary=f"Processed {self.stats.tasks_processed} tasks: "
                            f"{self.stats.tasks_completed} completed, "
                            f"{self.stats.tasks_failed} failed",
                    topics=["orchestration", project]
                )

            # Log final stats
            logger.info("Session complete: %s", json.dumps(self.stats.to_dict(), indent=2))

        return self.stats

    def _process_task(self, task, project: str):
        """Process a single task through the full pipeline."""
        self.stats.tasks_processed += 1
        logger.info("Processing task: %s", task.id)

        # Pre-gates
        if self.gate_runner:
            pre_report = self.gate_runner.run_pre_gates(
                task=task.description,
                current_cost=self.stats.total_cost,
                estimated_cost=0.01
            )
            self.stats.gates_checked += len(pre_report.checks)

            if not pre_report.passed:
                logger.warning("Task %s failed pre-gates: %s", task.id, pre_report.summary())
                self.stats.gates_failed += len(pre_report.failures)
                self.task_manager.fail(task.id, f"Pre-gate failed: {pre_report.failures[0].message}")
                self.stats.tasks_failed += 1
                return

        # Execute
        self.task_manager.start(task.id)
        result = self.executor.execute(task)
        self.stats.total_cost += 0.01  # Estimated cost

        if not result.success:
            self._handle_failure(task, result.error or "Execution failed")
            return

        # Post-gates
        if self.gate_runner:
            post_report = self.gate_runner.run_post_gates(
                task=task.description,
                result=result.output,
                confidence=result.confidence
            )
            self.stats.gates_checked += len(post_report.checks)

            if not post_report.passed:
                logger.warning("Task %s failed post-gates: %s", task.id, post_report.summary())
                self.stats.gates_failed += len(post_report.failures)
                self._handle_failure(task, f"Post-gate failed: {post_report.failures[0].message}")
                return

        # Verify
        if self.verifier:
            verification = self.verifier.verify(task, result)

            if not verification.passed:
                self._handle_failure(task, verification.feedback)
                return

        # Success!
        self.task_manager.complete(task.id, result.output)
        self.stats.tasks_completed += 1
        self.safety.record_success()

        # Store in memory
        if self.memory:
            self.memory.store(
                entity=project,
                attribute=f"completed_{task.id}",
                value=f"{task.description[:50]}... -> Success",
                source="orchestrator"
            )

        logger.info("Task %s completed successfully", task.id)

    def _handle_failure(self, task, reason: str):
        """Handle task failure with retry logic."""
        self.safety.record_failure(reason)

        if task.attempts < self.config.max_attempts_per_task:
            self.task_manager.retry(task.id, reason)
            logger.info("Task %s queued for retry (attempt %d)", task.id, task.attempts)
        else:
            self.task_manager.fail(task.id, reason)
            self.stats.tasks_failed += 1

            # Store failure in memory
            if self.memory:
                self.memory.store(
                    entity="error_patterns",
                    attribute="task_failure",
                    value=f"{task.description[:50]}... -> {reason}",
                    source="orchestrator"
                )

            logger.warning("Task %s failed permanently: %s", task.id, reason)

    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "pending_tasks": len(self.task_manager.get_pending()),
            "completed_tasks": len(self.task_manager.get_completed()),
            "failed_tasks": len(self.task_manager.get_failed()),
            "safety_status": "OK" if self.safety.allow_continue() else self.safety.get_stop_reason(),
            "memory_count": len(self.memory.memories) if self.memory else 0
        }

    def reset(self):
        """Reset the system for a new run."""
        self.task_manager.reset_all()
        self.safety = SimpleCircuitBreaker(
            max_iterations=self.config.max_iterations,
            max_consecutive_failures=self.config.max_consecutive_failures
        )
        if self.confidence_manager:
            self.confidence_manager.reset()
        logger.info("System reset")
```

---

## Step 2: Create the CLI

Create `cli.py`:

```python
"""
Ralph System CLI - Lab 15

Command-line interface for the complete system.
"""

import argparse
import json
from system import RalphSystem, SystemConfig


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Workflow System - Autonomous Task Orchestration"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the orchestrator")
    run_parser.add_argument("--project", default="default", help="Project name")
    run_parser.add_argument("--max-tasks", type=int, help="Max tasks to process")
    run_parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    run_parser.add_argument("--no-memory", action="store_true", help="Disable memory")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a task")
    add_parser.add_argument("description", help="Task description")
    add_parser.add_argument("--priority", type=int, default=5)

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Tasks command
    tasks_parser = subparsers.add_parser("tasks", help="List tasks")
    tasks_parser.add_argument("--status", help="Filter by status")

    # Reset command
    subparsers.add_parser("reset", help="Reset all tasks")

    args = parser.parse_args()

    # Create config
    config = SystemConfig()
    if hasattr(args, 'no_verify') and args.no_verify:
        config.verify_results = False
    if hasattr(args, 'no_memory') and args.no_memory:
        config.use_memory = False

    # Create system
    system = RalphSystem(config)

    if args.command == "run":
        print("Starting Ralph Workflow System...")
        print(f"Project: {args.project}")
        print()

        stats = system.run(project=args.project)

        print()
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(json.dumps(stats.to_dict(), indent=2))

    elif args.command == "add":
        task_id = system.add_task(args.description, priority=args.priority)
        print(f"Added task: {task_id}")

    elif args.command == "status":
        status = system.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "tasks":
        tasks = system.task_manager.get_all()
        if args.status:
            tasks = [t for t in tasks if t.status.value == args.status]

        for task in tasks:
            icon = {"pending": "○", "in_progress": "◐", "completed": "✓", "failed": "✗"}
            print(f"{icon.get(task.status.value, '?')} [{task.id}] {task.description[:50]}...")

    elif args.command == "reset":
        system.reset()
        print("System reset complete")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

## Step 3: Create Integration Tests

Create `test_integration.py`:

```python
"""
Integration Tests - Lab 15

End-to-end tests for the complete system.
"""

import unittest
import os
import json
from system import RalphSystem, SystemConfig


class TestEndToEnd(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up test environment."""
        self.config = SystemConfig(
            task_file="test_tasks.json",
            verify_results=False,  # Disable for faster tests
            use_memory=True,
            max_iterations=10
        )
        self.system = RalphSystem(self.config)

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists("test_tasks.json"):
            os.remove("test_tasks.json")

    def test_add_and_process_task(self):
        """Test adding and processing a single task."""
        task_id = self.system.add_task("Write hello world")

        self.assertIsNotNone(task_id)
        self.assertTrue(task_id.startswith("task-"))

        # Run would need actual Claude API, so just verify task was added
        status = self.system.get_status()
        self.assertEqual(status["pending_tasks"], 1)

    def test_multiple_tasks(self):
        """Test adding multiple tasks."""
        ids = self.system.add_tasks([
            "Task one",
            "Task two",
            "Task three"
        ])

        self.assertEqual(len(ids), 3)

        status = self.system.get_status()
        self.assertEqual(status["pending_tasks"], 3)

    def test_reset(self):
        """Test system reset."""
        self.system.add_task("Test task")
        self.system.reset()

        status = self.system.get_status()
        self.assertEqual(status["pending_tasks"], 1)  # Reset keeps tasks but resets status

    def test_config_options(self):
        """Test different configuration options."""
        config = SystemConfig(
            verify_results=False,
            use_gates=False,
            use_memory=False
        )
        system = RalphSystem(config)

        self.assertIsNone(system.verifier)
        self.assertIsNone(system.gate_runner)
        self.assertIsNone(system.memory)


class TestGatesIntegration(unittest.TestCase):
    """Test gate integration."""

    def test_pre_gate_blocks_short_task(self):
        """Test that pre-gate blocks too-short tasks."""
        config = SystemConfig(
            task_file="test_gates.json",
            use_gates=True,
            min_task_length=20,
            verify_results=False
        )
        system = RalphSystem(config)

        # This task is too short
        system.add_task("Hi")

        # Would need to mock executor for full test
        # Just verify gate runner is configured
        self.assertIsNotNone(system.gate_runner)

    def tearDown(self):
        if os.path.exists("test_gates.json"):
            os.remove("test_gates.json")


if __name__ == "__main__":
    unittest.main()
```

---

## Step 4: Create Deployment Script

Create `deploy.sh`:

```bash
#!/bin/bash
# Deployment script for Ralph Workflow System

set -e

echo "=== Ralph Workflow System Deployment ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -e .

# Run tests
echo "Running tests..."
python -m pytest test_integration.py -v

# Verify MCP server
echo "Verifying MCP server..."
python -c "from mcp_server import mcp; print('MCP server OK')"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start the MCP server:"
echo "  ralph-mcp"
echo ""
echo "To use the CLI:"
echo "  python cli.py --help"
echo ""
echo "To configure Claude Desktop, add to config:"
echo '  {"mcpServers": {"ralph-workflow": {"command": "ralph-mcp"}}}'
```

---

## Real-World Usage Example

```bash
# Add tasks
python cli.py add "Review PR #123 for security issues"
python cli.py add "Write unit tests for auth module"
python cli.py add "Update API documentation"

# Check status
python cli.py status

# Run orchestrator
python cli.py run --project myproject

# View results
python cli.py tasks --status completed
```

Output:
```
Starting Ralph Workflow System...
Project: myproject

Processing task-001: Review PR #123...
  ✓ Completed (attempt 1)

Processing task-002: Write unit tests...
  ↻ Retry (verification failed)
  ✓ Completed (attempt 2)

Processing task-003: Update API documentation...
  ✓ Completed (attempt 1)

==================================================
RESULTS
==================================================
{
  "session_id": "session-a1b2c3d4",
  "tasks_processed": 3,
  "tasks_completed": 3,
  "tasks_failed": 0,
  "success_rate": 100.0,
  "duration_seconds": 45.2,
  "total_cost": 0.03
}
```

---

## Exercises

### Exercise 1: Add Monitoring Dashboard

Create a simple web dashboard showing real-time stats:

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/status")
def status():
    return jsonify(system.get_status())

@app.route("/stats")
def stats():
    return jsonify(system.stats.to_dict() if system.stats else {})
```

### Exercise 2: Add Webhook Notifications

Send notifications when tasks complete:

```python
def notify_completion(task_id: str, result: str):
    """Send webhook on task completion."""
    requests.post(WEBHOOK_URL, json={
        "event": "task_completed",
        "task_id": task_id,
        "result": result[:100]
    })
```

### Exercise 3: Add Task Dependencies

Implement task dependencies:

```python
system.add_task("Build project", depends_on=[])
system.add_task("Run tests", depends_on=["task-001"])
system.add_task("Deploy", depends_on=["task-002"])
```

---

## Checkpoint

Before finishing, verify:
- [ ] All components integrate correctly
- [ ] CLI commands work as expected
- [ ] Tests pass
- [ ] MCP server connects to Claude Desktop
- [ ] Skill commands trigger orchestration
- [ ] Memory persists across sessions

---

## Key Takeaway

> Real systems require all pieces working together.

A production orchestration system needs:
- **Modularity**: Pluggable components
- **Reliability**: Gates, verification, circuit breakers
- **Observability**: Logging, metrics, status
- **Persistence**: Task state and memory
- **Accessibility**: CLI, MCP, and skills

---

## Congratulations!

You've completed the entire 8me Labs Curriculum!

### What You've Built

| Lab | Component |
|-----|-----------|
| 01-03 | Core loop, state, verification |
| 04-07 | Task management, tools, safety |
| 08-12 | Orchestrator, multi-agent, memory |
| 13-15 | MCP server, skill, full system |

### Next Steps

1. **Deploy**: Host your MCP server
2. **Extend**: Add custom components
3. **Share**: Contribute to the community
4. **Explore**: Try LangChain, CrewAI, AutoGen

---

## Get the Code

Complete implementation: [8me repository](https://github.com/fbratten/8me)

---

<div class="lab-navigation">
  <a href="./14-skill.md" class="prev">← Previous: Lab 14 - Claude Code Skill</a>
  <span class="next">🎉 Curriculum Complete!</span>
</div>
