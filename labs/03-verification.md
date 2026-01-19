---
layout: lab
title: "Lab 03: Simple Verification"
lab_number: 3
difficulty: beginner
time: 45 minutes
prerequisites: Lab 02 completed
---

# Lab 03: Simple Verification

Verify AI outputs before accepting them as complete.

## Objectives

By the end of this lab, you will:
- Understand why verification matters
- Implement basic verification patterns
- Handle verification failures with retry
- Know when to accept "good enough"

## Prerequisites

- Lab 02 completed (external state)
- Understanding of the loop pattern

## The Core Problem

AI models can be **confidently wrong**. A loop without verification might:

```python
# Without verification - dangerous!
result = ai.complete(task)
mark_complete(task, result)  # Did it actually succeed?
```

Consider these failure modes:
- Task: "Write a function that returns the sum of two numbers"
- Result: A function that returns the product (wrong but plausible)
- Task: "List 5 US state capitals"
- Result: Lists 4 capitals, or includes a city that isn't a capital

**Verification catches these errors before they compound.**

## The Core Concept

```python
result = ai.complete(task)

if verify(result, task.criteria):
    mark_complete(task)
else:
    mark_for_retry(task)
```

Verification creates a quality gate between "AI produced output" and "task is done."

---

## Step 1: Define Verification Criteria

Extend the task schema to include verification criteria:

```json
{
  "tasks": [
    {
      "id": 1,
      "description": "Write a haiku about loops",
      "status": "pending",
      "result": null,
      "criteria": {
        "type": "haiku",
        "requirements": ["exactly 3 lines", "5-7-5 syllable pattern"]
      },
      "attempts": 0,
      "max_attempts": 3
    }
  ]
}
```

## Step 2: Create the Verifier

Create `verifier.py`:

```python
"""
Verifier - Lab 03

Verifies AI outputs against task criteria.
Uses Claude to evaluate results for quality and correctness.
"""

import anthropic


client = anthropic.Anthropic()


def verify_result(task: dict, result: str) -> dict:
    """
    Verify a result against task criteria.

    Returns:
        {
            "passed": bool,
            "confidence": float (0-1),
            "feedback": str
        }
    """
    criteria = task.get("criteria", {})

    # Build verification prompt
    prompt = f"""You are a quality verification assistant. Your job is to determine if a result meets the specified criteria.

TASK: {task['description']}

CRITERIA:
{format_criteria(criteria)}

RESULT TO VERIFY:
{result}

Evaluate whether the result meets ALL criteria. Be strict but fair.

Respond in this exact format:
PASSED: yes or no
CONFIDENCE: a number from 0.0 to 1.0
FEEDBACK: brief explanation of your evaluation
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_verification_response(response.content[0].text)


def format_criteria(criteria: dict) -> str:
    """Format criteria dictionary for the prompt."""
    if not criteria:
        return "- Result should reasonably complete the task"

    lines = []
    if "type" in criteria:
        lines.append(f"- Type: {criteria['type']}")
    if "requirements" in criteria:
        for req in criteria["requirements"]:
            lines.append(f"- {req}")

    return "\n".join(lines) if lines else "- Result should reasonably complete the task"


def parse_verification_response(text: str) -> dict:
    """Parse Claude's verification response."""
    lines = text.strip().split("\n")

    passed = False
    confidence = 0.5
    feedback = ""

    for line in lines:
        line = line.strip()
        if line.upper().startswith("PASSED:"):
            value = line.split(":", 1)[1].strip().lower()
            passed = value in ["yes", "true", "1"]
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            except ValueError:
                confidence = 0.5
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {
        "passed": passed,
        "confidence": confidence,
        "feedback": feedback
    }


def quick_verify(result: str, expected_type: str) -> bool:
    """
    Quick heuristic verification without AI.
    Use for simple checks before calling Claude.
    """
    if expected_type == "haiku":
        lines = [l for l in result.strip().split("\n") if l.strip()]
        return len(lines) == 3

    if expected_type == "list":
        # Check if result contains bullet points or numbers
        return any(c in result for c in ["-", "•", "1.", "1)"])

    if expected_type == "code":
        # Check for code indicators
        return any(kw in result for kw in ["def ", "function ", "class ", "const ", "let ", "var "])

    # Default: assume valid
    return True
```

## Step 3: Update State Manager

Add retry tracking to `state_manager.py`:

```python
def add_task(self, description: str, criteria: dict = None) -> dict:
    """Add a new task with optional verification criteria."""
    task = {
        "id": len(self.tasks) + 1,
        "description": description,
        "status": "pending",
        "result": None,
        "criteria": criteria or {},
        "attempts": 0,
        "max_attempts": 3,
        "verification": None  # Will hold verification result
    }
    self.tasks.append(task)
    self.save()
    return task

def increment_attempts(self, task_id: int):
    """Increment attempt counter for a task."""
    for task in self.tasks:
        if task["id"] == task_id:
            task["attempts"] = task.get("attempts", 0) + 1
            self.save()
            return

def mark_failed(self, task_id: int, reason: str):
    """Mark a task as permanently failed."""
    for task in self.tasks:
        if task["id"] == task_id:
            task["status"] = "failed"
            task["failure_reason"] = reason
            self.save()
            return

def mark_for_retry(self, task_id: int, feedback: str):
    """Reset task to pending for retry."""
    for task in self.tasks:
        if task["id"] == task_id:
            task["status"] = "pending"
            task["last_feedback"] = feedback
            self.save()
            return
```

## Step 4: Update the Main Loop

Create `loop_with_verification.py`:

```python
"""
Loop with Verification - Lab 03

Demonstrates verified task completion with retry logic.
"""

import anthropic
from state_manager import StateManager
from verifier import verify_result, quick_verify


client = anthropic.Anthropic()


def complete_task(task: dict) -> str:
    """Send task to Claude, including any feedback from previous attempts."""
    prompt = task["description"]

    # Include feedback from failed verification attempts
    if task.get("last_feedback"):
        prompt += f"\n\nPrevious attempt feedback: {task['last_feedback']}"
        prompt += "\nPlease address this feedback in your response."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def process_task(state: StateManager, task: dict) -> bool:
    """
    Process a single task with verification.
    Returns True if task completed successfully.
    """
    task_id = task["id"]
    attempt = task.get("attempts", 0) + 1
    max_attempts = task.get("max_attempts", 3)

    print(f"\n[Task {task_id}] Attempt {attempt}/{max_attempts}")
    print(f"  Description: {task['description'][:50]}...")

    # Increment attempt counter
    state.increment_attempts(task_id)
    state.mark_in_progress(task_id)

    # Generate result
    result = complete_task(task)
    print(f"  Generated result ({len(result)} chars)")

    # Quick heuristic check first (cheap)
    expected_type = task.get("criteria", {}).get("type")
    if expected_type and not quick_verify(result, expected_type):
        print(f"  ✗ Failed quick verification (not a valid {expected_type})")
        handle_failure(state, task, f"Result doesn't appear to be a valid {expected_type}")
        return False

    # Full AI verification (more expensive)
    print(f"  Verifying with AI...")
    verification = verify_result(task, result)

    if verification["passed"]:
        print(f"  ✓ Verified! (confidence: {verification['confidence']:.0%})")
        state.mark_complete(task_id, result)
        return True
    else:
        print(f"  ✗ Verification failed: {verification['feedback']}")
        handle_failure(state, task, verification["feedback"])
        return False


def handle_failure(state: StateManager, task: dict, feedback: str):
    """Handle a failed verification - retry or mark failed."""
    task_id = task["id"]
    attempts = task.get("attempts", 0)
    max_attempts = task.get("max_attempts", 3)

    if attempts >= max_attempts:
        print(f"  Max attempts reached. Marking as failed.")
        state.mark_failed(task_id, f"Failed after {attempts} attempts: {feedback}")
    else:
        print(f"  Marking for retry...")
        state.mark_for_retry(task_id, feedback)


def main():
    state = StateManager("state.json")

    # Load sample tasks if empty
    if len(state.tasks) == 0:
        print("Loading sample tasks with verification criteria...")

        state.add_task(
            "Write a haiku about programming",
            criteria={
                "type": "haiku",
                "requirements": [
                    "exactly 3 lines",
                    "follows 5-7-5 syllable pattern approximately",
                    "relates to programming or coding"
                ]
            }
        )

        state.add_task(
            "List exactly 5 benefits of version control",
            criteria={
                "type": "list",
                "requirements": [
                    "exactly 5 items",
                    "each item describes a benefit",
                    "items are about version control (git, etc.)"
                ]
            }
        )

        state.add_task(
            "Write a Python function that checks if a number is prime",
            criteria={
                "type": "code",
                "requirements": [
                    "valid Python syntax",
                    "function named is_prime or similar",
                    "returns True for prime, False otherwise",
                    "handles edge cases (0, 1, negative)"
                ]
            }
        )

    # Show stats
    stats = state.get_stats()
    print(f"\nTasks: {stats['completed']} completed, "
          f"{stats['pending']} pending, "
          f"{stats.get('failed', 0)} failed")

    # Process tasks
    while state.has_pending():
        task = state.get_next()
        process_task(state, task)

    # Final report
    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)

    for task in state.tasks:
        status_icon = {
            "completed": "✓",
            "failed": "✗",
            "pending": "○"
        }.get(task["status"], "?")

        print(f"\n{status_icon} Task {task['id']}: {task['description'][:40]}...")
        print(f"  Status: {task['status']}")
        print(f"  Attempts: {task.get('attempts', 0)}")

        if task["status"] == "failed":
            print(f"  Reason: {task.get('failure_reason', 'Unknown')}")


if __name__ == "__main__":
    main()
```

---

## Step 5: Run and Observe

```bash
python loop_with_verification.py
```

Example output:
```
Loading sample tasks with verification criteria...

Tasks: 0 completed, 3 pending, 0 failed

[Task 1] Attempt 1/3
  Description: Write a haiku about programming...
  Generated result (47 chars)
  Verifying with AI...
  ✓ Verified! (confidence: 95%)

[Task 2] Attempt 1/3
  Description: List exactly 5 benefits of version control...
  Generated result (312 chars)
  Verifying with AI...
  ✗ Verification failed: Listed 6 benefits instead of 5
  Marking for retry...

[Task 2] Attempt 2/3
  Description: List exactly 5 benefits of version control...
  Generated result (245 chars)
  Verifying with AI...
  ✓ Verified! (confidence: 92%)

[Task 3] Attempt 1/3
  Description: Write a Python function that checks if a n...
  Generated result (421 chars)
  ✗ Failed quick verification (not a valid code)
  Marking for retry...

[Task 3] Attempt 2/3
  Description: Write a Python function that checks if a n...
  Generated result (523 chars)
  Verifying with AI...
  ✓ Verified! (confidence: 98%)

==================================================
FINAL REPORT
==================================================

✓ Task 1: Write a haiku about programming...
  Status: completed
  Attempts: 1

✓ Task 2: List exactly 5 benefits of version contro...
  Status: completed
  Attempts: 2

✓ Task 3: Write a Python function that checks if a ...
  Status: completed
  Attempts: 2
```

---

## Understanding the Code

### The Verification Pipeline

```
Generate Result → Quick Check → AI Verification → Accept/Retry
       ↓              ↓               ↓              ↓
   (Claude)      (Heuristic)     (Claude)      (Decision)
```

1. **Quick Check**: Fast, cheap heuristics (line count, format checks)
2. **AI Verification**: Detailed evaluation against criteria
3. **Decision**: Accept, retry, or fail based on results

### Two-Tier Verification

```python
# Tier 1: Quick (free, fast)
if not quick_verify(result, expected_type):
    handle_failure(...)
    return

# Tier 2: AI (costs tokens, more accurate)
verification = verify_result(task, result)
```

This saves API costs by catching obvious failures early.

### Feedback Loop

When verification fails, we pass feedback to the next attempt:

```python
if task.get("last_feedback"):
    prompt += f"\nPrevious attempt feedback: {task['last_feedback']}"
```

This helps Claude learn from its mistakes within the same task.

---

## When to Verify

| Scenario | Verification Level |
|----------|-------------------|
| Low stakes (fun facts) | Quick check only |
| Medium stakes (content) | AI verification |
| High stakes (code, data) | AI + tests/execution |
| Critical (financial, medical) | Human review |

---

## Exercises

### Exercise 1: Add Test Execution

For code tasks, actually run the generated code:

```python
def verify_code(code: str, test_cases: list) -> bool:
    """Execute code and verify against test cases."""
    # Hint: Use exec() carefully with restricted globals
    pass
```

### Exercise 2: Confidence Thresholds

Modify the loop to accept results below 90% confidence only after 2+ attempts:

```python
if verification["confidence"] >= 0.9:
    accept()
elif verification["confidence"] >= 0.7 and attempts >= 2:
    accept()  # Good enough after retries
else:
    retry()
```

### Exercise 3: Verification Cost Tracking

Track how many verification API calls are made and estimate cost:

```python
verification_calls = 0
estimated_cost = verification_calls * 0.003  # ~$0.003 per call
```

---

## Checkpoint

Before moving on, verify:
- [ ] Your loop retries failed verifications
- [ ] Feedback from failures is passed to retry attempts
- [ ] Tasks are marked failed after max attempts
- [ ] You understand the two-tier verification approach

---

## Key Takeaway

> Verification turns "probably correct" into "verified correct."

Without verification, you're trusting that every AI response is perfect. With verification:
- Errors are caught before they compound
- Retries improve quality
- Failed tasks are clearly identified
- You know what actually succeeded

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

The Tier 1 implementation includes verification via tool calling (Lab 05) and circuit breakers (Lab 06).

---

<div class="lab-navigation">
  <a href="./02-external-state.md" class="prev">← Previous: Lab 02 - External State</a>
  <a href="./04-json-tasks.md" class="next">Next: Lab 04 - JSON Task Management →</a>
</div>
