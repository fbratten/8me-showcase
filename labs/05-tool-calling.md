---
layout: default
title: "Lab 05: Tool Calling"
nav_order: 5
parent: Labs
lab_number: 5
difficulty: intermediate
time: 1 hour
prerequisites: Lab 04 completed
---

# Lab 05: Tool Calling

Use Claude's native tool calling for structured, reliable AI output.

## Objectives

By the end of this lab, you will:
- Understand Claude's tool calling mechanism
- Define tools with JSON schemas
- Force structured output with tool_choice
- Parse and handle tool responses reliably

## Prerequisites

- Lab 04 completed (TaskManager)
- Basic understanding of JSON Schema

## The Problem with Free-Form Output

In earlier labs, we parsed Claude's text responses:

```python
# Lab 03: Parsing free-form verification response
response_text = "PASSED: yes\nCONFIDENCE: 0.95\nFEEDBACK: Looks good"

# Fragile parsing
for line in response_text.split("\n"):
    if line.startswith("PASSED:"):
        passed = "yes" in line.lower()  # What if Claude says "Yeah"?
```

This approach is **fragile** because:
- Claude might use different formats ("Yes", "TRUE", "Affirmative")
- Line order might change
- Extra text might appear
- Regex patterns break on edge cases

## The Solution: Tool Calling

Tool calling forces Claude to respond with **structured JSON**:

```python
# Define the structure you want
tools = [{
    "name": "submit_result",
    "description": "Submit the completed task result",
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["result", "confidence"]
    }
}]

# Claude MUST respond with this structure
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    tool_choice={"type": "any"},  # Force tool use
    messages=[...]
)

# Guaranteed structure!
tool_input = response.content[0].input
result = tool_input["result"]       # Always exists
confidence = tool_input["confidence"]  # Always a number 0-1
```

---

## Step 1: Define Your Tools

Create `tools.py`:

```python
"""
Tool Definitions - Lab 05

Defines tools for structured AI output in task processing.
"""

# Tool for submitting a completed task
SUBMIT_RESULT_TOOL = {
    "name": "submit_result",
    "description": "Submit the completed result for a task. Use this when you have successfully completed the task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": "The complete result/output for the task"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Your confidence in the result (0.0 to 1.0)"
            },
            "notes": {
                "type": "string",
                "description": "Optional notes about the approach or limitations"
            }
        },
        "required": ["result", "confidence"]
    }
}

# Tool for reporting inability to complete
REPORT_FAILURE_TOOL = {
    "name": "report_failure",
    "description": "Report that you cannot complete the task. Use this when the task is impossible, unclear, or you've encountered an insurmountable problem.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the task cannot be completed"
            },
            "category": {
                "type": "string",
                "enum": ["impossible", "unclear", "missing_info", "out_of_scope", "other"],
                "description": "Category of failure"
            },
            "suggestion": {
                "type": "string",
                "description": "Suggested modification to make the task completable"
            }
        },
        "required": ["reason", "category"]
    }
}

# Tool for requesting clarification
REQUEST_CLARIFICATION_TOOL = {
    "name": "request_clarification",
    "description": "Request clarification before proceeding. Use this when the task is ambiguous and you need more information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The specific question you need answered"
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Possible answers/options if applicable"
            },
            "default": {
                "type": "string",
                "description": "What you'll assume if no clarification is provided"
            }
        },
        "required": ["question"]
    }
}

# Tool for verification results
VERIFICATION_RESULT_TOOL = {
    "name": "verification_result",
    "description": "Submit the result of verifying a task output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": "Whether the output passes verification"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in this verification judgment"
            },
            "feedback": {
                "type": "string",
                "description": "Specific feedback about what passed or failed"
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific issues found (if any)"
            }
        },
        "required": ["passed", "confidence", "feedback"]
    }
}

# Combined tool sets for different operations
TASK_COMPLETION_TOOLS = [
    SUBMIT_RESULT_TOOL,
    REPORT_FAILURE_TOOL,
    REQUEST_CLARIFICATION_TOOL
]

VERIFICATION_TOOLS = [
    VERIFICATION_RESULT_TOOL
]

ALL_TOOLS = [
    SUBMIT_RESULT_TOOL,
    REPORT_FAILURE_TOOL,
    REQUEST_CLARIFICATION_TOOL,
    VERIFICATION_RESULT_TOOL
]
```

---

## Step 2: Create the Tool Handler

Create `tool_handler.py`:

```python
"""
Tool Handler - Lab 05

Processes Claude's tool call responses.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ToolCallType(Enum):
    """Types of tool calls we handle."""
    SUBMIT_RESULT = "submit_result"
    REPORT_FAILURE = "report_failure"
    REQUEST_CLARIFICATION = "request_clarification"
    VERIFICATION_RESULT = "verification_result"
    UNKNOWN = "unknown"


@dataclass
class ToolCallResult:
    """Parsed result from a tool call."""
    tool_name: str
    tool_type: ToolCallType
    inputs: Dict[str, Any]
    raw_response: Any

    @property
    def is_success(self) -> bool:
        """Check if this represents a successful completion."""
        return self.tool_type == ToolCallType.SUBMIT_RESULT

    @property
    def is_failure(self) -> bool:
        """Check if this represents a failure."""
        return self.tool_type == ToolCallType.REPORT_FAILURE

    @property
    def needs_clarification(self) -> bool:
        """Check if clarification is needed."""
        return self.tool_type == ToolCallType.REQUEST_CLARIFICATION

    @property
    def is_verification(self) -> bool:
        """Check if this is a verification result."""
        return self.tool_type == ToolCallType.VERIFICATION_RESULT


def parse_tool_response(response) -> Optional[ToolCallResult]:
    """
    Parse a Claude API response containing tool calls.

    Args:
        response: The full response from client.messages.create()

    Returns:
        ToolCallResult if a tool was called, None otherwise
    """
    # Find the tool_use block in the response
    tool_use_block = None
    for block in response.content:
        if block.type == "tool_use":
            tool_use_block = block
            break

    if not tool_use_block:
        return None

    # Determine tool type
    tool_name = tool_use_block.name
    try:
        tool_type = ToolCallType(tool_name)
    except ValueError:
        tool_type = ToolCallType.UNKNOWN

    return ToolCallResult(
        tool_name=tool_name,
        tool_type=tool_type,
        inputs=tool_use_block.input,
        raw_response=response
    )


def extract_result(tool_result: ToolCallResult) -> Tuple[str, float]:
    """
    Extract result and confidence from a submit_result tool call.

    Returns:
        Tuple of (result_text, confidence)
    """
    if tool_result.tool_type != ToolCallType.SUBMIT_RESULT:
        raise ValueError(f"Expected submit_result, got {tool_result.tool_type}")

    return (
        tool_result.inputs.get("result", ""),
        tool_result.inputs.get("confidence", 0.5)
    )


def extract_failure(tool_result: ToolCallResult) -> Tuple[str, str, Optional[str]]:
    """
    Extract failure details from a report_failure tool call.

    Returns:
        Tuple of (reason, category, suggestion)
    """
    if tool_result.tool_type != ToolCallType.REPORT_FAILURE:
        raise ValueError(f"Expected report_failure, got {tool_result.tool_type}")

    return (
        tool_result.inputs.get("reason", "Unknown reason"),
        tool_result.inputs.get("category", "other"),
        tool_result.inputs.get("suggestion")
    )


def extract_verification(tool_result: ToolCallResult) -> Tuple[bool, float, str, list]:
    """
    Extract verification details from a verification_result tool call.

    Returns:
        Tuple of (passed, confidence, feedback, issues)
    """
    if tool_result.tool_type != ToolCallType.VERIFICATION_RESULT:
        raise ValueError(f"Expected verification_result, got {tool_result.tool_type}")

    return (
        tool_result.inputs.get("passed", False),
        tool_result.inputs.get("confidence", 0.5),
        tool_result.inputs.get("feedback", ""),
        tool_result.inputs.get("issues", [])
    )
```

---

## Step 3: Update the Task Executor

Create `executor.py`:

```python
"""
Task Executor with Tool Calling - Lab 05

Executes tasks using Claude with structured tool responses.
"""

import anthropic
from tools import TASK_COMPLETION_TOOLS, VERIFICATION_TOOLS
from tool_handler import (
    parse_tool_response,
    extract_result,
    extract_failure,
    extract_verification,
    ToolCallType
)


client = anthropic.Anthropic()


def execute_task(task: dict) -> dict:
    """
    Execute a task and get structured response via tool calling.

    Args:
        task: Task dict with 'description' and optional 'criteria'

    Returns:
        Dict with execution result:
        {
            "status": "completed" | "failed" | "needs_clarification",
            "result": str (if completed),
            "confidence": float (if completed),
            "reason": str (if failed),
            "question": str (if needs_clarification),
            ...
        }
    """
    # Build the prompt
    prompt = f"""Complete the following task. Use the appropriate tool to submit your response.

TASK: {task['description']}
"""

    if task.get("criteria"):
        criteria = task["criteria"]
        prompt += f"\nREQUIREMENTS:\n"
        if criteria.get("type"):
            prompt += f"- Output type: {criteria['type']}\n"
        for req in criteria.get("requirements", []):
            prompt += f"- {req}\n"

    if task.get("last_feedback"):
        prompt += f"\nPREVIOUS FEEDBACK: {task['last_feedback']}\n"
        prompt += "Please address this feedback in your response.\n"

    # Call Claude with tools
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=TASK_COMPLETION_TOOLS,
        tool_choice={"type": "any"},  # Force tool use
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the tool response
    tool_result = parse_tool_response(response)

    if not tool_result:
        # Shouldn't happen with tool_choice="any", but handle it
        return {
            "status": "failed",
            "reason": "No tool call in response",
            "category": "other"
        }

    # Handle based on tool type
    if tool_result.is_success:
        result, confidence = extract_result(tool_result)
        return {
            "status": "completed",
            "result": result,
            "confidence": confidence,
            "notes": tool_result.inputs.get("notes")
        }

    elif tool_result.is_failure:
        reason, category, suggestion = extract_failure(tool_result)
        return {
            "status": "failed",
            "reason": reason,
            "category": category,
            "suggestion": suggestion
        }

    elif tool_result.needs_clarification:
        return {
            "status": "needs_clarification",
            "question": tool_result.inputs.get("question"),
            "options": tool_result.inputs.get("options", []),
            "default": tool_result.inputs.get("default")
        }

    else:
        return {
            "status": "failed",
            "reason": f"Unknown tool: {tool_result.tool_name}",
            "category": "other"
        }


def verify_result(task: dict, result: str) -> dict:
    """
    Verify a result using tool calling for structured output.

    Args:
        task: The original task with criteria
        result: The result to verify

    Returns:
        Dict with verification result:
        {
            "passed": bool,
            "confidence": float,
            "feedback": str,
            "issues": list
        }
    """
    criteria = task.get("criteria", {})

    prompt = f"""You are a verification assistant. Evaluate whether this result meets the criteria.

ORIGINAL TASK: {task['description']}

CRITERIA:
"""
    if criteria.get("type"):
        prompt += f"- Type: {criteria['type']}\n"
    for req in criteria.get("requirements", []):
        prompt += f"- {req}\n"

    if not criteria:
        prompt += "- Result should reasonably complete the task\n"

    prompt += f"""
RESULT TO VERIFY:
{result}

Use the verification_result tool to submit your evaluation. Be strict but fair.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        tools=VERIFICATION_TOOLS,
        tool_choice={"type": "tool", "name": "verification_result"},
        messages=[{"role": "user", "content": prompt}]
    )

    tool_result = parse_tool_response(response)

    if tool_result and tool_result.is_verification:
        passed, confidence, feedback, issues = extract_verification(tool_result)
        return {
            "passed": passed,
            "confidence": confidence,
            "feedback": feedback,
            "issues": issues
        }

    # Fallback if something went wrong
    return {
        "passed": False,
        "confidence": 0.0,
        "feedback": "Verification failed to produce structured output",
        "issues": ["Tool call parsing failed"]
    }
```

---

## Step 4: Update the Main Loop

Create `loop_with_tools.py`:

```python
"""
Loop with Tool Calling - Lab 05

Demonstrates structured AI output via tool calling.
"""

from task_manager import TaskManager
from executor import execute_task, verify_result


def process_task(manager: TaskManager, task) -> bool:
    """Process a single task with tool calling."""
    task_id = task.id
    attempt = task.attempts + 1

    print(f"\n[{task_id}] Attempt {attempt}/{task.max_attempts}")
    print(f"  Task: {task.description[:50]}...")

    manager.start(task_id)

    # Execute with tool calling
    result = execute_task(task.to_dict())

    if result["status"] == "completed":
        print(f"  Result: {result['result'][:50]}...")
        print(f"  Confidence: {result['confidence']:.0%}")

        # Verify if criteria exist
        if task.criteria:
            print(f"  Verifying...")
            verification = verify_result(task.to_dict(), result["result"])

            if verification["passed"]:
                print(f"  ✓ Verified ({verification['confidence']:.0%})")
                manager.complete(task_id, result["result"])
                return True
            else:
                print(f"  ✗ Verification failed: {verification['feedback']}")
                if verification["issues"]:
                    for issue in verification["issues"]:
                        print(f"    - {issue}")

                # Retry or fail
                if attempt < task.max_attempts:
                    manager.retry(task_id, verification["feedback"])
                else:
                    manager.fail(task_id, f"Failed verification: {verification['feedback']}")
                return False
        else:
            # No criteria, accept based on confidence
            if result["confidence"] >= 0.7:
                manager.complete(task_id, result["result"])
                return True
            else:
                if attempt < task.max_attempts:
                    manager.retry(task_id, "Low confidence")
                else:
                    manager.complete(task_id, result["result"])  # Accept anyway
                return True

    elif result["status"] == "failed":
        print(f"  ✗ Failed: {result['reason']}")
        print(f"    Category: {result['category']}")
        if result.get("suggestion"):
            print(f"    Suggestion: {result['suggestion']}")

        manager.fail(task_id, result["reason"])
        return False

    elif result["status"] == "needs_clarification":
        print(f"  ? Needs clarification: {result['question']}")
        if result.get("options"):
            print(f"    Options: {', '.join(result['options'])}")
        if result.get("default"):
            print(f"    Default: {result['default']}")

        # In a real system, you'd prompt the user here
        # For now, use the default or retry
        if result.get("default"):
            # Update task with clarification and retry
            manager.update(task_id, last_feedback=f"Clarification: {result['default']}")
            manager.retry(task_id, f"Using default: {result['default']}")
        else:
            manager.fail(task_id, f"Needs clarification: {result['question']}")
        return False

    return False


def main():
    manager = TaskManager("tasks.json")

    # Create sample tasks if empty
    if not manager.tasks:
        print("Creating sample tasks...\n")

        manager.create(
            "Write a haiku about Python programming",
            criteria={
                "type": "haiku",
                "requirements": [
                    "Exactly 3 lines",
                    "Approximately 5-7-5 syllable pattern",
                    "Related to Python programming"
                ]
            }
        )

        manager.create(
            "List exactly 3 advantages of type hints in Python",
            criteria={
                "type": "list",
                "requirements": [
                    "Exactly 3 items",
                    "Each item is an advantage of type hints",
                    "Clear and concise explanations"
                ]
            }
        )

        manager.create(
            "Calculate the factorial of 7"
            # No criteria - will accept based on confidence
        )

    # Process loop
    stats = manager.get_stats()
    print(f"Tasks: {stats['pending']} pending, {stats['completed']} completed, {stats['failed']} failed")

    while manager.has_pending():
        task = manager.get_next()
        process_task(manager, task)

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    stats = manager.get_stats()
    print(f"\nCompletion rate: {stats['completion_rate']:.0f}%")
    print(f"Failure rate: {stats['failure_rate']:.0f}%")
    print(f"Average attempts: {stats['average_attempts']}")

    for task in manager.get_all():
        icon = {"completed": "✓", "failed": "✗", "pending": "○"}.get(task.status, "?")
        print(f"\n{icon} {task.description[:50]}...")
        print(f"  Status: {task.status}, Attempts: {task.attempts}")
        if task.status == "completed" and task.result:
            print(f"  Result: {task.result[:60]}...")


if __name__ == "__main__":
    main()
```

---

## Understanding Tool Calling

### How It Works

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Your Code     │──────│    Claude API   │──────│   Claude LLM    │
│                 │      │                 │      │                 │
│ tools=[...]     │      │ Validates       │      │ Generates       │
│ tool_choice=any │─────▶│ schema          │─────▶│ structured JSON │
│                 │      │                 │      │                 │
│ response.content│◀─────│ Returns         │◀─────│ via tool_use    │
│ [0].input       │      │ tool_use block  │      │ block           │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### Tool Choice Options

```python
# Let Claude decide whether to use a tool
tool_choice={"type": "auto"}

# Force Claude to use ANY of the provided tools
tool_choice={"type": "any"}

# Force Claude to use a SPECIFIC tool
tool_choice={"type": "tool", "name": "submit_result"}
```

### JSON Schema Power

```python
{
    "type": "object",
    "properties": {
        "confidence": {
            "type": "number",
            "minimum": 0,        # Guaranteed range!
            "maximum": 1
        },
        "category": {
            "type": "string",
            "enum": ["a", "b", "c"]  # Guaranteed values!
        }
    },
    "required": ["confidence"]  # Guaranteed presence!
}
```

---

## Comparison: Text Parsing vs Tool Calling

| Aspect | Text Parsing | Tool Calling |
|--------|--------------|--------------|
| Output format | Unpredictable | Guaranteed JSON |
| Parsing | Regex/string ops | Direct dict access |
| Validation | Manual | Schema-enforced |
| Error handling | Complex | Simple |
| Type safety | None | Built-in |
| Maintenance | Fragile | Robust |

---

## Exercises

### Exercise 1: Add a Progress Tool

Create a tool for reporting partial progress:

```python
REPORT_PROGRESS_TOOL = {
    "name": "report_progress",
    "input_schema": {
        "properties": {
            "percent_complete": {"type": "integer", "minimum": 0, "maximum": 100},
            "current_step": {"type": "string"},
            "partial_result": {"type": "string"}
        }
    }
}
```

### Exercise 2: Multi-Tool Responses

Handle cases where Claude might want to use multiple tools in sequence (e.g., clarify then submit).

### Exercise 3: Tool Call Logging

Add detailed logging of all tool calls for debugging:

```python
def log_tool_call(tool_result: ToolCallResult):
    """Log tool call details to a file."""
    pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Tool definitions compile without errors
- [ ] Claude responds with tool_use blocks
- [ ] Tool responses parse correctly
- [ ] Verification uses structured output
- [ ] You understand tool_choice options

---

## Key Takeaway

> Tool calling beats regex parsing for structured AI output.

With tool calling:
- **Guaranteed structure**: JSON schema enforces format
- **Type safety**: Numbers are numbers, booleans are booleans
- **Validation built-in**: Enums, ranges, required fields
- **Simpler code**: No regex, no string parsing
- **More reliable**: No edge cases from format variations

---

## Get the Code

Full implementation: [8me/src/tier1-ralph-loop/claude_tools.py](https://github.com/fbratten/8me/tree/main/src/tier1-ralph-loop)

---

<div class="lab-navigation">
  <a href="./04-json-tasks" class="prev">← Previous: Lab 04 - JSON Task Management</a>
  <a href="./06-circuit-breakers" class="next">Next: Lab 06 - Circuit Breakers →</a>
</div>
