---
layout: default
title: "Lab 10: Gating and Drift Prevention"
nav_order: 10
parent: Labs
lab_number: 10
difficulty: advanced
time: 1.5 hours
prerequisites: Lab 09 completed
---

# Lab 10: Gating and Drift Prevention

Implement quality gates and detect when outputs drift from intent.

## Objectives

By the end of this lab, you will:
- Implement pre-execution gates (preconditions)
- Implement post-execution gates (validation)
- Detect semantic drift from original intent
- Handle gate failures gracefully

## Prerequisites

- Lab 09 completed (multi-agent)
- Understanding of the orchestrator pattern

## Why Gates?

Without gates, errors compound:

```
Task: "Write a sorting function"
     │
     ▼
Iteration 1: Writes search function (wrong!)
     │
     ▼
Iteration 2: "Improves" the search function
     │
     ▼
Iteration 3: Adds features to the search function
     │
     ▼
Result: Great search function, but task was SORTING
```

Gates catch problems early:

```
Task: "Write a sorting function"
     │
     ▼
PRE-GATE: Is task well-defined? ✓
     │
     ▼
Iteration 1: Writes search function
     │
     ▼
POST-GATE: Does output match intent? ✗ DRIFT DETECTED
     │
     ▼
Correction: Re-execute with explicit "must sort, not search"
```

---

## The Gate Pattern

```python
def gated_execute(task):
    # Pre-gate: Check if we should proceed
    if not passes_pre_gate(task):
        return reject("Pre-conditions not met")

    # Execute
    result = execute(task)

    # Post-gate: Check if result is valid
    if not passes_post_gate(task, result):
        return rollback_or_retry("Post-conditions failed")

    return commit(result)
```

---

## Step 1: Create the Gate Framework

Create `gates.py`:

```python
"""
Gating Framework - Lab 10

Pre and post execution gates for quality control.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class GateResult(Enum):
    """Result of a gate check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"  # Pass with warning


@dataclass
class GateCheckResult:
    """Detailed result from a gate check."""
    result: GateResult
    gate_name: str
    message: str
    details: Optional[Dict[str, Any]] = None

    @property
    def passed(self) -> bool:
        return self.result in (GateResult.PASS, GateResult.WARN)


@dataclass
class GateReport:
    """Full report from all gate checks."""
    checks: List[GateCheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.result == GateResult.WARN for c in self.checks)

    @property
    def failures(self) -> List[GateCheckResult]:
        return [c for c in self.checks if c.result == GateResult.FAIL]

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.result == GateResult.PASS)
        warned = sum(1 for c in self.checks if c.result == GateResult.WARN)
        failed = sum(1 for c in self.checks if c.result == GateResult.FAIL)
        return f"Gates: {passed} passed, {warned} warned, {failed} failed"


class Gate(ABC):
    """Abstract base class for gates."""

    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required

    @abstractmethod
    def check(self, **kwargs) -> GateCheckResult:
        """Run the gate check."""
        pass


class PreGate(Gate):
    """Gate that runs before execution."""
    pass


class PostGate(Gate):
    """Gate that runs after execution."""
    pass


class GateRunner:
    """
    Runs a collection of gates.

    Usage:
        runner = GateRunner()
        runner.add_pre_gate(TaskDefinedGate())
        runner.add_post_gate(OutputLengthGate())

        pre_report = runner.run_pre_gates(task=task)
        if pre_report.passed:
            result = execute(task)
            post_report = runner.run_post_gates(task=task, result=result)
    """

    def __init__(self):
        self.pre_gates: List[PreGate] = []
        self.post_gates: List[PostGate] = []

    def add_pre_gate(self, gate: PreGate) -> "GateRunner":
        self.pre_gates.append(gate)
        return self

    def add_post_gate(self, gate: PostGate) -> "GateRunner":
        self.post_gates.append(gate)
        return self

    def run_pre_gates(self, **kwargs) -> GateReport:
        """Run all pre-execution gates."""
        results = []
        for gate in self.pre_gates:
            try:
                result = gate.check(**kwargs)
                results.append(result)

                # Stop on required gate failure
                if gate.required and result.result == GateResult.FAIL:
                    break
            except Exception as e:
                results.append(GateCheckResult(
                    result=GateResult.FAIL,
                    gate_name=gate.name,
                    message=f"Gate error: {e}"
                ))
                if gate.required:
                    break

        return GateReport(checks=results)

    def run_post_gates(self, **kwargs) -> GateReport:
        """Run all post-execution gates."""
        results = []
        for gate in self.post_gates:
            try:
                result = gate.check(**kwargs)
                results.append(result)
            except Exception as e:
                results.append(GateCheckResult(
                    result=GateResult.FAIL,
                    gate_name=gate.name,
                    message=f"Gate error: {e}"
                ))

        return GateReport(checks=results)
```

---

## Step 2: Implement Common Gates

Add to `gates.py`:

```python
# ==================== Pre-Gates ====================

class TaskDefinedGate(PreGate):
    """Ensure task is properly defined."""

    def __init__(self, min_length: int = 10):
        super().__init__("task_defined")
        self.min_length = min_length

    def check(self, task: str, **kwargs) -> GateCheckResult:
        if not task or len(task.strip()) < self.min_length:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Task too short (min {self.min_length} chars)"
            )

        # Check for vague tasks
        vague_phrases = ["do something", "fix it", "make it work", "whatever"]
        if any(phrase in task.lower() for phrase in vague_phrases):
            return GateCheckResult(
                result=GateResult.WARN,
                gate_name=self.name,
                message="Task may be too vague"
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message="Task is well-defined"
        )


class ResourceAvailableGate(PreGate):
    """Check if required resources are available."""

    def __init__(self, required_resources: List[str]):
        super().__init__("resource_available")
        self.required = required_resources

    def check(self, resources: Dict[str, Any] = None, **kwargs) -> GateCheckResult:
        resources = resources or {}
        missing = [r for r in self.required if r not in resources]

        if missing:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Missing resources: {missing}"
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message="All resources available"
        )


class BudgetGate(PreGate):
    """Check if budget allows execution."""

    def __init__(self, max_cost: float):
        super().__init__("budget")
        self.max_cost = max_cost

    def check(self, current_cost: float = 0, estimated_cost: float = 0, **kwargs) -> GateCheckResult:
        if current_cost + estimated_cost > self.max_cost:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Would exceed budget (${current_cost + estimated_cost:.2f} > ${self.max_cost:.2f})"
            )

        if current_cost + estimated_cost > self.max_cost * 0.8:
            return GateCheckResult(
                result=GateResult.WARN,
                gate_name=self.name,
                message=f"Approaching budget limit ({(current_cost + estimated_cost) / self.max_cost * 100:.0f}%)"
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message="Within budget"
        )


# ==================== Post-Gates ====================

class OutputLengthGate(PostGate):
    """Check if output meets length requirements."""

    def __init__(self, min_length: int = 1, max_length: int = 10000):
        super().__init__("output_length")
        self.min_length = min_length
        self.max_length = max_length

    def check(self, result: str, **kwargs) -> GateCheckResult:
        length = len(result) if result else 0

        if length < self.min_length:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Output too short ({length} < {self.min_length})"
            )

        if length > self.max_length:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Output too long ({length} > {self.max_length})"
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message=f"Output length OK ({length} chars)"
        )


class FormatGate(PostGate):
    """Check if output matches expected format."""

    def __init__(self, expected_format: str):
        """
        Args:
            expected_format: "json", "markdown", "code", "list", etc.
        """
        super().__init__("format")
        self.expected_format = expected_format

    def check(self, result: str, **kwargs) -> GateCheckResult:
        if self.expected_format == "json":
            return self._check_json(result)
        elif self.expected_format == "markdown":
            return self._check_markdown(result)
        elif self.expected_format == "code":
            return self._check_code(result)
        elif self.expected_format == "list":
            return self._check_list(result)
        else:
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message="Format check skipped (unknown format)"
            )

    def _check_json(self, result: str) -> GateCheckResult:
        import json
        try:
            json.loads(result)
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message="Valid JSON"
            )
        except:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message="Invalid JSON"
            )

    def _check_markdown(self, result: str) -> GateCheckResult:
        if "#" in result or "**" in result or "-" in result:
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message="Appears to be Markdown"
            )
        return GateCheckResult(
            result=GateResult.WARN,
            gate_name=self.name,
            message="May not be Markdown"
        )

    def _check_code(self, result: str) -> GateCheckResult:
        code_indicators = ["def ", "function ", "class ", "const ", "let ", "var ", "import ", "from "]
        if any(ind in result for ind in code_indicators):
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message="Contains code"
            )
        return GateCheckResult(
            result=GateResult.WARN,
            gate_name=self.name,
            message="May not contain code"
        )

    def _check_list(self, result: str) -> GateCheckResult:
        lines = result.strip().split("\n")
        list_lines = [l for l in lines if l.strip().startswith(("-", "*", "•")) or
                      (l.strip() and l.strip()[0].isdigit() and "." in l[:3])]
        if len(list_lines) >= 2:
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message=f"Contains list ({len(list_lines)} items)"
            )
        return GateCheckResult(
            result=GateResult.FAIL,
            gate_name=self.name,
            message="Does not appear to be a list"
        )


class ConfidenceGate(PostGate):
    """Check if confidence meets threshold."""

    def __init__(self, min_confidence: float = 0.7):
        super().__init__("confidence")
        self.min_confidence = min_confidence

    def check(self, confidence: float = 0, **kwargs) -> GateCheckResult:
        if confidence >= self.min_confidence:
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message=f"Confidence OK ({confidence:.0%})"
            )

        if confidence >= self.min_confidence * 0.8:
            return GateCheckResult(
                result=GateResult.WARN,
                gate_name=self.name,
                message=f"Confidence marginal ({confidence:.0%})"
            )

        return GateCheckResult(
            result=GateResult.FAIL,
            gate_name=self.name,
            message=f"Confidence too low ({confidence:.0%})"
        )
```

---

## Step 3: Implement Drift Detection

Create `drift_detection.py`:

```python
"""
Drift Detection - Lab 10

Detect when outputs drift from original intent.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import anthropic

from gates import PostGate, GateCheckResult, GateResult


@dataclass
class DriftAnalysis:
    """Result of drift analysis."""
    is_drifting: bool
    similarity_score: float  # 0-1, higher = more similar
    drift_description: str
    suggested_correction: Optional[str] = None


class DriftDetector:
    """
    Detects semantic drift from original intent.

    Uses AI to compare output against original task intent.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def analyze(self, task: str, output: str, context: str = "") -> DriftAnalysis:
        """
        Analyze if output has drifted from task intent.

        Args:
            task: Original task description
            output: The generated output
            context: Optional additional context

        Returns:
            DriftAnalysis with similarity score and description
        """
        prompt = f"""Analyze whether this output matches the original task intent.

ORIGINAL TASK:
{task}

{f"CONTEXT: {context}" if context else ""}

OUTPUT TO ANALYZE:
{output}

Evaluate:
1. Does the output address the task?
2. Has the output drifted to a different topic?
3. Are there any misunderstandings?

Respond in this format:
SIMILARITY: (0.0 to 1.0, where 1.0 = perfect match)
DRIFTING: (yes/no)
DESCRIPTION: (brief explanation)
CORRECTION: (if drifting, how to correct - otherwise "N/A")
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response.content[0].text)

    def _parse_response(self, text: str) -> DriftAnalysis:
        """Parse the AI response."""
        similarity = 0.5
        is_drifting = False
        description = ""
        correction = None

        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SIMILARITY:"):
                try:
                    similarity = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.upper().startswith("DRIFTING:"):
                is_drifting = "yes" in line.lower()
            elif line.upper().startswith("DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CORRECTION:"):
                corr = line.split(":", 1)[1].strip()
                if corr.lower() != "n/a":
                    correction = corr

        return DriftAnalysis(
            is_drifting=is_drifting,
            similarity_score=similarity,
            drift_description=description,
            suggested_correction=correction
        )


class DriftGate(PostGate):
    """Post-gate that checks for semantic drift."""

    def __init__(
        self,
        min_similarity: float = 0.7,
        detector: Optional[DriftDetector] = None
    ):
        super().__init__("drift_detection")
        self.min_similarity = min_similarity
        self.detector = detector or DriftDetector()

    def check(self, task: str, result: str, **kwargs) -> GateCheckResult:
        analysis = self.detector.analyze(task, result)

        if analysis.is_drifting or analysis.similarity_score < self.min_similarity:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Drift detected: {analysis.drift_description}",
                details={
                    "similarity": analysis.similarity_score,
                    "correction": analysis.suggested_correction
                }
            )

        if analysis.similarity_score < self.min_similarity + 0.1:
            return GateCheckResult(
                result=GateResult.WARN,
                gate_name=self.name,
                message=f"Possible drift: {analysis.drift_description}",
                details={"similarity": analysis.similarity_score}
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message=f"No drift detected (similarity: {analysis.similarity_score:.0%})"
        )


class KeywordDriftGate(PostGate):
    """
    Simple drift detection based on keyword overlap.

    Faster and cheaper than AI-based detection.
    """

    def __init__(self, min_overlap: float = 0.3):
        super().__init__("keyword_drift")
        self.min_overlap = min_overlap

    def check(self, task: str, result: str, **kwargs) -> GateCheckResult:
        # Extract keywords (simple approach)
        task_words = self._extract_keywords(task)
        result_words = self._extract_keywords(result)

        if not task_words:
            return GateCheckResult(
                result=GateResult.PASS,
                gate_name=self.name,
                message="No keywords to compare"
            )

        # Calculate overlap
        overlap = len(task_words & result_words) / len(task_words)

        if overlap < self.min_overlap:
            return GateCheckResult(
                result=GateResult.FAIL,
                gate_name=self.name,
                message=f"Low keyword overlap ({overlap:.0%})",
                details={
                    "task_keywords": list(task_words),
                    "result_keywords": list(result_words),
                    "overlap": overlap
                }
            )

        return GateCheckResult(
            result=GateResult.PASS,
            gate_name=self.name,
            message=f"Keyword overlap OK ({overlap:.0%})"
        )

    def _extract_keywords(self, text: str) -> set:
        """Extract significant words from text."""
        # Simple keyword extraction
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "and", "but", "if", "or", "because", "until",
            "while", "this", "that", "these", "those", "it", "its"
        }

        words = set()
        for word in text.lower().split():
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2 and word not in stop_words:
                words.add(word)

        return words
```

---

## Step 4: Complete Example

Create `gating_demo.py`:

```python
"""
Gating Demo - Lab 10

Demonstrates pre/post gates and drift detection.
"""

from gates import (
    GateRunner, TaskDefinedGate, BudgetGate,
    OutputLengthGate, FormatGate, ConfidenceGate
)
from drift_detection import DriftGate, KeywordDriftGate


def demo_gates():
    """Demonstrate gating in action."""
    print("=" * 60)
    print("GATING DEMO")
    print("=" * 60)

    # Set up gate runner
    runner = GateRunner()

    # Pre-gates
    runner.add_pre_gate(TaskDefinedGate(min_length=10))
    runner.add_pre_gate(BudgetGate(max_cost=1.0))

    # Post-gates
    runner.add_post_gate(OutputLengthGate(min_length=10, max_length=5000))
    runner.add_post_gate(FormatGate("list"))
    runner.add_post_gate(ConfidenceGate(min_confidence=0.7))
    runner.add_post_gate(KeywordDriftGate(min_overlap=0.2))

    # Test cases
    test_cases = [
        {
            "name": "Good task",
            "task": "List 5 benefits of using Python for data science",
            "result": """Here are 5 benefits of using Python for data science:
1. Rich ecosystem of libraries (NumPy, Pandas, Scikit-learn)
2. Easy to learn and readable syntax
3. Strong community support
4. Excellent data visualization tools
5. Integration with big data platforms""",
            "confidence": 0.9,
            "current_cost": 0.10
        },
        {
            "name": "Vague task",
            "task": "Do something",
            "result": "OK",
            "confidence": 0.5,
            "current_cost": 0.10
        },
        {
            "name": "Drifted output",
            "task": "Write a sorting algorithm",
            "result": """Here's a great recipe for chocolate cake:
1. Preheat oven to 350°F
2. Mix flour and sugar
3. Add eggs and butter
4. Bake for 30 minutes""",
            "confidence": 0.8,
            "current_cost": 0.10
        },
        {
            "name": "Over budget",
            "task": "Analyze this large dataset in detail",
            "result": "Analysis complete",
            "confidence": 0.9,
            "current_cost": 0.95,
            "estimated_cost": 0.10
        }
    ]

    for case in test_cases:
        print(f"\n--- Test: {case['name']} ---")
        print(f"Task: {case['task'][:50]}...")

        # Run pre-gates
        pre_report = runner.run_pre_gates(
            task=case["task"],
            current_cost=case.get("current_cost", 0),
            estimated_cost=case.get("estimated_cost", 0.01)
        )

        print(f"\nPre-gates: {pre_report.summary()}")
        for check in pre_report.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[check.result.value]
            print(f"  {icon} {check.gate_name}: {check.message}")

        if not pre_report.passed:
            print("  → Execution blocked by pre-gate")
            continue

        # Simulate execution (already have result)
        print(f"\nExecuted. Result: {case['result'][:50]}...")

        # Run post-gates
        post_report = runner.run_post_gates(
            task=case["task"],
            result=case["result"],
            confidence=case.get("confidence", 0.8)
        )

        print(f"\nPost-gates: {post_report.summary()}")
        for check in post_report.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[check.result.value]
            print(f"  {icon} {check.gate_name}: {check.message}")

        if not post_report.passed:
            print("  → Result rejected by post-gate")
        else:
            print("  → Result accepted!")


if __name__ == "__main__":
    demo_gates()
```

---

## Understanding Gates

### Gate Hierarchy

```
                    ┌─────────────┐
                    │    Task     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  PRE-GATES  │
                    │             │
                    │ • Defined?  │
                    │ • Budget?   │
                    │ • Resources?│
                    └──────┬──────┘
                           │ Pass?
                    ┌──────▼──────┐
                    │   EXECUTE   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ POST-GATES  │
                    │             │
                    │ • Length?   │
                    │ • Format?   │
                    │ • Drift?    │
                    │ • Confidence│
                    └──────┬──────┘
                           │ Pass?
                    ┌──────▼──────┐
                    │   COMMIT    │
                    └─────────────┘
```

### Drift Detection Strategies

| Strategy | Speed | Cost | Accuracy |
|----------|-------|------|----------|
| Keyword overlap | Fast | Free | Low |
| Embedding similarity | Medium | Low | Medium |
| AI evaluation | Slow | Higher | High |

Choose based on your needs:
- **Keyword**: Quick sanity check
- **AI**: Important/complex tasks

---

## Exercises

### Exercise 1: Custom Validator Gate

Create a gate that runs custom validation functions:

```python
class CustomValidatorGate(PostGate):
    def __init__(self, validators: List[Callable[[str], bool]]):
        pass

    def check(self, result: str, **kwargs) -> GateCheckResult:
        pass
```

### Exercise 2: Progressive Gating

Implement gates that become stricter over retries:

```python
class ProgressiveConfidenceGate(PostGate):
    def __init__(self, base_threshold: float = 0.7):
        pass

    def check(self, confidence: float, attempt: int, **kwargs) -> GateCheckResult:
        # Higher threshold on first try, lower on retries
        pass
```

### Exercise 3: Gate Analytics

Track gate pass/fail rates over time:

```python
class GateAnalytics:
    def record(self, gate_name: str, result: GateResult):
        pass

    def get_pass_rate(self, gate_name: str) -> float:
        pass

    def report(self) -> dict:
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Pre-gates block invalid tasks
- [ ] Post-gates catch bad outputs
- [ ] Drift detection identifies off-topic outputs
- [ ] Gate reports summarize all checks
- [ ] You understand when to use each gate type

---

## Key Takeaway

> Gates catch problems before they compound.

Gating provides:
- **Early rejection** of bad inputs (pre-gates)
- **Quality assurance** on outputs (post-gates)
- **Drift prevention** to stay on track
- **Clear failure reasons** for debugging

---

## Get the Code

Related concepts: [8me/src/tier3.5-orchestration-concepts/02-patterns.md](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)

---

<div class="lab-navigation">
  <a href="./09-multi-agent" class="prev">← Previous: Lab 09 - Multi-Agent Patterns</a>
  <a href="./11-self-play" class="next">Next: Lab 11 - Self-Play Oscillation →</a>
</div>
