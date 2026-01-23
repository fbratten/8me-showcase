---
layout: page
title: Verification Strategies
permalink: /concepts/verification/
parent: Concepts
---

# Verification Strategies

How to confirm AI outputs are correct.

---

## The Verification Problem

AI models can:
- Claim success when they failed
- Produce plausible-looking but wrong output
- Miss edge cases
- Hallucinate confidently

**Never trust AI self-assessment alone.**

---

## Verification Types

### 1. Structural Verification
Check the output format matches expectations:

```python
def verify_json(output):
    try:
        data = json.loads(output)
        return "required_field" in data
    except json.JSONDecodeError:
        return False

def verify_haiku(output):
    lines = output.strip().split('\n')
    return len(lines) == 3
```

### 2. Content Verification
Check the output content is valid:

```python
def verify_code(output):
    try:
        compile(output, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def verify_url(output):
    import requests
    response = requests.head(output, timeout=5)
    return response.status_code == 200
```

### 3. External Verification
Use external tools or services:

```python
def verify_with_tests(code):
    # Run actual tests against the generated code
    result = subprocess.run(['pytest', 'tests/'], capture_output=True)
    return result.returncode == 0

def verify_with_linter(code):
    result = subprocess.run(['ruff', 'check', '-'], input=code)
    return result.returncode == 0
```

### 4. AI-Assisted Verification
Use a second AI call to verify (with caution):

```python
def verify_with_ai(task, result):
    prompt = f"""
    Task: {task}
    Result: {result}

    Does this result correctly complete the task?
    Answer only YES or NO.
    """
    response = client.messages.create(...)
    return "YES" in response.content[0].text.upper()
```

**Warning:** AI verification is still probabilistic. Use as one signal among many.

---

## Confidence Scoring

Instead of binary pass/fail, use confidence scores:

```python
def calculate_confidence(result, checks):
    passed = sum(1 for check in checks if check(result))
    return passed / len(checks)

checks = [
    lambda r: len(r) > 100,           # Minimum length
    lambda r: "error" not in r.lower(), # No error mentions
    lambda r: verify_json(r),          # Valid JSON
    lambda r: has_required_fields(r),  # Required fields
]

confidence = calculate_confidence(result, checks)
if confidence >= 0.75:
    accept(result)
else:
    retry()
```

---

## Verification Patterns

### Threshold-Based
```python
CONFIDENCE_THRESHOLD = 0.8

if result.confidence >= CONFIDENCE_THRESHOLD:
    accept(result)
else:
    retry()
```

### Multi-Stage
```python
# Stage 1: Quick structural check
if not verify_structure(result):
    retry_immediately()

# Stage 2: Content validation
if not verify_content(result):
    retry_with_feedback()

# Stage 3: External verification
if not verify_externally(result):
    flag_for_review()
```

### Consensus
```python
# Run the same task multiple times
results = [run_task() for _ in range(3)]

# Accept if majority agree
if results[0] == results[1] or results[0] == results[2]:
    accept(results[0])
elif results[1] == results[2]:
    accept(results[1])
else:
    fail("No consensus")
```

---

## Tool-Based Verification

Use Claude's tool calling for structured verification:

```python
tools = [{
    "name": "submit_result",
    "description": "Submit verified result",
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "verification_notes": {"type": "string"}
        },
        "required": ["result", "confidence"]
    }
}]

# AI must use the tool, giving us structured output
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    tool_choice={"type": "tool", "name": "submit_result"},
    messages=[...]
)
```

---

## When to Verify

| Scenario | Verification Level |
|----------|-------------------|
| Low-stakes, reversible | Light (structural) |
| User-facing output | Medium (content + structural) |
| Code generation | Heavy (tests + linting) |
| Financial/security | Maximum (external + human review) |

---

## Next Steps

- Learn about [Circuit Breakers](./circuit-breakers)
- Understand [State Management](./state-management)
- Practice: [Lab 03: Simple Verification](../labs/03-verification)

---

<div style="text-align: center;">
  <a href="./">← Back to Concepts</a>
</div>
