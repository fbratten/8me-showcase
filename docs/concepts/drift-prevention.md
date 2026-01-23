---
layout: page
title: Drift Prevention
permalink: /concepts/drift-prevention/
parent: Concepts
---

# Drift Prevention

Keeping agents aligned with original intent over multiple iterations.

---

## The Drift Problem

Over many iterations, agents can drift from the original goal:

```
┌─────────────────────────────────────────────────────────┐
│                 Drift Prevention Pattern                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    Original Intent ──────────────────────────────►     │
│          │                                              │
│          │    Iteration 1    Iteration 2    Iteration 3│
│          │        │              │              │       │
│          │        ▼              ▼              ▼       │
│          │    ┌───────┐     ┌───────┐     ┌───────┐    │
│          └───►│ CHECK │────►│ CHECK │────►│ CHECK │    │
│               │ DRIFT │     │ DRIFT │     │ DRIFT │    │
│               └───┬───┘     └───┬───┘     └───┬───┘    │
│                   │             │             │         │
│              OK ──┘        DRIFT ─► REALIGN   │         │
│                                               │         │
│                                          COMPLETE       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation

```python
def execute_with_drift_prevention(task, max_drift=0.3):
    """
    Execute task while monitoring for drift from original intent.
    Realign if drift detected.
    """
    # Capture original intent
    original_intent = extract_intent(task)
    intent_embedding = embed(original_intent)

    result = None
    iterations = 0

    while not is_complete(result):
        iterations += 1

        # Execute iteration
        result = execute_iteration(task, result)

        # Measure drift
        current_embedding = embed(summarize(result))
        drift_score = 1 - cosine_similarity(
            intent_embedding,
            current_embedding
        )

        if drift_score > max_drift:
            # Drift detected! Realign
            log_warning(f"Drift detected: {drift_score:.2f}")
            result = realign(task, result, original_intent)

        if iterations > 10:
            break

    return result
```

---

## Intent Extraction

Capture the core intent before starting:

```python
def extract_intent(task):
    """
    Distill the core intent from a task.
    This becomes the anchor for drift detection.
    """
    return call_ai(
        prompt=f"""
        Extract the core intent from this task.
        Focus on WHAT needs to be achieved, not HOW.

        Task: {task.description}

        Core intent (1-2 sentences):
        """
    )
```

---

## Drift Measurement

Use embeddings to measure semantic distance:

```python
def measure_drift(current_state, original_intent):
    """
    Calculate how far current state has drifted from intent.
    Returns 0.0 (no drift) to 1.0 (completely off track).
    """
    intent_embedding = embed(original_intent)
    current_embedding = embed(summarize(current_state))

    similarity = cosine_similarity(intent_embedding, current_embedding)
    drift = 1 - similarity

    return drift
```

---

## Realignment

Bring execution back on track:

```python
def realign(task, current_result, original_intent):
    """
    Bring execution back in line with original intent.
    """
    return call_ai(
        prompt=f"""
        The execution has drifted from the original intent.

        ORIGINAL INTENT:
        {original_intent}

        CURRENT STATE:
        {current_result}

        TASK:
        {task.description}

        Please realign the work with the original intent.
        Keep what's valuable, discard what's drifted.
        """
    )
```

---

## Drift Thresholds

| Drift Score | Action |
|-------------|--------|
| 0.0 - 0.2 | Continue normally |
| 0.2 - 0.4 | Log warning, continue |
| 0.4 - 0.6 | Realign and continue |
| 0.6+ | Stop and request review |

---

## Next Steps

- Learn about [Multi-Agent Coordination](./multi-agent)
- See concepts in action: [Lab 10: Gating and Drift Prevention](../labs/10-gating)

---

<div style="text-align: center;">
  <a href="./">← Back to Concepts</a>
</div>
