---
layout: page
title: Gating Mechanisms
permalink: /concepts/gating/
parent: Concepts
---

# Gating Mechanisms

Pre and post-condition enforcement with rollback.

---

## The Gating Pattern

Nothing proceeds without passing gates:

```
┌─────────────────────────────────────────────────────────┐
│                    Gating Pattern                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│    │ PRE-GATE │───►│ EXECUTE  │───►│POST-GATE │        │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘        │
│         │               │               │               │
│    FAIL │          FAIL │          FAIL │               │
│         ▼               ▼               ▼               │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│    │  REJECT  │    │ ROLLBACK │    │  REVERT  │        │
│    └──────────┘    └──────────┘    └──────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation

```python
def gated_execution(task, executor):
    """
    Execute task only if gates pass.
    Rollback on post-gate failure.
    """
    # === PRE-GATE ===
    pre_gate_result = check_pre_gates(task)
    if not pre_gate_result.passed:
        return {
            "status": "rejected",
            "reason": pre_gate_result.reason,
            "gate": "pre"
        }

    # === EXECUTION ===
    checkpoint = create_checkpoint()

    try:
        result = executor.execute(task)
    except Exception as e:
        restore_checkpoint(checkpoint)
        return {"status": "execution_failed", "reason": str(e)}

    # === POST-GATE ===
    post_gate_result = check_post_gates(task, result)
    if not post_gate_result.passed:
        restore_checkpoint(checkpoint)
        return {
            "status": "reverted",
            "reason": post_gate_result.reason,
            "gate": "post"
        }

    commit_changes(checkpoint)
    return {"status": "committed", "result": result}
```

---

## Pre-Gate Checks

Verify conditions before execution:

```python
def check_pre_gates(task):
    """
    Pre-execution checks:
    - Task is well-formed
    - Dependencies are satisfied
    - Resources are available
    - Permissions are granted
    """
    gates = [
        ("well_formed", lambda t: t.description and len(t.description) > 10),
        ("dependencies", lambda t: all_deps_ready(t)),
        ("resources", lambda t: resources_available(t)),
        ("permissions", lambda t: has_permissions(t)),
    ]

    for gate_name, gate_fn in gates:
        if not gate_fn(task):
            return GateResult(passed=False, reason=f"Failed: {gate_name}")

    return GateResult(passed=True)
```

---

## Post-Gate Checks

Verify results after execution:

```python
def check_post_gates(task, result):
    """
    Post-execution checks:
    - Tests pass
    - Types check
    - No security issues
    - Output is valid
    """
    gates = [
        ("tests", lambda r: run_tests(r)),
        ("types", lambda r: type_check(r)),
        ("security", lambda r: security_scan(r)),
        ("valid", lambda r: validate_output(r)),
    ]

    for gate_name, gate_fn in gates:
        if not gate_fn(result):
            return GateResult(passed=False, reason=f"Failed: {gate_name}")

    return GateResult(passed=True)
```

---

## Tiered Enforcement

SPINE uses task tiers to determine gating requirements:

| Tier | Description | Requirements |
|------|-------------|--------------|
| **1** | Simple, single-file | Direct execution OK |
| **2** | Multi-file, features | SHOULD use subagents |
| **3** | Architecture, research | MUST use subagents + MCP |

---

## Rollback Strategies

When post-gates fail:

```python
def rollback_strategy(checkpoint, failure_type):
    if failure_type == "tests":
        # Partial rollback - keep structure, revert logic
        restore_logic_only(checkpoint)
    elif failure_type == "security":
        # Full rollback - security issues are serious
        restore_checkpoint(checkpoint)
        alert_security_team()
    else:
        # Standard rollback
        restore_checkpoint(checkpoint)
```

---

## Next Steps

- Learn about [Drift Prevention](./drift-prevention)
- See implementation: [Lab 10: Gating](../labs/10-gating)

---

<div style="text-align: center;">
  <a href="./">← Back to Concepts</a>
</div>
