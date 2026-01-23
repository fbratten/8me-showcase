---
layout: default
title: Tier 2 Commands
nav_order: 9
parent: Tutorials
permalink: /tutorials/tier2-commands/
---

# Tier 2: Using Ralph Commands

Use the Ralph skill in Claude Code.

---

## Available Commands

### /ralph

Start the loop on current tasks:

```
/ralph
```

Claude will:
1. Read your task file
2. Execute each pending task
3. Verify completion
4. Mark done and continue

---

### /ralph add

Add a new task:

```
/ralph add "Refactor the login function"
```

---

### /ralph status

Check queue status:

```
/ralph status
```

Output:
```
Tasks: 3 pending, 1 in_progress, 5 completed
```

---

## How It Works

The skill teaches Claude to:

1. **Persist** - Keep working until verified complete
2. **Verify** - Check results before marking done
3. **Retry** - Try again on low confidence
4. **Stop** - Respect circuit breaker limits

---

## Example Session

```
You: /ralph add "Write unit tests for auth module"
Claude: Added task. Starting loop...

Working on: Write unit tests for auth module
Created: tests/test_auth.py
Running tests... 5/5 passed
Confidence: 0.95 - Complete!

All tasks done!
```

---

## Next Steps

- [Tier 3: MCP Server](../tier3-install/)
- [Lab 14: Build Your Own Skill](../../labs/14-skill)

---

<div style="text-align: center;">
  <a href="../">← Back to Tutorials</a>
</div>
