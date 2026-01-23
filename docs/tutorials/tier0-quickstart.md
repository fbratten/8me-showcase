---
layout: page
title: Tier 0 Quickstart
permalink: /tutorials/tier0-quickstart/
parent: Tutorials
---

# Tier 0: Hello World Quickstart

Run the simplest autonomous loop.

---

## Prerequisites

- Python 3.8+
- Anthropic API key

---

## Setup

```bash
# Clone the repo
git clone https://github.com/fbratten/8me.git
cd 8me

# Install dependency
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Run It

```bash
cd src/tier0-hello-world
python ralph_hello_world.py
```

---

## What Happens

1. Script reads `tasks.txt`
2. Finds first task not marked `[DONE]`
3. Sends to Claude
4. Prints response
5. Marks task `[DONE]`
6. Repeats until all done

---

## Sample Output

```
=== Starting Ralph Hello World Loop ===
Processing task: Write a haiku about programming

Claude's response:
Code flows like water
Through silicon valleys deep
Bugs bloom, then they die

Marked as done. Moving to next task...
...
=== All tasks completed! ===
```

---

## Try It Yourself

Edit `tasks.txt`:

```
Write a limerick about debugging
Explain why the sky is blue
Give me a fun fact about loops
```

---

## Next Steps

- [Understand the Code](../tier0-explained/)
- [Lab 01: Your First Loop](../../labs/01-first-loop)

---

<div style="text-align: center;">
  <a href="../">← Back to Tutorials</a>
</div>
