---
layout: page
title: Tier 2 Install
permalink: /tutorials/tier2-install/
parent: Tutorials
---

# Tier 2: Installing the Skill

Install the Ralph skill for Claude Code.

---

## What is a Skill?

Claude Code skills are markdown instructions that teach Claude new workflows. The Ralph skill teaches the loop-until-done methodology.

---

## Installation

### Option 1: Copy to Skills Directory

```bash
# Find your Claude Code skills directory
# Usually: ~/.claude/skills/

cp -r src/tier2-ralph-skill ~/.claude/skills/ralph
```

### Option 2: Use /install-skill

In Claude Code:
```
/install-skill https://github.com/fbratten/8me/tree/main/src/tier2-ralph-skill
```

---

## Verify Installation

```bash
ls ~/.claude/skills/ralph/
# Should show: SKILL.md, README.md, references/
```

---

## Skill Structure

```
tier2-ralph-skill/
├── SKILL.md              # Main instructions
├── README.md             # Documentation
├── sample.ralph-config   # Example config
└── references/
    └── ralph-methodology.md  # Background
```

---

## Next Steps

- [Using Ralph Commands](../tier2-commands/)
- [Lab 14: Skill Development](../../labs/14-skill)

---

<div style="text-align: center;">
  <a href="./">← Back to Tutorials</a>
</div>
