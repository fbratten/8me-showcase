---
layout: lab
title: "Lab 14: Claude Code Skill"
lab_number: 14
difficulty: integration
time: 1.5 hours
prerequisites: Lab 13 completed
---

# Lab 14: Claude Code Skill

Create a custom Claude Code skill for natural language orchestration.

## Objectives

By the end of this lab, you will:
- Understand Claude Code skills
- Create a custom skill with commands
- Integrate with your MCP server
- Use natural language to orchestrate tasks

## Prerequisites

- Lab 13 completed (MCP server)
- Claude Code CLI installed

## What is a Claude Code Skill?

A **skill** teaches Claude Code new behaviors through markdown instructions:

```
Without skill:
  User: "Process my tasks"
  Claude: "I'm not sure what you mean by process tasks..."

With skill:
  User: "Process my tasks"
  Claude: [Reads SKILL.md]
  Claude: [Follows instructions to run orchestrator]
  Claude: "I've processed 5 tasks. 4 completed, 1 failed..."
```

Skills provide:
- **Custom commands**: `/ralph`, `/tasks`, `/loop`
- **Behavioral instructions**: How to handle specific requests
- **Reference materials**: Background knowledge

---

## Skill Structure

```
my-skill/
├── SKILL.md           # Main instructions (required)
├── README.md          # Installation guide
├── references/        # Supporting materials
│   ├── methodology.md
│   └── examples.md
└── sample.config      # Sample configuration
```

---

## Step 1: Create SKILL.md

Create `ralph-skill/SKILL.md`:

```markdown
---
name: ralph
description: Autonomous task loop orchestration
commands:
  - /ralph
  - /tasks
  - /loop
---

# Ralph Workflow Skill

You are enhanced with the Ralph Workflow methodology for autonomous task processing.

## Core Principle

> "Loop persistently until the job is done."

When the user asks you to process tasks, run a loop, or use Ralph methodology:
1. Check for pending tasks
2. Process each task with verification
3. Retry on failure (up to max attempts)
4. Report results

## Commands

### /ralph

Start the Ralph workflow loop.

**When the user runs `/ralph`:**

1. Check if the Ralph MCP server is available
   - Look for `mcp__ralph-workflow__*` tools

2. If MCP server is available:
   - Read `task://queue` to see pending tasks
   - For each pending task:
     a. Use `get_next_task` to get the task
     b. Process the task (complete the described work)
     c. Use `complete_task` with your result
     d. Or use `fail_task` if unable to complete
   - Continue until no pending tasks remain

3. If MCP server is NOT available:
   - Look for a `tasks.json` file in the current directory
   - Process tasks manually using the file
   - Update task status in the file

4. Report summary when done:
   - Tasks completed
   - Tasks failed
   - Any issues encountered

### /tasks

Manage the task queue.

**When the user runs `/tasks`:**

Show the current task queue status:
1. Read `task://queue` or `tasks.json`
2. Display tasks grouped by status:
   - Pending (ready to process)
   - In Progress (currently being worked on)
   - Completed (finished successfully)
   - Failed (could not complete)
3. Show statistics (completion rate, etc.)

**Subcommands:**

- `/tasks add <description>` - Add a new task
- `/tasks clear` - Remove completed tasks
- `/tasks reset` - Reset all tasks to pending
- `/tasks show <id>` - Show task details

### /loop

Run a verification loop on the current task.

**When the user runs `/loop`:**

1. Get the current task context
2. Implement a loop:
   ```
   while not verified:
       result = attempt_task()
       verified = verify_result(result)
       if not verified:
           incorporate_feedback()
   ```
3. Use circuit breaker (max 5 iterations)
4. Report when complete or max iterations reached

## Task Processing Guidelines

### Verification

For each task result:
1. Check if the result addresses the task description
2. Verify any specific criteria (format, length, content)
3. If verification fails, retry with feedback

### Circuit Breaker

Stop processing if:
- More than 5 consecutive failures
- Same error repeating 3+ times
- User requests stop

### Error Handling

When a task fails:
1. Record the error reason
2. Check if retryable (< max attempts)
3. If retryable, queue for retry with feedback
4. If not retryable, mark as failed and continue

## Integration with MCP Server

When the Ralph MCP server is available, use these tools:

**Resources (read-only):**
- `task://queue` - All tasks
- `task://current` - Current task
- `task://stats` - Statistics
- `task://pending` - Pending tasks only
- `task://completed` - Completed tasks

**Tools (actions):**
- `add_task(description, priority?, max_attempts?)` - Add task
- `start_task(task_id)` - Mark as in progress
- `complete_task(task_id, result)` - Mark complete
- `fail_task(task_id, reason)` - Mark failed
- `retry_task(task_id, feedback?)` - Queue for retry
- `get_next_task()` - Get next pending task
- `clear_completed()` - Remove completed tasks
- `reset_all_tasks()` - Reset all to pending

## Examples

**User:** "Process my tasks"
**You:** Check for Ralph MCP or tasks.json, then process all pending tasks.

**User:** "/ralph"
**You:** Run the full Ralph loop with verification.

**User:** "/tasks add Write unit tests for the login module"
**You:** Add a new task with the given description.

**User:** "/loop"
**You:** Run verification loop on current work.

## Fallback Behavior

If no MCP server or tasks.json exists:
1. Ask the user what tasks they want to accomplish
2. Create a mental task list
3. Process each task systematically
4. Report progress and results

Remember: The Ralph methodology is about **persistence**. Keep trying until the job is done, but know when to stop (circuit breaker).
```

---

## Step 2: Create Supporting Files

Create `ralph-skill/references/ralph-methodology.md`:

```markdown
# Ralph Wiggum Methodology

## Origin

Named after Ralph Wiggum from The Simpsons, who persistently tries things
until they work (or hilariously fail). The methodology embraces:

- **Persistence**: Keep trying
- **Simplicity**: Don't overthink
- **Acceptance**: Know when to stop

## Core Loop

```
while tasks_remain and not circuit_breaker_tripped:
    task = get_next_task()
    result = attempt(task)

    if verify(result):
        complete(task, result)
    else:
        if can_retry(task):
            retry(task, feedback)
        else:
            fail(task)
```

## Key Principles

### 1. External State

Always persist state externally:
- Task status in JSON/database
- Progress survives crashes
- Enables debugging

### 2. Verification

Never trust first results:
- Check against criteria
- Use confidence scores
- Allow for "good enough"

### 3. Circuit Breakers

Prevent infinite loops:
- Max iterations
- Consecutive failure limits
- Cost limits
- Output repetition detection

### 4. Feedback Loops

Learn from failures:
- Pass feedback to retries
- Adjust approach based on errors
- Remember patterns

## When to Use Ralph

✓ Repetitive task processing
✓ Quality through iteration
✓ Long-running workflows
✓ Automated pipelines

✗ Simple one-shot tasks
✗ Real-time requirements
✗ Creative exploration
```

Create `ralph-skill/README.md`:

```markdown
# Ralph Workflow Skill

A Claude Code skill for autonomous task loop orchestration.

## Installation

### Option 1: Add to Project

Copy the `ralph-skill` folder to your project and add to `.claude/settings.json`:

```json
{
  "skills": ["./ralph-skill"]
}
```

### Option 2: Global Installation

Copy to your Claude Code skills directory:

```bash
# Linux/Mac
cp -r ralph-skill ~/.claude/skills/

# Windows
copy ralph-skill %USERPROFILE%\.claude\skills\
```

## Usage

### Commands

- `/ralph` - Start the Ralph workflow loop
- `/tasks` - View and manage task queue
- `/loop` - Run verification loop on current task

### With MCP Server

For best results, also install the Ralph MCP server:

```bash
pip install ralph-mcp-server
```

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "ralph-workflow": {
      "command": "ralph-mcp"
    }
  }
}
```

### Without MCP Server

The skill works standalone using a `tasks.json` file in your project.

## Examples

```
User: /ralph
Claude: Checking for pending tasks...
        Found 3 tasks. Processing...

        [Task 1] Write a haiku about loops
        ✓ Completed (attempt 1)

        [Task 2] Explain recursion
        ✓ Completed (attempt 1)

        [Task 3] Solve impossible math problem
        ✗ Failed after 3 attempts

        Summary: 2/3 tasks completed
```

## License

MIT
```

---

## Step 3: Create Sample Configuration

Create `ralph-skill/sample.ralph-config`:

```json
{
  "maxAttempts": 3,
  "verificationThreshold": 0.8,
  "circuitBreaker": {
    "maxIterations": 50,
    "maxConsecutiveFailures": 5,
    "maxCostDollars": 1.0
  },
  "taskFile": "tasks.json",
  "useMcp": true
}
```

---

## Step 4: Test the Skill

### Manual Testing

1. Copy the skill to your project
2. Start Claude Code in the project directory
3. Run commands:

```
> /ralph
> /tasks
> /tasks add "Write a hello world function"
> /loop
```

### Verify Skill Loading

```
> /help

Should show:
- /ralph - Start the Ralph workflow loop
- /tasks - View and manage task queue
- /loop - Run verification loop
```

---

## How Skills Work

### Loading Process

```
1. Claude Code starts
         │
         ▼
2. Scans for SKILL.md files
   - Project: ./.claude/skills/
   - Global: ~/.claude/skills/
         │
         ▼
3. Parses frontmatter (name, commands)
         │
         ▼
4. Loads into system prompt
         │
         ▼
5. Commands become available
```

### Skill vs MCP

| Aspect | Skill | MCP Server |
|--------|-------|------------|
| Purpose | Teach behavior | Provide capabilities |
| Format | Markdown | Code + Protocol |
| State | Stateless | Can be stateful |
| Integration | System prompt | Tool calls |
| Best for | Workflows | Data/Actions |

### Combining Both

Skills and MCP work together:
- **Skill**: Defines *how* to use tools
- **MCP**: Provides *what* tools are available

```
User: "/ralph"
         │
         ▼
Skill: "When /ralph is called, use these MCP tools..."
         │
         ▼
MCP: Provides add_task, complete_task, etc.
         │
         ▼
Result: Coordinated task processing
```

---

## Exercises

### Exercise 1: Add Progress Reporting

Enhance the skill to show progress bars:

```markdown
### /ralph --verbose

Show detailed progress:
- Current task description
- Attempt number
- Verification status
- Time elapsed
```

### Exercise 2: Add Task Templates

Add support for task templates:

```markdown
### /tasks template <name>

Load a predefined task template:
- `code-review` - Review code for issues
- `documentation` - Generate docs
- `testing` - Write tests
```

### Exercise 3: Add History Command

Track and display task history:

```markdown
### /tasks history

Show recent task processing history:
- Last 10 tasks processed
- Success/failure status
- Processing time
```

---

## Checkpoint

Before moving on, verify:
- [ ] Skill loads without errors
- [ ] /ralph command starts processing
- [ ] /tasks shows queue status
- [ ] Skill integrates with MCP server
- [ ] Fallback works without MCP

---

## Key Takeaway

> Skills make orchestration accessible via natural language.

With a skill:
- **Natural commands**: `/ralph` instead of API calls
- **Guided behavior**: Claude knows *how* to orchestrate
- **Flexible integration**: Works with or without MCP
- **User-friendly**: Anyone can use complex workflows

---

## Get the Code

Full implementation: [8me/src/tier2-ralph-skill/](https://github.com/fbratten/8me/tree/main/src/tier2-ralph-skill)

---

<div class="lab-navigation">
  <a href="./13-mcp-server.md" class="prev">← Previous: Lab 13 - MCP Server Deployment</a>
  <a href="./15-end-to-end.md" class="next">Next: Lab 15 - End-to-End Workflow →</a>
</div>
