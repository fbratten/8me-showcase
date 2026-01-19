---
layout: lab
title: "Lab 09: Multi-Agent Patterns"
lab_number: 9
difficulty: advanced
time: 2 hours
prerequisites: Lab 08 completed
---

# Lab 09: Multi-Agent Patterns

Coordinate multiple specialized agents for complex workflows.

## Objectives

By the end of this lab, you will:
- Understand agent specialization and roles
- Implement agent communication patterns
- Build pipelines and workflows with multiple agents
- Know when multi-agent is better than single-agent

## Prerequisites

- Lab 08 completed (orchestrator)
- Understanding of the orchestrator pattern

## Why Multiple Agents?

A single generalist agent must do everything:

```python
# Single agent: jack of all trades
result = agent.do_everything(
    "Plan the feature, implement it, review the code, write tests"
)
```

Problems:
- **Context overload**: Too much in one prompt
- **Conflicting goals**: "Be creative" vs "Be critical"
- **No checks**: Same agent reviews its own work

Multiple specialized agents divide and conquer:

```python
# Multiple agents: specialists
plan = planner.create_plan(task)
code = implementer.implement(plan)
review = reviewer.critique(code)
tests = tester.write_tests(code)
```

---

## Agent Patterns

### Pattern 1: Pipeline

Sequential processing where each agent's output feeds the next:

```
Task → [Planner] → [Implementer] → [Reviewer] → Result
```

### Pattern 2: Debate

Two agents with opposing goals reach consensus:

```
        ┌──────────┐
        │ Proposer │───────┐
        └──────────┘       │
                           ▼
                    ┌─────────────┐
                    │  Moderator  │
                    └─────────────┘
                           ▲
        ┌──────────┐       │
        │  Critic  │───────┘
        └──────────┘
```

### Pattern 3: Ensemble

Multiple agents work independently, results are merged:

```
          ┌──────────┐
     ┌────│ Agent A  │────┐
     │    └──────────┘    │
     │                    │
Task ├────┌──────────┐────┼────► Merge ────► Result
     │    │ Agent B  │    │
     │    └──────────┘    │
     │                    │
     └────┌──────────┐────┘
          │ Agent C  │
          └──────────┘
```

### Pattern 4: Hierarchy

Supervisor delegates to worker agents:

```
              ┌────────────┐
              │ Supervisor │
              └──────┬─────┘
         ┌──────────┬┴───────────┐
         ▼          ▼            ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Worker A│ │Worker B│ │Worker C│
    └────────┘ └────────┘ └────────┘
```

---

## Step 1: Create the Agent Base Class

Create `agent.py`:

```python
"""
Agent Base Class - Lab 09

Foundation for specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import anthropic


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class AgentMessage:
    """A message in agent communication."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all agents.

    Provides common functionality for AI-powered agents.
    """

    def __init__(
        self,
        name: str,
        role: str,
        config: Optional[AgentConfig] = None
    ):
        self.name = name
        self.role = role
        self.config = config or AgentConfig()
        self.client = anthropic.Anthropic()
        self.history: List[AgentMessage] = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define the agent's system prompt."""
        pass

    def execute(self, task: str, context: Optional[Dict] = None) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: The task description
            context: Optional context from other agents

        Returns:
            AgentResponse with result
        """
        # Build messages
        messages = self._build_messages(task, context)

        # Call Claude
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.system_prompt,
            messages=messages
        )

        content = response.content[0].text

        # Record in history
        self.history.append(AgentMessage(role="user", content=task))
        self.history.append(AgentMessage(role="assistant", content=content))

        return AgentResponse(
            content=content,
            confidence=0.85,
            metadata={
                "agent": self.name,
                "role": self.role,
                "model": self.config.model
            }
        )

    def _build_messages(
        self,
        task: str,
        context: Optional[Dict] = None
    ) -> List[Dict[str, str]]:
        """Build message list for the API call."""
        prompt = task

        if context:
            prompt = self._format_context(context) + "\n\n" + task

        return [{"role": "user", "content": prompt}]

    def _format_context(self, context: Dict) -> str:
        """Format context from other agents."""
        parts = ["## Context from other agents:\n"]
        for key, value in context.items():
            parts.append(f"### {key}:\n{value}\n")
        return "\n".join(parts)

    def reset(self):
        """Clear agent history."""
        self.history.clear()
```

---

## Step 2: Create Specialized Agents

Create `specialized_agents.py`:

```python
"""
Specialized Agents - Lab 09

Role-specific agent implementations.
"""

from typing import Optional, Dict
from agent import Agent, AgentConfig, AgentResponse


class PlannerAgent(Agent):
    """Agent specialized for planning and decomposition."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="Planner",
            role="planner",
            config=config
        )

    @property
    def system_prompt(self) -> str:
        return """You are a planning specialist. Your job is to:
1. Analyze tasks and break them into clear steps
2. Identify dependencies between steps
3. Estimate complexity and potential issues
4. Create actionable plans

Always output structured plans with numbered steps.
Be thorough but concise. Focus on what needs to be done, not how."""


class ImplementerAgent(Agent):
    """Agent specialized for implementation."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="Implementer",
            role="implementer",
            config=config
        )

    @property
    def system_prompt(self) -> str:
        return """You are an implementation specialist. Your job is to:
1. Follow plans precisely
2. Write clean, working code or content
3. Handle edge cases
4. Document your work

Focus on execution, not planning. If the plan is unclear, ask for clarification.
Produce complete, working solutions."""


class ReviewerAgent(Agent):
    """Agent specialized for review and critique."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="Reviewer",
            role="reviewer",
            config=config
        )

    @property
    def system_prompt(self) -> str:
        return """You are a review specialist. Your job is to:
1. Critically evaluate work from other agents
2. Find bugs, errors, and issues
3. Suggest improvements
4. Verify requirements are met

Be constructive but thorough. Your role is quality assurance.
Output structured reviews with clear pass/fail assessments."""


class RefinerAgent(Agent):
    """Agent specialized for refinement based on feedback."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="Refiner",
            role="refiner",
            config=config
        )

    @property
    def system_prompt(self) -> str:
        return """You are a refinement specialist. Your job is to:
1. Take existing work and feedback
2. Address all issues raised
3. Improve quality while maintaining intent
4. Produce polished final output

Focus on improvement, not replacement. Preserve what works, fix what doesn't."""


class SummarizerAgent(Agent):
    """Agent specialized for summarization."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="Summarizer",
            role="summarizer",
            config=config
        )

    @property
    def system_prompt(self) -> str:
        return """You are a summarization specialist. Your job is to:
1. Distill complex information into key points
2. Maintain accuracy while reducing length
3. Highlight the most important elements
4. Create clear, readable summaries

Be concise but complete. Capture essence, not details."""
```

---

## Step 3: Create Pipeline Coordinator

Create `pipeline.py`:

```python
"""
Multi-Agent Pipeline - Lab 09

Coordinates multiple agents in sequence.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from agent import Agent, AgentResponse


@dataclass
class PipelineStep:
    """A single step in the pipeline."""
    agent: Agent
    input_key: str  # Key to get input from context
    output_key: str  # Key to store output in context
    prompt_template: str  # Template for the task prompt
    required: bool = True


@dataclass
class PipelineResult:
    """Result of running a pipeline."""
    success: bool
    final_output: str
    steps_completed: int
    context: Dict[str, Any]
    error: Optional[str] = None


class AgentPipeline:
    """
    Coordinates multiple agents in a pipeline.

    Usage:
        pipeline = AgentPipeline()
        pipeline.add_step(planner, "task", "plan", "Create a plan for: {task}")
        pipeline.add_step(implementer, "plan", "impl", "Implement this plan:\n{plan}")
        pipeline.add_step(reviewer, "impl", "review", "Review this:\n{impl}")

        result = pipeline.run("Build a login form")
    """

    def __init__(self):
        self.steps: List[PipelineStep] = []
        self.hooks: Dict[str, List[Callable]] = {
            "before_step": [],
            "after_step": [],
            "on_error": []
        }

    def add_step(
        self,
        agent: Agent,
        input_key: str,
        output_key: str,
        prompt_template: str,
        required: bool = True
    ) -> "AgentPipeline":
        """Add a step to the pipeline."""
        self.steps.append(PipelineStep(
            agent=agent,
            input_key=input_key,
            output_key=output_key,
            prompt_template=prompt_template,
            required=required
        ))
        return self

    def add_hook(self, event: str, callback: Callable) -> "AgentPipeline":
        """Add a lifecycle hook."""
        if event in self.hooks:
            self.hooks[event].append(callback)
        return self

    def run(self, initial_input: str) -> PipelineResult:
        """
        Run the pipeline.

        Args:
            initial_input: The initial task/input

        Returns:
            PipelineResult with final output and context
        """
        context = {"task": initial_input, "input": initial_input}
        steps_completed = 0

        for i, step in enumerate(self.steps):
            # Trigger before hooks
            self._trigger_hooks("before_step", step, context)

            try:
                # Build prompt from template
                prompt = step.prompt_template.format(**context)

                # Execute agent
                response = step.agent.execute(prompt, context)

                # Store output
                context[step.output_key] = response.content
                context["last_output"] = response.content
                steps_completed += 1

                # Trigger after hooks
                self._trigger_hooks("after_step", step, context, response)

            except Exception as e:
                self._trigger_hooks("on_error", step, context, error=e)

                if step.required:
                    return PipelineResult(
                        success=False,
                        final_output="",
                        steps_completed=steps_completed,
                        context=context,
                        error=f"Step {i+1} ({step.agent.name}) failed: {e}"
                    )

        return PipelineResult(
            success=True,
            final_output=context.get("last_output", ""),
            steps_completed=steps_completed,
            context=context
        )

    def _trigger_hooks(self, event: str, step: PipelineStep, context: Dict, **kwargs):
        """Trigger hooks for an event."""
        for hook in self.hooks.get(event, []):
            try:
                hook(step, context, **kwargs)
            except Exception as e:
                print(f"Hook error: {e}")


class DebatePipeline:
    """
    Two agents debate until consensus or max rounds.

    Usage:
        debate = DebatePipeline(proposer, critic, max_rounds=3)
        result = debate.run("What's the best approach for X?")
    """

    def __init__(
        self,
        proposer: Agent,
        critic: Agent,
        max_rounds: int = 3,
        consensus_threshold: float = 0.8
    ):
        self.proposer = proposer
        self.critic = critic
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold

    def run(self, topic: str) -> PipelineResult:
        """Run the debate."""
        context = {"topic": topic, "rounds": []}

        proposal = None
        critique = None

        for round_num in range(self.max_rounds):
            # Proposer makes or refines proposal
            if proposal is None:
                proposer_prompt = f"Propose a solution for: {topic}"
            else:
                proposer_prompt = f"""Refine your proposal based on this critique:

Original topic: {topic}
Your previous proposal: {proposal}
Critique: {critique}

Provide an improved proposal."""

            proposer_response = self.proposer.execute(proposer_prompt)
            proposal = proposer_response.content

            # Critic evaluates
            critic_prompt = f"""Evaluate this proposal:

Topic: {topic}
Proposal: {proposal}

Provide:
1. SCORE: 0.0 to 1.0 (how good is this proposal?)
2. ISSUES: What problems remain?
3. SUGGESTIONS: How to improve?

If score >= {self.consensus_threshold}, the proposal is accepted."""

            critic_response = self.critic.execute(critic_prompt)
            critique = critic_response.content

            # Record round
            context["rounds"].append({
                "round": round_num + 1,
                "proposal": proposal,
                "critique": critique
            })

            # Check for consensus
            score = self._extract_score(critique)
            if score >= self.consensus_threshold:
                return PipelineResult(
                    success=True,
                    final_output=proposal,
                    steps_completed=round_num + 1,
                    context=context
                )

        # Max rounds reached
        return PipelineResult(
            success=True,  # Still succeeded, just hit limit
            final_output=proposal,
            steps_completed=self.max_rounds,
            context=context,
            error="Max rounds reached without consensus"
        )

    def _extract_score(self, critique: str) -> float:
        """Extract score from critique."""
        for line in critique.split("\n"):
            if "SCORE:" in line.upper():
                try:
                    score_str = line.split(":")[1].strip()
                    return float(score_str.split()[0])
                except:
                    pass
        return 0.5  # Default
```

---

## Step 4: Complete Example

Create `multi_agent_demo.py`:

```python
"""
Multi-Agent Demo - Lab 09

Demonstrates different multi-agent patterns.
"""

from specialized_agents import (
    PlannerAgent, ImplementerAgent, ReviewerAgent, RefinerAgent
)
from pipeline import AgentPipeline, DebatePipeline


def demo_pipeline():
    """Demonstrate a sequential pipeline."""
    print("=" * 60)
    print("PIPELINE DEMO: Plan → Implement → Review → Refine")
    print("=" * 60)

    # Create agents
    planner = PlannerAgent()
    implementer = ImplementerAgent()
    reviewer = ReviewerAgent()
    refiner = RefinerAgent()

    # Build pipeline
    pipeline = AgentPipeline()
    pipeline.add_step(
        planner, "task", "plan",
        "Create a detailed plan for: {task}"
    )
    pipeline.add_step(
        implementer, "plan", "implementation",
        "Implement this plan:\n\n{plan}"
    )
    pipeline.add_step(
        reviewer, "implementation", "review",
        "Review this implementation:\n\n{implementation}"
    )
    pipeline.add_step(
        refiner, "implementation", "final",
        "Refine based on review:\n\nOriginal:\n{implementation}\n\nReview:\n{review}"
    )

    # Add logging hooks
    def log_step(step, context, **kwargs):
        print(f"\n▶ {step.agent.name} working...")

    def log_complete(step, context, response=None, **kwargs):
        if response:
            print(f"  ✓ Output: {response.content[:100]}...")

    pipeline.add_hook("before_step", log_step)
    pipeline.add_hook("after_step", log_complete)

    # Run
    result = pipeline.run("Write a Python function that validates email addresses")

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"Steps completed: {result.steps_completed}")
    print(f"\nFinal output:\n{result.final_output}")


def demo_debate():
    """Demonstrate a debate pipeline."""
    print("\n" + "=" * 60)
    print("DEBATE DEMO: Proposer vs Critic")
    print("=" * 60)

    # Create debaters
    proposer = PlannerAgent()  # Reuse planner as proposer
    proposer.name = "Proposer"

    critic = ReviewerAgent()
    critic.name = "Critic"

    # Create debate
    debate = DebatePipeline(
        proposer=proposer,
        critic=critic,
        max_rounds=3,
        consensus_threshold=0.8
    )

    # Run debate
    result = debate.run("What's the best way to handle errors in a REST API?")

    print(f"\nDebate completed in {result.steps_completed} rounds")
    print(f"Consensus reached: {'Yes' if not result.error else 'No (max rounds)'}")

    print("\n--- Debate History ---")
    for round_data in result.context["rounds"]:
        print(f"\nRound {round_data['round']}:")
        print(f"Proposal: {round_data['proposal'][:200]}...")
        print(f"Critique: {round_data['critique'][:200]}...")

    print("\n--- Final Proposal ---")
    print(result.final_output)


def main():
    demo_pipeline()
    print("\n\n")
    demo_debate()


if __name__ == "__main__":
    main()
```

---

## When to Use Multi-Agent

| Scenario | Single Agent | Multi-Agent |
|----------|--------------|-------------|
| Simple task | ✓ Better | Overkill |
| Complex workflow | Struggles | ✓ Better |
| Self-review needed | Biased | ✓ Better |
| Specialized knowledge | Good enough | ✓ Better |
| Debugging needed | Hard to trace | ✓ Easier |
| Cost sensitive | ✓ Cheaper | More expensive |

### Rules of Thumb

1. **Start single** → Add agents only when needed
2. **Clear roles** → Each agent has one job
3. **Minimize handoffs** → Each handoff adds latency and error risk
4. **Test individually** → Verify each agent works before combining

---

## Exercises

### Exercise 1: Ensemble Pattern

Implement an ensemble that runs 3 agents in parallel and merges results:

```python
class EnsemblePipeline:
    def __init__(self, agents: List[Agent], merger: Agent):
        pass

    def run(self, task: str) -> PipelineResult:
        # Run all agents in parallel
        # Merge results with merger agent
        pass
```

### Exercise 2: Hierarchical Delegation

Implement a supervisor that delegates subtasks:

```python
class SupervisorAgent(Agent):
    def __init__(self, workers: List[Agent]):
        pass

    def execute(self, task: str) -> AgentResponse:
        # Decompose task
        # Delegate to appropriate workers
        # Aggregate results
        pass
```

### Exercise 3: Agent Memory

Add persistent memory so agents remember past interactions:

```python
class MemoryAgent(Agent):
    def __init__(self, *args, memory_file: str, **kwargs):
        pass

    def remember(self, key: str, value: str):
        pass

    def recall(self, key: str) -> Optional[str]:
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Pipeline executes agents in sequence
- [ ] Context passes between pipeline steps
- [ ] Debate reaches consensus or max rounds
- [ ] Hooks fire at correct points
- [ ] You understand when to use multi-agent

---

## Key Takeaway

> Specialized agents can outperform generalist approaches.

Multi-agent systems enable:
- **Separation of concerns** (each agent has one job)
- **Quality through review** (agents check each other)
- **Complex workflows** (pipeline, debate, ensemble)
- **Better traceability** (see what each agent did)

But remember: **complexity has costs**. Start simple.

---

## Get the Code

Related implementation: [8me/src/tier3.5-orchestration-concepts/02-patterns.md](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)

---

<div class="lab-navigation">
  <a href="./08-orchestrator.md" class="prev">← Previous: Lab 08 - Building an Orchestrator</a>
  <a href="./10-gating.md" class="next">Next: Lab 10 - Gating and Drift Prevention →</a>
</div>
