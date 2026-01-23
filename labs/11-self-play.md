---
layout: default
title: "Lab 11: Self-Play Oscillation"
nav_order: 11
parent: Labs
lab_number: 11
difficulty: advanced
time: 2 hours
prerequisites: Lab 10 completed
---

# Lab 11: Self-Play Oscillation

Use internal debate to refine outputs through proposer/critic cycles.

## Objectives

By the end of this lab, you will:
- Understand self-play and DIALECTIC patterns
- Implement proposer/critic feedback loops
- Detect and handle convergence
- Prevent infinite oscillation

## Prerequisites

- Lab 10 completed (gating)
- Understanding of multi-agent patterns

## What is Self-Play?

Self-play uses **internal debate** to improve outputs:

```
Single Pass:            Self-Play:

Task → AI → Output      Task → Proposer ──┐
                                          │
                              ┌───────────┘
                              ▼
                           Critic ◄──────┐
                              │          │
                              ▼          │
                           Good? ──No────┘
                              │
                             Yes
                              ▼
                           Output
```

Benefits:
- **Catches errors** the first pass missed
- **Improves quality** through iteration
- **Provides reasoning** about why choices were made

---

## The DIALECTIC Pattern

```
      ┌──────────────────────────────────────────┐
      │                                          │
      │    ┌──────────┐      ┌──────────┐       │
      │    │ Proposer │      │  Critic  │       │
      │    └────┬─────┘      └────┬─────┘       │
      │         │                 │             │
      │         ▼                 │             │
      │    "Here's my            │             │
      │     proposal"             │             │
      │         │                 │             │
      │         └────────────────►│             │
      │                          │             │
      │                          ▼             │
      │                    "Issues found:      │
      │                     - Problem A        │
      │                     - Problem B"       │
      │                          │             │
      │         ◄────────────────┘             │
      │         │                              │
      │         ▼                              │
      │    "Refined based                      │
      │     on feedback"                       │
      │         │                              │
      └─────────┴──────────────────────────────┘
                │
                ▼
          Convergence?
           /       \
         No         Yes
          │          │
          ▼          ▼
      Continue    Output
```

---

## Step 1: Create the Self-Play Engine

Create `self_play.py`:

```python
"""
Self-Play Engine - Lab 11

Implements proposer/critic feedback loops with convergence detection.
"""

from typing import Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import anthropic


class ConvergenceState(Enum):
    """State of the self-play loop."""
    RUNNING = "running"
    CONVERGED = "converged"
    OSCILLATING = "oscillating"
    MAX_ROUNDS = "max_rounds"
    STALLED = "stalled"


@dataclass
class Critique:
    """Critique from the critic."""
    approved: bool
    score: float  # 0-1
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SelfPlayRound:
    """Record of a single round."""
    round_number: int
    proposal: str
    critique: Critique
    proposal_hash: str  # For oscillation detection


@dataclass
class SelfPlayResult:
    """Result of self-play loop."""
    final_output: str
    convergence_state: ConvergenceState
    rounds: List[SelfPlayRound]
    final_score: float

    @property
    def converged(self) -> bool:
        return self.convergence_state == ConvergenceState.CONVERGED

    @property
    def round_count(self) -> int:
        return len(self.rounds)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""
    max_rounds: int = 5
    approval_threshold: float = 0.85
    improvement_threshold: float = 0.05  # Min improvement to continue
    oscillation_window: int = 3  # Rounds to check for oscillation
    model: str = "claude-sonnet-4-20250514"


class SelfPlayEngine:
    """
    Runs self-play loops with proposer and critic roles.

    Usage:
        engine = SelfPlayEngine()
        result = engine.run("Write a sorting algorithm")

        if result.converged:
            print(result.final_output)
        else:
            print(f"Did not converge: {result.convergence_state}")
    """

    def __init__(self, config: Optional[SelfPlayConfig] = None):
        self.config = config or SelfPlayConfig()
        self.client = anthropic.Anthropic()
        self.rounds: List[SelfPlayRound] = []

    def run(self, task: str, context: str = "") -> SelfPlayResult:
        """
        Run self-play loop until convergence or max rounds.

        Args:
            task: The task to complete
            context: Optional additional context

        Returns:
            SelfPlayResult with final output and convergence state
        """
        self.rounds = []
        proposal = None
        critique = None
        scores: List[float] = []

        for round_num in range(1, self.config.max_rounds + 1):
            # Generate proposal
            proposal = self._generate_proposal(task, context, proposal, critique)
            proposal_hash = self._hash_proposal(proposal)

            # Get critique
            critique = self._get_critique(task, proposal)
            scores.append(critique.score)

            # Record round
            self.rounds.append(SelfPlayRound(
                round_number=round_num,
                proposal=proposal,
                critique=critique,
                proposal_hash=proposal_hash
            ))

            # Check for convergence
            if critique.approved and critique.score >= self.config.approval_threshold:
                return SelfPlayResult(
                    final_output=proposal,
                    convergence_state=ConvergenceState.CONVERGED,
                    rounds=self.rounds,
                    final_score=critique.score
                )

            # Check for oscillation
            if self._is_oscillating():
                return SelfPlayResult(
                    final_output=proposal,
                    convergence_state=ConvergenceState.OSCILLATING,
                    rounds=self.rounds,
                    final_score=critique.score
                )

            # Check for stalling (no improvement)
            if len(scores) >= 2:
                improvement = scores[-1] - scores[-2]
                if improvement < self.config.improvement_threshold and scores[-1] < self.config.approval_threshold:
                    # Allow one more try
                    if len(scores) >= 3 and all(
                        scores[i] - scores[i-1] < self.config.improvement_threshold
                        for i in range(-2, 0)
                    ):
                        return SelfPlayResult(
                            final_output=proposal,
                            convergence_state=ConvergenceState.STALLED,
                            rounds=self.rounds,
                            final_score=critique.score
                        )

        # Max rounds reached
        return SelfPlayResult(
            final_output=proposal,
            convergence_state=ConvergenceState.MAX_ROUNDS,
            rounds=self.rounds,
            final_score=scores[-1] if scores else 0.0
        )

    def _generate_proposal(
        self,
        task: str,
        context: str,
        previous_proposal: Optional[str],
        previous_critique: Optional[Critique]
    ) -> str:
        """Generate or refine a proposal."""
        if previous_proposal is None:
            # Initial proposal
            prompt = f"""You are a proposal generator. Create a high-quality response for this task.

TASK: {task}

{f"CONTEXT: {context}" if context else ""}

Provide your best proposal. Focus on completeness and correctness."""
        else:
            # Refined proposal
            prompt = f"""You are a proposal generator. Refine your previous proposal based on feedback.

TASK: {task}

{f"CONTEXT: {context}" if context else ""}

YOUR PREVIOUS PROPOSAL:
{previous_proposal}

CRITIQUE RECEIVED:
Score: {previous_critique.score:.0%}
Feedback: {previous_critique.feedback}
Issues: {', '.join(previous_critique.issues) if previous_critique.issues else 'None'}
Suggestions: {', '.join(previous_critique.suggestions) if previous_critique.suggestions else 'None'}

Provide an improved proposal that addresses ALL the issues raised. Keep what works, fix what doesn't."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _get_critique(self, task: str, proposal: str) -> Critique:
        """Get critique for a proposal."""
        prompt = f"""You are a strict but fair critic. Evaluate this proposal thoroughly.

ORIGINAL TASK: {task}

PROPOSAL TO EVALUATE:
{proposal}

Evaluate:
1. Does it fully address the task?
2. Is it correct and accurate?
3. Is it well-structured?
4. Are there any issues or errors?
5. How could it be improved?

Respond in this exact format:
APPROVED: yes or no (yes = ready to submit, no = needs improvement)
SCORE: 0.0 to 1.0 (overall quality score)
FEEDBACK: (2-3 sentence overall assessment)
ISSUES: (comma-separated list of problems, or "None")
SUGGESTIONS: (comma-separated list of improvements, or "None")

Be constructive but thorough. Don't approve unless quality is genuinely high."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_critique(response.content[0].text)

    def _parse_critique(self, text: str) -> Critique:
        """Parse critique response."""
        approved = False
        score = 0.5
        feedback = ""
        issues = []
        suggestions = []

        for line in text.split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("APPROVED:"):
                approved = "yes" in line.lower()
            elif upper.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip().split()[0])
                    score = max(0.0, min(1.0, score))
                except:
                    pass
            elif upper.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()
            elif upper.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() != "none":
                    issues = [i.strip() for i in issues_str.split(",") if i.strip()]
            elif upper.startswith("SUGGESTIONS:"):
                sugg_str = line.split(":", 1)[1].strip()
                if sugg_str.lower() != "none":
                    suggestions = [s.strip() for s in sugg_str.split(",") if s.strip()]

        return Critique(
            approved=approved,
            score=score,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions
        )

    def _hash_proposal(self, proposal: str) -> str:
        """Create hash for oscillation detection."""
        import hashlib
        normalized = " ".join(proposal.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _is_oscillating(self) -> bool:
        """Check if proposals are oscillating between states."""
        if len(self.rounds) < self.config.oscillation_window:
            return False

        recent_hashes = [r.proposal_hash for r in self.rounds[-self.config.oscillation_window:]]

        # Check if any hash appears more than once
        seen = set()
        for h in recent_hashes:
            if h in seen:
                return True
            seen.add(h)

        return False
```

---

## Step 2: Add Oscillation Prevention

Add to `self_play.py`:

```python
class OscillationPreventer:
    """
    Prevents and handles oscillation in self-play loops.

    Strategies:
    1. Temperature increase: Make proposals more varied
    2. Constraint injection: Add explicit constraints
    3. Best-so-far: Return the best proposal seen
    """

    def __init__(self):
        self.seen_proposals: dict[str, float] = {}  # hash -> score
        self.oscillation_count: int = 0

    def record_proposal(self, proposal_hash: str, score: float):
        """Record a proposal and its score."""
        if proposal_hash in self.seen_proposals:
            self.oscillation_count += 1
        self.seen_proposals[proposal_hash] = max(
            self.seen_proposals.get(proposal_hash, 0),
            score
        )

    def get_best_proposal(self, rounds: List[SelfPlayRound]) -> SelfPlayRound:
        """Get the best proposal seen so far."""
        return max(rounds, key=lambda r: r.critique.score)

    def suggest_escape_strategy(self) -> str:
        """Suggest a strategy to escape oscillation."""
        if self.oscillation_count <= 1:
            return "increase_temperature"
        elif self.oscillation_count <= 2:
            return "add_constraints"
        else:
            return "return_best"

    def get_constraint_injection(self, issues: List[str]) -> str:
        """Generate explicit constraints from issues."""
        if not issues:
            return ""

        constraints = ["You MUST address these specific issues:"]
        for i, issue in enumerate(issues, 1):
            constraints.append(f"{i}. {issue}")
        constraints.append("\nDo NOT repeat previous approaches that had these issues.")

        return "\n".join(constraints)


class AdaptiveSelfPlayEngine(SelfPlayEngine):
    """
    Self-play engine with adaptive strategies to prevent oscillation.
    """

    def __init__(self, config: Optional[SelfPlayConfig] = None):
        super().__init__(config)
        self.preventer = OscillationPreventer()
        self.base_temperature = 0.7

    def run(self, task: str, context: str = "") -> SelfPlayResult:
        """Run with adaptive oscillation prevention."""
        self.rounds = []
        self.preventer = OscillationPreventer()  # Reset

        proposal = None
        critique = None
        temperature = self.base_temperature
        extra_constraints = ""

        for round_num in range(1, self.config.max_rounds + 1):
            # Generate proposal with current settings
            proposal = self._generate_proposal_adaptive(
                task, context, proposal, critique,
                temperature=temperature,
                extra_constraints=extra_constraints
            )
            proposal_hash = self._hash_proposal(proposal)

            # Record and check for oscillation
            self.preventer.record_proposal(
                proposal_hash,
                critique.score if critique else 0
            )

            # Get critique
            critique = self._get_critique(task, proposal)

            # Record round
            self.rounds.append(SelfPlayRound(
                round_number=round_num,
                proposal=proposal,
                critique=critique,
                proposal_hash=proposal_hash
            ))

            # Check for convergence
            if critique.approved and critique.score >= self.config.approval_threshold:
                return SelfPlayResult(
                    final_output=proposal,
                    convergence_state=ConvergenceState.CONVERGED,
                    rounds=self.rounds,
                    final_score=critique.score
                )

            # Handle oscillation adaptively
            if self._is_oscillating():
                strategy = self.preventer.suggest_escape_strategy()

                if strategy == "return_best":
                    best = self.preventer.get_best_proposal(self.rounds)
                    return SelfPlayResult(
                        final_output=best.proposal,
                        convergence_state=ConvergenceState.OSCILLATING,
                        rounds=self.rounds,
                        final_score=best.critique.score
                    )
                elif strategy == "increase_temperature":
                    temperature = min(1.0, temperature + 0.1)
                elif strategy == "add_constraints":
                    extra_constraints = self.preventer.get_constraint_injection(
                        critique.issues
                    )

        # Max rounds - return best
        best = self.preventer.get_best_proposal(self.rounds)
        return SelfPlayResult(
            final_output=best.proposal,
            convergence_state=ConvergenceState.MAX_ROUNDS,
            rounds=self.rounds,
            final_score=best.critique.score
        )

    def _generate_proposal_adaptive(
        self,
        task: str,
        context: str,
        previous_proposal: Optional[str],
        previous_critique: Optional[Critique],
        temperature: float = 0.7,
        extra_constraints: str = ""
    ) -> str:
        """Generate proposal with adaptive parameters."""
        # Build prompt (similar to base class)
        if previous_proposal is None:
            prompt = f"""You are a proposal generator. Create a high-quality response.

TASK: {task}

{f"CONTEXT: {context}" if context else ""}
{f"CONSTRAINTS: {extra_constraints}" if extra_constraints else ""}

Provide your best proposal."""
        else:
            prompt = f"""Refine your previous proposal based on feedback.

TASK: {task}
{f"CONTEXT: {context}" if context else ""}
{f"CONSTRAINTS: {extra_constraints}" if extra_constraints else ""}

PREVIOUS PROPOSAL:
{previous_proposal}

CRITIQUE:
Score: {previous_critique.score:.0%}
Issues: {', '.join(previous_critique.issues)}

Address ALL issues. Try a DIFFERENT approach if the same approach keeps failing."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
```

---

## Step 3: Complete Example

Create `self_play_demo.py`:

```python
"""
Self-Play Demo - Lab 11

Demonstrates self-play with convergence and oscillation handling.
"""

from self_play import (
    SelfPlayEngine, AdaptiveSelfPlayEngine, SelfPlayConfig,
    ConvergenceState
)


def demo_basic_self_play():
    """Demonstrate basic self-play."""
    print("=" * 60)
    print("BASIC SELF-PLAY DEMO")
    print("=" * 60)

    config = SelfPlayConfig(
        max_rounds=5,
        approval_threshold=0.85
    )

    engine = SelfPlayEngine(config)

    task = "Write a Python function that validates email addresses. Include edge cases."

    print(f"\nTask: {task}\n")
    print("Running self-play loop...\n")

    result = engine.run(task)

    print(f"Convergence: {result.convergence_state.value}")
    print(f"Rounds: {result.round_count}")
    print(f"Final Score: {result.final_score:.0%}")

    print("\n--- Round History ---")
    for round_data in result.rounds:
        print(f"\nRound {round_data.round_number}:")
        print(f"  Score: {round_data.critique.score:.0%}")
        print(f"  Approved: {round_data.critique.approved}")
        print(f"  Feedback: {round_data.critique.feedback[:100]}...")
        if round_data.critique.issues:
            print(f"  Issues: {', '.join(round_data.critique.issues[:3])}")

    print("\n--- Final Output ---")
    print(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output)


def demo_adaptive_self_play():
    """Demonstrate adaptive self-play with oscillation prevention."""
    print("\n" + "=" * 60)
    print("ADAPTIVE SELF-PLAY DEMO")
    print("=" * 60)

    config = SelfPlayConfig(
        max_rounds=7,
        approval_threshold=0.80,
        oscillation_window=3
    )

    engine = AdaptiveSelfPlayEngine(config)

    # Intentionally tricky task that might cause oscillation
    task = """Write a function that finds the optimal solution.
    It should be efficient but also readable.
    Balance performance with maintainability.
    Consider edge cases but don't over-engineer."""

    print(f"\nTask: {task}\n")
    print("Running adaptive self-play loop...\n")

    result = engine.run(task)

    print(f"Convergence: {result.convergence_state.value}")
    print(f"Rounds: {result.round_count}")
    print(f"Final Score: {result.final_score:.0%}")

    # Show score progression
    scores = [r.critique.score for r in result.rounds]
    print(f"\nScore progression: {' → '.join(f'{s:.0%}' for s in scores)}")

    if result.convergence_state == ConvergenceState.OSCILLATING:
        print("\n⚠️ Oscillation detected - returned best proposal")
    elif result.convergence_state == ConvergenceState.CONVERGED:
        print("\n✓ Converged successfully")
    else:
        print(f"\n⚡ Stopped: {result.convergence_state.value}")


def main():
    demo_basic_self_play()
    demo_adaptive_self_play()


if __name__ == "__main__":
    main()
```

---

## Understanding Self-Play

### When It Works Well

| Good For | Why |
|----------|-----|
| Code writing | Bugs are catchable |
| Writing/editing | Style is improvable |
| Problem solving | Solutions can be verified |
| Data validation | Errors are detectable |

### When It Struggles

| Challenging | Why |
|-------------|-----|
| Subjective tasks | No clear "better" |
| Creative tasks | Different ≠ better |
| Speed-critical | Multiple rounds = slow |
| Simple tasks | Overkill |

### Oscillation Patterns

```
Type 1: Flip-Flop
Round 1: "Use approach A" → "Too complex"
Round 2: "Use approach B" → "Too simple"
Round 3: "Use approach A" → "Too complex"
...

Type 2: Incremental Reversal
Round 1: "Add feature X" → "Missing Y"
Round 2: "Add feature Y" → "X is now broken"
Round 3: "Fix X" → "Y is now broken"
...

Type 3: Perfection Loop
Round 1: Score 0.82 → "Improve error handling"
Round 2: Score 0.84 → "Simplify error handling"
Round 3: Score 0.83 → "Improve error handling"
...
```

---

## Exercises

### Exercise 1: Weighted Critic

Implement a critic that weighs different aspects:

```python
class WeightedCritique:
    correctness: float  # Weight: 0.4
    completeness: float  # Weight: 0.3
    style: float  # Weight: 0.2
    efficiency: float  # Weight: 0.1

    @property
    def weighted_score(self) -> float:
        pass
```

### Exercise 2: Multi-Critic Ensemble

Use multiple critics and aggregate their feedback:

```python
class CriticEnsemble:
    def __init__(self, critics: List[Critic]):
        pass

    def evaluate(self, proposal: str) -> Critique:
        # Run all critics
        # Aggregate scores and feedback
        pass
```

### Exercise 3: Convergence Prediction

Predict if the loop will converge based on early rounds:

```python
class ConvergencePredictor:
    def predict(self, rounds: List[SelfPlayRound]) -> float:
        """Predict probability of convergence."""
        # Analyze score trend
        # Check for oscillation patterns
        # Return probability
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Self-play loop runs multiple rounds
- [ ] Convergence is detected correctly
- [ ] Oscillation is detected and handled
- [ ] Adaptive strategies improve outcomes
- [ ] You understand when to use self-play

---

## Key Takeaway

> Self-play catches issues that single-pass misses.

Self-play provides:
- **Iterative improvement** through feedback
- **Quality assurance** via internal critique
- **Robustness** by catching errors early
- **Transparency** through round-by-round history

But watch out for:
- **Oscillation** (going in circles)
- **Over-refinement** (diminishing returns)
- **Cost** (multiple API calls)

---

## Get the Code

Related concepts: [8me/src/tier3.5-orchestration-concepts/02-patterns.md](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)

---

<div class="lab-navigation">
  <a href="./10-gating" class="prev">← Previous: Lab 10 - Gating and Drift Prevention</a>
  <a href="./12-memory" class="next">Next: Lab 12 - Memory Integration →</a>
</div>
