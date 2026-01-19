---
layout: lab
title: "Lab 12: Memory Integration"
lab_number: 12
difficulty: advanced
time: 1.5 hours
prerequisites: Lab 11 completed
---

# Lab 12: Memory Integration

Enable cross-session learning with persistent memory.

## Objectives

By the end of this lab, you will:
- Understand persistent memory patterns
- Integrate memory into your orchestrator
- Enable cross-session continuity
- Learn from past errors and successes

## Prerequisites

- Lab 11 completed (self-play)
- Understanding of the orchestrator pattern

## Why Memory?

Without memory, every session starts fresh:

```
Session 1: "User prefers TypeScript" → learned
Session 2: "What language?" → forgotten!

Session 1: "This approach failed" → learned
Session 2: *tries same failed approach* → wasted time!
```

With memory:

```
Session 1: Store("user", "prefers", "TypeScript")
Session 2: Recall("user", "prefers") → "TypeScript" ✓

Session 1: Store("approach_x", "status", "failed: caused timeout")
Session 2: Recall("approach_x") → avoid timeout approach ✓
```

---

## Memory Patterns

### Pattern 1: Entity-Attribute-Value

```python
memory.store(
    entity="project_alpha",
    attribute="tech_stack",
    value="Python, FastAPI, PostgreSQL"
)

# Later...
stack = memory.recall(entity="project_alpha", attribute="tech_stack")
```

### Pattern 2: Session Summaries

```python
# End of session
memory.summarize_session(
    session_id="2026-01-19-001",
    summary="Implemented login flow, fixed CORS issue",
    topics=["authentication", "cors", "api"]
)

# Next session
context = memory.get_context(limit=5)  # Last 5 session summaries
```

### Pattern 3: Error Patterns

```python
# When error occurs
memory.store(
    entity="error_patterns",
    attribute="timeout_api_call",
    value="Solution: Add retry with exponential backoff"
)

# When similar error occurs
solutions = memory.search("timeout")
```

---

## Step 1: Create the Memory Interface

Create `memory_interface.py`:

```python
"""
Memory Interface - Lab 12

Abstract interface for memory operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    """A single memory entry."""
    id: int
    entity: str
    attribute: str
    value: str
    source: str
    confidence: float
    created_at: datetime
    context: Optional[str] = None


@dataclass
class Entity:
    """An entity in the memory system."""
    id: int
    name: str
    entity_type: str
    aliases: List[str]


@dataclass
class SessionSummary:
    """Summary of a session."""
    session_id: str
    summary: str
    topics: List[str]
    created_at: datetime


class MemoryStore(ABC):
    """Abstract interface for memory storage."""

    @abstractmethod
    def store(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str = "orchestrator",
        confidence: float = 1.0,
        context: Optional[str] = None
    ) -> int:
        """Store a memory. Returns memory ID."""
        pass

    @abstractmethod
    def recall(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Recall memories with optional filters."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Full-text search across memories."""
        pass

    @abstractmethod
    def add_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None
    ) -> int:
        """Add a new entity. Returns entity ID."""
        pass

    @abstractmethod
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name or alias."""
        pass

    @abstractmethod
    def summarize_session(
        self,
        session_id: str,
        summary: str,
        topics: Optional[List[str]] = None
    ) -> int:
        """Store a session summary."""
        pass

    @abstractmethod
    def get_context(self, limit: int = 5) -> List[SessionSummary]:
        """Get recent session summaries for context."""
        pass


class InMemoryStore(MemoryStore):
    """Simple in-memory implementation for testing."""

    def __init__(self):
        self.memories: List[Memory] = []
        self.entities: Dict[str, Entity] = {}
        self.summaries: List[SessionSummary] = []
        self._next_id = 1

    def store(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str = "orchestrator",
        confidence: float = 1.0,
        context: Optional[str] = None
    ) -> int:
        memory = Memory(
            id=self._next_id,
            entity=entity,
            attribute=attribute,
            value=value,
            source=source,
            confidence=confidence,
            created_at=datetime.now(),
            context=context
        )
        self.memories.append(memory)
        self._next_id += 1
        return memory.id

    def recall(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        results = self.memories

        if entity:
            results = [m for m in results if m.entity == entity]
        if attribute:
            results = [m for m in results if m.attribute == attribute]

        # Return most recent first
        return sorted(results, key=lambda m: m.created_at, reverse=True)[:limit]

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        query_lower = query.lower()
        results = [
            m for m in self.memories
            if query_lower in m.value.lower() or
               query_lower in m.entity.lower() or
               query_lower in m.attribute.lower()
        ]
        return sorted(results, key=lambda m: m.created_at, reverse=True)[:limit]

    def add_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None
    ) -> int:
        entity = Entity(
            id=self._next_id,
            name=name,
            entity_type=entity_type,
            aliases=aliases or []
        )
        self.entities[name] = entity
        self._next_id += 1
        return entity.id

    def get_entity(self, name: str) -> Optional[Entity]:
        if name in self.entities:
            return self.entities[name]
        # Check aliases
        for entity in self.entities.values():
            if name in entity.aliases:
                return entity
        return None

    def summarize_session(
        self,
        session_id: str,
        summary: str,
        topics: Optional[List[str]] = None
    ) -> int:
        sess_summary = SessionSummary(
            session_id=session_id,
            summary=summary,
            topics=topics or [],
            created_at=datetime.now()
        )
        self.summaries.append(sess_summary)
        return len(self.summaries)

    def get_context(self, limit: int = 5) -> List[SessionSummary]:
        return sorted(
            self.summaries,
            key=lambda s: s.created_at,
            reverse=True
        )[:limit]
```

---

## Step 2: Create Memory-Aware Orchestrator

Create `memory_orchestrator.py`:

```python
"""
Memory-Aware Orchestrator - Lab 12

Orchestrator that learns from and contributes to memory.
"""

from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
import uuid

from memory_interface import MemoryStore, InMemoryStore, Memory
from orchestrator import Orchestrator, OrchestratorConfig
from interfaces import Task, TaskStore, Executor, Verifier, SafetyCheck


@dataclass
class SessionContext:
    """Context for the current session."""
    session_id: str
    started_at: datetime
    relevant_memories: List[Memory]
    prior_summaries: List[str]


class MemoryOrchestrator(Orchestrator):
    """
    Orchestrator with persistent memory integration.

    Extends base orchestrator with:
    - Session context loading
    - Progress persistence
    - Error pattern learning
    - Session summarization
    """

    def __init__(
        self,
        task_store: TaskStore,
        executor: Executor,
        memory: Optional[MemoryStore] = None,
        verifier: Optional[Verifier] = None,
        safety: Optional[SafetyCheck] = None,
        config: Optional[OrchestratorConfig] = None
    ):
        super().__init__(task_store, executor, verifier, safety, config)
        self.memory = memory or InMemoryStore()
        self.session: Optional[SessionContext] = None

    def start_session(self, project: str = "default") -> SessionContext:
        """
        Start a new session with memory context.

        Args:
            project: Project name for context filtering

        Returns:
            SessionContext with relevant memories
        """
        session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        # Load relevant memories
        relevant = self.memory.recall(entity=project, limit=20)

        # Get prior session summaries
        prior = self.memory.get_context(limit=5)
        prior_summaries = [s.summary for s in prior]

        self.session = SessionContext(
            session_id=session_id,
            started_at=datetime.now(),
            relevant_memories=relevant,
            prior_summaries=prior_summaries
        )

        # Store session start
        self.memory.store(
            entity=project,
            attribute="session_started",
            value=session_id,
            source="orchestrator"
        )

        return self.session

    def end_session(self, project: str = "default", summary: str = ""):
        """
        End session and store summary.

        Args:
            project: Project name
            summary: Session summary (auto-generated if empty)
        """
        if not self.session:
            return

        # Auto-generate summary if not provided
        if not summary:
            summary = self._generate_summary()

        # Extract topics from completed tasks
        topics = self._extract_topics()

        # Store session summary
        self.memory.summarize_session(
            session_id=self.session.session_id,
            summary=summary,
            topics=topics
        )

        # Store session end
        self.memory.store(
            entity=project,
            attribute="session_ended",
            value=f"{self.session.session_id}: {summary[:100]}",
            source="orchestrator"
        )

        self.session = None

    def remember_task_completion(self, task: Task, result: str):
        """Remember a successful task completion."""
        self.memory.store(
            entity=task.metadata.get("project", "default"),
            attribute=f"completed_{task.id}",
            value=f"Task: {task.description[:100]}... Result: {result[:100]}...",
            source="task_completion",
            context=f"Session: {self.session.session_id if self.session else 'unknown'}"
        )

    def remember_task_failure(self, task: Task, error: str):
        """Remember a task failure for future avoidance."""
        self.memory.store(
            entity="error_patterns",
            attribute=self._categorize_error(error),
            value=f"Task: {task.description[:100]}... Error: {error}",
            source="task_failure",
            confidence=0.9
        )

    def get_relevant_context(self, task: Task) -> str:
        """Get memory-based context relevant to a task."""
        # Search for relevant memories
        keywords = self._extract_keywords(task.description)
        relevant = []

        for keyword in keywords[:5]:  # Limit keyword searches
            memories = self.memory.search(keyword, limit=3)
            relevant.extend(memories)

        if not relevant:
            return ""

        # Deduplicate and format
        seen_ids = set()
        unique = []
        for m in relevant:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                unique.append(m)

        context_parts = ["## Relevant memories:"]
        for m in unique[:10]:
            context_parts.append(f"- [{m.entity}:{m.attribute}] {m.value}")

        return "\n".join(context_parts)

    def _generate_summary(self) -> str:
        """Generate session summary from stats."""
        stats = self.stats
        return (
            f"Processed {stats.tasks_processed} tasks: "
            f"{stats.tasks_completed} completed, "
            f"{stats.tasks_failed} failed. "
            f"Duration: {stats.elapsed_seconds:.0f}s."
        )

    def _extract_topics(self) -> List[str]:
        """Extract topics from tasks."""
        topics = set()
        for task in self.task_store.get_all():
            # Simple keyword extraction
            words = task.description.lower().split()
            for word in words:
                if len(word) > 5:
                    topics.add(word)
        return list(topics)[:10]

    def _categorize_error(self, error: str) -> str:
        """Categorize error for storage."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout_error"
        elif "rate limit" in error_lower:
            return "rate_limit_error"
        elif "authentication" in error_lower or "auth" in error_lower:
            return "auth_error"
        elif "not found" in error_lower:
            return "not_found_error"
        else:
            return "general_error"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple extraction - in production use NLP
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "in", "for", "on", "with"}
        words = []
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stop_words:
                words.append(word)
        return words


# Hook implementations for memory integration
def on_task_complete_memory(orchestrator: MemoryOrchestrator, task: Task, result):
    """Hook to remember task completion."""
    if hasattr(orchestrator, 'remember_task_completion'):
        orchestrator.remember_task_completion(task, result.output if result else "")


def on_task_fail_memory(orchestrator: MemoryOrchestrator, task: Task, result):
    """Hook to remember task failure."""
    if hasattr(orchestrator, 'remember_task_failure'):
        error = result.error if result and hasattr(result, 'error') else "Unknown error"
        orchestrator.remember_task_failure(task, error)
```

---

## Step 3: Minna Memory Integration

Create `minna_adapter.py`:

```python
"""
Minna Memory Adapter - Lab 12

Adapter for integrating with Minna Memory MCP server.
"""

from typing import Optional, List
from datetime import datetime

from memory_interface import MemoryStore, Memory, Entity, SessionSummary


class MinnaMemoryAdapter(MemoryStore):
    """
    Adapter for Minna Memory MCP server.

    In production, this would make actual MCP calls.
    This implementation shows the interface pattern.

    Usage with MCP:
        # Via Claude Code / MCP client
        memory_store(entity="project", attribute="status", value="in progress")
        memory_recall(entity="project")
        memory_search(query="timeout error")
    """

    def __init__(self, mcp_client=None):
        """
        Initialize adapter.

        Args:
            mcp_client: MCP client for making tool calls
        """
        self.mcp_client = mcp_client

    def store(
        self,
        entity: str,
        attribute: str,
        value: str,
        source: str = "orchestrator",
        confidence: float = 1.0,
        context: Optional[str] = None
    ) -> int:
        """Store using Minna's memory_store tool."""
        if self.mcp_client:
            # Real MCP call would look like:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_store",
            #     params={
            #         "entity": entity,
            #         "attribute": attribute,
            #         "value": value,
            #         "source": source,
            #         "confidence": confidence,
            #         "context": context
            #     }
            # )
            # return result["memory_id"]
            pass

        # Mock implementation for demonstration
        return 1

    def recall(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Recall using Minna's memory_recall tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_recall",
            #     params={"entity": entity, "attribute": attribute, "limit": limit}
            # )
            # return [Memory(**m) for m in result["memories"]]
            pass

        return []

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """Search using Minna's memory_search tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_search",
            #     params={"query": query, "limit": limit}
            # )
            # return [Memory(**m) for m in result["memories"]]
            pass

        return []

    def add_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None
    ) -> int:
        """Add entity using Minna's memory_add_entity tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_add_entity",
            #     params={"name": name, "entity_type": entity_type, "aliases": aliases}
            # )
            # return result["entity_id"]
            pass

        return 1

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity using Minna's memory_get_entity tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_get_entity",
            #     params={"name": name}
            # )
            # return Entity(**result) if result else None
            pass

        return None

    def summarize_session(
        self,
        session_id: str,
        summary: str,
        topics: Optional[List[str]] = None
    ) -> int:
        """Summarize using Minna's memory_summarize_session tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_summarize_session",
            #     params={"session_id": session_id, "summary": summary, "topics": topics}
            # )
            # return result["summary_id"]
            pass

        return 1

    def get_context(self, limit: int = 5) -> List[SessionSummary]:
        """Get context using Minna's memory_get_context tool."""
        if self.mcp_client:
            # Real MCP call:
            # result = self.mcp_client.call_tool(
            #     "mcp__minna-memory__memory_get_context",
            #     params={"limit": limit}
            # )
            # return [SessionSummary(**s) for s in result["summaries"]]
            pass

        return []
```

---

## Step 4: Complete Example

Create `memory_demo.py`:

```python
"""
Memory Integration Demo - Lab 12

Demonstrates memory-aware orchestration.
"""

from memory_interface import InMemoryStore
from memory_orchestrator import (
    MemoryOrchestrator,
    on_task_complete_memory,
    on_task_fail_memory
)
from components import JSONTaskStore, ClaudeExecutor, SimpleCircuitBreaker
from orchestrator import OrchestratorConfig


def simulate_multiple_sessions():
    """Simulate multiple sessions with memory persistence."""
    print("=" * 60)
    print("MEMORY INTEGRATION DEMO")
    print("=" * 60)

    # Shared memory across sessions
    memory = InMemoryStore()

    # Pre-populate with some "historical" memories
    memory.add_entity("my_project", "project", aliases=["proj", "myproj"])
    memory.store("my_project", "tech_stack", "Python, FastAPI", source="initial_setup")
    memory.store("error_patterns", "timeout_error", "Use retry with backoff", source="past_learning")

    # SESSION 1
    print("\n--- SESSION 1 ---")
    run_session(memory, session_num=1)

    # SESSION 2 (with memory from session 1)
    print("\n--- SESSION 2 ---")
    run_session(memory, session_num=2)

    # Show accumulated memory
    print("\n--- ACCUMULATED MEMORY ---")
    all_memories = memory.recall(limit=20)
    for m in all_memories:
        print(f"  [{m.entity}:{m.attribute}] {m.value[:60]}...")

    print("\n--- SESSION SUMMARIES ---")
    summaries = memory.get_context(limit=5)
    for s in summaries:
        print(f"  [{s.session_id}] {s.summary}")


def run_session(memory: InMemoryStore, session_num: int):
    """Run a single orchestrator session."""
    # Create components
    store = JSONTaskStore(f"tasks_session_{session_num}.json")
    executor = ClaudeExecutor()
    safety = SimpleCircuitBreaker(max_iterations=10)

    # Create memory-aware orchestrator
    orchestrator = MemoryOrchestrator(
        task_store=store,
        executor=executor,
        memory=memory,
        safety=safety,
        config=OrchestratorConfig(verify_results=False)
    )

    # Add memory hooks
    orchestrator.add_hook("on_task_complete", on_task_complete_memory)
    orchestrator.add_hook("on_task_fail", on_task_fail_memory)

    # Start session and load context
    context = orchestrator.start_session(project="my_project")

    print(f"Session: {context.session_id}")
    print(f"Loaded {len(context.relevant_memories)} memories")
    print(f"Prior summaries: {len(context.prior_summaries)}")

    # Show relevant context
    if context.relevant_memories:
        print("Relevant memories:")
        for m in context.relevant_memories[:3]:
            print(f"  - {m.entity}:{m.attribute} = {m.value[:50]}...")

    # Add sample tasks
    if len(store.tasks) == 0:
        store.add(f"Task for session {session_num}: Write a hello world")
        store.add(f"Task for session {session_num}: Explain Python basics")

    # Run
    stats = orchestrator.run()
    print(f"Completed: {stats.tasks_completed}/{stats.tasks_processed}")

    # End session
    orchestrator.end_session(
        project="my_project",
        summary=f"Session {session_num}: Processed {stats.tasks_processed} tasks"
    )


def demo_error_learning():
    """Demonstrate learning from errors."""
    print("\n--- ERROR LEARNING DEMO ---")

    memory = InMemoryStore()

    # Simulate first encounter with error
    memory.store(
        entity="error_patterns",
        attribute="api_timeout",
        value="Caused by large payload. Solution: chunk requests into smaller batches",
        source="session_1"
    )

    # Later session encounters similar error
    print("\nSearching for timeout solutions...")
    results = memory.search("timeout")
    for r in results:
        print(f"Found: {r.value}")


if __name__ == "__main__":
    simulate_multiple_sessions()
    demo_error_learning()
```

---

## Memory Best Practices

### What to Remember

| Good to Store | Why |
|--------------|-----|
| User preferences | Personalization |
| Error solutions | Avoid repeating mistakes |
| Project context | Continuity |
| Session summaries | Quick catchup |
| Entity relationships | Understanding connections |

### What NOT to Store

| Avoid Storing | Why |
|--------------|-----|
| Raw API responses | Too verbose |
| Temporary state | Not useful long-term |
| Sensitive data | Security risk |
| Duplicate info | Wastes space |

### Memory Decay

Older memories may become less relevant:

```python
# Minna Memory supports confidence decay
memory.recall(
    entity="project",
    include_decay=True,      # Apply time-based decay
    half_life_days=90,       # Confidence halves every 90 days
    min_confidence=0.3       # Filter out low-confidence memories
)
```

---

## Exercises

### Exercise 1: Semantic Memory Search

Implement similarity-based memory search:

```python
class SemanticMemoryStore(MemoryStore):
    def search_similar(self, text: str, threshold: float = 0.7) -> List[Memory]:
        """Find memories semantically similar to text."""
        # Use embeddings to find similar memories
        pass
```

### Exercise 2: Memory Consolidation

Implement automatic consolidation of related memories:

```python
class MemoryConsolidator:
    def consolidate(self, memories: List[Memory]) -> Memory:
        """Merge related memories into a single summary."""
        pass
```

### Exercise 3: Conflict Detection

Detect and resolve conflicting memories:

```python
class ConflictResolver:
    def detect_conflicts(self, entity: str, attribute: str) -> List[Memory]:
        """Find memories that contradict each other."""
        pass

    def resolve(self, conflicts: List[Memory]) -> Memory:
        """Resolve conflicts by recency, confidence, or AI."""
        pass
```

---

## Checkpoint

Before moving on, verify:
- [ ] Memory persists across sessions
- [ ] Context is loaded at session start
- [ ] Errors are stored for future reference
- [ ] Session summaries enable quick catchup
- [ ] You understand entity-attribute-value pattern

---

## Key Takeaway

> Memory enables learning across sessions.

With persistent memory:
- **Continuity**: Don't start from scratch each time
- **Learning**: Remember what worked and what didn't
- **Personalization**: Adapt to user preferences
- **Efficiency**: Don't repeat discovered solutions

---

## Get the Code

Minna Memory: Available as an MCP server at [minna-memory](https://github.com/fbratten/minna-memory)

Related concepts: [8me/src/tier3.5-orchestration-concepts/05-minna-memory.md](https://github.com/fbratten/8me/tree/main/src/tier3.5-orchestration-concepts)

---

<div class="lab-navigation">
  <a href="./11-self-play.md" class="prev">← Previous: Lab 11 - Self-Play Oscillation</a>
  <a href="./13-mcp-server.md" class="next">Next: Lab 13 - MCP Server Deployment →</a>
</div>
