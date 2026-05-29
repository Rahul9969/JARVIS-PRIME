"""
JARVIS-PRIME Hierarchical Memory System
========================================

Four-tier memory architecture:
    1. Working Memory — Current context, active goals (in-memory / Redis)
    2. Procedural Memory — Tool usage patterns, workflows (versioned YAML)
    3. Semantic Memory — Domain knowledge, ontologies (Neo4j + Weaviate)
    4. Episodic Memory — Session logs, conversation history (PostgreSQL/SQLite)

Phase 0: All tiers use in-memory implementations with optional persistence.
"""
from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5
    domain: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "importance": self.importance,
            "domain": self.domain,
        }


class WorkingMemory:
    """
    Tier 1: Working Memory (in-memory).
    Holds current context, active goals, and constraints.
    Fast access, limited capacity, auto-evicts old entries.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._store: dict[str, MemoryEntry] = {}
        self.active_goals: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    def set(self, key: str, content: str, **metadata: Any) -> None:
        """Store a working memory entry."""
        entry_id = hashlib.md5(key.encode()).hexdigest()[:12]
        self._store[key] = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
        )
        self._evict_if_needed()

    def get(self, key: str) -> str | None:
        """Retrieve a working memory entry."""
        entry = self._store.get(key)
        if entry:
            entry.access_count += 1
            return entry.content
        return None

    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        """Get a context variable."""
        return self.context.get(key)

    def add_goal(self, goal: dict[str, Any]) -> None:
        """Add an active goal."""
        self.active_goals.append(goal)

    def clear_goals(self) -> None:
        """Clear all active goals."""
        self.active_goals.clear()

    def _evict_if_needed(self) -> None:
        """Evict least-accessed entries when over capacity."""
        if len(self._store) > self.max_entries:
            sorted_entries = sorted(
                self._store.items(),
                key=lambda x: (x[1].access_count, x[1].timestamp),
            )
            to_remove = len(self._store) - self.max_entries
            for key, _ in sorted_entries[:to_remove]:
                del self._store[key]

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of working memory state."""
        return {
            "entries": {k: v.to_dict() for k, v in self._store.items()},
            "active_goals": self.active_goals,
            "context": self.context,
        }


class ProceduralMemory:
    """
    Tier 2: Procedural Memory.
    Stores tool usage patterns, agent workflows, and learned skills.
    Persisted as versioned JSON files.
    """

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = persist_dir
        self._procedures: dict[str, dict[str, Any]] = {}
        self._tool_patterns: dict[str, list[dict[str, Any]]] = {}
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def store_procedure(
        self,
        name: str,
        steps: list[str],
        domain: str = "general",
        success_rate: float = 1.0,
    ) -> None:
        """Store a learned procedure."""
        self._procedures[name] = {
            "name": name,
            "steps": steps,
            "domain": domain,
            "success_rate": success_rate,
            "usage_count": 0,
            "last_used": time.time(),
        }
        self._persist()

    def get_procedure(self, name: str) -> dict[str, Any] | None:
        """Retrieve a stored procedure."""
        proc = self._procedures.get(name)
        if proc:
            proc["usage_count"] += 1
            proc["last_used"] = time.time()
        return proc

    def record_tool_usage(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result_quality: float,
        context: str = "",
    ) -> None:
        """Record a tool usage pattern for future optimization."""
        if tool_name not in self._tool_patterns:
            self._tool_patterns[tool_name] = []
        self._tool_patterns[tool_name].append({
            "parameters": parameters,
            "result_quality": result_quality,
            "context": context,
            "timestamp": time.time(),
        })

    def get_best_tool_params(self, tool_name: str) -> dict[str, Any] | None:
        """Get the best-performing parameters for a tool."""
        patterns = self._tool_patterns.get(tool_name, [])
        if not patterns:
            return None
        best = max(patterns, key=lambda x: x["result_quality"])
        return best["parameters"]

    def list_procedures(self, domain: str | None = None) -> list[str]:
        """List available procedures, optionally filtered by domain."""
        if domain:
            return [
                name for name, proc in self._procedures.items()
                if proc["domain"] == domain
            ]
        return list(self._procedures.keys())

    def _persist(self) -> None:
        """Persist procedures to disk."""
        if self.persist_dir:
            filepath = self.persist_dir / "procedures.json"
            filepath.write_text(json.dumps(self._procedures, indent=2, default=str))

    def _load(self) -> None:
        """Load procedures from disk."""
        if self.persist_dir:
            filepath = self.persist_dir / "procedures.json"
            if filepath.exists():
                self._procedures = json.loads(filepath.read_text())


class SemanticMemory:
    """
    Tier 3: Semantic Memory (In-Memory implementation for Phase 0).
    Stores domain knowledge, research findings, and ontological relationships.

    Phase 0: Simple in-memory store with keyword search.
    Phase 1+: Neo4j knowledge graph + Weaviate vector embeddings.
    """

    def __init__(self) -> None:
        self._facts: dict[str, MemoryEntry] = {}
        self._relationships: list[tuple[str, str, str]] = []  # (subject, predicate, object)

    def store_fact(
        self,
        fact_id: str,
        content: str,
        domain: str = "general",
        importance: float = 0.5,
        **metadata: Any,
    ) -> None:
        """Store a semantic fact."""
        self._facts[fact_id] = MemoryEntry(
            id=fact_id,
            content=content,
            domain=domain,
            importance=importance,
            metadata=metadata,
        )

    def store_relationship(self, subject: str, predicate: str, obj: str) -> None:
        """Store a semantic relationship (knowledge graph triple)."""
        self._relationships.append((subject, predicate, obj))

    async def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Query semantic memory with keyword matching.
        Phase 1+: Will use vector similarity + graph traversal.
        """
        query_lower = query_text.lower()
        scored = []
        for fact in self._facts.values():
            # Simple keyword overlap scoring
            words = set(query_lower.split())
            fact_words = set(fact.content.lower().split())
            overlap = len(words & fact_words)
            if overlap > 0:
                score = overlap * fact.importance
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry.to_dict() for _, entry in scored[:top_k]]

    def get_relationships(self, entity: str) -> list[tuple[str, str, str]]:
        """Get all relationships involving an entity."""
        return [
            (s, p, o) for s, p, o in self._relationships
            if s == entity or o == entity
        ]

    def stats(self) -> dict[str, int]:
        """Get memory statistics."""
        return {
            "facts": len(self._facts),
            "relationships": len(self._relationships),
            "domains": len(set(f.domain for f in self._facts.values())),
        }


class EpisodicMemory:
    """
    Tier 4: Episodic Memory.
    Stores session logs, goal processing history, and agent interactions.
    Enables the system to learn from past experiences.

    Phase 0: JSON file-based persistence.
    Phase 1+: PostgreSQL + pgvector for semantic search over episodes.
    """

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = persist_dir
        self._episodes: list[dict[str, Any]] = []
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def store_episode(
        self,
        query: str,
        result: dict[str, Any],
        agents_used: list[str] | None = None,
        duration_seconds: float = 0.0,
        success: bool = True,
    ) -> str:
        """Store a goal-processing episode."""
        episode_id = f"EP-{len(self._episodes):06d}"
        episode = {
            "id": episode_id,
            "timestamp": time.time(),
            "query": query,
            "result": result,
            "agents_used": agents_used or [],
            "duration_seconds": duration_seconds,
            "success": success,
        }
        self._episodes.append(episode)
        self._persist()
        return episode_id

    def get_similar_episodes(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Find similar past episodes (keyword matching for Phase 0)."""
        query_words = set(query.lower().split())
        scored = []
        for episode in self._episodes:
            ep_words = set(episode["query"].lower().split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                scored.append((overlap, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_failure_patterns(self) -> list[dict[str, Any]]:
        """Identify recurring failure patterns for SICA analysis."""
        failures = [ep for ep in self._episodes if not ep["success"]]
        return failures

    def stats(self) -> dict[str, Any]:
        """Get episodic memory statistics."""
        total = len(self._episodes)
        successes = sum(1 for ep in self._episodes if ep["success"])
        return {
            "total_episodes": total,
            "success_rate": successes / max(total, 1),
            "domains_covered": len(set(
                agent
                for ep in self._episodes
                for agent in ep.get("agents_used", [])
            )),
        }

    def _persist(self) -> None:
        if self.persist_dir:
            filepath = self.persist_dir / "episodes.json"
            filepath.write_text(json.dumps(self._episodes, indent=2, default=str))

    def _load(self) -> None:
        if self.persist_dir:
            filepath = self.persist_dir / "episodes.json"
            if filepath.exists():
                self._episodes = json.loads(filepath.read_text())


class HierarchicalMemory:
    """
    Unified interface to all four memory tiers.
    Provides cross-tier queries and automatic memory promotion.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
    ):
        self.working = WorkingMemory()
        self.procedural = ProceduralMemory(
            persist_dir=data_dir / "procedural" if data_dir else None
        )
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory(
            persist_dir=data_dir / "episodic" if data_dir else None
        )

    async def get_relevant_context(self, query: str) -> dict[str, Any]:
        """
        Query all memory tiers for relevant context.
        Returns a combined context dictionary.
        """
        # Working memory context
        working_ctx = self.working.snapshot()

        # Semantic facts
        semantic_results = await self.semantic.query(query)

        # Similar past episodes
        similar_episodes = self.episodic.get_similar_episodes(query)

        # Best procedures for the domain
        procedures = self.procedural.list_procedures()

        return {
            "working_context": working_ctx.get("context", {}),
            "active_goals": working_ctx.get("active_goals", []),
            "semantic_facts": semantic_results,
            "similar_episodes": similar_episodes,
            "available_procedures": procedures,
            "cross_domain": self._find_cross_domain_links(semantic_results),
        }

    async def store_episode(self, query: str, result: Any) -> str:
        """Store a completed goal as an episode."""
        result_dict = result if isinstance(result, dict) else {"result": str(result)}
        return self.episodic.store_episode(
            query=query,
            result=result_dict,
        )

    def _find_cross_domain_links(
        self, facts: list[dict[str, Any]]
    ) -> list[str]:
        """Identify potential cross-domain connections."""
        domains = set()
        for fact in facts:
            domains.add(fact.get("domain", "general"))

        insights = []
        if len(domains) > 1:
            insights.append(
                f"Cross-domain connection detected across: {', '.join(domains)}"
            )
        return insights

    def stats(self) -> dict[str, Any]:
        """Get statistics from all memory tiers."""
        return {
            "working": {
                "entries": len(self.working._store),
                "active_goals": len(self.working.active_goals),
            },
            "procedural": {
                "procedures": len(self.procedural._procedures),
                "tool_patterns": len(self.procedural._tool_patterns),
            },
            "semantic": self.semantic.stats(),
            "episodic": self.episodic.stats(),
        }
