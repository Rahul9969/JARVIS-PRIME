"""
NEXUS: Neural Executive for Unified eXpert Systems
=====================================================

The Meta-Orchestrator that coordinates all JARVIS-PRIME subsystems.

Architecture:
    LangGraph-inspired state machine with:
    - Goal decomposition (Tree-of-Thoughts + domain awareness)
    - Agent routing (A2A protocol with performance-based selection)
    - Parallel task execution
    - Cross-domain result synthesis
    - SICA-driven self-improvement loop

Flow:
    PERCEIVE → DECOMPOSE → ROUTE → EXECUTE → SYNTHESIZE → VALIDATE
                    ↑                                          ↓
                    ╰─────── REFLECT (if iteration needed) ←───╯

Phase 0: Simplified orchestration with async execution.
Phase 1+: Full LangGraph state machine with checkpointing.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from jarvis.agents.base_agent import BaseAgent
from jarvis.core.memory import HierarchicalMemory
from jarvis.core.protocols import MCPGateway, A2ACoordinator, AgentCard
from jarvis.core.self_improvement import SICAEngine


class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    BACKGROUND = auto()


class TaskStatus(Enum):
    PENDING = "pending"
    PERCEIVING = "perceiving"
    DECOMPOSING = "decomposing"
    ROUTING = "routing"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GoalState:
    """State object for goal processing pipeline."""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_query: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    decomposed_tasks: list[dict[str, Any]] = field(default_factory=list)
    agent_assignments: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    synthesized_output: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    iteration: int = 0
    max_iterations: int = 5
    cross_domain_insights: list[str] = field(default_factory=list)
    error_log: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    duration_seconds: float = 0.0


@dataclass
class RegisteredAgent:
    """An agent registered with NEXUS."""
    agent: BaseAgent
    card: AgentCard
    priority_tasks: list[str] = field(default_factory=list)


class Nexus:
    """
    NEXUS Meta-Orchestrator.

    Coordinates the entire JARVIS-PRIME agent mesh through a
    structured pipeline that decomposes high-level goals into
    domain-specific tasks, executes them in parallel where possible,
    synthesizes cross-domain results, and drives self-improvement.
    """

    def __init__(
        self,
        memory: HierarchicalMemory,
        sica: SICAEngine,
        mcp: MCPGateway,
        a2a: A2ACoordinator,
        max_iterations: int = 5,
        cognitive_core: Any | None = None,
    ):
        self.memory = memory
        self.sica = sica
        self.mcp = mcp
        self.a2a = a2a
        self.max_iterations = max_iterations
        self.llm = cognitive_core  # CognitiveCore instance (optional)

        self._agents: dict[str, RegisteredAgent] = {}
        self._goal_history: list[GoalState] = []

    def bind_llm(self, cognitive_core: Any) -> None:
        """Bind a CognitiveCore LLM router to enable intelligent decomposition."""
        self.llm = cognitive_core

    # ──────────────────────────────────────────────────────
    # Agent Registration
    # ──────────────────────────────────────────────────────

    def register_agent(
        self,
        agent: BaseAgent,
        priority_tasks: list[str] | None = None,
    ) -> None:
        """
        Register a domain specialist agent with NEXUS.

        Creates an A2A agent card for discovery and routing.
        """
        card = AgentCard(
            name=agent.name,
            domain=agent.domain,
            capabilities=agent.get_capabilities(),
            supported_tasks=priority_tasks or [],
        )
        self.a2a.register_agent(card)
        self._agents[agent.name] = RegisteredAgent(
            agent=agent,
            card=card,
            priority_tasks=priority_tasks or [],
        )

        # Grant MCP access based on domain
        self.mcp.grant_domain_access(agent.name, agent.domain)

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents with their capabilities."""
        return [
            {
                "name": reg.agent.name,
                "domain": reg.agent.domain,
                "capabilities": reg.card.capabilities,
                "metrics": {
                    "success_rate": reg.agent.metrics.success_rate,
                    "quality": reg.agent.metrics.avg_quality_score,
                    "tasks_completed": reg.agent.metrics.tasks_completed,
                },
            }
            for reg in self._agents.values()
        ]

    # ──────────────────────────────────────────────────────
    # Goal Processing Pipeline
    # ──────────────────────────────────────────────────────

    async def process_goal(
        self,
        query: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> dict[str, Any]:
        """
        Main entry point: process a high-level goal through the pipeline.

        Pipeline: PERCEIVE → DECOMPOSE → ROUTE → EXECUTE →
                  SYNTHESIZE → VALIDATE → (REFLECT if needed)
        """
        state = GoalState(
            original_query=query,
            priority=priority,
            max_iterations=self.max_iterations,
        )

        try:
            # Phase 1: PERCEIVE — Gather context from memory
            state = await self._perceive(state)

            # Iterative loop with self-improvement
            while state.iteration < state.max_iterations:
                # Phase 2: DECOMPOSE — Break goal into subtasks
                state = await self._decompose(state)

                # Phase 3: ROUTE — Assign tasks to agents
                state = await self._route(state)

                # Phase 4: EXECUTE — Run tasks in parallel
                state = await self._execute(state)

                # Phase 5: SYNTHESIZE — Combine cross-domain results
                state = await self._synthesize(state)

                # Phase 6: VALIDATE — Check result quality
                state = await self._validate(state)

                # Phase 7: REFLECT — Self-improve if needed
                if state.status == TaskStatus.COMPLETED:
                    break
                elif state.status == TaskStatus.REFLECTING:
                    state = await self._reflect(state)
                    state.iteration += 1
                else:
                    break

            # Record completion
            state.duration_seconds = time.time() - state.start_time
            state.status = TaskStatus.COMPLETED
            self._goal_history.append(state)

            # Store in episodic memory
            await self.memory.store_episode(query, state.synthesized_output)

            return {
                "goal_id": state.goal_id,
                "status": state.status.value,
                "iterations": state.iteration,
                "duration_seconds": state.duration_seconds,
                "results": state.synthesized_output,
                "cross_domain_insights": state.cross_domain_insights,
                "agents_used": list(state.agent_assignments.values()),
                "errors": state.error_log,
            }

        except Exception as e:
            state.status = TaskStatus.FAILED
            state.error_log.append(f"NEXUS pipeline error: {str(e)}")
            state.duration_seconds = time.time() - state.start_time
            self._goal_history.append(state)
            return {
                "goal_id": state.goal_id,
                "status": "failed",
                "error": str(e),
                "error_log": state.error_log,
            }

    # ──────────────────────────────────────────────────────
    # Pipeline Stages
    # ──────────────────────────────────────────────────────

    async def _perceive(self, state: GoalState) -> GoalState:
        """Stage 1: Gather context from all memory tiers."""
        state.status = TaskStatus.PERCEIVING
        state.context = await self.memory.get_relevant_context(state.original_query)

        # Set working memory context
        self.memory.working.set_context("current_goal", state.original_query)
        self.memory.working.set_context("goal_id", state.goal_id)

        return state

    async def _decompose(self, state: GoalState) -> GoalState:
        """
        Stage 2: Decompose goal into domain-specific subtasks.

        Phase 2: LLM-powered decomposition with keyword fallback.
        """
        state.status = TaskStatus.DECOMPOSING
        query = state.original_query.lower()

        # Try LLM-powered decomposition first
        if self.llm is not None:
            try:
                available_domains = list(set(
                    reg.agent.domain for reg in self._agents.values()
                ))
                llm_result = await self.llm.generate_structured(
                    f"Decompose this goal into domain-specific subtasks.\n"
                    f"GOAL: {state.original_query}\n"
                    f"AVAILABLE DOMAINS: {available_domains}\n"
                    f"CONTEXT: {json.dumps(state.context.get('semantic_facts', [])[:3], default=str)}\n\n"
                    f'Return JSON: {{"tasks": [{{"domain": "...", "type": "...", "description": "..."}}]}}',
                    temperature=0.3,
                )
                if "tasks" in llm_result:
                    tasks = []
                    for i, t in enumerate(llm_result["tasks"][:5]):
                        domain = t.get("domain", "scientific")
                        if domain not in available_domains:
                            domain = "scientific"
                        tasks.append({
                            "id": f"TASK-{state.goal_id}-{domain}-{i}",
                            "domain": domain,
                            "description": t.get("description", state.original_query),
                            "type": t.get("type", self._infer_task_type(query, domain)),
                            "relevance_score": 5 - i,
                            "parameters": self._extract_parameters(query, domain),
                        })
                    if tasks:
                        state.decomposed_tasks = tasks
                        return state
            except Exception:
                pass  # Fall through to keyword decomposition

        # Keyword-based fallback decomposition
        domain_keywords = {
            "exotic_physics": [
                "gravity", "anti-gravity", "warp", "alcubierre", "casimir",
                "lense-thirring", "frame-dragging", "propulsion", "exotic",
                "thruster", "woodward", "mach effect", "gravitomagnetic",
                "spacetime", "metric", "general relativity",
            ],
            "scientific": [
                "research", "hypothesis", "experiment", "discovery",
                "literature", "survey", "gap", "synthesis", "paper",
                "publish", "analyze", "simulate",
            ],
            "cybersecurity": [
                "cve", "vulnerability", "exploit", "security", "threat",
                "attack", "malware", "hack", "pentest", "firewall",
                "mitre", "apt", "zero-day", "encryption", "cyber",
            ],
            "biotech": [
                "protein", "gene", "crispr", "drug", "dna", "rna",
                "cell", "enzyme", "antibody", "clinical", "trial",
                "mutation", "sequencing", "biotech", "longevity",
            ],
            "quantum": [
                "quantum", "qubit", "entanglement", "superposition",
                "circuit", "qiskit", "pennylane", "vqe", "grover",
            ],
            "creative": [
                "write", "draft", "paper", "report", "visualize",
                "chart", "graph", "presentation", "format", "summary",
            ],
            "energy": [
                "fusion", "tokamak", "plasma", "solar", "energy",
                "confinement", "lawson", "iter", "sparc", "reactor",
                "fission", "nuclear", "battery", "grid", "renewable",
            ],
            "legal_financial": [
                "patent", "legal", "contract", "compliance", "gdpr",
                "financial", "investment", "stock", "option", "risk",
                "dcf", "valuation", "black-scholes", "portfolio", "var",
                "trading", "hedge", "regulation",
            ],
            "robotics": [
                "robot", "kinematics", "actuator", "servo", "motor",
                "path planning", "pid", "manipulator", "arm", "ros",
                "trajectory", "control", "navigation", "sensor",
            ],
        }

        tasks = []
        matched_domains = set()

        for domain, keywords in domain_keywords.items():
            relevance = sum(1 for kw in keywords if kw in query)
            if relevance > 0:
                matched_domains.add(domain)
                tasks.append({
                    "id": f"TASK-{state.goal_id}-{domain}",
                    "domain": domain,
                    "description": state.original_query,
                    "type": self._infer_task_type(query, domain),
                    "relevance_score": relevance,
                    "parameters": self._extract_parameters(query, domain),
                })

        if not tasks:
            tasks.append({
                "id": f"TASK-{state.goal_id}-scientific",
                "domain": "scientific",
                "description": state.original_query,
                "type": "research_summary",
                "relevance_score": 1,
                "parameters": {"topic": state.original_query},
            })

        tasks.sort(key=lambda t: t["relevance_score"], reverse=True)
        state.decomposed_tasks = tasks
        return state

    async def _route(self, state: GoalState) -> GoalState:
        """Stage 3: Route tasks to the best available agents."""
        state.status = TaskStatus.ROUTING

        for task in state.decomposed_tasks:
            domain = task["domain"]
            # Find best agent for domain
            agent = self._select_agent(domain)
            if agent:
                state.agent_assignments[task["id"]] = agent.agent.name
                await self.a2a.delegate(agent.agent.name, task)
            else:
                state.error_log.append(
                    f"No agent available for domain: {domain}"
                )

        return state

    async def _execute(self, state: GoalState) -> GoalState:
        """Stage 4: Execute all routed tasks in parallel."""
        state.status = TaskStatus.EXECUTING

        # Gather all execution coroutines
        coros = []
        task_ids = []
        for task in state.decomposed_tasks:
            agent_name = state.agent_assignments.get(task["id"])
            if agent_name and agent_name in self._agents:
                reg = self._agents[agent_name]
                tools = await self.mcp.get_tools(
                    [t["name"] for t in self.mcp.list_tools(domain=reg.agent.domain)]
                )
                coros.append(reg.agent.safe_execute(task, tools))
                task_ids.append(task["id"])

        # Execute in parallel
        if coros:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for task_id, result in zip(task_ids, results):
                if isinstance(result, Exception):
                    state.error_log.append(f"{task_id}: {str(result)}")
                    state.results[task_id] = {"error": str(result)}
                else:
                    state.results[task_id] = result

        return state

    async def _synthesize(self, state: GoalState) -> GoalState:
        """
        Stage 5: Cross-domain synthesis of results.

        Phase 2: LLM-powered coherent synthesis with fallback.
        """
        state.status = TaskStatus.SYNTHESIZING

        combined = {
            "query": state.original_query,
            "goal_id": state.goal_id,
            "domain_results": {},
            "cross_domain_insights": [],
            "llm_synthesis": None,
        }

        domains_involved = set()
        for task_id, result in state.results.items():
            if isinstance(result, dict) and "error" not in result:
                task = next(
                    (t for t in state.decomposed_tasks if t["id"] == task_id),
                    None,
                )
                if task:
                    domain = task["domain"]
                    domains_involved.add(domain)
                    combined["domain_results"][domain] = result

        # LLM-powered synthesis
        if self.llm is not None and len(domains_involved) >= 1:
            try:
                # Truncate results to fit in context
                results_summary = json.dumps(
                    combined["domain_results"], indent=2, default=str
                )[:4000]
                synthesis = await self.llm.generate(
                    f"Synthesize these multi-domain research results into a coherent analysis.\n\n"
                    f"ORIGINAL GOAL: {state.original_query}\n\n"
                    f"DOMAIN RESULTS:\n{results_summary}\n\n"
                    f"Provide: 1) Key findings 2) Cross-domain insights 3) Recommendations",
                    temperature=0.5,
                    max_tokens=1024,
                )
                combined["llm_synthesis"] = synthesis
            except Exception:
                pass

        if len(domains_involved) > 1:
            combined["cross_domain_insights"].append(
                f"Analysis spans {len(domains_involved)} domains: "
                f"{', '.join(domains_involved)}. "
                "Cross-domain synthesis opportunities identified."
            )
            state.cross_domain_insights = combined["cross_domain_insights"]

        state.synthesized_output = combined
        return state

    async def _validate(self, state: GoalState) -> GoalState:
        """Stage 6: Validate output quality."""
        state.status = TaskStatus.VALIDATING

        # Check for completeness
        has_results = bool(state.results)
        has_errors = bool(state.error_log)
        all_tasks_done = len(state.results) >= len(state.decomposed_tasks)

        if has_results and all_tasks_done and not has_errors:
            state.status = TaskStatus.COMPLETED
        elif has_errors and state.iteration < state.max_iterations - 1:
            state.status = TaskStatus.REFLECTING
        else:
            state.status = TaskStatus.COMPLETED

        return state

    async def _reflect(self, state: GoalState) -> GoalState:
        """Stage 7: SICA self-reflection and improvement."""
        state.status = TaskStatus.REFLECTING

        # Run SICA improvement cycle
        cycle_result = await self.sica.run_improvement_cycle(
            performance_data={
                "error_count": len(state.error_log),
                "tasks_completed": len(state.results),
                "tasks_total": len(state.decomposed_tasks),
            },
            error_log=state.error_log,
        )

        # Clear errors for retry
        state.error_log.clear()
        state.results.clear()

        return state

    # ──────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────

    def _select_agent(self, domain: str) -> RegisteredAgent | None:
        """Select the best agent for a given domain."""
        candidates = [
            reg for reg in self._agents.values()
            if reg.agent.domain == domain
        ]
        if not candidates:
            # Fallback: try any agent
            candidates = list(self._agents.values())[:1]

        if not candidates:
            return None

        # Select by performance score
        return max(
            candidates,
            key=lambda r: r.agent.metrics.avg_quality_score,
        )

    def _infer_task_type(self, query: str, domain: str) -> str:
        """Infer the specific task type from query and domain."""
        if domain == "exotic_physics":
            if any(kw in query for kw in ["lense", "thirring", "frame-dragging", "gravitomagnetic"]):
                return "lense_thirring"
            elif any(kw in query for kw in ["alcubierre", "warp"]):
                return "alcubierre_energy"
            elif any(kw in query for kw in ["casimir"]):
                return "casimir_analysis"
            elif any(kw in query for kw in ["propulsion", "thruster", "survey"]):
                return "propulsion_survey"
            else:
                return "full_analysis"
        elif domain == "scientific":
            if any(kw in query for kw in ["survey", "literature", "review"]):
                return "literature_survey"
            elif any(kw in query for kw in ["hypothesis", "hypotheses"]):
                return "hypothesis_generation"
            elif any(kw in query for kw in ["gap", "missing"]):
                return "gap_analysis"
            elif any(kw in query for kw in ["synthesis", "cross-domain", "intersection"]):
                return "cross_domain_synthesis"
            else:
                return "research_summary"
        return "research_summary"

    def _extract_parameters(self, query: str, domain: str) -> dict[str, Any]:
        """Extract task parameters from query text."""
        params: dict[str, Any] = {"topic": query}

        # Simple parameter extraction
        # Phase 1+: LLM-powered parameter extraction
        if domain == "exotic_physics":
            params["type"] = self._infer_task_type(query, domain)

        return params

    # ──────────────────────────────────────────────────────
    # Status & Statistics
    # ──────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get NEXUS orchestrator statistics."""
        total_goals = len(self._goal_history)
        completed = sum(
            1 for g in self._goal_history
            if g.status == TaskStatus.COMPLETED
        )
        avg_duration = (
            sum(g.duration_seconds for g in self._goal_history)
            / max(total_goals, 1)
        )

        return {
            "total_goals_processed": total_goals,
            "completed": completed,
            "failed": total_goals - completed,
            "success_rate": completed / max(total_goals, 1),
            "avg_duration_seconds": avg_duration,
            "registered_agents": len(self._agents),
            "agent_details": self.list_agents(),
            "memory_stats": self.memory.stats(),
            "sica_stats": self.sica.stats(),
            "mcp_stats": self.mcp.get_invocation_stats(),
            "a2a_stats": self.a2a.stats(),
        }
