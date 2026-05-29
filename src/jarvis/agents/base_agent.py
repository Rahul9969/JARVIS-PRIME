"""
JARVIS-PRIME Base Agent
========================

Abstract base class for all domain specialist agents.
Provides common infrastructure: tool access, memory, logging,
and the standard execute/reflect interface.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.protocols import MCPGateway, MCPToolResult


@dataclass
class AgentMetrics:
    """Performance metrics for a domain agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_seconds: float = 0.0
    avg_quality_score: float = 1.0
    last_active: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / max(total, 1)

    def record_task(self, success: bool, duration: float, quality: float = 1.0) -> None:
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.total_duration_seconds += duration
        # Exponential moving average for quality
        alpha = 0.3
        self.avg_quality_score = alpha * quality + (1 - alpha) * self.avg_quality_score
        self.last_active = time.time()


class BaseAgent(ABC):
    """
    Abstract base class for JARVIS-PRIME domain specialist agents.

    Every agent must implement:
    - execute(): Process a task and return results
    - get_capabilities(): Describe what this agent can do

    Optional overrides:
    - reflect(): Self-analyze performance for SICA
    - initialize(): Setup resources on first use
    """

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.metrics = AgentMetrics()
        self._mcp: MCPGateway | None = None
        self._initialized = False
        self._log: list[dict[str, Any]] = []

    def bind_mcp(self, gateway: MCPGateway) -> None:
        """Bind the MCP gateway for tool access."""
        self._mcp = gateway

    async def initialize(self) -> None:
        """Initialize agent resources. Override for custom setup."""
        self._initialized = True

    @abstractmethod
    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        """
        Execute a domain-specific task.

        Args:
            task: Task specification with type, parameters, etc.
            tools: Available MCP tools for this execution.

        Returns:
            Result dictionary with task outputs.
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Return a list of capabilities this agent provides."""
        ...

    async def safe_execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        """
        Execute with error handling and metrics tracking.
        This is the primary entry point used by NEXUS.
        """
        if not self._initialized:
            await self.initialize()

        start = time.monotonic()
        try:
            result = await self.execute(task, tools)
            duration = time.monotonic() - start
            quality = result.get("quality_score", 1.0)
            self.metrics.record_task(True, duration, quality)

            self._log.append({
                "task": task.get("type", "unknown"),
                "success": True,
                "duration": duration,
                "timestamp": time.time(),
            })

            return result
        except Exception as e:
            duration = time.monotonic() - start
            self.metrics.record_task(False, duration, 0.0)

            self._log.append({
                "task": task.get("type", "unknown"),
                "success": False,
                "error": str(e),
                "duration": duration,
                "timestamp": time.time(),
            })

            return {
                "error": str(e),
                "agent": self.name,
                "task_type": task.get("type", "unknown"),
            }

    async def invoke_tool(self, tool_name: str, **params: Any) -> MCPToolResult:
        """Invoke an MCP tool through the gateway."""
        if not self._mcp:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="MCP gateway not bound",
            )
        return await self._mcp.invoke(tool_name, self.name, params)

    def reflect(self) -> dict[str, Any]:
        """
        Self-reflect on performance for SICA analysis.

        Returns insights about what's working, what's failing,
        and suggested improvements.
        """
        recent_failures = [
            log for log in self._log[-20:]
            if not log.get("success", True)
        ]
        return {
            "agent": self.name,
            "domain": self.domain,
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "avg_quality": self.metrics.avg_quality_score,
                "total_tasks": self.metrics.tasks_completed + self.metrics.tasks_failed,
            },
            "recent_failures": recent_failures,
            "improvement_suggestions": self._generate_suggestions(recent_failures),
        }

    def _generate_suggestions(self, failures: list[dict[str, Any]]) -> list[str]:
        """Generate improvement suggestions based on failure patterns."""
        suggestions = []
        if len(failures) > 3:
            suggestions.append(
                f"High failure rate detected ({len(failures)} recent failures). "
                "Consider revising task decomposition strategy."
            )

        error_types: dict[str, int] = {}
        for f in failures:
            err = f.get("error", "unknown")[:50]
            error_types[err] = error_types.get(err, 0) + 1

        for err, count in error_types.items():
            if count >= 2:
                suggestions.append(
                    f"Recurring error pattern: '{err}' (×{count}). "
                    "Add specific error handling or retry logic."
                )

        return suggestions
