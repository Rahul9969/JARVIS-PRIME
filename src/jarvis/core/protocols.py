"""
JARVIS-PRIME Protocol Implementations
======================================

MCP (Model Context Protocol) — Agent-to-tool connectivity.
A2A (Agent-to-Agent Protocol) — Inter-agent coordination.

Phase 0: Lightweight in-process implementations.
Phase 1+: Full JSON-RPC MCP servers and A2A discovery.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class MCPTool:
    """Representation of an MCP-compatible tool."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    domain: str = "general"
    requires_approval: bool = False


@dataclass
class MCPToolResult:
    """Result from an MCP tool invocation."""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MCPGateway:
    """
    MCP Gateway — The control plane for tool access.

    Provides:
    - Tool registration and discovery
    - RBAC (Role-Based Access Control) for tool access
    - Invocation logging for observability
    - Rate limiting (Phase 1+)

    Phase 0: In-process tool registry with direct function calls.
    Phase 1+: JSON-RPC client connecting to external MCP servers.
    """

    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}
        self._invocation_log: list[MCPToolResult] = []
        self._permissions: dict[str, set[str]] = {}  # agent_name -> {tool_names}

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool in the gateway."""
        self._tools[tool.name] = tool

    def grant_access(self, agent_name: str, tool_names: list[str]) -> None:
        """Grant an agent access to specific tools."""
        if agent_name not in self._permissions:
            self._permissions[agent_name] = set()
        self._permissions[agent_name].update(tool_names)

    def grant_domain_access(self, agent_name: str, domain: str) -> None:
        """Grant an agent access to all tools in a domain."""
        domain_tools = [
            t.name for t in self._tools.values() if t.domain == domain
        ]
        self.grant_access(agent_name, domain_tools)

    async def invoke(
        self,
        tool_name: str,
        agent_name: str,
        parameters: dict[str, Any],
    ) -> MCPToolResult:
        """
        Invoke a tool through the gateway.

        Checks permissions, executes the tool, and logs the result.
        """
        # Permission check
        if agent_name in self._permissions:
            if tool_name not in self._permissions[agent_name]:
                result = MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Agent '{agent_name}' lacks permission for tool '{tool_name}'",
                )
                self._invocation_log.append(result)
                return result

        # Tool existence check
        tool = self._tools.get(tool_name)
        if not tool:
            result = MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )
            self._invocation_log.append(result)
            return result

        # Execute
        start = time.monotonic()
        try:
            output = await tool.handler(**parameters)
            duration = (time.monotonic() - start) * 1000
            result = MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=output,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            result = MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration,
            )

        self._invocation_log.append(result)
        return result

    async def get_tools(self, tool_names: list[str]) -> list[MCPTool]:
        """Get tool objects by name."""
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self, domain: str | None = None) -> list[dict[str, str]]:
        """List available tools."""
        tools = self._tools.values()
        if domain:
            tools = [t for t in tools if t.domain == domain]
        return [
            {"name": t.name, "description": t.description, "domain": t.domain}
            for t in tools
        ]

    def get_invocation_stats(self) -> dict[str, Any]:
        """Get tool invocation statistics."""
        total = len(self._invocation_log)
        successes = sum(1 for r in self._invocation_log if r.success)
        avg_duration = (
            sum(r.duration_ms for r in self._invocation_log) / max(total, 1)
        )
        return {
            "total_invocations": total,
            "success_rate": successes / max(total, 1),
            "avg_duration_ms": avg_duration,
        }


@dataclass
class AgentCard:
    """A2A Agent Card — describes an agent's capabilities for discovery."""
    name: str
    domain: str
    capabilities: list[str]
    supported_tasks: list[str]
    performance_score: float = 1.0
    status: str = "available"


@dataclass
class A2ATask:
    """A task delegated via A2A protocol."""
    task_id: str
    source_agent: str
    target_agent: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Any = None


class A2ACoordinator:
    """
    A2A (Agent-to-Agent) Coordinator.

    Handles:
    - Agent discovery (registration and capability matching)
    - Task delegation between agents
    - Result aggregation
    - Load balancing (Phase 1+)

    Phase 0: In-process agent registry with direct delegation.
    Phase 1+: gRPC-based inter-process agent communication.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}
        self._task_queue: list[A2ATask] = []
        self._completed_tasks: list[A2ATask] = []

    def register_agent(self, card: AgentCard) -> None:
        """Register an agent with the coordinator."""
        self._agents[card.name] = card

    def discover_agents(
        self,
        domain: str | None = None,
        capability: str | None = None,
    ) -> list[AgentCard]:
        """Discover available agents by domain or capability."""
        agents = list(self._agents.values())

        if domain:
            agents = [a for a in agents if a.domain == domain]
        if capability:
            agents = [a for a in agents if capability in a.capabilities]

        return sorted(agents, key=lambda a: a.performance_score, reverse=True)

    async def delegate(self, target_agent: str, task: dict[str, Any]) -> str:
        """Delegate a task to a specific agent."""
        a2a_task = A2ATask(
            task_id=f"A2A-{len(self._task_queue):06d}",
            source_agent="nexus",
            target_agent=target_agent,
            description=task.get("description", ""),
            parameters=task,
        )
        self._task_queue.append(a2a_task)
        return a2a_task.task_id

    def complete_task(self, task_id: str, result: Any) -> None:
        """Mark a task as completed with result."""
        for task in self._task_queue:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                self._completed_tasks.append(task)
                self._task_queue.remove(task)
                break

    def get_agent_card(self, name: str) -> AgentCard | None:
        """Get an agent's capability card."""
        return self._agents.get(name)

    def stats(self) -> dict[str, Any]:
        """Get coordination statistics."""
        return {
            "registered_agents": len(self._agents),
            "pending_tasks": len(self._task_queue),
            "completed_tasks": len(self._completed_tasks),
        }
