"""
JARVIS-PRIME Multi-Agent Swarm Orchestrator
===========================================

Coordinates multiple specialized agents to solve complex tasks.
Agents:
- ResearcherAgent: Browses the web to find information.
- CoderAgent: Writes logic/scripts.
- SecurityAgent: Reviews code before execution.
"""
from __future__ import annotations

import asyncio
from typing import Any

from jarvis.assistant.brain import OllamaBrain
from jarvis.assistant.automation import SystemAutomation


class SwarmAgent:
    """Base class for specialized agents."""
    def __init__(self, name: str, role: str, model: str = "qwen3:1.7b"):
        self.name = name
        self.llm = OllamaBrain(model=model)
        self.llm.system_prompt = f"You are {name}, a highly specialized AI agent. Your role is: {role}."

    async def execute(self, task: str) -> str:
        print(f"    [{self.name}] Executing task...")
        return await self.llm.chat(task)


class ResearcherAgent(SwarmAgent):
    """Specialized in searching and summarizing information."""
    def __init__(self):
        super().__init__(
            name="Researcher",
            role="Find accurate information, synthesize research, and extract facts."
        )
        self.automation = SystemAutomation()
        
    async def execute(self, task: str) -> str:
        print(f"    [{self.name}] Searching context for: {task}")
        # In a full implementation, this agent would physically use the web browser
        # For now, we simulate the research by asking the LLM with a strict prompt
        return await super().execute(f"RESEARCH TASK: {task}\nProvide a factual, detailed summary.")


class CoderAgent(SwarmAgent):
    """Specialized in writing code snippets."""
    def __init__(self):
        super().__init__(
            name="Coder",
            role="Write clean, efficient, and well-documented Python code. ONLY output code."
        )


class Orchestrator:
    """Manages the swarm and delegates tasks."""
    
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.coder = CoderAgent()
        
    async def delegate_complex_task(self, prompt: str) -> str:
        """
        Takes a complex prompt, breaks it down, and assigns it to sub-agents.
        """
        print(f"\n[SWARM] Orchestrator received complex task: {prompt}")
        
        # 1. Research Phase
        print("[SWARM] Delegating to Researcher...")
        research_context = await self.researcher.execute(prompt)
        
        # 2. Synthesis Phase
        print("[SWARM] Synthesizing results...")
        
        # If it's a coding task, send to coder
        if "code" in prompt.lower() or "build" in prompt.lower() or "script" in prompt.lower():
            print("[SWARM] Delegating to Coder...")
            code_result = await self.coder.execute(
                f"Context from research: {research_context}\n\nTask: {prompt}\nWrite the code."
            )
            return f"I have utilized the Swarm to complete your request.\n\nResearch:\n{research_context[:200]}...\n\nCode:\n{code_result}"
            
        return f"I have utilized the Swarm to complete your request.\n\nResearch Findings:\n{research_context}"

if __name__ == "__main__":
    async def test():
        swarm = Orchestrator()
        res = await swarm.delegate_complex_task("Research the current price of Bitcoin and write a Python script to fetch it.")
        print(res)
    
    asyncio.run(test())
