"""
JARVIS-PRIME Metacognition Engine
=================================

The Self-Improvement Agent.
Allows JARVIS to analyze his own code, write new tools, test them,
and inject them into the framework dynamically.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from jarvis.assistant.brain import OllamaBrain
from jarvis.infrastructure.version_control import VersionControl
from jarvis.tools.code_editor import CodeEditor


class SelfImprovementAgent:
    """Agent responsible for writing and safely injecting new code."""

    def __init__(self, llm_model: str = "qwen3:1.7b"):
        self.llm = OllamaBrain(model=llm_model)
        self.vcs = VersionControl()
        self.editor = CodeEditor()
        
        # We need a system prompt specifically tailored for coding
        self.llm.system_prompt = (
            "You are JARVIS, an autonomous AI writing your own source code. "
            "You are tasked with generating a Python function/script to add a new capability. "
            "IMPORTANT RULES:\n"
            "1. ONLY output valid Python code inside ```python ... ``` blocks.\n"
            "2. Do not include extra conversational text outside the code block.\n"
            "3. Ensure your code has proper imports and is syntactically perfect.\n"
            "4. Follow standard Python conventions (PEP 8).\n"
        )

    def extract_python_code(self, llm_output: str) -> str:
        """Extract Python code from markdown blocks."""
        matches = re.findall(r"```python(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
            
        # Fallback if the LLM didn't use the markdown tags properly
        return llm_output.strip()

    async def attempt_improvement(self, target_file: str, instruction: str) -> dict[str, Any]:
        """
        The core self-improvement loop:
        1. Create safety checkpoint.
        2. Prompt LLM to write code.
        3. Validate syntax.
        4. Apply code.
        5. Return status.
        """
        print(f"\n[METACOGNITION] Initiating self-improvement on {target_file}")
        print(f"[METACOGNITION] Instruction: {instruction}")
        
        # 1. Create a safety checkpoint
        if not self.vcs.create_checkpoint(f"Pre-improvement: {instruction[:30]}"):
            return {"status": "error", "message": "Failed to create safety checkpoint. Aborting."}
            
        # 2. Read existing code context (if any)
        context = ""
        try:
            existing_code = self.editor.read_file(target_file)
            context = f"The existing file looks like this:\n```python\n{existing_code}\n```\n\n"
        except FileNotFoundError:
            context = "The file is currently empty or does not exist. You are creating it from scratch.\n\n"

        # 3. Ask LLM to generate code
        prompt = (
            f"{context}"
            f"INSTRUCTION: {instruction}\n"
            "Write the complete Python code to satisfy this instruction. "
            "Provide ONLY the code."
        )
        
        print("[METACOGNITION] Brain is generating code...")
        raw_output = await self.llm.chat(prompt)
        new_code = self.extract_python_code(raw_output)
        
        if not new_code:
            return {"status": "error", "message": "The brain failed to generate valid Python code."}
            
        # 4. Validate syntax
        is_valid, msg = self.editor.validate_python_syntax(new_code)
        if not is_valid:
            print("[METACOGNITION] Syntax error detected. Rolling back.")
            self.vcs.rollback()
            return {"status": "error", "message": f"Generated code had syntax errors: {msg}"}
            
        # 5. Apply the code
        print("[METACOGNITION] Syntax valid. Applying code...")
        result = self.editor.write_file(target_file, new_code, validate_syntax=False)
        
        if result["status"] == "success":
            print("[METACOGNITION] Code successfully injected.")
            return {"status": "success", "message": "New capability successfully added.", "code": new_code}
        else:
            print("[METACOGNITION] File write failed. Rolling back.")
            self.vcs.rollback()
            return {"status": "error", "message": result["message"]}


if __name__ == "__main__":
    # Test script
    async def run_test():
        agent = SelfImprovementAgent()
        result = await agent.attempt_improvement(
            "src/jarvis/tools/joke_generator.py",
            "Write a Python function named get_joke() that returns a programming joke as a string. Include necessary imports."
        )
        print(result)
        
    asyncio.run(run_test())
