"""
JARVIS-PRIME Metacognition Engine
=================================

The Self-Improvement Agent.
Fully autonomous TDD (Test-Driven Development) loop.
JARVIS writes code, writes a pytest suite, runs it in a sandbox,
self-corrects if it fails, and only injects if 100% successful.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from jarvis.assistant.brain import OllamaBrain
from jarvis.infrastructure.version_control import VersionControl
from jarvis.tools.code_editor import CodeEditor


class SelfImprovementAgent:
    """Agent responsible for writing, testing, and safely injecting new code."""

    def __init__(self, llm_model: str = "qwen3:1.7b"):
        self.llm = OllamaBrain(model=llm_model)
        self.vcs = VersionControl()
        self.editor = CodeEditor()
        
        self.sandbox_dir = Path(__file__).parent.parent / "sandbox"
        self.sandbox_dir.mkdir(exist_ok=True)
        
        self.llm.system_prompt = (
            "You are JARVIS, an autonomous AI writing your own source code. "
            "You are an expert Python developer who writes robust, tested code. "
            "IMPORTANT RULES:\n"
            "1. ONLY output valid Python code inside ```python ... ``` blocks.\n"
            "2. Do not include extra conversational text outside the code block.\n"
            "3. Ensure your code has proper imports and follows PEP 8.\n"
        )

    def extract_python_code(self, llm_output: str) -> str:
        """Extract Python code from markdown blocks."""
        matches = re.findall(r"```python(.*?)```", llm_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        return llm_output.strip()

    async def _generate_and_test(self, instruction: str, max_retries: int = 3) -> tuple[bool, str, str]:
        """
        TDD Loop: Generate code -> Generate test -> Run test -> Self-correct.
        Returns (success_bool, final_code, message)
        """
        # 1. Generate the implementation code
        code_prompt = (
            f"INSTRUCTION: {instruction}\n"
            "Write a self-contained Python module to satisfy this instruction. "
            "Provide ONLY the implementation code."
        )
        print("  [METACOGNITION] Generating implementation code...")
        raw_code = await self.llm.chat(code_prompt)
        impl_code = self.extract_python_code(raw_code)
        
        # Reset LLM conversation so it doesn't get confused
        self.llm.conversation = []

        # 2. Write it to sandbox
        target_file = self.sandbox_dir / "generated_tool.py"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        # 3. Generate the test
        test_prompt = (
            f"Here is a Python module I wrote:\n```python\n{impl_code}\n```\n\n"
            "Write a complete `pytest` script that thoroughly tests this module. "
            "The test file will be saved in the same directory, so you can import the module as `import generated_tool`. "
            "Provide ONLY the test code."
        )
        print("  [METACOGNITION] Generating unit tests...")
        raw_test = await self.llm.chat(test_prompt)
        test_code = self.extract_python_code(raw_test)
        
        self.llm.conversation = []
        
        test_file = self.sandbox_dir / "test_generated_tool.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)

        # 4. The Self-Correction Loop
        for attempt in range(max_retries):
            print(f"  [METACOGNITION] Running pytest (Attempt {attempt + 1}/{max_retries})...")
            result = subprocess.run(
                ["pytest", str(test_file)],
                capture_output=True,
                text=True,
                cwd=str(self.sandbox_dir)
            )

            if result.returncode == 0:
                print("  [METACOGNITION] ✓ All tests passed!")
                return True, impl_code, "Tests passed."

            print(f"  [METACOGNITION] ✗ Tests failed. Initiating self-correction...")
            
            # Feed the error back to the LLM
            error_msg = result.stdout + "\n" + result.stderr
            correction_prompt = (
                f"The test failed with this output:\n```\n{error_msg}\n```\n\n"
                f"Here is the current implementation code:\n```python\n{impl_code}\n```\n\n"
                f"Here is the current test code:\n```python\n{test_code}\n```\n\n"
                "Fix the implementation code so the tests pass. Provide ONLY the corrected implementation code."
            )
            
            raw_corrected = await self.llm.chat(correction_prompt)
            impl_code = self.extract_python_code(raw_corrected)
            
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(impl_code)
                
            self.llm.conversation = []

        return False, impl_code, "Failed to pass unit tests after maximum retries."

    async def attempt_improvement(self, target_file: str, instruction: str) -> dict[str, Any]:
        """
        The autonomous improvement loop.
        """
        print(f"\n[METACOGNITION] Autonomous TDD Improvement Initiated.")
        print(f"[METACOGNITION] Target: {target_file}")
        
        # 1. Create safety checkpoint
        if not self.vcs.create_checkpoint(f"Auto-TDD: {instruction[:30]}"):
            return {"status": "error", "message": "Failed to create Git checkpoint. Aborting for safety."}
            
        # 2. Generate and test code via Sandbox TDD
        success, final_code, msg = await self._generate_and_test(instruction)
        
        if not success:
            print("[METACOGNITION] TDD loop failed. Code was rejected.")
            self.vcs.rollback()
            return {"status": "error", "message": f"Code failed testing: {msg}"}
            
        # 3. Final Syntax Validation
        is_valid, syn_msg = self.editor.validate_python_syntax(final_code)
        if not is_valid:
            print("[METACOGNITION] Syntax error detected prior to injection. Rolling back.")
            self.vcs.rollback()
            return {"status": "error", "message": f"Syntax error: {syn_msg}"}
            
        # 4. Inject into actual system
        print("[METACOGNITION] Tests passed. Injecting code into framework...")
        
        # We append it for now, assuming they are mostly standalone functions/classes
        # In the future, we could use more precise AST patching
        result = self.editor.append_to_file(target_file, final_code, validate_syntax=False)
        
        if result["status"] == "success":
            print("[METACOGNITION] Code successfully injected and active.")
            return {"status": "success", "message": "New tool written, tested, and injected successfully."}
        else:
            print("[METACOGNITION] Injection failed. Rolling back.")
            self.vcs.rollback()
            return {"status": "error", "message": result["message"]}


if __name__ == "__main__":
    # Test script
    async def run_test():
        agent = SelfImprovementAgent()
        result = await agent.attempt_improvement(
            "src/jarvis/tools/math_tools.py",
            "Write a function named calculate_factorial(n: int) -> int. It should raise ValueError for negative numbers."
        )
        print(result)
        
    asyncio.run(run_test())
