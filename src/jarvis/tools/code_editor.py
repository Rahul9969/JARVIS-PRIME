"""
JARVIS-PRIME Code Editor
========================

Provides tools for JARVIS to safely read, validate, and write Python code.
Ensures that generated code is at least syntactically valid before saving.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path


class CodeEditor:
    """Safe self-editing tools for JARVIS."""

    def __init__(self, root_dir: str | Path | None = None):
        if root_dir is None:
            self.root_dir = Path(__file__).parent.parent.parent.parent
        else:
            self.root_dir = Path(root_dir)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a path relative to the root_dir."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root_dir / path
            
        # Security: Prevent escaping the root directory
        try:
            path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Access denied: {file_path} is outside the project root.")
            
        return path

    def read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        path = self._resolve_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def validate_python_syntax(self, code: str) -> tuple[bool, str]:
        """Check if Python code is syntactically valid using AST."""
        try:
            ast.parse(code)
            return True, "Syntax is valid."
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}\nCode snippet: {e.text}"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def write_file(self, file_path: str, code: str, validate_syntax: bool = True) -> dict[str, str]:
        """
        Write code to a file. 
        If validate_syntax is True, it will reject invalid Python code.
        """
        path = self._resolve_path(file_path)
        
        if validate_syntax and path.suffix == ".py":
            is_valid, msg = self.validate_python_syntax(code)
            if not is_valid:
                return {"status": "error", "message": f"Syntax validation failed: {msg}"}
                
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
            
        return {"status": "success", "message": f"Successfully wrote {len(code)} bytes to {file_path}"}
        
    def append_to_file(self, file_path: str, code: str, validate_syntax: bool = True) -> dict[str, str]:
        """Append code to an existing file."""
        path = self._resolve_path(file_path)
        
        if not path.exists():
            return {"status": "error", "message": f"File does not exist: {file_path}"}
            
        existing_code = self.read_file(file_path)
        new_code = existing_code + "\n" + code
        
        if validate_syntax and path.suffix == ".py":
            is_valid, msg = self.validate_python_syntax(new_code)
            if not is_valid:
                return {"status": "error", "message": f"Appending this code creates a SyntaxError: {msg}"}
                
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + code)
            
        return {"status": "success", "message": f"Successfully appended to {file_path}"}
