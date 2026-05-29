"""
JARVIS-PRIME Version Control System
===================================

Provides Git-based safety checkpoints for the self-improvement loop.
Allows JARVIS to safely rollback code changes if new code crashes.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


class VersionControl:
    """Manages local git commits and rollbacks for safety."""

    def __init__(self, repo_path: str | Path | None = None):
        if repo_path is None:
            # Default to the src folder containing jarvis
            self.repo_path = Path(__file__).parent.parent.parent.parent
        else:
            self.repo_path = Path(repo_path)
            
        self._ensure_git_initialized()

    def _run_git(self, *args) -> tuple[int, str, str]:
        """Run a git command in the repo directory."""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError:
            return 1, "", "Git is not installed or not in PATH."

    def _ensure_git_initialized(self):
        """Initialize git repo if it doesn't exist."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            print("[VCS] Initializing Git repository for safety rollbacks...")
            self._run_git("init")
            
            # Create default .gitignore
            gitignore = self.repo_path / ".gitignore"
            if not gitignore.exists():
                with open(gitignore, "w", encoding="utf-8") as f:
                    f.write("__pycache__/\n*.pyc\n.env\n.pytest_cache/\n")
            
            self.create_checkpoint("Initial JARVIS-PRIME commit")

    def create_checkpoint(self, message: str = "JARVIS Self-Improvement Checkpoint") -> bool:
        """Commit the current state of the codebase."""
        print(f"[VCS] Creating safety checkpoint: {message}")
        # Add all files
        self._run_git("add", ".")
        
        # Check if there are changes to commit
        code, out, err = self._run_git("status", "--porcelain")
        if not out:
            print("[VCS] No changes to commit.")
            return True
            
        code, out, err = self._run_git("commit", "-m", message)
        if code == 0:
            return True
        else:
            print(f"[VCS] Failed to create checkpoint: {err}")
            return False

    def get_current_commit(self) -> str:
        """Get the hash of the current HEAD commit."""
        code, out, err = self._run_git("rev-parse", "HEAD")
        return out if code == 0 else ""

    def rollback(self, commit_hash: str | None = None) -> bool:
        """
        Rollback the codebase. 
        If commit_hash is provided, hard resets to that commit.
        If not, resets to HEAD (undoing any uncommitted changes).
        """
        if commit_hash:
            print(f"[VCS] Rolling back to commit {commit_hash[:7]}...")
            code, out, err = self._run_git("reset", "--hard", commit_hash)
        else:
            print("[VCS] Undoing uncommitted changes...")
            code, out, err = self._run_git("reset", "--hard", "HEAD")
            self._run_git("clean", "-fd") # Remove untracked files
            
        if code == 0:
            return True
        else:
            print(f"[VCS] Rollback failed: {err}")
            return False
            
    def has_uncommitted_changes(self) -> bool:
        """Check if there are modified/untracked files."""
        code, out, err = self._run_git("status", "--porcelain")
        return bool(out)
