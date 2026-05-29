"""
JARVIS-PRIME Global Configuration System
=========================================

Centralized configuration using Pydantic Settings with environment
variable support and hierarchical defaults.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    MOCK = "mock"  # For testing without API keys


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    provider: LLMProvider = LLMProvider.MOCK
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    model_config = {"env_prefix": "JARVIS_LLM_"}


class MemoryConfig(BaseSettings):
    """Memory system configuration."""
    # Episodic memory
    episodic_db_url: str = "sqlite:///jarvis_episodic.db"

    # Semantic memory (Knowledge Graph)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    weaviate_url: str = "http://localhost:8080"

    # Working memory
    redis_url: str = "redis://localhost:6379"

    # Fallback: use in-memory stores when services unavailable
    use_in_memory_fallback: bool = True

    model_config = {"env_prefix": "JARVIS_MEMORY_"}


class SimulationConfig(BaseSettings):
    """Physics simulation configuration."""
    output_dir: Path = Path("simulation_results")
    grid_resolution: int = 200
    export_format: str = "hdf5"  # hdf5 or json
    max_compute_seconds: int = 300

    model_config = {"env_prefix": "JARVIS_SIM_"}


class JarvisConfig(BaseSettings):
    """Master configuration for JARVIS-PRIME."""
    # System
    project_name: str = "JARVIS-PRIME"
    version: str = "0.1.0"
    codename: str = "PRIME"
    log_level: LogLevel = LogLevel.INFO
    data_dir: Path = Path("jarvis_data")
    research_journal_dir: Path = Path("research_journal")

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # Agent mesh
    max_concurrent_agents: int = 8
    agent_timeout_seconds: int = 300
    max_goal_iterations: int = 10

    # Self-improvement
    sica_enabled: bool = True
    sica_auto_promote: bool = False  # Require human approval
    sica_significance_threshold: float = 0.05
    sica_min_improvement_pct: float = 2.0

    model_config = {"env_prefix": "JARVIS_"}

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.research_journal_dir.mkdir(parents=True, exist_ok=True)
        self.simulation.output_dir.mkdir(parents=True, exist_ok=True)
        (self.research_journal_dir / "hypotheses").mkdir(exist_ok=True)
        (self.research_journal_dir / "experiments").mkdir(exist_ok=True)
        (self.research_journal_dir / "results").mkdir(exist_ok=True)


# Global singleton
_config: JarvisConfig | None = None


def get_config() -> JarvisConfig:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = JarvisConfig()
        _config.ensure_dirs()
    return _config


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _config
    _config = None
