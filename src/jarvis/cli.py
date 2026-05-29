"""
JARVIS-PRIME Interactive CLI
==============================

Command-line interface for interacting with the JARVIS-PRIME system.
Provides:
- Natural language goal processing
- Direct agent commands
- System status and diagnostics
- Research journal access
- SICA self-improvement monitoring

Usage:
    python -m jarvis.cli              # Interactive mode
    python -m jarvis.cli --query "..." # Single query mode
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from jarvis import __version__, __codename__
from jarvis.infrastructure.config import get_config, JarvisConfig
from jarvis.core.cognitive_core import CognitiveCore
from jarvis.core.memory import HierarchicalMemory
from jarvis.core.protocols import MCPGateway, A2ACoordinator
from jarvis.core.self_improvement import SICAEngine
from jarvis.core.nexus import Nexus, TaskPriority
from jarvis.agents.exotic_physics import ExoticPhysicsAgent
from jarvis.agents.scientific import ScientificDiscoveryAgent
from jarvis.agents.cybersecurity import CybersecurityAgent
from jarvis.agents.biotech import BiotechAgent
from jarvis.agents.quantum import QuantumAgent
from jarvis.agents.creative import CreativeAgent
from jarvis.agents.energy import EnergyAgent
from jarvis.agents.legal_financial import LegalFinancialAgent
from jarvis.agents.robotics import RoboticsAgent


# ══════════════════════════════════════════════════════════
# ANSI Color Constants
# ══════════════════════════════════════════════════════════

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# ══════════════════════════════════════════════════════════
# System Bootstrap
# ══════════════════════════════════════════════════════════

class JarvisSystem:
    """
    Complete JARVIS-PRIME system initialization and management.
    """

    def __init__(self, config: JarvisConfig | None = None):
        self.config = config or get_config()
        self.config.ensure_dirs()

        # Initialize core subsystems
        self.memory = HierarchicalMemory(data_dir=self.config.data_dir)
        self.mcp = MCPGateway()
        self.a2a = A2ACoordinator()
        self.sica = SICAEngine(
            archive_dir=self.config.data_dir / "sica_archive",
            auto_promote=self.config.sica_auto_promote,
        )

        # Initialize NEXUS
        self.nexus = Nexus(
            memory=self.memory,
            sica=self.sica,
            mcp=self.mcp,
            a2a=self.a2a,
            max_iterations=self.config.max_goal_iterations,
        )

        # Initialize LLM cognitive core
        self.cognitive_core = CognitiveCore()
        self.nexus.bind_llm(self.cognitive_core)

        # Initialize and register agents
        self._init_agents()

        # Seed knowledge base
        self._seed_knowledge()

    def _init_agents(self) -> None:
        """Initialize and register all domain agents."""
        # Exotic Physics Agent
        physics_agent = ExoticPhysicsAgent(
            journal_dir=self.config.research_journal_dir
        )
        self.nexus.register_agent(
            physics_agent,
            priority_tasks=[
                "lense_thirring", "alcubierre_energy", "casimir_analysis",
                "propulsion_survey", "gravity_probe_b_validation", "full_analysis",
            ],
        )

        # Scientific Discovery Agent
        sci_agent = ScientificDiscoveryAgent()
        self.nexus.register_agent(
            sci_agent,
            priority_tasks=[
                "literature_survey", "hypothesis_generation",
                "gap_analysis", "cross_domain_synthesis",
            ],
        )

        # Cybersecurity Agent
        self.nexus.register_agent(
            CybersecurityAgent(),
            priority_tasks=["cve_lookup", "threat_model", "mitre_mapping", "security_audit"],
        )

        # Biotech Agent
        self.nexus.register_agent(
            BiotechAgent(),
            priority_tasks=["protein_analysis", "crispr_design", "sequence_stats"],
        )

        # Quantum Computing Agent
        self.nexus.register_agent(
            QuantumAgent(),
            priority_tasks=["bell_state", "ghz_state", "quantum_teleportation"],
        )

        # Creative Studio Agent
        self.nexus.register_agent(
            CreativeAgent(),
            priority_tasks=["draft_paper", "write_report", "create_presentation"],
        )

        # Energy/Fusion Agent
        self.nexus.register_agent(
            EnergyAgent(),
            priority_tasks=["lawson", "tokamak_confinement", "fusion_power", "solar"],
        )

        # Legal/Financial Agent
        self.nexus.register_agent(
            LegalFinancialAgent(),
            priority_tasks=["dcf", "options", "var", "patent", "compliance"],
        )

        # Robotics Agent
        self.nexus.register_agent(
            RoboticsAgent(),
            priority_tasks=["forward_kinematics", "inverse_kinematics", "pid", "path_planning"],
        )

    def _seed_knowledge(self) -> None:
        """Seed the knowledge base with foundational facts."""
        facts = [
            ("CONST-G", "Gravitational constant G = 6.67430e-11 m³/(kg·s²)", "physics"),
            ("CONST-c", "Speed of light c = 2.99792458e8 m/s", "physics"),
            ("CONST-hbar", "Reduced Planck constant ℏ = 1.054571817e-34 J·s", "physics"),
            ("GPB-LT", "Gravity Probe B measured Lense-Thirring precession: 37.2 ± 7.2 mas/yr", "exotic_physics"),
            ("GPB-GEO", "Gravity Probe B measured geodetic precession: 6601.8 ± 18.3 mas/yr", "exotic_physics"),
            ("CASIMIR-INC", "Casimir Inc (2026) claims MicroSparc chips: 1.5V @ 25μA from Casimir cavities", "exotic_physics"),
            ("QET-2025", "Quantum Energy Teleportation demonstrated experimentally in <40ms (2024-2025)", "quantum_physics"),
            ("SICA-2025", "SICA agents achieved 17% → 53% on SWE-Bench Verified via self-modification", "ai"),
            ("MCP-2026", "MCP (Model Context Protocol) July 2026 spec: stateless HTTP for scalability", "ai"),
            ("A2A-2026", "A2A (Agent-to-Agent Protocol) by Google for inter-agent coordination", "ai"),
            ("SPARC-2026", "CFS SPARC tokamak 75% complete, first plasma targeted 2027", "fusion"),
            ("HELION-2026", "Helion Polaris: first private D-T fusion at 150M°C", "fusion"),
            ("JEPA-2026", "LeJEPA formal stability proofs published May 2026", "ai"),
            ("NEURALINK-2026", "Neuralink N1 expanding to high-volume production, PRIME/CONVOY trials", "neuroscience"),
        ]

        for fact_id, content, domain in facts:
            self.memory.semantic.store_fact(
                fact_id=fact_id,
                content=content,
                domain=domain,
                importance=0.8,
            )

        # Seed relationships
        relationships = [
            ("Casimir_effect", "enables", "vacuum_energy_extraction"),
            ("Casimir_effect", "produces", "attractive_force"),
            ("metamaterials", "enhance", "Casimir_effect"),
            ("Lense_Thirring", "confirms", "general_relativity"),
            ("Gravity_Probe_B", "measured", "Lense_Thirring"),
            ("SICA", "enables", "self_improvement"),
            ("LangGraph", "implements", "agent_orchestration"),
            ("MCP", "standardizes", "tool_integration"),
            ("A2A", "standardizes", "agent_communication"),
            ("JEPA", "models", "world_understanding"),
            ("active_inference", "minimizes", "free_energy"),
        ]

        for s, p, o in relationships:
            self.memory.semantic.store_relationship(s, p, o)


# ══════════════════════════════════════════════════════════
# CLI Display Functions
# ══════════════════════════════════════════════════════════

BANNER = f"""\
{Colors.CYAN}{Colors.BOLD}
 ╔══════════════════════════════════════════════════════════════╗
 ║                                                              ║
 ║     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                ║
 ║     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                ║
 ║     ██║███████║██████╔╝██║   ██║██║███████╗                 ║
 ║██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                ║
 ║╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                ║
 ║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                ║
 ║                    P R I M E                                 ║
 ║                                                              ║
 ║  Omnidisciplinary Autonomous Intelligence System v{__version__}      ║
 ║  Codename: {__codename__}                                           ║
 ╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}"""


HELP_TEXT = f"""
{colored("Available Commands:", Colors.YELLOW)}

  {colored("Natural Language:", Colors.GREEN)}
    Just type your query -- NEXUS will route it to the right agents.

  {colored("Direct Commands:", Colors.GREEN)}
    {colored("/status", Colors.CYAN)}         -- System status and agent metrics
    {colored("/agents", Colors.CYAN)}         -- List registered domain agents
    {colored("/memory", Colors.CYAN)}         -- Memory system statistics
    {colored("/journal", Colors.CYAN)}        -- Research journal (hypotheses & experiments)
    {colored("/sica", Colors.CYAN)}           -- Self-improvement engine status
    {colored("/llm", Colors.CYAN)}            -- LLM provider status (Groq/Gemini/Ollama)
    {colored("/physics", Colors.CYAN)}        -- Run full exotic physics analysis
    {colored("/casimir", Colors.CYAN)}        -- Run Casimir force analysis
    {colored("/alcubierre", Colors.CYAN)}     -- Run Alcubierre metric analysis
    {colored("/gpb", Colors.CYAN)}            -- Validate against Gravity Probe B
    {colored("/quantum", Colors.CYAN)}        -- Create Bell state and run quantum sim
    {colored("/biotech", Colors.CYAN)}        -- Run biotech protein analysis
    {colored("/cyber", Colors.CYAN)}          -- Run cybersecurity threat model
    {colored("/gaps", Colors.CYAN)}           -- Run knowledge gap analysis
    {colored("/synthesis", Colors.CYAN)}      -- Run cross-domain synthesis
    {colored("/help", Colors.CYAN)}           -- Show this help message
    {colored("/quit", Colors.CYAN)}           -- Exit JARVIS-PRIME
"""


def print_result(result: dict[str, Any], indent: int = 0) -> None:
    """Pretty-print a result dictionary."""
    prefix = "  " * indent
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{prefix}{colored(key + ':', Colors.CYAN)}")
            print_result(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{colored(key + ':', Colors.CYAN)}")
            for item in value:
                if isinstance(item, dict):
                    print_result(item, indent + 1)
                    print(f"{prefix}  {'─' * 40}")
                else:
                    print(f"{prefix}  • {item}")
        elif isinstance(value, float):
            if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
                print(f"{prefix}{colored(key + ':', Colors.CYAN)} {value:.6e}")
            else:
                print(f"{prefix}{colored(key + ':', Colors.CYAN)} {value:.6f}")
        else:
            print(f"{prefix}{colored(key + ':', Colors.CYAN)} {value}")


# ══════════════════════════════════════════════════════════
# Command Handlers
# ══════════════════════════════════════════════════════════

async def handle_command(system: JarvisSystem, command: str) -> None:
    """Handle a slash command."""
    cmd = command.strip().lower()

    if cmd == "/help":
        print(HELP_TEXT)

    elif cmd == "/status":
        print(colored("\n━━━ JARVIS-PRIME System Status ━━━\n", Colors.YELLOW))
        stats = system.nexus.stats()
        print_result(stats)

    elif cmd == "/agents":
        print(colored("\n━━━ Registered Domain Agents ━━━\n", Colors.YELLOW))
        agents = system.nexus.list_agents()
        for agent in agents:
            print(f"  {colored(agent['name'], Colors.GREEN)} [{agent['domain']}]")
            print(f"    Capabilities: {', '.join(agent['capabilities'][:5])}")
            print(f"    Success Rate: {agent['metrics']['success_rate']:.1%}")
            print(f"    Tasks Done: {agent['metrics']['tasks_completed']}")
            print()

    elif cmd == "/memory":
        print(colored("\n━━━ Memory System ━━━\n", Colors.YELLOW))
        print_result(system.memory.stats())

    elif cmd == "/journal":
        print(colored("\n━━━ Research Journal ━━━\n", Colors.YELLOW))
        # Get the exotic physics agent's journal
        for reg in system.nexus._agents.values():
            if hasattr(reg.agent, 'journal'):
                journal = reg.agent.journal.get_all()
                print_result(journal)
                break

    elif cmd == "/sica":
        print(colored("\n━━━ SICA Self-Improvement Engine ━━━\n", Colors.YELLOW))
        print_result(system.sica.stats())

    elif cmd == "/physics":
        print(colored("\n━━━ Running Full Exotic Physics Analysis ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Run full exotic physics analysis including Lense-Thirring, Alcubierre, and Casimir",
            priority=TaskPriority.HIGH,
        )
        print_result(result)

    elif cmd == "/casimir":
        print(colored("\n━━━ Casimir Force Analysis ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Analyze Casimir force engineering with metamaterial enhancement",
        )
        print_result(result)

    elif cmd == "/alcubierre":
        print(colored("\n━━━ Alcubierre Warp Metric Analysis ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Compute Alcubierre warp drive energy requirements",
        )
        print_result(result)

    elif cmd == "/gpb":
        print(colored("\n━━━ Gravity Probe B Validation ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Validate Lense-Thirring frame-dragging computation against Gravity Probe B data",
        )
        print_result(result)

    elif cmd == "/gaps":
        print(colored("\n━━━ Knowledge Gap Analysis ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Perform gap analysis across all research domains",
        )
        print_result(result)

    elif cmd == "/synthesis":
        print(colored("\n━━━ Cross-Domain Synthesis ━━━\n", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Find cross-domain synthesis breakthroughs at intersections",
        )
        print_result(result)

    elif cmd == "/llm":
        print(colored("\n=== LLM Provider Status ===", Colors.YELLOW))
        print_result(system.cognitive_core.stats())

    elif cmd == "/quantum":
        print(colored("\n=== Quantum Simulation ===", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Create a Bell state and demonstrate quantum entanglement",
        )
        print_result(result)

    elif cmd == "/biotech":
        print(colored("\n=== Biotech Analysis ===", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Analyze protein sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        )
        print_result(result)

    elif cmd == "/cyber":
        print(colored("\n=== Cybersecurity Analysis ===", Colors.YELLOW))
        result = await system.nexus.process_goal(
            "Perform threat modeling and security audit",
        )
        print_result(result)

    elif cmd == "/quit" or cmd == "/exit":
        print(colored("\nShutting down JARVIS-PRIME. All research logged.\n", Colors.DIM))
        sys.exit(0)

    else:
        print(colored(f"Unknown command: {command}. Type /help for available commands.", Colors.RED))


async def handle_query(system: JarvisSystem, query: str) -> None:
    """Handle a natural language query."""
    print(colored(f"\n⟡ Processing: ", Colors.DIM) + colored(query, Colors.WHITE))
    print(colored("  Decomposing goal → Routing to agents → Executing...\n", Colors.DIM))

    start = time.time()
    result = await system.nexus.process_goal(query)
    duration = time.time() - start

    print(colored(f"━━━ Results ({duration:.2f}s) ━━━\n", Colors.GREEN))
    print_result(result)
    print()


# ══════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════

async def interactive_loop(system: JarvisSystem) -> None:
    """Main interactive loop."""
    print(BANNER)
    print(colored("  System initialized. Type /help for commands or ask anything.\n", Colors.DIM))

    # Show initial status
    agents = system.nexus.list_agents()
    print(colored(f"  [OK] {len(agents)} domain agents online", Colors.GREEN))
    print(colored(f"  [OK] {system.memory.semantic.stats()['facts']} knowledge facts loaded", Colors.GREEN))
    print(colored(f"  [OK] LLM provider: {system.cognitive_core.get_active_provider()}", Colors.GREEN))
    print(colored(f"  [OK] SICA self-improvement engine: {'ENABLED' if system.config.sica_enabled else 'DISABLED'}", Colors.GREEN))
    print()

    while True:
        try:
            user_input = input(colored("JARVIS", Colors.CYAN) + colored(" > ", Colors.YELLOW))
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                await handle_command(system, user_input)
            else:
                await handle_query(system, user_input)

        except KeyboardInterrupt:
            print(colored("\n\n  Interrupted. Use /quit to exit cleanly.\n", Colors.YELLOW))
        except EOFError:
            break


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="JARVIS-PRIME: Omnidisciplinary Autonomous Intelligence System"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Process a single query and exit",
    )
    parser.add_argument(
        "--command", "-c",
        type=str,
        default=None,
        help="Execute a single command and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    system = JarvisSystem()

    if args.query:
        result = asyncio.run(
            system.nexus.process_goal(args.query)
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_result(result)
    elif args.command:
        asyncio.run(handle_command(system, args.command))
    else:
        asyncio.run(interactive_loop(system))


if __name__ == "__main__":
    main()
