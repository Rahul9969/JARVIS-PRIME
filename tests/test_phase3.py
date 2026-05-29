"""
JARVIS-PRIME Phase 3-5 Test Suite
====================================
Tests all new modules: Agents, Simulation, Perception,
Cognitive Core, Infrastructure, Knowledge
"""
import asyncio
import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, success, detail=""):
        if success:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────
# Energy Agent Tests
# ──────────────────────────────────────────────────────

def test_energy_agent(results):
    print("\n[Energy/Fusion Agent Tests]")
    from jarvis.agents.energy import EnergyAgent, TokamakAnalyzer, SolarAnalyzer

    agent = EnergyAgent()

    # Lawson criterion
    lawson = agent.tokamak.lawson_criterion(density_m3=1e20, temperature_keV=15, confinement_time_s=3)
    results.record("Lawson criterion computes", lawson["triple_product"] > 0)
    results.record("Lawson status valid", lawson["status"] in ("IGNITION", "BREAKEVEN", "SUB-BREAKEVEN"))

    # Magnetic confinement
    conf = agent.tokamak.magnetic_confinement()
    results.record("Tokamak confinement works", conf["plasma_volume_m3"] > 0)
    results.record("Safety factor computed", conf["safety_factor_q"] > 0)
    results.record("Troyon beta limit", conf["troyon_beta_limit_pct"] > 0)

    # Fusion power
    power = agent.tokamak.fusion_power()
    results.record("Fusion power estimated", power["fusion_power_MW"] > 0)

    # Solar
    solar = agent.solar.panel_output()
    results.record("Solar output computed", solar["daily_energy_kWh"] > 0)
    results.record("CO2 avoided calculated", solar["co2_avoided_kg_yr"] > 0)


# ──────────────────────────────────────────────────────
# Legal/Financial Agent Tests
# ──────────────────────────────────────────────────────

def test_legal_financial(results):
    print("\n[Legal/Financial Agent Tests]")
    from jarvis.agents.legal_financial import LegalFinancialAgent

    agent = LegalFinancialAgent()

    # Compound interest
    ci = agent.finance.compound_interest(10000, 0.08, 10, 12)
    results.record("Compound interest works", ci["final_value"] > 10000)
    results.record("Interest accrued", ci["total_interest"] > 0)

    # DCF
    dcf = agent.finance.dcf_valuation()
    results.record("DCF valuation works", dcf["total_present_value"] > 0)

    # Black-Scholes
    bs = agent.finance.black_scholes()
    results.record("Call price positive", bs["call_price"] > 0)
    results.record("Put price positive", bs["put_price"] > 0)
    results.record("Greeks computed", "delta_call" in bs["greeks"])

    # Monte Carlo VaR
    var = agent.finance.monte_carlo_var()
    results.record("VaR computed", var["VaR"] > 0)
    results.record("CVaR >= VaR", var["CVaR_expected_shortfall"] >= var["VaR"])

    # Legal
    patent = agent.legal.patent_framework()
    results.record("Patent framework has sections", len(patent["sections"]) >= 5)

    compliance = agent.legal.compliance_frameworks()
    results.record("GDPR in compliance", "GDPR" in compliance["frameworks"])


# ──────────────────────────────────────────────────────
# Robotics Agent Tests
# ──────────────────────────────────────────────────────

def test_robotics(results):
    print("\n[Robotics Agent Tests]")
    from jarvis.agents.robotics import RoboticsAgent

    agent = RoboticsAgent()

    # Forward kinematics
    fk = agent.kinematics.forward(1.0, 0.8, 45, 30)
    results.record("Forward kinematics works", len(fk["end_effector"]) == 2)
    results.record("End effector within reach", fk["reach"] <= fk["max_reach"])

    # Inverse kinematics
    ik = agent.kinematics.inverse(1.0, 0.8, 1.0, 0.5)
    results.record("Inverse kinematics works", "elbow_up" in ik)

    # PID controller
    pid = agent.pid.simulate()
    results.record("PID simulation works", pid["final_output"] > 0)
    results.record("PID rises to setpoint", abs(pid["final_output"] - 1.0) < 0.5)
    results.record("PID performance metrics", "rise_time_s" in pid["performance"])

    # Path planning
    path = agent.planner.a_star()
    results.record("A* finds path", path["path_found"])
    results.record("A* path reasonable length", path["path_length"] > 10)


# ──────────────────────────────────────────────────────
# Plasma Simulator Tests
# ──────────────────────────────────────────────────────

def test_plasma(results):
    print("\n[Plasma Physics Tests]")
    from jarvis.simulation.plasma import PlasmaSimulator

    sim = PlasmaSimulator()

    # Plasma frequency
    pf = sim.plasma_frequency(1e18)
    results.record("Plasma frequency computed", pf["plasma_frequency_Hz"] > 0)
    results.record("Plasma frequency ~GHz range", pf["plasma_frequency_GHz"] > 0.1)

    # Debye length
    dl = sim.debye_length(10.0, 1e18)
    results.record("Debye length computed", dl["debye_length_m"] > 0)
    results.record("Plasma condition valid", dl["is_plasma"])

    # Cyclotron frequency
    cf = sim.cyclotron_frequency(5.0, species="electron")
    results.record("Electron cyclotron freq", cf["cyclotron_frequency_GHz"] > 0)
    results.record("Larmor radius computed", cf["larmor_radius_mm"] > 0)

    # Magnetic mirror
    mm = sim.magnetic_mirror(1.0, 5.0, 45.0)
    results.record("Mirror ratio correct", mm["mirror_ratio"] == 5.0)
    results.record("Trapped fraction reasonable", 0 < mm["trapped_fraction"] < 1)

    # MHD beta
    beta = sim.mhd_beta(1e5, 5.0)
    results.record("Beta computed", beta["beta"] > 0)


# ──────────────────────────────────────────────────────
# Molecular Dynamics Tests
# ──────────────────────────────────────────────────────

def test_molecular(results):
    print("\n[Molecular Dynamics Tests]")
    from jarvis.simulation.molecular import LennardJones, MolecularDynamics, EnergyMinimizer

    # LJ potential
    lj = LennardJones()
    results.record("LJ minimum at ~1.12 sigma", abs(lj.potential_curve()["r_min_sigma"] - 1.12) < 0.1)
    results.record("LJ well depth = 1 epsilon", abs(lj.potential_curve()["well_depth"] - 1.0) < 0.05)

    # MD simulation
    md = MolecularDynamics(n_particles=20, box_size=5.0, temperature=1.0, dt=0.001)
    result = md.run(steps=200)
    results.record("MD simulation completes", result["results"]["avg_temperature"] > 0)
    results.record("MD particles correct", result["n_particles"] == 20)

    # Energy minimizer
    em = EnergyMinimizer.minimize_2d("sphere")
    results.record("Sphere minimization converges", em["converged"])
    results.record("Sphere minimum near origin", em["distance_to_minimum"] < 0.01)


# ──────────────────────────────────────────────────────
# Perception Tests
# ──────────────────────────────────────────────────────

def test_perception(results):
    print("\n[Perception Tests]")

    # Audio
    from jarvis.perception.audio import AudioAnalyzer, WakeWordDetector

    audio = AudioAnalyzer()
    signal = audio.generate_test_signal([440], duration_s=0.5)
    results.record("Test signal generated", len(signal) == 8000)

    analysis = audio.analyze_signal(signal)
    results.record("Peak frequency near 440Hz", abs(analysis["peak_frequency_Hz"] - 440) < 50)
    results.record("RMS amplitude > 0", analysis["rms_amplitude"] > 0)

    spec = audio.compute_spectrogram(signal)
    results.record("Spectrogram computed", spec["n_frames"] > 0)

    mfcc = audio.compute_mfcc(signal)
    results.record("MFCC computed", len(mfcc["mean_coefficients"]) == 13)

    wake = WakeWordDetector()
    results.record("Wake word detector status", wake.status()["wake_word"] == "jarvis")

    # Vision
    from jarvis.perception.vision import ImageAnalyzer, VisionPipeline

    img = ImageAnalyzer.generate_test_image(64, 64, "gradient")
    results.record("Test image generated", img.shape == (64, 64, 3))

    analysis = ImageAnalyzer.analyze(img)
    results.record("Image analysis works", analysis["total_pixels"] == 4096)

    edges = ImageAnalyzer.edge_detect(img)
    results.record("Edge detection works", "edge_density_pct" in edges)

    pipeline = VisionPipeline()
    result = pipeline.process()
    results.record("Vision pipeline works", "analysis" in result)


# ──────────────────────────────────────────────────────
# Cognitive Core Tests
# ──────────────────────────────────────────────────────

def test_world_model(results):
    print("\n[World Model Tests]")
    from jarvis.core.world_model import WorldModel

    wm = WorldModel(state_dim=32)

    # Encode observation
    wm.encode_observation({"temperature": 300, "pressure": 1e5}, domain="physics")
    results.record("Observation encoded", wm.current_state.to_dict()["norm"] > 0)

    # Predict
    pred = wm.predict_next_state({"heat": 50})
    results.record("Prediction works", pred["similarity_to_current"] > 0)

    # Anomaly detection
    anomaly = wm.detect_anomaly({"temperature": 300})
    results.record("Anomaly detection works", "prediction_error" in anomaly)

    # Scenario simulation
    scenario = wm.simulate_scenario([{"action": 1}, {"action": 2}])
    results.record("Scenario simulation works", len(scenario["trajectory"]) == 2)

    # Stats
    stats = wm.stats()
    results.record("World model stats", stats["state_dim"] == 32)


def test_active_inference(results):
    print("\n[Active Inference Tests]")
    from jarvis.core.active_inference import ActiveInferenceAgent, BeliefState

    # Belief state
    bs = BeliefState(8)
    results.record("Uniform entropy = max entropy", abs(bs.entropy() - bs.max_entropy()) < 0.01)
    results.record("Uniform certainty near 0", bs.certainty() < 0.05)

    # Bayesian update
    likelihood = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])
    bs.update(likelihood)
    results.record("Belief updates toward state 0", bs.most_likely_state() == 0)
    results.record("Certainty increases after update", bs.certainty() > 0.3)

    # Full agent
    agent = ActiveInferenceAgent(n_states=4, n_obs=4, n_actions=3)

    # Perceive
    perception = agent.perceive(0)
    results.record("Perception works", "free_energy" in perception)

    # Plan
    plan = agent.plan(planning_horizon=2)
    results.record("Planning works", "best_action" in plan)

    # Run episode
    episode = agent.run_episode(n_steps=10)
    results.record("Episode runs", episode["n_steps"] == 10)
    results.record("State estimation > 0%", episode["state_estimation_accuracy"] >= 0)


def test_reasoning(results):
    print("\n[Reasoning Engine Tests]")
    from jarvis.core.reasoning import ReasoningEngine, MCTSReasoner

    engine = ReasoningEngine()

    # Chain of thought (without LLM)
    cot = asyncio.run(engine.reason("What is 2+2?", strategy="cot"))
    results.record("CoT reasoning works", cot["n_steps"] > 0)

    # Tree of thoughts
    tot = asyncio.run(engine.reason("Compare fusion vs fission", strategy="tot"))
    results.record("ToT reasoning works", tot["total_nodes_explored"] > 1)
    results.record("ToT best path found", len(tot["best_path"]) > 0)

    # MCTS
    mcts = MCTSReasoner(n_simulations=50)
    result = mcts.search("Should we invest in fusion?", ["invest", "wait", "diversify"])
    results.record("MCTS search works", result["total_nodes"] > 1)
    results.record("MCTS selects action", result["best_action"] in ["invest", "wait", "diversify"])


# ──────────────────────────────────────────────────────
# Infrastructure Tests
# ──────────────────────────────────────────────────────

def test_telemetry(results):
    print("\n[Telemetry Tests]")
    from jarvis.infrastructure.telemetry import Telemetry

    tele = Telemetry()

    # Metrics
    tele.metrics.increment("test.counter", 5)
    results.record("Counter increment", tele.metrics.get_counter("test.counter") == 5)

    tele.metrics.set_gauge("test.gauge", 42.0)
    results.record("Gauge set", tele.metrics.get_gauge("test.gauge") == 42.0)

    for i in range(100):
        tele.metrics.record("test.latency", i * 0.1)
    hist = tele.metrics.get_histogram_stats("test.latency")
    results.record("Histogram stats", hist["count"] == 100)
    results.record("P95 computed", hist["p95"] > 0)

    # Spans
    span = tele.start_span("test_operation", agent="physics")
    span.add_event("step_1")
    tele.end_span(span)
    results.record("Span tracing works", len(tele._completed_spans) == 1)

    # System info
    info = tele.get_system_info()
    results.record("System info available", "platform" in info)


def test_auth(results):
    print("\n[Auth/RBAC Tests]")
    from jarvis.infrastructure.auth import AuthManager

    auth = AuthManager()

    # Create user
    user_info = auth.create_user("researcher_1", "researcher")
    results.record("User created", "api_key" in user_info)

    # Authenticate
    user = auth.authenticate(user_info["api_key"])
    results.record("Auth works", user is not None and user.username == "researcher_1")

    # Authorize
    results.record("Researcher can submit goals", auth.authorize(user, "goal.submit"))
    results.record("Researcher cannot shutdown", not auth.authorize(user, "system.shutdown"))

    # Viewer
    viewer_info = auth.create_user("viewer_1", "viewer")
    viewer = auth.authenticate(viewer_info["api_key"])
    results.record("Viewer can view goals", auth.authorize(viewer, "goal.view"))
    results.record("Viewer cannot submit goals", not auth.authorize(viewer, "goal.submit"))

    # Stats
    stats = auth.stats()
    results.record("Auth stats work", stats["total_users"] == 3)  # admin + 2


def test_ontology(results):
    print("\n[Ontology Tests]")
    from jarvis.knowledge.ontology import OntologyManager

    mgr = OntologyManager()

    # Physics ontology
    physics = mgr.get_ontology("physics")
    results.record("Physics ontology loaded", physics is not None)

    subclasses = physics.get_subclasses("Force")
    results.record("Force has subclasses", len(subclasses) >= 4)

    props = physics.get_properties("CasimirForce")
    results.record("CasimirForce has properties", "origin" in props)

    # Inherited properties
    results.record("CasimirForce inherits from EM", "mediator" in props)

    # AI ontology
    ai = mgr.get_ontology("ai")
    results.record("AI ontology loaded", ai is not None)

    # Cross-domain query
    results_list = mgr.query_across_domains("Force")
    results.record("Cross-domain query works", len(results_list) > 0)

    # Stats
    stats = mgr.stats()
    results.record("Ontology stats work", stats["total_classes"] > 20)


# ──────────────────────────────────────────────────────
# Full Integration Test
# ──────────────────────────────────────────────────────

def test_full_integration(results):
    print("\n[Full Integration Test]")
    from jarvis.core.cognitive_core import CognitiveCore
    from jarvis.core.memory import HierarchicalMemory
    from jarvis.core.protocols import MCPGateway, A2ACoordinator
    from jarvis.core.self_improvement import SICAEngine
    from jarvis.core.nexus import Nexus
    from jarvis.agents.exotic_physics import ExoticPhysicsAgent
    from jarvis.agents.scientific import ScientificDiscoveryAgent
    from jarvis.agents.cybersecurity import CybersecurityAgent
    from jarvis.agents.biotech import BiotechAgent
    from jarvis.agents.quantum import QuantumAgent
    from jarvis.agents.creative import CreativeAgent
    from jarvis.agents.energy import EnergyAgent
    from jarvis.agents.legal_financial import LegalFinancialAgent
    from jarvis.agents.robotics import RoboticsAgent

    # Build full 9-agent system
    memory = HierarchicalMemory()
    nexus = Nexus(
        memory=memory,
        sica=SICAEngine(),
        mcp=MCPGateway(),
        a2a=A2ACoordinator(),
        cognitive_core=CognitiveCore(),
    )

    all_agents = [
        ExoticPhysicsAgent(), ScientificDiscoveryAgent(),
        CybersecurityAgent(), BiotechAgent(),
        QuantumAgent(), CreativeAgent(),
        EnergyAgent(), LegalFinancialAgent(), RoboticsAgent(),
    ]
    for agent in all_agents:
        nexus.register_agent(agent)

    # Verify 9 agents
    agents = nexus.list_agents()
    results.record("9 agents registered", len(agents) == 9)

    # Test each domain routes correctly
    result = asyncio.run(nexus.process_goal("Analyze tokamak fusion energy confinement"))
    results.record("Energy goal routes", result["status"] == "completed")

    result = asyncio.run(nexus.process_goal("Calculate Black-Scholes options pricing"))
    results.record("Financial goal routes", result["status"] == "completed")

    result = asyncio.run(nexus.process_goal("Plan robotic arm path with A* algorithm"))
    results.record("Robotics goal routes", result["status"] == "completed")

    # Cross-domain
    result = asyncio.run(nexus.process_goal("Research quantum computing security implications for fusion reactor control systems"))
    results.record("Cross-domain routes", result["status"] == "completed")


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  JARVIS-PRIME Phase 3-5 Test Suite")
    print("=" * 60)

    results = Results()

    test_energy_agent(results)
    test_legal_financial(results)
    test_robotics(results)
    test_plasma(results)
    test_molecular(results)
    test_perception(results)
    test_world_model(results)
    test_active_inference(results)
    test_reasoning(results)
    test_telemetry(results)
    test_auth(results)
    test_ontology(results)
    test_full_integration(results)

    results.summary()

    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
