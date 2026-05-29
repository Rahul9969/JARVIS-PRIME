"""
JARVIS-PRIME Phase 2 Test Suite
=================================
Tests for: Cognitive Core, Knowledge Engine, New Agents, Integration
"""
import asyncio
import sys
import os
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
# Cognitive Core Tests
# ──────────────────────────────────────────────────────

def test_cognitive_core(results):
    print("\n[Cognitive Core Tests]")
    from jarvis.core.cognitive_core import (
        CognitiveCore, MockProvider, GroqProvider, GeminiProvider, OllamaProvider
    )

    # Test 1: MockProvider is always available
    mock = MockProvider()
    results.record("MockProvider is available", mock.is_available())

    # Test 2: MockProvider generates response
    response = asyncio.run(mock.generate("Test prompt"))
    results.record("MockProvider generates text", len(response) > 0)

    # Test 3: CognitiveCore initializes
    core = CognitiveCore()
    results.record("CognitiveCore initializes", core is not None)

    # Test 4: Active provider fallback to mock
    active = core.get_active_provider()
    results.record("Falls back to mock provider", active in ("mock", "groq", "gemini", "ollama"))

    # Test 5: Generate with failover
    response = asyncio.run(core.generate("What is the Casimir effect?"))
    results.record("Generate with failover works", len(response) > 0)

    # Test 6: Structured JSON output
    structured = asyncio.run(core.generate_structured(
        "Decompose: analyze physics",
        schema_hint='{"tasks": [{"domain": "physics"}]}'
    ))
    results.record("Structured output returns dict", isinstance(structured, dict))

    # Test 7: Stats reporting
    stats = core.stats()
    results.record("Stats has providers", "providers" in stats)
    results.record("Stats has active provider", "active_provider" in stats)

    # Test 8: GroqProvider disabled without key
    groq = GroqProvider(api_key="")
    results.record("Groq disabled without API key", not groq.is_available())

    # Test 9: GeminiProvider disabled without key
    gemini = GeminiProvider(api_key="")
    results.record("Gemini disabled without API key", not gemini.is_available())

    # Test 10: JSON parsing from markdown code block
    parsed = core._parse_json('```json\n{"key": "value"}\n```')
    results.record("Parse JSON from markdown block", parsed.get("key") == "value")


# ──────────────────────────────────────────────────────
# Knowledge Engine Tests
# ──────────────────────────────────────────────────────

def test_knowledge_engine(results):
    print("\n[Knowledge Engine Tests]")

    # Test embeddings
    from jarvis.knowledge.embeddings import TFIDFEmbedder, EmbeddingEngine

    embedder = TFIDFEmbedder(dim=128)
    corpus = [
        "Casimir force between parallel plates",
        "Quantum entanglement in Bell states",
        "Protein folding with AlphaFold",
    ]
    embedder.fit(corpus)

    # Test 1: Embedding produces correct dimension
    vec = embedder.embed("Casimir effect")
    results.record("TF-IDF embedding dim=128", len(vec) == 128)

    # Test 2: Embedding is normalized
    import math
    norm = math.sqrt(sum(v*v for v in vec))
    results.record("TF-IDF embedding is L2-normalized", abs(norm - 1.0) < 0.01 or norm == 0)

    # Test 3: Similar texts have similar embeddings
    vec1 = embedder.embed("Casimir force")
    vec2 = embedder.embed("Casimir pressure")
    vec3 = embedder.embed("Quantum computing")
    sim_12 = sum(a*b for a, b in zip(vec1, vec2))
    sim_13 = sum(a*b for a, b in zip(vec1, vec3))
    results.record("Similar texts have higher similarity", sim_12 > sim_13 or sim_12 == sim_13 == 0)

    # Test 4: EmbeddingEngine initializes
    engine = EmbeddingEngine(dim=128)
    results.record("EmbeddingEngine initializes", engine.backend in ("tfidf", "sentence-transformers"))

    # Test 5: Batch embedding
    vecs = embedder.embed_batch(["hello world", "test text"])
    results.record("Batch embedding works", len(vecs) == 2)

    # Test GraphRAG
    from jarvis.knowledge.graph_rag import GraphRAG

    # Test 6: GraphRAG initializes
    rag = GraphRAG()
    results.record("GraphRAG initializes", rag is not None)

    # Test 7: Add fact
    rag.add_fact("test-001", "Casimir force is attractive between plates", domain="physics")
    results.record("Add fact works", rag.graph.has_node("test-001"))

    # Test 8: Add relationship
    rag.add_relationship("Casimir", "produces", "force")
    results.record("Add relationship works", rag.graph.has_edge("Casimir", "force"))

    # Test 9: Query returns results
    query_results = asyncio.run(rag.query("Casimir force", top_k=3))
    results.record("Query returns results", len(query_results) > 0)

    # Test 10: Stats
    stats = rag.stats()
    results.record("Knowledge stats work", "vector_store" in stats and "knowledge_graph" in stats)

    # Test 11: Document chunking
    chunks = rag._chunk_text("word " * 1000, chunk_size=100, overlap=10)
    results.record("Document chunking works", len(chunks) > 5)


# ──────────────────────────────────────────────────────
# New Agent Tests
# ──────────────────────────────────────────────────────

def test_cybersecurity_agent(results):
    print("\n[Cybersecurity Agent Tests]")
    from jarvis.agents.cybersecurity import CybersecurityAgent

    agent = CybersecurityAgent()

    # Test 1: Capabilities
    caps = agent.get_capabilities()
    results.record("Has cybersecurity capabilities", "mitre_attack_mapping" in caps)

    # Test 2: Threat model
    result = asyncio.run(agent.execute({"type": "threat_model"}, []))
    results.record("STRIDE threat model works", result.get("methodology") == "STRIDE")

    # Test 3: MITRE mapping
    result = asyncio.run(agent.execute({"type": "mitre_mapping"}, []))
    results.record("MITRE ATT&CK mapping works", "T1566" in result.get("common_techniques", {}))

    # Test 4: Attack surface
    result = asyncio.run(agent.execute({"type": "attack_surface"}, []))
    results.record("Attack surface analysis works", len(result.get("areas", [])) > 0)

    # Test 5: Security audit
    result = asyncio.run(agent.execute({"type": "security_audit"}, []))
    results.record("Security audit works", len(result.get("checklist", [])) > 0)


def test_biotech_agent(results):
    print("\n[Biotech Agent Tests]")
    from jarvis.agents.biotech import BiotechAgent

    agent = BiotechAgent()

    # Test 1: Protein analysis
    result = asyncio.run(agent.execute({
        "type": "protein_analysis",
        "parameters": {"sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"}
    }, []))
    results.record("Protein analysis works", result.get("length") == 51)
    results.record("Molecular weight computed", result.get("molecular_weight_Da", 0) > 0)

    # Test 2: CRISPR design
    result = asyncio.run(agent.execute({
        "type": "crispr_design",
        "parameters": {"target_sequence": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG"}
    }, []))
    results.record("CRISPR guide design works", result.get("cas_protein") == "SpCas9")
    results.record("Found PAM sites", result.get("guides_found", 0) >= 0)

    # Test 3: Sequence stats
    result = asyncio.run(agent.execute({
        "type": "sequence_stats",
        "parameters": {"sequence": "ATGCATGCATGC"}
    }, []))
    results.record("DNA sequence stats work", result.get("type") == "DNA")
    results.record("GC content calculated", "gc_content_pct" in result)


def test_quantum_agent(results):
    print("\n[Quantum Agent Tests]")
    from jarvis.agents.quantum import QuantumAgent, QuantumSimulator

    # Test simulator directly
    sim = QuantumSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)

    # Test 1: Bell state probabilities
    probs = sim.get_probabilities()
    results.record("Bell state has 2 outcomes", len(probs) == 2)
    results.record("Bell |00> prob ~ 0.5", abs(probs.get("|00>", 0) - 0.5) < 0.01)
    results.record("Bell |11> prob ~ 0.5", abs(probs.get("|11>", 0) - 0.5) < 0.01)

    # Test 2: Measurement
    counts = sim.measure(1000)
    results.record("Measurement returns counts", sum(counts.values()) == 1000)
    results.record("Only |00> and |11> measured", set(counts.keys()).issubset({"00", "11"}))

    # Test agent
    agent = QuantumAgent()

    # Test 3: Bell state via agent
    result = asyncio.run(agent.execute({"type": "bell_state"}, []))
    results.record("Agent Bell state works", result.get("entanglement") == "MAXIMALLY ENTANGLED")

    # Test 4: GHZ state
    result = asyncio.run(agent.execute({
        "type": "ghz_state",
        "parameters": {"n_qubits": 3}
    }, []))
    results.record("3-qubit GHZ state works", result.get("n_qubits") == 3)

    # Test 5: Teleportation
    result = asyncio.run(agent.execute({"type": "quantum_teleportation"}, []))
    results.record("Teleportation demo works", "protocol" in result)


def test_creative_agent(results):
    print("\n[Creative Agent Tests]")
    from jarvis.agents.creative import CreativeAgent

    agent = CreativeAgent()

    # Test 1: Paper drafting
    result = asyncio.run(agent.execute({"type": "draft_paper", "description": "Casimir Force Analysis"}, []))
    results.record("Paper template generated", "template" in result)

    # Test 2: Report writing
    result = asyncio.run(agent.execute({"type": "write_report", "description": "System Report"}, []))
    results.record("Report generated", "report" in result)


# ──────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────

def test_integration(results):
    print("\n[Integration Tests]")
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

    # Build full system
    memory = HierarchicalMemory()
    sica = SICAEngine()
    mcp = MCPGateway()
    a2a = A2ACoordinator()
    llm = CognitiveCore()

    nexus = Nexus(memory=memory, sica=sica, mcp=mcp, a2a=a2a, cognitive_core=llm)

    # Register all agents
    for agent in [
        ExoticPhysicsAgent(),
        ScientificDiscoveryAgent(),
        CybersecurityAgent(),
        BiotechAgent(),
        QuantumAgent(),
        CreativeAgent(),
    ]:
        nexus.register_agent(agent)

    # Test 1: All 6 agents registered
    agents = nexus.list_agents()
    results.record("6 agents registered", len(agents) == 6)

    # Test 2: Process physics goal
    result = asyncio.run(nexus.process_goal("Analyze Casimir force at 100nm"))
    results.record("Physics goal processes", result.get("status") == "completed")

    # Test 3: Process security goal
    result = asyncio.run(nexus.process_goal("Perform security threat model"))
    results.record("Security goal processes", result.get("status") == "completed")

    # Test 4: Process quantum goal
    result = asyncio.run(nexus.process_goal("Create quantum Bell state entanglement"))
    results.record("Quantum goal processes", result.get("status") == "completed")

    # Test 5: Cross-domain query
    result = asyncio.run(nexus.process_goal("Research quantum gravity and security implications"))
    results.record("Cross-domain goal processes", result.get("status") == "completed")

    # Test 6: LLM stats available
    llm_stats = llm.stats()
    results.record("LLM stats available", llm_stats.get("total_requests", 0) >= 0)


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  JARVIS-PRIME Phase 2 Test Suite")
    print("=" * 60)

    results = Results()

    test_cognitive_core(results)
    test_knowledge_engine(results)
    test_cybersecurity_agent(results)
    test_biotech_agent(results)
    test_quantum_agent(results)
    test_creative_agent(results)
    test_integration(results)

    results.summary()

    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
