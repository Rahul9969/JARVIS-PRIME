"""
JARVIS-PRIME Dashboard — FastAPI REST API & Web UI
=====================================================

Provides:
- REST API for all JARVIS-PRIME operations
- Auto-generated OpenAPI docs at /docs
- Static web dashboard at /
- Real-time system status monitoring

Run:
    python -m jarvis.interfaces.dashboard
    → http://localhost:8000
    → http://localhost:8000/docs  (Swagger UI)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────
# JARVIS System Bootstrap
# ──────────────────────────────────────────────────────

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jarvis.core.cognitive_core import CognitiveCore
from jarvis.core.memory import HierarchicalMemory
from jarvis.core.protocols import MCPGateway, A2ACoordinator
from jarvis.core.self_improvement import SICAEngine
from jarvis.core.nexus import Nexus, TaskPriority
from jarvis.knowledge.graph_rag import GraphRAG

from jarvis.agents.exotic_physics import ExoticPhysicsAgent
from jarvis.agents.scientific import ScientificDiscoveryAgent
from jarvis.agents.cybersecurity import CybersecurityAgent
from jarvis.agents.biotech import BiotechAgent
from jarvis.agents.quantum import QuantumAgent
from jarvis.agents.creative import CreativeAgent
from jarvis.agents.energy import EnergyAgent
from jarvis.agents.legal_financial import LegalFinancialAgent
from jarvis.agents.robotics import RoboticsAgent
from jarvis.infrastructure.telemetry import Telemetry
from jarvis.infrastructure.auth import AuthManager
from jarvis.knowledge.ontology import OntologyManager
from jarvis.core.reasoning import ReasoningEngine
from jarvis.core.world_model import WorldModel
from jarvis.core.active_inference import ActiveInferenceAgent as AIFAgent


# ──────────────────────────────────────────────────────
# Initialize JARVIS System
# ──────────────────────────────────────────────────────

data_dir = Path("jarvis_data")
data_dir.mkdir(exist_ok=True)

# Core systems
memory = HierarchicalMemory(persist_dir=data_dir / "memory")
sica = SICAEngine()
mcp = MCPGateway()
a2a = A2ACoordinator()
cognitive_core = CognitiveCore()
knowledge = GraphRAG(persist_dir=data_dir / "knowledge")

# NEXUS orchestrator with LLM
nexus = Nexus(
    memory=memory,
    sica=sica,
    mcp=mcp,
    a2a=a2a,
    cognitive_core=cognitive_core,
)

# Register all domain agents
agents = [
    ExoticPhysicsAgent(),
    ScientificDiscoveryAgent(),
    CybersecurityAgent(),
    BiotechAgent(),
    QuantumAgent(),
    CreativeAgent(),
    EnergyAgent(),
    LegalFinancialAgent(),
    RoboticsAgent(),
]
for agent in agents:
    nexus.register_agent(agent)

# Phase 3-5 systems
telemetry = Telemetry()
auth_manager = AuthManager()
ontology = OntologyManager()
reasoning = ReasoningEngine(llm=cognitive_core)
world_model = WorldModel()
aif_agent = AIFAgent()

# Seed knowledge base with initial facts
knowledge.add_fact("F001", "Casimir force scales as 1/d^4 between parallel plates", domain="physics", importance=0.9)
knowledge.add_fact("F002", "Lense-Thirring precession rate: Omega_LT = GJ/(c^2 r^3)", domain="physics", importance=0.9)
knowledge.add_fact("F003", "Alcubierre warp metric requires negative energy density (exotic matter)", domain="physics", importance=0.8)
knowledge.add_fact("F004", "SICA agents achieved 17% to 53% improvement on SWE-Bench", domain="ai", importance=0.8)
knowledge.add_fact("F005", "MCP is the USB-C of AI for agent-to-tool connectivity", domain="ai", importance=0.7)
knowledge.add_fact("F006", "Google Willow demonstrated below-threshold quantum error correction", domain="quantum", importance=0.8)
knowledge.add_relationship("Casimir_effect", "produces", "attractive_force")
knowledge.add_relationship("Casimir_effect", "requires", "parallel_plates")
knowledge.add_relationship("Casimir_effect", "scales_as", "inverse_d4")
knowledge.add_relationship("Alcubierre_metric", "requires", "exotic_matter")
knowledge.add_relationship("Lense_Thirring", "predicts", "frame_dragging")
knowledge.save()

# ──────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────

app = FastAPI(
    title="JARVIS-PRIME",
    description="Omnidisciplinary Autonomous Intelligence System - Phase 5 Dashboard",
    version="0.5.0",
)

START_TIME = time.time()


class GoalRequest(BaseModel):
    """Request body for submitting a goal."""
    query: str = Field(..., description="The goal/query to process", min_length=1)
    priority: str = Field("medium", description="Priority: critical, high, medium, low")


class KnowledgeEntry(BaseModel):
    """Request body for adding knowledge."""
    fact_id: str = Field(..., description="Unique fact ID")
    content: str = Field(..., description="Fact content")
    domain: str = Field("general", description="Knowledge domain")


# ──────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the web dashboard."""
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content=_fallback_html())


@app.get("/api/status")
async def get_status():
    """Get full system status."""
    return {
        "system": "JARVIS-PRIME",
        "version": "0.5.0",
        "phase": 5,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "llm_provider": cognitive_core.get_active_provider(),
        "llm_stats": cognitive_core.stats(),
        "nexus": nexus.stats(),
        "knowledge": knowledge.stats(),
    }


@app.post("/api/goal")
async def process_goal(request: GoalRequest):
    """Submit a goal for NEXUS processing."""
    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "medium": TaskPriority.MEDIUM,
        "low": TaskPriority.LOW,
    }
    priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)

    try:
        result = await nexus.process_goal(request.query, priority)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def list_agents():
    """List all registered agents."""
    return {"agents": nexus.list_agents()}


@app.get("/api/memory")
async def memory_stats():
    """Get memory system statistics."""
    return memory.stats()


@app.get("/api/knowledge")
async def knowledge_stats():
    """Get knowledge engine statistics."""
    return knowledge.stats()


@app.post("/api/knowledge")
async def add_knowledge(entry: KnowledgeEntry):
    """Add a fact to the knowledge base."""
    knowledge.add_fact(entry.fact_id, entry.content, entry.domain)
    knowledge.save()
    return {"status": "added", "fact_id": entry.fact_id}


@app.get("/api/knowledge/query")
async def query_knowledge(q: str, top_k: int = 5):
    """Query the knowledge base."""
    results = await knowledge.query(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/api/sica")
async def sica_status():
    """Get SICA self-improvement engine status."""
    return sica.stats()


@app.get("/api/llm")
async def llm_status():
    """Get LLM provider status."""
    return cognitive_core.stats()


@app.get("/api/telemetry")
async def telemetry_status():
    """Get telemetry metrics."""
    return telemetry.stats()


@app.get("/api/auth/users")
async def list_users():
    """List all users."""
    return {"users": auth_manager.list_users()}


@app.get("/api/ontology")
async def ontology_status():
    """Get ontology statistics."""
    return ontology.stats()


@app.get("/api/ontology/query/{concept}")
async def query_ontology(concept: str):
    """Query a concept across all ontologies."""
    return {"concept": concept, "results": ontology.query_across_domains(concept)}


@app.get("/api/reasoning")
async def reasoning_status():
    """Get reasoning engine status."""
    return reasoning.stats()


@app.get("/api/world_model")
async def world_model_status():
    """Get world model status."""
    return world_model.stats()


# ──────────────────────────────────────────────────────
# Fallback HTML (used if web/index.html doesn't exist)
# ──────────────────────────────────────────────────────

def _fallback_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JARVIS-PRIME Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; margin: 0; padding: 20px; }
        h1 { color: #00d4ff; text-align: center; }
        .card { background: #1a1a2e; border: 1px solid #16213e; border-radius: 8px; padding: 20px; margin: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        code { color: #00ff88; }
        a { color: #00d4ff; }
        #output { white-space: pre-wrap; background: #0d0d1a; padding: 15px; border-radius: 8px; max-height: 500px; overflow-y: auto; }
        input, button { padding: 10px; border-radius: 5px; border: 1px solid #333; }
        input { background: #1a1a2e; color: white; width: 60%; }
        button { background: #00d4ff; color: black; cursor: pointer; font-weight: bold; }
        button:hover { background: #00b8d4; }
    </style>
</head>
<body>
    <h1>JARVIS-PRIME v0.2.0</h1>
    <p style="text-align:center">Omnidisciplinary Autonomous Intelligence System</p>
    <div style="text-align:center; margin: 20px;">
        <input id="query" placeholder="Enter a goal..." onkeypress="if(event.key==='Enter')submit()"/>
        <button onclick="submit()">Process Goal</button>
    </div>
    <div id="output">Ready. Enter a goal above or visit <a href="/docs">/docs</a> for the full API.</div>
    <div class="grid" id="cards"></div>
    <script>
        async function submit() {
            const q = document.getElementById('query').value;
            document.getElementById('output').textContent = 'Processing: ' + q + '...';
            try {
                const r = await fetch('/api/goal', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:q})});
                const d = await r.json();
                document.getElementById('output').textContent = JSON.stringify(d, null, 2);
            } catch(e) { document.getElementById('output').textContent = 'Error: ' + e; }
        }
        fetch('/api/status').then(r=>r.json()).then(d=>{
            const cards = document.getElementById('cards');
            cards.innerHTML = '<div class="card"><h3>LLM Provider</h3><code>'+d.llm_provider+'</code></div>'
                + '<div class="card"><h3>Agents</h3><code>'+d.nexus.registered_agents+' registered</code></div>'
                + '<div class="card"><h3>Knowledge</h3><code>'+JSON.stringify(d.knowledge.knowledge_graph)+'</code></div>';
        });
    </script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  JARVIS-PRIME Dashboard v0.5.0")
    print("=" * 60)
    print(f"  LLM Provider: {cognitive_core.get_active_provider()}")
    print(f"  Agents: {len(agents)} registered")
    print(f"  API Docs: http://localhost:8000/docs")
    print(f"  Dashboard: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
