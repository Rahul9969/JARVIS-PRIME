"""
JARVIS-PRIME Domain Ontology System
=======================================

Provides structured domain knowledge via ontologies:
- Class hierarchies (is-a relationships)
- Properties and constraints
- Cross-domain mappings
- Inference rules

Phase 3: Static ontology definitions
Phase 4+: OWL/RDF integration
"""
from __future__ import annotations

from typing import Any

import networkx as nx


class DomainOntology:
    """
    A domain-specific ontology with class hierarchy and properties.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.graph = nx.DiGraph()
        self._properties: dict[str, dict[str, Any]] = {}

    def add_class(
        self,
        name: str,
        parent: str | None = None,
        description: str = "",
        properties: dict[str, str] | None = None,
    ) -> None:
        """Add a class to the ontology."""
        self.graph.add_node(name, description=description, node_type="class")
        if parent:
            if parent not in self.graph:
                self.graph.add_node(parent, node_type="class")
            self.graph.add_edge(parent, name, relation="has_subclass")

        if properties:
            self._properties[name] = properties

    def add_relation(self, subject: str, predicate: str, obj: str) -> None:
        """Add a relation between concepts."""
        for node in [subject, obj]:
            if node not in self.graph:
                self.graph.add_node(node, node_type="concept")
        self.graph.add_edge(subject, obj, relation=predicate)

    def get_subclasses(self, class_name: str) -> list[str]:
        """Get all subclasses of a class."""
        if class_name not in self.graph:
            return []
        return [
            target for _, target, data in self.graph.edges(class_name, data=True)
            if data.get("relation") == "has_subclass"
        ]

    def get_superclasses(self, class_name: str) -> list[str]:
        """Get all superclasses (ancestors) of a class."""
        ancestors = []
        for source, _, data in self.graph.in_edges(class_name, data=True):
            if data.get("relation") == "has_subclass":
                ancestors.append(source)
                ancestors.extend(self.get_superclasses(source))
        return ancestors

    def get_properties(self, class_name: str) -> dict[str, str]:
        """Get properties of a class including inherited ones."""
        props = dict(self._properties.get(class_name, {}))
        for parent in self.get_superclasses(class_name):
            for k, v in self._properties.get(parent, {}).items():
                if k not in props:
                    props[k] = v
        return props

    def query(self, subject: str | None = None, predicate: str | None = None, obj: str | None = None) -> list[dict[str, str]]:
        """Query the ontology with optional subject/predicate/object filters."""
        results = []
        for s, o, data in self.graph.edges(data=True):
            rel = data.get("relation", "related")
            if subject and s != subject:
                continue
            if predicate and rel != predicate:
                continue
            if obj and o != obj:
                continue
            results.append({"subject": s, "predicate": rel, "object": o})
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "classes": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "top_classes": [
                n for n in self.graph.nodes()
                if self.graph.in_degree(n) == 0
            ][:10],
        }


class OntologyManager:
    """
    Manages multiple domain ontologies and provides cross-domain reasoning.
    """

    def __init__(self):
        self.ontologies: dict[str, DomainOntology] = {}
        self._cross_domain_mappings: list[dict[str, str]] = []

        # Initialize built-in ontologies
        self._init_physics_ontology()
        self._init_ai_ontology()
        self._init_biology_ontology()

    def _init_physics_ontology(self) -> None:
        onto = DomainOntology("physics")

        # Force hierarchy
        onto.add_class("Force", description="Interaction causing acceleration")
        onto.add_class("GravitationalForce", parent="Force", properties={"mediator": "graviton", "range": "infinite"})
        onto.add_class("ElectromagneticForce", parent="Force", properties={"mediator": "photon", "range": "infinite"})
        onto.add_class("StrongForce", parent="Force", properties={"mediator": "gluon", "range": "1e-15 m"})
        onto.add_class("WeakForce", parent="Force", properties={"mediator": "W/Z boson", "range": "1e-18 m"})
        onto.add_class("CasimirForce", parent="ElectromagneticForce", properties={"origin": "vacuum_fluctuations"})

        # Spacetime concepts
        onto.add_class("Spacetime", description="4D manifold of general relativity")
        onto.add_class("Metric", parent="Spacetime")
        onto.add_class("SchwarzschildMetric", parent="Metric")
        onto.add_class("KerrMetric", parent="Metric")
        onto.add_class("AlcubierreMetric", parent="Metric", properties={"requires": "exotic_matter"})

        # Relations
        onto.add_relation("CasimirForce", "arises_from", "VacuumFluctuations")
        onto.add_relation("LenseThirring", "predicts", "FrameDragging")
        onto.add_relation("KerrMetric", "describes", "RotatingBlackHole")
        onto.add_relation("AlcubierreMetric", "enables", "FTLTravel")

        self.ontologies["physics"] = onto

    def _init_ai_ontology(self) -> None:
        onto = DomainOntology("artificial_intelligence")

        onto.add_class("AISystem", description="Artificial intelligence system")
        onto.add_class("NeuralNetwork", parent="AISystem")
        onto.add_class("Transformer", parent="NeuralNetwork")
        onto.add_class("LLM", parent="Transformer", properties={"training": "next_token_prediction"})
        onto.add_class("VisionModel", parent="NeuralNetwork")
        onto.add_class("MultimodalModel", parent="NeuralNetwork")

        onto.add_class("LearningParadigm", description="How AI systems learn")
        onto.add_class("SupervisedLearning", parent="LearningParadigm")
        onto.add_class("ReinforcementLearning", parent="LearningParadigm")
        onto.add_class("SelfSupervisedLearning", parent="LearningParadigm")
        onto.add_class("JEPA", parent="SelfSupervisedLearning", properties={"inventor": "Yann LeCun"})
        onto.add_class("ActiveInference", parent="LearningParadigm", properties={"inventor": "Karl Friston"})

        onto.add_class("AgentArchitecture", description="Multi-agent system")
        onto.add_class("SICA", parent="AgentArchitecture", properties={"capability": "self_improvement"})
        onto.add_class("MCP", parent="AgentArchitecture", properties={"type": "tool_protocol"})
        onto.add_class("A2A", parent="AgentArchitecture", properties={"type": "agent_protocol"})

        onto.add_relation("LLM", "powers", "AgentArchitecture")
        onto.add_relation("SICA", "enables", "SelfImprovement")
        onto.add_relation("JEPA", "models", "WorldUnderstanding")

        self.ontologies["ai"] = onto

    def _init_biology_ontology(self) -> None:
        onto = DomainOntology("biology")

        onto.add_class("Biomolecule")
        onto.add_class("Protein", parent="Biomolecule", properties={"building_block": "amino_acid"})
        onto.add_class("Enzyme", parent="Protein")
        onto.add_class("Antibody", parent="Protein")
        onto.add_class("NucleicAcid", parent="Biomolecule")
        onto.add_class("DNA", parent="NucleicAcid")
        onto.add_class("RNA", parent="NucleicAcid")
        onto.add_class("mRNA", parent="RNA")

        onto.add_class("GeneEditingTool")
        onto.add_class("CRISPR_Cas9", parent="GeneEditingTool", properties={"pam": "NGG"})
        onto.add_class("PrimeEditing", parent="GeneEditingTool")
        onto.add_class("BaseEditing", parent="GeneEditingTool")

        onto.add_relation("CRISPR_Cas9", "edits", "DNA")
        onto.add_relation("DNA", "encodes", "Protein")
        onto.add_relation("mRNA", "translates_to", "Protein")

        self.ontologies["biology"] = onto

    def add_cross_domain_mapping(
        self, domain1: str, concept1: str, domain2: str, concept2: str, relation: str
    ) -> None:
        """Map concepts across domains."""
        self._cross_domain_mappings.append({
            "domain1": domain1, "concept1": concept1,
            "domain2": domain2, "concept2": concept2,
            "relation": relation,
        })

    def get_ontology(self, domain: str) -> DomainOntology | None:
        return self.ontologies.get(domain)

    def query_across_domains(self, concept: str) -> list[dict[str, Any]]:
        """Find a concept across all ontologies."""
        results = []
        for domain, onto in self.ontologies.items():
            if concept in onto.graph:
                results.append({
                    "domain": domain,
                    "concept": concept,
                    "description": onto.graph.nodes[concept].get("description", ""),
                    "properties": onto.get_properties(concept),
                    "subclasses": onto.get_subclasses(concept),
                    "superclasses": onto.get_superclasses(concept),
                })
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "domains": list(self.ontologies.keys()),
            "total_classes": sum(o.graph.number_of_nodes() for o in self.ontologies.values()),
            "total_relations": sum(o.graph.number_of_edges() for o in self.ontologies.values()),
            "cross_domain_mappings": len(self._cross_domain_mappings),
            "per_domain": {
                domain: onto.to_dict()
                for domain, onto in self.ontologies.items()
            },
        }
