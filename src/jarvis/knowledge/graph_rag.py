"""
JARVIS-PRIME GraphRAG Knowledge Engine
========================================

Hybrid vector + graph retrieval system.

Architecture:
    ChromaDB (embedded)  →  Semantic vector search (cosine similarity)
    NetworkX (in-memory) →  Knowledge graph traversal (relationships)
    Unified query()      →  Combines both for maximum recall + precision

Phase 2: ChromaDB + NetworkX (no servers, no Docker)
Phase 3+: Upgrade to Neo4j + Weaviate for scale
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import networkx as nx

from jarvis.knowledge.embeddings import EmbeddingEngine


class GraphRAG:
    """
    Hybrid Vector + Graph Knowledge Engine.

    Combines:
    1. ChromaDB for dense vector semantic search
    2. NetworkX for structured relationship traversal
    3. BM25-style keyword boost for exact matches

    Usage:
        engine = GraphRAG(persist_dir=Path("jarvis_data/knowledge"))
        engine.add_fact("F001", "Casimir force scales as 1/d^4", domain="physics")
        engine.add_relationship("Casimir_effect", "produces", "attractive_force")
        results = await engine.query("How does Casimir force scale with distance?")
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "jarvis_knowledge",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = EmbeddingEngine()

        # Initialize ChromaDB
        self._chroma_client = None
        self._collection = None
        self._init_chromadb()

        # Initialize NetworkX knowledge graph
        self.graph = nx.DiGraph()
        self._graph_file = persist_dir / "knowledge_graph.json" if persist_dir else None

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_graph()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB in embedded mode."""
        try:
            import chromadb
            if self.persist_dir:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.persist_dir / "chromadb")
                )
            else:
                self._chroma_client = chromadb.EphemeralClient()

            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            # ChromaDB not installed — use fallback
            self._chroma_client = None
            self._collection = None

    # ──────────────────────────────────────────────────────
    # Data Ingestion
    # ──────────────────────────────────────────────────────

    def add_fact(
        self,
        fact_id: str,
        content: str,
        domain: str = "general",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a fact to both vector store and graph."""
        meta = {
            "domain": domain,
            "importance": str(importance),
            "timestamp": str(time.time()),
            **(metadata or {}),
        }

        if self._collection is not None:
            embedding = self.embedder.embed(content)
            self._collection.upsert(
                ids=[fact_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[meta],
            )

        # Also add as a node in the graph
        self.graph.add_node(
            fact_id,
            content=content,
            domain=domain,
            importance=importance,
            node_type="fact",
        )

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relationship to the knowledge graph."""
        # Ensure nodes exist
        if subject not in self.graph:
            self.graph.add_node(subject, node_type="entity", content=subject)
        if obj not in self.graph:
            self.graph.add_node(obj, node_type="entity", content=obj)

        self.graph.add_edge(
            subject, obj,
            predicate=predicate,
            weight=weight,
        )

    def add_document(
        self,
        doc_id: str,
        text: str,
        domain: str = "general",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> int:
        """
        Ingest a document by chunking and adding each chunk.
        Returns the number of chunks created.
        """
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i:04d}"
            self.add_fact(
                fact_id=chunk_id,
                content=chunk,
                domain=domain,
                metadata={"source_doc": doc_id, "chunk_index": str(i)},
            )
        return len(chunks)

    # ──────────────────────────────────────────────────────
    # Query / Retrieval
    # ──────────────────────────────────────────────────────

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid query combining vector search + graph context.

        Returns ranked results with content, score, and related graph context.
        """
        results = []

        # 1. Vector search via ChromaDB
        if self._collection is not None:
            try:
                embedding = self.embedder.embed(query_text)
                where_filter = {"domain": domain_filter} if domain_filter else None

                chroma_results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    where=where_filter,
                )

                if chroma_results and chroma_results["documents"]:
                    for i, doc in enumerate(chroma_results["documents"][0]):
                        fact_id = chroma_results["ids"][0][i] if chroma_results["ids"] else f"unknown-{i}"
                        meta = chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {}
                        distance = chroma_results["distances"][0][i] if chroma_results["distances"] else 0.5
                        score = 1.0 - distance  # Convert distance to similarity

                        # Enrich with graph context
                        graph_context = self._get_graph_context(fact_id)

                        results.append({
                            "id": fact_id,
                            "content": doc,
                            "score": score,
                            "domain": meta.get("domain", "general"),
                            "importance": float(meta.get("importance", "0.5")),
                            "graph_context": graph_context,
                        })
            except Exception:
                pass

        # 2. Fallback: keyword search on graph nodes
        if not results:
            query_lower = query_text.lower()
            for node, data in self.graph.nodes(data=True):
                content = data.get("content", "")
                if isinstance(content, str):
                    words = set(query_lower.split())
                    node_words = set(content.lower().split())
                    overlap = len(words & node_words)
                    if overlap > 0:
                        results.append({
                            "id": node,
                            "content": content,
                            "score": overlap * 0.1,
                            "domain": data.get("domain", "general"),
                            "importance": data.get("importance", 0.5),
                            "graph_context": self._get_graph_context(node),
                        })

            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

        return results

    def _get_graph_context(self, node_id: str) -> dict[str, Any]:
        """Get relationship context from the knowledge graph for a node."""
        if node_id not in self.graph:
            return {"relationships": [], "neighbors": []}

        relationships = []
        # Outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            target_content = self.graph.nodes[target].get("content", target)
            relationships.append({
                "predicate": data.get("predicate", "related_to"),
                "target": target,
                "target_content": target_content[:100] if isinstance(target_content, str) else str(target),
            })

        # Incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            source_content = self.graph.nodes[source].get("content", source)
            relationships.append({
                "predicate": f"is_{data.get('predicate', 'related_to')}_by",
                "target": source,
                "target_content": source_content[:100] if isinstance(source_content, str) else str(source),
            })

        # 2-hop neighbors for broader context
        neighbors = []
        try:
            for neighbor in nx.single_source_shortest_path_length(self.graph, node_id, cutoff=2):
                if neighbor != node_id:
                    n_data = self.graph.nodes.get(neighbor, {})
                    neighbors.append({
                        "id": neighbor,
                        "domain": n_data.get("domain", "general"),
                    })
        except nx.NetworkXError:
            pass

        return {
            "relationships": relationships[:10],
            "neighbors": neighbors[:10],
        }

    def get_domain_summary(self, domain: str) -> dict[str, Any]:
        """Get a summary of knowledge in a specific domain."""
        domain_nodes = [
            (n, d) for n, d in self.graph.nodes(data=True)
            if d.get("domain") == domain
        ]
        return {
            "domain": domain,
            "fact_count": len(domain_nodes),
            "relationship_count": sum(
                1 for _, _, d in self.graph.edges(data=True)
            ),
            "sample_facts": [
                d.get("content", n)[:100]
                for n, d in domain_nodes[:5]
            ],
        }

    # ──────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the knowledge graph to disk. ChromaDB auto-persists."""
        if self._graph_file:
            data = nx.node_link_data(self.graph)
            self._graph_file.write_text(json.dumps(data, indent=2, default=str))

    def _load_graph(self) -> None:
        """Load the knowledge graph from disk."""
        if self._graph_file and self._graph_file.exists():
            try:
                data = json.loads(self._graph_file.read_text())
                self.graph = nx.node_link_graph(data)
            except Exception:
                self.graph = nx.DiGraph()

    # ──────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def stats(self) -> dict[str, Any]:
        """Get knowledge engine statistics."""
        chroma_count = 0
        if self._collection is not None:
            try:
                chroma_count = self._collection.count()
            except Exception:
                pass

        return {
            "vector_store": {
                "backend": "chromadb",
                "document_count": chroma_count,
                "collection": self.collection_name,
            },
            "knowledge_graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "domains": list(set(
                    d.get("domain", "general")
                    for _, d in self.graph.nodes(data=True)
                    if "domain" in d
                )),
            },
            "embeddings": self.embedder.stats(),
        }
