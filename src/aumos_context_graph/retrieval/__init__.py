"""Retrieval package for aumos-context-graph.

Provides hybrid vector+graph retrieval with two-phase search:
1. Vector similarity search (fast approximate nearest neighbor)
2. Graph expansion (follow knowledge graph edges from vector hits)
3. Combined scoring with configurable alpha weighting
"""

from __future__ import annotations

from aumos_context_graph.retrieval.hybrid_retriever import HybridGraphVectorRetriever, RetrievedDocument
from aumos_context_graph.retrieval.protocols import GraphStoreProtocol, VectorStoreProtocol

__all__ = [
    "GraphStoreProtocol",
    "HybridGraphVectorRetriever",
    "RetrievedDocument",
    "VectorStoreProtocol",
]
