"""Ingestion package for aumos-context-graph.

Provides document processing and knowledge graph construction:
- DocumentGraphConstructor: Extracts relationships from documents and builds edges
- GraphEdge: Typed edge representation for the knowledge graph
"""

from __future__ import annotations

from aumos_context_graph.ingestion.graph_constructor import DocumentGraphConstructor, GraphEdge

__all__ = ["DocumentGraphConstructor", "GraphEdge"]
