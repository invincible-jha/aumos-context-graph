"""Document graph constructor for knowledge graph ingestion.

Extracts structured relationships from raw documents and converts them into
typed graph edges. Supports citation, authorship, hierarchical (part-of),
supersession, and entity co-occurrence relationships.

The extracted edges are intended for ingestion into a GraphStoreProtocol-
compatible backend (Apache AGE, Neo4j, or in-memory test double).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Relationship types supported in the knowledge graph
RelationshipType = Literal["CITED_BY", "AUTHORED_BY", "PART_OF", "SUPERSEDES", "RELATED_TO"]

# Heuristic patterns for lightweight citation extraction
_CITATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(([A-Z][a-z]+ et al\.,? \d{4})\)"),  # (Smith et al., 2020)
    re.compile(r"\[(\d+)\]"),  # [42] — numeric reference
    re.compile(r"\(([A-Z][a-z]+,? \d{4})\)"),  # (Smith, 2020)
]

# Heuristic patterns for author extraction
_AUTHOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:by|author[s]?:|written by)\s+([A-Z][a-z]+ [A-Z][a-z]+)"),
    re.compile(r"([A-Z][a-z]+ [A-Z][a-z]+) et al\."),
]


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph.

    Attributes:
        source_id: Document ID of the edge source.
        target_id: Document or entity ID of the edge target.
        relationship: Typed relationship between source and target.
        weight: Edge weight (0.0 to 1.0). Higher = stronger relationship.
    """

    source_id: str
    target_id: str
    relationship: RelationshipType
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate edge constraints."""
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must be different")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0.0, 1.0], got {self.weight}")


@dataclass
class _ExtractionResult:
    """Internal holder for extraction passes."""

    citations: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    parent_documents: list[str] = field(default_factory=list)
    related_to: list[str] = field(default_factory=list)


class DocumentGraphConstructor:
    """Extracts relationships from documents and constructs knowledge graph edges.

    Processes document metadata and content to infer:
    - Citation edges (CITED_BY) from reference patterns
    - Authorship edges (AUTHORED_BY) from author metadata
    - Hierarchical edges (PART_OF) from parent_document metadata
    - Supersession edges (SUPERSEDES) from version metadata
    - Co-occurrence edges (RELATED_TO) from shared entity tags

    All extraction is heuristic and best-effort. Missing metadata fields are
    handled gracefully — no edges are created if the source information is absent.
    """

    async def process_document(self, document: dict) -> list[GraphEdge]:
        """Extract graph edges from a document dict.

        Inspects both the document metadata and content body to extract
        typed relationships. All edge extraction is additive — a document
        can produce edges of multiple types.

        Expected document dict keys (all optional except 'document_id'):
            document_id (str): Unique identifier for this document. Required.
            content (str): Raw document text for heuristic extraction.
            metadata (dict): Structured metadata with optional subkeys:
                authors (list[str]): Author names or IDs.
                citations (list[str]): Explicitly listed citation IDs.
                parent_document_id (str): Parent document ID for PART_OF edges.
                supersedes (list[str]): Document IDs this document supersedes.
                related_document_ids (list[str]): Explicitly related document IDs.
                entity_tags (list[str]): Shared entity tags for co-occurrence.

        Args:
            document: Document dict with id, content, and metadata.

        Returns:
            List of GraphEdge objects extracted from the document.
            May be empty if no relationships are detectable.

        Raises:
            ValueError: If document_id is missing or empty.
        """
        document_id = document.get("document_id") or document.get("id", "")
        if not document_id:
            raise ValueError("Document must have a non-empty 'document_id' or 'id' field")

        document_id = str(document_id)
        content = str(document.get("content", ""))
        metadata = document.get("metadata") or {}

        logger.info(
            "graph_constructor_processing_document",
            document_id=document_id,
            content_length=len(content),
        )

        extraction = _ExtractionResult()

        # Extract from explicit metadata first (higher confidence)
        self._extract_from_metadata(metadata=metadata, result=extraction)

        # Supplement with heuristic content extraction
        if content:
            self._extract_from_content(content=content, result=extraction)

        edges = self._build_edges(document_id=document_id, extraction=extraction)

        logger.info(
            "graph_constructor_edges_extracted",
            document_id=document_id,
            edge_count=len(edges),
            edge_types=[e.relationship for e in edges],
        )

        return edges

    def _extract_from_metadata(
        self,
        metadata: dict,
        result: _ExtractionResult,
    ) -> None:
        """Extract relationships from structured document metadata.

        Args:
            metadata: Metadata dict from the document.
            result: Mutable extraction result to populate.
        """
        # Authors
        authors = metadata.get("authors", [])
        if isinstance(authors, list):
            result.authors.extend(str(a) for a in authors if a)
        elif isinstance(authors, str) and authors:
            result.authors.append(authors)

        # Explicit citations
        citations = metadata.get("citations", [])
        if isinstance(citations, list):
            result.citations.extend(str(c) for c in citations if c)

        # Parent document (hierarchical membership)
        parent = metadata.get("parent_document_id", "")
        if parent:
            result.parent_documents.append(str(parent))

        # Supersession
        supersedes = metadata.get("supersedes", [])
        if isinstance(supersedes, list):
            result.supersedes.extend(str(s) for s in supersedes if s)
        elif isinstance(supersedes, str) and supersedes:
            result.supersedes.append(supersedes)

        # Explicit related documents
        related = metadata.get("related_document_ids", [])
        if isinstance(related, list):
            result.related_to.extend(str(r) for r in related if r)

    def _extract_from_content(
        self,
        content: str,
        result: _ExtractionResult,
    ) -> None:
        """Extract relationships heuristically from document content.

        Uses regex patterns to find citations and author references in text.

        Args:
            content: Raw document text.
            result: Mutable extraction result to populate.
        """
        # Heuristic citation extraction
        for pattern in _CITATION_PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                # Normalise to a slug-like form for use as an edge target
                citation_id = re.sub(r"[^\w]", "_", match.strip()).lower()
                if citation_id and citation_id not in result.citations:
                    result.citations.append(citation_id)

        # Heuristic author extraction
        for pattern in _AUTHOR_PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                author_id = match.strip()
                if author_id and author_id not in result.authors:
                    result.authors.append(author_id)

    def _build_edges(
        self,
        document_id: str,
        extraction: _ExtractionResult,
    ) -> list[GraphEdge]:
        """Convert extraction results into typed GraphEdge objects.

        Args:
            document_id: Source document ID.
            extraction: Populated extraction result.

        Returns:
            List of deduplicated GraphEdge objects.
        """
        edges: list[GraphEdge] = []
        seen: set[tuple[str, str, str]] = set()

        def add_edge(
            target_id: str,
            relationship: RelationshipType,
            weight: float = 1.0,
        ) -> None:
            """Add an edge if not already seen (deduplication)."""
            key = (document_id, target_id, relationship)
            if key in seen or not target_id or target_id == document_id:
                return
            seen.add(key)
            try:
                edges.append(
                    GraphEdge(
                        source_id=document_id,
                        target_id=target_id,
                        relationship=relationship,
                        weight=weight,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "graph_constructor_invalid_edge",
                    document_id=document_id,
                    target_id=target_id,
                    relationship=relationship,
                    error=str(exc),
                )

        for citation in extraction.citations:
            add_edge(citation, "CITED_BY", weight=0.9)

        for author in extraction.authors:
            add_edge(author, "AUTHORED_BY", weight=1.0)

        for parent in extraction.parent_documents:
            add_edge(parent, "PART_OF", weight=1.0)

        for superseded in extraction.supersedes:
            add_edge(superseded, "SUPERSEDES", weight=0.8)

        for related in extraction.related_to:
            add_edge(related, "RELATED_TO", weight=0.6)

        return edges


__all__ = ["DocumentGraphConstructor", "GraphEdge", "RelationshipType"]
