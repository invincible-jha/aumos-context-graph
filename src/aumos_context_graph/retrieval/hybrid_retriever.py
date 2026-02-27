"""Hybrid graph-vector retriever for RAG pipelines.

Implements a two-phase retrieval strategy:
  Phase 1: Vector similarity search (top-20 candidates)
  Phase 2: Graph expansion via knowledge graph traversal
  Phase 3: Combined re-ranking with configurable alpha weighting
  Phase 4: Deduplication and top-k selection

The combined score formula:
  score = alpha * vector_sim + (1 - alpha) * graph_decay(hops)

Where graph_decay is 1.0 for direct hits, 0.7 for 1-hop, 0.4 for 2-hop,
0.2 for 3-hop, and 0.1 for any deeper hop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from aumos_common.observability import get_logger

from aumos_context_graph.retrieval.protocols import GraphStoreProtocol, VectorStoreProtocol

logger = get_logger(__name__)

# Graph decay factors indexed by hop count (index 0 = direct vector hit)
_GRAPH_DECAY_FACTORS: list[float] = [1.0, 0.7, 0.4, 0.2, 0.1]


@dataclass(frozen=True)
class RetrievedDocument:
    """A document retrieved by the hybrid retriever.

    Attributes:
        document_id: Unique identifier for the document.
        content: Document text content.
        vector_similarity: Cosine similarity score from vector search (0-1).
        graph_distance: Number of graph hops from the nearest vector hit.
            0 = direct vector hit, 1+ = graph-expanded neighbors.
        combined_score: Weighted combination of vector similarity and graph decay.
        relationships: List of relationship type strings that connected this doc.
            Empty for direct vector hits.
    """

    document_id: str
    content: str
    vector_similarity: float
    graph_distance: int
    combined_score: float
    relationships: list[str] = field(default_factory=list)


class HybridGraphVectorRetriever:
    """Two-phase retriever combining vector similarity and graph expansion.

    Retrieves relevant documents by first performing vector similarity search,
    then expanding each hit through the knowledge graph, and finally re-ranking
    all candidates using a weighted combination of vector similarity and graph
    proximity.

    Args:
        vector_store: Backend for vector similarity search.
        graph_store: Backend for knowledge graph traversal.
        alpha: Weight for vector similarity in combined score (0-1).
            Higher alpha favors vector similarity; lower favors graph proximity.
        max_graph_hops: Maximum number of graph hops for expansion.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        graph_store: GraphStoreProtocol,
        alpha: float = 0.6,
        max_graph_hops: int = 2,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            vector_store: Backend for vector similarity search.
            graph_store: Backend for knowledge graph traversal.
            alpha: Weighting for vector similarity (0.0 to 1.0).
            max_graph_hops: Maximum graph hops for expansion (1-3 recommended).
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        if max_graph_hops < 1:
            raise ValueError(f"max_graph_hops must be >= 1, got {max_graph_hops}")

        self._vector_store = vector_store
        self._graph_store = graph_store
        self._alpha = alpha
        self._max_graph_hops = max_graph_hops

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 10,
        graph_hops: int = 2,
        relationship_types: list[str] | None = None,
        vector_threshold: float = 0.0,
    ) -> list[RetrievedDocument]:
        """Retrieve documents using hybrid vector+graph search.

        Phase 1: Vector search for top-20 candidates above threshold.
        Phase 2: Graph expand each result by graph_hops via knowledge graph.
        Phase 3: Re-rank with alpha * vector_sim + (1-alpha) * graph_decay(hops).
        Phase 4: Deduplicate by document_id and return top-k.

        Args:
            query: Original query string (used for logging only).
            query_embedding: Pre-computed query embedding vector.
            k: Number of final results to return.
            graph_hops: Number of graph expansion hops (capped at max_graph_hops).
            relationship_types: Optional filter for graph traversal relationship types.
            vector_threshold: Minimum vector similarity for Phase 1 candidates.

        Returns:
            List of RetrievedDocument ordered by combined_score descending.
        """
        effective_hops = min(graph_hops, self._max_graph_hops)
        vector_candidates = 20  # Always retrieve 20 for Phase 1

        logger.info(
            "hybrid_retrieval_start",
            query_length=len(query),
            k=k,
            graph_hops=effective_hops,
            alpha=self._alpha,
        )

        # Phase 1: Vector similarity search
        vector_results = await self._vector_store.search(
            embedding=query_embedding,
            k=vector_candidates,
            threshold=vector_threshold,
        )

        if not vector_results:
            logger.info("hybrid_retrieval_no_vector_results")
            return []

        # Phase 2: Graph expansion — expand each vector hit in parallel
        expansion_tasks = [
            self._expand_document(
                doc_id=result["document_id"],
                vector_similarity=float(result.get("similarity", 0.0)),
                hops=effective_hops,
                relationship_types=relationship_types,
            )
            for result in vector_results
        ]

        expanded_batches = await asyncio.gather(*expansion_tasks)

        # Phase 3: Re-rank — collect all candidates into a flat dict keyed by document_id
        candidates: dict[str, RetrievedDocument] = {}

        for batch in expanded_batches:
            for doc in batch:
                existing = candidates.get(doc.document_id)
                if existing is None or doc.combined_score > existing.combined_score:
                    candidates[doc.document_id] = doc

        # Phase 4: Deduplicate and return top-k
        ranked = sorted(candidates.values(), key=lambda d: d.combined_score, reverse=True)
        result = ranked[:k]

        logger.info(
            "hybrid_retrieval_complete",
            vector_candidates=len(vector_results),
            total_candidates=len(candidates),
            returned=len(result),
        )

        return result

    async def _expand_document(
        self,
        doc_id: str,
        vector_similarity: float,
        hops: int,
        relationship_types: list[str] | None,
    ) -> list[RetrievedDocument]:
        """Expand a single document through the knowledge graph.

        Creates a direct-hit RetrievedDocument for the source, then queries
        the graph for neighbors and creates RetrievedDocuments for each.

        Args:
            doc_id: Document ID to expand from.
            vector_similarity: Vector similarity score for the source document.
            hops: Number of graph traversal hops.
            relationship_types: Optional relationship type filter.

        Returns:
            List of RetrievedDocuments (source + graph neighbors).
        """
        results: list[RetrievedDocument] = []

        # Direct hit at graph_distance=0
        direct_score = self._combined_score(vector_sim=vector_similarity, graph_dist=0)
        results.append(
            RetrievedDocument(
                document_id=doc_id,
                content="",  # Content populated from vector results upstream
                vector_similarity=vector_similarity,
                graph_distance=0,
                combined_score=direct_score,
                relationships=[],
            )
        )

        # Graph expansion
        try:
            neighbors = await self._graph_store.get_neighbors(
                node_id=doc_id,
                relationship_types=relationship_types,
                max_hops=hops,
            )
        except Exception as exc:
            logger.warning(
                "graph_expansion_failed",
                doc_id=doc_id,
                error=str(exc),
            )
            return results

        for neighbor in neighbors:
            neighbor_doc_id = neighbor.get("document_id", "")
            if not neighbor_doc_id or neighbor_doc_id == doc_id:
                continue

            hop_count = int(neighbor.get("hops", 1))
            relationship = str(neighbor.get("relationship", "RELATED_TO"))

            # Neighbors discovered via graph expansion have lower vector_similarity
            # (we don't have their embedding similarity), so we use 0.0 and rely
            # on graph_decay for their score contribution.
            neighbor_vector_sim = float(neighbor.get("similarity", 0.0))
            neighbor_score = self._combined_score(
                vector_sim=neighbor_vector_sim,
                graph_dist=hop_count,
            )

            results.append(
                RetrievedDocument(
                    document_id=neighbor_doc_id,
                    content=str(neighbor.get("content", "")),
                    vector_similarity=neighbor_vector_sim,
                    graph_distance=hop_count,
                    combined_score=neighbor_score,
                    relationships=[relationship],
                )
            )

        return results

    def _graph_distance_decay(self, hops: int) -> float:
        """Compute graph distance decay factor.

        Returns progressively lower scores for documents reached via more hops:
          0 hops (direct hit): 1.0
          1 hop: 0.7
          2 hops: 0.4
          3 hops: 0.2
          4+ hops: 0.1

        Args:
            hops: Number of graph hops from nearest vector hit.

        Returns:
            Decay factor in range (0, 1].
        """
        if hops < 0:
            return 0.0
        if hops >= len(_GRAPH_DECAY_FACTORS):
            return _GRAPH_DECAY_FACTORS[-1]
        return _GRAPH_DECAY_FACTORS[hops]

    def _combined_score(self, vector_sim: float, graph_dist: int) -> float:
        """Compute the combined ranking score.

        Formula: alpha * vector_sim + (1 - alpha) * graph_decay(graph_dist)

        Args:
            vector_sim: Vector cosine similarity (0-1).
            graph_dist: Number of graph hops (0 = direct hit).

        Returns:
            Combined score in range [0, 1].
        """
        graph_decay = self._graph_distance_decay(graph_dist)
        return self._alpha * vector_sim + (1.0 - self._alpha) * graph_decay


__all__ = ["HybridGraphVectorRetriever", "RetrievedDocument"]
