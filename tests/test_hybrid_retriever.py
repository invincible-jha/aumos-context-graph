"""Tests for HybridGraphVectorRetriever and DocumentGraphConstructor.

These tests run without aumos_common installed. The conftest uses sys.path
insertion to make the source importable, and all aumos_common references are
patched before the module is imported.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.path setup — run without installing the package
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---------------------------------------------------------------------------
# Stub aumos_common so the modules can be imported without it installed
# ---------------------------------------------------------------------------
_aumos_common = types.ModuleType("aumos_common")
_aumos_common_observability = types.ModuleType("aumos_common.observability")
_aumos_common_observability.get_logger = lambda name: MagicMock()
_aumos_common.observability = _aumos_common_observability
sys.modules.setdefault("aumos_common", _aumos_common)
sys.modules.setdefault("aumos_common.observability", _aumos_common_observability)

from aumos_context_graph.retrieval.hybrid_retriever import (  # noqa: E402
    HybridGraphVectorRetriever,
    RetrievedDocument,
    _GRAPH_DECAY_FACTORS,
)
from aumos_context_graph.ingestion.graph_constructor import (  # noqa: E402
    DocumentGraphConstructor,
    GraphEdge,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

class _FakeVectorStore:
    """In-memory vector store for testing."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results

    async def search(
        self,
        embedding: list[float],
        k: int,
        threshold: float = 0.0,
    ) -> list[dict]:
        return [r for r in self._results if r.get("similarity", 0.0) >= threshold][:k]


class _FakeGraphStore:
    """In-memory graph store for testing."""

    def __init__(self, neighbors: dict[str, list[dict]]) -> None:
        self._neighbors = neighbors

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None,
        max_hops: int,
    ) -> list[dict]:
        return self._neighbors.get(node_id, [])


class _FailingGraphStore:
    """Graph store that always raises on get_neighbors."""

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None,
        max_hops: int,
    ) -> list[dict]:
        raise RuntimeError("Graph DB unavailable")


# ---------------------------------------------------------------------------
# Combined score calculation
# ---------------------------------------------------------------------------


class TestCombinedScore:
    def _retriever(self, alpha: float = 0.6) -> HybridGraphVectorRetriever:
        return HybridGraphVectorRetriever(
            vector_store=_FakeVectorStore([]),
            graph_store=_FakeGraphStore({}),
            alpha=alpha,
        )

    def test_direct_hit_uses_full_graph_decay(self) -> None:
        r = self._retriever(alpha=0.6)
        score = r._combined_score(1.0, 0)
        assert score == pytest.approx(0.6 * 1.0 + 0.4 * 1.0)

    def test_one_hop_applies_decay_07(self) -> None:
        r = self._retriever(alpha=0.6)
        score = r._combined_score(0.8, 1)
        assert score == pytest.approx(0.6 * 0.8 + 0.4 * 0.7)

    def test_two_hop_applies_decay_04(self) -> None:
        r = self._retriever(alpha=0.6)
        score = r._combined_score(0.5, 2)
        assert score == pytest.approx(0.6 * 0.5 + 0.4 * 0.4)

    def test_three_hop_applies_decay_02(self) -> None:
        r = self._retriever(alpha=0.6)
        score = r._combined_score(0.0, 3)
        assert score == pytest.approx(0.6 * 0.0 + 0.4 * 0.2)

    def test_beyond_max_hop_clamps_to_last_factor(self) -> None:
        r = self._retriever(alpha=0.6)
        score_10 = r._combined_score(0.0, 10)
        # 10 hops should clamp to the last factor (0.1)
        last_factor = _GRAPH_DECAY_FACTORS[-1]
        assert score_10 == pytest.approx(0.4 * last_factor)

    def test_alpha_zero_ignores_vector(self) -> None:
        r = self._retriever(alpha=0.0)
        score = r._combined_score(1.0, 0)
        assert score == pytest.approx(1.0 * 1.0)  # beta=1.0, decay=1.0 at hop 0

    def test_alpha_one_ignores_graph(self) -> None:
        r = self._retriever(alpha=1.0)
        score = r._combined_score(0.75, 2)
        assert score == pytest.approx(0.75)

    def test_custom_alpha_beta_tenant_config(self) -> None:
        """Custom alpha=0.3 beta=0.7 per tenant should weight graph more."""
        r = self._retriever(alpha=0.3)
        score = r._combined_score(0.5, 0)
        assert score == pytest.approx(0.3 * 0.5 + 0.7 * 1.0)


# ---------------------------------------------------------------------------
# Graph distance decay
# ---------------------------------------------------------------------------


class TestGraphDecay:
    def _retriever(self) -> HybridGraphVectorRetriever:
        return HybridGraphVectorRetriever(
            vector_store=_FakeVectorStore([]),
            graph_store=_FakeGraphStore({}),
        )

    def test_decay_at_zero_hops(self) -> None:
        r = self._retriever()
        assert r._graph_distance_decay(0) == 1.0

    def test_decay_at_one_hop(self) -> None:
        r = self._retriever()
        assert r._graph_distance_decay(1) == 0.7

    def test_decay_at_two_hops(self) -> None:
        r = self._retriever()
        assert r._graph_distance_decay(2) == 0.4

    def test_decay_at_three_hops(self) -> None:
        r = self._retriever()
        assert r._graph_distance_decay(3) == 0.2

    def test_negative_hops_returns_zero(self) -> None:
        r = self._retriever()
        assert r._graph_distance_decay(-1) == 0.0


# ---------------------------------------------------------------------------
# Retriever construction validation
# ---------------------------------------------------------------------------


class TestRetrieverConstruction:
    def test_invalid_alpha_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            HybridGraphVectorRetriever(
                vector_store=_FakeVectorStore([]),
                graph_store=_FakeGraphStore({}),
                alpha=1.5,
            )

    def test_invalid_alpha_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            HybridGraphVectorRetriever(
                vector_store=_FakeVectorStore([]),
                graph_store=_FakeGraphStore({}),
                alpha=-0.1,
            )

    def test_invalid_max_hops_raises(self) -> None:
        with pytest.raises(ValueError, match="max_graph_hops"):
            HybridGraphVectorRetriever(
                vector_store=_FakeVectorStore([]),
                graph_store=_FakeGraphStore({}),
                max_graph_hops=0,
            )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeduplication:
    async def test_document_found_by_both_vector_and_graph_deduped(self) -> None:
        """A doc appearing in vector results and as a graph neighbor must appear once."""
        vector_store = _FakeVectorStore(
            [
                {"document_id": "doc-a", "content": "Content A", "similarity": 0.9},
                {"document_id": "doc-b", "content": "Content B", "similarity": 0.8},
            ]
        )
        # doc-b is also a 1-hop neighbor of doc-a
        graph_store = _FakeGraphStore(
            {
                "doc-a": [
                    {
                        "document_id": "doc-b",
                        "hops": 1,
                        "relationship": "CITED_BY",
                        "content": "Content B",
                        "similarity": 0.0,
                    }
                ],
                "doc-b": [],
            }
        )
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test",
            query_embedding=[0.0] * 4,
            k=10,
        )
        ids = [r.document_id for r in results]
        assert ids.count("doc-b") == 1, "doc-b should appear exactly once"

    async def test_dedup_keeps_highest_combined_score(self) -> None:
        """When doc appears via vector (hop=0) and graph (hop=2), keep higher score."""
        vector_store = _FakeVectorStore(
            [
                {"document_id": "doc-a", "content": "A", "similarity": 0.95},
                {"document_id": "doc-b", "content": "B", "similarity": 0.70},
            ]
        )
        # doc-b also reachable from doc-a at 1 hop with lower score
        graph_store = _FakeGraphStore(
            {
                "doc-a": [
                    {
                        "document_id": "doc-b",
                        "hops": 1,
                        "relationship": "RELATED_TO",
                        "content": "B",
                        "similarity": 0.0,
                    }
                ],
                "doc-b": [],
            }
        )
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        doc_b = next(r for r in results if r.document_id == "doc-b")
        # doc-b direct-hit score: 0.6*0.70 + 0.4*1.0 = 0.82
        # doc-b graph-hit score:  0.6*0.00 + 0.4*0.7 = 0.28
        assert doc_b.combined_score == pytest.approx(0.6 * 0.70 + 0.4 * 1.0)


# ---------------------------------------------------------------------------
# Re-ranking order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReRanking:
    async def test_results_ordered_by_combined_score_descending(self) -> None:
        vector_store = _FakeVectorStore(
            [
                {"document_id": "low", "content": "Low", "similarity": 0.5},
                {"document_id": "high", "content": "High", "similarity": 0.9},
                {"document_id": "mid", "content": "Mid", "similarity": 0.7},
            ]
        )
        graph_store = _FakeGraphStore({"low": [], "high": [], "mid": []})
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_respected(self) -> None:
        vector_store = _FakeVectorStore(
            [
                {"document_id": f"doc-{i}", "content": f"Content {i}", "similarity": 0.9 - i * 0.05}
                for i in range(10)
            ]
        )
        graph_store = _FakeGraphStore({f"doc-{i}": [] for i in range(10)})
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=3
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Vector-only results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVectorOnlyResults:
    async def test_empty_graph_returns_vector_only(self) -> None:
        """When graph returns nothing, retriever should still return vector hits."""
        vector_store = _FakeVectorStore(
            [{"document_id": "doc-x", "content": "X", "similarity": 0.8}]
        )
        graph_store = _FakeGraphStore({})  # no neighbors
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        assert len(results) == 1
        assert results[0].document_id == "doc-x"

    async def test_graph_failure_falls_back_to_vector(self) -> None:
        """Graph store that raises should degrade gracefully to vector-only results."""
        vector_store = _FakeVectorStore(
            [{"document_id": "doc-y", "content": "Y", "similarity": 0.75}]
        )
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store,
            graph_store=_FailingGraphStore(),
            alpha=0.6,
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        assert len(results) == 1
        assert results[0].document_id == "doc-y"

    async def test_empty_vector_results_returns_empty(self) -> None:
        vector_store = _FakeVectorStore([])
        graph_store = _FakeGraphStore({})
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        assert results == []


# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGraphExpansion:
    async def test_graph_expansion_adds_related_documents(self) -> None:
        vector_store = _FakeVectorStore(
            [{"document_id": "seed", "content": "Seed", "similarity": 0.9}]
        )
        graph_store = _FakeGraphStore(
            {
                "seed": [
                    {
                        "document_id": "related-1",
                        "hops": 1,
                        "relationship": "CITED_BY",
                        "content": "Related 1",
                        "similarity": 0.0,
                    },
                    {
                        "document_id": "related-2",
                        "hops": 2,
                        "relationship": "AUTHORED_BY",
                        "content": "Related 2",
                        "similarity": 0.0,
                    },
                ]
            }
        )
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        ids = {r.document_id for r in results}
        assert "related-1" in ids
        assert "related-2" in ids

    async def test_graph_distance_propagated_correctly(self) -> None:
        vector_store = _FakeVectorStore(
            [{"document_id": "seed", "content": "Seed", "similarity": 0.9}]
        )
        graph_store = _FakeGraphStore(
            {
                "seed": [
                    {
                        "document_id": "hop1",
                        "hops": 1,
                        "relationship": "CITED_BY",
                        "content": "Hop 1",
                        "similarity": 0.0,
                    }
                ]
            }
        )
        retriever = HybridGraphVectorRetriever(
            vector_store=vector_store, graph_store=graph_store, alpha=0.6
        )
        results = await retriever.retrieve(
            query="test", query_embedding=[0.0] * 4, k=10
        )
        hop1 = next(r for r in results if r.document_id == "hop1")
        assert hop1.graph_distance == 1
        assert hop1.relationships == ["CITED_BY"]


# ---------------------------------------------------------------------------
# RetrievedDocument immutability
# ---------------------------------------------------------------------------


class TestRetrievedDocumentImmutability:
    def test_frozen_dataclass_cannot_be_mutated(self) -> None:
        doc = RetrievedDocument(
            document_id="x",
            content="content",
            vector_similarity=0.9,
            graph_distance=0,
            combined_score=0.9,
            relationships=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            doc.document_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCitationExtraction:
    async def test_numeric_citation_extracted(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-1",
            "content": "See [42] for details and also [7] for more.",
        }
        edges = await constructor.process_document(doc)
        citation_targets = {e.target_id for e in edges if e.relationship == "CITED_BY"}
        assert len(citation_targets) >= 2

    async def test_author_citation_extracted(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-2",
            "content": "As shown by (Smith et al., 2020) and (Jones, 2019).",
        }
        edges = await constructor.process_document(doc)
        cited = [e for e in edges if e.relationship == "CITED_BY"]
        assert len(cited) >= 2

    async def test_no_citations_produces_no_cited_by_edges(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-3",
            "content": "This document has no citations at all.",
        }
        edges = await constructor.process_document(doc)
        cited = [e for e in edges if e.relationship == "CITED_BY"]
        assert cited == []

    async def test_metadata_citations_take_priority(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-4",
            "content": "",
            "metadata": {"citations": ["ref-001", "ref-002"]},
        }
        edges = await constructor.process_document(doc)
        cited = {e.target_id for e in edges if e.relationship == "CITED_BY"}
        assert "ref-001" in cited
        assert "ref-002" in cited


# ---------------------------------------------------------------------------
# Entity co-occurrence / entity extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEntityExtraction:
    async def test_author_extracted_from_text(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-5",
            "content": "Written by John Smith, a leading researcher.",
        }
        edges = await constructor.process_document(doc)
        authored = [e for e in edges if e.relationship == "AUTHORED_BY"]
        assert len(authored) >= 1
        targets = {e.target_id for e in authored}
        assert "John Smith" in targets

    async def test_metadata_authors_produce_authored_by_edges(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "doc-6",
            "content": "",
            "metadata": {"authors": ["Alice Brown", "Bob Johnson"]},
        }
        edges = await constructor.process_document(doc)
        authored = {e.target_id for e in edges if e.relationship == "AUTHORED_BY"}
        assert "Alice Brown" in authored
        assert "Bob Johnson" in authored

    async def test_part_of_edge_created_from_parent_document(self) -> None:
        constructor = DocumentGraphConstructor()
        doc = {
            "document_id": "chapter-3",
            "content": "",
            "metadata": {"parent_document_id": "book-1"},
        }
        edges = await constructor.process_document(doc)
        part_of = [e for e in edges if e.relationship == "PART_OF"]
        assert len(part_of) == 1
        assert part_of[0].target_id == "book-1"


# ---------------------------------------------------------------------------
# GraphMutation / GraphEdge validation
# ---------------------------------------------------------------------------


class TestGraphEdgeValidation:
    def test_graph_edge_created_successfully(self) -> None:
        edge = GraphEdge(
            source_id="doc-a",
            target_id="doc-b",
            relationship="CITED_BY",
            weight=0.9,
        )
        assert edge.source_id == "doc-a"
        assert edge.relationship == "CITED_BY"

    def test_graph_edge_same_source_target_raises(self) -> None:
        with pytest.raises(ValueError, match="different"):
            GraphEdge(
                source_id="same",
                target_id="same",
                relationship="RELATED_TO",
            )

    def test_graph_edge_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            GraphEdge(source_id="", target_id="doc-b", relationship="RELATED_TO")

    def test_graph_edge_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            GraphEdge(
                source_id="doc-a",
                target_id="doc-b",
                relationship="RELATED_TO",
                weight=1.5,
            )

    def test_supersedes_edge_created(self) -> None:
        edge = GraphEdge(
            source_id="v2",
            target_id="v1",
            relationship="SUPERSEDES",
            weight=0.8,
        )
        assert edge.relationship == "SUPERSEDES"
