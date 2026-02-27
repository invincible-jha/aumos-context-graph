"""Retrieval API routes for hybrid graph-vector search and document ingestion.

Provides three endpoints:
  POST /api/v1/retrieval/hybrid      - Hybrid vector+graph retrieval
  POST /api/v1/retrieval/graph-expand - Expand a document via the knowledge graph
  POST /api/v1/ingestion/documents    - Ingest a document and build graph edges

All routes use Annotated[TenantContext, Depends(get_current_tenant)] for tenant
isolation — body-supplied tenant_id is never trusted.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from aumos_common.auth import TenantContext, get_current_user
from aumos_common.observability import get_logger

from aumos_context_graph.ingestion.graph_constructor import DocumentGraphConstructor, GraphEdge
from aumos_context_graph.retrieval.hybrid_retriever import HybridGraphVectorRetriever, RetrievedDocument
from aumos_context_graph.retrieval.protocols import GraphStoreProtocol, VectorStoreProtocol

logger = get_logger(__name__)

retrieval_router = APIRouter(prefix="/retrieval", tags=["retrieval"])
ingestion_router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class HybridRetrievalRequest(BaseModel):
    """Request body for hybrid vector+graph retrieval."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1, max_length=4096, description="Natural language query")
    query_embedding: list[float] = Field(
        ...,
        min_length=1,
        description="Pre-computed query embedding vector",
    )
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    graph_hops: int = Field(default=2, ge=1, le=4, description="Graph expansion depth")
    alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity vs graph proximity",
    )
    relationship_types: list[str] | None = Field(
        default=None,
        description="Optional filter for graph relationship types",
    )
    vector_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity score",
    )


class RetrievedDocumentResponse(BaseModel):
    """Response schema for a single retrieved document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    content: str
    vector_similarity: float
    graph_distance: int
    combined_score: float
    relationships: list[str]


class HybridRetrievalResponse(BaseModel):
    """Response for hybrid retrieval endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: list[RetrievedDocumentResponse]
    total_returned: int
    alpha_used: float
    graph_hops_used: int


class GraphExpandRequest(BaseModel):
    """Request body for graph expansion from a seed document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., min_length=1, description="Seed document ID to expand from")
    max_hops: int = Field(default=2, ge=1, le=4, description="Maximum graph traversal hops")
    relationship_types: list[str] | None = Field(
        default=None,
        description="Optional relationship type filter",
    )


class GraphExpandResponse(BaseModel):
    """Response for graph expansion endpoint."""

    model_config = ConfigDict(frozen=True)

    seed_document_id: str
    neighbors: list[dict[str, Any]]
    total_neighbors: int
    hops_explored: int


class IngestDocumentRequest(BaseModel):
    """Request body for document ingestion with graph edge construction."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., min_length=1, description="Unique document identifier")
    content: str = Field(default="", description="Document text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class GraphEdgeResponse(BaseModel):
    """Response schema for a single extracted graph edge."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    relationship: str
    weight: float


class IngestDocumentResponse(BaseModel):
    """Response for document ingestion endpoint."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    edges_extracted: list[GraphEdgeResponse]
    total_edges: int


# ---------------------------------------------------------------------------
# Dependency stubs — these resolve real adapters in production via
# dependency injection. In tests, they are overridden via app.dependency_overrides.
# ---------------------------------------------------------------------------


async def _get_vector_store() -> VectorStoreProtocol:
    """Provide the vector store adapter.

    Override via app.dependency_overrides in production and tests.
    """
    raise NotImplementedError("Vector store adapter not configured — override this dependency")


async def _get_graph_store() -> GraphStoreProtocol:
    """Provide the graph store adapter.

    Override via app.dependency_overrides in production and tests.
    """
    raise NotImplementedError("Graph store adapter not configured — override this dependency")


# ---------------------------------------------------------------------------
# Retrieval endpoints
# ---------------------------------------------------------------------------


@retrieval_router.post(
    "/hybrid",
    response_model=HybridRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid graph-vector retrieval",
    description=(
        "Performs two-phase retrieval: vector similarity search followed by "
        "knowledge graph expansion, then re-ranks results with configurable "
        "alpha weighting."
    ),
)
async def hybrid_retrieval(
    request: HybridRetrievalRequest,
    tenant: Annotated[TenantContext, Depends(get_current_user)],
    vector_store: Annotated[VectorStoreProtocol, Depends(_get_vector_store)],
    graph_store: Annotated[GraphStoreProtocol, Depends(_get_graph_store)],
) -> HybridRetrievalResponse:
    """Hybrid vector+graph retrieval endpoint.

    Args:
        request: Retrieval request with query, embedding, and parameters.
        tenant: Authenticated tenant context (injected — not from body).
        vector_store: Vector similarity backend.
        graph_store: Knowledge graph backend.

    Returns:
        HybridRetrievalResponse with ranked documents.
    """
    logger.info(
        "hybrid_retrieval_request",
        tenant_id=str(tenant.tenant_id),
        k=request.k,
        graph_hops=request.graph_hops,
        alpha=request.alpha,
    )

    retriever = HybridGraphVectorRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        alpha=request.alpha,
        max_graph_hops=request.graph_hops,
    )

    retrieved: list[RetrievedDocument] = await retriever.retrieve(
        query=request.query,
        query_embedding=request.query_embedding,
        k=request.k,
        graph_hops=request.graph_hops,
        relationship_types=request.relationship_types,
        vector_threshold=request.vector_threshold,
    )

    results = [
        RetrievedDocumentResponse(
            document_id=doc.document_id,
            content=doc.content,
            vector_similarity=doc.vector_similarity,
            graph_distance=doc.graph_distance,
            combined_score=doc.combined_score,
            relationships=doc.relationships,
        )
        for doc in retrieved
    ]

    return HybridRetrievalResponse(
        query=request.query,
        results=results,
        total_returned=len(results),
        alpha_used=request.alpha,
        graph_hops_used=request.graph_hops,
    )


@retrieval_router.post(
    "/graph-expand",
    response_model=GraphExpandResponse,
    status_code=status.HTTP_200_OK,
    summary="Expand a document via the knowledge graph",
    description=(
        "Traverses the knowledge graph from a seed document and returns all "
        "neighboring documents within the specified hop count."
    ),
)
async def graph_expand(
    request: GraphExpandRequest,
    tenant: Annotated[TenantContext, Depends(get_current_user)],
    graph_store: Annotated[GraphStoreProtocol, Depends(_get_graph_store)],
) -> GraphExpandResponse:
    """Knowledge graph expansion from a seed document.

    Args:
        request: Graph expansion request with document_id and hop count.
        tenant: Authenticated tenant context (injected — not from body).
        graph_store: Knowledge graph backend.

    Returns:
        GraphExpandResponse with neighboring documents.
    """
    logger.info(
        "graph_expand_request",
        document_id=request.document_id,
        max_hops=request.max_hops,
        tenant_id=str(tenant.tenant_id),
    )

    try:
        neighbors = await graph_store.get_neighbors(
            node_id=request.document_id,
            relationship_types=request.relationship_types,
            max_hops=request.max_hops,
        )
    except Exception as exc:
        logger.error(
            "graph_expand_error",
            document_id=request.document_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Graph store query failed: {exc}",
        ) from exc

    return GraphExpandResponse(
        seed_document_id=request.document_id,
        neighbors=neighbors,
        total_neighbors=len(neighbors),
        hops_explored=request.max_hops,
    )


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------


@ingestion_router.post(
    "/documents",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest document and build graph edges",
    description=(
        "Processes a document to extract graph edges (citations, authorship, "
        "hierarchy, supersession, and co-occurrence relationships)."
    ),
)
async def ingest_document(
    request: IngestDocumentRequest,
    tenant: Annotated[TenantContext, Depends(get_current_user)],
) -> IngestDocumentResponse:
    """Document ingestion with knowledge graph edge construction.

    Args:
        request: Document ingestion request with content and metadata.
        tenant: Authenticated tenant context (injected — not from body).

    Returns:
        IngestDocumentResponse with extracted graph edges.
    """
    logger.info(
        "ingest_document_request",
        document_id=request.document_id,
        tenant_id=str(tenant.tenant_id),
    )

    constructor = DocumentGraphConstructor()

    try:
        edges: list[GraphEdge] = await constructor.process_document(
            document={
                "document_id": request.document_id,
                "content": request.content,
                "metadata": request.metadata,
            }
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    edge_responses = [
        GraphEdgeResponse(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relationship=edge.relationship,
            weight=edge.weight,
        )
        for edge in edges
    ]

    return IngestDocumentResponse(
        document_id=request.document_id,
        edges_extracted=edge_responses,
        total_edges=len(edge_responses),
    )


__all__ = ["ingestion_router", "retrieval_router"]
