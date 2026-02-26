"""API router for aumos-context-graph.

All endpoints are registered here and included in main.py under /api/v1.
Routes delegate all logic to service layer — no business logic in routes.

Routes:
  POST   /context/documents              - Ingest document
  GET    /context/documents/{id}         - Document metadata
  POST   /context/graph/query            - Cypher graph query
  POST   /context/graph/entities         - Create entity
  GET    /context/graph/entities/{id}    - Entity with relationships
  POST   /context/rag/query              - RAG query
  GET    /context/rag/sources            - Sources for response
  POST   /context/search                 - Unified search
  POST   /context/ontology               - Create ontology
  GET    /context/ontology/{id}          - Ontology detail
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_user
from aumos_common.database import get_db_session

from aumos_context_graph.adapters.embedding_engine import OpenAIEmbeddingEngine
from aumos_context_graph.adapters.graph_client import AGEGraphClient
from aumos_context_graph.adapters.repositories import (
    DocumentRepository,
    EntityRepository,
    OntologyRepository,
    SearchLogRepository,
)
from aumos_context_graph.adapters.vector_store import PgVectorStore
from aumos_context_graph.api.schemas import (
    CreateEntityRequest,
    CreateOntologyRequest,
    DocumentDetailResponse,
    DocumentResponse,
    EntityDetailResponse,
    EntityResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    IngestDocumentRequest,
    OntologyResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSourcesResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from aumos_context_graph.core.services import (
    DocumentService,
    EntityResolverService,
    GraphService,
    OntologyService,
    RAGService,
    VectorSearchService,
)
from aumos_context_graph.settings import Settings

router = APIRouter(tags=["context-graph"])
settings = Settings()


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_document_service(session: AsyncSession = Depends(get_db_session)) -> DocumentService:
    """Construct DocumentService with injected dependencies."""
    document_repo = DocumentRepository(session)
    embedding_engine = OpenAIEmbeddingEngine(settings)
    return DocumentService(document_repo, embedding_engine)


def get_graph_service(session: AsyncSession = Depends(get_db_session)) -> GraphService:
    """Construct GraphService with injected dependencies."""
    entity_repo = EntityRepository(session)
    graph_client = AGEGraphClient(session)
    embedding_engine = OpenAIEmbeddingEngine(settings)
    return GraphService(entity_repo, graph_client, embedding_engine)


def get_search_service(session: AsyncSession = Depends(get_db_session)) -> VectorSearchService:
    """Construct VectorSearchService with injected dependencies."""
    vector_store = PgVectorStore(session)
    embedding_engine = OpenAIEmbeddingEngine(settings)
    search_log_repo = SearchLogRepository(session)
    return VectorSearchService(vector_store, embedding_engine, search_log_repo)


def get_rag_service(session: AsyncSession = Depends(get_db_session)) -> RAGService:
    """Construct RAGService with injected dependencies."""
    vector_store = PgVectorStore(session)
    embedding_engine = OpenAIEmbeddingEngine(settings)
    document_repo = DocumentRepository(session)
    return RAGService(vector_store, embedding_engine, document_repo)


def get_ontology_service(session: AsyncSession = Depends(get_db_session)) -> OntologyService:
    """Construct OntologyService with injected dependencies."""
    ontology_repo = OntologyRepository(session)
    return OntologyService(ontology_repo)


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


@router.post("/context/documents", response_model=DocumentResponse, status_code=201)
async def ingest_document(
    request: IngestDocumentRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Ingest a document and index it for search and RAG.

    Deduplicates by content hash. Returns existing record if already indexed.
    """
    document = await service.ingest_document(
        title=request.title,
        content=request.content,
        tenant=tenant,
        document_type=request.document_type,
        source_uri=request.source_uri,
        extra_metadata=request.extra_metadata,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return DocumentResponse(
        id=document.id,
        title=document.title,
        source_uri=document.source_uri,
        document_type=document.document_type,
        status=document.status,
        chunk_count=document.chunk_count,
        token_count=document.token_count,
        size_bytes=document.size_bytes,
        language=document.language,
        extra_metadata=document.extra_metadata,
        created_at=document.created_at.isoformat(),
        updated_at=document.updated_at.isoformat(),
    )


@router.get("/context/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentDetailResponse:
    """Retrieve document metadata and chunk list by ID."""
    document = await service.get_document(document_id, tenant)
    return DocumentDetailResponse(
        id=document.id,
        title=document.title,
        source_uri=document.source_uri,
        document_type=document.document_type,
        status=document.status,
        chunk_count=document.chunk_count,
        token_count=document.token_count,
        size_bytes=document.size_bytes,
        language=document.language,
        extra_metadata=document.extra_metadata,
        created_at=document.created_at.isoformat(),
        updated_at=document.updated_at.isoformat(),
        chunks=[],
    )


# ---------------------------------------------------------------------------
# Knowledge graph endpoints
# ---------------------------------------------------------------------------


@router.post("/context/graph/query", response_model=GraphQueryResponse)
async def execute_graph_query(
    request: GraphQueryRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: GraphService = Depends(get_graph_service),
) -> GraphQueryResponse:
    """Execute a Cypher query against the tenant-scoped knowledge graph."""
    results = await service.execute_graph_query(
        cypher_query=request.query,
        parameters=request.parameters,
        tenant=tenant,
    )
    return GraphQueryResponse(
        results=results,
        result_count=len(results),
        query=request.query,
    )


@router.post("/context/graph/entities", response_model=EntityResponse, status_code=201)
async def create_entity(
    request: CreateEntityRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: GraphService = Depends(get_graph_service),
) -> EntityResponse:
    """Create a new knowledge graph entity with embedding."""
    entity = await service.create_entity(
        name=request.name,
        entity_type=request.entity_type,
        tenant=tenant,
        description=request.description,
        aliases=request.aliases,
        properties=request.properties,
        ontology_id=request.ontology_id,
    )
    return EntityResponse(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        description=entity.description,
        aliases=entity.aliases,
        properties=entity.properties,
        ontology_id=entity.ontology_id,
        confidence_score=entity.confidence_score,
        created_at=entity.created_at.isoformat(),
        updated_at=entity.updated_at.isoformat(),
    )


@router.get("/context/graph/entities/{entity_id}", response_model=EntityDetailResponse)
async def get_entity(
    entity_id: uuid.UUID,
    depth: int = Query(default=1, ge=1, le=3),
    tenant: TenantContext = Depends(get_current_user),
    service: GraphService = Depends(get_graph_service),
) -> EntityDetailResponse:
    """Retrieve an entity with its relationships up to the specified depth."""
    entity = await service.get_entity_with_relationships(entity_id, tenant, depth)
    return EntityDetailResponse(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        description=entity.description,
        aliases=entity.aliases,
        properties=entity.properties,
        ontology_id=entity.ontology_id,
        confidence_score=entity.confidence_score,
        created_at=entity.created_at.isoformat(),
        updated_at=entity.updated_at.isoformat(),
        source_relationships=[],
        target_relationships=[],
    )


# ---------------------------------------------------------------------------
# RAG endpoints
# ---------------------------------------------------------------------------


@router.post("/context/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Perform retrieval-augmented generation — retrieve context and construct prompt."""
    result = await service.query(
        question=request.question,
        tenant=tenant,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_ids=request.document_ids,
        max_tokens=request.max_tokens,
    )
    from aumos_context_graph.api.schemas import RAGChunkContext, RAGSourceReference

    return RAGQueryResponse(
        question=result["question"],
        context_chunks=[
            RAGChunkContext(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                similarity_score=chunk["score"],
            )
            for chunk in result["context_chunks"]
        ],
        prompt=result["prompt"],
        sources=[
            RAGSourceReference(
                chunk_id=source["chunk_id"],
                document_id=source["document_id"],
                chunk_index=source["chunk_index"],
                similarity_score=source["similarity_score"],
                section_title=source.get("section_title"),
            )
            for source in result["sources"]
        ],
        context_count=len(result["context_chunks"]),
    )


@router.get("/context/rag/sources", response_model=RAGSourcesResponse)
async def get_rag_sources(
    chunk_ids: list[uuid.UUID] = Query(...),
    tenant: TenantContext = Depends(get_current_user),
    service: RAGService = Depends(get_rag_service),
) -> RAGSourcesResponse:
    """Retrieve source document metadata for a list of chunk IDs."""
    sources = await service.get_sources(chunk_ids, tenant)
    return RAGSourcesResponse(sources=sources, total=len(sources))


# ---------------------------------------------------------------------------
# Unified search endpoint
# ---------------------------------------------------------------------------


@router.post("/context/search", response_model=UnifiedSearchResponse)
async def unified_search(
    request: UnifiedSearchRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: VectorSearchService = Depends(get_search_service),
) -> UnifiedSearchResponse:
    """Perform unified search across documents, chunks, and entities."""
    result = await service.search(
        query=request.query,
        tenant=tenant,
        search_type=request.search_type,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_ids=request.document_ids,
        entity_type=request.entity_type,
        include_entities=request.include_entities,
        include_chunks=request.include_chunks,
    )
    from aumos_context_graph.api.schemas import ChunkSearchResult, EntitySearchResult

    return UnifiedSearchResponse(
        query=result["query"],
        search_type=result["search_type"],
        chunks=[
            ChunkSearchResult(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                similarity_score=chunk["similarity_score"],
                chunk_index=chunk["chunk_index"],
                section_title=chunk.get("section_title"),
            )
            for chunk in result["chunks"]
        ],
        entities=[
            EntitySearchResult(
                entity_id=entity["entity_id"],
                name=entity["name"],
                entity_type=entity["entity_type"],
                description=entity.get("description"),
                similarity_score=entity["similarity_score"],
            )
            for entity in result["entities"]
        ],
        total=result["total"],
        latency_ms=result["latency_ms"],
    )


# ---------------------------------------------------------------------------
# Ontology endpoints
# ---------------------------------------------------------------------------


@router.post("/context/ontology", response_model=OntologyResponse, status_code=201)
async def create_ontology(
    request: CreateOntologyRequest,
    tenant: TenantContext = Depends(get_current_user),
    service: OntologyService = Depends(get_ontology_service),
) -> OntologyResponse:
    """Create a domain ontology definition for structuring knowledge."""
    ontology = await service.create_ontology(
        name=request.name,
        domain=request.domain,
        schema_definition=request.schema_definition,
        entity_types=request.entity_types,
        relationship_types=request.relationship_types,
        tenant=tenant,
        description=request.description,
        version=request.version,
    )
    return OntologyResponse(
        id=ontology.id,
        name=ontology.name,
        domain=ontology.domain,
        version=ontology.version,
        description=ontology.description,
        schema_definition=ontology.schema_definition,
        entity_types=ontology.entity_types,
        relationship_types=ontology.relationship_types,
        is_active=ontology.is_active,
        is_public=ontology.is_public,
        created_at=ontology.created_at.isoformat(),
        updated_at=ontology.updated_at.isoformat(),
    )


@router.get("/context/ontology/{ontology_id}", response_model=OntologyResponse)
async def get_ontology(
    ontology_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_user),
    service: OntologyService = Depends(get_ontology_service),
) -> OntologyResponse:
    """Retrieve an ontology definition by ID."""
    ontology = await service.get_ontology(ontology_id, tenant)
    return OntologyResponse(
        id=ontology.id,
        name=ontology.name,
        domain=ontology.domain,
        version=ontology.version,
        description=ontology.description,
        schema_definition=ontology.schema_definition,
        entity_types=ontology.entity_types,
        relationship_types=ontology.relationship_types,
        is_active=ontology.is_active,
        is_public=ontology.is_public,
        created_at=ontology.created_at.isoformat(),
        updated_at=ontology.updated_at.isoformat(),
    )
