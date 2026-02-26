"""Pydantic request/response schemas for aumos-context-graph API.

All API inputs and outputs are defined here as Pydantic models.
Never return raw dicts from API endpoints.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------


class IngestDocumentRequest(BaseModel):
    """Request to ingest a document into the knowledge graph."""

    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Raw document text content")
    document_type: str = Field(
        default="text",
        pattern="^(pdf|docx|txt|html|markdown|text)$",
        description="Document type",
    )
    source_uri: str | None = Field(default=None, max_length=2000, description="Source URI")
    extra_metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")
    chunk_size: int = Field(default=512, ge=64, le=2048, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=256, description="Chunk overlap in tokens")


class DocumentChunkResponse(BaseModel):
    """Document chunk summary in document response."""

    chunk_id: uuid.UUID
    chunk_index: int
    token_count: int
    section_title: str | None


class DocumentResponse(BaseModel):
    """Response with document metadata and optional chunks."""

    id: uuid.UUID
    title: str
    source_uri: str | None
    document_type: str
    status: str
    chunk_count: int
    token_count: int
    size_bytes: int
    language: str | None
    extra_metadata: dict[str, Any] | None
    created_at: str
    updated_at: str


class DocumentDetailResponse(DocumentResponse):
    """Document response including chunk list."""

    chunks: list[DocumentChunkResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph query schemas
# ---------------------------------------------------------------------------


class GraphQueryRequest(BaseModel):
    """Request to execute a Cypher query against the knowledge graph."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Cypher query string",
    )
    parameters: dict[str, Any] | None = Field(default=None, description="Query parameters")


class GraphQueryResponse(BaseModel):
    """Response from a Cypher graph query."""

    results: list[dict[str, Any]]
    result_count: int
    query: str


# ---------------------------------------------------------------------------
# Entity schemas
# ---------------------------------------------------------------------------


class CreateEntityRequest(BaseModel):
    """Request to create a new knowledge graph entity."""

    name: str = Field(..., min_length=1, max_length=500, description="Entity name")
    entity_type: str = Field(..., min_length=1, max_length=100, description="Entity type")
    description: str | None = Field(default=None, description="Entity description")
    aliases: list[str] | None = Field(default=None, description="Alternative names")
    properties: dict[str, Any] | None = Field(default=None, description="Additional properties")
    ontology_id: uuid.UUID | None = Field(default=None, description="Optional ontology ID")


class RelationshipResponse(BaseModel):
    """Relationship to or from an entity."""

    relationship_id: uuid.UUID
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    relationship_type: str
    properties: dict[str, Any] | None
    weight: float | None
    confidence_score: float | None


class EntityResponse(BaseModel):
    """Response with entity data."""

    id: uuid.UUID
    name: str
    entity_type: str
    description: str | None
    aliases: list[str] | None
    properties: dict[str, Any] | None
    ontology_id: uuid.UUID | None
    confidence_score: float | None
    created_at: str
    updated_at: str


class EntityDetailResponse(EntityResponse):
    """Entity response with related entities."""

    source_relationships: list[RelationshipResponse] = Field(default_factory=list)
    target_relationships: list[RelationshipResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG schemas
# ---------------------------------------------------------------------------


class RAGQueryRequest(BaseModel):
    """Request to perform a RAG query."""

    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Optional: restrict to specific documents",
    )
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Max tokens in response")


class RAGChunkContext(BaseModel):
    """A chunk used as context in a RAG response."""

    chunk_id: uuid.UUID
    content: str
    similarity_score: float


class RAGSourceReference(BaseModel):
    """Source document reference for a RAG response."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    similarity_score: float
    section_title: str | None


class RAGQueryResponse(BaseModel):
    """Response from a RAG query including context and prompt."""

    question: str
    context_chunks: list[RAGChunkContext]
    prompt: str
    sources: list[RAGSourceReference]
    context_count: int


class RAGSourcesResponse(BaseModel):
    """Response listing sources for a set of chunk IDs."""

    sources: list[dict[str, Any]]
    total: int


# ---------------------------------------------------------------------------
# Unified search schemas
# ---------------------------------------------------------------------------


class UnifiedSearchRequest(BaseModel):
    """Request for unified search across documents, chunks, and entities."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    search_type: str = Field(
        default="unified",
        pattern="^(keyword|semantic|graph|unified)$",
        description="Type of search to perform",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = Field(default=None)
    entity_type: str | None = Field(default=None)
    include_entities: bool = Field(default=True)
    include_chunks: bool = Field(default=True)


class ChunkSearchResult(BaseModel):
    """A document chunk matching a search query."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    similarity_score: float
    chunk_index: int
    section_title: str | None


class EntitySearchResult(BaseModel):
    """An entity matching a search query."""

    entity_id: uuid.UUID
    name: str
    entity_type: str
    description: str | None
    similarity_score: float


class UnifiedSearchResponse(BaseModel):
    """Response from unified search."""

    query: str
    search_type: str
    chunks: list[ChunkSearchResult]
    entities: list[EntitySearchResult]
    total: int
    latency_ms: int


# ---------------------------------------------------------------------------
# Ontology schemas
# ---------------------------------------------------------------------------


class CreateOntologyRequest(BaseModel):
    """Request to create a domain ontology definition."""

    name: str = Field(..., min_length=1, max_length=255, description="Ontology name")
    domain: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Domain: healthcare, finance, legal, general, etc.",
    )
    schema_definition: dict[str, Any] = Field(..., description="JSON-LD or OWL-like schema")
    entity_types: list[str] = Field(..., min_length=1, description="Valid entity types")
    relationship_types: list[str] = Field(..., min_length=1, description="Valid relationship types")
    description: str | None = Field(default=None)
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")


class OntologyResponse(BaseModel):
    """Response with ontology definition."""

    id: uuid.UUID
    name: str
    domain: str
    version: str
    description: str | None
    schema_definition: dict[str, Any]
    entity_types: list[str]
    relationship_types: list[str]
    is_active: bool
    is_public: bool
    created_at: str
    updated_at: str


__all__ = [
    "ChunkSearchResult",
    "CreateEntityRequest",
    "CreateOntologyRequest",
    "DocumentChunkResponse",
    "DocumentDetailResponse",
    "DocumentResponse",
    "EntityDetailResponse",
    "EntityResponse",
    "EntitySearchResult",
    "GraphQueryRequest",
    "GraphQueryResponse",
    "IngestDocumentRequest",
    "OntologyResponse",
    "RAGChunkContext",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "RAGSourceReference",
    "RAGSourcesResponse",
    "RelationshipResponse",
    "UnifiedSearchRequest",
    "UnifiedSearchResponse",
]
