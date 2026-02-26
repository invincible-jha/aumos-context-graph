"""Abstract interfaces (Protocol classes) for aumos-context-graph.

Defines the contracts between the core domain and adapters.
Services depend on interfaces, not concrete implementations.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from aumos_common.auth import TenantContext

from aumos_context_graph.core.models import (
    Document,
    DocumentChunk,
    EntityRelationship,
    GraphEntity,
    Ontology,
    SearchLog,
)


@runtime_checkable
class IDocumentRepository(Protocol):
    """Repository interface for Document and DocumentChunk persistence."""

    async def get_by_id(self, document_id: uuid.UUID, tenant: TenantContext) -> Document | None: ...

    async def get_by_content_hash(
        self, content_hash: str, tenant: TenantContext
    ) -> Document | None: ...

    async def list_all(
        self,
        tenant: TenantContext,
        status: str | None = None,
        document_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Document]: ...

    async def create(
        self,
        title: str,
        source_uri: str | None,
        document_type: str,
        content_hash: str,
        size_bytes: int,
        extra_metadata: dict[str, Any] | None,
        tenant: TenantContext,
    ) -> Document: ...

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: str,
        chunk_count: int,
        token_count: int,
        tenant: TenantContext,
    ) -> Document | None: ...

    async def delete(self, document_id: uuid.UUID, tenant: TenantContext) -> None: ...

    async def create_chunk(
        self,
        document_id: uuid.UUID,
        chunk_index: int,
        content: str,
        token_count: int,
        tenant: TenantContext,
        embedding: list[float] | None = None,
        start_char: int | None = None,
        end_char: int | None = None,
        section_title: str | None = None,
    ) -> DocumentChunk: ...

    async def get_chunks_by_document(
        self, document_id: uuid.UUID, tenant: TenantContext
    ) -> list[DocumentChunk]: ...

    async def update_chunk_embedding(
        self,
        chunk_id: uuid.UUID,
        embedding: list[float],
        tenant: TenantContext,
    ) -> None: ...


@runtime_checkable
class IEntityRepository(Protocol):
    """Repository interface for GraphEntity and EntityRelationship persistence."""

    async def get_by_id(self, entity_id: uuid.UUID, tenant: TenantContext) -> GraphEntity | None: ...

    async def get_with_relationships(
        self, entity_id: uuid.UUID, tenant: TenantContext, depth: int = 1
    ) -> GraphEntity | None: ...

    async def find_by_name(
        self, name: str, entity_type: str | None, tenant: TenantContext
    ) -> list[GraphEntity]: ...

    async def create(
        self,
        name: str,
        entity_type: str,
        tenant: TenantContext,
        description: str | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        ontology_id: uuid.UUID | None = None,
    ) -> GraphEntity: ...

    async def update_embedding(
        self, entity_id: uuid.UUID, embedding: list[float], tenant: TenantContext
    ) -> None: ...

    async def create_relationship(
        self,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        tenant: TenantContext,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        source_document_id: uuid.UUID | None = None,
    ) -> EntityRelationship: ...

    async def get_relationships(
        self,
        entity_id: uuid.UUID,
        tenant: TenantContext,
        relationship_type: str | None = None,
    ) -> list[EntityRelationship]: ...

    async def delete(self, entity_id: uuid.UUID, tenant: TenantContext) -> None: ...


@runtime_checkable
class IOntologyRepository(Protocol):
    """Repository interface for Ontology management."""

    async def get_by_id(self, ontology_id: uuid.UUID, tenant: TenantContext) -> Ontology | None: ...

    async def list_all(
        self,
        tenant: TenantContext,
        domain: str | None = None,
        is_active: bool | None = None,
    ) -> list[Ontology]: ...

    async def create(
        self,
        name: str,
        domain: str,
        schema_definition: dict[str, Any],
        entity_types: list[str],
        relationship_types: list[str],
        tenant: TenantContext,
        description: str | None = None,
        version: str = "1.0.0",
    ) -> Ontology: ...

    async def update(
        self,
        ontology_id: uuid.UUID,
        updates: dict[str, Any],
        tenant: TenantContext,
    ) -> Ontology | None: ...

    async def delete(self, ontology_id: uuid.UUID, tenant: TenantContext) -> None: ...


@runtime_checkable
class ISearchLogRepository(Protocol):
    """Repository interface for search query logging."""

    async def create(
        self,
        query_text: str,
        search_type: str,
        result_count: int,
        tenant: TenantContext,
        latency_ms: int | None = None,
        filters_applied: dict[str, Any] | None = None,
        top_result_ids: list[str] | None = None,
    ) -> SearchLog: ...


@runtime_checkable
class IVectorStore(Protocol):
    """Interface for vector similarity search operations."""

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        tenant: TenantContext,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[DocumentChunk, float]]: ...

    async def search_similar_entities(
        self,
        query_embedding: list[float],
        tenant: TenantContext,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[tuple[GraphEntity, float]]: ...


@runtime_checkable
class IEmbeddingEngine(Protocol):
    """Interface for generating text embeddings."""

    async def embed_text(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class IGraphClient(Protocol):
    """Interface for graph database queries (Neo4j / Apache AGE)."""

    async def execute_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]: ...

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> str: ...

    async def create_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str: ...


__all__ = [
    "IDocumentRepository",
    "IEmbeddingEngine",
    "IEntityRepository",
    "IGraphClient",
    "IOntologyRepository",
    "ISearchLogRepository",
    "IVectorStore",
]
