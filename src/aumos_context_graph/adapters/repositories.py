"""SQLAlchemy repositories for aumos-context-graph.

Implements all repository interfaces using SQLAlchemy async ORM.
All queries are tenant-scoped via aumos-common RLS middleware.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext
from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_context_graph.core.models import (
    Document,
    DocumentChunk,
    EntityRelationship,
    GraphEntity,
    Ontology,
    SearchLog,
)

logger = get_logger(__name__)


class DocumentRepository(BaseRepository):
    """Repository for Document and DocumentChunk persistence.

    Args:
        session: Async SQLAlchemy session (RLS-scoped by aumos-common).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)
        self._session = session

    async def get_by_id(self, document_id: uuid.UUID, tenant: TenantContext) -> Document | None:
        """Fetch a document by primary key within tenant scope.

        Args:
            document_id: Document UUID.
            tenant: Tenant context for RLS.

        Returns:
            Document if found, None otherwise.
        """
        result = await self._session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_content_hash(
        self, content_hash: str, tenant: TenantContext
    ) -> Document | None:
        """Fetch a document by content hash for deduplication.

        Args:
            content_hash: SHA-256 hex digest of document content.
            tenant: Tenant context for RLS.

        Returns:
            Document if found, None otherwise.
        """
        result = await self._session.execute(
            select(Document).where(
                Document.content_hash == content_hash,
                Document.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        tenant: TenantContext,
        status: str | None = None,
        document_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Document]:
        """List documents for a tenant with optional filters.

        Args:
            tenant: Tenant context for RLS.
            status: Optional status filter.
            document_type: Optional document type filter.
            limit: Maximum records to return.
            offset: Pagination offset.

        Returns:
            List of matching Document records.
        """
        query = select(Document).where(Document.tenant_id == tenant.tenant_id)
        if status:
            query = query.where(Document.status == status)
        if document_type:
            query = query.where(Document.document_type == document_type)
        query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def create(
        self,
        title: str,
        source_uri: str | None,
        document_type: str,
        content_hash: str,
        size_bytes: int,
        extra_metadata: dict[str, Any] | None,
        tenant: TenantContext,
    ) -> Document:
        """Create a new document record.

        Args:
            title: Document title.
            source_uri: Optional source URI.
            document_type: Document type string.
            content_hash: SHA-256 hex digest.
            size_bytes: Document size in bytes.
            extra_metadata: Optional metadata dict.
            tenant: Tenant context for RLS.

        Returns:
            Newly created Document record.
        """
        document = Document(
            title=title,
            source_uri=source_uri,
            document_type=document_type,
            content_hash=content_hash,
            size_bytes=size_bytes,
            extra_metadata=extra_metadata,
            tenant_id=tenant.tenant_id,
            status="pending",
        )
        self._session.add(document)
        await self._session.flush()
        await self._session.refresh(document)
        return document

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: str,
        chunk_count: int,
        token_count: int,
        tenant: TenantContext,
    ) -> Document | None:
        """Update document status and chunk/token counts after processing.

        Args:
            document_id: Document UUID.
            status: New status string.
            chunk_count: Number of chunks created.
            token_count: Total token count.
            tenant: Tenant context for RLS.

        Returns:
            Updated Document if found, None otherwise.
        """
        await self._session.execute(
            update(Document)
            .where(Document.id == document_id, Document.tenant_id == tenant.tenant_id)
            .values(status=status, chunk_count=chunk_count, token_count=token_count)
        )
        await self._session.flush()
        return await self.get_by_id(document_id, tenant)

    async def delete(self, document_id: uuid.UUID, tenant: TenantContext) -> None:
        """Delete a document and its chunks by ID.

        Args:
            document_id: Document UUID.
            tenant: Tenant context for RLS.
        """
        document = await self.get_by_id(document_id, tenant)
        if document:
            await self._session.delete(document)
            await self._session.flush()

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
    ) -> DocumentChunk:
        """Create a document chunk with optional embedding.

        Args:
            document_id: Parent document UUID.
            chunk_index: Zero-based chunk sequence number.
            content: Chunk text content.
            token_count: Approximate token count.
            tenant: Tenant context for RLS.
            embedding: Optional pre-computed embedding vector.
            start_char: Start character offset in source document.
            end_char: End character offset in source document.
            section_title: Optional section heading for context.

        Returns:
            Newly created DocumentChunk record.
        """
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            tenant_id=tenant.tenant_id,
            embedding=embedding,
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
        )
        self._session.add(chunk)
        await self._session.flush()
        await self._session.refresh(chunk)
        return chunk

    async def get_chunks_by_document(
        self, document_id: uuid.UUID, tenant: TenantContext
    ) -> list[DocumentChunk]:
        """Fetch all chunks for a document ordered by chunk index.

        Args:
            document_id: Parent document UUID.
            tenant: Tenant context for RLS.

        Returns:
            Ordered list of DocumentChunk records.
        """
        result = await self._session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.tenant_id == tenant.tenant_id,
            )
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def update_chunk_embedding(
        self,
        chunk_id: uuid.UUID,
        embedding: list[float],
        tenant: TenantContext,
    ) -> None:
        """Update the embedding vector for a chunk.

        Args:
            chunk_id: Chunk UUID.
            embedding: New embedding vector.
            tenant: Tenant context for RLS.
        """
        await self._session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id == chunk_id, DocumentChunk.tenant_id == tenant.tenant_id)
            .values(embedding=embedding)
        )
        await self._session.flush()


class EntityRepository(BaseRepository):
    """Repository for GraphEntity and EntityRelationship persistence.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)
        self._session = session

    async def get_by_id(self, entity_id: uuid.UUID, tenant: TenantContext) -> GraphEntity | None:
        """Fetch an entity by primary key.

        Args:
            entity_id: Entity UUID.
            tenant: Tenant context for RLS.

        Returns:
            GraphEntity if found, None otherwise.
        """
        result = await self._session.execute(
            select(GraphEntity).where(
                GraphEntity.id == entity_id,
                GraphEntity.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_with_relationships(
        self, entity_id: uuid.UUID, tenant: TenantContext, depth: int = 1
    ) -> GraphEntity | None:
        """Fetch entity with eagerly loaded relationships.

        Args:
            entity_id: Entity UUID.
            tenant: Tenant context for RLS.
            depth: Relationship traversal depth (currently loads 1 level).

        Returns:
            GraphEntity with relationships if found, None otherwise.
        """
        result = await self._session.execute(
            select(GraphEntity).where(
                GraphEntity.id == entity_id,
                GraphEntity.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def find_by_name(
        self, name: str, entity_type: str | None, tenant: TenantContext
    ) -> list[GraphEntity]:
        """Find entities by exact name match.

        Args:
            name: Entity name to search for.
            entity_type: Optional type filter.
            tenant: Tenant context for RLS.

        Returns:
            List of matching GraphEntity records.
        """
        query = select(GraphEntity).where(
            GraphEntity.name == name,
            GraphEntity.tenant_id == tenant.tenant_id,
        )
        if entity_type:
            query = query.where(GraphEntity.entity_type == entity_type)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def create(
        self,
        name: str,
        entity_type: str,
        tenant: TenantContext,
        description: str | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        ontology_id: uuid.UUID | None = None,
    ) -> GraphEntity:
        """Create a new graph entity.

        Args:
            name: Entity name.
            entity_type: Entity type string.
            tenant: Tenant context for RLS.
            description: Optional description.
            aliases: Optional list of alternative names.
            properties: Optional additional properties dict.
            ontology_id: Optional ontology UUID.

        Returns:
            Newly created GraphEntity record.
        """
        entity = GraphEntity(
            name=name,
            entity_type=entity_type,
            description=description,
            aliases=aliases,
            properties=properties,
            ontology_id=ontology_id,
            tenant_id=tenant.tenant_id,
        )
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def update_embedding(
        self, entity_id: uuid.UUID, embedding: list[float], tenant: TenantContext
    ) -> None:
        """Update the embedding for an entity.

        Args:
            entity_id: Entity UUID.
            embedding: Embedding vector.
            tenant: Tenant context for RLS.
        """
        await self._session.execute(
            update(GraphEntity)
            .where(GraphEntity.id == entity_id, GraphEntity.tenant_id == tenant.tenant_id)
            .values(embedding=embedding)
        )
        await self._session.flush()

    async def create_relationship(
        self,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        tenant: TenantContext,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        source_document_id: uuid.UUID | None = None,
    ) -> EntityRelationship:
        """Create a directed relationship between two entities.

        Args:
            source_entity_id: Source entity UUID.
            target_entity_id: Target entity UUID.
            relationship_type: Relationship type string.
            tenant: Tenant context for RLS.
            properties: Optional properties dict.
            weight: Optional edge weight.
            source_document_id: Optional source document UUID.

        Returns:
            Newly created EntityRelationship record.
        """
        relationship = EntityRelationship(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            properties=properties,
            weight=weight,
            source_document_id=source_document_id,
            tenant_id=tenant.tenant_id,
        )
        self._session.add(relationship)
        await self._session.flush()
        await self._session.refresh(relationship)
        return relationship

    async def get_relationships(
        self,
        entity_id: uuid.UUID,
        tenant: TenantContext,
        relationship_type: str | None = None,
    ) -> list[EntityRelationship]:
        """Fetch all relationships involving an entity.

        Args:
            entity_id: Entity UUID (matches source or target).
            tenant: Tenant context for RLS.
            relationship_type: Optional type filter.

        Returns:
            List of EntityRelationship records.
        """
        from sqlalchemy import or_

        query = select(EntityRelationship).where(
            or_(
                EntityRelationship.source_entity_id == entity_id,
                EntityRelationship.target_entity_id == entity_id,
            ),
            EntityRelationship.tenant_id == tenant.tenant_id,
        )
        if relationship_type:
            query = query.where(EntityRelationship.relationship_type == relationship_type)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def delete(self, entity_id: uuid.UUID, tenant: TenantContext) -> None:
        """Delete an entity and its relationships.

        Args:
            entity_id: Entity UUID.
            tenant: Tenant context for RLS.
        """
        entity = await self.get_by_id(entity_id, tenant)
        if entity:
            await self._session.delete(entity)
            await self._session.flush()


class OntologyRepository(BaseRepository):
    """Repository for Ontology persistence.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)
        self._session = session

    async def get_by_id(self, ontology_id: uuid.UUID, tenant: TenantContext) -> Ontology | None:
        """Fetch an ontology by primary key.

        Args:
            ontology_id: Ontology UUID.
            tenant: Tenant context for RLS.

        Returns:
            Ontology if found, None otherwise.
        """
        result = await self._session.execute(
            select(Ontology).where(
                Ontology.id == ontology_id,
                Ontology.tenant_id == tenant.tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        tenant: TenantContext,
        domain: str | None = None,
        is_active: bool | None = None,
    ) -> list[Ontology]:
        """List ontologies for a tenant.

        Args:
            tenant: Tenant context for RLS.
            domain: Optional domain filter.
            is_active: Optional active status filter.

        Returns:
            List of matching Ontology records.
        """
        query = select(Ontology).where(Ontology.tenant_id == tenant.tenant_id)
        if domain:
            query = query.where(Ontology.domain == domain)
        if is_active is not None:
            query = query.where(Ontology.is_active == is_active)
        result = await self._session.execute(query)
        return list(result.scalars().all())

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
    ) -> Ontology:
        """Create a new ontology definition.

        Args:
            name: Ontology name.
            domain: Domain category.
            schema_definition: JSON-LD or OWL-like schema.
            entity_types: Valid entity type strings.
            relationship_types: Valid relationship type strings.
            tenant: Tenant context for RLS.
            description: Optional description.
            version: Semantic version string.

        Returns:
            Newly created Ontology record.
        """
        ontology = Ontology(
            name=name,
            domain=domain,
            schema_definition=schema_definition,
            entity_types=entity_types,
            relationship_types=relationship_types,
            description=description,
            version=version,
            tenant_id=tenant.tenant_id,
        )
        self._session.add(ontology)
        await self._session.flush()
        await self._session.refresh(ontology)
        return ontology

    async def update(
        self,
        ontology_id: uuid.UUID,
        updates: dict[str, Any],
        tenant: TenantContext,
    ) -> Ontology | None:
        """Apply partial updates to an ontology.

        Args:
            ontology_id: Ontology UUID.
            updates: Dict of field names to new values.
            tenant: Tenant context for RLS.

        Returns:
            Updated Ontology if found, None otherwise.
        """
        await self._session.execute(
            update(Ontology)
            .where(Ontology.id == ontology_id, Ontology.tenant_id == tenant.tenant_id)
            .values(**updates)
        )
        await self._session.flush()
        return await self.get_by_id(ontology_id, tenant)

    async def delete(self, ontology_id: uuid.UUID, tenant: TenantContext) -> None:
        """Delete an ontology by ID.

        Args:
            ontology_id: Ontology UUID.
            tenant: Tenant context for RLS.
        """
        ontology = await self.get_by_id(ontology_id, tenant)
        if ontology:
            await self._session.delete(ontology)
            await self._session.flush()


class SearchLogRepository(BaseRepository):
    """Repository for search query logging.

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        super().__init__(session)
        self._session = session

    async def create(
        self,
        query_text: str,
        search_type: str,
        result_count: int,
        tenant: TenantContext,
        latency_ms: int | None = None,
        filters_applied: dict[str, Any] | None = None,
        top_result_ids: list[str] | None = None,
    ) -> SearchLog:
        """Log a search query for analytics.

        Args:
            query_text: The raw search query string.
            search_type: Type of search performed.
            result_count: Number of results returned.
            tenant: Tenant context for RLS.
            latency_ms: Optional query latency in milliseconds.
            filters_applied: Optional dict of applied filters.
            top_result_ids: Optional list of top result IDs.

        Returns:
            Newly created SearchLog record.
        """
        log = SearchLog(
            query_text=query_text,
            search_type=search_type,
            result_count=result_count,
            latency_ms=latency_ms,
            filters_applied=filters_applied,
            top_result_ids=top_result_ids,
            tenant_id=tenant.tenant_id,
        )
        self._session.add(log)
        await self._session.flush()
        return log


__all__ = [
    "DocumentRepository",
    "EntityRepository",
    "OntologyRepository",
    "SearchLogRepository",
]
