"""SQLAlchemy ORM models for aumos-context-graph.

All tenant-scoped tables extend AumOSModel which provides:
  - id: UUID primary key
  - tenant_id: UUID (RLS-enforced)
  - created_at: datetime
  - updated_at: datetime

Table prefix: ctx_
"""

from __future__ import annotations

import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, BigInteger, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class Document(AumOSModel):
    """Ingested document with metadata and processing status.

    Table: ctx_documents
    """

    __tablename__ = "ctx_documents"

    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    source_uri: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    document_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="text",
        comment="pdf, docx, txt, html, markdown",
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        comment="pending, processing, indexed, failed",
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_ctx_documents_tenant_status", "tenant_id", "status"),
        Index("ix_ctx_documents_tenant_type", "tenant_id", "document_type"),
    )


class DocumentChunk(AumOSModel):
    """Document chunk with embedding for vector search.

    Table: ctx_chunks
    """

    __tablename__ = "ctx_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ctx_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_ctx_chunks_document_index", "document_id", "chunk_index"),
        Index("ix_ctx_chunks_tenant", "tenant_id"),
    )


class GraphEntity(AumOSModel):
    """Knowledge graph entity node.

    Table: ctx_entities
    """

    __tablename__ = "ctx_entities"

    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="person, organization, concept, location, product, event, etc.",
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    ontology_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Optional ontology this entity belongs to",
    )
    external_id: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="ID in external graph database (Neo4j node id, AGE vertex id)",
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    source_relationships: Mapped[list[EntityRelationship]] = relationship(
        "EntityRelationship",
        foreign_keys="EntityRelationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    target_relationships: Mapped[list[EntityRelationship]] = relationship(
        "EntityRelationship",
        foreign_keys="EntityRelationship.target_entity_id",
        back_populates="target_entity",
    )

    __table_args__ = (
        Index("ix_ctx_entities_tenant_type", "tenant_id", "entity_type"),
        Index("ix_ctx_entities_tenant_name", "tenant_id", "name"),
    )


class EntityRelationship(AumOSModel):
    """Directed relationship between two graph entities.

    Table: ctx_relationships
    """

    __tablename__ = "ctx_relationships"

    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ctx_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ctx_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="works_for, belongs_to, related_to, part_of, etc.",
    )
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Document where this relationship was extracted from",
    )

    # Relationships
    source_entity: Mapped[GraphEntity] = relationship(
        "GraphEntity",
        foreign_keys=[source_entity_id],
        back_populates="source_relationships",
    )
    target_entity: Mapped[GraphEntity] = relationship(
        "GraphEntity",
        foreign_keys=[target_entity_id],
        back_populates="target_relationships",
    )

    __table_args__ = (
        Index("ix_ctx_relationships_source_type", "source_entity_id", "relationship_type"),
        Index("ix_ctx_relationships_target_type", "target_entity_id", "relationship_type"),
        Index("ix_ctx_relationships_tenant", "tenant_id"),
    )


class Ontology(AumOSModel):
    """Domain ontology definition for structuring knowledge.

    Table: ctx_ontologies
    """

    __tablename__ = "ctx_ontologies"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="healthcare, finance, legal, general, etc.",
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_definition: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="JSON-LD or OWL-like ontology schema",
    )
    entity_types: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of valid entity type strings",
    )
    relationship_types: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of valid relationship type strings",
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_public: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        comment="If True, all tenants can use this ontology",
    )

    __table_args__ = (
        Index("ix_ctx_ontologies_tenant_domain", "tenant_id", "domain"),
        Index("ix_ctx_ontologies_tenant_name", "tenant_id", "name"),
    )


class SearchLog(AumOSModel):
    """Log of search queries for analytics and optimization.

    Table: ctx_search_logs
    """

    __tablename__ = "ctx_search_logs"

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="keyword, semantic, graph, unified",
    )
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filters_applied: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    top_result_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_ctx_search_logs_tenant_type", "tenant_id", "search_type"),
        Index("ix_ctx_search_logs_tenant_created", "tenant_id", "created_at"),
    )


__all__ = [
    "Document",
    "DocumentChunk",
    "EntityRelationship",
    "GraphEntity",
    "Ontology",
    "SearchLog",
]
