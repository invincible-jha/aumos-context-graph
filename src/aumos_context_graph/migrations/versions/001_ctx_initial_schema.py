"""ctx initial schema — documents, chunks, entities, relationships, ontologies, search logs.

Revision ID: 001_ctx_initial
Revises: —
Create Date: 2026-02-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001_ctx_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all ctx_ tables with RLS policies and pgvector indexes."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ctx_documents
    op.create_table(
        "ctx_documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("source_uri", sa.String(2000), nullable=True),
        sa.Column("document_type", sa.String(50), nullable=False, server_default="text"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("size_bytes", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("extra_metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_ctx_documents_tenant_status", "ctx_documents", ["tenant_id", "status"])
    op.create_index("ix_ctx_documents_tenant_type", "ctx_documents", ["tenant_id", "document_type"])
    op.create_index("ix_ctx_documents_content_hash", "ctx_documents", ["content_hash"])

    # RLS for ctx_documents
    op.execute("ALTER TABLE ctx_documents ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_documents_tenant_isolation ON ctx_documents "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid)"
    )

    # ctx_chunks
    op.create_table(
        "ctx_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("ctx_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding", sa.Text, nullable=True, comment="pgvector column — type set in migration"),
        sa.Column("start_char", sa.Integer, nullable=True),
        sa.Column("end_char", sa.Integer, nullable=True),
        sa.Column("section_title", sa.String(500), nullable=True),
        sa.Column("chunk_metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    # Convert embedding column to vector type after table creation
    op.execute("ALTER TABLE ctx_chunks ALTER COLUMN embedding TYPE vector(1536) USING NULL")
    op.create_index("ix_ctx_chunks_document_index", "ctx_chunks", ["document_id", "chunk_index"])
    op.create_index("ix_ctx_chunks_tenant", "ctx_chunks", ["tenant_id"])
    # IVFFlat index for approximate nearest neighbor search
    op.execute(
        "CREATE INDEX ix_ctx_chunks_embedding_ivfflat ON ctx_chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.execute("ALTER TABLE ctx_chunks ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_chunks_tenant_isolation ON ctx_chunks "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid)"
    )

    # ctx_ontologies
    op.create_table(
        "ctx_ontologies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("domain", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False, server_default="1.0.0"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("schema_definition", JSONB, nullable=False),
        sa.Column("entity_types", sa.JSON, nullable=False),
        sa.Column("relationship_types", sa.JSON, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_public", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_ctx_ontologies_tenant_domain", "ctx_ontologies", ["tenant_id", "domain"])
    op.create_index("ix_ctx_ontologies_tenant_name", "ctx_ontologies", ["tenant_id", "name"])

    op.execute("ALTER TABLE ctx_ontologies ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_ontologies_tenant_isolation ON ctx_ontologies "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid OR is_public = true)"
    )

    # ctx_entities
    op.create_table(
        "ctx_entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("entity_type", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("aliases", sa.JSON, nullable=True),
        sa.Column("properties", JSONB, nullable=True),
        sa.Column("ontology_id", UUID(as_uuid=True), nullable=True),
        sa.Column("external_id", sa.String(500), nullable=True),
        sa.Column("embedding", sa.Text, nullable=True, comment="pgvector column"),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.execute("ALTER TABLE ctx_entities ALTER COLUMN embedding TYPE vector(1536) USING NULL")
    op.create_index("ix_ctx_entities_tenant_type", "ctx_entities", ["tenant_id", "entity_type"])
    op.create_index("ix_ctx_entities_tenant_name", "ctx_entities", ["tenant_id", "name"])
    op.execute(
        "CREATE INDEX ix_ctx_entities_embedding_ivfflat ON ctx_entities "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.execute("ALTER TABLE ctx_entities ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_entities_tenant_isolation ON ctx_entities "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid)"
    )

    # ctx_relationships
    op.create_table(
        "ctx_relationships",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source_entity_id", UUID(as_uuid=True), sa.ForeignKey("ctx_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_entity_id", UUID(as_uuid=True), sa.ForeignKey("ctx_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relationship_type", sa.String(100), nullable=False),
        sa.Column("properties", JSONB, nullable=True),
        sa.Column("weight", sa.Float, nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("source_document_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_ctx_relationships_source_type", "ctx_relationships", ["source_entity_id", "relationship_type"])
    op.create_index("ix_ctx_relationships_target_type", "ctx_relationships", ["target_entity_id", "relationship_type"])
    op.create_index("ix_ctx_relationships_tenant", "ctx_relationships", ["tenant_id"])

    op.execute("ALTER TABLE ctx_relationships ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_relationships_tenant_isolation ON ctx_relationships "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid)"
    )

    # ctx_search_logs
    op.create_table(
        "ctx_search_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("search_type", sa.String(50), nullable=False),
        sa.Column("result_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("filters_applied", JSONB, nullable=True),
        sa.Column("top_result_ids", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_ctx_search_logs_tenant_type", "ctx_search_logs", ["tenant_id", "search_type"])
    op.create_index("ix_ctx_search_logs_tenant_created", "ctx_search_logs", ["tenant_id", "created_at"])

    op.execute("ALTER TABLE ctx_search_logs ENABLE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY ctx_search_logs_tenant_isolation ON ctx_search_logs "
        "USING (tenant_id = current_setting('app.current_tenant')::uuid)"
    )


def downgrade() -> None:
    """Drop all ctx_ tables."""
    op.drop_table("ctx_search_logs")
    op.drop_table("ctx_relationships")
    op.drop_table("ctx_entities")
    op.drop_table("ctx_ontologies")
    op.drop_table("ctx_chunks")
    op.drop_table("ctx_documents")
