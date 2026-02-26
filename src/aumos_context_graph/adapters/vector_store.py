"""Vector store adapter for aumos-context-graph.

Implements pgvector-based semantic similarity search for chunks and entities.
Uses cosine similarity with tenant-scoped queries.
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext
from aumos_common.observability import get_logger

from aumos_context_graph.core.models import DocumentChunk, GraphEntity

logger = get_logger(__name__)


class PgVectorStore:
    """Semantic similarity search using pgvector.

    Performs cosine similarity queries directly in PostgreSQL.
    All queries are tenant-scoped for RLS compliance.

    Args:
        session: Async SQLAlchemy session with pgvector support.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize PgVectorStore.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        tenant: TenantContext,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Find document chunks most similar to the query embedding.

        Uses cosine similarity (1 - cosine_distance) for ranking.

        Args:
            query_embedding: Query vector for similarity comparison.
            tenant: Tenant context for RLS scoping.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum similarity score (0-1).
            document_ids: Optional filter to specific documents.

        Returns:
            List of (DocumentChunk, similarity_score) tuples ordered by similarity.
        """
        cosine_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        similarity = 1 - cosine_distance

        query = (
            select(DocumentChunk, similarity.label("similarity"))
            .where(
                DocumentChunk.tenant_id == tenant.tenant_id,
                DocumentChunk.embedding.isnot(None),
                similarity >= similarity_threshold,
            )
            .order_by(cosine_distance)
            .limit(top_k)
        )

        if document_ids:
            query = query.where(DocumentChunk.document_id.in_(document_ids))

        result = await self._session.execute(query)
        rows = result.all()

        return [(row.DocumentChunk, float(row.similarity)) for row in rows]

    async def search_similar_entities(
        self,
        query_embedding: list[float],
        tenant: TenantContext,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[tuple[GraphEntity, float]]:
        """Find entities most similar to the query embedding.

        Args:
            query_embedding: Query vector for similarity comparison.
            tenant: Tenant context for RLS scoping.
            top_k: Maximum number of results to return.
            entity_type: Optional entity type filter.

        Returns:
            List of (GraphEntity, similarity_score) tuples ordered by similarity.
        """
        cosine_distance = GraphEntity.embedding.cosine_distance(query_embedding)
        similarity = 1 - cosine_distance

        query = (
            select(GraphEntity, similarity.label("similarity"))
            .where(
                GraphEntity.tenant_id == tenant.tenant_id,
                GraphEntity.embedding.isnot(None),
            )
            .order_by(cosine_distance)
            .limit(top_k)
        )

        if entity_type:
            query = query.where(GraphEntity.entity_type == entity_type)

        result = await self._session.execute(query)
        rows = result.all()

        return [(row.GraphEntity, float(row.similarity)) for row in rows]


__all__ = ["PgVectorStore"]
