"""Kafka event publishing for aumos-context-graph.

Publishes domain events after state-changing operations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentIndexedEvent:
    """Published when a document is successfully indexed.

    Attributes:
        tenant_id: Tenant UUID.
        document_id: Indexed document UUID.
        document_type: Type of document.
        chunk_count: Number of chunks created.
        correlation_id: Request correlation ID.
    """

    tenant_id: str
    document_id: str
    document_type: str
    chunk_count: int
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "document.indexed"


@dataclass
class EntityCreatedEvent:
    """Published when a graph entity is created.

    Attributes:
        tenant_id: Tenant UUID.
        entity_id: Created entity UUID.
        entity_type: Entity type string.
        name: Entity name.
        correlation_id: Request correlation ID.
    """

    tenant_id: str
    entity_id: str
    entity_type: str
    name: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "entity.created"


@dataclass
class SearchPerformedEvent:
    """Published when a search is executed.

    Attributes:
        tenant_id: Tenant UUID.
        search_type: Type of search.
        result_count: Number of results returned.
        latency_ms: Query latency.
        correlation_id: Request correlation ID.
    """

    tenant_id: str
    search_type: str
    result_count: int
    latency_ms: int
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "search.performed"


class ContextGraphEventPublisher:
    """Event publisher for context-graph domain events.

    Wraps the aumos-common EventPublisher with context-graph specific events.

    Args:
        publisher: EventPublisher from aumos-common.
    """

    def __init__(self, publisher: EventPublisher) -> None:
        """Initialize with injected EventPublisher.

        Args:
            publisher: aumos-common EventPublisher.
        """
        self._publisher = publisher

    async def publish_document_indexed(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        document_type: str,
        chunk_count: int,
    ) -> None:
        """Publish a document.indexed event.

        Args:
            tenant_id: Tenant UUID.
            document_id: Indexed document UUID.
            document_type: Document type string.
            chunk_count: Number of chunks created.
        """
        event = DocumentIndexedEvent(
            tenant_id=str(tenant_id),
            document_id=str(document_id),
            document_type=document_type,
            chunk_count=chunk_count,
        )
        await self._publisher.publish(Topics.DOCUMENT_LIFECYCLE, event)
        logger.info(
            "Published document.indexed event",
            document_id=str(document_id),
            tenant_id=str(tenant_id),
        )

    async def publish_entity_created(
        self,
        tenant_id: uuid.UUID,
        entity_id: uuid.UUID,
        entity_type: str,
        name: str,
    ) -> None:
        """Publish an entity.created event.

        Args:
            tenant_id: Tenant UUID.
            entity_id: Created entity UUID.
            entity_type: Entity type string.
            name: Entity name.
        """
        event = EntityCreatedEvent(
            tenant_id=str(tenant_id),
            entity_id=str(entity_id),
            entity_type=entity_type,
            name=name,
        )
        await self._publisher.publish(Topics.KNOWLEDGE_GRAPH, event)
        logger.info(
            "Published entity.created event",
            entity_id=str(entity_id),
            tenant_id=str(tenant_id),
        )


__all__ = [
    "ContextGraphEventPublisher",
    "DocumentIndexedEvent",
    "EntityCreatedEvent",
    "SearchPerformedEvent",
]
