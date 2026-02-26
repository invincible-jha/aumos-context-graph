"""Test fixtures for aumos-context-graph.

Uses aumos_common.testing fixtures for auth, database, and tenant context.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from aumos_common.auth import TenantContext
from aumos_common.testing import UserFactory


@pytest.fixture
def tenant() -> TenantContext:
    """Provide a test tenant context."""
    return TenantContext(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        privilege_level=3,
    )


@pytest.fixture
def mock_document_repository() -> MagicMock:
    """Provide a mock IDocumentRepository."""
    mock = MagicMock()
    mock.get_by_id = AsyncMock(return_value=None)
    mock.get_by_content_hash = AsyncMock(return_value=None)
    mock.create = AsyncMock()
    mock.update_status = AsyncMock()
    mock.create_chunk = AsyncMock()
    mock.get_chunks_by_document = AsyncMock(return_value=[])
    mock.update_chunk_embedding = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_engine() -> MagicMock:
    """Provide a mock IEmbeddingEngine returning zero vectors."""
    mock = MagicMock()
    mock.embed_text = AsyncMock(return_value=[0.0] * 1536)
    mock.embed_batch = AsyncMock(return_value=[[0.0] * 1536])
    return mock


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Provide a mock IVectorStore with empty results."""
    mock = MagicMock()
    mock.search_similar_chunks = AsyncMock(return_value=[])
    mock.search_similar_entities = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_entity_repository() -> MagicMock:
    """Provide a mock IEntityRepository."""
    mock = MagicMock()
    mock.get_by_id = AsyncMock(return_value=None)
    mock.get_with_relationships = AsyncMock(return_value=None)
    mock.find_by_name = AsyncMock(return_value=[])
    mock.create = AsyncMock()
    mock.update_embedding = AsyncMock()
    mock.create_relationship = AsyncMock()
    mock.get_relationships = AsyncMock(return_value=[])
    mock.delete = AsyncMock()
    return mock


@pytest.fixture
def mock_graph_client() -> MagicMock:
    """Provide a mock IGraphClient."""
    mock = MagicMock()
    mock.execute_cypher = AsyncMock(return_value=[])
    mock.create_node = AsyncMock(return_value="node_123")
    mock.create_relationship = AsyncMock(return_value="rel_456")
    return mock


@pytest.fixture
def mock_ontology_repository() -> MagicMock:
    """Provide a mock IOntologyRepository."""
    mock = MagicMock()
    mock.get_by_id = AsyncMock(return_value=None)
    mock.list_all = AsyncMock(return_value=[])
    mock.create = AsyncMock()
    mock.update = AsyncMock()
    mock.delete = AsyncMock()
    return mock


@pytest.fixture
def mock_search_log_repository() -> MagicMock:
    """Provide a mock ISearchLogRepository."""
    mock = MagicMock()
    mock.create = AsyncMock()
    return mock
