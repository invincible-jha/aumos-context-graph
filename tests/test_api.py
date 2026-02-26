"""API endpoint tests for aumos-context-graph."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aumos_context_graph.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a test client for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_tenant_context() -> MagicMock:
    """Provide a mock TenantContext for auth override."""
    mock = MagicMock()
    mock.tenant_id = uuid.uuid4()
    mock.user_id = uuid.uuid4()
    mock.privilege_level = 3
    return mock


@pytest.fixture
def sample_document_response() -> dict:
    """Provide a sample document dict matching DocumentResponse schema."""
    return {
        "id": str(uuid.uuid4()),
        "title": "Test Document",
        "source_uri": None,
        "document_type": "text",
        "status": "indexed",
        "chunk_count": 3,
        "token_count": 150,
        "size_bytes": 500,
        "language": None,
        "extra_metadata": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class TestDocumentEndpoints:
    """Tests for document ingestion and retrieval endpoints."""

    def test_ingest_document_requires_auth(self, client: TestClient) -> None:
        """POST /context/documents should return 401 without auth token."""
        response = client.post(
            "/api/v1/context/documents",
            json={"title": "Test", "content": "Content"},
        )
        # Auth rejected by aumos-common middleware
        assert response.status_code in (401, 403, 422)

    def test_get_document_requires_auth(self, client: TestClient) -> None:
        """GET /context/documents/{id} should return 401 without auth."""
        response = client.get(f"/api/v1/context/documents/{uuid.uuid4()}")
        assert response.status_code in (401, 403, 422)


class TestSearchEndpoints:
    """Tests for unified search endpoint."""

    def test_search_requires_auth(self, client: TestClient) -> None:
        """POST /context/search should return 401 without auth."""
        response = client.post(
            "/api/v1/context/search",
            json={"query": "test query"},
        )
        assert response.status_code in (401, 403, 422)


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_ingest_request_validates_empty_title(self, client: TestClient) -> None:
        """Ingest request with empty title should be rejected."""
        response = client.post(
            "/api/v1/context/documents",
            json={"title": "", "content": "content"},
        )
        assert response.status_code == 422

    def test_search_request_validates_invalid_search_type(self, client: TestClient) -> None:
        """Search request with invalid search_type should be rejected."""
        response = client.post(
            "/api/v1/context/search",
            json={"query": "test", "search_type": "invalid_type"},
        )
        assert response.status_code == 422

    def test_rag_query_validates_top_k_bounds(self, client: TestClient) -> None:
        """RAG query with top_k out of bounds should be rejected."""
        response = client.post(
            "/api/v1/context/rag/query",
            json={"question": "test", "top_k": 100},  # max is 20
        )
        assert response.status_code == 422

    def test_ontology_request_validates_version_format(self, client: TestClient) -> None:
        """Ontology request with invalid version should be rejected."""
        response = client.post(
            "/api/v1/context/ontology",
            json={
                "name": "Test",
                "domain": "general",
                "schema_definition": {},
                "entity_types": ["person"],
                "relationship_types": ["related_to"],
                "version": "invalid-version",
            },
        )
        assert response.status_code == 422
