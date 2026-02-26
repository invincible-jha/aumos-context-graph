"""Tests for core services in aumos-context-graph."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_context_graph.core.services import (
    DocumentService,
    EntityResolverService,
    GraphService,
    OntologyService,
    RAGService,
    VectorSearchService,
)


class TestDocumentService:
    """Tests for DocumentService."""

    @pytest.fixture
    def service(self, mock_document_repository: MagicMock, mock_embedding_engine: MagicMock) -> DocumentService:
        """Provide DocumentService with mocked dependencies."""
        return DocumentService(mock_document_repository, mock_embedding_engine)

    async def test_ingest_document_creates_record(
        self,
        service: DocumentService,
        mock_document_repository: MagicMock,
        mock_embedding_engine: MagicMock,
        tenant: object,
    ) -> None:
        """Ingest should create a document record and return indexed status."""
        from datetime import datetime, timezone

        created_doc = MagicMock()
        created_doc.id = uuid.uuid4()
        created_doc.status = "pending"
        created_doc.created_at = datetime.now(timezone.utc)
        created_doc.updated_at = datetime.now(timezone.utc)

        indexed_doc = MagicMock()
        indexed_doc.id = created_doc.id
        indexed_doc.status = "indexed"
        indexed_doc.chunk_count = 1
        indexed_doc.created_at = datetime.now(timezone.utc)
        indexed_doc.updated_at = datetime.now(timezone.utc)

        mock_document_repository.get_by_content_hash = AsyncMock(return_value=None)
        mock_document_repository.create = AsyncMock(return_value=created_doc)
        mock_document_repository.create_chunk = AsyncMock(return_value=MagicMock())
        mock_document_repository.update_status = AsyncMock(return_value=indexed_doc)
        mock_embedding_engine.embed_batch = AsyncMock(return_value=[[0.0] * 1536])

        result = await service.ingest_document(
            title="Test Document",
            content="This is test content for the document ingestion test.",
            tenant=tenant,
        )

        assert result.status == "indexed"
        mock_document_repository.create.assert_called_once()
        mock_document_repository.update_status.assert_called_once()

    async def test_ingest_duplicate_document_returns_existing(
        self,
        service: DocumentService,
        mock_document_repository: MagicMock,
        tenant: object,
    ) -> None:
        """Ingesting a duplicate document by content hash should return existing."""
        existing_doc = MagicMock()
        existing_doc.status = "indexed"

        mock_document_repository.get_by_content_hash = AsyncMock(return_value=existing_doc)

        result = await service.ingest_document(
            title="Duplicate Document",
            content="Same content as before",
            tenant=tenant,
        )

        assert result is existing_doc
        mock_document_repository.create.assert_not_called()

    async def test_get_document_raises_not_found(
        self,
        service: DocumentService,
        mock_document_repository: MagicMock,
        tenant: object,
    ) -> None:
        """get_document should raise NotFoundError when document does not exist."""
        from aumos_common.errors import NotFoundError

        mock_document_repository.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await service.get_document(uuid.uuid4(), tenant)

    def test_split_into_chunks_basic(self, service: DocumentService) -> None:
        """Chunking should split content into multiple chunks."""
        content = " ".join([f"word{i}" for i in range(200)])
        chunks = service._split_into_chunks(content, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["token_count"] <= 50
            assert len(chunk["content"]) > 0

    def test_split_into_chunks_short_content(self, service: DocumentService) -> None:
        """Short content should produce exactly one chunk."""
        content = "short content"
        chunks = service._split_into_chunks(content, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1


class TestVectorSearchService:
    """Tests for VectorSearchService."""

    @pytest.fixture
    def service(
        self,
        mock_vector_store: MagicMock,
        mock_embedding_engine: MagicMock,
        mock_search_log_repository: MagicMock,
    ) -> VectorSearchService:
        """Provide VectorSearchService with mocked dependencies."""
        return VectorSearchService(mock_vector_store, mock_embedding_engine, mock_search_log_repository)

    async def test_search_returns_structured_result(
        self,
        service: VectorSearchService,
        mock_vector_store: MagicMock,
        mock_embedding_engine: MagicMock,
        mock_search_log_repository: MagicMock,
        tenant: object,
    ) -> None:
        """Search should return dict with expected keys."""
        mock_vector_store.search_similar_chunks = AsyncMock(return_value=[])
        mock_vector_store.search_similar_entities = AsyncMock(return_value=[])

        result = await service.search(query="test query", tenant=tenant)

        assert "query" in result
        assert "chunks" in result
        assert "entities" in result
        assert "total" in result
        assert "latency_ms" in result
        assert result["query"] == "test query"
        assert result["total"] == 0

    async def test_search_logs_query(
        self,
        service: VectorSearchService,
        mock_search_log_repository: MagicMock,
        tenant: object,
    ) -> None:
        """Search should log the query to the search log repository."""
        await service.search(query="logged query", tenant=tenant)

        mock_search_log_repository.create.assert_called_once()
        call_kwargs = mock_search_log_repository.create.call_args.kwargs
        assert call_kwargs["query_text"] == "logged query"


class TestGraphService:
    """Tests for GraphService."""

    @pytest.fixture
    def service(
        self,
        mock_entity_repository: MagicMock,
        mock_graph_client: MagicMock,
        mock_embedding_engine: MagicMock,
    ) -> GraphService:
        """Provide GraphService with mocked dependencies."""
        return GraphService(mock_entity_repository, mock_graph_client, mock_embedding_engine)

    async def test_create_entity_creates_record_and_embedding(
        self,
        service: GraphService,
        mock_entity_repository: MagicMock,
        mock_embedding_engine: MagicMock,
        tenant: object,
    ) -> None:
        """create_entity should persist entity and generate embedding."""
        created_entity = MagicMock()
        created_entity.id = uuid.uuid4()
        created_entity.name = "Acme Corp"
        created_entity.entity_type = "organization"

        mock_entity_repository.create = AsyncMock(return_value=created_entity)
        mock_entity_repository.update_embedding = AsyncMock()

        result = await service.create_entity(
            name="Acme Corp",
            entity_type="organization",
            tenant=tenant,
        )

        assert result is created_entity
        mock_entity_repository.update_embedding.assert_called_once()

    async def test_get_entity_raises_not_found(
        self,
        service: GraphService,
        mock_entity_repository: MagicMock,
        tenant: object,
    ) -> None:
        """get_entity_with_relationships should raise NotFoundError when entity missing."""
        from aumos_common.errors import NotFoundError

        mock_entity_repository.get_with_relationships = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await service.get_entity_with_relationships(uuid.uuid4(), tenant)

    async def test_execute_graph_query_delegates_to_client(
        self,
        service: GraphService,
        mock_graph_client: MagicMock,
        tenant: object,
    ) -> None:
        """execute_graph_query should delegate to graph client."""
        mock_graph_client.execute_cypher = AsyncMock(return_value=[{"result": "test"}])

        results = await service.execute_graph_query(
            cypher_query="MATCH (n) RETURN n",
            parameters=None,
            tenant=tenant,
        )

        assert results == [{"result": "test"}]
        mock_graph_client.execute_cypher.assert_called_once()


class TestOntologyService:
    """Tests for OntologyService."""

    @pytest.fixture
    def service(self, mock_ontology_repository: MagicMock) -> OntologyService:
        """Provide OntologyService with mocked dependencies."""
        return OntologyService(mock_ontology_repository)

    async def test_create_ontology_validates_entity_types(
        self,
        service: OntologyService,
        tenant: object,
    ) -> None:
        """create_ontology should raise ValidationError when entity_types is empty."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            await service.create_ontology(
                name="Test Ontology",
                domain="general",
                schema_definition={},
                entity_types=[],
                relationship_types=["related_to"],
                tenant=tenant,
            )

    async def test_create_ontology_validates_relationship_types(
        self,
        service: OntologyService,
        tenant: object,
    ) -> None:
        """create_ontology should raise ValidationError when relationship_types is empty."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            await service.create_ontology(
                name="Test Ontology",
                domain="general",
                schema_definition={},
                entity_types=["person"],
                relationship_types=[],
                tenant=tenant,
            )

    async def test_get_ontology_raises_not_found(
        self,
        service: OntologyService,
        mock_ontology_repository: MagicMock,
        tenant: object,
    ) -> None:
        """get_ontology should raise NotFoundError when ontology is missing."""
        from aumos_common.errors import NotFoundError

        mock_ontology_repository.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await service.get_ontology(uuid.uuid4(), tenant)


class TestEntityResolverService:
    """Tests for EntityResolverService."""

    @pytest.fixture
    def service(
        self,
        mock_entity_repository: MagicMock,
        mock_embedding_engine: MagicMock,
        mock_vector_store: MagicMock,
    ) -> EntityResolverService:
        """Provide EntityResolverService with mocked dependencies."""
        return EntityResolverService(
            mock_entity_repository,
            mock_embedding_engine,
            mock_vector_store,
        )

    async def test_resolve_entity_by_exact_name_match(
        self,
        service: EntityResolverService,
        mock_entity_repository: MagicMock,
        tenant: object,
    ) -> None:
        """resolve_entity should return match on exact name."""
        existing_entity = MagicMock()
        existing_entity.id = uuid.uuid4()
        existing_entity.name = "Acme Corporation"

        mock_entity_repository.find_by_name = AsyncMock(return_value=[existing_entity])

        result = await service.resolve_entity(
            name="Acme Corporation",
            entity_type="organization",
            tenant=tenant,
        )

        assert result is existing_entity

    async def test_resolve_entity_returns_none_on_no_match(
        self,
        service: EntityResolverService,
        mock_entity_repository: MagicMock,
        mock_vector_store: MagicMock,
        tenant: object,
    ) -> None:
        """resolve_entity should return None when no match is found."""
        mock_entity_repository.find_by_name = AsyncMock(return_value=[])
        mock_vector_store.search_similar_entities = AsyncMock(return_value=[])

        result = await service.resolve_entity(
            name="Unknown Entity",
            entity_type="person",
            tenant=tenant,
        )

        assert result is None
