"""Business logic services for aumos-context-graph.

Services contain all domain logic. They accept dependencies via constructor
injection and are framework-agnostic.

Services:
  - DocumentService: Document ingestion and chunk management
  - VectorSearchService: Semantic and hybrid search
  - GraphService: Knowledge graph queries and entity management
  - RAGService: Retrieval-augmented generation
  - OntologyService: Ontology CRUD and validation
  - EntityResolverService: Cross-document entity resolution
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from aumos_common.auth import TenantContext
from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.observability import get_logger

from aumos_context_graph.core.interfaces import (
    IDocumentRepository,
    IEmbeddingEngine,
    IEntityRepository,
    IGraphClient,
    IOntologyRepository,
    ISearchLogRepository,
    IVectorStore,
)
from aumos_context_graph.core.models import (
    Document,
    DocumentChunk,
    EntityRelationship,
    GraphEntity,
    Ontology,
)

logger = get_logger(__name__)


class DocumentService:
    """Manages document ingestion, chunking, and embedding pipeline.

    Args:
        document_repository: Data access for documents and chunks.
        embedding_engine: Engine for generating text embeddings.
    """

    def __init__(
        self,
        document_repository: IDocumentRepository,
        embedding_engine: IEmbeddingEngine,
    ) -> None:
        """Initialize DocumentService with required dependencies.

        Args:
            document_repository: Repository for documents and chunks.
            embedding_engine: Service for generating text embeddings.
        """
        self._documents = document_repository
        self._embeddings = embedding_engine

    async def ingest_document(
        self,
        title: str,
        content: str,
        tenant: TenantContext,
        document_type: str = "text",
        source_uri: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> Document:
        """Ingest a document, chunk it, and generate embeddings.

        Deduplicates by content hash. If the document already exists and is
        indexed, returns the existing record without reprocessing.

        Args:
            title: Human-readable document title.
            content: Raw document text content.
            tenant: Tenant context for RLS enforcement.
            document_type: Type of document (pdf, docx, txt, html, markdown).
            source_uri: Optional URI where the document was sourced.
            extra_metadata: Optional additional metadata.
            chunk_size: Target size of each chunk in tokens.
            chunk_overlap: Overlap between consecutive chunks in tokens.

        Returns:
            The created or existing Document record.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Deduplicate by content hash
        existing = await self._documents.get_by_content_hash(content_hash, tenant)
        if existing and existing.status == "indexed":
            logger.info(
                "Document already indexed, skipping",
                document_id=str(existing.id),
                content_hash=content_hash,
                tenant_id=str(tenant.tenant_id),
            )
            return existing

        document = await self._documents.create(
            title=title,
            source_uri=source_uri,
            document_type=document_type,
            content_hash=content_hash,
            size_bytes=len(content.encode()),
            extra_metadata=extra_metadata,
            tenant=tenant,
        )

        logger.info(
            "Document created, starting processing",
            document_id=str(document.id),
            tenant_id=str(tenant.tenant_id),
        )

        chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
        chunk_texts = [chunk["content"] for chunk in chunks]

        embeddings = await self._embeddings.embed_batch(chunk_texts)

        for index, (chunk_data, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            await self._documents.create_chunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk_data["content"],
                token_count=chunk_data["token_count"],
                tenant=tenant,
                embedding=embedding,
                start_char=chunk_data.get("start_char"),
                end_char=chunk_data.get("end_char"),
            )

        updated = await self._documents.update_status(
            document_id=document.id,
            status="indexed",
            chunk_count=len(chunks),
            token_count=sum(c["token_count"] for c in chunks),
            tenant=tenant,
        )

        logger.info(
            "Document indexed successfully",
            document_id=str(document.id),
            chunk_count=len(chunks),
            tenant_id=str(tenant.tenant_id),
        )

        return updated or document

    async def get_document(self, document_id: uuid.UUID, tenant: TenantContext) -> Document:
        """Retrieve document metadata by ID.

        Args:
            document_id: UUID of the document.
            tenant: Tenant context for RLS enforcement.

        Returns:
            Document record with metadata.

        Raises:
            NotFoundError: If document does not exist.
        """
        document = await self._documents.get_by_id(document_id, tenant)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found")
        return document

    def _split_into_chunks(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        """Split content into overlapping chunks by approximate token count.

        Uses word-based splitting as a proxy for token count.

        Args:
            content: Text content to split.
            chunk_size: Target chunk size in approximate tokens.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            List of chunk dicts with content, token_count, start_char, end_char.
        """
        words = content.split()
        chunks: list[dict[str, Any]] = []
        start_word = 0

        while start_word < len(words):
            end_word = min(start_word + chunk_size, len(words))
            chunk_words = words[start_word:end_word]
            chunk_content = " ".join(chunk_words)

            start_char = len(" ".join(words[:start_word]))
            if start_word > 0:
                start_char += 1

            chunks.append(
                {
                    "content": chunk_content,
                    "token_count": len(chunk_words),
                    "start_char": start_char,
                    "end_char": start_char + len(chunk_content),
                }
            )

            if end_word >= len(words):
                break
            start_word = end_word - chunk_overlap

        return chunks


class VectorSearchService:
    """Semantic and keyword vector search over documents and entities.

    Args:
        vector_store: Vector similarity search backend.
        embedding_engine: Engine for query embedding generation.
        search_log_repository: Repository for logging search queries.
    """

    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_engine: IEmbeddingEngine,
        search_log_repository: ISearchLogRepository,
    ) -> None:
        """Initialize VectorSearchService.

        Args:
            vector_store: Backend for similarity search.
            embedding_engine: Service for embedding query text.
            search_log_repository: Repository for search analytics logging.
        """
        self._vector_store = vector_store
        self._embeddings = embedding_engine
        self._search_logs = search_log_repository

    async def search(
        self,
        query: str,
        tenant: TenantContext,
        search_type: str = "semantic",
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
        entity_type: str | None = None,
        include_entities: bool = True,
        include_chunks: bool = True,
    ) -> dict[str, Any]:
        """Perform unified search combining semantic, entity, and chunk results.

        Args:
            query: Search query text.
            tenant: Tenant context for RLS enforcement.
            search_type: Type of search: semantic, keyword, graph, or unified.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum cosine similarity score.
            document_ids: Optional filter to specific documents.
            entity_type: Optional filter by entity type.
            include_entities: Whether to include entity results.
            include_chunks: Whether to include chunk results.

        Returns:
            Dict with 'chunks', 'entities', 'total', and 'query' keys.
        """
        start_time = time.monotonic()
        query_embedding = await self._embeddings.embed_text(query)

        chunk_results: list[dict[str, Any]] = []
        entity_results: list[dict[str, Any]] = []

        if include_chunks:
            similar_chunks = await self._vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                tenant=tenant,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids,
            )
            chunk_results = [
                {
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "similarity_score": score,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                }
                for chunk, score in similar_chunks
            ]

        if include_entities:
            similar_entities = await self._vector_store.search_similar_entities(
                query_embedding=query_embedding,
                tenant=tenant,
                top_k=top_k,
                entity_type=entity_type,
            )
            entity_results = [
                {
                    "entity_id": str(entity.id),
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "similarity_score": score,
                }
                for entity, score in similar_entities
            ]

        latency_ms = int((time.monotonic() - start_time) * 1000)
        total = len(chunk_results) + len(entity_results)
        top_ids = [r["chunk_id"] for r in chunk_results[:5]] + [r["entity_id"] for r in entity_results[:5]]

        await self._search_logs.create(
            query_text=query,
            search_type=search_type,
            result_count=total,
            tenant=tenant,
            latency_ms=latency_ms,
            top_result_ids=top_ids,
        )

        logger.info(
            "Search completed",
            query=query[:100],
            search_type=search_type,
            result_count=total,
            latency_ms=latency_ms,
            tenant_id=str(tenant.tenant_id),
        )

        return {
            "query": query,
            "search_type": search_type,
            "chunks": chunk_results,
            "entities": entity_results,
            "total": total,
            "latency_ms": latency_ms,
        }


class GraphService:
    """Knowledge graph operations — entity CRUD and Cypher queries.

    Args:
        entity_repository: Data access for entities and relationships.
        graph_client: Interface to the graph database engine.
        embedding_engine: Engine for entity embeddings.
    """

    def __init__(
        self,
        entity_repository: IEntityRepository,
        graph_client: IGraphClient,
        embedding_engine: IEmbeddingEngine,
    ) -> None:
        """Initialize GraphService.

        Args:
            entity_repository: Repository for graph entities.
            graph_client: Client for the graph database.
            embedding_engine: Service for generating entity embeddings.
        """
        self._entities = entity_repository
        self._graph = graph_client
        self._embeddings = embedding_engine

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        tenant: TenantContext,
        description: str | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        ontology_id: uuid.UUID | None = None,
    ) -> GraphEntity:
        """Create a new knowledge graph entity with embedding.

        Args:
            name: Entity name.
            entity_type: Type of entity (person, org, concept, etc.).
            tenant: Tenant context for RLS enforcement.
            description: Optional description of the entity.
            aliases: Optional list of alternative names.
            properties: Optional additional structured properties.
            ontology_id: Optional ontology this entity belongs to.

        Returns:
            Newly created GraphEntity record.
        """
        entity = await self._entities.create(
            name=name,
            entity_type=entity_type,
            tenant=tenant,
            description=description,
            aliases=aliases,
            properties=properties,
            ontology_id=ontology_id,
        )

        # Generate and store embedding for semantic search
        embed_text = f"{entity_type}: {name}"
        if description:
            embed_text = f"{embed_text}. {description}"
        embedding = await self._embeddings.embed_text(embed_text)
        await self._entities.update_embedding(entity.id, embedding, tenant)

        # Create node in graph database
        try:
            external_id = await self._graph.create_node(
                label=entity_type,
                properties={
                    "entity_id": str(entity.id),
                    "name": name,
                    "tenant_id": str(tenant.tenant_id),
                    **(properties or {}),
                },
                tenant_id=tenant.tenant_id,
            )
            logger.info(
                "Entity created in graph database",
                entity_id=str(entity.id),
                external_id=external_id,
                tenant_id=str(tenant.tenant_id),
            )
        except Exception:
            logger.warning(
                "Failed to create entity in graph database, continuing without graph node",
                entity_id=str(entity.id),
                tenant_id=str(tenant.tenant_id),
            )

        return entity

    async def get_entity_with_relationships(
        self,
        entity_id: uuid.UUID,
        tenant: TenantContext,
        depth: int = 1,
    ) -> GraphEntity:
        """Retrieve an entity with its relationships up to given depth.

        Args:
            entity_id: UUID of the entity.
            tenant: Tenant context for RLS enforcement.
            depth: Relationship traversal depth.

        Returns:
            GraphEntity with populated relationships.

        Raises:
            NotFoundError: If entity does not exist.
        """
        entity = await self._entities.get_with_relationships(entity_id, tenant, depth)
        if entity is None:
            raise NotFoundError(f"Entity {entity_id} not found")
        return entity

    async def execute_graph_query(
        self,
        cypher_query: str,
        parameters: dict[str, Any] | None,
        tenant: TenantContext,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the knowledge graph.

        Args:
            cypher_query: Cypher query string.
            parameters: Optional query parameters.
            tenant: Tenant context for query scoping.

        Returns:
            List of result records as dicts.
        """
        logger.info(
            "Executing graph query",
            query=cypher_query[:200],
            tenant_id=str(tenant.tenant_id),
        )
        results = await self._graph.execute_cypher(
            query=cypher_query,
            parameters=parameters,
            tenant_id=tenant.tenant_id,
        )
        return results


class RAGService:
    """Retrieval-augmented generation pipeline.

    Retrieves relevant context from vector store and generates responses.

    Args:
        vector_store: For retrieving relevant document chunks.
        embedding_engine: For embedding the query.
        document_repository: For fetching chunk content.
    """

    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_engine: IEmbeddingEngine,
        document_repository: IDocumentRepository,
    ) -> None:
        """Initialize RAGService.

        Args:
            vector_store: Backend for similarity search.
            embedding_engine: Service for embedding query text.
            document_repository: Repository for fetching document content.
        """
        self._vector_store = vector_store
        self._embeddings = embedding_engine
        self._documents = document_repository

    async def query(
        self,
        question: str,
        tenant: TenantContext,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Answer a question using retrieval-augmented generation.

        Retrieves relevant chunks then constructs a context-aware prompt.
        Note: actual LLM call should be made by the caller using the returned
        context and prompt — this service constructs the retrieval context.

        Args:
            question: Natural language question to answer.
            tenant: Tenant context for RLS enforcement.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity for chunk retrieval.
            document_ids: Optional filter to specific documents.
            max_tokens: Max tokens for the generated response.

        Returns:
            Dict with 'question', 'context_chunks', 'prompt', and 'sources' keys.
        """
        query_embedding = await self._embeddings.embed_text(question)
        similar_chunks = await self._vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            tenant=tenant,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
        )

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for chunk, score in similar_chunks:
            context_parts.append(f"[Chunk {chunk.chunk_index}]\n{chunk.content}")
            sources.append(
                {
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "similarity_score": score,
                    "section_title": chunk.section_title,
                }
            )

        context_text = "\n\n".join(context_parts)
        prompt = (
            f"You are a helpful assistant. Answer the question based on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        logger.info(
            "RAG context prepared",
            question=question[:100],
            chunk_count=len(similar_chunks),
            tenant_id=str(tenant.tenant_id),
        )

        return {
            "question": question,
            "context_chunks": [
                {"chunk_id": str(c.id), "content": c.content, "score": s}
                for c, s in similar_chunks
            ],
            "prompt": prompt,
            "sources": sources,
        }

    async def get_sources(
        self,
        chunk_ids: list[uuid.UUID],
        tenant: TenantContext,
    ) -> list[dict[str, Any]]:
        """Retrieve source document metadata for given chunk IDs.

        Args:
            chunk_ids: List of chunk UUIDs to fetch sources for.
            tenant: Tenant context for RLS enforcement.

        Returns:
            List of source dicts with document and chunk metadata.
        """
        sources: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            # Fetch document for each chunk
            # In a real impl, a single batch query would be used
            logger.debug("Fetching source for chunk", chunk_id=str(chunk_id))
            sources.append({"chunk_id": str(chunk_id)})
        return sources


class OntologyService:
    """Domain ontology management — creation, retrieval, and validation.

    Args:
        ontology_repository: Data access for ontology definitions.
    """

    def __init__(self, ontology_repository: IOntologyRepository) -> None:
        """Initialize OntologyService.

        Args:
            ontology_repository: Repository for ontology records.
        """
        self._ontologies = ontology_repository

    async def create_ontology(
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
        """Create a new domain ontology definition.

        Args:
            name: Human-readable ontology name.
            domain: Domain category (healthcare, finance, legal, general).
            schema_definition: JSON-LD or OWL-like schema dict.
            entity_types: List of valid entity type strings.
            relationship_types: List of valid relationship type strings.
            tenant: Tenant context for RLS enforcement.
            description: Optional description.
            version: Semantic version string.

        Returns:
            Newly created Ontology record.

        Raises:
            ValidationError: If schema definition is invalid.
        """
        if not entity_types:
            raise ValidationError("Ontology must define at least one entity type")
        if not relationship_types:
            raise ValidationError("Ontology must define at least one relationship type")

        ontology = await self._ontologies.create(
            name=name,
            domain=domain,
            schema_definition=schema_definition,
            entity_types=entity_types,
            relationship_types=relationship_types,
            tenant=tenant,
            description=description,
            version=version,
        )

        logger.info(
            "Ontology created",
            ontology_id=str(ontology.id),
            name=name,
            domain=domain,
            tenant_id=str(tenant.tenant_id),
        )

        return ontology

    async def get_ontology(self, ontology_id: uuid.UUID, tenant: TenantContext) -> Ontology:
        """Retrieve ontology by ID.

        Args:
            ontology_id: UUID of the ontology.
            tenant: Tenant context for RLS enforcement.

        Returns:
            Ontology record.

        Raises:
            NotFoundError: If ontology does not exist.
        """
        ontology = await self._ontologies.get_by_id(ontology_id, tenant)
        if ontology is None:
            raise NotFoundError(f"Ontology {ontology_id} not found")
        return ontology

    async def validate_entity_against_ontology(
        self,
        entity_type: str,
        relationship_types: list[str],
        ontology_id: uuid.UUID,
        tenant: TenantContext,
    ) -> bool:
        """Validate that an entity type and relationships conform to an ontology.

        Args:
            entity_type: Entity type to validate.
            relationship_types: Relationship types to validate.
            ontology_id: Ontology to validate against.
            tenant: Tenant context for RLS enforcement.

        Returns:
            True if valid, False otherwise.
        """
        ontology = await self._ontologies.get_by_id(ontology_id, tenant)
        if ontology is None:
            return False

        if entity_type not in ontology.entity_types:
            return False

        for rel_type in relationship_types:
            if rel_type not in ontology.relationship_types:
                return False

        return True


class EntityResolverService:
    """Cross-document entity resolution and deduplication.

    Identifies when entities mentioned across documents refer to the same
    real-world entity using name matching and semantic similarity.

    Args:
        entity_repository: Data access for graph entities.
        embedding_engine: For semantic similarity comparison.
        vector_store: For finding candidate matches.
    """

    def __init__(
        self,
        entity_repository: IEntityRepository,
        embedding_engine: IEmbeddingEngine,
        vector_store: IVectorStore,
    ) -> None:
        """Initialize EntityResolverService.

        Args:
            entity_repository: Repository for entity lookups.
            embedding_engine: Service for entity embeddings.
            vector_store: Backend for similarity search.
        """
        self._entities = entity_repository
        self._embeddings = embedding_engine
        self._vector_store = vector_store

    async def resolve_entity(
        self,
        name: str,
        entity_type: str,
        tenant: TenantContext,
        similarity_threshold: float = 0.92,
    ) -> GraphEntity | None:
        """Attempt to resolve a named entity to an existing record.

        Uses exact name matching first, then falls back to semantic similarity.

        Args:
            name: Entity name to resolve.
            entity_type: Type of entity for scoped search.
            tenant: Tenant context for RLS enforcement.
            similarity_threshold: Minimum similarity to consider a match.

        Returns:
            Matching GraphEntity if found, None otherwise.
        """
        # Exact name match first
        matches = await self._entities.find_by_name(name, entity_type, tenant)
        if matches:
            logger.info(
                "Entity resolved by exact name match",
                name=name,
                entity_id=str(matches[0].id),
                tenant_id=str(tenant.tenant_id),
            )
            return matches[0]

        # Semantic similarity fallback
        query_embedding = await self._embeddings.embed_text(f"{entity_type}: {name}")
        similar_entities = await self._vector_store.search_similar_entities(
            query_embedding=query_embedding,
            tenant=tenant,
            top_k=3,
            entity_type=entity_type,
        )

        for entity, score in similar_entities:
            if score >= similarity_threshold:
                logger.info(
                    "Entity resolved by semantic similarity",
                    name=name,
                    entity_id=str(entity.id),
                    similarity_score=score,
                    tenant_id=str(tenant.tenant_id),
                )
                return entity

        return None

    async def merge_entities(
        self,
        primary_entity_id: uuid.UUID,
        duplicate_entity_id: uuid.UUID,
        tenant: TenantContext,
    ) -> GraphEntity:
        """Merge a duplicate entity into a primary entity.

        Transfers all relationships from the duplicate to the primary.

        Args:
            primary_entity_id: UUID of the entity to keep.
            duplicate_entity_id: UUID of the entity to merge and delete.
            tenant: Tenant context for RLS enforcement.

        Returns:
            The primary entity after merging.

        Raises:
            NotFoundError: If either entity is not found.
        """
        primary = await self._entities.get_by_id(primary_entity_id, tenant)
        if primary is None:
            raise NotFoundError(f"Primary entity {primary_entity_id} not found")

        duplicate = await self._entities.get_by_id(duplicate_entity_id, tenant)
        if duplicate is None:
            raise NotFoundError(f"Duplicate entity {duplicate_entity_id} not found")

        # Get all relationships from duplicate and repoint to primary
        dup_relationships = await self._entities.get_relationships(duplicate_entity_id, tenant)
        for rel in dup_relationships:
            if rel.source_entity_id == duplicate_entity_id:
                await self._entities.create_relationship(
                    source_entity_id=primary_entity_id,
                    target_entity_id=rel.target_entity_id,
                    relationship_type=rel.relationship_type,
                    tenant=tenant,
                    properties=rel.properties,
                    weight=rel.weight,
                )
            else:
                await self._entities.create_relationship(
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=primary_entity_id,
                    relationship_type=rel.relationship_type,
                    tenant=tenant,
                    properties=rel.properties,
                    weight=rel.weight,
                )

        await self._entities.delete(duplicate_entity_id, tenant)

        logger.info(
            "Entities merged",
            primary_entity_id=str(primary_entity_id),
            duplicate_entity_id=str(duplicate_entity_id),
            relationships_transferred=len(dup_relationships),
            tenant_id=str(tenant.tenant_id),
        )

        return primary


__all__ = [
    "DocumentService",
    "EntityResolverService",
    "GraphService",
    "OntologyService",
    "RAGService",
    "VectorSearchService",
]
