# Changelog

All notable changes to aumos-context-graph are documented here.

## [0.1.0] — 2026-02-26

### Added

- Initial implementation of knowledge graph + RAG pipeline + vector search service
- `DocumentService`: document ingestion with chunking, embedding, and SHA-256 deduplication
- `VectorSearchService`: pgvector cosine similarity search over chunks and entities
- `GraphService`: entity CRUD + Cypher query execution via Apache AGE
- `RAGService`: retrieval-augmented generation context preparation
- `OntologyService`: domain ontology CRUD with entity/relationship type validation
- `EntityResolverService`: cross-document entity resolution via name + semantic similarity
- 10 REST endpoints under `/api/v1/context/`
- 6 database tables with RLS tenant isolation (`ctx_` prefix)
- pgvector IVFFlat indexes for approximate nearest neighbor search
- Apache AGE adapter for openCypher graph queries on PostgreSQL
- OpenAI-compatible embedding engine with batch processing
- Alembic migration `001_ctx_initial_schema` enabling pgvector and all ctx_ tables
- Domain events: `document.indexed`, `entity.created`, `search.performed`
- Full hexagonal architecture: api/ + core/ + adapters/
- Comprehensive test suite for all services
