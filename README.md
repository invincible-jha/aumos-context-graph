# aumos-context-graph

Knowledge graph + RAG pipeline + vector search + ontology management.
Unified context search API consumed by all 9 AumOS products.

## Overview

`aumos-context-graph` is a standalone shared service that provides:

- **Knowledge Graph** — Apache AGE (openCypher on PostgreSQL) with per-tenant graph isolation
- **RAG Pipeline** — Document ingestion → chunking → embedding → semantic retrieval → prompt construction
- **Vector Search** — pgvector cosine similarity over document chunks and graph entities
- **Ontology Management** — Domain ontology CRUD with entity and relationship type validation
- **Entity Resolution** — Cross-document entity deduplication via exact name + semantic similarity

All 9 AumOS products consume this service via the `/api/v1/context/*` API.

## API Surface

```
POST   /api/v1/context/documents              # Ingest document (PDF, DOCX, TXT, HTML)
GET    /api/v1/context/documents/{id}         # Document metadata + chunks

POST   /api/v1/context/graph/query            # Cypher graph query
POST   /api/v1/context/graph/entities         # Create graph entity
GET    /api/v1/context/graph/entities/{id}    # Entity with relationships

POST   /api/v1/context/rag/query              # RAG query — retrieve + generate prompt
GET    /api/v1/context/rag/sources            # Source documents for response

POST   /api/v1/context/search                 # Unified search (keyword + semantic + graph)

POST   /api/v1/context/ontology               # Create domain ontology
GET    /api/v1/context/ontology/{id}          # Ontology detail
```

## Architecture

```
src/aumos_context_graph/
├── __init__.py
├── main.py                    # FastAPI app entry point
├── settings.py                # Settings with AUMOS_CONTEXT_ prefix
├── api/
│   ├── router.py              # All endpoints — delegates to services
│   └── schemas.py             # Pydantic request/response models
├── core/
│   ├── models.py              # SQLAlchemy ORM: ctx_ tables
│   ├── services.py            # DocumentService, VectorSearchService, GraphService,
│   │                          #   RAGService, OntologyService, EntityResolverService
│   └── interfaces.py          # Protocol interfaces for DI
├── adapters/
│   ├── repositories.py        # SQLAlchemy repositories
│   ├── embedding_engine.py    # OpenAI-compatible embedding client
│   ├── vector_store.py        # pgvector similarity search
│   ├── graph_client.py        # Apache AGE Cypher client
│   └── kafka.py               # Domain event publishing
└── migrations/
    └── versions/001_ctx_initial_schema.py
```

## Database Schema

| Table | Purpose |
|-------|---------|
| `ctx_documents` | Ingested documents with metadata |
| `ctx_chunks` | Document chunks with 1536-dim embeddings |
| `ctx_entities` | Graph entities with 1536-dim embeddings |
| `ctx_relationships` | Directed entity relationships |
| `ctx_ontologies` | Domain ontology definitions |
| `ctx_search_logs` | Search query analytics |

All tables have RLS policies enforcing tenant isolation via `app.current_tenant`.

## Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy env config
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and database URLs

# Start infrastructure (PostgreSQL with pgvector, Redis, Kafka)
docker compose -f docker-compose.dev.yml up -d postgres redis kafka

# Run migrations
make migrate

# Start service
uvicorn aumos_context_graph.main:app --reload --port 8000

# Run tests
make test
```

## Configuration

All settings are prefixed `AUMOS_CONTEXT_`. See `.env.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_CONTEXT_GRAPH_BACKEND` | `age` | Graph backend: `age` or `neo4j` |
| `AUMOS_CONTEXT_VECTOR_BACKEND` | `pgvector` | Vector backend: `pgvector` or `milvus` |
| `AUMOS_CONTEXT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `AUMOS_CONTEXT_EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions |
| `AUMOS_CONTEXT_RAG_TOP_K` | `5` | Chunks retrieved per RAG query |
| `AUMOS_CONTEXT_CHUNK_SIZE` | `512` | Chunk size in tokens |
| `AUMOS_CONTEXT_SEMANTIC_SIMILARITY_THRESHOLD` | `0.7` | Min cosine similarity |

## License

Apache 2.0 — see LICENSE.
