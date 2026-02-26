# CLAUDE.md — AumOS Context Graph

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-context-graph`) is part of **Phase 4: Standalone Services**:
Knowledge graph and RAG pipeline shared across all AumOS products.

**Release Tier:** B — Open Core
**Product Mapping:** Shared Service — consumed by all 9 AumOS products
**Phase:** 4 (Months 22-30)

## Repo Purpose

Provides a unified knowledge graph (Apache AGE / Neo4j), RAG pipeline (LlamaIndex-compatible),
pgvector-based semantic search, ontology management, and entity resolution as shared services.
All 9 AumOS products consume this via the `/api/v1/context/*` endpoints for knowledge-enriched
AI features.

## Architecture Position

```
aumos-platform-core
    └── aumos-auth-gateway
            └── aumos-event-bus
                    └── THIS REPO (aumos-context-graph)
                            ├── aumos-governance-engine (consumes ontology + graph)
                            ├── aumos-agent-framework (consumes RAG + search)
                            ├── aumos-llm-serving (consumes document context)
                            └── all 9 products (consume unified search)
```

**Upstream dependencies:**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-event-bus` — Kafka topics for document.indexed, entity.created events

**Downstream dependents:**
- All 9 AumOS products consume the `/api/v1/context/*` endpoints
- `aumos-agent-framework` — uses RAG for agent context retrieval
- `aumos-governance-engine` — uses ontology for policy entity resolution

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| pgvector | 0.3+ | Vector similarity search in PostgreSQL |
| Apache AGE | — | Graph database on PostgreSQL (openCypher) |
| httpx | 0.27+ | HTTP client for embedding API calls |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.**
   ```python
   from aumos_common.auth import get_current_user, TenantContext
   from aumos_common.database import get_db_session, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ValidationError
   from aumos_common.observability import get_logger
   ```

2. **Type hints on EVERY function.**

3. **Pydantic models for ALL API inputs/outputs.**

4. **RLS tenant isolation via aumos-common.** Never bypass RLS.

5. **Structured logging via structlog.** Never use print().

6. **Publish domain events after state changes** — document.indexed, entity.created.

7. **Async by default.** All I/O must be async.

8. **Google-style docstrings** on all public classes and functions.

## API Conventions

- All endpoints under `/api/v1/context/` prefix
- Auth: Bearer JWT (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)

## Database Conventions

- Table prefix: `ctx_`
- All tables extend `AumOSModel` (id, tenant_id, created_at, updated_at)
- RLS policy on every table enforcing `app.current_tenant`
- pgvector columns on `ctx_chunks.embedding` and `ctx_entities.embedding` (1536 dims)
- IVFFlat indexes for approximate nearest neighbor search

## Graph Database (Apache AGE)

- Primary backend: Apache AGE on PostgreSQL (no separate server needed)
- Graph is tenant-scoped: one graph per tenant named `aumos_ctx_{tenant_id}`
- Queries executed via `cypher()` function in PostgreSQL
- Fallback to relational storage if AGE extension unavailable

## Vector Search (pgvector)

- Primary backend: pgvector on PostgreSQL
- Cosine similarity for semantic search
- IVFFlat index with lists=100 for approximate search
- Future: Milvus for billion-scale (configured via AUMOS_CONTEXT_VECTOR_BACKEND)

## Embedding Configuration

- Default model: `text-embedding-3-small` (1536 dimensions)
- Provider-agnostic: configured via `AUMOS_CONTEXT_EMBEDDING_MODEL`
- Batch processing: configurable batch size to manage API rate limits
- Dev fallback: zero vectors when no API key configured

## Repo-Specific Context

### Services

- **DocumentService**: Ingest documents, split into chunks, generate embeddings, deduplicate by content hash
- **VectorSearchService**: Semantic search over chunks and entities using pgvector cosine similarity
- **GraphService**: Entity CRUD + Cypher queries via Apache AGE
- **RAGService**: Retrieve context + construct LLM prompt for RAG responses
- **OntologyService**: CRUD for domain ontology definitions with entity/relationship type validation
- **EntityResolverService**: Cross-document entity resolution via exact name + semantic similarity fallback

### Key Patterns

- Content deduplication: SHA-256 hash of document content prevents re-indexing
- Tenant graph isolation: separate AGE graph per tenant (`aumos_ctx_{tenant_id}`)
- Graceful degradation: if graph DB unavailable, entity still stored relationally
- Search logging: all queries logged to `ctx_search_logs` for analytics

## What Claude Code Should NOT Do

1. **Do NOT reimplement aumos-common utilities** — auth, sessions, logging, errors.
2. **Do NOT bypass RLS** — all queries must be tenant-scoped.
3. **Do NOT hardcode API keys** — use Pydantic settings with env vars.
4. **Do NOT return raw dicts** from API endpoints.
5. **Do NOT write sync I/O** — everything must be async.
6. **Do NOT add AGPL/GPL dependencies** without explicit approval.
7. **Do NOT store embeddings outside PostgreSQL** (pgvector) without updating the vector backend abstraction.
8. **Do NOT write Cypher queries that bypass tenant scoping** — always pass `tenant_id` to graph client.
