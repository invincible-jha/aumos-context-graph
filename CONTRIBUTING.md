# Contributing to aumos-context-graph

## Development Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d
make migrate
```

## Code Standards

- Python 3.11+, type hints on all functions
- `ruff` for linting and formatting (`make lint`, `make format`)
- `mypy` strict mode (`make typecheck`)
- Google-style docstrings on all public APIs
- Tests alongside implementation (`make test`)

## Commit Convention

```
feat: add Milvus vector store adapter
fix: handle AGE graph creation race condition
refactor: extract embedding retry logic to base class
test: add integration tests for RAG pipeline
```

## Pull Request Process

1. Branch from `main`: `feature/`, `fix/`, `docs/`
2. Run `make all` (lint + typecheck + test) before opening PR
3. PRs require passing CI and one reviewer approval
4. Squash-merge to keep history clean

## Architecture Rules

- No business logic in API routes — delegate to services
- Services accept dependencies via constructor injection
- Repositories implement interface protocols
- All I/O must be async
- Never bypass RLS — all queries must respect tenant isolation
