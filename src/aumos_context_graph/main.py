"""AumOS Context Graph service entry point.

Knowledge graph + RAG pipeline + vector search + ontology management.
Unified search API consumed by all 9 AumOS products.

Port: 8000
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database

from aumos_context_graph.api.router import router
from aumos_context_graph.settings import Settings

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    # Startup
    init_database(settings.database)
    # TODO: Initialize Kafka publisher for domain events
    # TODO: Verify pgvector extension is enabled
    # TODO: Initialize Apache AGE search_path
    yield
    # Shutdown
    # TODO: Flush pending Kafka messages
    # TODO: Close graph database connections


app: FastAPI = create_app(
    service_name="aumos-context-graph",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db),
        # HealthCheck(name="pgvector", check_fn=check_pgvector),
    ],
)

app.include_router(router, prefix="/api/v1")
