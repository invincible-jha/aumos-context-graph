"""Alembic environment configuration for aumos-context-graph."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from aumos_context_graph.core.models import (  # noqa: F401 — imported to register ORM metadata
    Document,
    DocumentChunk,
    EntityRelationship,
    GraphEntity,
    Ontology,
    SearchLog,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from aumos_common.database import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode without a database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
