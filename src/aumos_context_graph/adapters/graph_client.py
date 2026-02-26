"""Graph database client adapter for aumos-context-graph.

Supports Apache AGE (running on PostgreSQL) as primary backend.
Apache AGE implements openCypher on PostgreSQL, enabling graph queries
without requiring a separate graph database server.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# AGE graph name per tenant — prefixed to isolate tenant graph namespaces
_AGE_GRAPH_PREFIX = "aumos_ctx_"


class AGEGraphClient:
    """Apache AGE graph client running on PostgreSQL.

    Executes openCypher queries via AGE extension. Graph is tenant-scoped
    by using per-tenant graph names: aumos_ctx_{tenant_id}.

    Args:
        session: Async SQLAlchemy session connected to AGE-enabled PostgreSQL.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the AGE graph client.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    def _graph_name(self, tenant_id: uuid.UUID) -> str:
        """Construct the tenant-scoped graph name.

        Args:
            tenant_id: Tenant UUID.

        Returns:
            AGE graph name string.
        """
        return f"{_AGE_GRAPH_PREFIX}{str(tenant_id).replace('-', '_')}"

    async def _ensure_graph_exists(self, tenant_id: uuid.UUID) -> None:
        """Create AGE graph for tenant if it does not exist.

        Args:
            tenant_id: Tenant UUID.
        """
        graph_name = self._graph_name(tenant_id)
        try:
            await self._session.execute(
                text("SELECT create_graph(:graph_name)"),
                {"graph_name": graph_name},
            )
            await self._session.flush()
            logger.info("Created AGE graph", graph_name=graph_name)
        except Exception:
            # Graph already exists — this is expected
            pass

    async def execute_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the tenant's graph.

        Args:
            query: openCypher query string.
            parameters: Optional query parameters.
            tenant_id: Tenant UUID for graph scoping.

        Returns:
            List of result records as dicts.
        """
        if tenant_id is None:
            logger.warning("No tenant_id provided for graph query, returning empty results")
            return []

        graph_name = self._graph_name(tenant_id)
        await self._ensure_graph_exists(tenant_id)

        # AGE requires SET search_path before Cypher queries
        await self._session.execute(text("SET search_path = ag_catalog, '$user', public"))

        age_query = text(
            f"SELECT * FROM cypher('{graph_name}', $$ {query} $$) AS (result agtype)"
        )

        try:
            result = await self._session.execute(age_query, parameters or {})
            rows = result.fetchall()
            return [{"result": str(row[0])} for row in rows]
        except Exception as exc:
            logger.error(
                "Graph query failed",
                query=query[:200],
                error=str(exc),
                tenant_id=str(tenant_id),
            )
            return []

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> str:
        """Create a node in the tenant's knowledge graph.

        Args:
            label: Node label (entity type).
            properties: Node properties dict.
            tenant_id: Tenant UUID.

        Returns:
            AGE node ID as string.
        """
        await self._ensure_graph_exists(tenant_id)
        graph_name = self._graph_name(tenant_id)
        await self._session.execute(text("SET search_path = ag_catalog, '$user', public"))

        # Build property string for Cypher
        props_str = ", ".join(f"{k}: ${k}" for k in properties)
        cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN id(n)"

        try:
            result = await self._session.execute(
                text(f"SELECT * FROM cypher('{graph_name}', $$ {cypher} $$) AS (node_id agtype)"),
                properties,
            )
            row = result.fetchone()
            node_id = str(row[0]) if row else "unknown"
            logger.info(
                "Graph node created",
                label=label,
                node_id=node_id,
                tenant_id=str(tenant_id),
            )
            return node_id
        except Exception as exc:
            logger.error(
                "Failed to create graph node",
                label=label,
                error=str(exc),
                tenant_id=str(tenant_id),
            )
            return "error"

    async def create_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a directed relationship between two graph nodes.

        This method is tenant-unscoped intentionally — node IDs already
        encode tenant membership.

        Args:
            source_node_id: AGE ID of source node.
            target_node_id: AGE ID of target node.
            relationship_type: Relationship type label.
            properties: Optional relationship properties.

        Returns:
            AGE relationship ID as string.
        """
        props = properties or {}
        props_str = ", ".join(f"{k}: ${k}" for k in props) if props else ""
        props_clause = f" {{{props_str}}}" if props_str else ""
        cypher = (
            f"MATCH (s), (t) WHERE id(s) = {source_node_id} AND id(t) = {target_node_id} "
            f"CREATE (s)-[r:{relationship_type}{props_clause}]->(t) RETURN id(r)"
        )

        try:
            result = await self._session.execute(text(cypher), props)
            row = result.fetchone()
            return str(row[0]) if row else "unknown"
        except Exception as exc:
            logger.error(
                "Failed to create graph relationship",
                relationship_type=relationship_type,
                error=str(exc),
            )
            return "error"


__all__ = ["AGEGraphClient"]
