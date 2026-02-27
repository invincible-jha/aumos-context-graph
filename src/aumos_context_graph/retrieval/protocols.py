"""Protocol interfaces for retrieval adapters.

These Protocol classes define the contracts that vector store and graph store
adapters must satisfy. They enable dependency injection and allow test doubles
to be used in place of real backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Contract for vector similarity search backends.

    Implementations include pgvector adapters, Milvus adapters, and in-memory
    test doubles.
    """

    async def search(
        self,
        embedding: list[float],
        k: int,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for documents by vector similarity.

        Args:
            embedding: Query embedding vector.
            k: Number of top results to return.
            threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of dicts with at minimum 'document_id', 'content',
            and 'similarity' keys.
        """
        ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Contract for graph database backends.

    Implementations include Apache AGE adapters, Neo4j adapters, and in-memory
    test doubles.
    """

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None,
        max_hops: int,
    ) -> list[dict]:
        """Retrieve graph neighbors within max_hops of the given node.

        Args:
            node_id: Starting node identifier (document_id).
            relationship_types: Optional filter for specific relationship types.
                If None, all relationship types are traversed.
            max_hops: Maximum number of hops from the starting node.

        Returns:
            List of dicts with at minimum 'document_id', 'relationship',
            and 'hops' keys.
        """
        ...


__all__ = ["GraphStoreProtocol", "VectorStoreProtocol"]
