"""Service-specific settings for aumos-context-graph.

Extends AumOSSettings with context graph configuration.
Environment variable prefix: AUMOS_CONTEXT_
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Settings for aumos-context-graph.

    Inherits all standard AumOS settings (database, kafka, keycloak, etc.)
    and adds context graph specific configuration.

    Environment variable prefix: AUMOS_CONTEXT_
    """

    service_name: str = "aumos-context-graph"

    # Graph database (Neo4j or Apache AGE via PostgreSQL)
    graph_backend: str = Field(default="age", description="Graph backend: 'neo4j' or 'age'")
    neo4j_url: str = Field(default="bolt://localhost:7687", description="Neo4j connection URL")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")

    # Vector store
    vector_backend: str = Field(default="pgvector", description="Vector backend: 'pgvector' or 'milvus'")
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")

    # Embedding model
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding generation")

    # LLM for RAG
    llm_model: str = Field(
        default="claude-sonnet-4-6",
        description="LLM model for RAG generation",
    )
    rag_max_tokens: int = Field(default=2048, description="Max tokens for RAG response")
    rag_top_k: int = Field(default=5, description="Number of chunks to retrieve for RAG")

    # Document processing
    chunk_size: int = Field(default=512, description="Document chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    max_document_size_mb: int = Field(default=50, description="Maximum document size in MB")

    # Search
    search_default_limit: int = Field(default=10, description="Default search result limit")
    search_max_limit: int = Field(default=100, description="Maximum search result limit")
    semantic_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity for semantic search",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_CONTEXT_")
