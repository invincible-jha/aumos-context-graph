"""Embedding engine adapter for aumos-context-graph.

Provides OpenAI-compatible embedding generation.
Configured via settings to support model switching.
"""

from __future__ import annotations

import httpx
from aumos_common.observability import get_logger

from aumos_context_graph.settings import Settings

logger = get_logger(__name__)


class OpenAIEmbeddingEngine:
    """Generates text embeddings using OpenAI-compatible API.

    Supports batched embedding generation with retry logic.

    Args:
        settings: Service settings for model and API configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the embedding engine.

        Args:
            settings: Service settings containing API keys and model config.
        """
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_batch_size
        self._api_key = settings.openai_api_key if hasattr(settings, "openai_api_key") else ""

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Splits into batches of self._batch_size to respect API limits.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_embeddings = await self._embed_batch_api(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for a single batch.

        Falls back to zero vectors when API key is not configured,
        allowing the service to function in development without credentials.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self._api_key:
            logger.warning(
                "No OpenAI API key configured, returning zero embeddings",
                batch_size=len(texts),
                model=self._model,
            )
            return [[0.0] * self._dimensions for _ in texts]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": texts,
                    "dimensions": self._dimensions,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


__all__ = ["OpenAIEmbeddingEngine"]
