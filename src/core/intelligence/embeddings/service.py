# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Embedding service for API-based embedding generation.

This module provides a unified interface for generating text embeddings
supporting multiple embedding providers.

Supported providers:
- Ollama: nomic-embed-text (768d), mxbai-embed-large (1024d) - via direct httpx
- OpenAI: text-embedding-3-small (1536d), text-embedding-3-large (3072d) - via LiteLLM
- Cohere: embed-english-v3.0, embed-multilingual-v3.0 - via LiteLLM

Note: Ollama embeddings use direct httpx calls because LiteLLM doesn't
properly pass the Authorization header for authenticated Ollama endpoints.

Example:
    >>> from src.core.intelligence.embeddings import EmbeddingService
    >>> service = EmbeddingService()
    >>> vector = await service.embed_text("Hello world")
    >>> vectors = await service.embed_batch(["Hello", "World"])
"""

import logging
from typing import Any, Optional

import httpx
import litellm
from litellm import aembedding

from src.core.config.llm_providers import (
    EmbeddingConfig,
    ProviderConfig,
    get_provider_manager,
)

logger = logging.getLogger(__name__)

# Model dimension mapping for known embedding models
MODEL_DIMENSIONS: dict[str, int] = {
    # Ollama models
    "ollama/nomic-embed-text": 768,
    "ollama/mxbai-embed-large": 1024,
    "ollama/all-minilm": 384,
    "ollama/snowflake-arctic-embed": 1024,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Cohere models
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    # Google Gemini models (via LiteLLM gemini/ prefix)
    "gemini/text-embedding-004": 768,
    "gemini/gemini-embedding-001": 3072,
    "text-embedding-004": 768,  # Without prefix for config compatibility
}


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails.

    Attributes:
        message: Error description.
        model: Model that caused the error.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize EmbeddingError.

        Args:
            message: Error description.
            model: Model that caused the error.
            original_error: Original exception if any.
        """
        self.message = message
        self.model = model
        self.original_error = original_error
        super().__init__(self.message)


class EmbeddingService:
    """Service for generating text embeddings via LiteLLM.

    This service provides a unified interface for embedding generation
    across multiple providers (Ollama, OpenAI, Cohere, etc.) using
    LiteLLM's abstraction layer.

    All embeddings are API-based, meaning no local model loading is required.
    This makes the service lightweight and suitable for containerized
    deployments.

    Attributes:
        model: The embedding model identifier in LiteLLM format.
        dimension: The output dimension of the embedding vectors.
        batch_size: Maximum number of texts to embed in a single batch.

    Example:
        >>> service = EmbeddingService()
        >>> vector = await service.embed_text("Hello world")
        >>> print(f"Dimension: {service.dimension}")
        Dimension: 768
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        provider_config: Optional[ProviderConfig] = None,
    ):
        """Initialize the embedding service.

        Args:
            model: Embedding model in LiteLLM format (e.g., 'ollama/nomic-embed-text').
                   Falls back to config if not provided.
            dimension: Vector dimension. Auto-detected from model if not provided.
            batch_size: Maximum batch size for embed_batch. Falls back to config.
            embedding_config: Embedding configuration. Uses provider manager if None.
            provider_config: Provider configuration for API. Uses provider manager if None.
        """
        # Get configuration from llm_providers.py (reads config/llm/providers.yaml)
        provider_manager = get_provider_manager()
        self._embedding_config = embedding_config or provider_manager.embedding

        # Get provider config for the embedding provider
        if provider_config:
            self._provider_config = provider_config
        else:
            provider = provider_manager.get_provider(self._embedding_config.provider)
            if not provider:
                raise ValueError(
                    f"Unknown embedding provider: {self._embedding_config.provider}"
                )
            self._provider_config = provider

        # Build model string with provider prefix if needed
        # LiteLLM requires provider prefix for routing (e.g., ollama/, gemini/)
        raw_model = model or self._embedding_config.model
        if self._provider_config.type == "ollama" and not raw_model.startswith("ollama/"):
            self._model = f"ollama/{raw_model}"
        elif self._provider_config.type == "google" and not raw_model.startswith("gemini/"):
            self._model = f"gemini/{raw_model}"
        else:
            self._model = raw_model

        self._batch_size = batch_size or self._embedding_config.batch_size

        # Determine dimension from model mapping or config
        if dimension is not None:
            self._dimension = dimension
        elif self._model in MODEL_DIMENSIONS:
            self._dimension = MODEL_DIMENSIONS[self._model]
        else:
            self._dimension = self._embedding_config.dimension

        # Build LiteLLM params (api_base, api_key) from provider config
        self._litellm_params = self._build_litellm_params()

        # Configure LiteLLM settings
        self._configure_litellm()

        logger.info(
            "EmbeddingService initialized with model=%s, dimension=%d, batch_size=%d, provider=%s",
            self._model,
            self._dimension,
            self._batch_size,
            self._provider_config.code,
        )

    def _build_litellm_params(self) -> dict[str, Any]:
        """Build parameters for LiteLLM aembedding() calls.

        Uses provider config from config/llm/providers.yaml to get
        api_base and api_key. These are passed directly to aembedding()
        instead of relying on environment variables.

        Returns:
            Dictionary with api_base and api_key if configured.
        """
        params: dict[str, Any] = {}

        if self._provider_config.api_base:
            params["api_base"] = self._provider_config.api_base

        if self._provider_config.api_key:
            params["api_key"] = self._provider_config.api_key

        return params

    def _configure_litellm(self) -> None:
        """Configure LiteLLM settings.

        Suppresses verbose logging in production.
        Note: For non-Ollama providers, api_base and api_key are passed
        directly to aembedding() via _litellm_params.
        """
        # Suppress LiteLLM verbose logging in production
        litellm.set_verbose = False

    def _is_ollama_provider(self) -> bool:
        """Check if the current provider is Ollama.

        Returns:
            True if provider type is 'ollama'.
        """
        return self._provider_config.type == "ollama"

    async def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using direct Ollama API call.

        LiteLLM doesn't properly pass Authorization header for Ollama
        embeddings, so we use direct httpx calls for authenticated
        Ollama endpoints (e.g., VastAI).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        # Extract model name without 'ollama/' prefix
        model_name = self._model
        if model_name.startswith("ollama/"):
            model_name = model_name[7:]

        api_base = self._provider_config.api_base
        api_key = self._provider_config.api_key

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{api_base}/api/embed",
                    headers=headers,
                    json={"model": model_name, "input": texts},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embeddings", [])

        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                message=f"Ollama API error: {e.response.status_code} - {e.response.text}",
                model=self._model,
                original_error=e,
            ) from e
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to call Ollama embedding API: {str(e)}",
                model=self._model,
                original_error=e,
            ) from e

    @property
    def model(self) -> str:
        """Get the embedding model identifier.

        Returns:
            Model name in LiteLLM format.
        """
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension.

        Returns:
            Vector dimension as integer.
        """
        return self._dimension

    @property
    def batch_size(self) -> int:
        """Get the maximum batch size for embedding operations.

        Returns:
            Maximum number of texts per batch.
        """
        return self._batch_size

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.

        Example:
            >>> vector = await service.embed_text("Machine learning is fascinating")
            >>> len(vector)
            768
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Use direct Ollama API for authenticated Ollama endpoints
            if self._is_ollama_provider():
                embeddings = await self._ollama_embed([text])
                embedding = embeddings[0] if embeddings else []
            else:
                # Use LiteLLM for other providers (OpenAI, Cohere, etc.)
                response = await aembedding(
                    model=self._model,
                    input=[text],
                    **self._litellm_params,
                )
                embedding = response.data[0]["embedding"]

            logger.debug(
                "Generated embedding for text of length %d, dimension=%d",
                len(text),
                len(embedding),
            )

            return embedding

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Failed to generate embedding: model=%s, text_length=%d, error=%s",
                self._model,
                len(text),
                str(e),
            )
            raise EmbeddingError(
                message=f"Failed to generate embedding: {str(e)}",
                model=self._model,
                original_error=e,
            ) from e

    async def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Handles batching automatically based on batch_size setting.
        Texts are processed in chunks to avoid API limits and memory issues.

        Args:
            texts: List of input texts to embed.
            show_progress: Whether to log progress (useful for large batches).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.

        Example:
            >>> texts = ["Hello", "World", "AI"]
            >>> vectors = await service.embed_batch(texts)
            >>> len(vectors)
            3
            >>> all(len(v) == 768 for v in vectors)
            True
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Filter out empty texts and track their indices
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            raise ValueError("All provided texts are empty")

        all_embeddings: list[list[float]] = []
        total_batches = (len(valid_texts) + self._batch_size - 1) // self._batch_size

        try:
            for batch_idx in range(0, len(valid_texts), self._batch_size):
                batch = valid_texts[batch_idx : batch_idx + self._batch_size]
                current_batch_num = batch_idx // self._batch_size + 1

                if show_progress:
                    logger.info(
                        "Processing batch %d/%d (%d texts)",
                        current_batch_num,
                        total_batches,
                        len(batch),
                    )

                # Use direct Ollama API for authenticated Ollama endpoints
                if self._is_ollama_provider():
                    batch_embeddings = await self._ollama_embed(batch)
                else:
                    # Use LiteLLM for other providers (OpenAI, Cohere, etc.)
                    response = await aembedding(
                        model=self._model,
                        input=batch,
                        **self._litellm_params,
                    )
                    batch_embeddings = [item["embedding"] for item in response.data]

                all_embeddings.extend(batch_embeddings)

            # Reconstruct result with None for empty texts
            result: list[list[float]] = [[] for _ in range(len(texts))]
            for idx, embedding in zip(valid_indices, all_embeddings):
                result[idx] = embedding

            # Fill empty slots with zero vectors
            zero_vector = [0.0] * self._dimension
            for i in range(len(texts)):
                if not result[i]:
                    result[i] = zero_vector

            logger.info(
                "Generated %d embeddings in %d batches",
                len(all_embeddings),
                total_batches,
            )

            return result

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Failed to generate batch embeddings: model=%s, count=%d, error=%s",
                self._model,
                len(texts),
                str(e),
            )
            raise EmbeddingError(
                message=f"Failed to generate batch embeddings: {str(e)}",
                model=self._model,
                original_error=e,
            ) from e

    async def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Calculate cosine similarity between two texts.

        Generates embeddings for both texts and computes their
        cosine similarity score.

        Args:
            text1: First text to compare.
            text2: Second text to compare.

        Returns:
            Cosine similarity score between -1 and 1.

        Example:
            >>> score = await service.similarity("Hello world", "Hi there")
            >>> print(f"Similarity: {score:.3f}")
            Similarity: 0.745
        """
        embeddings = await self.embed_batch([text1, text2])
        return self._cosine_similarity(embeddings[0], embeddings[1])

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def __repr__(self) -> str:
        """Return string representation of the service.

        Returns:
            String showing model, dimension, and batch size.
        """
        return (
            f"EmbeddingService(model={self._model!r}, "
            f"dimension={self._dimension}, batch_size={self._batch_size})"
        )
