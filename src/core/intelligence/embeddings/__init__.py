# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Embedding service module using LiteLLM.

This module provides API-based embedding generation through LiteLLM,
supporting multiple embedding providers without local model loading.

Supported models:
- ollama/nomic-embed-text (768 dimensions, default)
- ollama/mxbai-embed-large (1024 dimensions)
- text-embedding-3-small (1536 dimensions, OpenAI)
- text-embedding-3-large (3072 dimensions, OpenAI)

Example:
    >>> from src.core.intelligence.embeddings import EmbeddingService
    >>> service = EmbeddingService(model="ollama/nomic-embed-text")
    >>> vector = await service.embed_text("Hello world")
    >>> print(f"Vector dimension: {len(vector)}")
    Vector dimension: 768
"""

from src.core.intelligence.embeddings.service import EmbeddingService

__all__ = ["EmbeddingService"]
