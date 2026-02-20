# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Vector storage infrastructure using Qdrant.

This package provides Qdrant client for vector similarity search.
Used for RAG (Retrieval-Augmented Generation) and memory layer embeddings.

Tenant isolation is achieved via collection naming: tenant_{code}_{collection}

Collections:
- curriculum_knowledge: Curriculum content for RAG (shared)
- episodic_memories: Student learning events (tenant-isolated)
- student_interests: Interest-based personalization (tenant-isolated)
- successful_explanations: Effective teaching approaches (tenant-shared)

Example:
    from src.infrastructure.vectors import QdrantVectorClient, init_qdrant, get_qdrant

    # Initialize at application startup
    await init_qdrant(settings)

    # Get the Qdrant client
    qdrant = get_qdrant()

    # Search with tenant isolation
    results = await qdrant.search_with_tenant(
        "acme",
        "episodic_memories",
        query_vector=[0.1, 0.2, ...],
        limit=5
    )

    # Cleanup at shutdown
    await close_qdrant()
"""

from src.infrastructure.vectors.qdrant_client import (
    QdrantError,
    QdrantVectorClient,
    SearchResult,
    close_qdrant,
    get_qdrant,
    init_qdrant,
)

__all__ = [
    "QdrantError",
    "QdrantVectorClient",
    "SearchResult",
    "close_qdrant",
    "get_qdrant",
    "init_qdrant",
]
