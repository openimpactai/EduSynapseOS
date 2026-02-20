# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RAG (Retrieval-Augmented Generation) module for memory system.

This package provides retrieval and reranking capabilities for the
memory system, enabling context-aware responses by retrieving relevant
information from memory layers and curriculum content.

Components:
- RAGRetriever: Multi-source retrieval from memory and curriculum
- ResultReranker: LLM-based relevance reranking

Example:
    from src.core.memory.rag import RAGRetriever, ResultReranker

    retriever = RAGRetriever(
        memory_manager=memory_manager,
        qdrant_client=qdrant,
        embedding_service=embedding_service,
    )

    reranker = ResultReranker(llm_client=llm_client)

    # Retrieve relevant context
    results = await retriever.retrieve(
        tenant_code="acme",
        student_id=student_uuid,
        query="How do fractions work?",
    )

    # Rerank by relevance
    reranked = await reranker.rerank(
        query="How do fractions work?",
        results=results,
        top_k=5,
    )
"""

from src.core.memory.rag.reranker import RerankResult, ResultReranker
from src.core.memory.rag.retriever import RAGRetriever, RetrievalResult

__all__ = [
    "RAGRetriever",
    "RerankResult",
    "ResultReranker",
    "RetrievalResult",
]
