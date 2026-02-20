# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RAG retriever for multi-source context retrieval.

This module implements the RAGRetriever which retrieves relevant context
from multiple sources:
- Episodic memories (learning events)
- Associative memories (interests, analogies)
- Curriculum knowledge (educational content)

The retriever uses semantic similarity search via embeddings and provides
unified results that can be reranked for optimal relevance.

Example:
    retriever = RAGRetriever(
        memory_manager=memory_manager,
        qdrant_client=qdrant,
        embedding_service=embedding_service,
    )

    results = await retriever.retrieve(
        tenant_code="acme",
        student_id=student_uuid,
        query="How do fractions work?",
        sources=["episodic", "associative", "curriculum"],
        limit=10,
    )
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.core.intelligence.embeddings import EmbeddingService
from src.core.memory.manager import MemoryManager
from src.infrastructure.vectors import QdrantVectorClient, SearchResult

logger = logging.getLogger(__name__)

# Curriculum collection name (shared across tenants)
CURRICULUM_COLLECTION = "curriculum_knowledge"


class RetrievalSource(str, Enum):
    """Sources for retrieval."""

    EPISODIC = "episodic"
    ASSOCIATIVE = "associative"
    CURRICULUM = "curriculum"


@dataclass
class RetrievalResult:
    """Result from RAG retrieval.

    Attributes:
        source: Which source the result came from.
        content: The retrieved content text.
        score: Similarity score (0-1).
        metadata: Additional metadata about the result.
        memory_id: ID if from memory layer.
        chunk_id: ID if from curriculum chunk.
    """

    source: RetrievalSource
    content: str
    score: float
    metadata: dict[str, Any]
    memory_id: str | None = None
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        """Ensure source is RetrievalSource enum."""
        if isinstance(self.source, str):
            self.source = RetrievalSource(self.source)


class RAGRetrieverError(Exception):
    """Exception raised for RAG retrieval operations.

    Attributes:
        message: Error description.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class RAGRetriever:
    """Multi-source retriever for RAG context.

    Retrieves relevant context from memory layers and curriculum content
    using semantic similarity search.

    Attributes:
        memory_manager: Manager for memory layer access.
        qdrant_client: Client for vector similarity search.
        embedding_service: Service for generating query embeddings.

    Example:
        retriever = RAGRetriever(
            memory_manager=memory,
            qdrant_client=qdrant,
            embedding_service=embedding,
        )

        results = await retriever.retrieve(
            tenant_code="acme",
            student_id=student_uuid,
            query="explain multiplication",
            limit=10,
        )
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        qdrant_client: QdrantVectorClient,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize the RAG retriever.

        Args:
            memory_manager: Manager for memory layer access.
            qdrant_client: Client for Qdrant vector database.
            embedding_service: Service for generating text embeddings.
        """
        self._memory = memory_manager
        self._qdrant = qdrant_client
        self._embedding = embedding_service

    async def retrieve(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        query: str,
        sources: list[RetrievalSource | str] | None = None,
        limit: int = 10,
        min_score: float = 0.5,
        include_curriculum: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve relevant context from multiple sources.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            query: Search query text.
            sources: Sources to search (defaults to all).
            limit: Maximum total results.
            min_score: Minimum similarity score.
            include_curriculum: Whether to search curriculum content.

        Returns:
            List of RetrievalResult ordered by relevance.

        Raises:
            RAGRetrieverError: If retrieval fails.
        """
        # Normalize sources
        if sources is None:
            sources = [RetrievalSource.EPISODIC, RetrievalSource.ASSOCIATIVE]
            if include_curriculum:
                sources.append(RetrievalSource.CURRICULUM)
        else:
            sources = [
                RetrievalSource(s) if isinstance(s, str) else s for s in sources
            ]

        # Calculate per-source limits
        per_source_limit = max(3, limit // len(sources))

        results: list[RetrievalResult] = []

        # Retrieve from each source
        for source in sources:
            try:
                if source == RetrievalSource.EPISODIC:
                    source_results = await self._retrieve_episodic(
                        tenant_code=tenant_code,
                        student_id=student_id,
                        query=query,
                        limit=per_source_limit,
                        min_score=min_score,
                    )
                elif source == RetrievalSource.ASSOCIATIVE:
                    source_results = await self._retrieve_associative(
                        tenant_code=tenant_code,
                        student_id=student_id,
                        query=query,
                        limit=per_source_limit,
                        min_score=min_score,
                    )
                elif source == RetrievalSource.CURRICULUM:
                    source_results = await self._retrieve_curriculum(
                        tenant_code=tenant_code,
                        query=query,
                        limit=per_source_limit,
                        min_score=min_score,
                    )
                else:
                    continue

                results.extend(source_results)

            except Exception as e:
                logger.warning(
                    "Failed to retrieve from %s: %s", source.value, str(e)
                )
                # Continue with other sources

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:limit]

        logger.debug(
            "Retrieved %d results for query '%s' from %d sources",
            len(results),
            query[:50],
            len(sources),
        )

        return results

    async def _retrieve_episodic(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        """Retrieve from episodic memory layer.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            query: Search query text.
            limit: Maximum results.
            min_score: Minimum similarity score.

        Returns:
            List of RetrievalResult from episodic memories.
        """
        results = await self._memory.episodic.search(
            tenant_code=tenant_code,
            student_id=student_id,
            query=query,
            limit=limit,
            min_score=min_score,
        )

        return [
            RetrievalResult(
                source=RetrievalSource.EPISODIC,
                content=memory.summary,
                score=score,
                metadata={
                    "event_type": memory.event_type.value,
                    "emotional_state": (
                        memory.emotional_state.value if memory.emotional_state else None
                    ),
                    "importance": memory.importance,
                    "occurred_at": memory.occurred_at.isoformat(),
                },
                memory_id=str(memory.id),
            )
            for memory, score in results
        ]

    async def _retrieve_associative(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        """Retrieve from associative memory layer.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            query: Search query text.
            limit: Maximum results.
            min_score: Minimum similarity score.

        Returns:
            List of RetrievalResult from associative memories.
        """
        results = await self._memory.associative.search(
            tenant_code=tenant_code,
            student_id=student_id,
            query=query,
            limit=limit,
            min_score=min_score,
        )

        return [
            RetrievalResult(
                source=RetrievalSource.ASSOCIATIVE,
                content=memory.content,
                score=score,
                metadata={
                    "association_type": memory.association_type.value,
                    "strength": memory.strength,
                    "tags": memory.tags,
                    "times_effective": memory.times_effective,
                },
                memory_id=str(memory.id),
            )
            for memory, score in results
        ]

    async def _retrieve_curriculum(
        self,
        tenant_code: str,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        """Retrieve from curriculum knowledge base.

        Searches the shared curriculum content collection.

        Args:
            tenant_code: Unique tenant identifier (for future tenant-specific content).
            query: Search query text.
            limit: Maximum results.
            min_score: Minimum similarity score.

        Returns:
            List of RetrievalResult from curriculum content.
        """
        # Generate query embedding
        try:
            query_embedding = await self._embedding.embed_text(query)
        except Exception as e:
            logger.error("Failed to generate query embedding: %s", str(e))
            return []

        # Check if collection exists
        exists = await self._qdrant.collection_exists(CURRICULUM_COLLECTION)
        if not exists:
            logger.debug("Curriculum collection does not exist yet")
            return []

        # Search curriculum collection (shared, not tenant-isolated)
        try:
            search_results: list[SearchResult] = await self._qdrant.search(
                collection_name=CURRICULUM_COLLECTION,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
            )
        except Exception as e:
            logger.error("Failed to search curriculum: %s", str(e))
            return []

        return [
            RetrievalResult(
                source=RetrievalSource.CURRICULUM,
                content=result.payload.get("content", ""),
                score=result.score,
                metadata={
                    "subject": result.payload.get("subject"),
                    "topic": result.payload.get("topic"),
                    "chunk_type": result.payload.get("chunk_type", "text"),
                    "source_id": result.payload.get("source_id"),
                },
                chunk_id=result.id,
            )
            for result in search_results
            if result.payload
        ]

    async def retrieve_for_conversation(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        conversation_context: str,
        current_topic: str | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve context optimized for conversation continuation.

        Uses both the recent conversation context and current topic
        to find relevant information.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            conversation_context: Recent conversation summary.
            current_topic: Current topic being discussed.
            limit: Maximum results.

        Returns:
            List of relevant RetrievalResult.
        """
        # Combine context for query
        if current_topic:
            query = f"{current_topic}: {conversation_context}"
        else:
            query = conversation_context

        return await self.retrieve(
            tenant_code=tenant_code,
            student_id=student_id,
            query=query,
            limit=limit,
            min_score=0.4,  # Lower threshold for conversation context
            include_curriculum=True,
        )

    async def retrieve_for_question(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        question_text: str,
        topic_full_code: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve context optimized for question answering.

        Focuses on curriculum content and relevant student memories.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            question_text: The question being asked.
            topic_full_code: Optional topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            List of relevant RetrievalResult.
        """
        # Get more curriculum content for question answering
        curriculum_results = await self._retrieve_curriculum(
            tenant_code=tenant_code,
            query=question_text,
            limit=5,
            min_score=0.5,
        )

        # Get relevant student experiences
        episodic_results = await self._retrieve_episodic(
            tenant_code=tenant_code,
            student_id=student_id,
            query=question_text,
            limit=3,
            min_score=0.5,
        )

        # Get relevant interests for personalization
        associative_results = await self._retrieve_associative(
            tenant_code=tenant_code,
            student_id=student_id,
            query=question_text,
            limit=2,
            min_score=0.4,
        )

        # Combine and sort by score
        all_results = curriculum_results + episodic_results + associative_results
        all_results.sort(key=lambda r: r.score, reverse=True)

        return all_results[:10]

    def format_context_for_prompt(
        self,
        results: list[RetrievalResult],
        max_length: int = 2000,
    ) -> str:
        """Format retrieval results for inclusion in LLM prompt.

        Args:
            results: List of retrieval results.
            max_length: Maximum character length for context.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        sections = {
            RetrievalSource.CURRICULUM: [],
            RetrievalSource.EPISODIC: [],
            RetrievalSource.ASSOCIATIVE: [],
        }

        for result in results:
            sections[result.source].append(result)

        parts = []

        # Curriculum knowledge first
        if sections[RetrievalSource.CURRICULUM]:
            curriculum_text = "\n".join(
                f"- {r.content}" for r in sections[RetrievalSource.CURRICULUM]
            )
            parts.append(f"Relevant Knowledge:\n{curriculum_text}")

        # Student learning history
        if sections[RetrievalSource.EPISODIC]:
            episodic_text = "\n".join(
                f"- [{r.metadata.get('event_type', 'event')}] {r.content}"
                for r in sections[RetrievalSource.EPISODIC]
            )
            parts.append(f"Student's Learning History:\n{episodic_text}")

        # Student interests
        if sections[RetrievalSource.ASSOCIATIVE]:
            assoc_text = "\n".join(
                f"- [{r.metadata.get('association_type', 'interest')}] {r.content}"
                for r in sections[RetrievalSource.ASSOCIATIVE]
            )
            parts.append(f"Student's Interests:\n{assoc_text}")

        context = "\n\n".join(parts)

        # Truncate if too long
        if len(context) > max_length:
            context = context[: max_length - 3] + "..."

        return context
