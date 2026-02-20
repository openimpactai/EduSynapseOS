# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for RAG components.

Tests the RAG retriever and reranker:
- RAGRetriever: Multi-source context retrieval
- ResultReranker: LLM-based relevance reranking
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.memory.rag.retriever import (
    RAGRetriever,
    RetrievalResult,
    RetrievalSource,
)
from src.core.memory.rag.reranker import RerankResult, ResultReranker


# ============================================================================
# RAGRetriever tests
# ============================================================================


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    manager = MagicMock()
    manager.episodic = MagicMock()
    manager.associative = MagicMock()
    return manager


@pytest.fixture
def mock_qdrant():
    """Create a mock QdrantVectorClient."""
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.search = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_embedding():
    """Create a mock EmbeddingService."""
    service = MagicMock()
    service.embed_text = AsyncMock(return_value=[0.1] * 768)
    return service


@pytest.fixture
def retriever(mock_memory_manager, mock_qdrant, mock_embedding):
    """Create a RAGRetriever with mocked dependencies."""
    return RAGRetriever(
        memory_manager=mock_memory_manager,
        qdrant_client=mock_qdrant,
        embedding_service=mock_embedding,
    )


@pytest.mark.unit
class TestRAGRetrieverInit:
    """Test cases for RAGRetriever initialization."""

    def test_initialization(self, mock_memory_manager, mock_qdrant, mock_embedding):
        """Test retriever initializes with all dependencies."""
        retriever = RAGRetriever(
            memory_manager=mock_memory_manager,
            qdrant_client=mock_qdrant,
            embedding_service=mock_embedding,
        )

        assert retriever._memory == mock_memory_manager
        assert retriever._qdrant == mock_qdrant
        assert retriever._embedding == mock_embedding


@pytest.mark.unit
class TestRAGRetrieverRetrieve:
    """Test cases for RAGRetriever.retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_searches_all_sources(self, retriever, mock_memory_manager):
        """Test that retrieve searches all specified sources."""
        student_id = uuid.uuid4()

        # Mock search methods
        mock_memory_manager.episodic.search = AsyncMock(return_value=[])
        mock_memory_manager.associative.search = AsyncMock(return_value=[])

        results = await retriever.retrieve(
            tenant_code="test",
            student_id=student_id,
            query="fractions",
            sources=[RetrievalSource.EPISODIC, RetrievalSource.ASSOCIATIVE],
            include_curriculum=False,
        )

        mock_memory_manager.episodic.search.assert_called_once()
        mock_memory_manager.associative.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_respects_limit(self, retriever, mock_memory_manager):
        """Test that retrieve respects the limit parameter."""
        student_id = uuid.uuid4()

        # Create mock results
        from src.models.memory import (
            AssociativeMemoryResponse,
            AssociationType,
            EmotionalState,
            EpisodicEventType,
            EpisodicMemoryResponse,
        )
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        mock_episodic = [
            (
                EpisodicMemoryResponse(
                    id=uuid.uuid4(),
                    student_id=student_id,
                    event_type=EpisodicEventType.BREAKTHROUGH,
                    summary=f"Event {i}",
                    details={},
                    emotional_state=None,
                    importance=0.5,
                    session_id=None,
                    conversation_id=None,
                    topic_id=None,
                    topic_name=None,
                    occurred_at=now,
                    access_count=0,
                    last_accessed_at=None,
                ),
                0.8,
            )
            for i in range(5)
        ]

        mock_memory_manager.episodic.search = AsyncMock(return_value=mock_episodic)
        mock_memory_manager.associative.search = AsyncMock(return_value=[])

        results = await retriever.retrieve(
            tenant_code="test",
            student_id=student_id,
            query="test",
            sources=[RetrievalSource.EPISODIC],
            limit=3,
            include_curriculum=False,
        )

        assert len(results) <= 3


@pytest.mark.unit
class TestRAGRetrieverFormatContext:
    """Test cases for RAGRetriever.format_context_for_prompt method."""

    def test_format_context_organizes_by_source(self, retriever):
        """Test that format_context organizes results by source."""
        results = [
            RetrievalResult(
                source=RetrievalSource.CURRICULUM,
                content="Curriculum content about fractions",
                score=0.9,
                metadata={"subject": "math"},
            ),
            RetrievalResult(
                source=RetrievalSource.EPISODIC,
                content="Student struggled with fractions",
                score=0.8,
                metadata={"event_type": "struggle"},
            ),
            RetrievalResult(
                source=RetrievalSource.ASSOCIATIVE,
                content="Student likes pizza",
                score=0.7,
                metadata={"association_type": "interest"},
            ),
        ]

        formatted = retriever.format_context_for_prompt(results)

        assert "Relevant Knowledge" in formatted
        assert "Student's Learning History" in formatted
        assert "Student's Interests" in formatted
        assert "Curriculum content about fractions" in formatted

    def test_format_context_respects_max_length(self, retriever):
        """Test that format_context respects max_length."""
        # Create a result with very long content
        results = [
            RetrievalResult(
                source=RetrievalSource.CURRICULUM,
                content="x" * 5000,
                score=0.9,
                metadata={},
            ),
        ]

        formatted = retriever.format_context_for_prompt(results, max_length=500)

        assert len(formatted) <= 500
        assert formatted.endswith("...")


# ============================================================================
# ResultReranker tests
# ============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLMClient."""
    client = MagicMock()
    return client


@pytest.fixture
def reranker(mock_llm):
    """Create a ResultReranker with mocked LLM."""
    return ResultReranker(llm_client=mock_llm)


@pytest.fixture
def reranker_no_llm():
    """Create a ResultReranker without LLM."""
    return ResultReranker(llm_client=None, use_llm=False)


@pytest.mark.unit
class TestResultRerankerInit:
    """Test cases for ResultReranker initialization."""

    def test_initialization_with_llm(self, mock_llm):
        """Test reranker initializes with LLM."""
        reranker = ResultReranker(llm_client=mock_llm)
        assert reranker._llm == mock_llm
        assert reranker._use_llm is True

    def test_initialization_without_llm(self):
        """Test reranker initializes without LLM."""
        reranker = ResultReranker(llm_client=None, use_llm=False)
        assert reranker._llm is None
        assert reranker._use_llm is False


@pytest.mark.unit
class TestResultRerankerHeuristic:
    """Test cases for heuristic reranking."""

    def test_rerank_heuristic_applies_source_boost(self, reranker_no_llm):
        """Test that heuristic reranking applies source-based boosts."""
        results = [
            RetrievalResult(
                source=RetrievalSource.CURRICULUM,
                content="Curriculum content",
                score=0.8,
                metadata={},
            ),
            RetrievalResult(
                source=RetrievalSource.EPISODIC,
                content="Episodic content",
                score=0.8,
                metadata={"importance": 0.9},
            ),
        ]

        reranked = reranker_no_llm._rerank_heuristic(
            query="test query",
            results=results,
            top_k=2,
        )

        # Both should be reranked with adjusted scores
        assert len(reranked) == 2
        assert all(r.rerank_score > 0 for r in reranked)

    def test_rerank_heuristic_respects_top_k(self, reranker_no_llm):
        """Test that heuristic reranking respects top_k."""
        results = [
            RetrievalResult(
                source=RetrievalSource.CURRICULUM,
                content=f"Content {i}",
                score=0.5 + i * 0.1,
                metadata={},
            )
            for i in range(10)
        ]

        reranked = reranker_no_llm._rerank_heuristic(
            query="test",
            results=results,
            top_k=3,
        )

        assert len(reranked) == 3


@pytest.mark.unit
class TestResultRerankerDiversify:
    """Test cases for result diversification."""

    def test_diversify_limits_per_source(self, reranker_no_llm):
        """Test that diversify limits results per source."""
        results = [
            RerankResult(
                original=RetrievalResult(
                    source=RetrievalSource.CURRICULUM,
                    content=f"Curriculum {i}",
                    score=0.9 - i * 0.05,
                    metadata={},
                ),
                rerank_score=0.9 - i * 0.05,
                original_rank=i,
                new_rank=i,
            )
            for i in range(5)
        ]

        diversified = reranker_no_llm.diversify(results, max_per_source=2)

        curriculum_count = sum(
            1
            for r in diversified
            if r.original.source == RetrievalSource.CURRICULUM
        )
        assert curriculum_count <= 2


@pytest.mark.unit
class TestResultRerankerExplain:
    """Test cases for ranking explanation."""

    def test_explain_heuristic_generates_explanation(self, reranker_no_llm):
        """Test that heuristic explanation generates text."""
        result = RetrievalResult(
            source=RetrievalSource.CURRICULUM,
            content="Content about fractions",
            score=0.85,
            metadata={},
        )

        explanation = reranker_no_llm._explain_heuristic(
            query="fractions",
            result=result,
        )

        assert len(explanation) > 0
        assert "educational content" in explanation.lower()

    def test_explain_heuristic_handles_episodic(self, reranker_no_llm):
        """Test that heuristic explains episodic results."""
        result = RetrievalResult(
            source=RetrievalSource.EPISODIC,
            content="Struggled with fractions",
            score=0.75,
            metadata={"event_type": "struggle"},
        )

        explanation = reranker_no_llm._explain_heuristic(
            query="fractions",
            result=result,
        )

        assert "learning" in explanation.lower() or "struggle" in explanation.lower()
