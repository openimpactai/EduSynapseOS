# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for MemoryManager.

Tests the MemoryManager which orchestrates all four memory layers
and provides unified access to memory context.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.memory.manager import MemoryManager
from src.models.memory import (
    EpisodicEventType,
    EpisodicMemoryResponse,
    LearningPatterns,
    MasteryOverview,
    StudentInterests,
)


@pytest.fixture
def mock_tenant_db():
    """Create a mock TenantDatabaseManager."""
    return MagicMock()


@pytest.fixture
def mock_embedding_service():
    """Create a mock EmbeddingService."""
    service = MagicMock()
    service.dimension = 768
    service.embed_text = AsyncMock(return_value=[0.1] * 768)
    return service


@pytest.fixture
def mock_qdrant_client():
    """Create a mock QdrantVectorClient."""
    client = MagicMock()
    client._tenant_collection_name = MagicMock(return_value="collection")
    client.collection_exists = AsyncMock(return_value=True)
    client.create_tenant_collection = AsyncMock()
    return client


@pytest.fixture
def memory_manager(mock_tenant_db, mock_embedding_service, mock_qdrant_client):
    """Create a MemoryManager with mocked dependencies."""
    return MemoryManager(
        tenant_db_manager=mock_tenant_db,
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
    )


@pytest.mark.unit
class TestMemoryManagerInit:
    """Test cases for MemoryManager initialization."""

    def test_initialization_creates_all_layers(
        self, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test that manager creates all four memory layers."""
        manager = MemoryManager(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

        assert manager.episodic is not None
        assert manager.semantic is not None
        assert manager.procedural is not None
        assert manager.associative is not None


@pytest.mark.unit
class TestMemoryManagerEnsureCollections:
    """Test cases for MemoryManager.ensure_collections method."""

    @pytest.mark.asyncio
    async def test_ensure_collections_creates_for_episodic_and_associative(
        self, memory_manager
    ):
        """Test that ensure_collections creates collections for vector layers."""
        await memory_manager.ensure_collections("test_tenant")

        # Both episodic and associative layers should check/create collections
        assert memory_manager._qdrant.collection_exists.call_count >= 2


@pytest.mark.unit
class TestMemoryManagerGetFullContext:
    """Test cases for MemoryManager.get_full_context method."""

    @pytest.mark.asyncio
    async def test_get_full_context_returns_all_layers(self, memory_manager):
        """Test that get_full_context retrieves from all layers."""
        student_id = uuid.uuid4()

        # Mock layer methods
        memory_manager.episodic.get_recent = AsyncMock(return_value=[])
        memory_manager.episodic.get_important_memories = AsyncMock(return_value=[])
        memory_manager.semantic.get_mastery_overview = AsyncMock(
            return_value=MasteryOverview(
                student_id=student_id,
                overall_mastery=0.5,
                topics_mastered=2,
                topics_learning=5,
                topics_struggling=1,
                total_topics=8,
                by_subject={},
            )
        )
        memory_manager.procedural.get_learning_patterns = AsyncMock(
            return_value=LearningPatterns(
                student_id=student_id,
                best_time_of_day="morning",
                optimal_session_duration=30,
                preferred_content_format="visual",
                hint_dependency=None,
                avg_break_frequency=None,
                preferred_difficulty=None,
                favorite_persona=None,
                vark_profile=None,
            )
        )
        memory_manager.associative.get_student_interests = AsyncMock(
            return_value=StudentInterests(
                student_id=student_id,
                interests=[],
                effective_analogies=[],
            )
        )

        result = await memory_manager.get_full_context(
            tenant_code="test",
            student_id=student_id,
        )

        assert result.student_id == student_id
        assert result.semantic.overall_mastery == 0.5
        assert result.procedural.best_time_of_day == "morning"
        assert result.retrieved_at is not None


@pytest.mark.unit
class TestMemoryManagerSearchAll:
    """Test cases for MemoryManager.search_all method."""

    @pytest.mark.asyncio
    async def test_search_all_searches_specified_layers(self, memory_manager):
        """Test that search_all searches requested layers."""
        student_id = uuid.uuid4()

        # Mock search methods
        memory_manager.episodic.search = AsyncMock(return_value=[])
        memory_manager.semantic.get_all_for_student = AsyncMock(return_value=[])
        memory_manager.associative.search = AsyncMock(return_value=[])

        from src.models.memory import MemorySearchRequest

        request = MemorySearchRequest(
            query="fractions",
            layers=["episodic", "associative"],
            limit_per_layer=5,
        )

        result = await memory_manager.search_all(
            tenant_code="test",
            student_id=student_id,
            request=request,
        )

        # Episodic and associative should be searched
        memory_manager.episodic.search.assert_called_once()
        memory_manager.associative.search.assert_called_once()

        # Semantic should not be searched (not in layers)
        memory_manager.semantic.get_all_for_student.assert_not_called()


@pytest.mark.unit
class TestMemoryManagerGetLearningSummary:
    """Test cases for MemoryManager.get_learning_summary method."""

    @pytest.mark.asyncio
    async def test_get_learning_summary_returns_statistics(self, memory_manager):
        """Test that get_learning_summary returns comprehensive stats."""
        student_id = uuid.uuid4()

        # Mock layer methods
        memory_manager.semantic.get_mastery_overview = AsyncMock(
            return_value=MasteryOverview(
                student_id=student_id,
                overall_mastery=0.7,
                topics_mastered=5,
                topics_learning=3,
                topics_struggling=2,
                total_topics=10,
                by_subject={},
            )
        )
        memory_manager.episodic.get_event_type_stats = AsyncMock(
            return_value={
                "breakthrough": 5,
                "struggle": 2,
                "correct_answer": 20,
            }
        )
        memory_manager.episodic.count_by_student = AsyncMock(return_value=50)
        memory_manager.associative.count_by_student = AsyncMock(return_value=10)
        memory_manager.procedural.get_learning_patterns = AsyncMock(
            return_value=LearningPatterns(
                student_id=student_id,
                best_time_of_day="afternoon",
                optimal_session_duration=None,
                preferred_content_format="text",
                hint_dependency=None,
                avg_break_frequency=None,
                preferred_difficulty=None,
                favorite_persona=None,
                vark_profile=None,
            )
        )

        result = await memory_manager.get_learning_summary(
            tenant_code="test",
            student_id=student_id,
        )

        assert result["mastery"]["overall"] == 0.7
        assert result["mastery"]["topics_mastered"] == 5
        assert result["engagement"]["total_episodes"] == 50
        assert result["personalization"]["interests_recorded"] == 10
        assert result["personalization"]["preferred_time"] == "afternoon"
