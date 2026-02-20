# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for memory layer services.

Tests the four memory layers:
- EpisodicMemoryLayer: Session-based learning events
- SemanticMemoryLayer: Knowledge state tracking
- ProceduralMemoryLayer: Learning strategies
- AssociativeMemoryLayer: Interests and connections

These tests use mocking for database and vector operations.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.memory.layers.episodic import EpisodicMemoryLayer
from src.core.memory.layers.semantic import SemanticMemoryLayer
from src.core.memory.layers.procedural import ProceduralMemoryLayer
from src.core.memory.layers.associative import AssociativeMemoryLayer
from src.models.memory import (
    AssociationType,
    EmotionalState,
    EntityType,
    EpisodicEventType,
    StrategyType,
)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def mock_tenant_db():
    """Create a mock TenantDatabaseManager."""
    manager = MagicMock()
    manager.get_session = MagicMock()
    return manager


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
    client._tenant_collection_name = MagicMock(
        return_value="tenant_test_episodic_memories"
    )
    client.collection_exists = AsyncMock(return_value=True)
    client.create_tenant_collection = AsyncMock()
    client.upsert_with_tenant = AsyncMock()
    client.search_with_tenant = AsyncMock(return_value=[])
    client.delete_points = AsyncMock()
    return client


# ============================================================================
# EpisodicMemoryLayer tests
# ============================================================================


@pytest.mark.unit
class TestEpisodicMemoryLayerInit:
    """Test cases for EpisodicMemoryLayer initialization."""

    def test_initialization(
        self, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test layer initializes with all dependencies."""
        layer = EpisodicMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

        assert layer._tenant_db == mock_tenant_db
        assert layer._embedding == mock_embedding_service
        assert layer._qdrant == mock_qdrant_client


@pytest.mark.unit
class TestEpisodicMemoryLayerStore:
    """Test cases for EpisodicMemoryLayer.store method."""

    @pytest.fixture
    def layer(self, mock_tenant_db, mock_embedding_service, mock_qdrant_client):
        """Create layer with mocked dependencies."""
        return EpisodicMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

    @pytest.mark.asyncio
    async def test_store_creates_embedding(
        self, layer, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test that store generates embedding for summary."""
        student_id = uuid.uuid4()

        # Mock session context manager
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        result = await layer.store(
            tenant_code="test",
            student_id=student_id,
            event_type=EpisodicEventType.BREAKTHROUGH,
            summary="Student understood fractions",
            importance=0.9,
        )

        # Verify embedding was generated
        mock_embedding_service.embed_text.assert_called_once_with(
            "Student understood fractions"
        )

        # Verify result
        assert result.student_id == student_id
        assert result.event_type == EpisodicEventType.BREAKTHROUGH
        assert result.summary == "Student understood fractions"
        assert result.importance == 0.9

    @pytest.mark.asyncio
    async def test_store_upserts_to_qdrant(
        self, layer, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test that store upserts vector to Qdrant."""
        student_id = uuid.uuid4()

        # Mock session
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        await layer.store(
            tenant_code="test",
            student_id=student_id,
            event_type=EpisodicEventType.STRUGGLE,
            summary="Difficulty with algebra",
        )

        # Verify Qdrant upsert was called
        mock_qdrant_client.upsert_with_tenant.assert_called_once()
        call_args = mock_qdrant_client.upsert_with_tenant.call_args

        assert call_args.kwargs["tenant_code"] == "test"
        assert call_args.kwargs["collection"] == "episodic_memories"
        assert len(call_args.kwargs["points"]) == 1


@pytest.mark.unit
class TestEpisodicMemoryLayerSearch:
    """Test cases for EpisodicMemoryLayer.search method."""

    @pytest.fixture
    def layer(self, mock_tenant_db, mock_embedding_service, mock_qdrant_client):
        """Create layer with mocked dependencies."""
        return EpisodicMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

    @pytest.mark.asyncio
    async def test_search_generates_query_embedding(
        self, layer, mock_embedding_service, mock_qdrant_client, mock_tenant_db
    ):
        """Test that search generates embedding for query."""
        student_id = uuid.uuid4()

        # Mock empty session result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        await layer.search(
            tenant_code="test",
            student_id=student_id,
            query="fractions",
            limit=5,
        )

        mock_embedding_service.embed_text.assert_called_once_with("fractions")


# ============================================================================
# SemanticMemoryLayer tests
# ============================================================================


@pytest.mark.unit
class TestSemanticMemoryLayerInit:
    """Test cases for SemanticMemoryLayer initialization."""

    def test_initialization(self, mock_tenant_db):
        """Test layer initializes with tenant db manager."""
        layer = SemanticMemoryLayer(tenant_db_manager=mock_tenant_db)

        assert layer._tenant_db == mock_tenant_db


@pytest.mark.unit
class TestSemanticMemoryLayerMastery:
    """Test cases for SemanticMemoryLayer mastery calculation."""

    @pytest.fixture
    def layer(self, mock_tenant_db):
        """Create layer with mocked dependencies."""
        return SemanticMemoryLayer(tenant_db_manager=mock_tenant_db)

    def test_calculate_mastery_zero_attempts(self, layer):
        """Test mastery is 0 with no attempts."""
        mastery = layer._calculate_mastery(
            attempts_correct=0,
            attempts_total=0,
            current_streak=0,
            best_streak=0,
        )
        assert mastery == 0.0

    def test_calculate_mastery_perfect_accuracy(self, layer):
        """Test mastery with perfect accuracy."""
        mastery = layer._calculate_mastery(
            attempts_correct=10,
            attempts_total=10,
            current_streak=10,
            best_streak=10,
        )
        # 0.6 (accuracy) + 0.25 (streak cap) + 0.15 (best streak cap) = 1.0
        assert mastery == 1.0

    def test_calculate_mastery_low_accuracy(self, layer):
        """Test mastery with low accuracy."""
        mastery = layer._calculate_mastery(
            attempts_correct=3,
            attempts_total=10,
            current_streak=0,
            best_streak=2,
        )
        # 0.3 * 0.6 = 0.18 (accuracy) + 0 (streak) + 0.1 (best_streak) = 0.28
        assert 0.2 <= mastery <= 0.35

    def test_calculate_mastery_streak_bonus(self, layer):
        """Test that streaks add bonus."""
        mastery_no_streak = layer._calculate_mastery(
            attempts_correct=5,
            attempts_total=10,
            current_streak=0,
            best_streak=0,
        )

        mastery_with_streak = layer._calculate_mastery(
            attempts_correct=5,
            attempts_total=10,
            current_streak=5,
            best_streak=5,
        )

        assert mastery_with_streak > mastery_no_streak


@pytest.mark.unit
class TestSemanticMemoryLayerRecordAttempt:
    """Test cases for SemanticMemoryLayer.record_attempt method."""

    @pytest.fixture
    def layer(self, mock_tenant_db):
        """Create layer with mocked dependencies."""
        return SemanticMemoryLayer(tenant_db_manager=mock_tenant_db)

    @pytest.mark.asyncio
    async def test_record_attempt_creates_if_not_exists(
        self, layer, mock_tenant_db
    ):
        """Test that record_attempt creates new memory if none exists."""
        student_id = uuid.uuid4()
        entity_id = uuid.uuid4()

        # Mock no existing memory
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        result = await layer.record_attempt(
            tenant_code="test",
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_id=entity_id,
            is_correct=True,
            time_seconds=30,
        )

        # Session.add should have been called for new memory
        mock_session.add.assert_called_once()


# ============================================================================
# ProceduralMemoryLayer tests
# ============================================================================


@pytest.mark.unit
class TestProceduralMemoryLayerInit:
    """Test cases for ProceduralMemoryLayer initialization."""

    def test_initialization(self, mock_tenant_db):
        """Test layer initializes with tenant db manager."""
        layer = ProceduralMemoryLayer(tenant_db_manager=mock_tenant_db)

        assert layer._tenant_db == mock_tenant_db


@pytest.mark.unit
class TestProceduralMemoryLayerRecordObservation:
    """Test cases for ProceduralMemoryLayer.record_observation method."""

    @pytest.fixture
    def layer(self, mock_tenant_db):
        """Create layer with mocked dependencies."""
        return ProceduralMemoryLayer(tenant_db_manager=mock_tenant_db)

    @pytest.mark.asyncio
    async def test_record_observation_creates_new(self, layer, mock_tenant_db):
        """Test that record_observation creates new memory if none exists."""
        student_id = uuid.uuid4()

        # Mock no existing memory
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        result = await layer.record_observation(
            tenant_code="test",
            student_id=student_id,
            strategy_type=StrategyType.TIME_OF_DAY,
            strategy_value="morning",
            was_effective=True,
        )

        # Session.add should have been called for new memory
        mock_session.add.assert_called_once()


# ============================================================================
# AssociativeMemoryLayer tests
# ============================================================================


@pytest.mark.unit
class TestAssociativeMemoryLayerInit:
    """Test cases for AssociativeMemoryLayer initialization."""

    def test_initialization(
        self, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test layer initializes with all dependencies."""
        layer = AssociativeMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

        assert layer._tenant_db == mock_tenant_db
        assert layer._embedding == mock_embedding_service
        assert layer._qdrant == mock_qdrant_client


@pytest.mark.unit
class TestAssociativeMemoryLayerStore:
    """Test cases for AssociativeMemoryLayer.store method."""

    @pytest.fixture
    def layer(self, mock_tenant_db, mock_embedding_service, mock_qdrant_client):
        """Create layer with mocked dependencies."""
        return AssociativeMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

    @pytest.mark.asyncio
    async def test_store_creates_embedding(
        self, layer, mock_tenant_db, mock_embedding_service, mock_qdrant_client
    ):
        """Test that store generates embedding for content."""
        student_id = uuid.uuid4()

        # Mock session
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        result = await layer.store(
            tenant_code="test",
            student_id=student_id,
            association_type=AssociationType.INTEREST,
            content="Loves playing Minecraft",
            tags=["gaming", "creativity"],
        )

        # Verify embedding was generated
        mock_embedding_service.embed_text.assert_called_once_with(
            "Loves playing Minecraft"
        )

        # Verify result
        assert result.student_id == student_id
        assert result.association_type == AssociationType.INTEREST
        assert result.content == "Loves playing Minecraft"
        assert result.tags == ["gaming", "creativity"]


@pytest.mark.unit
class TestAssociativeMemoryLayerSearch:
    """Test cases for AssociativeMemoryLayer.search method."""

    @pytest.fixture
    def layer(self, mock_tenant_db, mock_embedding_service, mock_qdrant_client):
        """Create layer with mocked dependencies."""
        return AssociativeMemoryLayer(
            tenant_db_manager=mock_tenant_db,
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
        )

    @pytest.mark.asyncio
    async def test_search_generates_query_embedding(
        self, layer, mock_embedding_service, mock_qdrant_client, mock_tenant_db
    ):
        """Test that search generates embedding for query."""
        student_id = uuid.uuid4()

        # Mock empty session result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_tenant_db.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_tenant_db.get_session.return_value.__aexit__ = AsyncMock()

        await layer.search(
            tenant_code="test",
            student_id=student_id,
            query="building games",
            limit=5,
        )

        mock_embedding_service.embed_text.assert_called_once_with("building games")
