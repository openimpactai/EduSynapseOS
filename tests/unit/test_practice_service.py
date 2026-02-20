# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for practice service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.domains.practice.service import (
    PracticeService,
    SessionNotFoundError,
    SessionNotActiveError,
)
from src.models.practice import (
    StartPracticeRequest,
    SessionType,
    SessionStatus,
)


@pytest.fixture
def mock_db():
    """Create mock database session."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_agent_factory():
    """Create mock agent factory."""
    factory = MagicMock()
    return factory


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager."""
    manager = MagicMock()
    manager.get_full_context = AsyncMock(return_value=None)
    manager.record_learning_event = AsyncMock()
    return manager


@pytest.fixture
def mock_rag_retriever():
    """Create mock RAG retriever."""
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def mock_theory_orchestrator():
    """Create mock theory orchestrator."""
    orchestrator = MagicMock()
    orchestrator.get_recommendations = AsyncMock(return_value=None)
    return orchestrator


@pytest.fixture
def mock_persona_manager():
    """Create mock persona manager."""
    manager = MagicMock()
    manager.get_persona = MagicMock(return_value=None)
    return manager


class TestPracticeService:
    """Tests for PracticeService class."""

    @patch("src.domains.practice.service.PracticeWorkflow")
    @patch("src.domains.practice.service.get_checkpointer")
    def test_init_creates_workflow(
        self,
        mock_get_checkpointer,
        mock_workflow_class,
        mock_db,
        mock_agent_factory,
        mock_memory_manager,
        mock_rag_retriever,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test that init creates the workflow."""
        mock_get_checkpointer.return_value = None

        service = PracticeService(
            db=mock_db,
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            rag_retriever=mock_rag_retriever,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
        )

        mock_workflow_class.assert_called_once()

    @patch("src.domains.practice.service.PracticeWorkflow")
    @patch("src.domains.practice.service.get_checkpointer")
    @pytest.mark.asyncio
    async def test_start_session_creates_record(
        self,
        mock_get_checkpointer,
        mock_workflow_class,
        mock_db,
        mock_agent_factory,
        mock_memory_manager,
        mock_rag_retriever,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test that start_session creates a database record."""
        mock_get_checkpointer.return_value = None

        # Setup mock workflow
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value={
            "status": "active",
            "current_question": {
                "question_id": str(uuid4()),
                "content": "Test question?",
                "question_type": "short_answer",
            },
            "target_question_count": 10,
        })
        mock_workflow_class.return_value = mock_workflow

        service = PracticeService(
            db=mock_db,
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            rag_retriever=mock_rag_retriever,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
        )

        request = StartPracticeRequest(
            topic_id=uuid4(),
            session_type=SessionType.QUICK,
        )

        session, question = await service.start_session(
            student_id=uuid4(),
            tenant_id=uuid4(),
            request=request,
        )

        # Verify database operations
        mock_db.add.assert_called()
        mock_db.flush.assert_called()
        mock_db.commit.assert_called()

        # Verify workflow was run
        mock_workflow.run.assert_called_once()

    @patch("src.domains.practice.service.PracticeWorkflow")
    @patch("src.domains.practice.service.get_checkpointer")
    @pytest.mark.asyncio
    async def test_get_session_not_found_raises_error(
        self,
        mock_get_checkpointer,
        mock_workflow_class,
        mock_db,
        mock_agent_factory,
        mock_memory_manager,
        mock_rag_retriever,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test that get_session raises error when not found."""
        mock_get_checkpointer.return_value = None

        # Setup mock to return no result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = PracticeService(
            db=mock_db,
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            rag_retriever=mock_rag_retriever,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
        )

        with pytest.raises(SessionNotFoundError):
            await service.get_session(
                session_id=uuid4(),
                student_id=uuid4(),
            )


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError."""

    def test_error_message(self):
        """Test error message."""
        error = SessionNotFoundError("Session xyz not found")
        assert "xyz" in str(error)


class TestSessionNotActiveError:
    """Tests for SessionNotActiveError."""

    def test_error_message(self):
        """Test error message."""
        error = SessionNotActiveError("Session xyz is not active")
        assert "not active" in str(error)
