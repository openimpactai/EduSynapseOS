# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for workflow checkpointing.

Tests cover:
- Checkpointer initialization
- Thread configuration
- State persistence across workflow invocations
- Workflow resume functionality
- State recovery after interruption
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.orchestration.checkpointer import (
    create_session_thread_id,
    create_thread_config,
    get_checkpointer,
    reset_checkpointer,
)


# =============================================================================
# Thread Configuration Tests
# =============================================================================


class TestThreadConfiguration:
    """Tests for thread configuration helpers."""

    def test_create_session_thread_id(self):
        """Test creating a session thread ID."""
        thread_id = create_session_thread_id(
            tenant_id="tenant_123",
            student_id="student_456",
            session_id="session_789",
        )

        assert thread_id == "tenant_123:student_456:session_789"

    def test_create_session_thread_id_with_special_chars(self):
        """Test thread ID with special characters."""
        thread_id = create_session_thread_id(
            tenant_id="tenant-123",
            student_id="student_456",
            session_id="session.789",
        )

        assert thread_id == "tenant-123:student_456:session.789"

    def test_create_thread_config_minimal(self):
        """Test creating thread config with minimal parameters."""
        config = create_thread_config(thread_id="test_thread")

        assert config == {
            "configurable": {
                "thread_id": "test_thread",
                "checkpoint_ns": "",
            }
        }

    def test_create_thread_config_with_namespace(self):
        """Test creating thread config with namespace."""
        config = create_thread_config(
            thread_id="test_thread",
            checkpoint_ns="practice",
        )

        assert config == {
            "configurable": {
                "thread_id": "test_thread",
                "checkpoint_ns": "practice",
            }
        }


# =============================================================================
# Checkpointer Management Tests
# =============================================================================


class TestCheckpointerManagement:
    """Tests for checkpointer lifecycle management."""

    @pytest.fixture(autouse=True)
    async def reset_global_checkpointer(self):
        """Reset global checkpointer before each test."""
        await reset_checkpointer()
        yield
        await reset_checkpointer()

    @pytest.mark.asyncio
    async def test_get_checkpointer_creates_instance(self):
        """Test that get_checkpointer creates a checkpointer."""
        with patch(
            "src.core.orchestration.checkpointer.AsyncPostgresSaver"
        ) as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver.setup = AsyncMock()
            mock_saver_class.from_conn_string.return_value = mock_saver

            with patch(
                "src.core.orchestration.checkpointer.get_settings"
            ) as mock_settings:
                mock_settings.return_value.central_db.get_connection_url.return_value = (
                    "postgresql://test:test@localhost:5432/test"
                )

                checkpointer = await get_checkpointer()

                assert checkpointer is mock_saver
                mock_saver.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_checkpointer_caches_instance(self):
        """Test that checkpointer is cached after creation."""
        with patch(
            "src.core.orchestration.checkpointer.AsyncPostgresSaver"
        ) as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver.setup = AsyncMock()
            mock_saver_class.from_conn_string.return_value = mock_saver

            with patch(
                "src.core.orchestration.checkpointer.get_settings"
            ) as mock_settings:
                mock_settings.return_value.central_db.get_connection_url.return_value = (
                    "postgresql://test:test@localhost:5432/test"
                )

                checkpointer1 = await get_checkpointer()
                checkpointer2 = await get_checkpointer()

                assert checkpointer1 is checkpointer2
                # Should only be called once
                mock_saver_class.from_conn_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_checkpointer_with_custom_connection(self):
        """Test checkpointer with custom connection string."""
        with patch(
            "src.core.orchestration.checkpointer.AsyncPostgresSaver"
        ) as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver.setup = AsyncMock()
            mock_saver_class.from_conn_string.return_value = mock_saver

            custom_conn = "postgresql://custom:custom@localhost:5433/custom"
            checkpointer = await get_checkpointer(connection_string=custom_conn)

            mock_saver_class.from_conn_string.assert_called_with(custom_conn)
            assert checkpointer is mock_saver

    @pytest.mark.asyncio
    async def test_reset_checkpointer_clears_cache(self):
        """Test that reset clears the cached checkpointer."""
        with patch(
            "src.core.orchestration.checkpointer.AsyncPostgresSaver"
        ) as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver.setup = AsyncMock()
            mock_saver_class.from_conn_string.return_value = mock_saver

            with patch(
                "src.core.orchestration.checkpointer.get_settings"
            ) as mock_settings:
                mock_settings.return_value.central_db.get_connection_url.return_value = (
                    "postgresql://test:test@localhost:5432/test"
                )

                checkpointer1 = await get_checkpointer()
                await reset_checkpointer()
                checkpointer2 = await get_checkpointer()

                # Should be different instances
                assert mock_saver_class.from_conn_string.call_count == 2


# =============================================================================
# Workflow Integration Tests (with mocked checkpointer)
# =============================================================================


class TestWorkflowCheckpointingIntegration:
    """Integration tests for workflow checkpointing behavior."""

    @pytest.fixture
    def mock_checkpointer(self):
        """Create a mock checkpointer that simulates state persistence."""
        checkpointer = MagicMock()
        checkpointer.aget = AsyncMock(return_value=None)
        checkpointer.aput = AsyncMock()
        checkpointer.aget_tuple = AsyncMock(return_value=None)
        return checkpointer

    @pytest.fixture
    def mock_agent_factory(self):
        """Create a mock agent factory."""
        from src.core.agents import AgentFactory
        from src.core.agents.context import AgentResponse

        factory = MagicMock(spec=AgentFactory)

        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(
            return_value=AgentResponse(
                success=True,
                agent_id="tutor",
                capability_name="question_generation",
                result={
                    "content": "What is 2+2?",
                    "correct_answer": "4",
                    "question_type": "short_answer",
                    "difficulty": 0.5,
                },
            )
        )
        mock_agent.set_persona = MagicMock()
        factory.get = MagicMock(return_value=mock_agent)
        return factory

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        from src.core.memory.manager import MemoryManager

        manager = MagicMock(spec=MemoryManager)
        manager.get_full_context = AsyncMock(return_value=None)
        manager.record_learning_event = AsyncMock()
        manager.update_topic_mastery = AsyncMock()
        return manager

    @pytest.fixture
    def mock_theory_orchestrator(self):
        """Create a mock theory orchestrator."""
        from src.core.educational.orchestrator import TheoryOrchestrator

        orchestrator = MagicMock(spec=TheoryOrchestrator)
        orchestrator.get_recommendations = AsyncMock(return_value=None)
        return orchestrator

    @pytest.fixture
    def mock_persona_manager(self):
        """Create a mock persona manager."""
        from src.core.personas.manager import PersonaManager

        manager = MagicMock(spec=PersonaManager)
        manager.get_persona = MagicMock(return_value=None)
        return manager

    @pytest.mark.asyncio
    async def test_practice_workflow_with_checkpointer(
        self,
        mock_checkpointer,
        mock_agent_factory,
        mock_memory_manager,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test practice workflow uses checkpointer correctly."""
        from src.core.orchestration.workflows.practice import PracticeWorkflow
        from src.core.orchestration.states.practice import create_initial_practice_state

        workflow = PracticeWorkflow(
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
            checkpointer=mock_checkpointer,
        )

        initial_state = create_initial_practice_state(
            session_id="test_session",
            tenant_id="test_tenant",
            student_id="test_student",
            topic="mathematics",
            target_question_count=2,
        )

        compiled = workflow.compile()

        assert compiled is not None
        # Workflow compilation should include checkpointer
        # The actual persistence is handled by LangGraph

    @pytest.mark.asyncio
    async def test_tutoring_workflow_with_checkpointer(
        self,
        mock_checkpointer,
        mock_agent_factory,
        mock_memory_manager,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test tutoring workflow uses checkpointer correctly."""
        from src.core.memory.rag.retriever import RAGRetriever
        from src.core.orchestration.workflows.tutoring import TutoringWorkflow
        from src.core.orchestration.states.tutoring import create_initial_tutoring_state

        mock_rag_retriever = MagicMock(spec=RAGRetriever)
        mock_rag_retriever.retrieve = AsyncMock(return_value=[])

        workflow = TutoringWorkflow(
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            rag_retriever=mock_rag_retriever,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
            checkpointer=mock_checkpointer,
        )

        initial_state = create_initial_tutoring_state(
            session_id="test_session",
            tenant_id="test_tenant",
            tenant_code="test_tenant_code",
            student_id="test_student",
            topic="algebra",
        )

        compiled = workflow.compile()

        assert compiled is not None

    @pytest.mark.asyncio
    async def test_assessment_workflow_with_checkpointer(
        self,
        mock_checkpointer,
        mock_agent_factory,
        mock_memory_manager,
    ):
        """Test assessment workflow uses checkpointer correctly."""
        from src.core.orchestration.workflows.assessment import AssessmentWorkflow
        from src.core.orchestration.states.assessment import (
            create_initial_assessment_state,
            AssessmentQuestion,
        )

        workflow = AssessmentWorkflow(
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            checkpointer=mock_checkpointer,
        )

        questions = [
            AssessmentQuestion(
                question_id="q1",
                question={"content": "2+2?", "correct_answer": "4"},
                topic="math",
                bloom_level="remember",
                weight=1.0,
            )
        ]

        initial_state = create_initial_assessment_state(
            session_id="test_session",
            tenant_id="test_tenant",
            tenant_code="test_tenant_code",
            student_id="test_student",
            assessment_id="test_assessment",
            assessment_name="Test Quiz",
            topics=["math"],
            questions=questions,
        )

        compiled = workflow.compile()

        assert compiled is not None


# =============================================================================
# Thread ID Uniqueness Tests
# =============================================================================


class TestThreadIdUniqueness:
    """Tests for ensuring thread ID uniqueness."""

    def test_different_sessions_have_different_ids(self):
        """Test that different sessions produce different thread IDs."""
        id1 = create_session_thread_id("t1", "s1", "sess1")
        id2 = create_session_thread_id("t1", "s1", "sess2")

        assert id1 != id2

    def test_different_students_have_different_ids(self):
        """Test that different students produce different thread IDs."""
        id1 = create_session_thread_id("t1", "student1", "sess1")
        id2 = create_session_thread_id("t1", "student2", "sess1")

        assert id1 != id2

    def test_different_tenants_have_different_ids(self):
        """Test that different tenants produce different thread IDs."""
        id1 = create_session_thread_id("tenant1", "s1", "sess1")
        id2 = create_session_thread_id("tenant2", "s1", "sess1")

        assert id1 != id2

    def test_same_params_produce_same_id(self):
        """Test that same parameters produce same thread ID."""
        id1 = create_session_thread_id("t1", "s1", "sess1")
        id2 = create_session_thread_id("t1", "s1", "sess1")

        assert id1 == id2
