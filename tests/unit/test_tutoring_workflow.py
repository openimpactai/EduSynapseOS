# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for TutoringWorkflow.

Tests cover:
- Workflow initialization and graph building
- Node implementations (mocked dependencies)
- Intent detection
- State transitions and conditional edges
- Conversation history management
- Error handling
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents import AgentFactory, AgentExecutionContext
from src.core.agents.context import AgentResponse
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.personas.manager import PersonaManager
from src.core.orchestration.states.tutoring import (
    TutoringState,
    ConversationTurn,
    ExplanationRecord,
    TutoringMetrics,
    create_initial_tutoring_state,
)
from src.core.orchestration.workflows.tutoring import TutoringWorkflow


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_factory() -> MagicMock:
    """Create a mock agent factory."""
    from src.core.agents.capabilities.concept_explanation import ConceptExplanation

    factory = MagicMock(spec=AgentFactory)

    mock_agent = MagicMock()
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            success=True,
            agent_id="tutor",
            capability_name="concept_explanation",
            result=ConceptExplanation(
                success=True,
                capability_name="concept_explanation",
                concept="test concept",
                explanation="Here's an explanation of the concept...",
                summary="Brief summary of the concept",
            ),
        )
    )
    mock_agent.set_persona = MagicMock()

    factory.get = MagicMock(return_value=mock_agent)
    return factory


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock memory manager."""
    manager = MagicMock(spec=MemoryManager)
    manager.get_full_context = AsyncMock(return_value=None)
    manager.record_learning_event = AsyncMock()
    return manager


@pytest.fixture
def mock_rag_retriever() -> MagicMock:
    """Create a mock RAG retriever."""
    retriever = MagicMock(spec=RAGRetriever)
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def mock_theory_orchestrator() -> MagicMock:
    """Create a mock theory orchestrator."""
    orchestrator = MagicMock(spec=TheoryOrchestrator)
    orchestrator.get_recommendations = AsyncMock(return_value=None)
    return orchestrator


@pytest.fixture
def mock_persona_manager() -> MagicMock:
    """Create a mock persona manager."""
    manager = MagicMock(spec=PersonaManager)
    manager.get_persona = MagicMock(return_value=None)
    return manager


@pytest.fixture
def workflow(
    mock_agent_factory,
    mock_memory_manager,
    mock_rag_retriever,
    mock_theory_orchestrator,
    mock_persona_manager,
) -> TutoringWorkflow:
    """Create a TutoringWorkflow with mocked dependencies."""
    return TutoringWorkflow(
        agent_factory=mock_agent_factory,
        memory_manager=mock_memory_manager,
        rag_retriever=mock_rag_retriever,
        theory_orchestrator=mock_theory_orchestrator,
        persona_manager=mock_persona_manager,
        checkpointer=None,
    )


@pytest.fixture
def initial_state() -> TutoringState:
    """Create an initial tutoring state."""
    return create_initial_tutoring_state(
        session_id="session_123",
        tenant_id="tenant_456",
        tenant_code="test_tenant",
        student_id="student_789",
        topic="algebra",
    )


# =============================================================================
# State Creation Tests
# =============================================================================


class TestCreateInitialTutoringState:
    """Tests for create_initial_tutoring_state."""

    def test_create_with_minimal_params(self):
        """Test creating state with minimal parameters."""
        state = create_initial_tutoring_state(
            session_id="session_1",
            tenant_id="tenant_1",
            tenant_code="tenant_code_1",
            student_id="student_1",
            topic="math",
        )

        assert state["session_id"] == "session_1"
        assert state["tenant_id"] == "tenant_1"
        assert state["tenant_code"] == "tenant_code_1"
        assert state["student_id"] == "student_1"
        assert state["topic"] == "math"
        assert state["status"] == "pending"
        assert state["conversation_history"] == []
        assert state["awaiting_input"] is False

    def test_create_with_custom_persona(self):
        """Test creating state with custom persona."""
        state = create_initial_tutoring_state(
            session_id="session_1",
            tenant_id="tenant_1",
            tenant_code="tenant_code_1",
            student_id="student_1",
            topic="math",
            persona_id="friendly_tutor",
        )

        assert state["persona_id"] == "friendly_tutor"

    def test_initial_metrics(self):
        """Test that initial metrics are properly set."""
        state = create_initial_tutoring_state(
            session_id="s1",
            tenant_id="t1",
            tenant_code="tc1",
            student_id="st1",
            topic="math",
        )

        metrics = state["metrics"]
        assert metrics["turns_count"] == 0
        assert metrics["student_questions"] == 0
        assert metrics["explanations_given"] == 0
        assert metrics["clarifications_requested"] == 0


# =============================================================================
# Workflow Initialization Tests
# =============================================================================


class TestTutoringWorkflowInit:
    """Tests for TutoringWorkflow initialization."""

    def test_create_workflow(
        self,
        mock_agent_factory,
        mock_memory_manager,
        mock_rag_retriever,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test creating a workflow."""
        workflow = TutoringWorkflow(
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            rag_retriever=mock_rag_retriever,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
        )

        assert workflow._agent_factory is mock_agent_factory
        assert workflow._memory_manager is mock_memory_manager
        assert workflow._rag_retriever is mock_rag_retriever

    def test_workflow_has_graph(self, workflow):
        """Test that workflow has a built graph."""
        assert workflow._graph is not None

    def test_compile_workflow(self, workflow):
        """Test compiling the workflow."""
        compiled = workflow.compile()
        assert compiled is not None


# =============================================================================
# Node Implementation Tests
# =============================================================================


class TestTutoringWorkflowNodes:
    """Tests for individual workflow nodes."""

    @pytest.mark.asyncio
    async def test_initialize_node(self, workflow, initial_state, mock_memory_manager):
        """Test the initialize node."""
        result = await workflow._initialize(initial_state)

        assert result["status"] == "active"
        mock_memory_manager.get_full_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_handles_errors(
        self, workflow, initial_state, mock_memory_manager
    ):
        """Test that initialize handles errors gracefully."""
        mock_memory_manager.get_full_context.side_effect = Exception("DB error")

        result = await workflow._initialize(initial_state)

        assert result["status"] == "active"
        assert result["memory_context"] == {}

    @pytest.mark.asyncio
    async def test_wait_for_message_node(self, workflow, initial_state):
        """Test the wait_for_message node."""
        result = await workflow._wait_for_message(initial_state)

        assert result["awaiting_input"] is True
        assert "last_activity_at" in result

    @pytest.mark.asyncio
    async def test_analyze_intent_question(self, workflow, initial_state):
        """Test intent analysis for a question."""
        state = dict(initial_state)
        state["last_student_message"] = "What is a variable?"
        state["conversation_history"] = [
            ConversationTurn(
                role="student",
                content="What is a variable?",
                timestamp=datetime.now().isoformat(),
            )
        ]
        state["metrics"] = {"turns_count": 0, "student_questions": 0}

        result = await workflow._analyze_intent(state)

        assert result["metrics"]["turns_count"] == 1
        assert result["metrics"]["student_questions"] == 1

    @pytest.mark.asyncio
    async def test_analyze_intent_clarification(self, workflow, initial_state):
        """Test intent analysis for clarification request."""
        state = dict(initial_state)
        state["last_student_message"] = "I don't understand that"
        state["conversation_history"] = [
            ConversationTurn(
                role="student",
                content="I don't understand that",
                timestamp=datetime.now().isoformat(),
            )
        ]
        state["metrics"] = {
            "turns_count": 0,
            "student_questions": 0,
            "clarifications_requested": 0,
        }

        result = await workflow._analyze_intent(state)

        assert result["metrics"]["clarifications_requested"] == 1

    @pytest.mark.asyncio
    async def test_analyze_intent_empty_message(self, workflow, initial_state):
        """Test intent analysis with empty message."""
        state = dict(initial_state)
        state["last_student_message"] = ""

        result = await workflow._analyze_intent(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_retrieve_context_node(self, workflow, initial_state, mock_rag_retriever):
        """Test the retrieve_context node."""
        mock_rag_result = MagicMock()
        mock_rag_result.content = "Related content about algebra"
        mock_rag_result.source = MagicMock(value="textbook")
        mock_rag_result.score = 0.85
        mock_rag_retriever.retrieve.return_value = [mock_rag_result]

        state = dict(initial_state)
        state["last_student_message"] = "What is x in algebra?"

        result = await workflow._retrieve_context(state)

        assert "rag_context" in result
        assert len(result["rag_context"]) == 1
        mock_rag_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_context_handles_errors(
        self, workflow, initial_state, mock_rag_retriever
    ):
        """Test that retrieve_context handles errors."""
        mock_rag_retriever.retrieve.side_effect = Exception("RAG error")

        state = dict(initial_state)
        state["last_student_message"] = "Question?"

        result = await workflow._retrieve_context(state)

        assert result["rag_context"] == []

    @pytest.mark.asyncio
    async def test_generate_response_node(self, workflow, initial_state, mock_agent_factory):
        """Test the generate_response node."""
        state = dict(initial_state)
        state["last_student_message"] = "Explain variables"
        state["conversation_history"] = []
        state["concepts_explained"] = []
        state["rag_context"] = []
        state["metrics"] = {"explanations_given": 0, "concepts_covered": []}
        state["current_focus"] = "variables"

        result = await workflow._generate_response(state)

        assert "last_tutor_response" in result
        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["role"] == "tutor"
        assert result["awaiting_input"] is True
        mock_agent_factory.get.assert_called_with("tutor")

    @pytest.mark.asyncio
    async def test_generate_response_handles_failure(
        self, workflow, initial_state, mock_agent_factory
    ):
        """Test generate_response handles agent failure."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(
            return_value=AgentResponse(
                success=False,
                agent_id="tutor",
                capability_name="concept_explanation",
            )
        )
        mock_agent.set_persona = MagicMock()
        mock_agent_factory.get.return_value = mock_agent

        state = dict(initial_state)
        state["last_student_message"] = "Explain this"

        result = await workflow._generate_response(state)

        assert result["error"] == "Failed to generate response"
        assert result["awaiting_input"] is True

    @pytest.mark.asyncio
    async def test_update_memory_node(self, workflow, initial_state, mock_memory_manager):
        """Test the update_memory node."""
        state = dict(initial_state)
        state["last_student_message"] = "What is x?"
        state["last_tutor_response"] = "X is a variable..."
        state["current_focus"] = "variables"

        result = await workflow._update_memory(state)

        assert "last_activity_at" in result
        mock_memory_manager.record_learning_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_end_updates_duration(self, workflow, initial_state):
        """Test that check_end updates session duration."""
        state = dict(initial_state)
        state["started_at"] = datetime.now().isoformat()
        state["metrics"] = {}

        result = await workflow._check_end(state)

        assert "metrics" in result
        assert "total_duration_seconds" in result["metrics"]

    @pytest.mark.asyncio
    async def test_end_session_node(self, workflow, initial_state):
        """Test the end_session node."""
        state = dict(initial_state)
        state["metrics"] = {"turns_count": 5}

        result = await workflow._end_session(state)

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["awaiting_input"] is False


# =============================================================================
# Conditional Edge Tests
# =============================================================================


class TestTutoringWorkflowConditionalEdges:
    """Tests for workflow conditional routing."""

    def test_should_continue_normal_conversation(self, workflow):
        """Test continuing with normal conversation."""
        state: TutoringState = {
            "last_student_message": "What else?",
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "continue"

    def test_should_end_on_bye(self, workflow):
        """Test ending on goodbye."""
        state: TutoringState = {
            "last_student_message": "Thanks, bye!",
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "end"

    def test_should_end_on_thanks(self, workflow):
        """Test ending on thank you."""
        state: TutoringState = {
            "last_student_message": "Thank you for the help!",
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "end"

    def test_should_end_on_exit(self, workflow):
        """Test ending on exit."""
        state: TutoringState = {
            "last_student_message": "exit",
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "end"

    def test_should_end_on_error_status(self, workflow):
        """Test ending on error status."""
        state: TutoringState = {
            "last_student_message": "Continue",
            "status": "error",
        }

        result = workflow._should_continue(state)

        assert result == "end"


# =============================================================================
# Focus Extraction Tests
# =============================================================================


class TestFocusExtraction:
    """Tests for focus extraction helper."""

    def test_extract_focus_returns_topic(self, workflow):
        """Test that extract_focus returns the topic."""
        focus = workflow._extract_focus("What is algebra?", "mathematics")

        assert focus == "mathematics"


# =============================================================================
# Conversation History Tests
# =============================================================================


class TestConversationHistory:
    """Tests for conversation history management."""

    @pytest.mark.asyncio
    async def test_response_added_to_history(self, workflow, initial_state, mock_agent_factory):
        """Test that tutor response is added to conversation history."""
        state = dict(initial_state)
        state["last_student_message"] = "Explain this"
        state["conversation_history"] = [
            ConversationTurn(
                role="student",
                content="Explain this",
                timestamp=datetime.now().isoformat(),
            )
        ]
        state["concepts_explained"] = []
        state["rag_context"] = []
        state["metrics"] = {"explanations_given": 0, "concepts_covered": []}
        state["current_focus"] = "topic"

        result = await workflow._generate_response(state)

        assert len(result["conversation_history"]) == 2
        assert result["conversation_history"][1]["role"] == "tutor"

    @pytest.mark.asyncio
    async def test_concept_tracked_in_explanations(
        self, workflow, initial_state, mock_agent_factory
    ):
        """Test that explained concepts are tracked."""
        state = dict(initial_state)
        state["last_student_message"] = "What is a variable?"
        state["conversation_history"] = []
        state["concepts_explained"] = []
        state["rag_context"] = []
        state["metrics"] = {"explanations_given": 0, "concepts_covered": []}
        state["current_focus"] = "variables"

        result = await workflow._generate_response(state)

        assert len(result["concepts_explained"]) == 1
        assert result["concepts_explained"][0]["concept"] == "variables"
