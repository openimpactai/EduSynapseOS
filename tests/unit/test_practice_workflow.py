# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for PracticeWorkflow.

Tests cover:
- Workflow initialization and graph building
- Node implementations (mocked dependencies)
- State transitions and conditional edges
- Difficulty calculation
- Error handling
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents import AgentFactory, AgentExecutionContext
from src.core.agents.context import AgentResponse
from src.core.agents.capabilities.question_generation import GeneratedQuestion
from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
from src.core.memory.manager import MemoryManager
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.personas.manager import PersonaManager
from src.core.orchestration.states.practice import (
    PracticeState,
    QuestionRecord,
    SessionMetrics,
    create_initial_practice_state,
)
from src.core.orchestration.workflows.practice import PracticeWorkflow


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_factory() -> MagicMock:
    """Create a mock agent factory."""
    factory = MagicMock(spec=AgentFactory)

    mock_agent = MagicMock()
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            success=True,
            agent_id="tutor",
            capability_name="question_generation",
            result={
                "content": "What is 2 + 2?",
                "question_type": "short_answer",
                "correct_answer": "4",
                "difficulty": 0.5,
                "bloom_level": "remember",
            },
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
    manager.update_topic_mastery = AsyncMock()
    return manager


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
def mock_rag_retriever() -> MagicMock:
    """Create a mock RAG retriever."""
    from src.core.memory.rag.retriever import RAGRetriever

    retriever = MagicMock(spec=RAGRetriever)
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def workflow(
    mock_agent_factory,
    mock_memory_manager,
    mock_rag_retriever,
    mock_theory_orchestrator,
    mock_persona_manager,
) -> PracticeWorkflow:
    """Create a PracticeWorkflow with mocked dependencies."""
    return PracticeWorkflow(
        agent_factory=mock_agent_factory,
        memory_manager=mock_memory_manager,
        rag_retriever=mock_rag_retriever,
        theory_orchestrator=mock_theory_orchestrator,
        persona_manager=mock_persona_manager,
        checkpointer=None,
    )


@pytest.fixture
def initial_state() -> PracticeState:
    """Create an initial practice state."""
    return create_initial_practice_state(
        session_id="session_123",
        tenant_id="tenant_456",
        tenant_code="test_tenant",
        student_id="student_789",
        topic="mathematics",
        target_question_count=5,
        difficulty=0.5,
        mode="standard",
    )


# =============================================================================
# State Creation Tests
# =============================================================================


class TestCreateInitialPracticeState:
    """Tests for create_initial_practice_state."""

    def test_create_with_minimal_params(self):
        """Test creating state with minimal parameters."""
        state = create_initial_practice_state(
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
        assert state["target_question_count"] == 5
        assert state["difficulty"] == 0.5
        assert state["mode"] == "standard"

    def test_create_with_custom_params(self):
        """Test creating state with custom parameters."""
        state = create_initial_practice_state(
            session_id="session_1",
            tenant_id="tenant_1",
            tenant_code="tenant_code_1",
            student_id="student_1",
            topic="science",
            target_question_count=10,
            difficulty=0.8,
            mode="adaptive",
            persona_id="friendly_tutor",
        )

        assert state["target_question_count"] == 10
        assert state["difficulty"] == 0.8
        assert state["mode"] == "adaptive"
        assert state["persona_id"] == "friendly_tutor"

    def test_initial_metrics(self):
        """Test that initial metrics are properly set."""
        state = create_initial_practice_state(
            session_id="s1",
            tenant_id="t1",
            tenant_code="tc1",
            student_id="st1",
            topic="math",
        )

        metrics = state["metrics"]
        assert metrics["questions_answered"] == 0
        assert metrics["questions_correct"] == 0
        assert metrics["questions_partial"] == 0
        assert metrics["questions_incorrect"] == 0
        assert metrics["accuracy"] == 0.0
        assert metrics["streak_current"] == 0
        assert metrics["streak_max"] == 0


# =============================================================================
# Workflow Initialization Tests
# =============================================================================


class TestPracticeWorkflowInit:
    """Tests for PracticeWorkflow initialization."""

    def test_create_workflow(
        self,
        mock_agent_factory,
        mock_memory_manager,
        mock_theory_orchestrator,
        mock_persona_manager,
    ):
        """Test creating a workflow."""
        workflow = PracticeWorkflow(
            agent_factory=mock_agent_factory,
            memory_manager=mock_memory_manager,
            theory_orchestrator=mock_theory_orchestrator,
            persona_manager=mock_persona_manager,
        )

        assert workflow._agent_factory is mock_agent_factory
        assert workflow._memory_manager is mock_memory_manager
        assert workflow._theory_orchestrator is mock_theory_orchestrator
        assert workflow._persona_manager is mock_persona_manager

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


class TestPracticeWorkflowNodes:
    """Tests for individual workflow nodes."""

    @pytest.mark.asyncio
    async def test_initialize_node(self, workflow, initial_state):
        """Test the initialize node."""
        result = await workflow._initialize(initial_state)

        assert result["status"] == "active"
        assert result["current_question_index"] == 0
        assert result["questions"] == []

    @pytest.mark.asyncio
    async def test_load_context_node(self, workflow, initial_state, mock_memory_manager):
        """Test the load_context node."""
        result = await workflow._load_context(initial_state)

        assert "memory_context" in result
        assert "theory_recommendations" in result
        mock_memory_manager.get_full_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_context_handles_errors(
        self, workflow, initial_state, mock_memory_manager
    ):
        """Test that load_context handles errors gracefully."""
        mock_memory_manager.get_full_context.side_effect = Exception("DB error")

        result = await workflow._load_context(initial_state)

        assert result["memory_context"] == {}
        assert result["theory_recommendations"] == {}

    @pytest.mark.asyncio
    async def test_generate_question_node(self, workflow, initial_state, mock_agent_factory):
        """Test the generate_question node."""
        state = dict(initial_state)
        state["current_question_index"] = 0
        state["questions"] = []

        result = await workflow._generate_question(state)

        assert result["awaiting_answer"] is True
        assert result["error"] is None
        assert len(result["questions"]) == 1
        assert result["current_question"] is not None
        mock_agent_factory.get.assert_called_with("tutor")

    @pytest.mark.asyncio
    async def test_generate_question_handles_failure(
        self, workflow, initial_state, mock_agent_factory
    ):
        """Test generate_question handles agent failure."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(
            return_value=AgentResponse(
                success=False,
                agent_id="tutor",
                capability_name="question_generation",
            )
        )
        mock_agent.set_persona = MagicMock()
        mock_agent_factory.get.return_value = mock_agent

        state = dict(initial_state)
        state["current_question_index"] = 0

        result = await workflow._generate_question(state)

        assert result["status"] == "error"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_wait_for_answer_node(self, workflow, initial_state):
        """Test the wait_for_answer node."""
        result = await workflow._wait_for_answer(initial_state)

        assert result["awaiting_answer"] is True

    @pytest.mark.asyncio
    async def test_evaluate_answer_node(self, workflow, initial_state, mock_agent_factory):
        """Test the evaluate_answer node."""
        # Create a proper evaluation result structure
        from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
        from src.models.practice import EvaluationMethod

        eval_result = AnswerEvaluationResult(
            success=True,
            capability_name="answer_evaluation",
            is_correct=True,
            score=1.0,
            feedback="Correct!",
            correct_answer="4",
            student_answer="4",
            evaluation_method=EvaluationMethod.AI,
        )

        # Setup evaluation agent response
        eval_response = AgentResponse(
            success=True,
            agent_id="assessor",
            capability_name="answer_evaluation",
            result=eval_result,
        )
        mock_eval_agent = MagicMock()
        mock_eval_agent.execute = AsyncMock(return_value=eval_response)
        mock_agent_factory.get.return_value = mock_eval_agent

        state = dict(initial_state)
        state["current_question"] = {
            "content": "What is 2+2?",
            "correct_answer": "4",
            "question_type": "short_answer",
        }
        state["questions"] = [
            QuestionRecord(
                question_id="q1",
                question=state["current_question"],
                student_answer=None,
                evaluation=None,
                feedback=None,
                time_taken_seconds=None,
                hints_used=0,
            )
        ]
        state["_pending_answer"] = "4"
        state["metrics"] = {
            "questions_answered": 0,
            "questions_correct": 0,
            "questions_partial": 0,
            "questions_incorrect": 0,
            "accuracy": 0.0,
            "streak_current": 0,
            "streak_max": 0,
        }

        result = await workflow._evaluate_answer(state)

        assert result["awaiting_answer"] is False
        assert result["_pending_answer"] is None
        assert result["metrics"]["questions_answered"] == 1
        assert result["metrics"]["questions_correct"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_answer_incorrect(self, workflow, initial_state, mock_agent_factory):
        """Test evaluate_answer with incorrect answer."""
        from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
        from src.models.practice import EvaluationMethod

        eval_result = AnswerEvaluationResult(
            success=True,
            capability_name="answer_evaluation",
            is_correct=False,
            score=0.0,
            feedback="Incorrect.",
            correct_answer="4",
            student_answer="5",
            evaluation_method=EvaluationMethod.AI,
        )

        eval_response = AgentResponse(
            success=True,
            agent_id="assessor",
            capability_name="answer_evaluation",
            result=eval_result,
        )
        mock_eval_agent = MagicMock()
        mock_eval_agent.execute = AsyncMock(return_value=eval_response)
        mock_agent_factory.get.return_value = mock_eval_agent

        state = dict(initial_state)
        state["current_question"] = {"content": "2+2?", "correct_answer": "4"}
        state["questions"] = [QuestionRecord(
            question_id="q1",
            question=state["current_question"],
            student_answer=None,
            evaluation=None,
            feedback=None,
            time_taken_seconds=None,
            hints_used=0,
        )]
        state["_pending_answer"] = "5"
        state["metrics"] = {
            "questions_answered": 0,
            "questions_correct": 0,
            "questions_partial": 0,
            "questions_incorrect": 0,
            "accuracy": 0.0,
            "streak_current": 2,
            "streak_max": 2,
        }

        result = await workflow._evaluate_answer(state)

        assert result["metrics"]["questions_incorrect"] == 1
        assert result["metrics"]["streak_current"] == 0

    @pytest.mark.asyncio
    async def test_check_completion_increments_index(self, workflow, initial_state):
        """Test that check_completion increments question index."""
        state = dict(initial_state)
        state["current_question_index"] = 2

        result = await workflow._check_completion(state)

        assert result["current_question_index"] == 3

    @pytest.mark.asyncio
    async def test_complete_session_node(self, workflow, initial_state):
        """Test the complete_session node."""
        state = dict(initial_state)
        state["metrics"] = {
            "questions_answered": 5,
            "questions_correct": 4,
            "accuracy": 0.8,
        }

        result = await workflow._complete_session(state)

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["awaiting_answer"] is False


# =============================================================================
# Conditional Edge Tests
# =============================================================================


class TestPracticeWorkflowConditionalEdges:
    """Tests for workflow conditional routing."""

    def test_should_continue_with_more_questions(self, workflow):
        """Test routing when more questions needed."""
        state: PracticeState = {
            "current_question_index": 2,
            "target_question_count": 5,
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "continue"

    def test_should_complete_at_target_count(self, workflow):
        """Test routing when target reached."""
        state: PracticeState = {
            "current_question_index": 5,
            "target_question_count": 5,
            "status": "active",
        }

        result = workflow._should_continue(state)

        assert result == "complete"

    def test_should_complete_on_error(self, workflow):
        """Test routing when error occurred."""
        state: PracticeState = {
            "current_question_index": 2,
            "target_question_count": 5,
            "status": "error",
        }

        result = workflow._should_continue(state)

        assert result == "complete"


# =============================================================================
# Difficulty Calculation Tests
# =============================================================================


class TestDifficultyCalculation:
    """Tests for ZPD-based difficulty calculation.

    The new difficulty calculation uses ZPD theory recommendations
    when adaptive mode is enabled. It also respects mode_config
    settings for difficulty range constraints.
    """

    def test_standard_mode_uses_base_difficulty(self, workflow):
        """Test that standard mode uses base difficulty (no ZPD)."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "standard",
            "mode_config": {},
            "theory_recommendations": {"difficulty": 0.8},  # Should be ignored
            "metrics": {"accuracy": 1.0, "streak_current": 10},
        }

        result = workflow._calculate_next_difficulty(state)

        assert result == 0.5

    def test_adaptive_mode_uses_zpd_difficulty(self, workflow):
        """Test that adaptive mode uses ZPD theory difficulty."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "adaptive",
            "mode_config": {},
            "theory_recommendations": {"difficulty": 0.7},
            "metrics": {"accuracy": 0.9, "streak_current": 3},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should use ZPD difficulty, not hardcoded logic
        assert result == 0.7

    def test_adaptive_mode_fallback_to_base(self, workflow):
        """Test fallback to base difficulty when ZPD not available."""
        state: PracticeState = {
            "difficulty": 0.6,
            "mode": "adaptive",
            "mode_config": {},
            "theory_recommendations": {},  # No ZPD difficulty
            "metrics": {"accuracy": 0.3, "streak_current": 0},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should fallback to base difficulty
        assert result == 0.6

    def test_mode_config_adaptive_override(self, workflow):
        """Test mode_config can enable adaptive for non-adaptive mode."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "standard",  # Not adaptive
            "mode_config": {"adaptive_difficulty": True},  # But config says adaptive
            "theory_recommendations": {"difficulty": 0.75},
            "metrics": {},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should use ZPD because mode_config enables adaptive
        assert result == 0.75

    def test_mode_config_difficulty_range_constraint(self, workflow):
        """Test that difficulty_range from mode_config is applied."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "adaptive",
            "mode_config": {
                "adaptive_difficulty": True,
                "difficulty_range": [0.3, 0.7],  # Constrain to this range
            },
            "theory_recommendations": {"difficulty": 0.9},  # Above max
            "metrics": {},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should be clamped to max of range
        assert result == 0.7

    def test_mode_config_difficulty_range_min(self, workflow):
        """Test that difficulty_range minimum is enforced."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "adaptive",
            "mode_config": {
                "adaptive_difficulty": True,
                "difficulty_range": [0.4, 0.8],  # Constrain to this range
            },
            "theory_recommendations": {"difficulty": 0.2},  # Below min
            "metrics": {},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should be clamped to min of range
        assert result == 0.4

    def test_mode_config_disables_adaptive(self, workflow):
        """Test mode_config can disable adaptive for adaptive mode."""
        state: PracticeState = {
            "difficulty": 0.5,
            "mode": "adaptive",  # Normally adaptive
            "mode_config": {"adaptive_difficulty": False},  # But config disables
            "theory_recommendations": {"difficulty": 0.9},
            "metrics": {},
        }

        result = workflow._calculate_next_difficulty(state)

        # Should use base difficulty because mode_config disables adaptive
        assert result == 0.5


# =============================================================================
# Memory Update Tests
# =============================================================================


class TestPracticeWorkflowMemoryUpdates:
    """Tests for memory updates during workflow."""

    @pytest.mark.asyncio
    async def test_update_memory_on_correct_answer(
        self, workflow, initial_state, mock_memory_manager
    ):
        """Test memory update on correct answer."""
        from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
        from src.models.practice import EvaluationMethod

        eval_result = AnswerEvaluationResult(
            success=True,
            capability_name="answer_evaluation",
            is_correct=True,
            score=1.0,
            feedback="Correct!",
            correct_answer="4",
            student_answer="4",
            evaluation_method=EvaluationMethod.AI,
        )

        state = dict(initial_state)
        state["questions"] = [
            {
                "question_id": "q1",
                "evaluation": eval_result,
            }
        ]

        await workflow._update_memory(state)

        mock_memory_manager.record_learning_event.assert_called_once()
        mock_memory_manager.update_topic_mastery.assert_called_once_with(
            tenant_code=state["tenant_code"],
            student_id=state["student_id"],
            topic=state["topic"],
            score_delta=0.05,
        )

    @pytest.mark.asyncio
    async def test_update_memory_on_incorrect_answer(
        self, workflow, initial_state, mock_memory_manager
    ):
        """Test memory update on incorrect answer."""
        from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
        from src.models.practice import EvaluationMethod

        eval_result = AnswerEvaluationResult(
            success=True,
            capability_name="answer_evaluation",
            is_correct=False,
            score=0.2,
            feedback="Incorrect.",
            correct_answer="4",
            student_answer="5",
            evaluation_method=EvaluationMethod.AI,
        )

        state = dict(initial_state)
        state["questions"] = [
            {
                "question_id": "q1",
                "evaluation": eval_result,
            }
        ]

        await workflow._update_memory(state)

        mock_memory_manager.update_topic_mastery.assert_called_once_with(
            tenant_code=state["tenant_code"],
            student_id=state["student_id"],
            topic=state["topic"],
            score_delta=-0.02,
        )

    @pytest.mark.asyncio
    async def test_update_memory_handles_errors(
        self, workflow, initial_state, mock_memory_manager
    ):
        """Test that memory update handles errors gracefully."""
        from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
        from src.models.practice import EvaluationMethod

        mock_memory_manager.record_learning_event.side_effect = Exception("DB error")

        eval_result = AnswerEvaluationResult(
            success=True,
            capability_name="answer_evaluation",
            is_correct=True,
            score=1.0,
            feedback="Correct!",
            correct_answer="4",
            student_answer="4",
            evaluation_method=EvaluationMethod.AI,
        )

        state = dict(initial_state)
        state["questions"] = [
            {"question_id": "q1", "evaluation": eval_result}
        ]

        # Should not raise
        result = await workflow._update_memory(state)

        assert result == {}
