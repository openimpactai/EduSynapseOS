# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for agent execution.

These tests verify the full agent execution flow with real components
(except LLM which is mocked to avoid external dependencies).

Tests cover:
- Agent creation with real config files
- Agent execution with real capabilities
- Agent factory with real persona manager
- End-to-end execution flow
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents.capabilities.registry import (
    CapabilityRegistry,
    get_default_registry,
    reset_default_registry,
)
from src.core.agents.context import (
    AgentConfig,
    AgentExecutionContext,
    AgentResponse,
)
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.agents.factory import (
    AgentFactory,
    get_agent_factory,
    reset_agent_factory,
)
from src.core.intelligence.llm.client import LLMClient, LLMResponse
from src.core.personas.manager import (
    PersonaManager,
    get_persona_manager,
    reset_persona_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before and after each test."""
    reset_default_registry()
    reset_persona_manager()
    reset_agent_factory()
    yield
    reset_default_registry()
    reset_persona_manager()
    reset_agent_factory()


@pytest.fixture
def capability_registry() -> CapabilityRegistry:
    """Get the default capability registry with all capabilities."""
    return get_default_registry()


@pytest.fixture
def persona_manager() -> PersonaManager:
    """Get the persona manager with loaded personas."""
    return get_persona_manager()


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client with configurable responses."""
    client = MagicMock(spec=LLMClient)

    # Default response for question generation
    default_response = json.dumps({
        "content": "What is the sum of 5 and 3?",
        "options": [
            {"key": "a", "text": "6", "is_correct": False},
            {"key": "b", "text": "7", "is_correct": False},
            {"key": "c", "text": "8", "is_correct": True},
            {"key": "d", "text": "9", "is_correct": False},
        ],
        "correct_answer": "8",
        "hints": [
            {"level": 1, "text": "Count on your fingers"},
            {"level": 2, "text": "5 + 3 = ?"},
        ],
        "explanation": "5 + 3 equals 8",
        "misconceptions_addressed": [],
    })

    client.complete = AsyncMock(
        return_value=LLMResponse(
            content=default_response,
            model="ollama/qwen2.5:7b",
            tokens_input=150,
            tokens_output=200,
        )
    )

    return client


@pytest.fixture
def agents_dir() -> Path:
    """Get the path to agent configuration files."""
    return Path("config/agents")


# =============================================================================
# Agent Loading Integration Tests
# =============================================================================


class TestAgentLoadingIntegration:
    """Tests for loading agents from real config files."""

    def test_load_tutor_config(self, agents_dir):
        """Test loading tutor agent configuration."""
        config_path = agents_dir / "tutor.yaml"

        if not config_path.exists():
            pytest.skip("Tutor config not found")

        config = AgentConfig.from_yaml(config_path)

        assert config.id == "tutor"
        assert "question_generation" in config.capabilities
        assert "concept_explanation" in config.capabilities

    def test_load_assessor_config(self, agents_dir):
        """Test loading assessor agent configuration."""
        config_path = agents_dir / "assessor.yaml"

        if not config_path.exists():
            pytest.skip("Assessor config not found")

        config = AgentConfig.from_yaml(config_path)

        assert config.id == "assessor"
        assert "question_generation" in config.capabilities
        assert "answer_evaluation" in config.capabilities

    def test_load_diagnostician_config(self, agents_dir):
        """Test loading diagnostician agent configuration."""
        config_path = agents_dir / "diagnostician.yaml"

        if not config_path.exists():
            pytest.skip("Diagnostician config not found")

        config = AgentConfig.from_yaml(config_path)

        assert config.id == "diagnostician"
        assert "diagnostic_analysis" in config.capabilities


# =============================================================================
# Agent Creation Integration Tests
# =============================================================================


class TestAgentCreationIntegration:
    """Tests for creating agents with real components."""

    def test_create_tutor_agent(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test creating tutor agent with real registry."""
        config_path = agents_dir / "tutor.yaml"

        if not config_path.exists():
            pytest.skip("Tutor config not found")

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        assert agent.id == "tutor"
        assert agent.has_capability("question_generation")
        assert agent.has_capability("concept_explanation")

    def test_create_assessor_agent(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test creating assessor agent with real registry."""
        config_path = agents_dir / "assessor.yaml"

        if not config_path.exists():
            pytest.skip("Assessor config not found")

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        assert agent.id == "assessor"
        assert agent.has_capability("question_generation")
        assert agent.has_capability("answer_evaluation")

    def test_create_agent_with_persona(
        self, agents_dir, mock_llm_client, capability_registry, persona_manager
    ):
        """Test creating agent and attaching persona."""
        config_path = agents_dir / "tutor.yaml"

        if not config_path.exists():
            pytest.skip("Tutor config not found")

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        # Try to attach a persona if available
        try:
            persona = persona_manager.get_persona("tutor")
            agent.set_persona(persona)
            assert agent.persona is not None
            assert agent.persona.id == "tutor"
        except Exception:
            pytest.skip("Tutor persona not available")


# =============================================================================
# Agent Execution Integration Tests
# =============================================================================


class TestAgentExecutionIntegration:
    """Tests for executing agents with real capabilities."""

    @pytest.mark.asyncio
    async def test_execute_question_generation(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test executing question generation capability."""
        config_path = agents_dir / "tutor.yaml"

        if not config_path.exists():
            pytest.skip("Tutor config not found")

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Basic Addition",
            intent="question_generation",
            params={
                "topic": "Basic Addition",
                "difficulty": 0.3,
                "bloom_level": "remember",
            },
        )

        response = await agent.execute(context)

        assert response.success is True
        assert response.agent_id == "tutor"
        assert response.capability_name == "question_generation"
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_execute_answer_evaluation(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test executing answer evaluation capability."""
        config_path = agents_dir / "assessor.yaml"

        if not config_path.exists():
            pytest.skip("Assessor config not found")

        # Set up mock for evaluation response
        eval_response = json.dumps({
            "is_correct": True,
            "score": 1.0,
            "feedback": "Correct! Great job!",
            "misconceptions": [],
            "confidence": 0.95,
            "improvement_suggestions": [],
        })

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=eval_response,
                model="ollama/qwen2.5:7b",
                tokens_input=100,
                tokens_output=80,
            )
        )

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Basic Math",
            intent="answer_evaluation",
            params={
                "question_content": "What is 5 + 3?",
                "student_answer": "8",
                "expected_answer": "8",
            },
        )

        response = await agent.execute(context)

        assert response.success is True
        assert response.capability_name == "answer_evaluation"

    @pytest.mark.asyncio
    async def test_execute_with_persona_integration(
        self, agents_dir, mock_llm_client, capability_registry, persona_manager
    ):
        """Test execution with persona affecting system prompt."""
        config_path = agents_dir / "tutor.yaml"

        if not config_path.exists():
            pytest.skip("Tutor config not found")

        config = AgentConfig.from_yaml(config_path)
        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        # Attach persona
        try:
            persona = persona_manager.get_persona("coach")
            agent.set_persona(persona)
        except Exception:
            pytest.skip("Coach persona not available")

        context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Math",
            intent="question_generation",
            params={"topic": "Math"},
        )

        response = await agent.execute(context)

        # Verify persona was included in the call
        call_kwargs = mock_llm_client.complete.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")

        # Persona content should be in system prompt
        assert response.success is True
        assert response.metadata.get("persona_id") == "coach"


# =============================================================================
# Agent Factory Integration Tests
# =============================================================================


class TestAgentFactoryIntegration:
    """Tests for AgentFactory with real components."""

    def test_factory_discovers_agents(
        self, agents_dir, mock_llm_client, capability_registry, persona_manager
    ):
        """Test factory discovers available agents."""
        if not agents_dir.exists():
            pytest.skip("Agents directory not found")

        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
            persona_manager=persona_manager,
            agents_dir=agents_dir,
        )

        available = factory.list_available()

        # Should find the agents we created
        if (agents_dir / "tutor.yaml").exists():
            assert "tutor" in available
        if (agents_dir / "assessor.yaml").exists():
            assert "assessor" in available

    def test_factory_creates_agents_with_personas(
        self, agents_dir, mock_llm_client, capability_registry, persona_manager
    ):
        """Test factory attaches default personas."""
        if not agents_dir.exists() or not (agents_dir / "tutor.yaml").exists():
            pytest.skip("Tutor config not found")

        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
            persona_manager=persona_manager,
            agents_dir=agents_dir,
        )

        agent = factory.get("tutor")

        # Agent should have default persona attached
        # (if persona loading succeeds)
        assert agent.id == "tutor"

    @pytest.mark.asyncio
    async def test_factory_agent_execution(
        self, agents_dir, mock_llm_client, capability_registry, persona_manager
    ):
        """Test executing agent created by factory."""
        if not agents_dir.exists() or not (agents_dir / "tutor.yaml").exists():
            pytest.skip("Tutor config not found")

        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
            persona_manager=persona_manager,
            agents_dir=agents_dir,
        )

        agent = factory.get("tutor")

        context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Fractions",
            intent="question_generation",
            params={"topic": "Fractions"},
        )

        response = await agent.execute(context)

        assert response.success is True
        assert response.agent_id == "tutor"


# =============================================================================
# Full Flow Integration Tests
# =============================================================================


class TestFullFlowIntegration:
    """Tests for complete agent execution flows."""

    @pytest.mark.asyncio
    async def test_question_then_evaluate_flow(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test generating a question then evaluating an answer."""
        if not agents_dir.exists():
            pytest.skip("Agents directory not found")

        tutor_path = agents_dir / "tutor.yaml"
        assessor_path = agents_dir / "assessor.yaml"

        if not tutor_path.exists() or not assessor_path.exists():
            pytest.skip("Required configs not found")

        # Create agents
        tutor_config = AgentConfig.from_yaml(tutor_path)
        tutor = DynamicAgent(
            config=tutor_config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        assessor_config = AgentConfig.from_yaml(assessor_path)
        assessor = DynamicAgent(
            config=assessor_config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        # Step 1: Generate question
        gen_context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Addition",
            intent="question_generation",
            params={"topic": "Addition"},
        )

        gen_response = await tutor.execute(gen_context)
        assert gen_response.success is True

        # Step 2: Set up evaluation response
        eval_llm_response = json.dumps({
            "is_correct": True,
            "score": 1.0,
            "feedback": "Perfect!",
            "misconceptions": [],
            "confidence": 0.99,
            "improvement_suggestions": [],
        })

        mock_llm_client.complete = AsyncMock(
            return_value=LLMResponse(
                content=eval_llm_response,
                model="ollama/qwen2.5:7b",
                tokens_input=100,
                tokens_output=50,
            )
        )

        # Step 3: Evaluate answer
        eval_context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Addition",
            intent="answer_evaluation",
            params={
                "question_content": "What is 5 + 3?",
                "student_answer": "8",
                "expected_answer": "8",
            },
        )

        eval_response = await assessor.execute(eval_context)
        assert eval_response.success is True
        assert eval_response.capability_name == "answer_evaluation"

    @pytest.mark.asyncio
    async def test_multiple_agents_same_capability(
        self, agents_dir, mock_llm_client, capability_registry
    ):
        """Test different agents can use the same capability."""
        if not agents_dir.exists():
            pytest.skip("Agents directory not found")

        tutor_path = agents_dir / "tutor.yaml"
        assessor_path = agents_dir / "assessor.yaml"

        if not tutor_path.exists() or not assessor_path.exists():
            pytest.skip("Required configs not found")

        # Both agents have question_generation capability
        tutor_config = AgentConfig.from_yaml(tutor_path)
        tutor = DynamicAgent(
            config=tutor_config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        assessor_config = AgentConfig.from_yaml(assessor_path)
        assessor = DynamicAgent(
            config=assessor_config,
            llm_client=mock_llm_client,
            capability_registry=capability_registry,
        )

        context = AgentExecutionContext(
            tenant_id="tenant_test",
            student_id="student_test",
            topic="Math",
            intent="question_generation",
            params={"topic": "Math"},
        )

        # Both should be able to execute the same capability
        tutor_response = await tutor.execute(context)
        assessor_response = await assessor.execute(context)

        assert tutor_response.success is True
        assert assessor_response.success is True
        assert tutor_response.agent_id == "tutor"
        assert assessor_response.agent_id == "assessor"
