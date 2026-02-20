# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for DynamicAgent and related classes.

Tests cover:
- AgentConfig loading from YAML
- AgentExecutionContext building
- DynamicAgent initialization and execution
- AgentFactory operations
- Error handling
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityResult,
)
from src.core.agents.capabilities.registry import CapabilityRegistry
from src.core.agents.context import (
    AgentConfig,
    AgentDomainConfig,
    AgentError,
    AgentExecutionContext,
    AgentLLMConfig,
    AgentResponse,
    LLMRoutingStrategy,
)
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.agents.factory import AgentFactory, AgentNotFoundError
from src.core.intelligence.llm.client import LLMClient, LLMResponse
from src.core.personas.models import (
    Persona,
    PersonaIdentity,
    PersonaVoice,
    Tone,
    Formality,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Create a sample agent configuration."""
    return AgentConfig(
        id="test_agent",
        name="Test Agent",
        description="An agent for testing",
        version="1.0",
        capabilities=["question_generation", "answer_evaluation"],
        llm=AgentLLMConfig(
            routing_strategy=LLMRoutingStrategy.HYBRID,
            local_model="ollama/test-model",
            cloud_model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
        ),
        domain=AgentDomainConfig(
            supported_subjects=["mathematics", "science"],
            max_retries=3,
        ),
        default_persona="tutor",
    )


@pytest.fixture
def sample_persona() -> Persona:
    """Create a sample persona."""
    return Persona(
        id="test_persona",
        name="Test Persona",
        description="A persona for testing",
        identity=PersonaIdentity(
            role="Test Tutor",
            character="Helpful and patient",
        ),
        voice=PersonaVoice(
            tone=Tone.WARM,
            formality=Formality.INFORMAL,
        ),
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=LLMResponse(
            content='{"content": "Test question?", "correct_answer": "42"}',
            model="ollama/test-model",
            tokens_input=100,
            tokens_output=50,
        )
    )
    return client


@pytest.fixture
def mock_capability() -> MagicMock:
    """Create a mock capability."""
    capability = MagicMock(spec=Capability)
    capability.name = "question_generation"
    capability.description = "Generates questions"
    capability.validate_params = MagicMock()
    capability.build_prompt = MagicMock(
        return_value=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "Generate a question about math."},
        ]
    )
    capability.parse_response = MagicMock(
        return_value=CapabilityResult(
            success=True,
            capability_name="question_generation",
        )
    )
    return capability


@pytest.fixture
def mock_registry(mock_capability) -> MagicMock:
    """Create a mock capability registry."""
    registry = MagicMock(spec=CapabilityRegistry)
    registry.has = MagicMock(return_value=True)
    registry.get = MagicMock(return_value=mock_capability)
    registry.list_names = MagicMock(return_value=["question_generation", "answer_evaluation"])
    return registry


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_create_config_with_defaults(self):
        """Test creating config with minimal parameters."""
        config = AgentConfig(
            id="test",
            name="Test",
            capabilities=["question_generation"],
        )

        assert config.id == "test"
        assert config.name == "Test"
        assert config.version == "1.0"
        assert config.default_persona == "tutor"
        assert config.llm.routing_strategy == LLMRoutingStrategy.HYBRID

    def test_create_config_full(self, sample_agent_config):
        """Test creating config with all parameters."""
        config = sample_agent_config

        assert config.id == "test_agent"
        assert config.name == "Test Agent"
        assert len(config.capabilities) == 2
        assert config.llm.temperature == 0.7
        assert config.domain.max_retries == 3

    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
agent:
  id: yaml_agent
  name: YAML Agent
  description: Loaded from YAML
  capabilities:
    - question_generation
  llm:
    routing_strategy: hybrid
    local_model: ollama/test-model
  default_persona: tutor
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = AgentConfig.from_yaml(f.name)

            assert config.id == "yaml_agent"
            assert config.name == "YAML Agent"
            assert "question_generation" in config.capabilities

    def test_load_from_yaml_missing_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_yaml("/nonexistent/path.yaml")

    def test_load_from_yaml_invalid_structure(self):
        """Test loading invalid YAML raises error."""
        yaml_content = """
not_agent:
  id: test
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="missing 'agent' key"):
                AgentConfig.from_yaml(f.name)


class TestAgentLLMConfig:
    """Tests for AgentLLMConfig."""

    def test_default_values(self):
        """Test default LLM config values."""
        config = AgentLLMConfig()

        assert config.routing_strategy == LLMRoutingStrategy.HYBRID
        assert config.local_model == "ollama/qwen2.5:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout_seconds == 60

    def test_routing_strategies(self):
        """Test all routing strategies."""
        for strategy in LLMRoutingStrategy:
            config = AgentLLMConfig(routing_strategy=strategy)
            assert config.routing_strategy == strategy


# =============================================================================
# AgentExecutionContext Tests
# =============================================================================


class TestAgentExecutionContext:
    """Tests for AgentExecutionContext."""

    def test_create_minimal_context(self):
        """Test creating context with minimal parameters."""
        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
        )

        assert context.tenant_id == "tenant_123"
        assert context.student_id == "student_456"
        assert context.topic == ""
        assert context.params == {}
        assert context.memory is None

    def test_create_full_context(self):
        """Test creating context with all parameters."""
        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
            topic="fractions",
            intent="generate_question",
            params={"difficulty": 0.7},
            metadata={"session_id": "session_789"},
        )

        assert context.topic == "fractions"
        assert context.intent == "generate_question"
        assert context.params["difficulty"] == 0.7

    def test_to_capability_context(self):
        """Test converting to CapabilityContext."""
        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
            topic="fractions",
            intent="generate_question",
            params={"difficulty": 0.7},
        )

        cap_context = context.to_capability_context()

        assert isinstance(cap_context, CapabilityContext)
        assert cap_context.additional["topic"] == "fractions"
        assert cap_context.additional["difficulty"] == 0.7


# =============================================================================
# DynamicAgent Tests
# =============================================================================


class TestDynamicAgent:
    """Tests for DynamicAgent."""

    def test_create_agent(self, sample_agent_config, mock_llm_client, mock_registry):
        """Test creating a DynamicAgent."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        assert agent.id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.persona is None

    def test_create_agent_validates_capabilities(
        self, sample_agent_config, mock_llm_client
    ):
        """Test that agent creation validates capabilities."""
        registry = MagicMock(spec=CapabilityRegistry)
        registry.has = MagicMock(return_value=False)
        registry.list_names = MagicMock(return_value=["other_capability"])

        with pytest.raises(AgentError, match="Capabilities not found"):
            DynamicAgent(
                config=sample_agent_config,
                llm_client=mock_llm_client,
                capability_registry=registry,
            )

    def test_set_persona(
        self, sample_agent_config, mock_llm_client, mock_registry, sample_persona
    ):
        """Test setting a persona."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        agent.set_persona(sample_persona)

        assert agent.persona is sample_persona
        assert agent.persona.id == "test_persona"

    def test_clear_persona(
        self, sample_agent_config, mock_llm_client, mock_registry, sample_persona
    ):
        """Test clearing a persona."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        agent.set_persona(sample_persona)
        agent.clear_persona()

        assert agent.persona is None

    def test_has_capability(self, sample_agent_config, mock_llm_client, mock_registry):
        """Test checking for capabilities."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        assert agent.has_capability("question_generation") is True
        assert agent.has_capability("nonexistent") is False

    def test_list_capabilities(
        self, sample_agent_config, mock_llm_client, mock_registry
    ):
        """Test listing capabilities."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        caps = agent.list_capabilities()

        assert "question_generation" in caps
        assert "answer_evaluation" in caps

    @pytest.mark.asyncio
    async def test_execute_success(
        self, sample_agent_config, mock_llm_client, mock_registry
    ):
        """Test successful execution."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
            topic="fractions",
            intent="question_generation",
            params={"difficulty": 0.5},
        )

        response = await agent.execute(context)

        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert response.agent_id == "test_agent"
        assert response.capability_name == "question_generation"
        assert response.model_used == "ollama/test-model"

    @pytest.mark.asyncio
    async def test_execute_with_persona(
        self, sample_agent_config, mock_llm_client, mock_registry, sample_persona
    ):
        """Test execution with persona attached."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )
        agent.set_persona(sample_persona)

        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
            intent="question_generation",
        )

        response = await agent.execute(context)

        assert response.success is True
        assert response.metadata.get("persona_id") == "test_persona"

    @pytest.mark.asyncio
    async def test_execute_invalid_capability(
        self, sample_agent_config, mock_llm_client, mock_registry
    ):
        """Test execution with invalid capability."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        context = AgentExecutionContext(
            tenant_id="tenant_123",
            student_id="student_456",
            intent="nonexistent_capability",
        )

        with pytest.raises(AgentError, match="does not have capability"):
            await agent.execute(context)

    def test_repr(self, sample_agent_config, mock_llm_client, mock_registry):
        """Test string representation."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        repr_str = repr(agent)

        assert "test_agent" in repr_str
        assert "question_generation" in repr_str


class TestDynamicAgentPromptBuilding:
    """Tests for DynamicAgent prompt building."""

    def test_inject_persona_into_messages(
        self, sample_agent_config, mock_llm_client, mock_registry, sample_persona
    ):
        """Test persona is injected into system message."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )
        agent.set_persona(sample_persona)

        messages = [
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "Hello"},
        ]

        result = agent._inject_persona_into_messages(messages)

        # Persona should be prepended to system message
        assert result[0]["role"] == "system"
        assert sample_persona.identity.role in result[0]["content"]

    def test_inject_persona_creates_system_message(
        self, sample_agent_config, mock_llm_client, mock_registry, sample_persona
    ):
        """Test persona creates system message if none exists."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )
        agent.set_persona(sample_persona)

        messages = [{"role": "user", "content": "Hello"}]

        result = agent._inject_persona_into_messages(messages)

        # System message should be added
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_no_injection_without_persona(
        self, sample_agent_config, mock_llm_client, mock_registry
    ):
        """Test no injection when no persona is set."""
        agent = DynamicAgent(
            config=sample_agent_config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        messages = [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Hello"},
        ]

        result = agent._inject_persona_into_messages(messages)

        assert result == messages


class TestDynamicAgentModelSelection:
    """Tests for DynamicAgent model selection."""

    def test_select_model_privacy_first(self, mock_llm_client, mock_registry):
        """Test model selection with privacy_first strategy."""
        config = AgentConfig(
            id="test",
            name="Test",
            capabilities=["question_generation"],
            llm=AgentLLMConfig(
                routing_strategy=LLMRoutingStrategy.PRIVACY_FIRST,
                local_model="ollama/local-model",
                cloud_model="gpt-4",
            ),
        )

        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        model = agent._select_model()

        assert model == "ollama/local-model"

    def test_select_model_quality_first(self, mock_llm_client, mock_registry):
        """Test model selection with quality_first strategy."""
        config = AgentConfig(
            id="test",
            name="Test",
            capabilities=["question_generation"],
            llm=AgentLLMConfig(
                routing_strategy=LLMRoutingStrategy.QUALITY_FIRST,
                local_model="ollama/local-model",
                cloud_model="gpt-4",
            ),
        )

        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        model = agent._select_model()

        assert model == "gpt-4"

    def test_select_model_hybrid(self, mock_llm_client, mock_registry):
        """Test model selection with hybrid strategy."""
        config = AgentConfig(
            id="test",
            name="Test",
            capabilities=["question_generation"],
            llm=AgentLLMConfig(
                routing_strategy=LLMRoutingStrategy.HYBRID,
                local_model="ollama/local-model",
                cloud_model="gpt-4",
            ),
        )

        agent = DynamicAgent(
            config=config,
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
        )

        model = agent._select_model()

        # Hybrid defaults to local
        assert model == "ollama/local-model"


# =============================================================================
# AgentFactory Tests
# =============================================================================


class TestAgentFactory:
    """Tests for AgentFactory."""

    @pytest.fixture
    def temp_agents_dir(self):
        """Create a temporary agents directory with config files."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)

            # Create tutor config
            tutor_config = """
agent:
  id: tutor
  name: Tutor Agent
  capabilities:
    - question_generation
  default_persona: tutor
"""
            (agents_dir / "tutor.yaml").write_text(tutor_config)

            # Create assessor config
            assessor_config = """
agent:
  id: assessor
  name: Assessor Agent
  capabilities:
    - question_generation
    - answer_evaluation
  default_persona: tutor
"""
            (agents_dir / "assessor.yaml").write_text(assessor_config)

            yield agents_dir

    @pytest.fixture
    def mock_persona_manager(self, sample_persona):
        """Create a mock persona manager."""
        manager = MagicMock()
        manager.get_persona = MagicMock(return_value=sample_persona)
        return manager

    def test_create_factory(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test creating an AgentFactory."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        assert "tutor" in factory.list_available()
        assert "assessor" in factory.list_available()

    def test_get_agent(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test getting an agent."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        agent = factory.get("tutor")

        assert isinstance(agent, DynamicAgent)
        assert agent.id == "tutor"

    def test_get_agent_cached(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test that agents are cached."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        agent1 = factory.get("tutor")
        agent2 = factory.get("tutor")

        assert agent1 is agent2

    def test_create_agent_not_cached(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test that create returns new instance."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        agent1 = factory.create("tutor")
        agent2 = factory.create("tutor")

        assert agent1 is not agent2

    def test_get_nonexistent_agent(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test getting non-existent agent raises error."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        with pytest.raises(AgentNotFoundError) as exc_info:
            factory.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "tutor" in str(exc_info.value)  # Available agents listed

    def test_clear_cache(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test clearing agent cache."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        agent1 = factory.get("tutor")
        factory.clear_cache()
        agent2 = factory.get("tutor")

        assert agent1 is not agent2

    def test_get_config(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test getting agent configuration."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        config = factory.get_config("tutor")

        assert config.id == "tutor"
        assert "question_generation" in config.capabilities

    def test_set_persona_for_agent(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test setting persona for an agent via factory."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        agent = factory.set_persona_for_agent("tutor", "test_persona")

        assert agent.persona is not None
        mock_persona_manager.get_persona.assert_called_with("test_persona")

    def test_repr(
        self, temp_agents_dir, mock_llm_client, mock_registry, mock_persona_manager
    ):
        """Test string representation."""
        factory = AgentFactory(
            llm_client=mock_llm_client,
            capability_registry=mock_registry,
            persona_manager=mock_persona_manager,
            agents_dir=temp_agents_dir,
        )

        repr_str = repr(factory)

        assert "tutor" in repr_str
        assert "assessor" in repr_str


# =============================================================================
# AgentResponse Tests
# =============================================================================


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_create_response(self):
        """Test creating a response."""
        response = AgentResponse(
            success=True,
            agent_id="test_agent",
            capability_name="question_generation",
            result={"question": "What is 2+2?"},
        )

        assert response.success is True
        assert response.agent_id == "test_agent"
        assert response.result["question"] == "What is 2+2?"
        assert response.generated_at is not None

    def test_response_with_token_usage(self):
        """Test response with token usage."""
        response = AgentResponse(
            success=True,
            agent_id="test_agent",
            capability_name="question_generation",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        )

        assert response.token_usage["prompt_tokens"] == 100
        assert response.token_usage["completion_tokens"] == 50


# =============================================================================
# AgentError Tests
# =============================================================================


class TestAgentError:
    """Tests for AgentError."""

    def test_create_error(self):
        """Test creating an error."""
        error = AgentError(
            message="Something went wrong",
            agent_id="test_agent",
            capability_name="question_generation",
        )

        assert str(error) == "Something went wrong"
        assert error.agent_id == "test_agent"
        assert error.capability_name == "question_generation"

    def test_error_with_original(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        error = AgentError(
            message="Wrapped error",
            original_error=original,
        )

        assert error.original_error is original
