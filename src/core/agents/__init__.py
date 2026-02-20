# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agent system for AI-driven educational interactions.

This package contains the agent layer which orchestrates:
- Capabilities: What the agent can do (question generation, answer evaluation, etc.)
- Persona integration: How the agent communicates (tone, style)
- Theory integration: Why the agent makes certain decisions (educational theories)

The agent layer is responsible for all LLM interactions, combining context from
memory layers, RAG retrieval, and educational theories to generate personalized
responses.

Architecture:
    API/Workflow -> Agent -> Capabilities -> LLM
                      |
            Memory + RAG + Theory + Persona

Components:
    DynamicAgent: Config-driven agent that loads capabilities from YAML.
    AgentFactory: Creates and caches DynamicAgent instances.
    AgentConfig: Configuration model for agents.
    AgentExecutionContext: Runtime context for agent execution.
    AgentResponse: Structured response from agent execution.

Usage:
    # Get the factory singleton
    from src.core.agents import get_agent_factory

    factory = get_agent_factory()

    # Get or create an agent
    agent = factory.get("tutor")

    # Execute a capability
    context = AgentExecutionContext(
        tenant_id="tenant_123",
        student_id="student_456",
        intent="question_generation",
        params={"topic": "Fractions"},
    )
    response = await agent.execute(context)
"""

from src.core.agents.capabilities import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
    CapabilityRegistry,
)
from src.core.agents.context import (
    AgentConfig,
    AgentDomainConfig,
    AgentError,
    AgentExecutionContext,
    AgentLLMConfig,
    AgentResponse,
    LLMRoutingStrategy,
    # New: System Prompt Configuration
    SystemPromptConfig,
    SystemPromptContextSection,
    SystemPromptExample,
    SystemPromptRule,
    # New: Tool Configuration
    ToolDefinitionConfig,
    ToolsConfig,
)
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.agents.factory import (
    AgentFactory,
    AgentNotFoundError,
    get_agent_factory,
    reset_agent_factory,
)
from src.core.agents.prompt_builder import SystemPromptBuilder

__all__ = [
    # Capabilities
    "Capability",
    "CapabilityContext",
    "CapabilityError",
    "CapabilityResult",
    "CapabilityRegistry",
    # Agent
    "DynamicAgent",
    "AgentConfig",
    "AgentDomainConfig",
    "AgentError",
    "AgentExecutionContext",
    "AgentLLMConfig",
    "AgentResponse",
    "LLMRoutingStrategy",
    # System Prompt Configuration
    "SystemPromptConfig",
    "SystemPromptContextSection",
    "SystemPromptExample",
    "SystemPromptRule",
    # Tool Configuration
    "ToolDefinitionConfig",
    "ToolsConfig",
    # Factory
    "AgentFactory",
    "AgentNotFoundError",
    "get_agent_factory",
    "reset_agent_factory",
    # Prompt Builder
    "SystemPromptBuilder",
]
