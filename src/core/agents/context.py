# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agent configuration and execution context models.

This module provides:
- AgentLLMConfig: LLM routing configuration for an agent
- AgentConfig: Agent configuration loaded from YAML
- AgentExecutionContext: Runtime context for agent execution
- AgentResponse: Structured response from agent execution

Usage:
    # Load agent config
    config = AgentConfig.from_yaml("config/agents/tutor.yaml")

    # Get LLM params for the agent's provider
    from src.core.config import get_llm_params_for_provider
    params = get_llm_params_for_provider(config.llm.provider)

    # Create execution context
    context = AgentExecutionContext(
        tenant_id="tenant_abc",
        student_id="student_123",
        topic="fractions",
    )

    # Execute agent
    response = await agent.execute("explain_concept", context)
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import CapabilityContext, CapabilityResult
from src.core.educational.orchestrator import CombinedRecommendation
from src.core.memory.rag.retriever import RetrievalResult
from src.core.personas.models import Persona
from src.models.memory import FullMemoryContext


class LLMRoutingStrategy(str, Enum):
    """LLM routing strategy for an agent."""

    PRIVACY_FIRST = "privacy_first"  # Always use local/owned models
    HYBRID = "hybrid"  # Local first, cloud fallback
    QUALITY_FIRST = "quality_first"  # Prefer cloud models for quality
    LOCAL_ONLY = "local_only"  # Only local instance


class AgentLLMConfig(BaseModel):
    """LLM configuration for an agent.

    Uses provider codes from config/llm/providers.yaml.

    Attributes:
        provider: Provider code (e.g., 'vastai', 'local', 'openai').
        model: Optional model override. Uses provider's default if not specified.
        routing_strategy: How to route LLM requests.
        temperature: Default temperature for generation.
        max_tokens: Maximum tokens for generation.
        timeout_seconds: Request timeout.

    Example:
        llm:
          provider: vastai
          model: command-r:35b  # optional
          routing_strategy: hybrid
          temperature: 0.7
    """

    provider: str = Field(
        default="vastai",
        description="Provider code from config/llm/providers.yaml",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override. Uses provider's default if not specified.",
    )
    routing_strategy: LLMRoutingStrategy = Field(
        default=LLMRoutingStrategy.HYBRID,
        description="LLM routing strategy",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens for generation",
    )
    timeout_seconds: int = Field(
        default=180,
        ge=1,
        description="Request timeout in seconds",
    )


class AgentDomainConfig(BaseModel):
    """Domain-specific configuration for an agent.

    Attributes:
        supported_subjects: List of subjects the agent can handle.
        max_retries: Maximum retry attempts for failed operations.
        settings: Additional domain-specific settings.
    """

    supported_subjects: list[str] = Field(
        default_factory=list,
        description="Subjects this agent can handle",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional domain settings",
    )


# =============================================================================
# System Prompt Configuration Models
# =============================================================================


class SystemPromptRule(BaseModel):
    """A single rule in the system prompt.

    Rules define critical instructions that the agent must follow.
    They are formatted as numbered sections in the system prompt.

    Attributes:
        id: Unique rule identifier for reference.
        title: Rule title displayed as section header.
        content: Rule content with detailed instructions.

    Example YAML:
        rules:
          - id: never_teach
            title: "NEVER Teach"
            content: |
              If a student asks an academic question:
              - DO NOT explain or teach
              - Use handoff_to_tutor IMMEDIATELY
    """

    id: str = Field(
        description="Unique rule identifier",
    )
    title: str = Field(
        description="Rule title for section header",
    )
    content: str = Field(
        description="Rule content with instructions",
    )


class SystemPromptExample(BaseModel):
    """An example conversation for few-shot learning.

    Examples help the LLM understand expected behavior through
    demonstration. They are included in the system prompt to
    guide response patterns.

    Attributes:
        name: Example identifier for reference.
        title: Example title displayed as section header.
        conversation: Multi-turn conversation example.

    Example YAML:
        examples:
          - name: practice_flow
            title: "Practice Flow"
            conversation: |
              Student: "I want to practice"
              You: [Call get_subjects] "Great! Which subject?"
    """

    name: str = Field(
        description="Example identifier",
    )
    title: str = Field(
        description="Example title for display",
    )
    conversation: str = Field(
        description="Example conversation text",
    )


class SystemPromptContextSection(BaseModel):
    """A dynamic context section that can be conditionally included.

    Context sections are populated at runtime with actual values
    and can be conditionally included based on data availability.

    Attributes:
        id: Section identifier for reference.
        title: Section title displayed as header.
        template: Template string with {variable} placeholders.
        condition: Optional condition for inclusion (e.g., 'has_alerts').

    Example YAML:
        context_sections:
          - id: student_context
            title: "STUDENT CONTEXT"
            template: |
              - Grade Level: {grade_level}
              - Language: {language}
          - id: pending_alerts
            title: "PENDING ALERTS"
            template: "{alerts}"
            condition: has_alerts
    """

    id: str = Field(
        description="Section identifier",
    )
    title: str = Field(
        description="Section title for header",
    )
    template: str = Field(
        description="Template with {variable} placeholders",
    )
    condition: str | None = Field(
        default=None,
        description="Condition for inclusion, e.g., 'has_alerts'",
    )


class SystemPromptConfig(BaseModel):
    """System prompt configuration from YAML.

    Defines the complete structure of a system prompt that can be
    built from YAML configuration. The prompt is assembled by:
    1. Role definition (core identity)
    2. Rules (critical instructions)
    3. Examples (few-shot learning)
    4. Context sections (dynamic data)
    5. Intent-specific prompts (for capability-based execution)
    6. Tool instructions (if tools enabled)
    7. Response format
    8. Personality guidelines

    Attributes:
        role: Role definition - the core identity of the agent.
        rules: List of critical rules the agent must follow.
        examples: Few-shot learning examples.
        context_sections: Dynamic context sections.
        intent_prompts: Intent-specific prompt templates for capabilities.
        tool_instructions: Template for tool usage instructions.
        response_format: Expected response format instructions.
        personality: Personality and style guidelines.

    Example YAML:
        system_prompt:
          role: |
            You are {persona_name}, a friendly AI companion.
          rules:
            - id: never_teach
              title: "NEVER Teach"
              content: "..."
          examples:
            - name: practice_flow
              title: "Practice Flow"
              conversation: "..."
          intent_prompts:
            question_generation: |
              Generate a high-quality educational question...
            answer_evaluation: |
              Evaluate the student's answer...
    """

    role: str = Field(
        description="Role definition - core identity",
    )
    rules: list[SystemPromptRule] = Field(
        default_factory=list,
        description="Critical rules to follow",
    )
    examples: list[SystemPromptExample] = Field(
        default_factory=list,
        description="Few-shot learning examples",
    )
    context_sections: list[SystemPromptContextSection] = Field(
        default_factory=list,
        description="Dynamic context sections",
    )
    intent_prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Intent-specific prompt templates for capabilities",
    )
    tool_instructions: str | None = Field(
        default=None,
        description="Tool usage instructions template",
    )
    response_format: str | None = Field(
        default=None,
        description="Expected response format",
    )
    personality: str | None = Field(
        default=None,
        description="Personality and style guidelines",
    )


# =============================================================================
# Tool Configuration Models
# =============================================================================


class ToolDefinitionConfig(BaseModel):
    """Configuration for a single tool.

    Defines which tools are enabled and how they are organized.
    The tool must be registered in the ToolRegistry to be usable.

    Attributes:
        name: Tool name matching registered tool in ToolRegistry.
        enabled: Whether this tool is enabled for the agent.
        group: Tool group for organization (e.g., 'information_gathering').
        order: Order within group for display/instructions.

    Example YAML:
        definitions:
          - name: get_subjects
            enabled: true
            group: information_gathering
            order: 1
    """

    name: str = Field(
        description="Tool name matching registered tool",
    )
    enabled: bool = Field(
        default=True,
        description="Whether tool is enabled",
    )
    group: str | None = Field(
        default=None,
        description="Tool group for organization",
    )
    order: int = Field(
        default=0,
        description="Order within group",
    )


class ToolsConfig(BaseModel):
    """Tools configuration for an agent.

    Configures tool calling behavior for agents that support it.
    When enabled, the agent can use LLM tool calling to execute
    registered tools.

    Attributes:
        enabled: Whether tool calling is enabled for this agent.
        max_iterations: Maximum tool call rounds per turn (safety limit).
        definitions: List of tool definitions with enable/disable control.

    Example YAML:
        tools:
          enabled: true
          max_iterations: 3
          definitions:
            - name: get_subjects
              enabled: true
              group: information_gathering
              order: 1
    """

    enabled: bool = Field(
        default=True,
        description="Whether tool calling is enabled",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max tool call rounds per turn",
    )
    definitions: list[ToolDefinitionConfig] = Field(
        default_factory=list,
        description="Tool definitions",
    )
    tool_choice: str = Field(
        default="auto",
        description="How LLM should use tools: 'auto', 'none', 'required'",
    )

    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool names.

        Returns:
            List of tool names that are enabled.
        """
        return [t.name for t in self.definitions if t.enabled]

    def get_tools_by_group(self) -> dict[str, list[ToolDefinitionConfig]]:
        """Get tools organized by group.

        Returns:
            Dictionary mapping group names to tool definitions.
        """
        groups: dict[str, list[ToolDefinitionConfig]] = {}
        for tool in sorted(self.definitions, key=lambda t: t.order):
            if tool.enabled:
                group = tool.group or "default"
                if group not in groups:
                    groups[group] = []
                groups[group].append(tool)
        return groups


class AgentConfig(BaseModel):
    """Agent configuration loaded from YAML.

    This defines WHAT an agent can do (capabilities, tools) and HOW it should
    be configured (LLM settings, domain settings, system prompt). Personality
    and communication style come from Persona, which is referenced by default_persona.

    The configuration supports two execution modes:
    1. Capability-based: Uses capabilities for structured prompt/response (Tutor, Assessor)
    2. Tool-based: Uses tools for LLM tool calling (Companion)

    Both modes can be combined - an agent can have both capabilities and tools.

    Attributes:
        id: Unique agent identifier.
        name: Human-readable name.
        description: Agent description.
        version: Configuration version.
        capabilities: List of capability names the agent can use.
        llm: LLM configuration with provider code.
        domain: Domain-specific configuration.
        default_persona: Default persona ID to use.
        tools: Tool calling configuration (optional).
        system_prompt: System prompt configuration (optional).

    Example:
        # Capability-based agent (existing pattern)
        config = AgentConfig(
            id="assessor",
            name="Assessment Agent",
            capabilities=["question_generation", "answer_evaluation"],
            default_persona="tutor",
        )

        # Tool-based agent (new pattern)
        config = AgentConfig(
            id="companion",
            name="Learning Companion",
            capabilities=["companion_decision"],
            tools=ToolsConfig(enabled=True, definitions=[...]),
            system_prompt=SystemPromptConfig(role="You are..."),
            default_persona="companion",
        )
    """

    id: str = Field(
        description="Unique agent identifier",
    )
    name: str = Field(
        description="Human-readable agent name",
    )
    description: str = Field(
        default="",
        description="Agent description",
    )
    version: str = Field(
        default="1.0",
        description="Configuration version",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capability names",
    )
    llm: AgentLLMConfig = Field(
        default_factory=AgentLLMConfig,
        description="LLM configuration",
    )
    domain: AgentDomainConfig = Field(
        default_factory=AgentDomainConfig,
        description="Domain configuration",
    )
    default_persona: str = Field(
        default="tutor",
        description="Default persona ID",
    )
    tools: ToolsConfig | None = Field(
        default=None,
        description="Tool calling configuration (optional)",
    )
    system_prompt: SystemPromptConfig | None = Field(
        default=None,
        description="System prompt configuration (optional)",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """Load agent configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            AgentConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the YAML structure is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "agent" not in data:
            raise ValueError(f"Invalid agent config: missing 'agent' key in {path}")

        return cls.model_validate(data["agent"])

    def get_llm_params(self) -> dict[str, Any]:
        """Get LiteLLM parameters for this agent's LLM configuration.

        Returns:
            Dictionary with model, api_base, api_key for acompletion().

        Example:
            >>> config = AgentConfig.from_yaml("config/agents/tutor.yaml")
            >>> params = config.get_llm_params()
            >>> response = await acompletion(**params, messages=[...])
        """
        from src.core.config import get_llm_params_for_provider

        return get_llm_params_for_provider(self.llm.provider, self.llm.model)


class AgentExecutionContext(BaseModel):
    """Runtime context for agent execution.

    This context is assembled by the AgentFactory or Workflow and passed
    to the agent for execution. It contains:
    - Request info: tenant_id, student_id, topic
    - Memory context: 4-layer memory data
    - Theory recommendations: pedagogical guidance
    - RAG results: retrieved documents
    - Persona: the personality to use

    Attributes:
        tenant_id: Tenant identifier.
        student_id: Student identifier.
        topic: Current topic/subject.
        intent: What the agent should do.
        params: Additional parameters for the capability.
        memory: Full memory context.
        theory: Theory recommendations.
        rag_results: Retrieved documents.
        persona: Active persona.
        conversation_history: Previous messages if any.
        metadata: Additional metadata.

    Example:
        context = AgentExecutionContext(
            tenant_id="tenant_abc",
            student_id="student_123",
            topic="fractions",
            intent="generate_question",
            params={"difficulty": 0.7},
        )
    """

    tenant_id: str = Field(
        description="Tenant identifier",
    )
    student_id: str = Field(
        description="Student identifier",
    )
    topic: str = Field(
        default="",
        description="Current topic",
    )
    intent: str = Field(
        default="",
        description="What the agent should do (capability name)",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional capability parameters",
    )
    memory: FullMemoryContext | None = Field(
        default=None,
        description="Full memory context",
    )
    theory: CombinedRecommendation | None = Field(
        default=None,
        description="Theory recommendations",
    )
    rag_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="RAG retrieval results",
    )
    persona: Persona | None = Field(
        default=None,
        description="Active persona",
    )
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation messages",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"arbitrary_types_allowed": True}

    def to_capability_context(self) -> CapabilityContext:
        """Convert to CapabilityContext for capability execution.

        Returns:
            CapabilityContext with memory, theory, RAG, and persona.
        """
        return CapabilityContext(
            memory=self.memory,
            theory=self.theory,
            rag_results=self.rag_results,
            persona=self.persona,
            additional={
                "topic": self.topic,
                "tenant_id": self.tenant_id,
                "student_id": self.student_id,
                **self.params,
            },
        )


class AgentResponse(BaseModel):
    """Response from agent execution.

    Contains the structured result from capability execution along with
    metadata about the execution.

    Attributes:
        success: Whether the execution was successful.
        agent_id: ID of the agent that executed.
        capability_name: Name of the capability that was executed.
        result: Capability-specific result data.
        raw_response: Raw LLM response text.
        generated_at: Timestamp of generation.
        model_used: LLM model that was used.
        token_usage: Token usage information.
        metadata: Additional execution metadata.

    Example:
        response = AgentResponse(
            success=True,
            agent_id="assessor",
            capability_name="question_generation",
            result={"question": "What is 2+2?", "answer": 4},
        )
    """

    success: bool = Field(
        default=True,
        description="Whether execution succeeded",
    )
    agent_id: str = Field(
        description="ID of the executing agent",
    )
    capability_name: str = Field(
        description="Name of the executed capability",
    )
    result: CapabilityResult | dict[str, Any] | None = Field(
        default=None,
        description="Capability result",
    )
    raw_response: str | None = Field(
        default=None,
        description="Raw LLM response",
        repr=False,
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Generation timestamp",
    )
    model_used: str | None = Field(
        default=None,
        description="LLM model used",
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage: prompt_tokens, completion_tokens",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"arbitrary_types_allowed": True}


class AgentError(Exception):
    """Exception raised for agent-related errors.

    Attributes:
        message: Error description.
        agent_id: Agent that raised the error.
        capability_name: Capability being executed.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        capability_name: str | None = None,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.agent_id = agent_id
        self.capability_name = capability_name
        self.original_error = original_error
        super().__init__(self.message)
