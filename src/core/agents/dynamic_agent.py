# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""DynamicAgent: Config-driven agent that loads capabilities from configuration.

This module implements the core agent abstraction that:
- Loads capabilities from YAML configuration
- Composes with personas at runtime
- Orchestrates LLM calls through capabilities
- Supports tool calling for agents with tools configured
- Returns structured results

The DynamicAgent pattern eliminates the need for hardcoded agent classes.
New agents can be created by adding YAML configuration without code changes.

Usage:
    # Create agent from config (capability-based)
    config = AgentConfig.from_yaml("config/agents/assessor.yaml")
    agent = DynamicAgent(config, llm_client, capability_registry)
    agent.set_persona(persona)
    response = await agent.execute(context)

    # Create agent with tool support
    config = AgentConfig.from_yaml("config/agents/companion.yaml")
    agent = DynamicAgent(config, llm_client, capability_registry, tool_registry)
    agent.set_persona(persona)
    response = await agent.execute_with_tools(context, runtime_context)
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.core.agents.capabilities.registry import (
    CapabilityNotFoundError,
    CapabilityRegistry,
)
from src.core.agents.context import (
    AgentConfig,
    AgentError,
    AgentExecutionContext,
    AgentResponse,
    LLMRoutingStrategy,
)
from src.core.intelligence.llm.client import LLMClient, LLMError, LLMResponse
from src.core.personas.models import Persona

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.agents.prompt_builder import SystemPromptBuilder
    from src.core.emotional.context import EmotionalContext
    from src.core.tools import ToolContext, ToolRegistry, ToolResult
    from src.models.memory import FullMemoryContext

logger = logging.getLogger(__name__)


class DynamicAgent:
    """Config-driven agent that executes capabilities and/or tools.

    The DynamicAgent is a generic agent that loads its behavior from YAML
    configuration. It supports two execution modes:

    1. Capability-based (execute): Uses capabilities for structured prompts
       and response parsing. Used by Tutor, Assessor, etc.

    2. Tool-based (execute_with_tools): Uses tool calling for external actions.
       Used by Companion for navigation, data fetching, etc.

    Both modes can be combined - an agent can have both capabilities and tools.

    Responsibilities:
    - Load capabilities from registry based on config
    - Build system prompts from config + persona
    - Orchestrate LLM calls (with or without tools)
    - Parse responses using capability parsers
    - Execute tool calling loops when enabled

    The agent does NOT:
    - Define specific behaviors (that's in capabilities/tools)
    - Define personality (that's in personas)
    - Make educational decisions (that's in theories)

    Attributes:
        config: Agent configuration.
        persona: Currently attached persona.
        has_tools: Whether this agent has tool support.

    Example (capability-based):
        >>> config = AgentConfig.from_yaml("config/agents/tutor.yaml")
        >>> agent = DynamicAgent(config, llm_client, capability_registry)
        >>> agent.set_persona(persona)
        >>> response = await agent.execute(context)

    Example (tool-based):
        >>> config = AgentConfig.from_yaml("config/agents/companion.yaml")
        >>> agent = DynamicAgent(config, llm_client, capability_registry, tool_registry)
        >>> agent.set_persona(persona)
        >>> response = await agent.execute_with_tools(context, runtime_context)
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        capability_registry: CapabilityRegistry,
        tool_registry: "ToolRegistry | None" = None,
    ):
        """Initialize DynamicAgent.

        Args:
            config: Agent configuration loaded from YAML.
            llm_client: LLM client for text generation.
            capability_registry: Registry of available capabilities.
            tool_registry: Optional registry of tools for tool calling.

        Raises:
            AgentError: If required capabilities are not in registry.
        """
        self._config = config
        self._llm_client = llm_client
        self._capability_registry = capability_registry
        self._tool_registry = tool_registry
        self._persona: Persona | None = None
        self._prompt_builder: "SystemPromptBuilder | None" = None

        # Validate that all required capabilities are available
        self._validate_capabilities()

        logger.info(
            "DynamicAgent initialized: id=%s, capabilities=%s, has_tools=%s",
            config.id,
            config.capabilities,
            tool_registry is not None,
        )

    def _validate_capabilities(self) -> None:
        """Validate that all configured capabilities are in the registry.

        Raises:
            AgentError: If any capability is not found.
        """
        missing = []
        for cap_name in self._config.capabilities:
            if not self._capability_registry.has(cap_name):
                missing.append(cap_name)

        if missing:
            available = self._capability_registry.list_names()
            raise AgentError(
                message=f"Capabilities not found: {', '.join(missing)}. "
                f"Available: {', '.join(available)}",
                agent_id=self._config.id,
            )

    @property
    def id(self) -> str:
        """Get the agent ID.

        Returns:
            Agent identifier.
        """
        return self._config.id

    @property
    def name(self) -> str:
        """Get the agent name.

        Returns:
            Human-readable agent name.
        """
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration.

        Returns:
            Agent configuration.
        """
        return self._config

    @property
    def persona(self) -> Persona | None:
        """Get the currently attached persona.

        Returns:
            Active persona or None.
        """
        return self._persona

    @property
    def has_tools(self) -> bool:
        """Check if agent has tool support.

        Returns:
            True if tool registry is configured.
        """
        return self._tool_registry is not None

    def set_persona(self, persona: Persona) -> None:
        """Attach a persona to this agent.

        The persona defines HOW the agent communicates (tone, style,
        language). It will be incorporated into the system prompt.

        If the agent has system_prompt config, this also initializes
        the SystemPromptBuilder with the persona.

        Args:
            persona: Persona to attach.

        Example:
            >>> agent.set_persona(coach_persona)
        """
        self._persona = persona

        # Initialize prompt builder if system_prompt config exists
        if self._config.system_prompt:
            from src.core.agents.prompt_builder import SystemPromptBuilder

            self._prompt_builder = SystemPromptBuilder(
                config=self._config.system_prompt,
                persona=persona,
                tools_config=self._config.tools,
            )
            logger.debug(
                "SystemPromptBuilder initialized: agent=%s",
                self._config.id,
            )

        logger.debug(
            "Persona attached: agent=%s, persona=%s",
            self._config.id,
            persona.id,
        )

    def clear_persona(self) -> None:
        """Remove the currently attached persona."""
        self._persona = None
        self._prompt_builder = None
        logger.debug("Persona cleared: agent=%s", self._config.id)

    def has_capability(self, name: str) -> bool:
        """Check if the agent has a specific capability.

        Args:
            name: Capability name.

        Returns:
            True if agent has this capability.
        """
        return name in self._config.capabilities

    def list_capabilities(self) -> list[str]:
        """List all capabilities this agent has.

        Returns:
            List of capability names.
        """
        return list(self._config.capabilities)

    async def execute(
        self,
        context: AgentExecutionContext,
        runtime_context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Execute a capability with the given context.

        This is the main entry point for agent execution. It supports two modes:

        1. YAML-driven mode (if system_prompt config exists):
           - System prompt built from YAML via SystemPromptBuilder
           - Capability provides only user prompt via build_user_prompt()

        2. Legacy mode (if no system_prompt config):
           - Capability builds both system and user prompts
           - Persona injected into system message

        Args:
            context: Execution context with intent, params, memory, etc.
            runtime_context: Optional runtime context for YAML prompt building.
                Required if agent has system_prompt config.
                Expected keys: grade_level, age_range, subject_name, topic_name,
                curriculum, language, learning_style, mastery_level, etc.

        Returns:
            AgentResponse with structured result.

        Raises:
            AgentError: If execution fails.

        Example (YAML-driven):
            response = await agent.execute(
                context=context,
                runtime_context={
                    "grade_level": 5,
                    "age_range": "10-11",
                    "subject_name": "Mathematics",
                    "topic_name": "Fractions",
                    "language": "en",
                },
            )
        """
        intent = context.intent
        start_time = datetime.now()

        logger.info(
            "Agent execution started: agent=%s, intent=%s, student=%s, yaml_driven=%s",
            self._config.id,
            intent,
            context.student_id,
            self._prompt_builder is not None,
        )

        # Validate capability
        if not self.has_capability(intent):
            raise AgentError(
                message=f"Agent '{self._config.id}' does not have capability '{intent}'. "
                f"Available: {', '.join(self._config.capabilities)}",
                agent_id=self._config.id,
                capability_name=intent,
            )

        try:
            # Get capability
            capability = self._capability_registry.get(intent)

            # Validate params
            capability.validate_params(context.params)

            # Build capability context
            cap_context = context.to_capability_context()

            # Determine prompt building mode
            if self._prompt_builder and runtime_context is not None:
                # YAML-driven mode: system prompt from YAML, user prompt from capability
                system_prompt = self._prompt_builder.build(runtime_context, intent=intent)
                user_prompt = capability.build_user_prompt(context.params, cap_context)

                logger.debug(
                    "Using YAML-driven prompt building: agent=%s, intent=%s",
                    self._config.id,
                    intent,
                )
            else:
                # Legacy mode: capability builds both prompts
                messages = capability.build_prompt(context.params, cap_context)

                # Add persona to system message if attached
                messages = self._inject_persona_into_messages(messages)

                # Extract system and user messages
                system_prompt = self._extract_system_prompt(messages)
                user_prompt = self._extract_user_prompt(messages)

                logger.debug(
                    "Using legacy prompt building: agent=%s, intent=%s",
                    self._config.id,
                    intent,
                )

            # Select model based on routing strategy
            model = self._select_model()

            # Call LLM
            llm_response = await self._call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
            )

            # Parse response
            result = capability.parse_response(llm_response.content)

            # Store raw response in result if it's a CapabilityResult
            if isinstance(result, CapabilityResult):
                result.raw_response = llm_response.content

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                "Agent execution completed: agent=%s, intent=%s, "
                "model=%s, tokens=%d, duration_ms=%.0f",
                self._config.id,
                intent,
                llm_response.model,
                llm_response.total_tokens,
                duration_ms,
            )

            return AgentResponse(
                success=True,
                agent_id=self._config.id,
                capability_name=intent,
                result=result,
                raw_response=llm_response.content,
                model_used=llm_response.model,
                token_usage={
                    "prompt_tokens": llm_response.tokens_input,
                    "completion_tokens": llm_response.tokens_output,
                },
                metadata={
                    "duration_ms": duration_ms,
                    "persona_id": self._persona.id if self._persona else None,
                    "routing_strategy": self._config.llm.routing_strategy.value,
                    "yaml_driven": self._prompt_builder is not None and runtime_context is not None,
                },
            )

        except CapabilityNotFoundError as e:
            logger.error("Capability not found: %s", e.capability_name)
            raise AgentError(
                message=str(e),
                agent_id=self._config.id,
                capability_name=intent,
                original_error=e,
            ) from e

        except CapabilityError as e:
            logger.error(
                "Capability error: agent=%s, intent=%s, error=%s",
                self._config.id,
                intent,
                str(e),
            )
            raise AgentError(
                message=f"Capability error: {e.message}",
                agent_id=self._config.id,
                capability_name=intent,
                original_error=e,
            ) from e

        except LLMError as e:
            logger.error(
                "LLM error: agent=%s, intent=%s, model=%s, error=%s",
                self._config.id,
                intent,
                e.model,
                str(e),
            )
            raise AgentError(
                message=f"LLM error: {e.message}",
                agent_id=self._config.id,
                capability_name=intent,
                original_error=e,
            ) from e

        except Exception as e:
            logger.exception(
                "Unexpected error during agent execution: agent=%s, intent=%s",
                self._config.id,
                intent,
            )
            raise AgentError(
                message=f"Execution failed: {str(e)}",
                agent_id=self._config.id,
                capability_name=intent,
                original_error=e,
            ) from e

    def _inject_persona_into_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Inject persona into the system message.

        If a persona is attached, add its system prompt segment to the
        first system message, or create one if none exists.

        Args:
            messages: Original messages from capability.

        Returns:
            Messages with persona injected.
        """
        if not self._persona:
            return messages

        persona_prompt = self._persona.get_system_prompt_segment()
        if not persona_prompt:
            return messages

        # Find or create system message
        result = []
        system_found = False

        for msg in messages:
            if msg["role"] == "system" and not system_found:
                # Prepend persona to existing system message
                combined_content = f"{persona_prompt}\n\n{msg['content']}"
                result.append({"role": "system", "content": combined_content})
                system_found = True
            else:
                result.append(msg)

        # If no system message existed, add one at the beginning
        if not system_found:
            result.insert(0, {"role": "system", "content": persona_prompt})

        return result

    def _select_model(self) -> str:
        """Select the LLM model based on agent configuration.

        Uses the provider code from agent config to get the model.
        The provider configuration determines api_base and api_key.

        Returns:
            Model identifier in LiteLLM format.
        """
        # Get LLM params from agent config (uses provider code)
        params = self._config.get_llm_params()
        return params["model"]

    def _extract_system_prompt(self, messages: list[dict[str, str]]) -> str | None:
        """Extract system prompt from messages.

        Args:
            messages: List of message dicts.

        Returns:
            Combined system prompt content or None.
        """
        system_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])

        return "\n\n".join(system_parts) if system_parts else None

    def _extract_user_prompt(self, messages: list[dict[str, str]]) -> str:
        """Extract user prompt from messages.

        Takes the last user message as the prompt.

        Args:
            messages: List of message dicts.

        Returns:
            User prompt content.

        Raises:
            AgentError: If no user message found.
        """
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]

        raise AgentError(
            message="No user message found in capability prompt",
            agent_id=self._config.id,
        )

    async def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None,
        model: str,
    ) -> LLMResponse:
        """Call the LLM with the given prompts.

        Args:
            user_prompt: User prompt content.
            system_prompt: System prompt content.
            model: Model to use.

        Returns:
            LLM response.
        """
        return await self._llm_client.complete(
            prompt=user_prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
        )

    async def execute_raw(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Execute a raw LLM call without using a capability.

        This is a lower-level method for direct LLM access when
        a specific capability is not needed.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Raw LLM response.
        """
        # Inject persona if attached
        if self._persona and system_prompt:
            persona_prompt = self._persona.get_system_prompt_segment()
            if persona_prompt:
                system_prompt = f"{persona_prompt}\n\n{system_prompt}"
        elif self._persona:
            system_prompt = self._persona.get_system_prompt_segment()

        model = self._select_model()

        return await self._llm_client.complete(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
        )

    async def execute_conversational(
        self,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Execute agent in conversational mode without tools or capabilities.

        This method is for pure conversation agents that don't need tools
        or capability-based execution. It uses the YAML-configured system
        prompt with runtime context interpolation.

        The key difference from execute():
        - No capability required - uses YAML system_prompt directly
        - Messages-based conversation (like execute_with_tools)
        - No tool calling loop

        The key difference from execute_with_tools():
        - No tools - pure LLM conversation
        - Simpler return structure

        Args:
            messages: Conversation history in OpenAI format.
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                The last message should be the current user input.
            runtime_context: Variables for YAML prompt interpolation.
                Keys depend on the agent's system_prompt template.
                Common keys: student_name, grade_level, language, topic_name, etc.
            intent: Optional intent name for prompt section selection.
                Used if YAML has intent-specific prompt sections.

        Returns:
            Dictionary with:
                - success: bool
                - content: str (LLM response text)
                - model_used: str
                - token_usage: dict (prompt_tokens, completion_tokens)
                - duration_ms: float

        Raises:
            AgentError: If prompt builder not configured or LLM call fails.

        Example:
            # Create agent with YAML config that has system_prompt
            agent = DynamicAgent(config, llm_client, capability_registry)
            agent.set_persona(persona)  # Required for prompt builder

            # Execute conversation
            response = await agent.execute_conversational(
                messages=[
                    {"role": "user", "content": "I don't understand this question"},
                    {"role": "assistant", "content": "Let me help you understand..."},
                    {"role": "user", "content": "Can you explain step by step?"},
                ],
                runtime_context={
                    "student_name": "Harry",
                    "grade_level": 5,
                    "language": "en",
                    "topic_name": "Fractions",
                    "tutoring_mode": "guided",
                },
            )

            if response["success"]:
                print(response["content"])  # LLM's response
        """
        start_time = datetime.now()

        # Validate requirements
        if not self._prompt_builder:
            raise AgentError(
                message="Prompt builder not initialized. "
                "Ensure agent has system_prompt in YAML and set_persona() was called.",
                agent_id=self._config.id,
            )

        logger.info(
            "Conversational execution started: agent=%s, messages=%d, intent=%s",
            self._config.id,
            len(messages),
            intent,
        )

        try:
            # Build system prompt from YAML config with runtime context
            system_prompt = self._prompt_builder.build(runtime_context, intent=intent)

            # Prepare full message list with system prompt
            full_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            full_messages.extend(messages)

            # Select model
            model = self._select_model()

            logger.debug(
                "Calling LLM: model=%s, messages=%d, system_prompt_len=%d",
                model,
                len(full_messages),
                len(system_prompt),
            )

            # Call LLM with messages (not prompt/system_prompt pattern)
            llm_response = await self._llm_client.complete_with_messages(
                messages=full_messages,
                model=model,
                temperature=self._config.llm.temperature,
                max_tokens=self._config.llm.max_tokens,
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                "Conversational execution completed: agent=%s, "
                "model=%s, tokens=%d/%d, duration_ms=%.0f",
                self._config.id,
                model,
                llm_response.tokens_input,
                llm_response.tokens_output,
                duration_ms,
            )

            return {
                "success": True,
                "content": llm_response.content,
                "model_used": model,
                "token_usage": {
                    "prompt_tokens": llm_response.tokens_input,
                    "completion_tokens": llm_response.tokens_output,
                },
                "duration_ms": duration_ms,
            }

        except LLMError as e:
            logger.error(
                "LLM error in conversational execution: agent=%s, error=%s",
                self._config.id,
                str(e),
            )
            raise AgentError(
                message=f"LLM error: {e.message}",
                agent_id=self._config.id,
                original_error=e,
            ) from e

        except Exception as e:
            logger.exception(
                "Unexpected error in conversational execution: agent=%s",
                self._config.id,
            )
            raise AgentError(
                message=f"Conversational execution failed: {str(e)}",
                agent_id=self._config.id,
                original_error=e,
            ) from e

    async def execute_with_tools(
        self,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
        db_session: "AsyncSession | None" = None,
        memory_context: "FullMemoryContext | None" = None,
        emotional_context: "EmotionalContext | None" = None,
        memory_manager: "Any | None" = None,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Execute agent with tool calling support.

        This method is used for agents that need to call external tools
        (like Companion). It builds the system prompt from YAML config,
        sends messages to LLM with tool definitions, and handles the
        tool calling loop.

        The tool calling loop:
        1. Send messages + tools to LLM
        2. If LLM returns tool_calls, execute them
        3. Add tool results to messages
        4. Repeat until LLM returns final response or max iterations

        Args:
            messages: Conversation messages in OpenAI format.
                Should include the current user message.
            runtime_context: Context variables for prompt interpolation.
                Expected keys: tenant_code, student_id, grade_level, language,
                persona_name, current_emotion, interests, alerts, etc.
            db_session: Database session for tool execution (required for
                tools that access the database).
            memory_context: Optional 4-layer memory context for tools.
            emotional_context: Optional emotional context for tools.
            memory_manager: Optional MemoryManager for tools that need to
                write to memory layers (interests, episodic events).
            tool_choice: How LLM should use tools ("auto", "none", "required").

        Returns:
            Dictionary with:
                - success: bool
                - content: str (final response text)
                - tool_calls_made: list[dict] (all tool calls executed)
                - tool_results: list[dict] (all tool results with data)
                - pending_actions: list[dict] (actions extracted from tool results)
                - pending_emotional_signals: list[dict] (emotions extracted from tool results)
                - iterations: int (number of LLM calls made)
                - model_used: str
                - token_usage: dict

        Raises:
            AgentError: If tool registry or prompt builder not configured,
                or if execution fails.

        Example:
            response = await agent.execute_with_tools(
                messages=[{"role": "user", "content": "I want to practice math"}],
                runtime_context={
                    "tenant_code": "test_school",
                    "student_id": "...",
                    "grade_level": 5,
                    "language": "en",
                },
                db_session=session,
            )
        """
        start_time = datetime.now()

        # Validate requirements
        if not self._tool_registry:
            raise AgentError(
                message="Tool registry not configured for this agent",
                agent_id=self._config.id,
            )

        if not self._prompt_builder:
            raise AgentError(
                message="Prompt builder not initialized. Call set_persona() first.",
                agent_id=self._config.id,
            )

        if not self._config.tools:
            raise AgentError(
                message="Tools config not found in agent configuration",
                agent_id=self._config.id,
            )

        logger.info(
            "Agent tool execution started: agent=%s, messages=%d",
            self._config.id,
            len(messages),
        )

        try:
            # Build system prompt from config + persona + context
            system_prompt = self._prompt_builder.build(runtime_context)

            # Prepare messages with system prompt
            full_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            full_messages.extend(messages)

            # Get tool definitions
            tool_definitions = self._tool_registry.get_definitions()

            # Get model
            model = self._select_model()

            # Track execution
            all_tool_calls: list[dict[str, Any]] = []
            all_tool_results: list[dict[str, Any]] = []
            pending_actions: list[dict[str, Any]] = []
            pending_emotional_signals: list[dict[str, Any]] = []

            # Track UI elements and tool data for frontend
            collected_ui_elements: list[dict[str, Any]] = []
            collected_tool_data: dict[str, Any] = {}
            conversation_state_update: dict[str, Any] | None = None

            # Track called tools to prevent duplicates
            called_tool_signatures: set[str] = set()

            iterations = 0
            max_iterations = self._config.tools.max_iterations
            total_tokens_in = 0
            total_tokens_out = 0

            # Build tool context once (used for all tool executions)
            tool_context = self._build_tool_context(
                runtime_context=runtime_context,
                db_session=db_session,
                memory_context=memory_context,
                emotional_context=emotional_context,
                memory_manager=memory_manager,
            )

            # Current tools to send (may be emptied after action tools)
            current_tools = tool_definitions

            # Tool calling loop
            while iterations < max_iterations:
                iterations += 1

                logger.debug(
                    "Tool iteration %d/%d: messages=%d, tools=%d",
                    iterations,
                    max_iterations,
                    len(full_messages),
                    len(current_tools),
                )

                # Debug: Log message types being sent
                import json
                for i, msg in enumerate(full_messages):
                    role = msg.get("role", "?")
                    has_tool_calls = "tool_calls" in msg
                    content_preview = str(msg.get("content", ""))[:50]
                    logger.debug(
                        "  Message[%d]: role=%s, has_tool_calls=%s, content=%s...",
                        i, role, has_tool_calls, content_preview
                    )

                # Call LLM with tools
                response = await self._llm_client.complete_with_tools(
                    messages=full_messages,
                    tools=current_tools,
                    tool_choice=tool_choice,
                    model=model,
                    temperature=self._config.llm.temperature,
                    max_tokens=self._config.llm.max_tokens,
                )

                total_tokens_in += response.tokens_input
                total_tokens_out += response.tokens_output

                # Check if we have tool calls
                if not response.has_tool_calls:
                    # No more tool calls - we have final response
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                    logger.info(
                        "Agent tool execution completed: agent=%s, "
                        "iterations=%d, tool_calls=%d, actions=%d, duration_ms=%.0f",
                        self._config.id,
                        iterations,
                        len(all_tool_calls),
                        len(pending_actions),
                        duration_ms,
                    )

                    return {
                        "success": True,
                        "content": response.content,
                        "tool_calls_made": all_tool_calls,
                        "tool_results": all_tool_results,
                        "pending_actions": pending_actions,
                        "pending_emotional_signals": pending_emotional_signals,
                        "ui_elements": collected_ui_elements,
                        "tool_data": collected_tool_data,
                        "conversation_state": conversation_state_update,
                        "iterations": iterations,
                        "model_used": model,
                        "token_usage": {
                            "prompt_tokens": total_tokens_in,
                            "completion_tokens": total_tokens_out,
                        },
                    }

                # Execute tool calls
                logger.debug(
                    "Executing %d tool calls",
                    len(response.tool_calls),
                )

                # Add assistant message with tool calls
                # NOTE: Ollama expects arguments as dict, but Gemini/OpenAI expect JSON string
                # Detect provider from model string to determine format
                is_ollama = model.startswith("ollama/") or model.startswith("ollama_chat/")
                is_gemini = model.startswith("gemini/")

                def format_arguments(args: Any) -> str | dict:
                    """Format arguments based on provider requirements."""
                    if is_ollama:
                        # Ollama expects dict
                        return args if isinstance(args, dict) else json.loads(args)
                    else:
                        # Gemini/OpenAI/others expect JSON string
                        if isinstance(args, dict):
                            return json.dumps(args)
                        return args  # Already a string

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": format_arguments(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                full_messages.append(assistant_msg)

                # Execute each tool and collect results
                # Track if any tool requests to stop chaining
                should_stop_chaining = False

                logger.info(
                    "LLM requested tools: %s",
                    [tc.name for tc in response.tool_calls]
                )
                for tool_call in response.tool_calls:
                    # Build normalized signature to detect duplicates
                    # Treat null, missing, and empty string as equivalent
                    if isinstance(tool_call.arguments, dict):
                        normalized_args = {
                            k: v for k, v in tool_call.arguments.items()
                            if v is not None and v != ""
                        }
                        args_str = json.dumps(normalized_args, sort_keys=True)
                    else:
                        args_str = str(tool_call.arguments)
                    tool_signature = f"{tool_call.name}:{args_str}"


                    # Skip duplicate tool calls
                    if tool_signature in called_tool_signatures:
                        logger.warning(
                            "Skipping duplicate tool call: %s (already called with same params)",
                            tool_call.name,
                        )
                        # Add a tool result indicating it was skipped
                        # NOTE: Gemini doesn't accept 'name' field in tool messages
                        tool_result_msg: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool {tool_call.name} was already called with these parameters. Use the previous result.",
                        }
                        if not is_gemini:
                            tool_result_msg["name"] = tool_call.name
                        full_messages.append(tool_result_msg)
                        continue

                    called_tool_signatures.add(tool_signature)

                    tool_record = {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    all_tool_calls.append(tool_record)

                    try:
                        # Get tool from registry
                        tool = self._tool_registry.get(tool_call.name)

                        # Execute tool with proper ToolContext
                        result = await tool.execute(tool_call.arguments, tool_context)

                        # Extract actions and emotions from result
                        self._extract_actions_and_emotions(
                            result=result,
                            pending_actions=pending_actions,
                            pending_emotional_signals=pending_emotional_signals,
                        )

                        # Extract UI elements and passthrough data for frontend
                        if result.ui_element is not None:
                            # Serialize UIElement to dict for state storage
                            if hasattr(result.ui_element, "model_dump"):
                                collected_ui_elements.append(
                                    result.ui_element.model_dump()
                                )
                            else:
                                collected_ui_elements.append(result.ui_element)

                        if result.passthrough_data:
                            collected_tool_data.update(result.passthrough_data)

                        if result.state_update:
                            # Last state update wins (most recent tool's state)
                            conversation_state_update = result.state_update

                        # Check if tool requests to stop chaining
                        if result.stop_chaining:
                            should_stop_chaining = True
                            logger.info(
                                "Tool %s requested to stop chaining",
                                tool_call.name,
                            )

                        # Store tool result for return
                        tool_result = {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "success": result.success,
                            "data": result.data,
                            "error": result.error,
                        }

                        # Get human-readable message for LLM
                        result_content = result.to_llm_message()

                        logger.debug(
                            "Tool executed: %s -> success=%s",
                            tool_call.name,
                            result.success,
                        )

                    except KeyError:
                        # Unknown tool - LLM hallucinated a tool name
                        # Force text response immediately to avoid wasting iterations
                        logger.warning(
                            "Unknown tool: %s - forcing text response",
                            tool_call.name,
                        )
                        tool_result = {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "success": False,
                            "data": {},
                            "error": f"Unknown tool: {tool_call.name}",
                            "_unknown_tool": True,  # Flag for fallback detection
                        }
                        result_content = (
                            "This tool does not exist. "
                            "Please respond directly to the user without calling any tool."
                        )
                        # Force stop chaining - go straight to text response
                        should_stop_chaining = True

                    except Exception as e:
                        logger.exception("Tool %s execution failed", tool_call.name)
                        tool_result = {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "success": False,
                            "data": {},
                            "error": str(e),
                        }
                        result_content = f"Error: {str(e)}"

                    all_tool_results.append(tool_result)

                    # Add tool result message for LLM
                    # NOTE: Gemini doesn't accept 'name' field in tool messages
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                    if not is_gemini:
                        tool_result_msg["name"] = tool_call.name
                    full_messages.append(tool_result_msg)

                # Tool chaining control based on tool result
                # Tools that complete a flow (navigate, handoff) set stop_chaining=True
                # Info tools (get_subjects, get_topics) allow chaining to continue
                if should_stop_chaining:
                    logger.info(
                        "Tool requested stop chaining, forcing text response: tool_calls=%d",
                        len(all_tool_calls),
                    )
                    current_tools = []
                    tool_choice = "none"
                else:
                    logger.info(
                        "Tool chaining enabled, allowing more tool calls: tool_calls=%d",
                        len(all_tool_calls),
                    )
                    current_tools = tool_definitions
                    tool_choice = "auto"

            # Max iterations reached - force final response
            logger.warning(
                "Max iterations reached: agent=%s, iterations=%d",
                self._config.id,
                max_iterations,
            )

            # Force final text response using simple completion (no tool overhead)
            # This is faster than complete_with_tools since no tool definitions are sent
            #
            # Convert messages to clean format (remove tool-related fields)
            # because complete_with_messages doesn't support tool messages
            clean_messages = []
            for msg in full_messages:
                role = msg.get("role")
                if role == "tool":
                    # Convert tool result to user message for context
                    tool_name = msg.get("name", "tool")
                    content = msg.get("content", "")
                    clean_messages.append({
                        "role": "user",
                        "content": f"[Tool result from {tool_name}]: {content}",
                    })
                elif role == "assistant":
                    # Keep assistant message but remove tool_calls
                    content = msg.get("content", "")
                    if content:  # Only add if there's content
                        clean_messages.append({
                            "role": "assistant",
                            "content": content,
                        })
                else:
                    # Keep system and user messages as-is
                    clean_messages.append({
                        "role": role,
                        "content": msg.get("content", ""),
                    })

            response = await self._llm_client.complete_with_messages(
                messages=clean_messages,
                model=model,
                temperature=self._config.llm.temperature,
                max_tokens=self._config.llm.max_tokens,
            )

            total_tokens_in += response.tokens_input
            total_tokens_out += response.tokens_output

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "success": True,
                "content": response.content,
                "tool_calls_made": all_tool_calls,
                "tool_results": all_tool_results,
                "pending_actions": pending_actions,
                "pending_emotional_signals": pending_emotional_signals,
                "ui_elements": collected_ui_elements,
                "tool_data": collected_tool_data,
                "conversation_state": conversation_state_update,
                "iterations": iterations + 1,
                "model_used": model,
                "token_usage": {
                    "prompt_tokens": total_tokens_in,
                    "completion_tokens": total_tokens_out,
                },
            }

        except LLMError as e:
            logger.error(
                "LLM error during tool execution: agent=%s, error=%s",
                self._config.id,
                str(e),
            )
            raise AgentError(
                message=f"LLM error: {e.message}",
                agent_id=self._config.id,
                original_error=e,
            ) from e

        except Exception as e:
            logger.exception(
                "Unexpected error during tool execution: agent=%s",
                self._config.id,
            )
            raise AgentError(
                message=f"Tool execution failed: {str(e)}",
                agent_id=self._config.id,
                original_error=e,
            ) from e

    def _build_tool_context(
        self,
        runtime_context: dict[str, Any],
        db_session: "AsyncSession | None",
        memory_context: "FullMemoryContext | None",
        emotional_context: "EmotionalContext | None",
        memory_manager: "Any | None" = None,
    ) -> "ToolContext":
        """Build ToolContext for tool execution.

        Creates a proper ToolContext dataclass instance from runtime context
        and optional dependencies. Supports both student and teacher contexts.

        Args:
            runtime_context: Runtime context with tenant_code, user_id/student_id, etc.
                For students: student_id, grade_level, etc.
                For teachers: user_id, user_type="teacher", etc.
            db_session: Database session for tool queries.
            memory_context: Optional memory context (student only).
            emotional_context: Optional emotional context (student only).
            memory_manager: Optional MemoryManager for tools that need to
                write to memory layers (stored in extra dict).

        Returns:
            ToolContext instance ready for tool execution.

        Raises:
            AgentError: If required context values are missing.
        """
        from src.core.tools import ToolContext

        # Extract required values
        tenant_code = runtime_context.get("tenant_code")

        # Support both user_id (new) and student_id (legacy) for backwards compatibility
        user_id = runtime_context.get("user_id") or runtime_context.get("student_id")
        user_type = runtime_context.get("user_type", "student")

        grade_level = runtime_context.get("grade_level", 0)
        language = runtime_context.get("language", "en")
        framework_code = runtime_context.get("framework_code")
        grade_code = runtime_context.get("grade_code")

        if not tenant_code:
            raise AgentError(
                message="tenant_code is required in runtime_context",
                agent_id=self._config.id,
            )

        if not user_id:
            raise AgentError(
                message="user_id or student_id is required in runtime_context",
                agent_id=self._config.id,
            )

        # Convert user_id to UUID if string
        if isinstance(user_id, str):
            user_id = UUID(user_id)

        # Build extra dict with optional services and session context
        extra: dict[str, Any] = {}
        if memory_manager is not None:
            extra["memory_manager"] = memory_manager

        # Include session_id in extra for tools that need it (e.g., handoff_to_practice)
        session_id = runtime_context.get("session_id")
        if session_id:
            extra["session_id"] = session_id

        # Build ToolContext
        return ToolContext(
            tenant_code=tenant_code,
            user_id=user_id,
            user_type=user_type,
            grade_level=grade_level,
            language=language,
            framework_code=framework_code,
            grade_code=grade_code,
            session=db_session,
            memory_context=memory_context,
            emotional_context=emotional_context,
            extra=extra,
        )

    def _extract_actions_and_emotions(
        self,
        result: "ToolResult",
        pending_actions: list[dict[str, Any]],
        pending_emotional_signals: list[dict[str, Any]],
    ) -> None:
        """Extract actions and emotional signals from tool result.

        Processes the tool result data to extract:
        - Actions (from result.data["action"])
        - Emotional signals (from result.data["emotion"] or _action flag)

        Args:
            result: ToolResult from tool execution.
            pending_actions: List to append actions to.
            pending_emotional_signals: List to append emotions to.
        """
        if not result.success:
            return

        data = result.data

        # Check for action in tool result (from navigate, handoff_to_tutor, etc.)
        if "action" in data:
            action_data = data["action"]
            action_type = action_data.get("type")

            # All valid action types
            valid_action_types = [
                "practice", "learning", "game", "review",
                "break", "creative", "navigate", "handoff",
            ]

            if action_type in valid_action_types:
                pending_actions.append({
                    "type": action_type,
                    "label": action_data.get("label", ""),
                    "description": action_data.get("description"),
                    "icon": action_data.get("icon"),
                    "params": action_data.get("params", {}),
                    "route": action_data.get("route"),
                    "requires_confirmation": action_data.get("requires_confirmation", False),
                    "target": action_data.get("target"),  # Legacy field
                })

        # Check for emotional signal (from record_emotion tool)
        if "emotion" in data:
            pending_emotional_signals.append({
                "emotion": data.get("emotion", ""),
                "intensity": data.get("intensity", "medium"),
                "triggers": data.get("triggers", []),
                "context": data.get("context"),
            })

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String showing agent ID, capabilities, and tool status.
        """
        caps = ", ".join(self._config.capabilities)
        persona_id = self._persona.id if self._persona else "None"
        has_tools = "yes" if self._tool_registry else "no"
        return (
            f"DynamicAgent(id={self._config.id!r}, capabilities=[{caps}], "
            f"persona={persona_id}, tools={has_tools})"
        )
