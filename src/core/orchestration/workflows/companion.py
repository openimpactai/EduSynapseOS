# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion conversation workflow using LangGraph.

This workflow manages an interactive companion conversation with tool calling
support for actions like navigation, activity suggestions, and emotional support.

The workflow uses DynamicAgent for unified LLM/tool handling, following the
same architecture as Tutoring/Practice workflows.

Workflow Structure:
    initialize
        ↓
    load_context (4-layer memory + emotional context)
        ↓
    select_persona (based on emotional state + time of day)
        ↓
    generate_greeting (proactive first message)
        ↓
    wait_for_message [INTERRUPT POINT]
        ↓
    agent (DynamicAgent with tool calling)
        ↓
    process_response (aggregate actions, prepare response)
        ↓
    check_end
        ↓
    [conditional: continue → wait_for_message, end → end_session]

The workflow uses checkpointing with interrupt_before pattern,
matching the Tutoring workflow architecture for reliable pause/resume.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.core.agents.capabilities.registry import get_default_registry
from src.core.agents.context import AgentConfig
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.intelligence.llm import LLMClient, LLMToolResponse


class CompanionResponseSchema(BaseModel):
    """Schema for structured companion responses.

    Forces LLM to return consistent JSON format that can be reliably parsed.
    This eliminates the need for cleanup logic to handle inconsistent formats.
    """

    message: str  # The message to show to the student


from src.core.orchestration.states.companion import (
    CompanionAction,
    CompanionState,
    CompanionTurn,
    EmotionalSignalRecord,
    PendingAlertRecord,
    ToolCallRecord,
    create_initial_companion_state,
)
from src.core.tools import ToolContext, ToolRegistry
from src.tools import create_registry_from_config, get_default_tool_registry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.emotional import EmotionalStateService
    from src.core.memory.manager import MemoryManager
    from src.core.personas.manager import PersonaManager
    from src.core.proactive.service import ProactiveService
    from src.domains.analytics.events import EventTracker

logger = logging.getLogger(__name__)


class CompanionWorkflow:
    """LangGraph workflow for companion conversations.

    This workflow orchestrates a multi-turn companion conversation
    with tool calling support for personalized interactions:

    Tool Integration:
        - get_activities: Suggest learning activities
        - navigate: Navigate student to specific pages
        - record_emotion: Capture emotional signals
        - handoff_to_tutor: Transfer academic questions
        - get_student_context: Get personalization context
        - get_parent_notes: Get parent-provided context
        - get_review_schedule: Get spaced repetition schedule

    Memory Integration (4 Layers):
        - Episodic: Recent learning events for context
        - Semantic: Topic mastery levels
        - Procedural: Learning patterns
        - Associative: Student interests for personalization

    The workflow uses interrupt_before=["wait_for_message"] pattern
    for reliable pause/resume, matching Tutoring workflow architecture.

    Attributes:
        llm_client: LLM client for completions with tool calling.
        tool_registry: Registry of companion tools.
        memory_manager: Manager for 4-layer memory operations.
        persona_manager: Manager for companion personas.
        emotional_service: Service for recording emotional signals.

    Example:
        >>> workflow = CompanionWorkflow(llm_client, memory_manager, ...)
        >>> initial_state = create_initial_companion_state(...)
        >>> result = await workflow.run(initial_state, thread_id="session_123")
        >>> # result contains first_greeting from generate_greeting node
        >>> # Workflow is now paused at wait_for_message
        >>>
        >>> # To send a message:
        >>> result = await workflow.send_message(thread_id, "I feel tired today")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_manager: "MemoryManager",
        persona_manager: "PersonaManager",
        tool_registry: ToolRegistry | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: "EmotionalStateService | None" = None,
        proactive_service: "ProactiveService | None" = None,
        db_session: "AsyncSession | None" = None,
        event_tracker: "EventTracker | None" = None,
    ):
        """Initialize the companion workflow.

        Args:
            llm_client: LLM client for completions with tool calling.
            memory_manager: Manager for 4-layer memory operations.
            persona_manager: Manager for companion personas.
            tool_registry: Registry of companion tools. Uses default if None.
            checkpointer: Checkpointer for state persistence (required).
            emotional_service: Service for recording emotional signals.
            proactive_service: Service for loading pending alerts.
            db_session: Database session for tool execution.
            event_tracker: Tracker for publishing analytics events.
        """
        self._llm_client = llm_client
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager
        self._checkpointer = checkpointer
        self._emotional_service = emotional_service
        self._proactive_service = proactive_service
        self._db_session = db_session
        self._event_tracker = event_tracker

        # Load agent config and create tool registry
        self._agent_config = self._load_agent_config()
        self._tool_registry = self._create_tool_registry(tool_registry)

        # Create DynamicAgent with tool support
        self._agent = DynamicAgent(
            config=self._agent_config,
            llm_client=llm_client,
            capability_registry=get_default_registry(),
            tool_registry=self._tool_registry,
        )

        # Build the workflow graph
        self._graph = self._build_graph()

    def _load_agent_config(self) -> AgentConfig:
        """Load companion agent configuration from YAML.

        Returns:
            AgentConfig loaded from config/agents/companion.yaml.

        Raises:
            FileNotFoundError: If companion.yaml is not found.
        """
        config_path = Path("config/agents/companion.yaml")
        if not config_path.exists():
            logger.error("Companion agent config not found: %s", config_path)
            raise FileNotFoundError(f"Companion agent config not found: {config_path}")

        return AgentConfig.from_yaml(config_path)

    def _create_tool_registry(
        self, provided_registry: ToolRegistry | None
    ) -> ToolRegistry:
        """Create tool registry from config or use provided.

        Args:
            provided_registry: Optional pre-configured registry.

        Returns:
            ToolRegistry instance.
        """
        # DEBUG: Log tool registry creation
        print(f"\n{'='*60}")
        print(f"[TOOL REGISTRY] Creating tool registry")
        print(f"  provided_registry is None: {provided_registry is None}")
        print(f"  self._agent_config.tools: {self._agent_config.tools}")
        print(f"  self._agent_config.tools is None: {self._agent_config.tools is None}")
        if self._agent_config.tools:
            print(f"  self._agent_config.tools.enabled: {self._agent_config.tools.enabled}")
            print(f"  definitions count: {len(self._agent_config.tools.definitions)}")
        print(f"{'='*60}\n")

        # Use provided registry if given
        if provided_registry is not None:
            print("[TOOL REGISTRY] Using provided registry")
            return provided_registry

        # Create from YAML config if tools are defined
        if self._agent_config.tools and self._agent_config.tools.enabled:
            print("[TOOL REGISTRY] Creating from config - should have 6 tools")
            registry = create_registry_from_config(self._agent_config.tools)
            print(f"[TOOL REGISTRY] Created registry with {len(registry)} tools: {registry.list_names()}")
            return registry

        # Fallback to default registry
        print("[TOOL REGISTRY] FALLBACK to default registry - 29 tools!")
        return get_default_tool_registry()

    def set_db_session(self, session: "AsyncSession") -> None:
        """Set database session for tool execution.

        Should be called by the service before each workflow run/resume.

        Args:
            session: AsyncSession for database operations.
        """
        self._db_session = session

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Graph Structure (simplified - tool execution handled by DynamicAgent):
            initialize → load_context → select_persona → generate_greeting →
            wait_for_message [INTERRUPT] → agent → process_response →
            check_end → [continue/end]

        Returns:
            StateGraph configured for companion conversations.
        """
        graph = StateGraph(CompanionState)

        # Add nodes (execute_tools removed - DynamicAgent handles tool calling)
        graph.add_node("initialize", self._initialize)
        graph.add_node("load_context", self._load_context)
        graph.add_node("select_persona", self._select_persona)
        graph.add_node("generate_greeting", self._generate_greeting)
        graph.add_node("wait_for_message", self._wait_for_message)
        graph.add_node("agent", self._agent_node)
        graph.add_node("process_response", self._process_response)
        graph.add_node("check_end", self._check_end)
        graph.add_node("end_session", self._end_session)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add linear edges (simplified - no tool execution loop)
        graph.add_edge("initialize", "load_context")
        graph.add_edge("load_context", "select_persona")
        graph.add_edge("select_persona", "generate_greeting")
        graph.add_edge("generate_greeting", "wait_for_message")
        graph.add_edge("wait_for_message", "agent")
        graph.add_edge("agent", "process_response")
        graph.add_edge("process_response", "check_end")

        # Conditional edge from check_end
        graph.add_conditional_edges(
            "check_end",
            self._should_continue,
            {
                "continue": "wait_for_message",
                "end": "end_session",
            },
        )

        graph.add_edge("end_session", END)

        return graph

    def compile(self) -> Any:
        """Compile the workflow graph with interrupt support.

        Uses interrupt_before=["wait_for_message"] to pause workflow
        when waiting for student input, matching Tutoring pattern.

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_message"],
        )

    async def run(
        self,
        initial_state: CompanionState,
        thread_id: str,
    ) -> CompanionState:
        """Run the workflow from initial state.

        Executes initialize → load_context → select_persona → generate_greeting,
        then pauses at wait_for_message (interrupt point).

        Args:
            initial_state: Starting state for the workflow.
            thread_id: Thread ID for checkpointing.

        Returns:
            Workflow state with first_greeting populated.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}
        result = await compiled.ainvoke(initial_state, config=config)
        return result

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> CompanionState:
        """Send a message and get response.

        Uses aupdate_state + ainvoke(None) pattern for proper resume,
        matching Tutoring workflow architecture.

        Args:
            thread_id: Thread ID for the conversation.
            message: Student message.

        Returns:
            Updated workflow state with companion response.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow is paused
        state_snapshot = await compiled.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No state found for thread {thread_id}")

        # Update state with student message using aupdate_state
        await compiled.aupdate_state(
            config,
            {
                "_pending_message": message,
                "awaiting_input": False,
                # Reset tool call state for new turn
                "current_tool_calls": [],
                "tool_results": [],
                "tool_call_count": 0,
                "pending_actions": [],
                "pending_emotional_signals": [],
                # Reset UI elements to prevent accumulation
                "ui_elements": [],
                "tool_data": {},
            },
        )

        logger.info(
            "Resuming workflow for thread=%s with message: %s...",
            thread_id,
            message[:50],
        )

        # Resume workflow by passing None
        result = await compiled.ainvoke(None, config=config)

        logger.info(
            "Workflow resumed: last_companion_response=%s...",
            str(result.get("last_companion_response", ""))[:50] if result else "None",
        )
        return result

    # =========================================================================
    # Node Implementations
    # =========================================================================

    async def _initialize(self, state: CompanionState) -> dict:
        """Initialize the companion session.

        Sets status to active and prepares for context loading.

        Args:
            state: Current workflow state.

        Returns:
            State updates with active status.
        """
        logger.info(
            "Initializing companion session: session=%s, persona=%s",
            state.get("session_id"),
            state.get("persona_id"),
        )

        # Publish session started event (non-blocking)
        asyncio.create_task(self._publish_session_started(state))

        return {
            "status": "active",
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: CompanionState) -> dict:
        """Load 4-layer memory context, emotional context, and pending alerts.

        Args:
            state: Current workflow state.

        Returns:
            State updates with loaded context.
        """
        logger.info(
            "Loading context: student=%s",
            state["student_id"],
        )

        memory_context = {}
        emotional_context = None
        pending_alerts: list[PendingAlertRecord] = []

        # Convert student_id to UUID if needed
        student_uuid = (
            UUID(state["student_id"])
            if isinstance(state["student_id"], str)
            else state["student_id"]
        )

        try:
            # Load full 4-layer memory context
            full_context = await self._memory_manager.get_full_context(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                session=self._db_session,
            )

            if full_context:
                memory_context = full_context.model_dump()
                logger.debug(
                    "Loaded memory context: episodic=%d",
                    len(full_context.episodic),
                )

        except Exception as e:
            logger.warning("Failed to load memory context: %s", str(e))

        try:
            # Load current emotional context
            if self._emotional_service:
                emotional_state = await self._emotional_service.get_current_state(
                    student_id=student_uuid,
                )
                if emotional_state:
                    # EmotionalContext is a dataclass, use to_dict()
                    emotional_context = emotional_state.to_dict()
                    logger.debug(
                        "Loaded emotional context: state=%s",
                        emotional_state.current_state,
                    )

        except Exception as e:
            logger.warning("Failed to load emotional context: %s", str(e))

        try:
            # Load pending alerts from ProactiveService
            if self._proactive_service:
                alerts = await self._proactive_service.get_active_alerts(
                    tenant_code=state["tenant_code"],
                    student_id=student_uuid,
                    limit=5,
                )

                pending_alerts = [
                    PendingAlertRecord(
                        id=str(alert.id),
                        alert_type=alert.alert_type,
                        severity=alert.severity,
                        title=alert.title,
                        message=alert.message,
                        topic_full_code=alert.topic_full_code,
                        created_at=alert.created_at.isoformat() if alert.created_at else None,
                    )
                    for alert in alerts
                ]

                if pending_alerts:
                    logger.info(
                        "Loaded %d pending alerts for student %s",
                        len(pending_alerts),
                        student_uuid,
                    )

        except Exception as e:
            logger.warning("Failed to load pending alerts: %s", str(e))

        return {
            "memory_context": memory_context,
            "emotional_context": emotional_context,
            "pending_alerts": pending_alerts,
        }

    async def _select_persona(self, state: CompanionState) -> dict:
        """Select persona based on emotional state and time of day.

        Also sets the selected persona on the DynamicAgent for YAML-driven
        system prompt generation.

        Args:
            state: Current workflow state.

        Returns:
            State updates with selected persona.
        """
        persona_id = state.get("persona_id", "companion")
        persona_name = None

        try:
            # Get emotional state for persona selection
            emotional_context = state.get("emotional_context")
            current_emotion = None
            if emotional_context:
                current_emotion = emotional_context.get("current_state")

            # Time-based persona selection could be added here
            now = datetime.now()
            hour = now.hour

            # Select persona based on context
            # For now, use default persona but could be enhanced
            persona = self._persona_manager.get_persona(persona_id)
            persona_name = getattr(persona, "name", persona_id)

            # Set persona on the agent for YAML-driven prompt building
            self._agent.set_persona(persona)

            logger.info(
                "Selected persona: id=%s, name=%s, emotion=%s, hour=%d",
                persona_id,
                persona_name,
                current_emotion,
                hour,
            )

        except Exception as e:
            logger.warning("Failed to select persona, using default: %s", str(e))

        return {
            "persona_id": persona_id,
            "persona_name": persona_name,
        }

    async def _generate_greeting(self, state: CompanionState) -> dict:
        """Generate proactive first greeting from companion.

        Creates a personalized greeting based on:
        - Student's memory context (recent activities, interests)
        - Current emotional state
        - Time of day
        - Selected persona

        Args:
            state: Current workflow state.

        Returns:
            State updates with first_greeting and conversation history.
        """
        logger.info("Generating greeting for session: %s", state["session_id"])

        try:
            # Build greeting prompt with context
            memory_context = state.get("memory_context", {})
            emotional_context = state.get("emotional_context")

            # Extract context for personalization
            student_interests = []
            if memory_context.get("associative"):
                interests = memory_context["associative"].get("interests", [])
                student_interests = [i.get("name", "") for i in interests[:3]]

            recent_events = []
            if memory_context.get("episodic"):
                events = memory_context["episodic"][:3]
                recent_events = [e.get("event_type", "") for e in events]

            current_emotion = "neutral"
            if emotional_context:
                current_emotion = emotional_context.get("current_state", "neutral")

            # Determine time of day greeting
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_greeting = "Good morning"
            elif 12 <= hour < 17:
                time_greeting = "Good afternoon"
            elif 17 <= hour < 21:
                time_greeting = "Good evening"
            else:
                time_greeting = "Hello"

            # Get persona name and student name for greeting
            persona_name = state.get("persona_name", "Buddy")
            student_name = state.get("student_name", "there")
            language = state.get("language", "en")
            grade_level = state.get("grade_level", 5)

            # Build simple greeting system prompt
            system_prompt = f"""You are {persona_name}, a friendly AI companion for students.
Your role is to be warm, supportive, and age-appropriate for grade {grade_level}.
Keep responses short (1-2 sentences) and speak in {language}.
IMPORTANT: Address the student by their name: {student_name}.
Respond with JSON containing a "message" field."""

            greeting_context = f"""Generate a warm greeting for a student named {student_name}.

Context: {time_greeting}, grade {grade_level}, {current_emotion} mood, {'first visit' if not recent_events else 'returning student'}

Write 1-2 sentences that are warm and inviting. Use the student's name: {student_name}."""

            # Use structured output for consistent response format
            response = await self._llm_client.complete(
                prompt=greeting_context,
                system_prompt=system_prompt,
                temperature=0,  # Low temp for structured output
                max_tokens=150,
                response_format=CompanionResponseSchema,
            )

            greeting_text = response.content.strip()

            # Parse structured output
            try:
                parsed = json.loads(greeting_text)
                if isinstance(parsed, dict) and "message" in parsed:
                    greeting_text = parsed["message"]
                    logger.debug("Extracted greeting from structured output")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Failed to parse greeting structured output: %s", e)

            # Fallback if empty
            if not greeting_text:
                greeting_text = f"{time_greeting}! It's great to see you. How are you doing today?"

            # Add to conversation history
            history = [
                CompanionTurn(
                    role="companion",
                    content=greeting_text,
                    timestamp=datetime.now().isoformat(),
                )
            ]

            logger.info("Generated greeting: %s...", greeting_text[:50])

            return {
                "first_greeting": greeting_text,
                "last_companion_response": greeting_text,
                "conversation_history": history,
                "awaiting_input": True,
                "messages": [{"role": "assistant", "content": greeting_text}],
            }

        except Exception as e:
            logger.warning("Failed to generate greeting, using default: %s", str(e))

        # Default greeting if generation fails - use student name
        student_name = state.get("student_name", "there")
        default_greeting = f"Hello {student_name}! I'm here to help you today. How are you feeling?"

        return {
            "first_greeting": default_greeting,
            "last_companion_response": default_greeting,
            "conversation_history": [
                CompanionTurn(
                    role="companion",
                    content=default_greeting,
                    timestamp=datetime.now().isoformat(),
                )
            ],
            "awaiting_input": True,
            "messages": [{"role": "assistant", "content": default_greeting}],
        }

    async def _wait_for_message(self, state: CompanionState) -> dict:
        """Wait for student message (interrupt point).

        This node is the interrupt point. When workflow reaches here,
        it pauses and waits for send_message() to inject a message
        via aupdate_state.

        Args:
            state: Current workflow state.

        Returns:
            State updates extracting pending message.
        """
        pending_message = state.get("_pending_message")
        logger.info(
            "wait_for_message node: pending_message=%s, awaiting=%s",
            pending_message[:50] if pending_message else None,
            state.get("awaiting_input"),
        )

        if pending_message:
            # Add student message to conversation history
            history = list(state.get("conversation_history", []))
            history.append(
                CompanionTurn(
                    role="student",
                    content=pending_message,
                    timestamp=datetime.now().isoformat(),
                )
            )

            # Add to LangGraph messages for tool calling context
            messages = list(state.get("messages", []))
            messages.append({"role": "user", "content": pending_message})

            return {
                "last_student_message": pending_message,
                "_pending_message": None,  # Clear pending
                "conversation_history": history,
                "messages": messages,
                "awaiting_input": False,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "awaiting_input": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _agent_node(self, state: CompanionState) -> dict:
        """Execute DynamicAgent with tool calling support.

        This node uses DynamicAgent.execute_with_tools() which handles
        the entire tool calling loop internally, replacing the previous
        separate _agent and _execute_tools nodes.

        Args:
            state: Current workflow state.

        Returns:
            State updates with LLM response, actions, and emotional signals.
        """
        logger.info(
            "Agent node: message=%s",
            state.get("last_student_message", "")[:50]
            if state.get("last_student_message")
            else None,
        )

        # Ensure persona is set on agent (may be needed after workflow resume)
        if not self._agent._prompt_builder:
            persona_id = state.get("persona_id", "companion")
            try:
                persona = self._persona_manager.get_persona(persona_id)
                self._agent.set_persona(persona)
                logger.debug("Re-initialized persona on agent: %s", persona_id)
            except Exception as e:
                logger.warning("Failed to set persona on agent: %s", str(e))

        # Build runtime context for prompt interpolation
        memory_context = state.get("memory_context", {})
        emotional_context_dict = state.get("emotional_context")

        # Extract interests for context
        interests = []
        if memory_context.get("associative"):
            raw_interests = memory_context["associative"].get("interests", [])
            interests = [i.get("name", "") for i in raw_interests[:5]]

        # Get current emotional state
        current_emotion = "neutral"
        if emotional_context_dict:
            current_emotion = emotional_context_dict.get("current_state", "neutral")

        # Get pending alerts for context
        pending_alerts = state.get("pending_alerts", [])
        alerts_for_context = [
            {
                "severity": alert.get("severity", "info"),
                "title": alert.get("title", ""),
                "message": alert.get("message", ""),
            }
            for alert in pending_alerts[:3]
        ]

        runtime_context = {
            "tenant_code": state["tenant_code"],
            "student_id": state["student_id"],
            "user_id": state["student_id"],  # New field for ToolContext
            "user_type": "student",  # New field for ToolContext
            "session_id": state.get("session_id"),  # Companion session ID for handoff tools
            "student_name": state.get("student_name", "there"),
            "grade_level": state.get("grade_level", 5),
            "framework_code": state.get("framework_code"),  # Curriculum framework code
            "grade_code": state.get("grade_code"),  # Grade code within framework
            "language": state.get("language", "en"),
            "persona_name": state.get("persona_name", "Buddy"),
            "current_emotion": current_emotion,
            "interests": interests,
            "alerts": alerts_for_context,
        }

        # Build memory and emotional context objects for tools
        memory_context_obj = None
        emotional_context_obj = None

        if memory_context:
            try:
                from src.models.memory import FullMemoryContext

                memory_context_obj = FullMemoryContext.model_validate(memory_context)
            except Exception:
                pass

        if emotional_context_dict:
            try:
                from src.core.emotional.context import EmotionalContext

                emotional_context_obj = EmotionalContext.model_validate(
                    emotional_context_dict
                )
            except Exception:
                pass

        # Build messages from state - filter for LLM compatibility
        messages = []
        raw_messages = state.get("messages", [])

        # DEBUG: Log raw state messages
        print(f"\n{'='*60}")
        print(f"=== RAW STATE MESSAGES (count={len(raw_messages)}) ===")
        for i, msg in enumerate(raw_messages):
            if isinstance(msg, dict):
                role = msg.get("role", "?")
                content = msg.get("content", "")[:200]
                print(f"  [{i}] role={role} content={content}...")
            else:
                role = getattr(msg, "type", "?")
                content = getattr(msg, "content", "")[:200]
                print(f"  [{i}] type={role} content={content}...")

        for msg in raw_messages:
            if isinstance(msg, dict):
                # Only include role and content for Ollama compatibility
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
            else:
                # LangGraph message objects
                role = getattr(msg, "type", "assistant")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = getattr(msg, "content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # DEBUG: Log messages being sent to LLM
        print(f"\n=== MESSAGES TO LLM (count={len(messages)}) ===")
        for i, m in enumerate(messages):
            content = m["content"][:300] + "..." if len(m["content"]) > 300 else m["content"]
            print(f"  [{i}] role={m['role']} content={content}")

        try:
            # Get tool_choice from config (default: "auto")
            tool_choice = "auto"
            if self._agent_config.tools:
                tool_choice = self._agent_config.tools.tool_choice

            # Call DynamicAgent with tool support
            result = await self._agent.execute_with_tools(
                messages=messages,
                runtime_context=runtime_context,
                db_session=self._db_session,
                memory_context=memory_context_obj,
                emotional_context=emotional_context_obj,
                memory_manager=self._memory_manager,
                tool_choice=tool_choice,
            )

            if not result.get("success"):
                logger.error("Agent execution failed: %s", result.get("error"))
                return {
                    "error": result.get("error", "Unknown error"),
                    "last_companion_response": "",
                }

            response_text = result.get("content", "").strip()

            # Clean any accidental tool results echoing from response
            # This can happen if LLM sees tool results in history and echoes them
            if response_text.startswith("[Tool Results]") or response_text.startswith("Context from previous"):
                # Find the actual response after tool results
                lines = response_text.split("\n\n", 1)
                if len(lines) > 1:
                    response_text = lines[1].strip()
                else:
                    # Try to find after single newline
                    for i, line in enumerate(response_text.split("\n")):
                        if not line.startswith("-") and not line.startswith("•") and not line.startswith("[") and not line.startswith("Context"):
                            if line.strip():
                                response_text = "\n".join(response_text.split("\n")[i:]).strip()
                                break

            # Parse JSON if response looks like JSON (structured output)
            if response_text.startswith("{") and response_text.endswith("}"):
                try:
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict) and "message" in parsed:
                        response_text = parsed["message"]
                        logger.debug("Extracted message from JSON response")
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning("Failed to parse JSON response: %s", e)

            logger.info(
                "Agent generated response: %s... (tools=%d, actions=%d)",
                response_text[:50] if response_text else "empty",
                len(result.get("tool_calls_made", [])),
                len(result.get("pending_actions", [])),
            )

            # Convert pending actions to CompanionAction TypedDicts
            pending_actions = []
            for action in result.get("pending_actions", []):
                pending_actions.append(
                    CompanionAction(
                        type=action.get("type", ""),
                        label=action.get("label", ""),
                        description=action.get("description"),
                        icon=action.get("icon"),
                        params=action.get("params", {}),
                        route=action.get("route"),
                        requires_confirmation=action.get("requires_confirmation", False),
                        target=action.get("target"),
                    )
                )

            # Convert pending emotional signals to EmotionalSignalRecord TypedDicts
            pending_signals = []
            for signal in result.get("pending_emotional_signals", []):
                pending_signals.append(
                    EmotionalSignalRecord(
                        emotion=signal.get("emotion", ""),
                        intensity=signal.get("intensity", "medium"),
                        triggers=signal.get("triggers", []),
                        context=signal.get("context"),
                    )
                )

            # Build messages to save - include tool results for conversation history
            # This ensures LLM sees tool results WITH selection options and IDs
            # so it can match user's text input to the correct item
            messages_to_save = []
            tool_results = result.get("tool_results", [])

            # DEBUG: Log tool results being processed
            print(f"\n=== TOOL RESULTS TO SAVE (count={len(tool_results)}) ===")
            for i, tr in enumerate(tool_results):
                print(f"  [{i}] name={tr.get('name', '?')} data_keys={list(tr.get('data', {}).keys())}")

            if tool_results:
                for tr in tool_results:
                    tool_name = tr.get("name", "")
                    tool_data = tr.get("data", {})

                    # DEBUG: Print tool_data content
                    print(f"  DEBUG tool_data for {tool_name}: {list(tool_data.keys())}")
                    if "subjects" in tool_data:
                        print(f"  DEBUG subjects count: {len(tool_data['subjects'])}")
                        if tool_data['subjects']:
                            print(f"  DEBUG first subject: {tool_data['subjects'][0]}")

                    # Build context with selection options including IDs
                    context_parts = []

                    # Add message summary
                    msg = tool_data.get("message", "")
                    if msg:
                        context_parts.append(msg)
                        print(f"  DEBUG added message: {msg[:50]}...")

                    # Add subjects with codes for follow-up selection
                    if "subjects" in tool_data:
                        options = [
                            f"- {s['name']} (code: {s['code']})"
                            for s in tool_data["subjects"]
                        ]
                        subjects_text = "\nAvailable subjects:\n" + "\n".join(options)
                        context_parts.append(subjects_text)
                        print(f"  DEBUG added subjects: {len(options)} items")

                    # Add topics with codes for follow-up selection
                    if "topics" in tool_data:
                        options = [
                            f"- {t['name']} (code: {t['code']})"
                            for t in tool_data["topics"]
                        ]
                        context_parts.append(
                            "\nAvailable topics:\n" + "\n".join(options)
                        )

                    # Add games with codes for follow-up selection
                    if "games" in tool_data:
                        options = [
                            f"- {g['name']} (code: {g['code']})"
                            for g in tool_data["games"]
                        ]
                        context_parts.append(
                            "\nAvailable games:\n" + "\n".join(options)
                        )

                    # Add activities with codes for follow-up selection
                    if "activities" in tool_data:
                        options = [
                            f"- {a['name']} (code: {a['code']})"
                            for a in tool_data["activities"]
                        ]
                        context_parts.append(
                            "\nAvailable activities:\n" + "\n".join(options)
                        )

                    print(f"  DEBUG context_parts count: {len(context_parts)}")
                    if context_parts:
                        context = f"[{tool_name}]\n" + "\n".join(context_parts)
                        print(f"  DEBUG final context length: {len(context)} chars")
                        messages_to_save.append({"role": "system", "content": context})

            messages_to_save.append({"role": "assistant", "content": response_text})

            # DEBUG: Log messages being saved to state
            print(f"\n=== MESSAGES TO SAVE (count={len(messages_to_save)}) ===")
            for i, m in enumerate(messages_to_save):
                content = m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"]
                print(f"  [{i}] role={m['role']} content={content}")
            print(f"{'='*60}\n")

            return {
                "last_companion_response": response_text,
                "pending_actions": pending_actions,
                "pending_emotional_signals": pending_signals,
                "messages": messages_to_save,
                # UI elements and tool data for frontend
                "ui_elements": result.get("ui_elements", []),
                "tool_data": result.get("tool_data", {}),
                # Clear tool-related state (no longer used with DynamicAgent)
                "current_tool_calls": [],
                "tool_results": [],
                # Track actual tool calls made for metadata
                "tool_call_count": len(result.get("tool_calls_made", [])),
            }

        except Exception as e:
            logger.exception("Agent execution failed")
            return {
                "error": f"Agent execution failed: {str(e)}",
                "last_companion_response": "",
            }

    async def _process_response(self, state: CompanionState) -> dict:
        """Process the final response and prepare for output.

        Aggregates actions, records emotional signals, and updates
        conversation history with the companion response.

        Args:
            state: Current workflow state.

        Returns:
            State updates with finalized response.
        """
        response_text = state.get("last_companion_response", "")

        if not response_text:
            # This should not happen - agent node should always generate a response
            # Log error but don't mask with static fallback
            logger.error(
                "Agent failed to generate response for session=%s. "
                "This indicates a bug in the agent node.",
                state.get("session_id"),
            )

        # Add response to conversation history
        history = list(state.get("conversation_history", []))
        history.append(
            CompanionTurn(
                role="companion",
                content=response_text,
                timestamp=datetime.now().isoformat(),
            )
        )

        # Record emotional signals (fire-and-forget)
        pending_signals = state.get("pending_emotional_signals", [])
        if pending_signals:
            for signal in pending_signals:
                # Record to emotional service if available
                if self._emotional_service:
                    asyncio.create_task(
                        self._record_emotional_signal(
                            student_id=state["student_id"],
                            signal=signal,
                            session_id=state.get("session_id"),
                        )
                    )

                # Publish emotion detected event for analytics
                asyncio.create_task(
                    self._publish_emotion_detected(
                        state=state,
                        emotion=signal.get("emotion", "neutral"),
                        intensity=signal.get("intensity", "medium"),
                        triggers=signal.get("triggers"),
                    )
                )

        # Check for handoff or activity suggestion actions and publish events
        pending_actions = state.get("pending_actions", [])
        for action in pending_actions:
            action_type = action.get("type", "")

            if action_type == "handoff":
                target_workflow = action.get("target", "unknown")
                topic_code = action.get("params", {}).get("topic_code")

                # Publish handoff initiated event
                asyncio.create_task(
                    self._publish_handoff_initiated(
                        state=state,
                        target_workflow=target_workflow,
                        handoff_reason=action.get("description"),
                        topic_code=topic_code,
                    )
                )

                # Record handoff to memory
                asyncio.create_task(
                    self._record_handoff_memory(
                        state=state,
                        target_workflow=target_workflow,
                        topic_code=topic_code,
                    )
                )
            elif action_type in ("activity", "navigate"):
                # Publish activity suggested event
                asyncio.create_task(
                    self._publish_activity_suggested(
                        state=state,
                        activity_type=action_type,
                        activity_code=action.get("params", {}).get("code"),
                    )
                )

        logger.info(
            "Processed response: %d chars, %d actions, %d signals",
            len(response_text),
            len(state.get("pending_actions", [])),
            len(pending_signals),
        )

        return {
            "last_companion_response": response_text,
            "conversation_history": history,
            "awaiting_input": True,
            "pending_emotional_signals": [],  # Clear after processing
        }

    async def _record_emotional_signal(
        self,
        student_id: str,
        signal: EmotionalSignalRecord,
        session_id: str | None = None,
    ) -> None:
        """Record emotional signal (fire-and-forget).

        Args:
            student_id: Student ID.
            signal: Emotional signal to record.
            session_id: Optional session ID.
        """
        try:
            from src.core.emotional import EmotionalSignalSource

            await self._emotional_service.record_analyzed_signal(
                student_id=UUID(student_id),
                source=EmotionalSignalSource.SELF_REPORT,
                emotional_state=signal.get("emotion", "neutral"),
                intensity=signal.get("intensity", "medium"),
                confidence=0.9,  # Self-reported has high confidence
                triggers=signal.get("triggers", []),
                activity_id=session_id,
                activity_type="companion_conversation",
                context={"recorded_via": "record_emotion_tool"},
            )
            logger.debug(
                "Recorded emotional signal: student=%s, emotion=%s",
                student_id,
                signal.get("emotion"),
            )
        except Exception as e:
            logger.warning(
                "Failed to record emotional signal: student=%s, error=%s",
                student_id,
                str(e),
            )

    async def _check_end(self, state: CompanionState) -> dict:
        """Check if session should end.

        Args:
            state: Current workflow state.

        Returns:
            State updates (empty for continue, status for end).
        """
        return {}

    async def _end_session(self, state: CompanionState) -> dict:
        """End the companion session with final cleanup.

        Args:
            state: Current workflow state.

        Returns:
            State updates marking completion.
        """
        logger.info(
            "Companion session ended: session=%s, turns=%d",
            state.get("session_id"),
            len(state.get("conversation_history", [])),
        )

        # Publish session ended event (non-blocking)
        asyncio.create_task(self._publish_session_ended(state))

        # Record to memory (non-blocking)
        asyncio.create_task(self._record_session_memory(state))
        asyncio.create_task(self._record_procedural_observation(state))

        return {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "awaiting_input": False,
        }

    # =========================================================================
    # Routing Functions
    # =========================================================================

    def _should_continue(
        self, state: CompanionState
    ) -> Literal["continue", "end"]:
        """Determine if workflow should continue or end.

        Ends on:
        - Error status

        Note: Handoff actions no longer end the session. The handoff information
        is sent to the frontend for navigation, but the companion session
        continues so the student can choose to do something else instead.

        Args:
            state: Current workflow state.

        Returns:
            "continue" to wait for more messages, "end" to finish.
        """
        if state.get("status") == "error":
            return "end"

        return "continue"

    # =========================================================================
    # Event Publishing Methods
    # =========================================================================

    async def _publish_session_started(self, state: CompanionState) -> None:
        """Publish companion session started event.

        Args:
            state: Current workflow state.
        """
        try:
            # Create tracker locally if not provided (like practice workflow)
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="companion.session.started",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "companion",
                    "persona_id": state.get("persona_id"),
                    "persona_name": state.get("persona_name"),
                    "language": state.get("language"),
                    "grade_level": state.get("grade_level"),
                },
            )
            logger.debug("Published companion.session.started event")
        except Exception as e:
            logger.warning("Failed to publish session started event: %s", str(e))

    async def _publish_session_ended(self, state: CompanionState) -> None:
        """Publish companion session ended event.

        Args:
            state: Current workflow state.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="companion.session.ended",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "companion",
                    "turn_count": len(state.get("conversation_history", [])),
                    "tool_call_count": state.get("tool_call_count", 0),
                },
            )
            logger.debug("Published companion.session.ended event")
        except Exception as e:
            logger.warning("Failed to publish session ended event: %s", str(e))

    async def _publish_emotion_detected(
        self,
        state: CompanionState,
        emotion: str,
        intensity: str,
        triggers: list[str] | None = None,
    ) -> None:
        """Publish emotion detected event.

        Args:
            state: Current workflow state.
            emotion: Detected emotion.
            intensity: Emotion intensity.
            triggers: Optional list of triggers.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="companion.emotion.detected",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "companion",
                    "emotion": emotion,
                    "intensity": intensity,
                    "triggers": triggers or [],
                },
            )
            logger.debug("Published companion.emotion.detected event: %s", emotion)
        except Exception as e:
            logger.warning("Failed to publish emotion detected event: %s", str(e))

    async def _publish_activity_suggested(
        self,
        state: CompanionState,
        activity_type: str,
        activity_code: str | None = None,
    ) -> None:
        """Publish activity suggested event.

        Args:
            state: Current workflow state.
            activity_type: Type of activity suggested.
            activity_code: Optional activity code.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="companion.activity.suggested",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "companion",
                    "activity_type": activity_type,
                    "activity_code": activity_code,
                },
            )
            logger.debug("Published companion.activity.suggested event: %s", activity_type)
        except Exception as e:
            logger.warning("Failed to publish activity suggested event: %s", str(e))

    async def _publish_handoff_initiated(
        self,
        state: CompanionState,
        target_workflow: str,
        handoff_reason: str | None = None,
        topic_code: str | None = None,
    ) -> None:
        """Publish handoff initiated event.

        Args:
            state: Current workflow state.
            target_workflow: Workflow being handed off to.
            handoff_reason: Optional reason for handoff.
            topic_code: Optional topic code for the handoff.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="companion.handoff.initiated",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "companion",
                    "target_workflow": target_workflow,
                    "handoff_reason": handoff_reason,
                    "topic_code": topic_code,
                },
            )
            logger.debug("Published companion.handoff.initiated event: %s", target_workflow)
        except Exception as e:
            logger.warning("Failed to publish handoff initiated event: %s", str(e))

    # =========================================================================
    # Memory Recording Methods
    # =========================================================================

    async def _record_session_memory(self, state: CompanionState) -> None:
        """Record companion session completion in episodic memory.

        Args:
            state: Current workflow state.
        """
        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            conversation_history = state.get("conversation_history", [])

            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                event_type="companion_session",
                topic="companion_conversation",
                data={
                    "session_id": state["session_id"],
                    "turn_count": len(conversation_history),
                    "tool_call_count": state.get("tool_call_count", 0),
                    "persona_id": state.get("persona_id"),
                    "persona_name": state.get("persona_name"),
                },
                importance=0.4,
            )

            logger.debug("Recorded companion session to episodic memory")

        except Exception as e:
            logger.warning("Failed to record session memory: %s", str(e))

    async def _record_handoff_memory(
        self,
        state: CompanionState,
        target_workflow: str,
        topic_code: str | None = None,
    ) -> None:
        """Record handoff event in episodic memory.

        Handoffs are significant events that indicate student learning intent.

        Args:
            state: Current workflow state.
            target_workflow: Workflow being handed off to.
            topic_code: Optional topic code for the handoff.
        """
        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                event_type="companion_handoff",
                topic=topic_code or "companion_handoff",
                data={
                    "session_id": state["session_id"],
                    "target_workflow": target_workflow,
                    "topic_code": topic_code,
                },
                topic_full_code=topic_code,
                importance=0.6,
            )

            logger.debug("Recorded handoff to episodic memory: %s", target_workflow)

        except Exception as e:
            logger.warning("Failed to record handoff memory: %s", str(e))

    async def _record_procedural_observation(self, state: CompanionState) -> None:
        """Record interaction pattern observation in procedural memory.

        Tracks behavioral patterns that inform personalization:
        - When the student interacts with companion (time of day)
        - Emotional patterns during conversations
        - Tool usage patterns

        Args:
            state: Current workflow state.
        """
        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            # Determine time of day bucket
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                time_of_day = "morning"
            elif 12 <= current_hour < 17:
                time_of_day = "afternoon"
            elif 17 <= current_hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            conversation_history = state.get("conversation_history", [])
            emotional_context = state.get("emotional_context") or {}

            observation = {
                "session_type": "companion",
                "time_of_day": time_of_day,
                "turn_count": len(conversation_history),
                "tool_call_count": state.get("tool_call_count", 0),
                "emotional_state": emotional_context.get("current_state", "neutral"),
                "persona_id": state.get("persona_id"),
                "language": state.get("language"),
                "session_id": state["session_id"],
            }

            await self._memory_manager.record_procedural_observation(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                observation=observation,
                topic_full_code=None,
            )

            logger.debug(
                "Recorded procedural observation: time=%s, turns=%d",
                time_of_day,
                len(conversation_history),
            )

        except Exception as e:
            logger.warning("Failed to record procedural observation: %s", str(e))
