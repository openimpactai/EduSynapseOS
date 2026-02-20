# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher companion conversation workflow using LangGraph.

This workflow manages an interactive teacher assistant conversation with tool calling
support for class management, student monitoring, and analytics viewing.

The workflow uses DynamicAgent for unified LLM/tool handling, following the
same architecture as Companion workflow but tailored for teachers.

Workflow Structure:
    initialize
        ↓
    load_context (class info + alert summary)
        ↓
    select_persona
        ↓
    generate_greeting (professional greeting with status overview)
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
matching the Companion workflow architecture for reliable pause/resume.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.core.agents.capabilities.registry import get_default_registry
from src.core.agents.context import AgentConfig
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.intelligence.llm import LLMClient
from src.core.orchestration.states.teacher_companion import (
    TeacherAction,
    TeacherCompanionState,
    TeacherTurn,
)
from src.core.tools import ToolRegistry
from src.tools import create_registry_from_config, get_default_tool_registry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.personas.manager import PersonaManager

logger = logging.getLogger(__name__)


class TeacherResponseSchema(BaseModel):
    """Schema for structured teacher assistant responses."""

    message: str  # The message to show to the teacher


class TeacherCompanionWorkflow:
    """LangGraph workflow for teacher assistant conversations.

    This workflow orchestrates a multi-turn teacher assistant conversation
    with tool calling support for:

    Tool Integration:
        - get_my_classes: List teacher's assigned classes
        - get_class_students: List students in a class
        - get_student_progress: View student progress
        - get_student_mastery: View student mastery levels
        - get_class_analytics: View class performance analytics
        - get_struggling_students: Identify students who need help
        - get_topic_performance: View topic-level performance
        - get_student_notes: View notes about a student
        - get_alerts: View pending alerts
        - get_emotional_history: View student emotional history

    The workflow uses interrupt_before=["wait_for_message"] pattern
    for reliable pause/resume, matching Companion workflow architecture.

    Attributes:
        llm_client: LLM client for completions with tool calling.
        tool_registry: Registry of teacher tools.
        persona_manager: Manager for assistant personas.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        persona_manager: "PersonaManager",
        tool_registry: ToolRegistry | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        db_session: "AsyncSession | None" = None,
    ):
        """Initialize the teacher companion workflow.

        Args:
            llm_client: LLM client for completions with tool calling.
            persona_manager: Manager for assistant personas.
            tool_registry: Registry of teacher tools. Uses default if None.
            checkpointer: Checkpointer for state persistence (required).
            db_session: Database session for tool execution.
        """
        self._llm_client = llm_client
        self._persona_manager = persona_manager
        self._checkpointer = checkpointer
        self._db_session = db_session

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
        """Load teacher companion agent configuration from YAML.

        Returns:
            AgentConfig loaded from config/agents/teacher_companion.yaml.

        Raises:
            FileNotFoundError: If teacher_companion.yaml is not found.
        """
        config_path = Path("config/agents/teacher_companion.yaml")
        if not config_path.exists():
            logger.error("Teacher companion agent config not found: %s", config_path)
            raise FileNotFoundError(f"Teacher companion agent config not found: {config_path}")

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
        if provided_registry is not None:
            return provided_registry

        if self._agent_config.tools and self._agent_config.tools.enabled:
            return create_registry_from_config(self._agent_config.tools)

        return get_default_tool_registry()

    def set_db_session(self, session: "AsyncSession") -> None:
        """Set database session for tool execution.

        Args:
            session: AsyncSession for database operations.
        """
        self._db_session = session

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            StateGraph configured for teacher conversations.
        """
        graph = StateGraph(TeacherCompanionState)

        # Add nodes
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

        # Add linear edges
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

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_message"],
        )

    async def run(
        self,
        initial_state: TeacherCompanionState,
        thread_id: str,
    ) -> TeacherCompanionState:
        """Run the workflow from initial state.

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

    async def get_state(self, thread_id: str) -> dict | None:
        """Get current workflow state for a thread.

        Args:
            thread_id: Thread ID for the conversation.

        Returns:
            Current state dict or None if not found.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await compiled.aget_state(config)
        if state_snapshot and state_snapshot.values:
            return dict(state_snapshot.values)
        return None

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> TeacherCompanionState:
        """Send a message and get response.

        Args:
            thread_id: Thread ID for the conversation.
            message: Teacher message.

        Returns:
            Updated workflow state with assistant response.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow is paused
        state_snapshot = await compiled.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No state found for thread {thread_id}")

        # Update state with teacher message
        await compiled.aupdate_state(
            config,
            {
                "_pending_message": message,
                "awaiting_input": False,
                "current_tool_calls": [],
                "tool_results": [],
                "tool_call_count": 0,
                "pending_actions": [],
            },
        )

        logger.info(
            "Resuming teacher workflow for thread=%s with message: %s...",
            thread_id,
            message[:50],
        )

        result = await compiled.ainvoke(None, config=config)

        logger.info(
            "Teacher workflow resumed: last_assistant_response=%s...",
            str(result.get("last_assistant_response", ""))[:50] if result else "None",
        )
        return result

    # =========================================================================
    # Node Implementations
    # =========================================================================

    async def _initialize(self, state: TeacherCompanionState) -> dict:
        """Initialize the teacher session."""
        logger.info(
            "Initializing teacher session: session=%s, teacher=%s",
            state.get("session_id"),
            state.get("teacher_id"),
        )

        return {
            "status": "active",
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: TeacherCompanionState) -> dict:
        """Pass through pre-loaded teacher context from state.

        Context (class_summary, alert_summary) is loaded by the service layer
        before workflow execution to ensure proper transaction isolation.
        This node simply logs and passes through the pre-loaded context.

        Args:
            state: Current workflow state with pre-loaded context.

        Returns:
            State updates (context is already in state, this is a pass-through).
        """
        class_summary = state.get("class_summary", [])
        alert_summary = state.get("alert_summary")

        logger.info(
            "Using pre-loaded teacher context: teacher=%s, classes=%d, alerts=%s",
            state["teacher_id"],
            len(class_summary),
            alert_summary["total_count"] if alert_summary else 0,
        )

        # Context is already in state from service layer
        # Return empty dict as no state updates needed
        return {}

    async def _select_persona(self, state: TeacherCompanionState) -> dict:
        """Select persona for the teacher assistant."""
        persona_id = state.get("persona_id", "teacher_assistant")
        persona_name = None

        try:
            persona = self._persona_manager.get_persona(persona_id)
            persona_name = getattr(persona, "name", persona_id)

            # Set persona on the agent
            self._agent.set_persona(persona)

            logger.info(
                "Selected persona: id=%s, name=%s",
                persona_id,
                persona_name,
            )

        except Exception as e:
            logger.warning("Failed to select persona, using default: %s", str(e))

        return {
            "persona_id": persona_id,
            "persona_name": persona_name,
        }

    async def _generate_greeting(self, state: TeacherCompanionState) -> dict:
        """Generate professional greeting for teacher.

        Creates a greeting that includes:
        - Time-appropriate salutation
        - Summary of classes
        - Alert status if there are pending alerts
        """
        logger.info("Generating greeting for teacher session: %s", state["session_id"])

        try:
            class_summary = state.get("class_summary", [])
            alert_summary = state.get("alert_summary")
            language = state.get("language", "en")
            persona_name = state.get("persona_name", "Assistant")

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

            # Build context for greeting
            class_count = len(class_summary)
            total_students = sum(c.get("student_count", 0) for c in class_summary)

            alert_info = ""
            if alert_summary and alert_summary.get("total_count", 0) > 0:
                alert_count = alert_summary["total_count"]
                critical = alert_summary.get("critical_count", 0)
                if critical > 0:
                    alert_info = f"You have {alert_count} pending alerts, {critical} are critical."
                else:
                    alert_info = f"You have {alert_count} pending alerts."

            # Build greeting prompt
            system_prompt = f"""You are {persona_name}, a professional AI assistant for teachers.
Your role is to help teachers monitor their students, view analytics, and manage their classes.
Keep responses concise (1-2 sentences) and professional.
Speak in {language}.
Respond with JSON containing a "message" field."""

            greeting_context = f"""Generate a professional greeting for a teacher.

Context:
- {time_greeting}
- Teacher has {class_count} classes with {total_students} total students
- {alert_info if alert_info else "No pending alerts"}

Write 1-2 sentences that are professional and helpful."""

            response = await self._llm_client.complete(
                prompt=greeting_context,
                system_prompt=system_prompt,
                temperature=0,
                max_tokens=150,
                response_format=TeacherResponseSchema,
            )

            greeting_text = response.content.strip()

            # Parse structured output
            try:
                parsed = json.loads(greeting_text)
                if isinstance(parsed, dict) and "message" in parsed:
                    greeting_text = parsed["message"]
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

            if not greeting_text:
                greeting_text = f"{time_greeting}! I'm here to help you manage your classes and monitor your students."

            # Add to conversation history
            history = [
                TeacherTurn(
                    role="assistant",
                    content=greeting_text,
                    timestamp=datetime.now().isoformat(),
                )
            ]

            logger.info("Generated greeting: %s...", greeting_text[:50])

            return {
                "first_greeting": greeting_text,
                "last_assistant_response": greeting_text,
                "conversation_history": history,
                "awaiting_input": True,
                "messages": [{"role": "assistant", "content": greeting_text}],
            }

        except Exception as e:
            logger.warning("Failed to generate greeting: %s", str(e))

        default_greeting = "Hello! I'm here to help you manage your classes and monitor your students."

        return {
            "first_greeting": default_greeting,
            "last_assistant_response": default_greeting,
            "conversation_history": [
                TeacherTurn(
                    role="assistant",
                    content=default_greeting,
                    timestamp=datetime.now().isoformat(),
                )
            ],
            "awaiting_input": True,
            "messages": [{"role": "assistant", "content": default_greeting}],
        }

    async def _wait_for_message(self, state: TeacherCompanionState) -> dict:
        """Wait for teacher message (interrupt point)."""
        pending_message = state.get("_pending_message")
        logger.info(
            "wait_for_message node: pending_message=%s",
            pending_message[:50] if pending_message else None,
        )

        if pending_message:
            history = list(state.get("conversation_history", []))
            history.append(
                TeacherTurn(
                    role="teacher",
                    content=pending_message,
                    timestamp=datetime.now().isoformat(),
                )
            )

            messages = list(state.get("messages", []))
            messages.append({"role": "user", "content": pending_message})

            return {
                "last_teacher_message": pending_message,
                "_pending_message": None,
                "conversation_history": history,
                "messages": messages,
                "awaiting_input": False,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "awaiting_input": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _agent_node(self, state: TeacherCompanionState) -> dict:
        """Execute DynamicAgent with tool calling support for teacher."""
        logger.info(
            "Agent node: message=%s",
            state.get("last_teacher_message", "")[:50]
            if state.get("last_teacher_message")
            else None,
        )

        # Ensure persona is set
        if not self._agent._prompt_builder:
            persona_id = state.get("persona_id", "teacher_assistant")
            try:
                persona = self._persona_manager.get_persona(persona_id)
                self._agent.set_persona(persona)
            except Exception as e:
                logger.warning("Failed to set persona: %s", str(e))

        # Build runtime context for teacher
        class_summary = state.get("class_summary", [])
        alert_summary = state.get("alert_summary")

        # Format class info for context
        class_info = []
        for c in class_summary[:5]:
            info = f"{c.get('class_name', 'Unknown')}"
            if c.get("subject_name"):
                info += f" ({c.get('subject_name')})"
            info += f" - {c.get('student_count', 0)} students"
            class_info.append(info)

        # Format alert info
        alert_info = "No pending alerts"
        if alert_summary and alert_summary.get("total_count", 0) > 0:
            alert_info = f"{alert_summary['total_count']} alerts ({alert_summary.get('critical_count', 0)} critical)"

        runtime_context = {
            "tenant_code": state["tenant_code"],
            "user_id": state["teacher_id"],
            "user_type": "teacher",
            "grade_level": 0,  # Not applicable for teachers
            "language": state.get("language", "en"),
            "persona_name": state.get("persona_name", "Assistant"),
            "class_count": len(class_summary),
            "class_info": class_info,
            "alert_info": alert_info,
        }

        # Build messages from state
        messages = []
        raw_messages = state.get("messages", [])

        for msg in raw_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
            else:
                role = getattr(msg, "type", "assistant")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = getattr(msg, "content", "")
                if content:
                    messages.append({"role": role, "content": content})

        try:
            # Call DynamicAgent (no memory_context or emotional_context for teachers)
            result = await self._agent.execute_with_tools(
                messages=messages,
                runtime_context=runtime_context,
                db_session=self._db_session,
                memory_context=None,
                emotional_context=None,
                memory_manager=None,
            )

            if not result.get("success"):
                logger.error("Agent execution failed: %s", result.get("error"))
                return {
                    "error": result.get("error", "Unknown error"),
                    "last_assistant_response": "",
                }

            response_text = result.get("content", "").strip()

            # Parse JSON if structured output
            if response_text.startswith("{") and response_text.endswith("}"):
                try:
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict) and "message" in parsed:
                        response_text = parsed["message"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            logger.info(
                "Agent response: %s... (tools=%d, actions=%d)",
                response_text[:50] if response_text else "empty",
                len(result.get("tool_calls_made", [])),
                len(result.get("pending_actions", [])),
            )

            # Convert pending actions
            pending_actions = []
            for action in result.get("pending_actions", []):
                pending_actions.append(
                    TeacherAction(
                        type=action.get("type", "navigate"),
                        label=action.get("label", ""),
                        description=action.get("description"),
                        icon=action.get("icon"),
                        params=action.get("params", {}),
                        route=action.get("route"),
                    )
                )

            # Build messages to save
            messages_to_save = []
            tool_results = result.get("tool_results", [])

            if tool_results:
                for tr in tool_results:
                    tool_name = tr.get("name", "")
                    tool_data = tr.get("data", {})

                    context_parts = []
                    msg = tool_data.get("message", "")
                    if msg:
                        context_parts.append(msg)

                    if context_parts:
                        context = f"[{tool_name}]\n" + "\n".join(context_parts)
                        messages_to_save.append({"role": "system", "content": context})

            messages_to_save.append({"role": "assistant", "content": response_text})

            return {
                "last_assistant_response": response_text,
                "pending_actions": pending_actions,
                "messages": messages_to_save,
                "ui_elements": result.get("ui_elements", []),
                "tool_data": result.get("tool_data", {}),
                "current_tool_calls": [],
                "tool_results": [],
                "tool_call_count": len(result.get("tool_calls_made", [])),
            }

        except Exception as e:
            logger.exception("Agent execution failed")
            # Rollback to clean up any dirty transaction state from tool execution
            if self._db_session:
                try:
                    await self._db_session.rollback()
                    logger.debug("Rolled back db session after agent execution failure")
                except Exception as rollback_err:
                    logger.error("Failed to rollback db session: %s", rollback_err)
            return {
                "error": f"Agent execution failed: {str(e)}",
                "last_assistant_response": "",
            }

    async def _process_response(self, state: TeacherCompanionState) -> dict:
        """Process the final response."""
        response_text = state.get("last_assistant_response", "")

        if not response_text:
            logger.error(
                "Agent failed to generate response for session=%s",
                state.get("session_id"),
            )

        # Add response to conversation history
        history = list(state.get("conversation_history", []))
        history.append(
            TeacherTurn(
                role="assistant",
                content=response_text,
                timestamp=datetime.now().isoformat(),
            )
        )

        logger.info(
            "Processed response: %d chars, %d actions",
            len(response_text),
            len(state.get("pending_actions", [])),
        )

        return {
            "last_assistant_response": response_text,
            "conversation_history": history,
            "awaiting_input": True,
        }

    async def _check_end(self, state: TeacherCompanionState) -> dict:
        """Check if session should end."""
        return {}

    async def _end_session(self, state: TeacherCompanionState) -> dict:
        """End the teacher session."""
        logger.info(
            "Teacher session ended: session=%s, turns=%d",
            state.get("session_id"),
            len(state.get("conversation_history", [])),
        )

        return {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "awaiting_input": False,
        }

    # =========================================================================
    # Routing Functions
    # =========================================================================

    def _should_continue(
        self, state: TeacherCompanionState
    ) -> Literal["continue", "end"]:
        """Determine if workflow should continue or end."""
        if state.get("status") == "error":
            return "end"

        return "continue"
