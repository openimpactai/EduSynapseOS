# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion conversation workflow state.

This module defines the state structure for companion conversation workflows.
The state tracks:
- Pre-loaded memory and emotional context for personalization
- Conversation history
- Tool call execution state
- Actions to return to frontend (navigate, handoff)
- Emotional signals to record

The companion workflow is designed for multi-turn conversations where
the AI companion supports students emotionally, suggests activities,
and hands off academic questions to the tutor.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class CompanionTurn(TypedDict, total=False):
    """A single turn in the companion conversation."""

    role: Literal["student", "companion"]
    content: str
    timestamp: str  # ISO format
    emotional_state: str | None  # Detected emotional state


class ToolCallRecord(TypedDict, total=False):
    """Record of a tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["pending", "executed", "failed"]
    result: dict[str, Any] | None
    error: str | None


class CompanionAction(TypedDict, total=False):
    """An action to be sent to the frontend.

    Enhanced action structure with typed parameters and pre-computed routes.

    Action Types:
    - practice: Navigate to practice session
    - learning: Navigate to learning/tutoring page
    - game: Navigate to game page
    - review: Navigate to spaced repetition review
    - break: Navigate to break/relaxation page
    - creative: Navigate to creative activities
    - navigate: General navigation (dashboard, settings, etc.)
    - handoff: Hand off to another agent (tutor)
    """

    type: Literal["practice", "learning", "game", "review", "break", "creative", "navigate", "handoff"]
    label: str  # Button/action label for UI
    description: str | None  # Optional longer description
    icon: str | None  # Emoji icon for visual display
    params: dict[str, Any]  # Action-specific parameters
    route: str | None  # Pre-computed frontend route
    requires_confirmation: bool  # Whether action needs user confirmation
    target: str | None  # Legacy field for backward compatibility


class EmotionalSignalRecord(TypedDict, total=False):
    """Record of an emotional signal to record.

    Captured from record_emotion tool or message analysis.
    """

    emotion: str
    intensity: str  # low, medium, high
    triggers: list[str]
    context: str | None


class PendingAlertRecord(TypedDict, total=False):
    """Record of a pending proactive alert.

    Loaded from ProactiveService during context loading.
    Used to inform the companion about student situations.

    Uses code-based composite keys from Central Curriculum structure.
    """

    id: str
    alert_type: str  # struggle_detected, engagement_drop, milestone_achieved, etc.
    severity: str  # info, warning, critical
    title: str
    message: str
    topic_full_code: str | None  # Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001")
    created_at: str | None


class CompanionState(TypedDict, total=False):
    """State for companion conversation workflow.

    Manages the state of a multi-turn companion conversation with
    tool calling support and action aggregation.

    The state supports interrupt/resume pattern via _pending_message field,
    matching the Tutoring workflow architecture.

    Attributes:
        # Session Identity
        session_id: Unique session identifier.
        tenant_id: Tenant UUID (for database operations).
        tenant_code: Tenant code (for MemoryManager operations).
        student_id: Student identifier.
        grade_level: Student's grade level (1-12).
        language: Student's language preference (e.g., "en", "tr").

        # Persona
        persona_id: Selected persona for this session.
        persona_name: Display name of the persona.

        # Conversation Context
        conversation_history: Full conversation history.

        # Session Status
        status: Current session status.
        awaiting_input: Whether waiting for student input.

        # Current Turn
        last_student_message: Most recent student message.
        last_companion_response: Most recent companion response.
        first_greeting: Initial proactive greeting from companion.

        # Pending message (for interrupt/resume pattern)
        _pending_message: Message injected via aupdate_state during resume.

        # Tool Calling State
        current_tool_calls: Tool calls from current LLM response.
        tool_results: Results from executed tools.
        tool_call_count: Number of tool call rounds in current turn.

        # Actions and Signals
        pending_actions: Actions to send to frontend (navigate, handoff).
        pending_emotional_signals: Emotional signals to record.

        # UI Elements and Tool Data (for frontend)
        ui_elements: UI elements from tools for structured frontend interactions.
        tool_data: Raw tool data to pass through to frontend response.

        # Full Context (loaded at session start)
        memory_context: Full 4-layer memory context (serialized).
        emotional_context: Current emotional state (serialized).
        pending_alerts: Proactive alerts for the student (from ProactiveService).

        # LangGraph messages
        messages: Message history for LangGraph.

        # Timestamps
        started_at: Session start time.
        last_activity_at: Last interaction time.
        completed_at: Session completion time.

        # Error handling
        error: Error message if workflow failed.
    """

    # Session Identity
    session_id: str
    tenant_id: str
    tenant_code: str
    student_id: str
    student_name: str  # Student's first name for personalization
    grade_level: int
    framework_code: str | None  # Curriculum framework code (e.g., "UK-NC-2014")
    grade_code: str | None  # Grade code within framework (e.g., "Y5", "G5")
    language: str

    # Persona
    persona_id: str
    persona_name: str | None

    # Conversation Context
    conversation_history: list[CompanionTurn]

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error"]
    awaiting_input: bool

    # Current Turn
    last_student_message: str | None
    last_companion_response: str | None
    first_greeting: str | None

    # Pending message data (set via aupdate_state during send_message)
    _pending_message: str | None

    # Tool Calling State
    current_tool_calls: list[ToolCallRecord]
    tool_results: list[dict[str, Any]]  # Tool results to send back to LLM
    tool_call_count: int  # Safety limit for tool call loops

    # Actions and Signals
    pending_actions: list[CompanionAction]
    pending_emotional_signals: list[EmotionalSignalRecord]

    # UI Elements and Tool Data (for frontend)
    ui_elements: list[dict[str, Any]]  # UIElement instances from tools
    tool_data: dict[str, Any]  # Passthrough data from tools for frontend

    # Full Context (loaded at session start, serialized as dict)
    memory_context: dict[str, Any]  # FullMemoryContext.model_dump()
    emotional_context: dict[str, Any] | None  # EmotionalContext.model_dump()
    pending_alerts: list[PendingAlertRecord]  # From ProactiveService

    # LangGraph messages (for multi-turn tool calling)
    messages: Annotated[list[dict[str, Any]], add_messages]

    # Timestamps
    started_at: str
    last_activity_at: str
    completed_at: str | None

    # Error handling
    error: str | None


def create_initial_companion_state(
    session_id: str,
    tenant_id: str,
    tenant_code: str,
    student_id: str,
    grade_level: int = 5,
    framework_code: str | None = None,
    grade_code: str | None = None,
    language: str = "en",
    persona_id: str = "companion",
    student_name: str = "there",
) -> CompanionState:
    """Create initial state for a companion session.

    The initial state has empty context fields that will be populated
    by the load_context node during workflow initialization.

    Args:
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string (for database operations).
        tenant_code: Tenant code for MemoryManager operations.
        student_id: Student identifier.
        grade_level: Student's grade level sequence (1-12).
        framework_code: Curriculum framework code (e.g., "UK-NC-2014").
        grade_code: Grade code within framework (e.g., "Y5", "G5").
        language: Student's language preference.
        persona_id: Persona to use (default: companion).
        student_name: Student's first name for personalization.

    Returns:
        Initial CompanionState ready for workflow execution.
    """
    now = datetime.now().isoformat()

    return CompanionState(
        # Session Identity
        session_id=session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        student_id=student_id,
        student_name=student_name,
        grade_level=grade_level,
        framework_code=framework_code,
        grade_code=grade_code,
        language=language,
        # Persona
        persona_id=persona_id,
        persona_name=None,
        # Conversation Context
        conversation_history=[],
        # Session Status
        status="pending",
        awaiting_input=False,
        # Current Turn
        last_student_message=None,
        last_companion_response=None,
        first_greeting=None,
        # Pending message
        _pending_message=None,
        # Tool Calling State
        current_tool_calls=[],
        tool_results=[],
        tool_call_count=0,
        # Actions and Signals
        pending_actions=[],
        pending_emotional_signals=[],
        # UI Elements and Tool Data (populated by agent node)
        ui_elements=[],
        tool_data={},
        # Context (populated by load_context node)
        memory_context={},
        emotional_context=None,
        pending_alerts=[],
        # Messages
        messages=[],
        # Timestamps
        started_at=now,
        last_activity_at=now,
        completed_at=None,
        # Error
        error=None,
    )
