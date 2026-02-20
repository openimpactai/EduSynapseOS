# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher companion conversation workflow state.

This module defines the state structure for teacher companion conversation workflows.
The state tracks:
- Teacher's class information
- Pending alerts for students in teacher's classes
- Conversation history
- Tool call execution state
- Actions to return to frontend

The teacher companion workflow is designed for multi-turn conversations where
the AI assistant helps teachers monitor their students, view analytics,
and manage alerts.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class TeacherTurn(TypedDict, total=False):
    """A single turn in the teacher conversation."""

    role: Literal["teacher", "assistant"]
    content: str
    timestamp: str  # ISO format


class ToolCallRecord(TypedDict, total=False):
    """Record of a tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["pending", "executed", "failed"]
    result: dict[str, Any] | None
    error: str | None


class TeacherAction(TypedDict, total=False):
    """An action to be sent to the frontend.

    Action Types:
    - view_student: Navigate to student detail page
    - view_class: Navigate to class overview
    - view_analytics: Navigate to analytics dashboard
    - view_alerts: Navigate to alerts page
    - navigate: General navigation
    """

    type: Literal["view_student", "view_class", "view_analytics", "view_alerts", "navigate"]
    label: str  # Button/action label for UI
    description: str | None  # Optional longer description
    icon: str | None  # Emoji icon for visual display
    params: dict[str, Any]  # Action-specific parameters
    route: str | None  # Pre-computed frontend route


class ClassSummary(TypedDict, total=False):
    """Summary of a class the teacher is assigned to."""

    class_id: str
    class_name: str
    student_count: int
    subject_name: str | None
    is_homeroom: bool


class AlertSummary(TypedDict, total=False):
    """Summary of pending alerts for the teacher's students."""

    total_count: int
    critical_count: int
    warning_count: int
    info_count: int
    recent_alerts: list[dict[str, Any]]  # Last 5 alerts


class TeacherCompanionState(TypedDict, total=False):
    """State for teacher companion conversation workflow.

    Manages the state of a multi-turn teacher assistant conversation with
    tool calling support for class management and student monitoring.

    Attributes:
        # Session Identity
        session_id: Unique session identifier.
        tenant_id: Tenant UUID (for database operations).
        tenant_code: Tenant code.
        teacher_id: Teacher identifier.
        language: Teacher's language preference.

        # Persona
        persona_id: Selected persona for this session.
        persona_name: Display name of the persona.

        # Conversation Context
        conversation_history: Full conversation history.

        # Session Status
        status: Current session status.
        awaiting_input: Whether waiting for teacher input.

        # Current Turn
        last_teacher_message: Most recent teacher message.
        last_assistant_response: Most recent assistant response.
        first_greeting: Initial proactive greeting from assistant.

        # Pending message (for interrupt/resume pattern)
        _pending_message: Message injected via aupdate_state during resume.

        # Tool Calling State
        current_tool_calls: Tool calls from current LLM response.
        tool_results: Results from executed tools.
        tool_call_count: Number of tool call rounds in current turn.

        # Actions
        pending_actions: Actions to send to frontend.

        # UI Elements and Tool Data (for frontend)
        ui_elements: UI elements from tools for structured frontend interactions.
        tool_data: Raw tool data to pass through to frontend response.

        # Teacher Context (loaded at session start)
        class_summary: List of classes the teacher is assigned to.
        alert_summary: Summary of pending alerts for teacher's students.

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
    conversation_id: str
    tenant_id: str
    tenant_code: str
    teacher_id: str
    language: str

    # Persona
    persona_id: str
    persona_name: str | None

    # Conversation Context
    conversation_history: list[TeacherTurn]

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error"]
    awaiting_input: bool

    # Current Turn
    last_teacher_message: str | None
    last_assistant_response: str | None
    first_greeting: str | None

    # Pending message data (set via aupdate_state during send_message)
    _pending_message: str | None

    # Tool Calling State
    current_tool_calls: list[ToolCallRecord]
    tool_results: list[dict[str, Any]]
    tool_call_count: int

    # Actions
    pending_actions: list[TeacherAction]

    # UI Elements and Tool Data (for frontend)
    ui_elements: list[dict[str, Any]]
    tool_data: dict[str, Any]

    # Teacher Context (loaded at session start)
    class_summary: list[ClassSummary]
    alert_summary: AlertSummary | None

    # LangGraph messages (for multi-turn tool calling)
    messages: Annotated[list[dict[str, Any]], add_messages]

    # Timestamps
    started_at: str
    last_activity_at: str
    completed_at: str | None

    # Error handling
    error: str | None


def create_initial_teacher_companion_state(
    session_id: str,
    conversation_id: str,
    tenant_id: str,
    tenant_code: str,
    teacher_id: str,
    language: str = "en",
    persona_id: str = "teacher_assistant",
    class_summary: list[ClassSummary] | None = None,
    alert_summary: AlertSummary | None = None,
) -> TeacherCompanionState:
    """Create initial state for a teacher companion session.

    Context (class_summary, alert_summary) should be loaded by the service
    layer before calling workflow.run() to ensure proper transaction isolation.

    Args:
        session_id: Unique session identifier.
        conversation_id: Conversation ID for message persistence.
        tenant_id: Tenant UUID as string (for database operations).
        tenant_code: Tenant code.
        teacher_id: Teacher identifier.
        language: Teacher's language preference.
        persona_id: Persona to use (default: teacher_assistant).
        class_summary: Pre-loaded class summary from service layer.
        alert_summary: Pre-loaded alert summary from service layer.

    Returns:
        Initial TeacherCompanionState ready for workflow execution.
    """
    now = datetime.now().isoformat()

    return TeacherCompanionState(
        # Session Identity
        session_id=session_id,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        teacher_id=teacher_id,
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
        last_teacher_message=None,
        last_assistant_response=None,
        first_greeting=None,
        # Pending message
        _pending_message=None,
        # Tool Calling State
        current_tool_calls=[],
        tool_results=[],
        tool_call_count=0,
        # Actions
        pending_actions=[],
        # UI Elements and Tool Data
        ui_elements=[],
        tool_data={},
        # Teacher Context (pre-loaded by service layer for transaction isolation)
        class_summary=class_summary or [],
        alert_summary=alert_summary,
        # Messages
        messages=[],
        # Timestamps
        started_at=now,
        last_activity_at=now,
        completed_at=None,
        # Error
        error=None,
    )
