# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion API schemas.

This module defines request/response schemas for the companion chat API.
The companion uses a unified chat endpoint with tool calling for
all interactions including greetings, emotional support, and activity guidance.

The companion acts as a Navigator/Orchestrator, guiding students to
appropriate activities without teaching directly. It uses intelligent
tool chaining to gather information and present options to students.

UI Elements:
    When tools return UI elements, they are included in the response
    for the frontend to render structured selection controls (dropdowns,
    checkboxes, etc.) instead of relying on free-text input.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.tools.ui_elements import UIElement


class CompanionChatRequest(BaseModel):
    """Request for companion chat interaction.

    Supports both proactive (companion-initiated) and reactive
    (student-initiated) interactions through a unified interface.

    For new sessions: session_id is None, message is optional.
    For continuing: session_id is provided, message is required.
    """

    session_id: str | None = Field(
        default=None,
        description="Session ID for continuing conversation. None for new session.",
    )
    message: str | None = Field(
        default=None,
        description="Student message. Required for continuing, optional for new session.",
    )
    trigger: Literal[
        "checkin",
        "user_message",
        "idle",
        "milestone",
        "page_view",
        "error",
    ] = Field(
        default="checkin",
        description=(
            "What triggered this interaction:\n"
            "- checkin: New session greeting\n"
            "- user_message: Student sent a message\n"
            "- idle: Student idle for a while\n"
            "- milestone: Achievement to celebrate\n"
            "- page_view: Student viewed a page\n"
            "- error: Student encountered an error"
        ),
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context (current_page, event_data, etc.)",
    )


class EmotionalState(BaseModel):
    """Detected emotional state from conversation."""

    emotion: str = Field(
        description="Detected emotion (happy, frustrated, anxious, etc.)",
    )
    intensity: str = Field(
        description="Intensity level (low, medium, high)",
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="What triggered this emotion",
    )


class CompanionChatResponse(BaseModel):
    """Response from companion chat interaction.

    Contains the companion's message and optional UI elements for
    structured interactions.

    UI Elements:
        When tools return UI elements (e.g., subject/topic selection),
        they are included here for the frontend to render proper
        selection controls instead of relying on free-text input.

    Tool Data:
        Raw data from tools that the frontend may need for display
        or further processing (e.g., list of subjects with full details).

    Handoff:
        When the companion hands off to another agent (e.g., tutor),
        the handoff information is included here.
    """

    session_id: str = Field(
        description="Session ID for continuing conversation",
    )
    message: str = Field(
        description="Companion's response message",
    )
    suggestions: list[UIElement] = Field(
        default_factory=list,
        description="UI elements for structured suggestions (selections, options)",
    )
    tool_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw tool data for frontend (subjects, topics, etc.)",
    )
    handoff: dict[str, Any] | None = Field(
        default=None,
        description="Handoff information when transferring to another agent",
    )
    emotional_state: EmotionalState | None = Field(
        default=None,
        description="Detected emotional state from student's message",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool_calls count, etc.)",
    )


class CompanionSessionResponse(BaseModel):
    """Response for session info."""

    session_id: str
    conversation_id: str
    status: str
    message_count: int
    started_at: datetime
    last_message_at: datetime | None


class CompanionMessageResponse(BaseModel):
    """Response for a single message."""

    id: str
    role: Literal["student", "companion"]
    content: str
    emotional_context: dict[str, Any] | None = None
    created_at: datetime


class CompanionMessagesResponse(BaseModel):
    """Response for message history."""

    messages: list[CompanionMessageResponse]
    has_more: bool
