# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher API schemas.

This module defines request/response schemas for the teacher chat API.
The teacher assistant uses a chat endpoint with tool calling for
class management, student monitoring, and analytics viewing.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.tools.ui_elements import UIElement


class TeacherChatRequest(BaseModel):
    """Request for teacher chat interaction.

    Supports both new sessions and continuing conversations.

    For new sessions: session_id is None.
    For continuing: session_id is provided, message is required.
    """

    session_id: str | None = Field(
        default=None,
        description="Session ID for continuing conversation. None for new session.",
    )
    message: str | None = Field(
        default=None,
        description="Teacher message. Required for continuing, optional for new session.",
    )
    trigger: Literal[
        "checkin",
        "user_message",
        "alert_view",
        "student_select",
    ] = Field(
        default="checkin",
        description=(
            "What triggered this interaction:\n"
            "- checkin: New session greeting\n"
            "- user_message: Teacher sent a message\n"
            "- alert_view: Teacher clicked on an alert\n"
            "- student_select: Teacher selected a student"
        ),
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context (selected_student_id, selected_class_id, etc.)",
    )


class TeacherChatResponse(BaseModel):
    """Response from teacher chat interaction.

    Contains the assistant's message and optional UI elements for
    structured interactions.
    """

    session_id: str = Field(
        description="Session ID for continuing conversation",
    )
    message: str = Field(
        description="Assistant's response message",
    )
    suggestions: list[UIElement] = Field(
        default_factory=list,
        description="UI elements for structured suggestions (selections, options)",
    )
    tool_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw tool data for frontend (classes, students, alerts, etc.)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool_calls count, etc.)",
    )


class TeacherSessionResponse(BaseModel):
    """Response for session info."""

    session_id: str
    conversation_id: str
    status: str
    message_count: int
    started_at: datetime
    last_message_at: datetime | None


class TeacherMessageResponse(BaseModel):
    """Response for a single message."""

    id: str
    role: Literal["teacher", "assistant"]
    content: str
    created_at: datetime


class TeacherMessagesResponse(BaseModel):
    """Response for message history."""

    messages: list[TeacherMessageResponse]
    has_more: bool
