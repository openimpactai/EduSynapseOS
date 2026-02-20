# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System Explainer schemas for request/response models.

This module defines the Pydantic models for the System Explainer API,
which allows anyone to learn about EduSynapseOS through an AI agent.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class AudienceType(str, Enum):
    """Target audience types for response adaptation."""

    INVESTOR = "investor"
    EDUCATOR = "educator"
    TECHNICAL = "technical"
    GENERAL = "general"
    AUTO = "auto"  # Let the agent detect from context


class ExplainerChatRequest(BaseModel):
    """Request model for system explainer chat.

    Attributes:
        message: The user's question or message.
        session_id: Optional session ID for conversation continuity.
        audience: Target audience for response adaptation.
        language: Preferred response language (en/tr).
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question about EduSynapseOS",
        examples=["What is EduSynapseOS?", "How does the memory system work?"],
    )
    session_id: UUID | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )
    audience: AudienceType = Field(
        default=AudienceType.AUTO,
        description="Target audience for response style",
    )
    language: str = Field(
        default="en",
        pattern="^(en|tr)$",
        description="Response language",
    )


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplainerChatResponse(BaseModel):
    """Response model for system explainer chat.

    Attributes:
        message: The agent's response.
        session_id: Session ID for continuing the conversation.
        audience_detected: The detected audience type.
        suggested_topics: Topics the user might want to explore next.
        metadata: Additional response metadata.
    """

    message: str = Field(
        ...,
        description="Agent's response explaining EduSynapseOS",
    )
    session_id: UUID = Field(
        ...,
        description="Session ID for conversation continuity",
    )
    audience_detected: AudienceType = Field(
        default=AudienceType.GENERAL,
        description="Detected or specified audience type",
    )
    suggested_topics: list[str] = Field(
        default_factory=list,
        description="Related topics to explore",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class ExplainerSessionResponse(BaseModel):
    """Response model for session information.

    Attributes:
        session_id: The session identifier.
        created_at: When the session was created.
        message_count: Number of messages in the session.
        last_activity: Last activity timestamp.
    """

    session_id: UUID
    created_at: datetime
    message_count: int
    last_activity: datetime


class ExplainerTopicsResponse(BaseModel):
    """Response model for available topics.

    Lists the main topics that can be explained.
    """

    topics: list[dict[str, str]] = Field(
        ...,
        description="Available topics with id and description",
    )


class QuickExplainRequest(BaseModel):
    """Request for quick one-shot explanation without session.

    Used for simple questions that don't need conversation context.
    """

    topic: str = Field(
        ...,
        description="Topic to explain",
        examples=[
            "memory_system",
            "educational_theories",
            "agent_system",
            "business_model",
            "technical_architecture",
            "elevator_pitch",
        ],
    )
    audience: AudienceType = Field(
        default=AudienceType.GENERAL,
        description="Target audience",
    )
    language: str = Field(
        default="en",
        pattern="^(en|tr)$",
        description="Response language",
    )


class QuickExplainResponse(BaseModel):
    """Response for quick explanation."""

    topic: str
    explanation: str
    related_topics: list[str] = Field(default_factory=list)
