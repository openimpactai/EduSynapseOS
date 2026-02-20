# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation API schemas.

This module defines request/response schemas for the content creation API.
The content creation system uses a conversational interface with specialized
agents for generating H5P educational content.

Content Types:
    - Assessment: Multiple choice, true/false, fill blanks, question sets
    - Vocabulary: Flashcards, dialog cards, crossword, word search
    - Learning: Course presentation, interactive book, branching scenario
    - Game: Memory game, timeline, image sequencing
    - Media: Charts, image hotspots
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentChatRequest(BaseModel):
    """Request for content creation chat interaction.

    Supports both starting new content creation sessions and
    continuing existing conversations.

    For new sessions: session_id is None, message describes what to create.
    For continuing: session_id is provided, message continues the conversation.
    """

    session_id: str | None = Field(
        default=None,
        description="Session ID for continuing conversation. None for new session.",
    )
    message: str = Field(
        description="User message describing what content to create or response to agent.",
    )
    language: str | None = Field(
        default=None,
        description="Language code for content generation. If None, detected from user message or context.",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context: user_role, country_code, framework_code, subject_code, topic_code, grade_level, etc.",
    )


class ContentTypeInfo(BaseModel):
    """Information about an H5P content type."""

    content_type: str = Field(
        description="Content type identifier (e.g., 'multiple-choice').",
    )
    library: str = Field(
        description="H5P library identifier (e.g., 'H5P.MultiChoice 1.16').",
    )
    name: str = Field(
        description="Display name for the content type.",
    )
    description: str = Field(
        description="Description of what this content type does.",
    )
    category: str = Field(
        description="Category (assessment, vocabulary, learning, game, media).",
    )
    ai_support: Literal["full", "partial", "none"] = Field(
        description="Level of AI generation support.",
    )
    bloom_levels: list[str] = Field(
        default_factory=list,
        description="Bloom's taxonomy levels this content supports.",
    )
    requires_media: bool = Field(
        default=False,
        description="Whether this content type requires media assets.",
    )


class GeneratedContentResponse(BaseModel):
    """Response for a generated content item."""

    id: str = Field(
        description="Unique identifier for this generated content.",
    )
    content_type: str = Field(
        description="H5P content type identifier.",
    )
    title: str = Field(
        description="Content title.",
    )
    status: Literal["draft", "reviewed", "exported", "export_failed"] = Field(
        description="Current status of the content: draft (generated), reviewed (passed QA), exported (sent to H5P), export_failed (H5P export failed).",
    )
    preview_url: str | None = Field(
        default=None,
        description="URL to preview the content (if available).",
    )
    h5p_id: str | None = Field(
        default=None,
        description="H5P content ID after export.",
    )
    ai_content: dict[str, Any] = Field(
        default_factory=dict,
        description="AI-generated content in input format.",
    )
    quality_score: float | None = Field(
        default=None,
        description="Quality score from content review (0-100).",
    )
    created_at: datetime = Field(
        description="When the content was generated.",
    )
    exported_at: datetime | None = Field(
        default=None,
        description="When the content was exported to H5P.",
    )


class ContentChatResponse(BaseModel):
    """Response from content creation chat interaction.

    Contains the assistant's message and any generated content.
    """

    session_id: str = Field(
        description="Session ID for continuing conversation.",
    )
    message: str = Field(
        description="Assistant's response message.",
    )
    workflow_phase: Literal[
        "gathering_requirements",
        "awaiting_confirmation",
        "generating",
        "reviewing",
        "exporting",
        "completed",
    ] = Field(
        description="Current phase in the content creation workflow.",
    )
    current_agent: str | None = Field(
        default=None,
        description="Active agent handling the conversation.",
    )
    generated_content: GeneratedContentResponse | None = Field(
        default=None,
        description="Currently generated content (if any).",
    )
    recommended_types: list[ContentTypeInfo] = Field(
        default_factory=list,
        description="Recommended content types based on requirements.",
    )
    suggestions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="UI elements for structured suggestions.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tool calls, timing, etc.).",
    )


class ContentSessionResponse(BaseModel):
    """Response for session info."""

    session_id: str = Field(
        description="Session identifier.",
    )
    thread_id: str = Field(
        description="LangGraph thread ID for state persistence.",
    )
    status: Literal["active", "completed", "expired"] = Field(
        description="Session status.",
    )
    workflow_phase: str = Field(
        description="Current workflow phase.",
    )
    message_count: int = Field(
        description="Number of messages in session.",
    )
    generated_count: int = Field(
        description="Number of content items generated.",
    )
    exported_count: int = Field(
        description="Number of content items exported.",
    )
    language: str = Field(
        description="Session language.",
    )
    started_at: datetime = Field(
        description="When the session started.",
    )
    last_message_at: datetime | None = Field(
        default=None,
        description="When the last message was sent.",
    )


class ContentMessageResponse(BaseModel):
    """Response for a single message."""

    id: str = Field(
        description="Message identifier.",
    )
    role: Literal["user", "assistant"] = Field(
        description="Message role.",
    )
    content: str = Field(
        description="Message content.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent ID for assistant messages.",
    )
    generated_content_id: str | None = Field(
        default=None,
        description="Associated generated content ID.",
    )
    created_at: datetime = Field(
        description="When the message was created.",
    )


class ContentMessagesResponse(BaseModel):
    """Response for message history."""

    messages: list[ContentMessageResponse] = Field(
        description="List of messages.",
    )
    has_more: bool = Field(
        default=False,
        description="Whether there are more messages.",
    )


class BatchContentRequest(BaseModel):
    """Request for batch content generation."""

    content_requests: list[dict[str, Any]] = Field(
        description="List of content generation requests.",
        min_length=1,
        max_length=10,
    )
    language: str | None = Field(
        default=None,
        description="Language code for all content. If None, uses context or detects from content.",
    )
    auto_export: bool = Field(
        default=False,
        description="Automatically export generated content.",
    )


class BatchContentResponse(BaseModel):
    """Response for batch content generation."""

    batch_id: str = Field(
        description="Batch operation identifier.",
    )
    total: int = Field(
        description="Total number of content items requested.",
    )
    completed: int = Field(
        description="Number of items completed.",
    )
    failed: int = Field(
        description="Number of items failed.",
    )
    results: list[GeneratedContentResponse] = Field(
        default_factory=list,
        description="Generated content items.",
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Error details for failed items.",
    )


class ContentExportRequest(BaseModel):
    """Request to export content to H5P."""

    content_id: str = Field(
        description="ID of the generated content to export.",
    )
    folder_id: str | None = Field(
        default=None,
        description="Target folder in H5P library.",
    )
    publish: bool = Field(
        default=False,
        description="Publish content after export.",
    )


class ContentExportResponse(BaseModel):
    """Response for content export."""

    content_id: str = Field(
        description="Generated content ID.",
    )
    h5p_id: str = Field(
        description="H5P content ID.",
    )
    library: str = Field(
        description="H5P library used.",
    )
    preview_url: str = Field(
        description="URL to preview the content.",
    )
    embed_code: str | None = Field(
        default=None,
        description="HTML embed code for the content.",
    )
    exported_at: datetime = Field(
        description="When the content was exported.",
    )


class ContentTypesResponse(BaseModel):
    """Response for listing content types."""

    content_types: list[ContentTypeInfo] = Field(
        description="Available content types.",
    )
    categories: list[str] = Field(
        description="Available categories.",
    )
    total: int = Field(
        description="Total number of content types.",
    )
