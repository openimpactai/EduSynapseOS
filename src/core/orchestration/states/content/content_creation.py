# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation workflow state.

This module defines the state structure for content creation workflows.
The state tracks:
- User context and localization preferences
- Curriculum requirements (fetched via curriculum tools)
- Generated content across multiple agents
- Handoff state between orchestrator and generator agents
- Tool call execution state
- Content drafts and exports

All curriculum data (subject, topic, grade, framework, etc.) is populated
dynamically through curriculum tools - never hardcoded.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph.message import add_messages


class ContentTurn(TypedDict, total=False):
    """A single turn in the content creation conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str  # ISO format
    agent_id: str | None  # Which agent generated this


class GeneratedContent(TypedDict, total=False):
    """AI-generated content awaiting review/export."""

    id: str  # Unique content ID
    content_type: str  # H5P content type
    title: str  # Content title
    ai_content: dict[str, Any]  # AI input format content
    h5p_params: dict[str, Any] | None  # Converted H5P params
    status: Literal["draft", "reviewed", "exported", "export_failed"]
    generated_by: str  # Agent that generated this
    quality_score: float | None  # Review score (0-5)
    h5p_id: str | None  # H5P content ID after successful export
    preview_url: str | None  # H5P preview URL after export
    created_at: str  # ISO timestamp
    exported_at: str | None  # ISO timestamp when exported to H5P


class HandoffContext(TypedDict, total=False):
    """Context for agent handoff."""

    source_agent: str  # Agent initiating handoff
    target_agent: str  # Agent to hand off to
    task_type: str  # Type of task (generate, review, translate, etc.)
    content_type: str  # H5P content type
    h5p_library: str  # Full H5P library identifier
    generation_prompt: str  # Prompt for the target agent
    additional_data: dict[str, Any]  # Extra handoff data


class ReviewResult(TypedDict, total=False):
    """Content review result from reviewer agent."""

    pedagogical_score: float
    accuracy_score: float
    appropriateness_score: float
    accessibility_score: float
    engagement_score: float
    overall_score: float
    approved: bool
    critical_issues: list[str]
    improvements: list[dict[str, str]]


class UserMessageInference(TypedDict, total=False):
    """Inference results from user message analysis.

    This structure holds the AI's interpretation of what the user
    wants based on their message content. Used for smart inference
    in user-driven content creation mode.
    """

    content_type: str | None  # Inferred content type (e.g., "multiple-choice")
    wants_media: bool  # Whether user explicitly requested media/images
    media_description: str | None  # User's description of desired media
    confidence: float  # Confidence score (0.0-1.0)


class UserAction(TypedDict, total=False):
    """Classified user action from deterministic intent classification.

    Replaces stochastic tool-calling for state transitions. The orchestrator
    classifies user messages into discrete actions, then deterministic
    if/elif logic sets workflow flags.
    """

    action: Literal[
        "confirm_content_type",
        "request_generation",
        "approve",
        "modify",
        "end",
        "provide_info",
        "unclear",
    ]
    content_type: str | None  # Confirmed or inferred content type
    modification_detail: str | None  # What user wants changed (for "modify")
    confidence: float  # Classification confidence (0.0-1.0)


class ContentCreationState(TypedDict, total=False):
    """State for content creation workflow.

    This state manages the full content creation lifecycle:
    1. Role detection and user context gathering
    2. Requirements gathering via curriculum tools
    3. Content type recommendation
    4. Handoff to specialized generators (quiz, vocabulary, etc.)
    5. Content review and modification
    6. Export to H5P format

    IMPORTANT: All curriculum-related fields (subject_code, topic_code,
    grade_level, framework_code, country_code, learning_objectives)
    are populated dynamically via curriculum tools - NEVER hardcoded.

    The workflow uses interrupt_before pattern for user interactions.
    """

    # === Session Identifiers ===
    session_id: str
    tenant_code: str
    user_id: str

    # === User Context ===
    # These values come from user profile or are detected during conversation
    user_role: Literal["teacher", "student", "parent"] | None
    language: str | None  # Content language (tr, en, es, fr, de, ar, etc.)
    country_code: str | None  # User's country (TR, US, UK, DE, etc.)

    # === Curriculum Context ===
    # All curriculum data is fetched via curriculum tools - NEVER hardcoded
    framework_code: str | None  # Curriculum framework (e.g., "TR-MEB-2024", "UK-NC-2014")
    subject_code: str | None  # Subject code from curriculum API
    subject_name: str | None  # Subject display name
    topic_code: str | None  # Topic code from curriculum API
    topic_name: str | None  # Topic display name
    unit_code: str | None  # Unit code if applicable
    unit_name: str | None  # Unit display name
    grade_level: int | None  # Grade level (1-12), fetched from context
    learning_objectives: list[str]  # Learning objectives from curriculum API

    # === Content Creation Mode ===
    creation_mode: Literal["user_driven", "ai_driven"]  # Content creation mode
    user_inference: UserMessageInference | None  # Extracted from user message
    content_type_confirmed: bool  # Whether content type is confirmed by user

    # === Content Specification ===
    content_types: list[str]  # Selected H5P content types
    difficulty: Literal["easy", "medium", "hard"] | None
    bloom_level: str | None  # Bloom's taxonomy level
    count: int | None  # Number of items to generate
    include_images: bool  # Whether to generate images
    additional_instructions: str | None  # Extra user requirements

    # === Conversation State ===
    messages: Annotated[list[dict], add_messages]  # LangGraph message format
    conversation_history: list[ContentTurn]  # Structured history

    # === Agent State ===
    active_agent: str | None  # Currently active agent
    pending_handoff: str | None  # Target agent for pending handoff
    handoff_context: HandoffContext | None  # Context for handoff

    # === Generated Content ===
    pending_content: dict | None  # Content being generated
    generated_contents: list[GeneratedContent]  # All generated content
    current_content: GeneratedContent | None  # Currently being worked on
    h5p_params: dict | None  # Current H5P formatted content

    # === Review State ===
    review_result: ReviewResult | None
    approved: bool

    # === Export State ===
    exported_content_ids: list[str]  # Content IDs exported to H5P
    draft_ids: list[str]  # Saved draft IDs

    # === Workflow Control ===
    current_phase: Literal[
        "initialization",
        "role_detection",
        "requirements",
        "selection",
        "generation",
        "review",
        "export",
        "complete",
        "error_recovery",
    ]
    requires_input: bool  # Whether waiting for user input
    should_end: bool  # Whether to end the workflow
    error: str | None  # Error message if any

    # === Tool Execution ===
    tool_calls_pending: list[dict[str, Any]]
    tool_calls_completed: list[dict[str, Any]]

    # === Timestamps ===
    created_at: str
    started_at: str
    updated_at: str


def create_initial_content_state(
    session_id: str | None = None,
    tenant_code: str | None = None,
    user_id: str | None = None,
    user_role: Literal["teacher", "student", "parent"] | None = None,
    language: str | None = None,
    country_code: str | None = None,
    framework_code: str | None = None,
    subject_code: str | None = None,
    topic_code: str | None = None,
    grade_level: int | None = None,
) -> ContentCreationState:
    """Create initial state for content creation workflow.

    All parameters are optional. Curriculum-related values (framework_code,
    subject_code, topic_code, grade_level) should be provided from external
    sources (user profile, tenant settings, curriculum API) - NOT hardcoded.

    Args:
        session_id: Unique session identifier. Generated if not provided.
        tenant_code: Tenant identifier for multi-tenancy.
        user_id: User identifier.
        user_role: User's role (teacher, student, parent).
        language: Content language code (tr, en, etc.).
        country_code: User's country code (TR, US, UK, etc.).
        framework_code: Curriculum framework code from tenant/user settings.
        subject_code: Subject code if already known.
        topic_code: Topic code if already known.
        grade_level: Grade level if already known.

    Returns:
        Initial ContentCreationState ready for workflow execution.
    """
    now = datetime.utcnow().isoformat()

    return ContentCreationState(
        # Session identifiers
        session_id=session_id or str(uuid4()),
        tenant_code=tenant_code or "",
        user_id=user_id or "",
        # User context - from user profile or to be detected
        user_role=user_role,
        language=language,
        country_code=country_code,
        # Curriculum context - from external sources, never hardcoded
        framework_code=framework_code,
        subject_code=subject_code,
        subject_name=None,
        topic_code=topic_code,
        topic_name=None,
        unit_code=None,
        unit_name=None,
        grade_level=grade_level,
        learning_objectives=[],
        # Content creation mode
        creation_mode="user_driven",
        user_inference=None,
        content_type_confirmed=False,
        # Content specification - gathered during conversation
        content_types=[],
        difficulty=None,
        bloom_level=None,
        count=None,
        include_images=False,
        additional_instructions=None,
        # Conversation state
        messages=[],
        conversation_history=[],
        # Agent state
        active_agent="content_creation_orchestrator",
        pending_handoff=None,
        handoff_context=None,
        # Generated content
        pending_content=None,
        generated_contents=[],
        current_content=None,
        h5p_params=None,
        # Review state
        review_result=None,
        approved=False,
        # Export state
        exported_content_ids=[],
        draft_ids=[],
        # Workflow control
        current_phase="initialization",
        requires_input=False,
        should_end=False,
        error=None,
        # Tool execution
        tool_calls_pending=[],
        tool_calls_completed=[],
        # Timestamps
        created_at=now,
        started_at=now,
        updated_at=now,
    )
