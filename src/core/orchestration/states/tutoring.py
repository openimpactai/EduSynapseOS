# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tutoring conversation workflow state.

This module defines the state structure for tutoring conversation workflows.
The state tracks:
- Full 4-layer memory context for personalization
- Theory recommendations (ZPD, VARK, Scaffolding, Socratic, etc.)
- Conversation history and understanding tracking
- Emotional context and analysis
- Pending message for interrupt/resume pattern

The tutoring workflow is designed for multi-turn conversations where
the AI tutor explains concepts, answers questions, and adapts to
student needs using educational best practices.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class ConversationTurn(TypedDict, total=False):
    """A single turn in the tutoring conversation."""

    role: Literal["student", "tutor"]
    content: str
    timestamp: str  # ISO format
    intent: str | None  # Detected intent: question, clarification, acknowledgment
    concepts_mentioned: list[str]
    emotional_state: str | None  # Detected emotional state


class ExplanationRecord(TypedDict, total=False):
    """Record of a concept explanation with understanding tracking."""

    concept: str
    explanation: str
    bloom_level: str  # remember, understand, apply, analyze, evaluate, create
    scaffold_level: str  # minimal, moderate, maximum
    examples_provided: list[str]
    analogies_used: list[str]
    student_understood: bool | None  # None = not yet determined
    follow_up_questions: list[str]
    mastery_delta: float  # Change in mastery after explanation


class TutoringMetrics(TypedDict, total=False):
    """Comprehensive metrics for the tutoring session."""

    turns_count: int
    student_questions: int
    explanations_given: int
    concepts_covered: list[str]
    clarifications_requested: int
    total_duration_seconds: float
    understanding_signals: int  # Positive signals from student
    confusion_signals: int  # Negative/confusion signals
    support_interventions: int  # Times we provided emotional support


class MessageAnalysis(TypedDict, total=False):
    """Result of message analysis from emotional_analyzer agent."""

    intent: str  # question, clarification, acknowledgment, farewell, etc.
    intent_confidence: float
    emotional_state: str  # engaged, confused, frustrated, curious, etc.
    intensity: str  # low, medium, high
    sentiment_confidence: float
    triggers: list[str]
    requires_support: bool
    suggested_response_tone: str
    understanding_signal: bool | None  # Did student show understanding?


class TutoringState(TypedDict, total=False):
    """State for tutoring conversation workflow.

    Manages the state of a multi-turn tutoring conversation with full
    personalization support through 4-layer memory and educational theory
    integration.

    The state supports interrupt/resume pattern via _pending_message field,
    matching the Practice workflow architecture.

    Attributes:
        # Session Identity
        session_id: Unique session identifier.
        tenant_id: Tenant UUID (for database operations).
        tenant_code: Tenant code (for MemoryManager, RAG, theory).
        student_id: Student identifier.
        persona_id: Selected persona for this session.

        # Conversation Context
        topic: Main topic being discussed.
        subtopic: Current subtopic focus.
        conversation_history: Full conversation history.
        current_focus: What we're currently explaining/discussing.

        # Session Status
        status: Current session status.
        awaiting_input: Whether waiting for student input.

        # Understanding Tracking
        concepts_explained: Concepts we've explained with mastery tracking.
        concepts_student_knows: Concepts student demonstrates understanding.
        concepts_struggling: Concepts student is struggling with.

        # Current Turn
        last_student_message: Most recent student message.
        last_tutor_response: Most recent tutor response.
        first_greeting: Initial proactive greeting from tutor.

        # Pending message (for interrupt/resume pattern)
        _pending_message: Message injected via aupdate_state during resume.

        # Message Analysis
        last_message_analysis: Emotional/intent analysis of last message.

        # Metrics
        metrics: Comprehensive session metrics.

        # Full Context (loaded at session start)
        memory_context: Full 4-layer memory context (episodic, semantic,
                        procedural, associative).
        theory_recommendations: Combined recommendations from all 7 theories
                               (ZPD, Bloom, VARK, Scaffolding, Mastery,
                               Socratic, Spaced Repetition).
        emotional_context: Current emotional state from EmotionalStateService.
        rag_context: Retrieved context for current response.

        # Memory Update Tracking
        mastery_updates: Mastery changes to apply after session.
        concepts_for_review: Concepts to schedule for spaced repetition.

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
    persona_id: str

    # Conversation Context
    topic: str
    subtopic: str | None
    conversation_history: list[ConversationTurn]
    current_focus: str | None

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error"]
    awaiting_input: bool

    # Understanding Tracking
    concepts_explained: list[ExplanationRecord]
    concepts_student_knows: list[str]
    concepts_struggling: list[str]

    # Current Turn
    last_student_message: str | None
    last_tutor_response: str | None
    first_greeting: str | None  # Proactive greeting generated at start

    # Pending message data (set via aupdate_state during send_message)
    _pending_message: str | None

    # Message Analysis (from emotional_analyzer agent)
    last_message_analysis: MessageAnalysis | None

    # Metrics
    metrics: TutoringMetrics

    # Full Context (loaded at session start, refreshed periodically)
    memory_context: dict[str, Any]  # FullMemoryContext.model_dump()
    theory_recommendations: dict[str, Any]  # CombinedRecommendation.model_dump()
    emotional_context: dict[str, Any] | None  # EmotionalContext from service
    rag_context: list[dict[str, Any]]

    # Memory Update Tracking
    mastery_updates: dict[str, float]  # concept -> delta
    concepts_for_review: list[str]  # Concepts to schedule for FSRS

    # LangGraph messages
    messages: Annotated[list[dict[str, str]], add_messages]

    # Timestamps
    started_at: str
    last_activity_at: str
    completed_at: str | None

    # Error handling
    error: str | None


def create_initial_tutoring_state(
    session_id: str,
    tenant_id: str,
    tenant_code: str,
    student_id: str,
    topic: str,
    subtopic: str | None = None,
    persona_id: str = "tutor",
) -> TutoringState:
    """Create initial state for a tutoring session.

    The initial state has empty context fields that will be populated
    by the load_context node during workflow initialization.

    Args:
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string (for database operations).
        tenant_code: Tenant code for MemoryManager, RAG, and theory operations.
        student_id: Student identifier.
        topic: Main topic to discuss.
        subtopic: Optional subtopic focus.
        persona_id: Persona to use (default: tutor).

    Returns:
        Initial TutoringState ready for workflow execution.
    """
    now = datetime.now().isoformat()

    return TutoringState(
        # Session Identity
        session_id=session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        student_id=student_id,
        persona_id=persona_id,
        # Conversation Context
        topic=topic,
        subtopic=subtopic,
        conversation_history=[],
        current_focus=subtopic or topic,
        # Session Status
        status="pending",
        awaiting_input=False,
        # Understanding Tracking
        concepts_explained=[],
        concepts_student_knows=[],
        concepts_struggling=[],
        # Current Turn
        last_student_message=None,
        last_tutor_response=None,
        first_greeting=None,
        # Pending message
        _pending_message=None,
        # Message Analysis
        last_message_analysis=None,
        # Metrics
        metrics=TutoringMetrics(
            turns_count=0,
            student_questions=0,
            explanations_given=0,
            concepts_covered=[],
            clarifications_requested=0,
            total_duration_seconds=0.0,
            understanding_signals=0,
            confusion_signals=0,
            support_interventions=0,
        ),
        # Context (populated by load_context node)
        memory_context={},
        theory_recommendations={},
        emotional_context=None,
        rag_context=[],
        # Memory Update Tracking
        mastery_updates={},
        concepts_for_review=[],
        # Messages
        messages=[],
        # Timestamps
        started_at=now,
        last_activity_at=now,
        completed_at=None,
        # Error
        error=None,
    )
