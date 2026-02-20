# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice session workflow state.

This module defines the state structure for practice session workflows.
The state tracks:
- Session configuration (topic, difficulty, question count, mode)
- Mode configuration (hints, skips, time limits, adaptive difficulty)
- Current question and progress
- Student answers and evaluations
- Performance metrics
- Persona and memory context

The state is persisted via LangGraph checkpointing, allowing sessions
to be interrupted and resumed.

Topic and subject references use code-based composite keys from
the Central Curriculum structure:
- topic_full_code: e.g., "UK-NC-2014.MAT.Y4.NPV.001"
- subject_full_code: e.g., "UK-NC-2014.MAT"
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from src.core.agents.capabilities.question_generation import GeneratedQuestion
from src.core.agents.capabilities.answer_evaluation import AnswerEvaluationResult
from src.core.agents.capabilities.feedback_generation import GeneratedFeedback


class QuestionRecord(TypedDict, total=False):
    """Record of a question and its evaluation."""

    question_id: str
    question: GeneratedQuestion
    student_answer: str | None
    evaluation: AnswerEvaluationResult | None
    feedback: GeneratedFeedback | None
    time_taken_seconds: float | None
    hints_used: int


class SessionMetrics(TypedDict, total=False):
    """Aggregated metrics for the session."""

    questions_answered: int
    questions_correct: int
    questions_partial: int
    questions_incorrect: int
    total_time_seconds: float
    average_time_per_question: float
    accuracy: float
    streak_current: int
    streak_max: int
    hints_total: int
    skips_total: int


class HandoffContext(TypedDict, total=False):
    """Context passed during workflow handoffs.

    This structure captures context from the source workflow to enable
    seamless transitions and informed question selection.

    Attributes:
        source: Source workflow (learning_tutor, practice_helper, companion).
        topic_code: Topic being worked on.
        session_id: Source session ID for linking.

        # From Learning Tutor
        verified_understanding: Understanding score (0.0-1.0) if verified.
        concepts_verified: List of concepts the student understood.
        concepts_weak: List of concepts needing more practice.
        misconceptions_addressed: List of misconceptions addressed in learning.
        preferred_learning_style: Effective teaching approach identified.

        # From Practice Helper
        understood: Whether student understood after help.
        mode_that_helped: Which tutoring mode was effective.
        escalations_needed: Number of mode escalations needed.
        question_id: Question ID that can be retried.
        practice_session_id: Original practice session to return to.
        helper_session_id: Practice Helper session ID.

        # From Companion
        emotional_state: Emotional context from companion.
        conversation_context: Recent conversation turns.
        student_request: What the student asked for.
        suggested_mode: Suggested learning/practice mode.
    """

    source: str
    topic_code: str | None
    session_id: str | None

    # From Learning Tutor
    verified_understanding: float | None
    concepts_verified: list[str]
    concepts_weak: list[str]
    misconceptions_addressed: list[str]
    preferred_learning_style: str | None

    # From Practice Helper
    understood: bool | None
    mode_that_helped: str | None
    escalations_needed: int | None
    question_id: str | None
    practice_session_id: str | None
    helper_session_id: str | None

    # From Companion
    emotional_state: dict[str, Any] | None
    conversation_context: list[dict[str, Any]]
    student_request: str | None
    suggested_mode: str | None


class HelperIntervention(TypedDict, total=False):
    """Record of a Practice Helper intervention."""

    question_id: str
    mode_that_helped: str | None
    escalations: int


class PracticeState(TypedDict, total=False):
    """State for practice session workflow.

    This state is checkpointed after each node execution, allowing
    the session to be interrupted (e.g., waiting for student answer)
    and resumed later.

    Attributes:
        # Session Identity
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string (for database operations).
        tenant_code: Tenant code string (for MemoryManager, RAG, etc.).
        student_id: Student identifier.
        persona_id: Selected persona for this session.

        # Session Configuration
        topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
        topic: Topic name being practiced.
        learning_objective_full_code: Learning objective full code (optional).
        difficulty: Target difficulty (0.0-1.0).
        target_question_count: How many questions to ask.
        mode: Practice mode (quick, deep, mixed, exam_prep, weakness_focus).
        mode_config: Mode configuration dict with hints, skips, time limits, etc.

        # Handoff Context (from other workflows)
        handoff_context: Context passed from source workflow.
        priority_concepts: Concepts to prioritize in question selection.
        from_learning_tutor: Whether session started from Learning Tutor.
        learning_session_id: Linked Learning Tutor session ID.
        initial_difficulty: Difficulty adjusted based on handoff context.

        # Practice Helper Return Context
        _offer_retry: Whether to offer retry of last question.
        _retry_question_id: Question ID to retry after helper.
        _helper_interventions: Record of Practice Helper interventions.

        # Performance Tracking
        consecutive_wrong: Count of consecutive wrong answers.

        # Educational Context (from curriculum hierarchy)
        subject_full_code: Subject full code (e.g., "UK-NC-2014.MAT").
        subject_name: Subject display name (e.g., "Mathematics").
        grade_level: Grade level display name (e.g., "Year 4 (KS2)").
        grade_level_code: Grade level code (e.g., "year_4").
        age_range: Age range string (e.g., "8-9 years").
        curriculum: Curriculum display name (e.g., "UK National Curriculum - Primary").
        curriculum_code: Curriculum code (e.g., "uk_national_primary").
        language: Language code derived from curriculum (e.g., "en").
        unit_name: Unit name (e.g., "Number and Place Value").

        # Time Tracking
        time_limit_seconds: Total time limit in seconds (None = unlimited).
        time_remaining_seconds: Remaining time in seconds.

        # Skip Tracking
        skips_used: Number of questions skipped.
        max_skips: Maximum allowed skips (None = unlimited).

        # Progress
        status: Current session status.
        current_question_index: Index of current question (0-based).
        questions: List of question records.

        # Current State
        current_question: The current question being answered.
        awaiting_answer: Whether waiting for student input.

        # Metrics
        metrics: Aggregated session metrics.

        # Context
        memory_context: Memory context for personalization.
        theory_recommendations: Theory-based recommendations.
        messages: Conversation messages for context.

        # Timestamps
        started_at: Session start time.
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

    # Session Configuration
    topic_full_code: str | None  # Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001")
    topic: str  # Topic name for display and LLM prompts
    learning_objective_full_code: str | None  # Learning objective full code
    learning_objective_text: str | None  # LO objective text for LLM prompts
    difficulty: float
    target_question_count: int
    mode: str
    mode_config: dict[str, Any]
    question_type: str | None

    # Handoff Context (from other workflows)
    handoff_context: HandoffContext | None
    priority_concepts: list[str]  # Concepts to prioritize in question selection
    from_learning_tutor: bool  # Whether session started from Learning Tutor
    learning_session_id: str | None  # Linked Learning Tutor session ID
    initial_difficulty: float | None  # Difficulty adjusted based on handoff context

    # Practice Helper Return Context
    _offer_retry: bool  # Whether to offer retry of last question
    _retry_question_id: str | None  # Question ID to retry after helper
    _helper_interventions: list[HelperIntervention]  # Record of Practice Helper interventions

    # Performance Tracking
    consecutive_wrong: int  # Count of consecutive wrong answers

    # Random Mode Configuration (for mixed topics from subject)
    subject_full_code: str | None  # Subject full code for random mode (e.g., "UK-NC-2014.MAT")
    available_topics: list[dict[str, Any]]  # List of topics for random selection
    last_topic_full_code: str | None  # Last used topic full code (to avoid consecutive same topic)

    # Educational Context (from curriculum hierarchy)
    # These fields provide context for age-appropriate content generation
    subject_name: str | None  # e.g., "Mathematics"
    grade_level: str | None  # e.g., "Year 4 (KS2)"
    grade_level_code: str | None  # e.g., "year_4"
    age_range: str | None  # e.g., "8-9 years"
    curriculum: str | None  # e.g., "UK National Curriculum - Primary"
    curriculum_code: str | None  # e.g., "uk_national_primary"
    language: str  # e.g., "en" - derived from curriculum country code
    unit_name: str | None  # e.g., "Number and Place Value"

    # Time Tracking
    time_limit_seconds: int | None
    time_remaining_seconds: int | None

    # Skip Tracking
    skips_used: int
    max_skips: int | None

    # Progress
    status: Literal["pending", "active", "paused", "completed", "error"]
    current_question_index: int
    questions: list[QuestionRecord]

    # Current State
    current_question: GeneratedQuestion | None
    awaiting_answer: bool

    # Pending answer data (set via aupdate_state during submit_answer)
    _pending_answer: str | None
    _pending_time_spent: int | None
    _pending_hints_used: int | None

    # Last evaluation result (set by _evaluate_answer for service compatibility)
    _last_evaluation: Any

    # Help options available after evaluation
    _help_options: list[dict[str, Any]]

    # Metrics
    metrics: SessionMetrics

    # Context (these are not persisted to avoid bloat)
    memory_context: dict[str, Any]
    theory_recommendations: dict[str, Any]
    messages: Annotated[list[dict[str, str]], add_messages]

    # Timestamps
    started_at: str  # ISO format datetime
    completed_at: str | None

    # Error handling
    error: str | None


def create_initial_practice_state(
    session_id: str,
    tenant_id: str,
    tenant_code: str,
    student_id: str,
    topic_full_code: str | None,
    topic: str,
    learning_objective_full_code: str | None = None,
    learning_objective_text: str | None = None,
    difficulty: float = 0.5,
    target_question_count: int | None = None,
    mode: str = "quick",
    mode_config: dict[str, Any] | None = None,
    persona_id: str = "tutor",
    question_type: str | None = None,
    # Handoff context from other workflows
    handoff_context: HandoffContext | None = None,
    # Random mode configuration
    subject_full_code: str | None = None,
    available_topics: list[dict[str, Any]] | None = None,
    # Educational context from curriculum hierarchy
    subject_name: str | None = None,
    grade_level: str | None = None,
    grade_level_code: str | None = None,
    age_range: str | None = None,
    curriculum: str | None = None,
    curriculum_code: str | None = None,
    language: str = "en",
    unit_name: str | None = None,
    # Pre-loaded context (loaded before workflow execution for transaction isolation)
    memory_context: dict[str, Any] | None = None,
    theory_recommendations: dict[str, Any] | None = None,
) -> PracticeState:
    """Create initial state for a practice session.

    If mode_config is not provided, it will be auto-loaded based on the mode.
    If target_question_count is not provided, it will use the mode's default.

    When handoff_context is provided from Learning Tutor:
    - priority_concepts is set to concepts_weak from handoff
    - from_learning_tutor is set to True
    - learning_session_id is set from handoff session_id
    - initial_difficulty is capped at verified_understanding level

    When handoff_context is provided from Practice Helper:
    - _offer_retry is set if student understood
    - _retry_question_id is set to the question that needed help

    Args:
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory and RAG operations.
        student_id: Student identifier.
        topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
        topic: Topic name to practice.
        learning_objective_full_code: Learning objective full code (optional).
        learning_objective_text: Learning objective text for LLM prompts (optional).
        difficulty: Initial difficulty (0.0-1.0).
        target_question_count: Number of questions (None = use mode default).
        mode: Practice mode (quick, deep, random, mixed, exam_prep, weakness_focus).
        mode_config: Mode configuration dict (None = auto-load from mode).
        persona_id: Persona to use.
        question_type: Question type to generate (None = agent decides).
        handoff_context: Context from source workflow (Learning Tutor, Practice Helper, Companion).
        subject_full_code: Subject full code for random mode (e.g., "UK-NC-2014.MAT").
        available_topics: List of topic dicts for random selection (for random mode).
        subject_name: Subject display name (e.g., "Mathematics").
        grade_level: Grade level display name (e.g., "Year 4 (KS2)").
        grade_level_code: Grade level code (e.g., "year_4").
        age_range: Age range string (e.g., "8-9 years").
        curriculum: Curriculum display name.
        curriculum_code: Curriculum code.
        language: Language code (default: "en").
        unit_name: Unit name.
        memory_context: Pre-loaded memory context (from MemoryManager.get_full_context).
            If provided, _load_context node will skip database queries.
        theory_recommendations: Pre-loaded theory recommendations (from TheoryOrchestrator).
            If provided, _load_context node will skip database queries.

    Returns:
        Initial PracticeState.
    """
    from src.domains.practice.modes import get_mode_config, mode_config_to_dict

    # Auto-load mode config if not provided
    if mode_config is None:
        try:
            config = get_mode_config(mode)
            mode_config = mode_config_to_dict(config)
        except ValueError:
            # Unknown mode, use default quick config
            config = get_mode_config("quick")
            mode_config = mode_config_to_dict(config)
            mode = "quick"

    # Use mode's question count if not explicitly provided
    if target_question_count is None:
        target_question_count = mode_config.get("question_count", 5)

    # Calculate time limit in seconds
    time_limit_minutes = mode_config.get("time_limit_minutes")
    time_limit_seconds = time_limit_minutes * 60 if time_limit_minutes else None

    # Calculate max skips (half of question count if skipping enabled)
    skip_enabled = mode_config.get("skip_enabled", False)
    max_skips = target_question_count // 2 if skip_enabled else 0

    # Process handoff context
    priority_concepts: list[str] = []
    from_learning_tutor = False
    learning_session_id: str | None = None
    initial_difficulty: float | None = None
    offer_retry = False
    retry_question_id: str | None = None
    helper_interventions: list[HelperIntervention] = []

    if handoff_context:
        source = handoff_context.get("source", "")

        if source == "learning_tutor":
            # Handoff from Learning Tutor after verified understanding
            from_learning_tutor = True
            learning_session_id = handoff_context.get("session_id")

            # Prioritize weak concepts from learning session
            priority_concepts = handoff_context.get("concepts_weak", [])

            # Cap initial difficulty based on verified understanding
            verified = handoff_context.get("verified_understanding", 0.5)
            if verified is not None:
                initial_difficulty = min(0.6, verified)
                difficulty = initial_difficulty

        elif source == "practice_helper":
            # Return from Practice Helper after getting help
            if handoff_context.get("understood"):
                offer_retry = True
                retry_question_id = handoff_context.get("question_id")

                # Record the helper intervention
                helper_interventions.append(HelperIntervention(
                    question_id=retry_question_id or "",
                    mode_that_helped=handoff_context.get("mode_that_helped"),
                    escalations=handoff_context.get("escalations_needed", 0),
                ))

        elif source == "companion":
            # Handoff from Companion with emotional context
            # Emotional context will be used in _load_context for theory adjustments
            pass

    return PracticeState(
        # Session Identity
        session_id=session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        student_id=student_id,
        persona_id=persona_id,
        # Configuration
        topic_full_code=topic_full_code,
        topic=topic,
        learning_objective_full_code=learning_objective_full_code,
        learning_objective_text=learning_objective_text,
        difficulty=difficulty,
        target_question_count=target_question_count,
        mode=mode,
        mode_config=mode_config,
        question_type=question_type,
        # Handoff Context
        handoff_context=handoff_context,
        priority_concepts=priority_concepts,
        from_learning_tutor=from_learning_tutor,
        learning_session_id=learning_session_id,
        initial_difficulty=initial_difficulty,
        # Practice Helper Return Context
        _offer_retry=offer_retry,
        _retry_question_id=retry_question_id,
        _helper_interventions=helper_interventions,
        # Performance Tracking
        consecutive_wrong=0,
        # Random Mode Configuration
        subject_full_code=subject_full_code,
        available_topics=available_topics or [],
        last_topic_full_code=None,
        # Educational Context
        subject_name=subject_name,
        grade_level=grade_level,
        grade_level_code=grade_level_code,
        age_range=age_range,
        curriculum=curriculum,
        curriculum_code=curriculum_code,
        language=language,
        unit_name=unit_name,
        # Time Tracking
        time_limit_seconds=time_limit_seconds,
        time_remaining_seconds=time_limit_seconds,
        # Skip Tracking
        skips_used=0,
        max_skips=max_skips,
        # Progress
        status="pending",
        current_question_index=0,
        questions=[],
        # Current State
        current_question=None,
        awaiting_answer=False,
        # Pending answer data
        _pending_answer=None,
        _pending_time_spent=None,
        _pending_hints_used=None,
        # Last evaluation
        _last_evaluation=None,
        # Help options
        _help_options=[],
        # Metrics
        metrics=SessionMetrics(
            questions_answered=0,
            questions_correct=0,
            questions_partial=0,
            questions_incorrect=0,
            total_time_seconds=0.0,
            average_time_per_question=0.0,
            accuracy=0.0,
            streak_current=0,
            streak_max=0,
            hints_total=0,
            skips_total=0,
        ),
        # Context (pre-loaded by service layer for transaction isolation)
        memory_context=memory_context or {},
        theory_recommendations=theory_recommendations or {},
        messages=[],
        # Timestamps
        started_at=datetime.now().isoformat(),
        completed_at=None,
        # Error
        error=None,
    )
