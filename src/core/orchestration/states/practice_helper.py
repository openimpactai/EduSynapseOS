# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice Helper Tutor workflow state.

This module defines the state structure for practice helper tutoring workflows.
When a student answers incorrectly during practice and clicks "Get Help",
this workflow activates to help them understand the concept.

The state tracks:
- Original question context (from practice session)
- Tutoring mode (HINT, GUIDED, STEP_BY_STEP)
- Conversation history
- Current step (for STEP_BY_STEP mode)
- Understanding progress

The workflow uses subject-specific agents:
- practice_helper_tutor_math
- practice_helper_tutor_science
- practice_helper_tutor_history
- practice_helper_tutor_geography
- practice_helper_tutor_general
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class TutoringMode(str, Enum):
    """Tutoring mode for practice helper."""

    HINT = "hint"
    GUIDED = "guided"
    STEP_BY_STEP = "step_by_step"


class PracticeHelperTurn(TypedDict, total=False):
    """A single turn in the practice helper conversation."""

    role: Literal["student", "tutor"]
    content: str
    timestamp: str  # ISO format
    action: str | None  # respond, next_step, show_me, i_understand, end


class QuestionContext(TypedDict, total=False):
    """Context about the practice question being helped with."""

    question_id: str
    question_text: str
    question_type: str  # multiple_choice, true_false, short_answer, etc.
    options: dict[str, str] | None  # {A: "option1", B: "option2"} for MC
    correct_answer: Any
    student_answer: Any
    explanation: str | None


class StudentContext(TypedDict, total=False):
    """Student information for personalization."""

    student_id: str
    student_age: int | None
    student_gender: str | None
    grade_level: str | None
    grade_level_code: str | None
    language: str
    topic_mastery: float
    emotional_state: str
    interests: list[str]


class PracticeHelperMetrics(TypedDict, total=False):
    """Metrics for the practice helper session."""

    turn_count: int
    current_step: int
    total_steps: int | None
    mode_escalations: int
    understanding_progress: float  # 0.0 - 1.0


class PracticeHelperState(TypedDict, total=False):
    """State for practice helper tutoring workflow.

    This state is checkpointed after each node execution, allowing
    the session to be interrupted (e.g., waiting for student message)
    and resumed later.

    Attributes:
        # Session Identity
        session_id: Unique practice helper session identifier.
        practice_session_id: Reference to the original practice session.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.

        # Question Context
        question_context: Full context about the question being helped with.

        # Student Context
        student_context: Student information for personalization.

        # Subject and Agent
        subject: Subject of the question (mathematics, science, history, etc.).
        subject_code: Subject code for agent selection.
        topic_name: Topic name for display.
        agent_id: Selected agent ID (practice_helper_tutor_math, etc.).

        # Tutoring Mode
        tutoring_mode: Current mode (hint, guided, step_by_step).
        initial_mode: Mode determined at session start.
        mode_escalation_count: How many times mode was escalated.

        # Conversation
        conversation_history: Full conversation history.
        last_student_message: Most recent student message.
        last_tutor_response: Most recent tutor response.
        awaiting_input: Whether waiting for student input.

        # Progress Tracking
        current_step: Current step number (for STEP_BY_STEP mode).
        total_steps: Total steps planned (for STEP_BY_STEP mode).
        understanding_progress: Estimated understanding (0.0-1.0).
        i_dont_know_count: Times student said "I don't know".

        # Metrics
        metrics: Session metrics.

        # Session Status
        status: Current session status.
        completion_reason: Why session ended.
        understood: Whether student understood at completion.
        wants_retry: Whether student wants to retry the question.

        # Memory Context (loaded at start)
        memory_context: Full 4-layer memory context.
        emotional_context: Emotional context from service.

        # LangGraph messages
        messages: Message history for LangGraph.

        # Pending message (for interrupt/resume pattern)
        _pending_message: Message injected via aupdate_state during resume.
        _pending_action: Action injected via aupdate_state during resume.

        # Timestamps
        started_at: Session start time.
        last_activity_at: Last interaction time.
        completed_at: Session completion time.

        # Error handling
        error: Error message if workflow failed.
    """

    # Session Identity
    session_id: str
    practice_session_id: str
    tenant_id: str
    tenant_code: str

    # Question Context
    question_context: QuestionContext

    # Student Context
    student_context: StudentContext

    # Subject and Agent
    subject: str
    subject_code: str
    topic_name: str
    agent_id: str

    # Topic Codes (for event publishing)
    topic_framework_code: str | None
    topic_subject_code: str | None
    topic_grade_code: str | None
    topic_unit_code: str | None
    topic_code: str | None
    topic_full_code: str | None

    # Tutoring Mode
    tutoring_mode: str  # hint, guided, step_by_step
    initial_mode: str
    mode_escalation_count: int

    # Conversation
    conversation_history: list[PracticeHelperTurn]
    last_student_message: str | None
    last_tutor_response: str | None
    last_action: str | None  # Last action from student (respond, show_me, etc.)
    awaiting_input: bool

    # Progress Tracking
    current_step: int
    total_steps: int | None
    understanding_progress: float
    i_dont_know_count: int

    # Metrics
    metrics: PracticeHelperMetrics

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error", "escalating"]
    completion_reason: str | None  # understood, max_turns, timeout, user_ended, escalated_to_learning_tutor
    understood: bool | None
    wants_retry: bool | None

    # Escalation Detection
    confusion_pattern_count: int  # Tracks confusion signals for escalation
    discovered_misconceptions: list[dict[str, Any]]  # Misconceptions found during tutoring
    concepts_student_lacks: list[str]  # Concepts identified as gaps
    student_wants_full_lesson: bool  # Explicit request for Learning Tutor
    _escalate_to_learning_tutor: bool  # Flag to trigger escalation

    # Handoff Context (for workflow transitions)
    handoff_context: dict[str, Any] | None  # Context for Practice return or LT escalation
    ui_action: dict[str, Any] | None  # UI action for frontend navigation

    # Memory Context
    memory_context: dict[str, Any]
    emotional_context: dict[str, Any] | None

    # LangGraph messages
    messages: Annotated[list[dict[str, str]], add_messages]

    # Pending data (for interrupt/resume pattern)
    _pending_message: str | None
    _pending_action: str | None

    # Timestamps
    started_at: str
    last_activity_at: str
    completed_at: str | None

    # Error handling
    error: str | None


def select_tutoring_mode(
    emotional_state: str,
    topic_mastery: float,
) -> TutoringMode:
    """Select initial tutoring mode based on student context.

    Mode selection rules (from V4 documentation):
    1. If emotional_state is frustrated or anxious -> STEP_BY_STEP
    2. If topic_mastery < 0.5 -> GUIDED
    3. Otherwise -> HINT

    Args:
        emotional_state: Student's current emotional state.
        topic_mastery: Student's mastery on this topic (0.0-1.0).

    Returns:
        Selected TutoringMode.
    """
    if emotional_state.lower() in ("frustrated", "anxious"):
        return TutoringMode.STEP_BY_STEP

    if topic_mastery < 0.5:
        return TutoringMode.GUIDED

    return TutoringMode.HINT


def select_agent_id(subject_code: str) -> str:
    """Select agent based on subject.

    Agent selection (from V4 documentation):
    - mathematics -> practice_helper_tutor_math
    - science, physics, chemistry, biology -> practice_helper_tutor_science
    - history, social_studies -> practice_helper_tutor_history
    - geography, earth_science -> practice_helper_tutor_geography
    - everything else -> practice_helper_tutor_general

    Args:
        subject_code: Subject code from curriculum.

    Returns:
        Agent ID to use.
    """
    subject_lower = subject_code.lower()

    if subject_lower in ("mathematics", "math", "maths"):
        return "practice_helper_tutor_math"

    if subject_lower in ("science", "physics", "chemistry", "biology"):
        return "practice_helper_tutor_science"

    if subject_lower in ("history", "social_studies"):
        return "practice_helper_tutor_history"

    if subject_lower in ("geography", "earth_science"):
        return "practice_helper_tutor_geography"

    return "practice_helper_tutor_general"


def create_initial_practice_helper_state(
    session_id: str,
    practice_session_id: str,
    tenant_id: str,
    tenant_code: str,
    # Question context
    question_id: str,
    question_text: str,
    question_type: str,
    correct_answer: Any,
    student_answer: Any,
    options: dict[str, str] | None = None,
    explanation: str | None = None,
    # Student context
    student_id: str = "",
    student_age: int | None = None,
    student_gender: str | None = None,
    grade_level: str | None = None,
    grade_level_code: str | None = None,
    language: str = "en",
    topic_mastery: float = 0.5,
    emotional_state: str = "neutral",
    interests: list[str] | None = None,
    # Subject context
    subject: str = "other",
    subject_code: str = "other",
    topic_name: str = "",
    # Topic codes (for event publishing)
    topic_framework_code: str | None = None,
    topic_subject_code: str | None = None,
    topic_grade_code: str | None = None,
    topic_unit_code: str | None = None,
    topic_code: str | None = None,
    topic_full_code: str | None = None,
) -> PracticeHelperState:
    """Create initial state for a practice helper session.

    This function:
    1. Creates the question context
    2. Creates the student context
    3. Selects the tutoring mode based on emotional state and mastery
    4. Selects the appropriate agent based on subject

    Args:
        session_id: Unique session identifier.
        practice_session_id: Reference to original practice session.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.
        question_id: ID of the question being helped with.
        question_text: The question text.
        question_type: Type of question (multiple_choice, etc.).
        correct_answer: The correct answer.
        student_answer: The student's incorrect answer.
        options: Answer options for multiple choice.
        explanation: Explanation for the correct answer.
        student_id: Student identifier.
        student_age: Student's age.
        student_gender: Student's gender.
        grade_level: Grade level display name.
        grade_level_code: Grade level code.
        language: Language preference.
        topic_mastery: Student's mastery on this topic (0.0-1.0).
        emotional_state: Student's current emotional state.
        interests: Student's interests.
        subject: Subject name.
        subject_code: Subject code.
        topic_name: Topic name.
        topic_framework_code: Framework code for topic.
        topic_subject_code: Subject code for topic.
        topic_grade_code: Grade code for topic.
        topic_unit_code: Unit code for topic.
        topic_code: Topic code.
        topic_full_code: Full topic code.

    Returns:
        Initial PracticeHelperState ready for workflow execution.
    """
    now = datetime.now().isoformat()

    # Create question context
    question_context = QuestionContext(
        question_id=question_id,
        question_text=question_text,
        question_type=question_type,
        options=options,
        correct_answer=correct_answer,
        student_answer=student_answer,
        explanation=explanation,
    )

    # Create student context
    student_context = StudentContext(
        student_id=student_id,
        student_age=student_age,
        student_gender=student_gender,
        grade_level=grade_level,
        grade_level_code=grade_level_code,
        language=language,
        topic_mastery=topic_mastery,
        emotional_state=emotional_state,
        interests=interests or [],
    )

    # Select mode based on emotional state and mastery
    tutoring_mode = select_tutoring_mode(emotional_state, topic_mastery)

    # Select agent based on subject
    agent_id = select_agent_id(subject_code)

    return PracticeHelperState(
        # Session Identity
        session_id=session_id,
        practice_session_id=practice_session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        # Question Context
        question_context=question_context,
        # Student Context
        student_context=student_context,
        # Subject and Agent
        subject=subject,
        subject_code=subject_code,
        topic_name=topic_name,
        agent_id=agent_id,
        # Topic Codes
        topic_framework_code=topic_framework_code,
        topic_subject_code=topic_subject_code,
        topic_grade_code=topic_grade_code,
        topic_unit_code=topic_unit_code,
        topic_code=topic_code,
        topic_full_code=topic_full_code,
        # Tutoring Mode
        tutoring_mode=tutoring_mode.value,
        initial_mode=tutoring_mode.value,
        mode_escalation_count=0,
        # Conversation
        conversation_history=[],
        last_student_message=None,
        last_tutor_response=None,
        last_action=None,
        awaiting_input=False,
        # Progress Tracking
        current_step=0,
        total_steps=None,
        understanding_progress=0.0,
        i_dont_know_count=0,
        # Metrics
        metrics=PracticeHelperMetrics(
            turn_count=0,
            current_step=0,
            total_steps=None,
            mode_escalations=0,
            understanding_progress=0.0,
        ),
        # Session Status
        status="pending",
        completion_reason=None,
        understood=None,
        wants_retry=None,
        # Escalation Detection
        confusion_pattern_count=0,
        discovered_misconceptions=[],
        concepts_student_lacks=[],
        student_wants_full_lesson=False,
        _escalate_to_learning_tutor=False,
        # Handoff Context
        handoff_context=None,
        ui_action=None,
        # Memory Context (populated by load_context node)
        memory_context={},
        emotional_context=None,
        # Messages
        messages=[],
        # Pending data
        _pending_message=None,
        _pending_action=None,
        # Timestamps
        started_at=now,
        last_activity_at=now,
        completed_at=None,
        # Error
        error=None,
    )


def escalate_mode(current_mode: str) -> str:
    """Escalate tutoring mode one level.

    Mode escalation order: HINT -> GUIDED -> STEP_BY_STEP

    Args:
        current_mode: Current tutoring mode.

    Returns:
        New tutoring mode (escalated by one level).
    """
    if current_mode == TutoringMode.HINT.value:
        return TutoringMode.GUIDED.value
    if current_mode == TutoringMode.GUIDED.value:
        return TutoringMode.STEP_BY_STEP.value
    # Already at highest level
    return TutoringMode.STEP_BY_STEP.value
