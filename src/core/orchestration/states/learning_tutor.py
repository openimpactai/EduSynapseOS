# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning Tutor workflow state.

This module defines the state structure for Learning Tutor workflows.
Learning Tutor is a proactive tutoring system that helps students learn
new concepts through various learning modes.

The state tracks:
- Topic and subject context (using composite keys from Central Curriculum)
- Learning mode (DISCOVERY, EXPLANATION, WORKED_EXAMPLE, GUIDED_PRACTICE, ASSESSMENT)
- Conversation history
- Theory recommendations
- Understanding progress

Topic and subject references use code-based composite keys from
the Central Curriculum structure:
- topic_full_code: e.g., "UK-NC-2014.MAT.Y4.NPV.001"
- subject_full_code: e.g., "UK-NC-2014.MAT"

The workflow uses subject-specific agents:
- learning_tutor_math
- learning_tutor_science
- learning_tutor_history
- learning_tutor_geography
- learning_tutor_general

Entry Points:
- companion_handoff: From Companion when student asks to learn
- practice_help: "I need to learn this" button in practice
- direct: Direct access from learning menu
- lms: External LMS deep link
- review: Spaced repetition review trigger
- weakness: Suggested based on mastery gaps
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class LearningMode(str, Enum):
    """Learning mode for the tutor workflow.

    Modes represent different teaching approaches:
    - DISCOVERY: Socratic questioning, student-led exploration
    - EXPLANATION: Clear concept explanation with examples
    - WORKED_EXAMPLE: Step-by-step problem demonstration
    - GUIDED_PRACTICE: Practice with scaffolded support
    - ASSESSMENT: Check understanding with questions
    """

    DISCOVERY = "discovery"
    EXPLANATION = "explanation"
    WORKED_EXAMPLE = "worked_example"
    GUIDED_PRACTICE = "guided_practice"
    ASSESSMENT = "assessment"


class LearningTutorTurn(TypedDict, total=False):
    """A single turn in the learning tutor conversation."""

    role: Literal["student", "tutor"]
    content: str
    timestamp: str  # ISO format
    action: str | None  # respond, more_examples, let_me_try, simpler, etc.
    learning_mode: str | None  # Mode at time of turn


class LearningTutorMetrics(TypedDict, total=False):
    """Metrics for the learning tutor session."""

    turn_count: int
    mode_transitions: int
    practice_attempted: int
    practice_correct: int
    understanding_progress: float  # 0.0 - 1.0
    total_duration_seconds: float
    comprehension_checks: int  # Number of comprehension evaluations performed
    comprehension_verified: int  # Number of times understanding was verified


class TheoryRecommendation(TypedDict, total=False):
    """Recommendations from educational theory orchestrator."""

    difficulty: float  # 0.0 - 1.0
    bloom_level: str  # remember, understand, apply, analyze, evaluate, create
    content_format: str  # visual, auditory, kinesthetic, reading
    scaffold_level: str  # high, medium, low
    guide_ratio: float  # 0.0 - 1.0, how much to guide vs let explore


class LearningTutorState(TypedDict, total=False):
    """State for learning tutor workflow.

    This state is checkpointed after each node execution, allowing
    the session to be interrupted (e.g., waiting for student message)
    and resumed later.

    Attributes:
        # Session Identity
        session_id: Unique learning session identifier.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.

        # Topic Context (using Central Curriculum composite keys)
        topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
        topic_name: Topic display name.
        subject_full_code: Subject full code (e.g., "UK-NC-2014.MAT").
        subject_name: Subject display name (Mathematics, Science, etc.).
        subject_code: Subject code for agent selection (extracted from full_code).

        # Student Context
        student_id: Student UUID as string.
        student_age: Student's age.
        grade_level: Grade level (number or string).
        language: Language preference.
        topic_mastery: Student's mastery on this topic (0.0-1.0).
        emotional_state: Current emotional state.
        interests: List of student interests.

        # Entry Point
        entry_point: How session was triggered.

        # Agent Selection
        agent_id: Selected agent ID.

        # Learning Mode
        learning_mode: Current learning mode.
        initial_mode: Mode determined at session start.
        mode_transition_count: How many times mode changed.

        # Conversation
        conversation_history: Full conversation history.
        last_student_message: Most recent student message.
        last_tutor_response: Most recent tutor response.
        first_message: First tutor message (greeting/opening).
        awaiting_input: Whether waiting for student input.

        # Progress Tracking
        understanding_progress: Estimated understanding (0.0-1.0).
        practice_question_count: Questions attempted in GUIDED_PRACTICE.
        practice_correct_count: Correct answers in GUIDED_PRACTICE.

        # Comprehension Evaluation State
        key_concepts: Key concepts for this topic (from curriculum or AI-extracted).
        understanding_verified: Whether understanding has been AI-verified.
        _comprehension_check_pending: Whether a comprehension check is pending.
        _comprehension_trigger: What triggered the check (self_reported, mode_transition, etc.).
        _awaiting_comprehension_response: Waiting for student's explanation.
        _ai_explanation_for_comparison: Store AI's explanation for parroting detection.
        _last_comprehension_check_turn: Turn number of last comprehension check.
        _comprehension_evaluation: Current evaluation result (internal).
        last_comprehension_evaluation: Most recent evaluation result (public).
        _all_comprehension_evaluations: History of all evaluations.
        _intended_mode_after_verification: Mode to transition to after verification.
        _misconceptions_to_address: Misconceptions detected and needing attention.
        _concepts_to_clarify: Concepts needing clarification.

        # Metrics
        metrics: Session metrics.

        # Theory Integration
        theory_recommendation: Recommendations from TheoryOrchestrator.

        # Curriculum Content (loaded from tools)
        learning_objectives: Learning objectives for the topic.
        topic_description: Generated topic description.

        # Session Status
        status: Current session status.
        completion_reason: Why session ended.
        understood: Whether student understood at completion.

        # Memory Context (loaded at start)
        memory_context: Full 4-layer memory context.
        emotional_context: Emotional context from service.

        # LangGraph messages
        messages: Message history for LangGraph.

        # Pending data (for interrupt/resume pattern)
        _pending_message: Message injected via aupdate_state during resume.
        _pending_action: Action injected via aupdate_state during resume.

        # Tool execution state
        current_tool_calls: Tool calls in current turn.
        tool_results: Results from tool execution.

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

    # Topic Context (using Central Curriculum composite keys)
    topic_full_code: str  # Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001")
    topic_name: str
    subject_full_code: str  # Subject full code (e.g., "UK-NC-2014.MAT")
    subject_name: str  # Subject display name
    subject_code: str  # Subject code for agent selection (extracted from full_code)

    # Student Context
    student_id: str
    student_age: int | None
    grade_level: int | str | None
    language: str
    topic_mastery: float
    emotional_state: str
    interests: list[str]

    # Entry Point
    entry_point: str  # companion_handoff, practice_help, direct, lms, review, weakness

    # Agent Selection
    agent_id: str

    # Learning Mode
    learning_mode: str  # discovery, explanation, worked_example, guided_practice, assessment
    initial_mode: str
    mode_transition_count: int

    # Conversation
    conversation_history: list[LearningTutorTurn]
    last_student_message: str | None
    last_tutor_response: str | None
    first_message: str | None
    last_action: str | None
    awaiting_input: bool

    # Progress Tracking
    understanding_progress: float
    practice_question_count: int
    practice_correct_count: int

    # Comprehension Evaluation State
    key_concepts: list[str]  # Key concepts for this topic
    understanding_verified: bool  # Whether understanding has been AI-verified
    _comprehension_check_pending: bool  # Whether a comprehension check is pending
    _comprehension_trigger: str | None  # What triggered the check (self_reported, mode_transition, etc.)
    _awaiting_comprehension_response: bool  # Waiting for student's explanation
    _ai_explanation_for_comparison: str  # Store AI's explanation for parroting detection
    _last_comprehension_check_turn: int  # Turn number of last comprehension check
    _comprehension_evaluation: dict[str, Any] | None  # Current evaluation result
    last_comprehension_evaluation: dict[str, Any] | None  # Most recent evaluation (public)
    _all_comprehension_evaluations: list[dict[str, Any]]  # History of all evaluations
    _intended_mode_after_verification: str | None  # Mode to transition to after verification
    _misconceptions_to_address: list[dict[str, Any]]  # Misconceptions detected
    _concepts_to_clarify: list[str]  # Concepts needing clarification

    # Metrics
    metrics: LearningTutorMetrics

    # Theory Integration
    theory_recommendation: TheoryRecommendation | None

    # Curriculum Content
    learning_objectives: list[dict[str, Any]]
    topic_description: str | None

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error"]
    completion_reason: str | None  # understood, max_turns, timeout, user_ended, assessment_passed
    understood: bool | None

    # Memory Context
    memory_context: dict[str, Any]
    emotional_context: dict[str, Any] | None

    # LangGraph messages
    messages: Annotated[list[dict[str, str]], add_messages]

    # Pending data (for interrupt/resume pattern)
    _pending_message: str | None
    _pending_action: str | None

    # Tool execution state
    current_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Timestamps
    started_at: str
    last_activity_at: str
    completed_at: str | None

    # Error handling
    error: str | None


def select_learning_mode(
    emotional_state: str,
    topic_mastery: float,
    entry_point: str,
    theory_recommendation: TheoryRecommendation | None = None,
) -> LearningMode:
    """Select initial learning mode based on context.

    Mode selection rules (from documentation):
    1. If entry_point is "review" -> WORKED_EXAMPLE (refresh memory)
    2. If emotional_state is frustrated/anxious -> EXPLANATION (gentle approach)
    3. If topic_mastery < 0.3 -> WORKED_EXAMPLE (needs concrete examples)
    4. If topic_mastery < 0.6 -> EXPLANATION (standard teaching)
    5. If topic_mastery >= 0.6 and student is curious -> DISCOVERY (exploration)
    6. Default -> EXPLANATION

    Theory recommendations can further influence the mode based on
    guide_ratio and scaffold_level.

    Args:
        emotional_state: Student's current emotional state.
        topic_mastery: Student's mastery on this topic (0.0-1.0).
        entry_point: How the session was triggered.
        theory_recommendation: Optional recommendations from TheoryOrchestrator.

    Returns:
        Selected LearningMode.
    """
    # Rule 1: Review entry point
    if entry_point == "review":
        return LearningMode.WORKED_EXAMPLE

    # Rule 2: Emotional state consideration
    if emotional_state.lower() in ("frustrated", "anxious", "stressed"):
        return LearningMode.EXPLANATION

    # Rule 3: Very low mastery
    if topic_mastery < 0.3:
        return LearningMode.WORKED_EXAMPLE

    # Rule 4: Low mastery
    if topic_mastery < 0.6:
        return LearningMode.EXPLANATION

    # Rule 5: Higher mastery with positive emotion
    if topic_mastery >= 0.6:
        if emotional_state.lower() in ("curious", "confident", "excited"):
            return LearningMode.DISCOVERY

    # Consider theory recommendations
    if theory_recommendation:
        guide_ratio = theory_recommendation.get("guide_ratio", 0.5)
        scaffold_level = theory_recommendation.get("scaffold_level", "medium")

        # Low guide ratio suggests more student-led (discovery)
        if guide_ratio < 0.3 and topic_mastery >= 0.5:
            return LearningMode.DISCOVERY

        # High scaffold level suggests more support
        if scaffold_level == "high":
            return LearningMode.WORKED_EXAMPLE

    # Default
    return LearningMode.EXPLANATION


def select_agent_id(subject_code: str) -> str:
    """Select agent based on subject.

    Agent selection (from documentation):
    - mathematics -> learning_tutor_math
    - science, physics, chemistry, biology -> learning_tutor_science
    - history, social_studies -> learning_tutor_history
    - geography, earth_science -> learning_tutor_geography
    - everything else -> learning_tutor_general

    Args:
        subject_code: Subject code from curriculum.

    Returns:
        Agent ID to use.
    """
    subject_lower = subject_code.lower()

    if subject_lower in ("mathematics", "math", "maths"):
        return "learning_tutor_math"

    if subject_lower in ("science", "physics", "chemistry", "biology"):
        return "learning_tutor_science"

    if subject_lower in ("history", "social_studies"):
        return "learning_tutor_history"

    if subject_lower in ("geography", "earth_science"):
        return "learning_tutor_geography"

    return "learning_tutor_general"


def create_initial_learning_tutor_state(
    session_id: str,
    tenant_id: str,
    tenant_code: str,
    # Topic context (using Central Curriculum composite keys)
    topic_full_code: str,
    topic_name: str,
    subject_full_code: str,
    subject_name: str = "General",
    subject_code: str = "general",
    # Student context
    student_id: str = "",
    student_age: int | None = None,
    grade_level: int | str | None = None,
    language: str = "en",
    topic_mastery: float = 0.5,
    emotional_state: str = "neutral",
    interests: list[str] | None = None,
    # Entry point
    entry_point: str = "direct",
    # Theory recommendation (if available)
    theory_recommendation: TheoryRecommendation | None = None,
) -> LearningTutorState:
    """Create initial state for a learning tutor session.

    This function:
    1. Selects the learning mode based on context
    2. Selects the appropriate agent based on subject
    3. Creates the initial state structure

    Args:
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.
        topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
        topic_name: Topic display name.
        subject_full_code: Subject full code (e.g., "UK-NC-2014.MAT").
        subject_name: Subject display name.
        subject_code: Subject code for agent selection.
        student_id: Student identifier.
        student_age: Student's age.
        grade_level: Grade level.
        language: Language preference.
        topic_mastery: Student's mastery on this topic (0.0-1.0).
        emotional_state: Student's current emotional state.
        interests: Student's interests.
        entry_point: How session was triggered.
        theory_recommendation: Optional theory recommendations.

    Returns:
        Initial LearningTutorState ready for workflow execution.
    """
    now = datetime.now().isoformat()

    # Select mode based on context
    learning_mode = select_learning_mode(
        emotional_state=emotional_state,
        topic_mastery=topic_mastery,
        entry_point=entry_point,
        theory_recommendation=theory_recommendation,
    )

    # Select agent based on subject
    agent_id = select_agent_id(subject_code)

    return LearningTutorState(
        # Session Identity
        session_id=session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        # Topic Context (using Central Curriculum composite keys)
        topic_full_code=topic_full_code,
        topic_name=topic_name,
        subject_full_code=subject_full_code,
        subject_name=subject_name,
        subject_code=subject_code,
        # Student Context
        student_id=student_id,
        student_age=student_age,
        grade_level=grade_level,
        language=language,
        topic_mastery=topic_mastery,
        emotional_state=emotional_state,
        interests=interests or [],
        # Entry Point
        entry_point=entry_point,
        # Agent Selection
        agent_id=agent_id,
        # Learning Mode
        learning_mode=learning_mode.value,
        initial_mode=learning_mode.value,
        mode_transition_count=0,
        # Conversation
        conversation_history=[],
        last_student_message=None,
        last_tutor_response=None,
        first_message=None,
        last_action=None,
        awaiting_input=False,
        # Progress Tracking
        understanding_progress=0.0,
        practice_question_count=0,
        practice_correct_count=0,
        # Comprehension Evaluation State
        key_concepts=[],
        understanding_verified=False,
        _comprehension_check_pending=False,
        _comprehension_trigger=None,
        _awaiting_comprehension_response=False,
        _ai_explanation_for_comparison="",
        _last_comprehension_check_turn=0,
        _comprehension_evaluation=None,
        last_comprehension_evaluation=None,
        _all_comprehension_evaluations=[],
        _intended_mode_after_verification=None,
        _misconceptions_to_address=[],
        _concepts_to_clarify=[],
        # Metrics
        metrics=LearningTutorMetrics(
            turn_count=0,
            mode_transitions=0,
            practice_attempted=0,
            practice_correct=0,
            understanding_progress=0.0,
            comprehension_checks=0,
            comprehension_verified=0,
        ),
        # Theory Integration
        theory_recommendation=theory_recommendation,
        # Curriculum Content (populated by workflow)
        learning_objectives=[],
        topic_description=None,
        # Session Status
        status="pending",
        completion_reason=None,
        understood=None,
        # Memory Context (populated by load_context node)
        memory_context={},
        emotional_context=None,
        # Messages
        messages=[],
        # Pending data
        _pending_message=None,
        _pending_action=None,
        # Tool state
        current_tool_calls=[],
        tool_results=[],
        # Timestamps
        started_at=now,
        last_activity_at=now,
        completed_at=None,
        # Error
        error=None,
    )


def escalate_mode(current_mode: str) -> str:
    """Escalate learning mode to provide more support.

    Mode escalation order (more support):
    DISCOVERY -> EXPLANATION -> WORKED_EXAMPLE -> GUIDED_PRACTICE

    Use when student shows confusion or explicitly asks for more help.

    Args:
        current_mode: Current learning mode.

    Returns:
        New learning mode (escalated by one level).
    """
    if current_mode == LearningMode.DISCOVERY.value:
        return LearningMode.EXPLANATION.value
    if current_mode == LearningMode.EXPLANATION.value:
        return LearningMode.WORKED_EXAMPLE.value
    if current_mode == LearningMode.WORKED_EXAMPLE.value:
        return LearningMode.GUIDED_PRACTICE.value
    # Already at highest support level
    return LearningMode.GUIDED_PRACTICE.value


def advance_mode(current_mode: str) -> str:
    """Advance learning mode when student shows understanding.

    Mode advancement order (toward mastery):
    WORKED_EXAMPLE -> EXPLANATION -> GUIDED_PRACTICE -> ASSESSMENT

    Use when student demonstrates understanding and is ready to progress.

    Args:
        current_mode: Current learning mode.

    Returns:
        New learning mode (advanced by one level).
    """
    if current_mode == LearningMode.WORKED_EXAMPLE.value:
        return LearningMode.EXPLANATION.value
    if current_mode == LearningMode.EXPLANATION.value:
        return LearningMode.GUIDED_PRACTICE.value
    if current_mode == LearningMode.GUIDED_PRACTICE.value:
        return LearningMode.ASSESSMENT.value
    if current_mode == LearningMode.DISCOVERY.value:
        return LearningMode.GUIDED_PRACTICE.value
    # Already at assessment
    return LearningMode.ASSESSMENT.value


def get_mode_actions(mode: str) -> list[dict[str, str]]:
    """Get available UI actions for a learning mode.

    Different modes offer different action buttons to the student.

    Args:
        mode: Current learning mode.

    Returns:
        List of action dictionaries with 'action' and 'label' keys.
    """
    base_actions = [
        {"action": "respond", "label": "Reply"},
        {"action": "end", "label": "End Session"},
    ]

    mode_specific: dict[str, list[dict[str, str]]] = {
        LearningMode.DISCOVERY.value: [
            {"action": "respond", "label": "Reply"},
            {"action": "give_hint", "label": "Give Me a Hint"},
            {"action": "show_me", "label": "Show Me"},
            {"action": "end", "label": "End Session"},
        ],
        LearningMode.EXPLANATION.value: [
            {"action": "respond", "label": "Reply"},
            {"action": "more_examples", "label": "More Examples"},
            {"action": "let_me_try", "label": "Let Me Try"},
            {"action": "simpler", "label": "Explain Simpler"},
            {"action": "end", "label": "End Session"},
        ],
        LearningMode.WORKED_EXAMPLE.value: [
            {"action": "respond", "label": "Reply"},
            {"action": "another_example", "label": "Another Example"},
            {"action": "i_understand", "label": "I Understand"},
            {"action": "simpler", "label": "Explain Simpler"},
            {"action": "end", "label": "End Session"},
        ],
        LearningMode.GUIDED_PRACTICE.value: [
            {"action": "respond", "label": "Submit Answer"},
            {"action": "hint", "label": "Give Hint"},
            {"action": "show_solution", "label": "Show Solution"},
            {"action": "end", "label": "End Session"},
        ],
        LearningMode.ASSESSMENT.value: [
            {"action": "respond", "label": "Submit Answer"},
            {"action": "skip", "label": "Skip Question"},
            {"action": "end", "label": "End Assessment"},
        ],
    }

    return mode_specific.get(mode, base_actions)
