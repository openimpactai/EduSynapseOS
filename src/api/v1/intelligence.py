# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Intelligence API endpoints.

This module provides endpoints for AI intelligence introspection:
- GET /theory-decisions/{session_id} - Get theory decision log for a session
- GET /session-timeline/{session_id} - Get session timeline with events

These endpoints expose the AI's decision-making process for the playground UI,
showing how the 7 educational theories combine to produce recommendations.

Example:
    GET /api/v1/intelligence/theory-decisions/{session_id}
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    require_auth,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models - Theory Decisions
# ============================================================================


class BloomState(BaseModel):
    """Bloom's Taxonomy current state."""

    current: str = Field(description="Current cognitive level (remember, understand, apply, analyze, evaluate, create)")
    progress: float = Field(ge=0.0, le=1.0, description="Progress within current level")
    target: str | None = Field(description="Target level for this session")


class ZPDState(BaseModel):
    """Zone of Proximal Development state."""

    difficulty: float = Field(ge=0.0, le=1.0, description="Current difficulty level")
    comfort_zone: tuple[float, float] = Field(description="Comfort zone range [min, max]")
    zpd_zone: tuple[float, float] = Field(description="ZPD range [min, max]")
    frustration_zone: tuple[float, float] = Field(description="Frustration zone range [min, max]")
    current_zone: str = Field(description="Current zone: comfort, zpd, or frustration")


class MasteryState(BaseModel):
    """Mastery Learning current state."""

    current: float = Field(ge=0.0, le=1.0, description="Current mastery level")
    threshold: float = Field(ge=0.0, le=1.0, description="Mastery threshold")
    can_advance: bool = Field(description="Whether student can advance to next topic")
    prerequisites_met: bool = Field(description="Whether prerequisites are met")


class ScaffoldingState(BaseModel):
    """Scaffolding support level state."""

    level: int = Field(ge=1, le=5, description="Current scaffold level (1=minimal, 5=maximum)")
    description: str = Field(description="Description of current support level")
    hints_remaining: int = Field(description="Hints remaining in current session")


class SocraticState(BaseModel):
    """Socratic Method current state."""

    guide_ratio: float = Field(ge=0.0, le=1.0, description="Guide vs tell ratio")
    questioning_style: str = Field(description="Current questioning style")
    questions_asked: int = Field(description="Socratic questions asked this session")


class SpacedRepetitionState(BaseModel):
    """FSRS Spaced Repetition state."""

    next_review: datetime | None = Field(description="Next scheduled review time")
    retention_target: float = Field(ge=0.0, le=1.0, description="Target retention rate")
    current_interval_days: int = Field(description="Current review interval in days")
    stability: float = Field(description="Memory stability score")
    difficulty: float = Field(description="Item difficulty rating")


class VARKState(BaseModel):
    """VARK Learning Styles state."""

    visual: float = Field(ge=0.0, le=1.0, description="Visual preference score")
    auditory: float = Field(ge=0.0, le=1.0, description="Auditory preference score")
    reading: float = Field(ge=0.0, le=1.0, description="Reading/writing preference score")
    kinesthetic: float = Field(ge=0.0, le=1.0, description="Kinesthetic preference score")
    recommended: str = Field(description="Recommended content format")


class TheoryState(BaseModel):
    """Combined state of all 7 educational theories."""

    bloom: BloomState = Field(description="Bloom's Taxonomy state")
    zpd: ZPDState = Field(description="Zone of Proximal Development state")
    mastery: MasteryState = Field(description="Mastery Learning state")
    scaffolding: ScaffoldingState = Field(description="Scaffolding state")
    socratic: SocraticState = Field(description="Socratic Method state")
    spaced_repetition: SpacedRepetitionState = Field(description="FSRS state")
    vark: VARKState = Field(description="VARK Learning Styles state")


class TheoryDecision(BaseModel):
    """A single theory decision event."""

    timestamp: datetime = Field(description="When decision was made")
    trigger: str = Field(description="What triggered this decision")
    theory_inputs: dict[str, Any] = Field(description="Input from each theory")
    combined_output: dict[str, Any] = Field(description="Combined recommendation")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")


class TheoryDecisionsResponse(BaseModel):
    """Theory decisions response for a session."""

    session_id: UUID = Field(description="Session ID")
    session_type: str = Field(description="Session type: learning_tutor, practice, etc.")
    current_state: TheoryState = Field(description="Current theory states")
    decisions: list[TheoryDecision] = Field(description="Decision history")
    total_decisions: int = Field(description="Total decisions made")


# ============================================================================
# Response Models - Session Timeline
# ============================================================================


class TimelineEvent(BaseModel):
    """A single event in the session timeline."""

    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Event type category")
    event_subtype: str = Field(description="Specific event subtype")
    description: str = Field(description="Human-readable description")
    actor: str = Field(description="Who triggered: student, tutor, system")
    data: dict[str, Any] = Field(description="Event-specific data")
    impact: dict[str, Any] | None = Field(description="Impact on learning state")


class AdaptationEvent(BaseModel):
    """An AI adaptation event showing real-time adjustments."""

    id: str = Field(description="Unique event ID")
    type: str = Field(description="Adaptation type: difficulty_increase, bloom_level_advance, etc.")
    from_value: str = Field(description="Previous value")
    to_value: str = Field(description="New value")
    reason: str = Field(description="Reason for adaptation")
    timestamp: datetime = Field(description="When adaptation occurred")


class SessionTimelineResponse(BaseModel):
    """Session timeline with all events and adaptations."""

    session_id: UUID = Field(description="Session ID")
    session_type: str = Field(description="Session type")
    started_at: datetime = Field(description="Session start time")
    ended_at: datetime | None = Field(description="Session end time if completed")
    duration_seconds: int = Field(description="Total duration in seconds")
    events: list[TimelineEvent] = Field(description="All timeline events")
    adaptations: list[AdaptationEvent] = Field(description="AI adaptation events")
    summary: dict[str, Any] = Field(description="Session summary statistics")


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/theory-decisions/{session_id}",
    response_model=TheoryDecisionsResponse,
    summary="Get theory decisions",
    description="Get the theory decision log for a session, showing how the AI made pedagogical decisions.",
)
async def get_theory_decisions(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TheoryDecisionsResponse:
    """Get theory decisions for a session.

    Shows how the 7 educational theories combined to produce recommendations
    at each decision point during the session.

    Args:
        session_id: The session ID (learning tutor or practice).
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        TheoryDecisionsResponse with current state and decision history.

    Raises:
        HTTPException: 404 if session not found.
    """
    logger.info(
        "Getting theory decisions: session=%s, user=%s",
        session_id,
        current_user.id,
    )

    # Try to find the session in learning_sessions first, then practice_sessions
    session_type, session_data = await _find_session(db, session_id, current_user.id)

    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Build theory state from session data
    theory_state = _build_theory_state(session_data)

    # Build decision history from session events
    decisions = _build_decision_history(session_data)

    return TheoryDecisionsResponse(
        session_id=session_id,
        session_type=session_type,
        current_state=theory_state,
        decisions=decisions,
        total_decisions=len(decisions),
    )


@router.get(
    "/session-timeline/{session_id}",
    response_model=SessionTimelineResponse,
    summary="Get session timeline",
    description="Get a detailed timeline of all events and AI adaptations during a session.",
)
async def get_session_timeline(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SessionTimelineResponse:
    """Get session timeline with all events.

    Provides a detailed view of everything that happened during a session,
    including student actions, AI responses, and real-time adaptations.

    Args:
        session_id: The session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        SessionTimelineResponse with events and adaptations.

    Raises:
        HTTPException: 404 if session not found.
    """
    logger.info(
        "Getting session timeline: session=%s, user=%s",
        session_id,
        current_user.id,
    )

    # Find the session
    session_type, session_data = await _find_session(db, session_id, current_user.id)

    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Build timeline events
    events = _build_timeline_events(session_type, session_data)

    # Build adaptation events
    adaptations = _build_adaptation_events(session_data)

    # Calculate duration
    started_at = session_data.get("started_at") or session_data.get("created_at")
    ended_at = session_data.get("ended_at")

    now = datetime.now(timezone.utc)
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if isinstance(ended_at, str):
        ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))

    duration = int((ended_at or now).timestamp() - started_at.timestamp()) if started_at else 0

    return SessionTimelineResponse(
        session_id=session_id,
        session_type=session_type,
        started_at=started_at or now,
        ended_at=ended_at,
        duration_seconds=duration,
        events=events,
        adaptations=adaptations,
        summary=_build_session_summary(session_type, session_data, events),
    )


# ============================================================================
# Helper Functions
# ============================================================================


async def _find_session(
    db: AsyncSession,
    session_id: UUID,
    user_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """Find a session by ID in either learning or practice sessions.

    Args:
        db: Database session.
        session_id: Session ID to find.
        user_id: User ID for authorization check.

    Returns:
        Tuple of (session_type, session_data) or (None, None) if not found.
    """
    from src.infrastructure.database.models.tenant.learning import LearningSession
    from src.infrastructure.database.models.tenant.practice import PracticeSession

    # Try learning session first
    result = await db.execute(
        select(LearningSession).where(
            LearningSession.id == str(session_id),
            LearningSession.student_id == user_id,
        )
    )
    learning_session = result.scalars().first()

    if learning_session:
        return "learning_tutor", _session_to_dict(learning_session)

    # Try practice session
    result = await db.execute(
        select(PracticeSession).where(
            PracticeSession.id == str(session_id),
            PracticeSession.student_id == user_id,
        )
    )
    practice_session = result.scalars().first()

    if practice_session:
        return "practice", _session_to_dict(practice_session)

    return "", None


def _session_to_dict(session: Any) -> dict[str, Any]:
    """Convert a session model to a dictionary.

    Args:
        session: SQLAlchemy model instance.

    Returns:
        Dictionary representation.
    """
    result = {}
    for column in session.__table__.columns:
        value = getattr(session, column.name)
        if isinstance(value, datetime):
            result[column.name] = value.isoformat()
        elif isinstance(value, UUID):
            result[column.name] = str(value)
        else:
            result[column.name] = value
    return result


def _build_theory_state(session_data: dict[str, Any]) -> TheoryState:
    """Build the current theory state from session data.

    Args:
        session_data: Session data dictionary.

    Returns:
        TheoryState with current values.
    """
    # Extract theory state from session checkpoint_data or use defaults
    checkpoint_data = session_data.get("checkpoint_data") or {}
    theory_data = checkpoint_data.get("theory_state") or {}

    # Get bloom level from session data
    current_bloom = session_data.get("current_bloom_level") or theory_data.get("bloom_level", "understand")

    # Get difficulty from session
    difficulty = session_data.get("current_difficulty") or theory_data.get("difficulty", 0.5)

    # Get mastery
    mastery = session_data.get("topic_mastery") or theory_data.get("mastery", 0.5)

    # Get scaffold level (1-5 scale matching ScaffoldLevel enum)
    scaffold_level = theory_data.get("scaffold_level", 3)

    return TheoryState(
        bloom=BloomState(
            current=current_bloom.lower() if isinstance(current_bloom, str) else "understand",
            progress=theory_data.get("bloom_progress", 0.5),
            target=theory_data.get("bloom_target"),
        ),
        zpd=ZPDState(
            difficulty=difficulty,
            comfort_zone=(0.0, 0.3),
            zpd_zone=(0.3, 0.7),
            frustration_zone=(0.7, 1.0),
            current_zone="zpd" if 0.3 <= difficulty <= 0.7 else ("comfort" if difficulty < 0.3 else "frustration"),
        ),
        mastery=MasteryState(
            current=mastery,
            threshold=theory_data.get("mastery_threshold", 0.8),
            can_advance=mastery >= theory_data.get("mastery_threshold", 0.8),
            prerequisites_met=theory_data.get("prerequisites_met", True),
        ),
        scaffolding=ScaffoldingState(
            level=scaffold_level,
            description=_scaffold_description(scaffold_level),
            hints_remaining=session_data.get("hints_remaining") or 3,
        ),
        socratic=SocraticState(
            guide_ratio=theory_data.get("guide_vs_tell_ratio", 0.7),
            questioning_style=theory_data.get("questioning_style", "guided"),
            questions_asked=theory_data.get("socratic_questions", 0),
        ),
        spaced_repetition=SpacedRepetitionState(
            next_review=None,  # Would come from FSRS scheduler
            retention_target=0.9,
            current_interval_days=1,
            stability=theory_data.get("fsrs_stability", 1.0),
            difficulty=theory_data.get("fsrs_difficulty", 0.3),
        ),
        vark=VARKState(
            visual=theory_data.get("vark_visual", 0.25),
            auditory=theory_data.get("vark_auditory", 0.25),
            reading=theory_data.get("vark_reading", 0.25),
            kinesthetic=theory_data.get("vark_kinesthetic", 0.25),
            recommended=theory_data.get("content_format", "multimodal"),
        ),
    )


def _scaffold_description(level: int) -> str:
    """Get description for scaffold level.

    Args:
        level: Scaffold level 1-5 (matching ScaffoldLevel enum).

    Returns:
        Human-readable description.
    """
    descriptions = {
        1: "Minimal support - mostly independent work",
        2: "Low support with occasional hints",
        3: "Moderate support with guided hints",
        4: "High support with examples and guidance",
        5: "Maximum support with step-by-step help",
    }
    return descriptions.get(level, "Moderate support with guided hints")


def _build_decision_history(session_data: dict[str, Any]) -> list[TheoryDecision]:
    """Build decision history from session data.

    Args:
        session_data: Session data dictionary.

    Returns:
        List of theory decisions.
    """
    decisions = []
    checkpoint_data = session_data.get("checkpoint_data") or {}
    decision_log = checkpoint_data.get("theory_decisions") or []

    for decision in decision_log:
        decisions.append(
            TheoryDecision(
                timestamp=datetime.fromisoformat(decision.get("timestamp", datetime.now(timezone.utc).isoformat())),
                trigger=decision.get("trigger", "turn_start"),
                theory_inputs=decision.get("inputs", {}),
                combined_output=decision.get("output", {}),
                confidence=decision.get("confidence", 0.8),
            )
        )

    # If no decisions logged, create a synthetic one from current state
    if not decisions:
        now = datetime.now(timezone.utc)
        decisions.append(
            TheoryDecision(
                timestamp=now,
                trigger="session_start",
                theory_inputs={
                    "bloom": {"recommended_level": "understand"},
                    "zpd": {"difficulty": 0.5},
                    "mastery": {"current": 0.5},
                    "scaffolding": {"level": 2},
                    "socratic": {"guide_ratio": 0.7},
                    "spaced_repetition": {"interval": 1},
                    "vark": {"format": "multimodal"},
                },
                combined_output={
                    "difficulty": 0.5,
                    "bloom_level": "understand",
                    "scaffold_level": 2,
                    "content_format": "multimodal",
                },
                confidence=0.8,
            )
        )

    return decisions


def _build_timeline_events(
    session_type: str,
    session_data: dict[str, Any],
) -> list[TimelineEvent]:
    """Build timeline events from session data.

    Args:
        session_type: Type of session.
        session_data: Session data dictionary.

    Returns:
        List of timeline events.
    """
    events = []

    # Session start event
    started_at = session_data.get("started_at") or session_data.get("created_at")
    if started_at:
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        events.append(
            TimelineEvent(
                timestamp=started_at,
                event_type="session",
                event_subtype="started",
                description=f"{session_type.replace('_', ' ').title()} session started",
                actor="student",
                data={
                    "topic": session_data.get("topic_name") or session_data.get("topic_full_code"),
                    "entry_point": session_data.get("entry_point", "direct"),
                },
                impact=None,
            )
        )

    # Add message/turn events from history
    checkpoint_data = session_data.get("checkpoint_data") or {}
    turn_history = checkpoint_data.get("turn_history") or []

    for i, turn in enumerate(turn_history):
        turn_time = turn.get("timestamp", started_at)
        if isinstance(turn_time, str):
            turn_time = datetime.fromisoformat(turn_time.replace("Z", "+00:00"))

        # Student message
        if turn.get("student_message"):
            events.append(
                TimelineEvent(
                    timestamp=turn_time,
                    event_type="message",
                    event_subtype="student_input",
                    description=turn.get("student_message", "")[:100],
                    actor="student",
                    data={"action": turn.get("action", "respond")},
                    impact=None,
                )
            )

        # Tutor response
        if turn.get("tutor_response"):
            events.append(
                TimelineEvent(
                    timestamp=turn_time,
                    event_type="message",
                    event_subtype="tutor_response",
                    description=turn.get("tutor_response", "")[:100],
                    actor="tutor",
                    data={"learning_mode": turn.get("learning_mode")},
                    impact=turn.get("state_changes"),
                )
            )

    # Session end event
    ended_at = session_data.get("ended_at")
    if ended_at:
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        events.append(
            TimelineEvent(
                timestamp=ended_at,
                event_type="session",
                event_subtype="completed",
                description=f"Session completed: {session_data.get('completion_reason', 'unknown')}",
                actor="system",
                data={
                    "status": session_data.get("status"),
                    "completion_reason": session_data.get("completion_reason"),
                },
                impact=None,
            )
        )

    return events


def _build_adaptation_events(session_data: dict[str, Any]) -> list[AdaptationEvent]:
    """Build adaptation events from session data.

    Infers adaptations by comparing consecutive theory decisions to detect
    changes in difficulty, bloom_level, and scaffold_level.

    Args:
        session_data: Session data dictionary.

    Returns:
        List of adaptation events.
    """
    adaptations = []
    checkpoint_data = session_data.get("checkpoint_data") or {}

    # First check explicit adaptations log
    adaptation_log = checkpoint_data.get("adaptations") or []
    for i, adaptation in enumerate(adaptation_log):
        timestamp = adaptation.get("timestamp", datetime.now(timezone.utc).isoformat())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        adaptations.append(
            AdaptationEvent(
                id=f"adapt-{i+1}",
                type=adaptation.get("type", "unknown"),
                from_value=str(adaptation.get("from", "")),
                to_value=str(adaptation.get("to", "")),
                reason=adaptation.get("reason", "Based on student performance"),
                timestamp=timestamp,
            )
        )

    # Infer adaptations from theory_decisions by comparing consecutive decisions
    theory_decisions = checkpoint_data.get("theory_decisions") or []

    for i in range(1, len(theory_decisions)):
        prev_decision = theory_decisions[i - 1]
        curr_decision = theory_decisions[i]

        prev_output = prev_decision.get("output", {})
        curr_output = curr_decision.get("output", {})
        curr_inputs = curr_decision.get("inputs", {})

        timestamp = curr_decision.get("timestamp", datetime.now(timezone.utc).isoformat())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Check difficulty change
        prev_diff = prev_output.get("difficulty")
        curr_diff = curr_output.get("difficulty")
        if prev_diff is not None and curr_diff is not None and abs(curr_diff - prev_diff) > 0.01:
            adapt_type = "difficulty_increase" if curr_diff > prev_diff else "difficulty_decrease"
            reason = "Correct answer - increasing challenge" if curr_inputs.get("is_correct") else "Incorrect answer - adjusting difficulty"
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-diff-{i}",
                    type=adapt_type,
                    from_value=f"{prev_diff:.2f}",
                    to_value=f"{curr_diff:.2f}",
                    reason=reason,
                    timestamp=timestamp,
                )
            )

        # Check bloom level change
        prev_bloom = prev_output.get("bloom_level")
        curr_bloom = curr_output.get("bloom_level")
        if prev_bloom and curr_bloom and prev_bloom != curr_bloom:
            bloom_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
            prev_idx = bloom_order.index(prev_bloom.lower()) if prev_bloom.lower() in bloom_order else 0
            curr_idx = bloom_order.index(curr_bloom.lower()) if curr_bloom.lower() in bloom_order else 0
            adapt_type = "bloom_level_advance" if curr_idx > prev_idx else "bloom_level_regress"
            reason = "Advancing cognitive complexity" if curr_idx > prev_idx else "Reinforcing fundamentals"
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-bloom-{i}",
                    type=adapt_type,
                    from_value=str(prev_bloom),
                    to_value=str(curr_bloom),
                    reason=reason,
                    timestamp=timestamp,
                )
            )

        # Check scaffold level change
        prev_scaffold = prev_output.get("scaffold_level")
        curr_scaffold = curr_output.get("scaffold_level")
        if prev_scaffold and curr_scaffold and prev_scaffold != curr_scaffold:
            # Handle both string and int scaffold levels
            prev_val = int(prev_scaffold) if isinstance(prev_scaffold, (int, str)) and str(prev_scaffold).isdigit() else 3
            curr_val = int(curr_scaffold) if isinstance(curr_scaffold, (int, str)) and str(curr_scaffold).isdigit() else 3
            adapt_type = "scaffold_increase" if curr_val > prev_val else "scaffold_decrease"
            reason = "Adding more support" if curr_val > prev_val else "Fading support as mastery grows"
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-scaffold-{i}",
                    type=adapt_type,
                    from_value=str(prev_scaffold),
                    to_value=str(curr_scaffold),
                    reason=reason,
                    timestamp=timestamp,
                )
            )

        # Check content format change (VARK theory)
        prev_format = prev_output.get("content_format")
        curr_format = curr_output.get("content_format")
        if prev_format and curr_format and prev_format != curr_format:
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-format-{i}",
                    type="format_change",
                    from_value=str(prev_format),
                    to_value=str(curr_format),
                    reason="Adjusting content format based on learning style",
                    timestamp=timestamp,
                )
            )

        # Check questioning style change (Socratic theory)
        prev_style = prev_output.get("questioning_style")
        curr_style = curr_output.get("questioning_style")
        if prev_style and curr_style and prev_style != curr_style:
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-socratic-{i}",
                    type="socratic_adjust",
                    from_value=str(prev_style),
                    to_value=str(curr_style),
                    reason="Adapting questioning approach to student needs",
                    timestamp=timestamp,
                )
            )

        # Check guide vs tell ratio change (Socratic theory)
        prev_ratio = prev_output.get("guide_vs_tell_ratio")
        curr_ratio = curr_output.get("guide_vs_tell_ratio")
        if prev_ratio is not None and curr_ratio is not None and abs(curr_ratio - prev_ratio) > 0.05:
            adapt_type = "socratic_adjust"
            if curr_ratio > prev_ratio:
                reason = "Increasing guided discovery approach"
            else:
                reason = "Providing more direct explanations"
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-guide-{i}",
                    type=adapt_type,
                    from_value=f"{prev_ratio:.2f}",
                    to_value=f"{curr_ratio:.2f}",
                    reason=reason,
                    timestamp=timestamp,
                )
            )

        # Check hints enabled change
        prev_hints = prev_output.get("hints_enabled")
        curr_hints = curr_output.get("hints_enabled")
        if prev_hints is not None and curr_hints is not None and prev_hints != curr_hints:
            adapt_type = "scaffold_increase" if curr_hints else "scaffold_decrease"
            reason = "Enabling hints for additional support" if curr_hints else "Disabling hints as confidence grows"
            adaptations.append(
                AdaptationEvent(
                    id=f"adapt-hints-{i}",
                    type=adapt_type,
                    from_value="hints_off" if not prev_hints else "hints_on",
                    to_value="hints_on" if curr_hints else "hints_off",
                    reason=reason,
                    timestamp=timestamp,
                )
            )

    return adaptations


def _build_session_summary(
    session_type: str,
    session_data: dict[str, Any],
    events: list[TimelineEvent],
) -> dict[str, Any]:
    """Build session summary statistics.

    Args:
        session_type: Type of session.
        session_data: Session data dictionary.
        events: Timeline events.

    Returns:
        Summary statistics dictionary.
    """
    return {
        "total_events": len(events),
        "student_messages": len([e for e in events if e.event_subtype == "student_input"]),
        "tutor_responses": len([e for e in events if e.event_subtype == "tutor_response"]),
        "turn_count": session_data.get("turn_count", 0),
        "questions_answered": session_data.get("questions_answered", 0),
        "questions_correct": session_data.get("questions_correct", 0),
        "understanding_progress": session_data.get("understanding_progress", 0.0),
        "final_mastery": session_data.get("topic_mastery") or session_data.get("score"),
    }
