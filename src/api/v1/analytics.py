# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics API endpoints.

This module provides endpoints for learning analytics:
- GET /dashboard - Get student dashboard
- GET /progress/{student_id} - Get progress report for a student
- GET /class/{class_id} - Get class analytics (for teachers)

Example:
    GET /api/v1/analytics/dashboard
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    require_auth,
    require_teacher_or_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.emotional import EmotionalStateService
from src.domains.analytics import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class DashboardStats(BaseModel):
    """Dashboard statistics."""

    total_sessions: int = Field(description="Total practice sessions")
    total_time_minutes: int = Field(description="Total learning time in minutes")
    questions_answered: int = Field(description="Total questions answered")
    questions_correct: int = Field(description="Total correct answers")
    average_accuracy: float | None = Field(description="Average accuracy percentage")
    streak_days: int = Field(description="Current daily streak")


class TopicProgress(BaseModel):
    """Progress for a specific topic."""

    topic_code: str = Field(description="Topic code")
    topic_name: str = Field(description="Topic name")
    mastery_level: float = Field(description="Mastery level 0.0-1.0")
    questions_answered: int = Field(description="Questions answered")
    accuracy: float | None = Field(description="Accuracy for this topic")
    last_practiced_at: datetime | None = Field(description="Last practice time")


class WeakArea(BaseModel):
    """Identified weak area."""

    topic_code: str = Field(description="Topic code")
    topic_name: str = Field(description="Topic name")
    mastery_level: float = Field(description="Current mastery")
    recommended_action: str = Field(description="Recommended action")


class RecentActivity(BaseModel):
    """Recent learning activity."""

    date: datetime = Field(description="Activity date")
    activity_type: str = Field(description="Type of activity")
    topic_name: str | None = Field(description="Related topic")
    duration_minutes: int = Field(description="Duration in minutes")
    score: float | None = Field(description="Score if applicable")


class DashboardResponse(BaseModel):
    """Student dashboard response."""

    stats: DashboardStats = Field(description="Overall statistics")
    topic_progress: list[TopicProgress] = Field(description="Progress by topic")
    weak_areas: list[WeakArea] = Field(description="Identified weak areas")
    recent_activity: list[RecentActivity] = Field(description="Recent activities")


class ProgressReportResponse(BaseModel):
    """Detailed progress report."""

    student_id: str = Field(description="Student ID")
    student_name: str | None = Field(description="Student name")
    period_start: datetime = Field(description="Report period start")
    period_end: datetime = Field(description="Report period end")
    stats: DashboardStats = Field(description="Period statistics")
    topic_progress: list[TopicProgress] = Field(description="Topic progress")
    growth_percentage: float | None = Field(description="Growth from previous period")
    recommendations: list[str] = Field(description="AI recommendations")


class StudentSummary(BaseModel):
    """Summary for a single student (for class view)."""

    student_id: str = Field(description="Student ID")
    student_name: str = Field(description="Student name")
    total_sessions: int = Field(description="Total sessions")
    average_accuracy: float | None = Field(description="Average accuracy")
    total_time_minutes: int = Field(description="Total time")
    last_active_at: datetime | None = Field(description="Last activity")


class ClassAnalyticsResponse(BaseModel):
    """Class analytics for teachers."""

    class_id: str = Field(description="Class ID")
    class_name: str = Field(description="Class name")
    total_students: int = Field(description="Total students")
    active_students: int = Field(description="Active students this period")
    average_mastery: float = Field(description="Class average mastery")
    topic_performance: list[dict[str, Any]] = Field(description="Performance by topic")
    student_summaries: list[StudentSummary] = Field(description="Individual summaries")


# ============================================================================
# Learning Path Response Models
# ============================================================================


class LearningPathNode(BaseModel):
    """A single node in the learning path."""

    id: str = Field(description="Node ID (topic code)")
    name: str = Field(description="Topic name")
    type: str = Field(description="Node type: topic, unit, strand")
    status: str = Field(description="Status: completed, in_progress, available, locked")
    mastery: float = Field(ge=0.0, le=1.0, description="Mastery level")
    prerequisites: list[str] = Field(description="Prerequisite node IDs")
    recommended_next: bool = Field(description="Whether this is recommended next")
    session_count: int = Field(description="Number of sessions on this topic")
    last_practiced_at: datetime | None = Field(description="Last practice time")


class LearningPathEdge(BaseModel):
    """An edge connecting learning path nodes."""

    from_node: str = Field(description="Source node ID")
    to_node: str = Field(description="Target node ID")
    type: str = Field(description="Edge type: prerequisite, related, next")
    strength: float = Field(ge=0.0, le=1.0, description="Connection strength")


class LearningPathStats(BaseModel):
    """Statistics about the learning path."""

    total_topics: int = Field(description="Total topics in path")
    completed_topics: int = Field(description="Topics with mastery >= threshold")
    in_progress_topics: int = Field(description="Topics being worked on")
    available_topics: int = Field(description="Topics available to start")
    locked_topics: int = Field(description="Topics not yet available")
    overall_progress: float = Field(ge=0.0, le=1.0, description="Overall path completion")
    current_strand: str | None = Field(description="Current strand being studied")
    next_milestone: str | None = Field(description="Next milestone topic")


class LearningPathResponse(BaseModel):
    """Learning path visualization data."""

    student_id: str = Field(description="Student ID")
    subject: str = Field(description="Subject code")
    framework: str = Field(description="Curriculum framework")
    nodes: list[LearningPathNode] = Field(description="Learning path nodes")
    edges: list[LearningPathEdge] = Field(description="Connections between nodes")
    stats: LearningPathStats = Field(description="Path statistics")
    recommended_path: list[str] = Field(description="Recommended sequence of node IDs")


# ============================================================================
# Emotional State Response Models
# ============================================================================


class EmotionalTrendResponse(BaseModel):
    """Emotional trend information."""

    direction: str = Field(description="Trend direction: improving, stable, declining")
    dominant_emotion: str = Field(description="Most frequent emotion")
    volatility: float = Field(ge=0.0, le=1.0, description="How much emotion is changing")


class EmotionalStateResponse(BaseModel):
    """Student emotional state response."""

    student_id: str = Field(description="Student ID")
    current_state: str = Field(description="Current emotion: frustrated, confused, anxious, confident, curious, excited, bored, neutral")
    intensity: str = Field(description="Intensity level: low, moderate, high")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in detection")
    triggers: list[str] = Field(description="What triggered this emotion")
    trend: EmotionalTrendResponse | None = Field(description="Recent trend")
    is_negative: bool = Field(description="Whether current state is negative")
    is_positive: bool = Field(description="Whether current state is positive")
    needs_support: bool = Field(description="Whether student needs immediate support")
    parent_mood: str | None = Field(description="Parent-reported mood if available")
    recommended_actions: list[str] = Field(description="Recommended actions for this state")
    updated_at: datetime = Field(description="When this was last updated")


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get dashboard",
    description="Get learning dashboard for the current user.",
)
async def get_dashboard(
    period_days: Annotated[int, Query(ge=1, le=365)] = 7,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> DashboardResponse:
    """Get student dashboard.

    Args:
        period_days: Number of days for statistics.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        DashboardResponse with statistics and progress.
    """
    logger.info("Getting dashboard for user: %s", current_user.id)

    service = AnalyticsService(db=db, tenant_code=tenant.code)

    dashboard = await service.get_dashboard(
        student_id=current_user.id,
        period_days=period_days,
    )

    return DashboardResponse(
        stats=DashboardStats(
            total_sessions=dashboard.stats.total_sessions,
            total_time_minutes=dashboard.stats.total_time_minutes,
            questions_answered=dashboard.stats.questions_answered,
            questions_correct=dashboard.stats.questions_correct,
            average_accuracy=dashboard.stats.average_accuracy,
            streak_days=dashboard.stats.streak_days,
        ),
        topic_progress=[
            TopicProgress(
                topic_code=tp.topic_code,
                topic_name=tp.topic_name,
                mastery_level=tp.mastery_level,
                questions_answered=tp.questions_answered,
                accuracy=tp.accuracy,
                last_practiced_at=tp.last_practiced_at,
            )
            for tp in dashboard.topic_progress
        ],
        weak_areas=[
            WeakArea(
                topic_code=wa.topic_code,
                topic_name=wa.topic_name,
                mastery_level=wa.mastery_level,
                recommended_action=wa.recommended_action,
            )
            for wa in dashboard.weak_areas
        ],
        recent_activity=[
            RecentActivity(
                date=ra.date,
                activity_type=ra.activity_type,
                topic_name=ra.topic_name,
                duration_minutes=ra.duration_minutes,
                score=ra.score,
            )
            for ra in dashboard.recent_activity
        ],
    )


@router.get(
    "/progress/{student_id}",
    response_model=ProgressReportResponse,
    summary="Get progress report",
    description="Get detailed progress report for a student.",
)
async def get_progress_report(
    student_id: UUID,
    period_days: Annotated[int, Query(ge=1, le=365)] = 30,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ProgressReportResponse:
    """Get progress report for a student.

    Args:
        student_id: The student ID.
        period_days: Number of days for the report.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ProgressReportResponse with detailed progress.

    Raises:
        HTTPException: If unauthorized to view student's progress.
    """
    # Check authorization
    # Students can view their own progress, teachers/admins can view their students
    if str(student_id) != current_user.id:
        if not (current_user.is_teacher or current_user.is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this student's progress",
            )

    logger.info(
        "Getting progress report: student=%s, period=%d days",
        student_id,
        period_days,
    )

    service = AnalyticsService(db=db, tenant_code=tenant.code)

    report = await service.get_progress_report(
        student_id=student_id,
        period_days=period_days,
    )

    return ProgressReportResponse(
        student_id=report.student_id,
        student_name=report.student_name,
        period_start=report.period_start,
        period_end=report.period_end,
        stats=DashboardStats(
            total_sessions=report.stats.total_sessions,
            total_time_minutes=report.stats.total_time_minutes,
            questions_answered=report.stats.questions_answered,
            questions_correct=report.stats.questions_correct,
            average_accuracy=report.stats.average_accuracy,
            streak_days=report.stats.streak_days,
        ),
        topic_progress=[
            TopicProgress(
                topic_code=tp.topic_code,
                topic_name=tp.topic_name,
                mastery_level=tp.mastery_level,
                questions_answered=tp.questions_answered,
                accuracy=tp.accuracy,
                last_practiced_at=tp.last_practiced_at,
            )
            for tp in report.topic_progress
        ],
        growth_percentage=report.growth_percentage,
        recommendations=report.recommendations,
    )


@router.get(
    "/class/{class_id}",
    response_model=ClassAnalyticsResponse,
    summary="Get class analytics",
    description="Get analytics for a class. Requires teacher or admin role.",
)
async def get_class_analytics(
    class_id: UUID,
    period_days: Annotated[int, Query(ge=1, le=365)] = 30,
    current_user: CurrentUser = Depends(require_teacher_or_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassAnalyticsResponse:
    """Get class analytics.

    Args:
        class_id: The class ID.
        period_days: Number of days for the report.
        current_user: Authenticated teacher or admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ClassAnalyticsResponse with class-level analytics.

    Raises:
        HTTPException: If class not found or unauthorized.
    """
    logger.info(
        "Getting class analytics: class=%s, period=%d days",
        class_id,
        period_days,
    )

    service = AnalyticsService(db=db, tenant_code=tenant.code)

    class_data = await service.get_class_analytics(
        class_id=class_id,
        period_days=period_days,
    )

    return ClassAnalyticsResponse(
        class_id=class_data.class_id,
        class_name=class_data.class_name,
        total_students=class_data.total_students,
        active_students=class_data.active_students,
        average_mastery=class_data.average_mastery,
        topic_performance=class_data.topic_performance,
        student_summaries=[
            StudentSummary(
                student_id=ss.student_id,
                student_name=ss.student_name,
                total_sessions=ss.total_sessions,
                average_accuracy=ss.average_accuracy,
                total_time_minutes=ss.total_time_minutes,
                last_active_at=ss.last_active_at,
            )
            for ss in class_data.student_summaries
        ],
    )


@router.get(
    "/learning-path/{student_id}",
    response_model=LearningPathResponse,
    summary="Get learning path",
    description="Get the learning path visualization data for a student.",
)
async def get_learning_path(
    student_id: UUID,
    subject: Annotated[str, Query(description="Subject code (e.g., MATHS, ENGLISH)")] = "MATHS",
    framework: Annotated[str | None, Query(description="Curriculum framework code")] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> LearningPathResponse:
    """Get learning path visualization data.

    Returns a graph structure representing the student's learning path
    through the curriculum, including:
    - Topics as nodes with mastery status
    - Prerequisites as edges
    - Recommended path based on current progress
    - Statistics about overall progress

    This data is designed for visualization in a node-graph or tree view.

    Args:
        student_id: The student ID.
        subject: Subject code (default: MATHS).
        framework: Optional curriculum framework (auto-detected if not provided).
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        LearningPathResponse with nodes, edges, and stats.

    Raises:
        HTTPException: 403 if not authorized to view student's data.
    """
    # Authorization check
    if str(student_id) != current_user.id:
        if not (current_user.is_teacher or current_user.is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this student's learning path",
            )

    logger.info(
        "Getting learning path: student=%s, subject=%s",
        student_id,
        subject,
    )

    # Build learning path from curriculum and mastery data
    nodes, edges = await _build_learning_path_graph(
        db=db,
        student_id=student_id,
        subject=subject,
        framework=framework,
    )

    # Calculate statistics
    stats = _calculate_path_stats(nodes)

    # Generate recommended path
    recommended = _generate_recommended_path(nodes, edges)

    return LearningPathResponse(
        student_id=str(student_id),
        subject=subject,
        framework=framework or "UK-NC-2014",  # Default framework
        nodes=nodes,
        edges=edges,
        stats=stats,
        recommended_path=recommended,
    )


async def _build_learning_path_graph(
    db: AsyncSession,
    student_id: UUID,
    subject: str,
    framework: str | None,
) -> tuple[list[LearningPathNode], list[LearningPathEdge]]:
    """Build learning path graph from curriculum and mastery data.

    Args:
        db: Database session.
        student_id: Student ID.
        subject: Subject code.
        framework: Curriculum framework.

    Returns:
        Tuple of (nodes, edges).
    """
    from sqlalchemy import select
    from src.infrastructure.database.models.tenant.curriculum import Topic

    nodes: list[LearningPathNode] = []
    edges: list[LearningPathEdge] = []

    # Get topics for the subject
    try:
        query = select(Topic).where(
            Topic.subject_code == subject,
        )
        if framework:
            query = query.where(Topic.framework_code == framework)

        result = await db.execute(query.limit(50))  # Limit for performance
        topics = result.scalars().all()

        # Get mastery data for these topics
        mastery_map = await _get_topic_mastery_map(db, student_id)

        for topic in topics:
            topic_code = topic.full_code or f"{topic.framework_code}.{topic.subject_code}.{topic.code}"
            mastery_data = mastery_map.get(topic_code, {})

            mastery_level = mastery_data.get("mastery", 0.0)
            session_count = mastery_data.get("sessions", 0)
            last_practiced = mastery_data.get("last_practiced")

            # Determine status based on mastery
            if mastery_level >= 0.8:
                status = "completed"
            elif session_count > 0:
                status = "in_progress"
            else:
                status = "available"  # Simplified - would check prerequisites

            nodes.append(
                LearningPathNode(
                    id=topic_code,
                    name=topic.name or topic.code,
                    type="topic",
                    status=status,
                    mastery=mastery_level,
                    prerequisites=[],  # Would come from curriculum relationships
                    recommended_next=status == "available" and mastery_level < 0.8,
                    session_count=session_count,
                    last_practiced_at=last_practiced,
                )
            )

            # Create edges based on topic sequence (unit is the parent)
            unit_code = f"{topic.framework_code}.{topic.subject_code}.{topic.grade_code}.{topic.unit_code}"
            edges.append(
                LearningPathEdge(
                    from_node=unit_code,
                    to_node=topic_code,
                    type="contains",
                    strength=1.0,
                )
            )

    except Exception as e:
        logger.warning("Could not load curriculum topics: %s", e)
        # Return minimal default node
        nodes.append(
            LearningPathNode(
                id="default",
                name="Mathematics",
                type="subject",
                status="available",
                mastery=0.0,
                prerequisites=[],
                recommended_next=True,
                session_count=0,
                last_practiced_at=None,
            )
        )

    return nodes, edges


async def _get_topic_mastery_map(
    db: AsyncSession,
    student_id: UUID,
) -> dict[str, dict[str, Any]]:
    """Get mastery data for all topics for a student.

    Args:
        db: Database session.
        student_id: Student ID.

    Returns:
        Dict mapping topic code to mastery data.
    """
    from sqlalchemy import select, and_
    from src.infrastructure.database.models.tenant.memory import SemanticMemory

    mastery_map: dict[str, dict[str, Any]] = {}

    try:
        result = await db.execute(
            select(SemanticMemory).where(
                and_(
                    SemanticMemory.student_id == str(student_id),
                    SemanticMemory.entity_type == "topic",
                )
            )
        )
        masteries = result.scalars().all()

        for m in masteries:
            mastery_map[m.entity_full_code] = {
                "mastery": float(m.mastery_level),
                "sessions": m.attempts_total,
                "last_practiced": m.updated_at,
            }

    except Exception as e:
        logger.debug("Could not load mastery data: %s", e)

    return mastery_map


def _calculate_path_stats(nodes: list[LearningPathNode]) -> LearningPathStats:
    """Calculate learning path statistics.

    Args:
        nodes: Learning path nodes.

    Returns:
        Path statistics.
    """
    total = len(nodes)
    completed = len([n for n in nodes if n.status == "completed"])
    in_progress = len([n for n in nodes if n.status == "in_progress"])
    available = len([n for n in nodes if n.status == "available"])
    locked = len([n for n in nodes if n.status == "locked"])

    # Find current strand (most recent in-progress topic's parent)
    in_progress_nodes = [n for n in nodes if n.status == "in_progress"]
    current_strand = in_progress_nodes[0].id.rsplit(".", 1)[0] if in_progress_nodes else None

    # Find next milestone (first available topic)
    available_nodes = [n for n in nodes if n.status == "available"]
    next_milestone = available_nodes[0].name if available_nodes else None

    return LearningPathStats(
        total_topics=total,
        completed_topics=completed,
        in_progress_topics=in_progress,
        available_topics=available,
        locked_topics=locked,
        overall_progress=completed / total if total > 0 else 0.0,
        current_strand=current_strand,
        next_milestone=next_milestone,
    )


def _generate_recommended_path(
    nodes: list[LearningPathNode],
    edges: list[LearningPathEdge],
) -> list[str]:
    """Generate recommended learning path sequence.

    Args:
        nodes: Learning path nodes.
        edges: Learning path edges.

    Returns:
        List of recommended node IDs in order.
    """
    # Simple recommendation: in_progress first, then available, sorted by mastery
    recommended = []

    # Add in-progress topics first
    in_progress = [n for n in nodes if n.status == "in_progress"]
    in_progress.sort(key=lambda n: n.mastery, reverse=True)
    recommended.extend([n.id for n in in_progress])

    # Add available topics
    available = [n for n in nodes if n.status == "available"]
    recommended.extend([n.id for n in available[:5]])  # Limit to 5

    return recommended


# ============================================================================
# Emotional State Endpoint
# ============================================================================


@router.get(
    "/emotional-state",
    response_model=EmotionalStateResponse,
    summary="Get emotional state",
    description="Get the current student's emotional state based on recent signals.",
)
async def get_emotional_state(
    window_minutes: Annotated[int, Query(ge=5, le=120)] = 30,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> EmotionalStateResponse:
    """Get student's current emotional state.

    Retrieves the current emotional context by analyzing recent emotional
    signals from learning activities, chat interactions, and performance data.

    Args:
        window_minutes: Time window for signal aggregation (default 30 min).
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        EmotionalStateResponse with current state and recommendations.
    """
    logger.info(
        "Getting emotional state for user: %s (window: %d min)",
        current_user.id,
        window_minutes,
    )

    emotional_service = EmotionalStateService(db=db)

    try:
        context = await emotional_service.get_current_state(
            student_id=current_user.id,
            window_minutes=window_minutes,
        )

        return EmotionalStateResponse(
            student_id=str(context.student_id),
            current_state=context.current_state.value,
            intensity=context.intensity.value,
            confidence=context.confidence,
            triggers=context.triggers,
            trend=EmotionalTrendResponse(
                direction=context.trend.direction,
                dominant_emotion=context.trend.dominant_emotion.value,
                volatility=context.trend.volatility,
            ) if context.trend else None,
            is_negative=context.is_negative,
            is_positive=context.is_positive,
            needs_support=context.needs_support,
            parent_mood=context.parent_mood.mood if context.parent_mood else None,
            recommended_actions=[a.value for a in context.recommended_actions],
            updated_at=context.updated_at,
        )
    except Exception as e:
        logger.exception("Failed to get emotional state: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emotional state",
        )
