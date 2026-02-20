# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory and progress API endpoints.

This module provides endpoints for accessing student memory context:
- GET /context - Get full 4-layer memory context
- GET /mastery - Get mastery levels for topics/objectives
- GET /weak-areas - Get identified weak areas
- GET /review-schedule - Get spaced repetition schedule

Example:
    GET /api/v1/memory/context
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    get_tenant_db_manager,
    require_auth,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.infrastructure.database.tenant_manager import TenantDatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class EpisodicMemoryItem(BaseModel):
    """An episodic memory item (recent event)."""

    id: UUID = Field(description="Memory ID")
    event_type: str = Field(description="Type of event")
    summary: str = Field(description="Event summary")
    topic: str | None = Field(description="Related topic")
    timestamp: datetime = Field(description="When it happened")
    importance: float = Field(description="Importance score 0.0-1.0")


class SemanticMemoryItem(BaseModel):
    """A semantic memory item (knowledge state)."""

    entity_type: str = Field(description="topic or learning_objective")
    entity_code: str = Field(description="Entity code")
    entity_name: str = Field(description="Entity name")
    mastery_level: float = Field(ge=0.0, le=1.0, description="Mastery level")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in estimate")
    last_assessed_at: datetime | None = Field(description="Last assessment time")


class ProceduralMemoryItem(BaseModel):
    """A procedural memory item (learning strategy)."""

    strategy_type: str = Field(description="Type of strategy")
    value: str = Field(description="Strategy value")
    effectiveness: float = Field(description="Effectiveness score")
    usage_count: int = Field(description="Times used")


class AssociativeMemoryItem(BaseModel):
    """An associative memory item (interests/connections)."""

    id: UUID = Field(description="Memory ID")
    content: str = Field(description="Interest or connection")
    category: str = Field(description="Category")
    strength: float = Field(description="Association strength")


class MemoryContextResponse(BaseModel):
    """Full 4-layer memory context."""

    student_id: UUID = Field(description="Student ID")
    episodic: list[EpisodicMemoryItem] = Field(description="Recent events")
    semantic: list[SemanticMemoryItem] = Field(description="Knowledge state")
    procedural: list[ProceduralMemoryItem] = Field(description="Learning strategies")
    associative: list[AssociativeMemoryItem] = Field(description="Interests")
    retrieved_at: datetime = Field(description="When context was retrieved")


class MasteryItem(BaseModel):
    """Mastery level for an entity."""

    entity_type: str = Field(description="topic or learning_objective")
    entity_code: str = Field(description="Entity code")
    entity_name: str = Field(description="Entity name")
    parent_name: str | None = Field(description="Parent entity name")
    mastery_level: float = Field(ge=0.0, le=1.0, description="Mastery level")
    trend: str = Field(description="improving, stable, or declining")
    last_practiced_at: datetime | None = Field(description="Last practice time")


class MasteryResponse(BaseModel):
    """Mastery levels response."""

    student_id: UUID = Field(description="Student ID")
    mastery_items: list[MasteryItem] = Field(description="Mastery items")
    overall_mastery: float = Field(description="Overall mastery score")


class WeakAreaItem(BaseModel):
    """An identified weak area."""

    entity_type: str = Field(description="topic or learning_objective")
    entity_code: str = Field(description="Entity code")
    entity_name: str = Field(description="Entity name")
    mastery_level: float = Field(description="Current mastery")
    gap_from_target: float = Field(description="Gap from target mastery")
    recommended_action: str = Field(description="Recommended action")
    priority: int = Field(description="Priority (1=highest)")


class WeakAreasResponse(BaseModel):
    """Weak areas response."""

    student_id: UUID = Field(description="Student ID")
    weak_areas: list[WeakAreaItem] = Field(description="Identified weak areas")


class ReviewItem(BaseModel):
    """An item scheduled for review using FSRS algorithm."""

    entity_code: str = Field(description="Full entity code (topic/objective)")
    entity_name: str = Field(description="Entity name for display")
    due_at: datetime = Field(description="When review is due")
    fsrs_state: str = Field(description="FSRS state: new, learning, review, relearning")
    stability: float = Field(description="FSRS stability value")
    difficulty: float = Field(description="FSRS difficulty rating (0-1)")
    priority: str = Field(description="Priority: critical, high, medium, low")


class ReviewScheduleResponse(BaseModel):
    """Spaced repetition review schedule."""

    student_id: UUID = Field(description="Student ID")
    due_now: list[ReviewItem] = Field(description="Items due for review")
    upcoming: list[ReviewItem] = Field(description="Items coming up")
    total_due: int = Field(description="Total items due")
    next_review_at: datetime | None = Field(description="When next review is due")


class ActivityFeedItem(BaseModel):
    """A single item in the memory activity feed."""

    id: str = Field(description="Activity ID")
    timestamp: datetime = Field(description="When activity occurred")
    layer: str = Field(description="Memory layer: episodic, semantic, procedural, associative")
    operation: str = Field(description="Operation: create, update, read, consolidate")
    description: str = Field(description="Human-readable description")
    entity_type: str | None = Field(description="Entity type affected")
    entity_name: str | None = Field(description="Entity name affected")
    data: dict[str, Any] = Field(description="Operation-specific data")
    source_workflow: str | None = Field(description="Workflow that triggered this")


class ActivityFeedResponse(BaseModel):
    """Real-time memory activity feed."""

    student_id: UUID = Field(description="Student ID")
    activities: list[ActivityFeedItem] = Field(description="Recent activities")
    total_count: int = Field(description="Total activities in period")
    layers_active: dict[str, bool] = Field(description="Which layers have recent activity")
    context_building: dict[str, Any] = Field(description="Context building status")


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/context",
    response_model=MemoryContextResponse,
    summary="Get memory context",
    description="Get the full 4-layer memory context for the current user.",
)
async def get_memory_context(
    topic_code: Annotated[str | None, Query(description="Topic code filter")] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
) -> MemoryContextResponse:
    """Get full memory context.

    Args:
        topic_code: Optional topic filter.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.

    Returns:
        MemoryContextResponse with all 4 memory layers.
    """
    import asyncio
    from src.core.intelligence.embeddings import EmbeddingService
    from src.core.memory.manager import MemoryManager
    from src.infrastructure.vectors import get_qdrant

    logger.info(
        "Getting memory context: user=%s, topic_code=%s",
        current_user.id,
        topic_code,
    )

    # Initialize MemoryManager (same pattern as practice.py)
    qdrant_client = get_qdrant()
    embedding_service = EmbeddingService()
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    # Get full memory context using sync method via asyncio.to_thread
    # to avoid greenlet issues (same pattern as PracticeService)
    student_id = UUID(current_user.id)
    full_context = await asyncio.to_thread(
        memory_manager.get_full_context_sync,
        tenant.code,
        student_id,
        topic_code,
    )

    # Transform episodic memories to API response format
    episodic_items = []
    for ep in full_context.episodic:
        # EpisodicMemoryResponse has: id, event_type, summary, details, topic_full_code, occurred_at, importance, emotional_state
        episodic_items.append(
            EpisodicMemoryItem(
                id=UUID(ep.id) if isinstance(ep.id, str) else ep.id,
                event_type=ep.event_type.value if hasattr(ep.event_type, 'value') else str(ep.event_type),
                summary=ep.summary,
                topic=ep.topic_full_code,
                timestamp=ep.occurred_at,
                importance=float(ep.importance),
            )
        )

    # Transform semantic memories (MasteryOverview has aggregate stats, not individual topics)
    # MasteryOverview: overall_mastery, topics_mastered, topics_learning, topics_struggling, by_subject
    # We create a summary item from the aggregate data
    semantic_items = []
    if full_context.semantic:
        # Create items from by_subject breakdown
        for subject_code, mastery_level in full_context.semantic.by_subject.items():
            semantic_items.append(
                SemanticMemoryItem(
                    entity_type="subject",
                    entity_code=subject_code,
                    entity_name=subject_code,
                    mastery_level=float(mastery_level),
                    confidence=0.8,
                    last_assessed_at=None,
                )
            )
        # Add topic-specific mastery if available
        if full_context.topic_mastery:
            semantic_items.append(
                SemanticMemoryItem(
                    entity_type="topic",
                    entity_code=full_context.topic_mastery.entity_full_code,
                    entity_name=full_context.topic_mastery.entity_full_code.split(".")[-1] if full_context.topic_mastery.entity_full_code else "",
                    mastery_level=float(full_context.topic_mastery.mastery_level),
                    confidence=float(full_context.topic_mastery.confidence),
                    last_assessed_at=full_context.topic_mastery.last_practiced_at,
                )
            )

    # Transform procedural patterns to API response format
    # LearningPatterns: best_time_of_day, optimal_session_duration, preferred_content_format, hint_dependency, vark_profile
    procedural_items = []
    if full_context.procedural:
        if full_context.procedural.best_time_of_day:
            procedural_items.append(
                ProceduralMemoryItem(
                    strategy_type="time_of_day",
                    value=full_context.procedural.best_time_of_day,
                    effectiveness=0.8,
                    usage_count=10,
                )
            )
        if full_context.procedural.preferred_content_format:
            procedural_items.append(
                ProceduralMemoryItem(
                    strategy_type="content_format",
                    value=full_context.procedural.preferred_content_format,
                    effectiveness=0.8,
                    usage_count=10,
                )
            )
        if full_context.procedural.hint_dependency is not None:
            procedural_items.append(
                ProceduralMemoryItem(
                    strategy_type="hint_dependency",
                    value=f"{full_context.procedural.hint_dependency:.2f}",
                    effectiveness=1.0 - full_context.procedural.hint_dependency,
                    usage_count=10,
                )
            )
        if full_context.procedural.vark_profile:
            # VARKProfile has: visual, auditory, reading_writing, kinesthetic
            dominant = max(
                [
                    ("visual", full_context.procedural.vark_profile.visual),
                    ("auditory", full_context.procedural.vark_profile.auditory),
                    ("reading_writing", full_context.procedural.vark_profile.reading_writing),
                    ("kinesthetic", full_context.procedural.vark_profile.kinesthetic),
                ],
                key=lambda x: x[1],
            )
            procedural_items.append(
                ProceduralMemoryItem(
                    strategy_type="vark_preference",
                    value=dominant[0],
                    effectiveness=dominant[1],
                    usage_count=getattr(full_context.procedural.vark_profile, 'sample_size', 10),
                )
            )

    # Transform associative memories to API response format
    # StudentInterests: interests (list[InterestItem]), effective_analogies
    # InterestItem has: content, category, strength (no id, no association_type)
    associative_items = []
    if full_context.associative:
        for idx, interest in enumerate(full_context.associative.interests):
            associative_items.append(
                AssociativeMemoryItem(
                    id=uuid4(),  # Generate UUID since InterestItem doesn't have id
                    content=interest.content,
                    category=interest.category or "interest",
                    strength=float(interest.strength),
                )
            )

    return MemoryContextResponse(
        student_id=student_id,
        episodic=episodic_items,
        semantic=semantic_items,
        procedural=procedural_items,
        associative=associative_items,
        retrieved_at=datetime.now(timezone.utc),
    )


@router.get(
    "/mastery",
    response_model=MasteryResponse,
    summary="Get mastery levels",
    description="Get mastery levels for topics and objectives.",
)
async def get_mastery_levels(
    entity_type: Annotated[str | None, Query(description="Entity type filter")] = None,
    parent_code: Annotated[str | None, Query(description="Parent entity code")] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> MasteryResponse:
    """Get mastery levels.

    Args:
        entity_type: Filter by entity type.
        parent_code: Filter by parent entity code.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        MasteryResponse with mastery levels.
    """
    from sqlalchemy import select, and_, func
    from src.infrastructure.database.models.tenant.memory import SemanticMemory
    from src.infrastructure.database.models.tenant.curriculum import Topic

    logger.info(
        "Getting mastery levels: user=%s, type=%s, parent=%s",
        current_user.id,
        entity_type,
        parent_code,
    )

    student_id = UUID(current_user.id)

    # Build query for SemanticMemory
    query = select(SemanticMemory).where(
        SemanticMemory.student_id == str(student_id),
    )

    if entity_type:
        query = query.where(SemanticMemory.entity_type == entity_type)

    if parent_code:
        # Filter by parent code (entity_full_code starts with parent_code)
        query = query.where(SemanticMemory.entity_full_code.startswith(parent_code))

    result = await db.execute(query)
    memories = result.scalars().all()

    # Calculate overall mastery
    overall_mastery = 0.0
    if memories:
        overall_mastery = sum(float(m.mastery_level) for m in memories) / len(memories)

    # Build mastery items
    mastery_items = []
    for memory in memories:
        # Determine trend based on recent performance
        if memory.current_streak >= 3:
            trend = "improving"
        elif memory.current_streak == 0 and memory.attempts_total > 0:
            trend = "declining"
        else:
            trend = "stable"

        # Get topic name from entity_full_code
        entity_name = memory.entity_full_code.split(".")[-1] if memory.entity_full_code else ""
        parent_name = ".".join(memory.entity_full_code.split(".")[:-1]) if memory.entity_full_code else None

        mastery_items.append(
            MasteryItem(
                entity_type=memory.entity_type,
                entity_code=memory.entity_full_code,
                entity_name=entity_name,
                parent_name=parent_name,
                mastery_level=float(memory.mastery_level),
                trend=trend,
                last_practiced_at=memory.last_practiced_at,
            )
        )

    return MasteryResponse(
        student_id=student_id,
        mastery_items=mastery_items,
        overall_mastery=round(overall_mastery, 2),
    )


@router.get(
    "/weak-areas",
    response_model=WeakAreasResponse,
    summary="Get weak areas",
    description="Get identified weak areas that need attention.",
)
async def get_weak_areas(
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.6,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> WeakAreasResponse:
    """Get weak areas.

    Args:
        threshold: Mastery threshold below which is considered weak.
        limit: Maximum number of weak areas to return.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        WeakAreasResponse with identified weak areas.
    """
    from sqlalchemy import select, and_
    from src.infrastructure.database.models.tenant.memory import SemanticMemory

    logger.info(
        "Getting weak areas: user=%s, threshold=%f",
        current_user.id,
        threshold,
    )

    student_id = UUID(current_user.id)

    # Query for topics with low mastery (below threshold)
    query = (
        select(SemanticMemory)
        .where(
            and_(
                SemanticMemory.student_id == str(student_id),
                SemanticMemory.entity_type == "topic",
                SemanticMemory.mastery_level < threshold,
            )
        )
        .order_by(SemanticMemory.mastery_level.asc())
        .limit(limit)
    )

    result = await db.execute(query)
    memories = result.scalars().all()

    # Build weak area items
    weak_areas = []
    for memory in memories:
        mastery = float(memory.mastery_level)
        # Generate recommendation based on mastery level
        if mastery < 0.2:
            recommendation = "Start with foundational concepts and basic examples"
            priority = "critical"
        elif mastery < 0.3:
            recommendation = "Review core principles and practice basic exercises"
            priority = "high"
        else:
            recommendation = "Practice more exercises to reinforce understanding"
            priority = "medium"

        weak_areas.append(
            WeakAreaItem(
                topic_code=memory.entity_full_code,
                topic_name=memory.entity_full_code.split(".")[-1] if memory.entity_full_code else "",
                mastery_level=mastery,
                attempts_total=memory.attempts_total,
                last_practiced_at=memory.last_practiced_at,
                recommendation=recommendation,
                priority=priority,
            )
        )

    return WeakAreasResponse(
        student_id=student_id,
        weak_areas=weak_areas,
    )


@router.get(
    "/review-schedule",
    response_model=ReviewScheduleResponse,
    summary="Get review schedule",
    description="Get spaced repetition review schedule.",
)
async def get_review_schedule(
    days_ahead: Annotated[int, Query(ge=1, le=30)] = 7,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ReviewScheduleResponse:
    """Get review schedule.

    Args:
        days_ahead: Number of days to look ahead.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ReviewScheduleResponse with scheduled reviews.
    """
    from datetime import timedelta
    from sqlalchemy import select, and_, or_
    from src.infrastructure.database.models.tenant.memory import SemanticMemory

    logger.info(
        "Getting review schedule: user=%s, days=%d",
        current_user.id,
        days_ahead,
    )

    student_id = UUID(current_user.id)
    now = datetime.now(timezone.utc)
    future_date = now + timedelta(days=days_ahead)

    # Query for items due for review (using FSRS fields)
    query = (
        select(SemanticMemory)
        .where(
            and_(
                SemanticMemory.student_id == str(student_id),
                SemanticMemory.entity_type == "topic",
                SemanticMemory.fsrs_due_at.isnot(None),
                SemanticMemory.fsrs_due_at <= future_date,
            )
        )
        .order_by(SemanticMemory.fsrs_due_at.asc())
    )

    result = await db.execute(query)
    memories = result.scalars().all()

    # Split into due_now (overdue or due today) and upcoming
    due_now = []
    upcoming = []
    next_review_at = None

    for memory in memories:
        if memory.fsrs_due_at:
            # Calculate priority based on how overdue it is
            days_overdue = (now - memory.fsrs_due_at).days if memory.fsrs_due_at <= now else 0
            if days_overdue > 7:
                priority = "critical"
            elif days_overdue > 0:
                priority = "high"
            elif memory.fsrs_due_at.date() == now.date():
                priority = "medium"
            else:
                priority = "low"

            item = ReviewItem(
                entity_code=memory.entity_full_code,
                entity_name=memory.entity_full_code.split(".")[-1] if memory.entity_full_code else "",
                due_at=memory.fsrs_due_at,
                fsrs_state=memory.fsrs_state or "review",
                stability=float(memory.fsrs_stability) if memory.fsrs_stability else 0.0,
                difficulty=float(memory.fsrs_difficulty) if memory.fsrs_difficulty else 0.5,
                priority=priority,
            )

            # Due now if overdue or due today
            if memory.fsrs_due_at <= now:
                due_now.append(item)
            else:
                upcoming.append(item)

            # Track next review
            if next_review_at is None or memory.fsrs_due_at < next_review_at:
                next_review_at = memory.fsrs_due_at

    return ReviewScheduleResponse(
        student_id=student_id,
        due_now=due_now,
        upcoming=upcoming,
        total_due=len(due_now) + len(upcoming),
        next_review_at=next_review_at,
    )


@router.get(
    "/activity-feed",
    response_model=ActivityFeedResponse,
    summary="Get memory activity feed",
    description="Get real-time memory activity feed showing recent operations across all 4 memory layers.",
)
async def get_activity_feed(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ActivityFeedResponse:
    """Get memory activity feed.

    Returns recent memory operations showing how the 4-layer memory system
    is being used during learning sessions. This is useful for the playground
    to visualize context building in real-time.

    Activities include:
    - Episodic: Event recording (answers, hints, emotions)
    - Semantic: Mastery updates, topic state changes
    - Procedural: Learning strategy updates, preferences
    - Associative: Interest connections, analogies

    Args:
        limit: Maximum activities to return (default: 20, max: 100).
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ActivityFeedResponse with recent activities and layer status.
    """
    logger.info(
        "Getting memory activity feed: user=%s, limit=%d",
        current_user.id,
        limit,
    )

    student_id = UUID(current_user.id)

    # Get recent episodic events
    activities = await _get_recent_memory_activities(db, student_id, limit)

    # Determine which layers are active (have recent activity)
    layers_active = {
        "episodic": any(a.layer == "episodic" for a in activities),
        "semantic": any(a.layer == "semantic" for a in activities),
        "procedural": any(a.layer == "procedural" for a in activities),
        "associative": any(a.layer == "associative" for a in activities),
    }

    return ActivityFeedResponse(
        student_id=student_id,
        activities=activities,
        total_count=len(activities),
        layers_active=layers_active,
        context_building={
            "episodic_events": len([a for a in activities if a.layer == "episodic"]),
            "semantic_updates": len([a for a in activities if a.layer == "semantic"]),
            "procedural_patterns": len([a for a in activities if a.layer == "procedural"]),
            "associative_connections": len([a for a in activities if a.layer == "associative"]),
            "total_context_items": len(activities),
            "is_building": len(activities) > 0,
        },
    )


async def _get_recent_memory_activities(
    db: AsyncSession,
    student_id: UUID,
    limit: int,
) -> list[ActivityFeedItem]:
    """Get recent memory activities from episodic memory.

    Args:
        db: Database session.
        student_id: Student ID.
        limit: Maximum activities to return.

    Returns:
        List of activity feed items.
    """
    from sqlalchemy import select, desc
    from src.infrastructure.database.models.tenant.memory import (
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        AssociativeMemory,
    )

    activities: list[ActivityFeedItem] = []

    # Get recent episodic memories
    try:
        result = await db.execute(
            select(EpisodicMemory)
            .where(EpisodicMemory.student_id == str(student_id))
            .order_by(desc(EpisodicMemory.created_at))
            .limit(limit // 2)
        )
        episodic_items = result.scalars().all()

        for item in episodic_items:
            topic_full_code = item.topic_full_code
            activities.append(
                ActivityFeedItem(
                    id=f"ep-{item.id}",
                    timestamp=item.created_at,
                    layer="episodic",
                    operation="create",
                    description=_describe_episodic_event(item),
                    entity_type=item.event_type,
                    entity_name=topic_full_code,
                    data={
                        "event_type": item.event_type,
                        "importance": float(item.importance),
                        "topic": topic_full_code,
                        "emotional_state": item.emotional_state,
                    },
                    source_workflow=str(item.session_id) if item.session_id else None,
                )
            )
    except Exception as e:
        logger.debug("Could not load episodic memories: %s", e)

    # Get recent mastery updates (semantic layer)
    try:
        result = await db.execute(
            select(SemanticMemory)
            .where(SemanticMemory.student_id == str(student_id))
            .order_by(desc(SemanticMemory.updated_at))
            .limit(limit // 4)
        )
        mastery_items = result.scalars().all()

        for item in mastery_items:
            entity_name = item.entity_full_code.split(".")[-1] if item.entity_full_code else ""
            activities.append(
                ActivityFeedItem(
                    id=f"sem-{item.id}",
                    timestamp=item.updated_at,
                    layer="semantic",
                    operation="update",
                    description=f"Mastery updated for {item.entity_full_code}: {float(item.mastery_level):.0%}",
                    entity_type="topic_mastery",
                    entity_name=item.entity_full_code,
                    data={
                        "mastery_level": float(item.mastery_level),
                        "confidence": float(item.confidence),
                        "attempts": item.attempts_total,
                        "streak": item.current_streak,
                    },
                    source_workflow=None,
                )
            )
    except Exception as e:
        logger.debug("Could not load mastery items: %s", e)

    # Get recent procedural patterns
    try:
        result = await db.execute(
            select(ProceduralMemory)
            .where(ProceduralMemory.student_id == str(student_id))
            .order_by(desc(ProceduralMemory.updated_at))
            .limit(limit // 8)
        )
        procedural_items = result.scalars().all()

        for item in procedural_items:
            activities.append(
                ActivityFeedItem(
                    id=f"proc-{item.id}",
                    timestamp=item.updated_at,
                    layer="procedural",
                    operation="update",
                    description=f"Learning pattern updated: {item.strategy_type} = {item.strategy_value}",
                    entity_type=item.strategy_type,
                    entity_name=item.strategy_value,
                    data={
                        "strategy_type": item.strategy_type,
                        "strategy_value": item.strategy_value,
                        "effectiveness": float(item.effectiveness),
                        "sample_size": item.sample_size,
                    },
                    source_workflow=None,
                )
            )
    except Exception as e:
        logger.debug("Could not load procedural memories: %s", e)

    # Get recent associative connections
    try:
        result = await db.execute(
            select(AssociativeMemory)
            .where(AssociativeMemory.student_id == str(student_id))
            .order_by(desc(AssociativeMemory.updated_at))
            .limit(limit // 8)
        )
        associative_items = result.scalars().all()

        for item in associative_items:
            activities.append(
                ActivityFeedItem(
                    id=f"assoc-{item.id}",
                    timestamp=item.updated_at,
                    layer="associative",
                    operation="update",
                    description=f"Interest connection: {item.content[:50]}...",
                    entity_type=item.association_type,
                    entity_name=item.content[:30],
                    data={
                        "association_type": item.association_type,
                        "content": item.content,
                        "strength": float(item.strength),
                        "times_used": item.times_used,
                    },
                    source_workflow=None,
                )
            )
    except Exception as e:
        logger.debug("Could not load associative memories: %s", e)

    # Sort by timestamp and limit
    activities.sort(key=lambda a: a.timestamp, reverse=True)
    return activities[:limit]


def _describe_episodic_event(item: Any) -> str:
    """Generate human-readable description for episodic event.

    Args:
        item: EpisodicMemory instance.

    Returns:
        Human-readable description.
    """
    event_type = getattr(item, "event_type", "event")
    # Use topic_full_code property if available
    topic = getattr(item, "topic_full_code", None) or ""
    # Get short topic name for display
    topic_display = topic.split(".")[-1] if topic else ""

    descriptions = {
        "correct_answer": f"Answered correctly on {topic_display}" if topic_display else "Answered correctly",
        "incorrect_answer": f"Answered incorrectly on {topic_display}" if topic_display else "Answered incorrectly",
        "hint_requested": f"Requested a hint for {topic_display}" if topic_display else "Requested a hint",
        "session_started": f"Started learning {topic_display}" if topic_display else "Started learning session",
        "session_completed": f"Completed session for {topic_display}" if topic_display else "Completed session",
        "understanding_expressed": f"Expressed understanding of {topic_display}" if topic_display else "Expressed understanding",
        "confusion_detected": f"Showed confusion about {topic_display}" if topic_display else "Showed confusion",
        "emotional_state": f"Emotional state recorded: {getattr(item, 'emotional_state', 'unknown')}",
        "struggle": f"Struggled with {topic_display}" if topic_display else "Struggled",
        "success": f"Succeeded on {topic_display}" if topic_display else "Succeeded",
        "insight": f"Had insight about {topic_display}" if topic_display else "Had insight",
    }

    return descriptions.get(event_type, f"{event_type} on {topic_display}" if topic_display else event_type)
