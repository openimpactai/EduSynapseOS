# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Semantic memory layer for knowledge state tracking.

This module implements the semantic memory layer which tracks mastery levels,
knowledge state, and learning progress for curriculum entities (topics,
learning objectives).

Uses code-based composite keys from Central Curriculum structure.
entity_full_code format: "UK-NC-2014.MAT.Y4.NPV.001" (for topics)

Uses PostgreSQL only (no vector embeddings) as semantic memories are
structured knowledge states rather than text-based memories.

Session Injection Pattern:
    All methods accept an optional `session` parameter. When provided,
    operations use the given session (sharing transaction context).
    When not provided, a new session is created for the operation.

Example:
    layer = SemanticMemoryLayer(tenant_db_manager=tenant_db)

    # Without session injection (creates own session)
    memory = await layer.record_attempt(
        tenant_code="acme",
        student_id=student_uuid,
        entity_type=EntityType.TOPIC,
        entity_full_code="UK-NC-2014.MAT.Y4.NPV.001",
        is_correct=True,
        time_seconds=45,
    )

    # With session injection (uses provided session)
    async with tenant_db.get_session("acme") as session:
        memory = await layer.record_attempt(
            tenant_code="acme",
            student_id=student_uuid,
            ...,
            session=session,
        )
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.infrastructure.database.models.tenant.memory import SemanticMemory
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.models.memory import (
    EntityType,
    MasteryBySubject,
    MasteryOverview,
    SemanticMemoryResponse,
    TopicMastery,
)
from src.utils.datetime import ensure_utc, utc_now

logger = logging.getLogger(__name__)


class SemanticMemoryError(Exception):
    """Exception raised for semantic memory operations.

    Attributes:
        message: Error description.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class SemanticMemoryLayer:
    """Service layer for semantic memory operations.

    This layer manages knowledge state and mastery tracking for students.
    All data is stored in PostgreSQL without vector embeddings.

    Uses code-based composite keys from Central Curriculum structure:
    - entity_full_code: Full code path (e.g., "UK-NC-2014.MAT.Y4.NPV.001")

    Session Injection:
        All methods accept an optional `session` parameter for transaction
        sharing. This enables callers to pass an existing database session
        to avoid creating new connections and to share transaction context.

    Attributes:
        tenant_db_manager: Manager for tenant database connections.

    Example:
        layer = SemanticMemoryLayer(tenant_db_manager=tenant_db)

        # Standalone operation (creates own session)
        memory = await layer.get_or_create_by_full_code(
            tenant_code="acme",
            student_id=uuid.UUID("..."),
            entity_type=EntityType.TOPIC,
            entity_full_code="UK-NC-2014.MAT.Y4.NPV.001",
        )

        # With session injection (shares transaction)
        async with tenant_db.get_session("acme") as session:
            memory = await layer.get_or_create_by_full_code(
                tenant_code="acme",
                student_id=uuid.UUID("..."),
                entity_type=EntityType.TOPIC,
                entity_full_code="UK-NC-2014.MAT.Y4.NPV.001",
                session=session,
            )
    """

    def __init__(
        self,
        tenant_db_manager: TenantDatabaseManager,
    ) -> None:
        """Initialize the semantic memory layer.

        Args:
            tenant_db_manager: Manager for tenant database connections.
        """
        self._tenant_db = tenant_db_manager

    async def get_or_create_by_full_code(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str,
        entity_full_code: str,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse:
        """Get or create a semantic memory for an entity using full code.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Type of curriculum entity.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            SemanticMemoryResponse for the entity.
        """
        entity_type_str = (
            entity_type.value if isinstance(entity_type, EntityType) else entity_type
        )

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == entity_type_str,
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                now = utc_now()
                memory = SemanticMemory(
                    id=str(uuid.uuid4()),
                    student_id=str(student_id),
                    entity_type=entity_type_str,
                    entity_full_code=entity_full_code,
                    mastery_level=Decimal("0.00"),
                    attempts_total=0,
                    attempts_correct=0,
                    total_time_seconds=0,
                    confidence=Decimal("0.50"),
                    current_streak=0,
                    best_streak=0,
                    created_at=now,
                    updated_at=now,
                )
                db.add(memory)
                # Flush to ensure the record is visible to other operations
                # using the same session (prevents race conditions)
                await db.flush()

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def record_attempt(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str,
        entity_full_code: str,
        is_correct: bool,
        time_seconds: int = 0,
        update_mastery: bool = True,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse:
        """Record a practice attempt for an entity.

        Updates attempt counts, streaks, and optionally mastery level.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Type of curriculum entity.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            is_correct: Whether the attempt was correct.
            time_seconds: Time spent on the attempt.
            update_mastery: Whether to recalculate mastery level.
            session: Optional database session for transaction sharing.

        Returns:
            Updated SemanticMemoryResponse.
        """
        entity_type_str = (
            entity_type.value if isinstance(entity_type, EntityType) else entity_type
        )

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == entity_type_str,
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                now = utc_now()
                memory = SemanticMemory(
                    id=str(uuid.uuid4()),
                    student_id=str(student_id),
                    entity_type=entity_type_str,
                    entity_full_code=entity_full_code,
                    mastery_level=Decimal("0.00"),
                    attempts_total=0,
                    attempts_correct=0,
                    total_time_seconds=0,
                    confidence=Decimal("0.50"),
                    current_streak=0,
                    best_streak=0,
                    created_at=now,
                    updated_at=now,
                )
                db.add(memory)
                # Flush to ensure the record is visible to other operations
                # using the same session (prevents race conditions with FSRS scheduling)
                await db.flush()

            # Record the attempt
            memory.record_attempt(is_correct, time_seconds)

            # Update mastery level if requested
            if update_mastery:
                new_mastery = self._calculate_mastery(
                    attempts_correct=memory.attempts_correct,
                    attempts_total=memory.attempts_total,
                    current_streak=memory.current_streak,
                    best_streak=memory.best_streak,
                )
                memory.mastery_level = Decimal(str(new_mastery))

                # Update confidence based on sample size
                confidence = min(0.95, 0.5 + (memory.attempts_total * 0.05))
                memory.confidence = Decimal(str(confidence))

            logger.debug(
                "Recorded attempt for student %s on %s/%s: correct=%s, mastery=%.2f",
                student_id,
                entity_type_str,
                entity_full_code,
                is_correct,
                float(memory.mastery_level),
            )

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_by_entity(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str,
        entity_full_code: str,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse | None:
        """Get semantic memory for a specific entity.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Type of curriculum entity.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            SemanticMemoryResponse if found, None otherwise.
        """
        entity_type_str = (
            entity_type.value if isinstance(entity_type, EntityType) else entity_type
        )

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse | None:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == entity_type_str,
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_all_for_student(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str | None = None,
        session: AsyncSession | None = None,
    ) -> list[SemanticMemoryResponse]:
        """Get all semantic memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Optional filter by entity type.
            session: Optional database session for transaction sharing.

        Returns:
            List of semantic memory responses.
        """

        async def _execute(db: AsyncSession) -> list[SemanticMemoryResponse]:
            query = select(SemanticMemory).where(
                SemanticMemory.student_id == str(student_id)
            )

            if entity_type:
                entity_type_str = (
                    entity_type.value
                    if isinstance(entity_type, EntityType)
                    else entity_type
                )
                query = query.where(SemanticMemory.entity_type == entity_type_str)

            query = query.order_by(desc(SemanticMemory.last_practiced_at))

            result = await db.execute(query)
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_mastery_overview(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> MasteryOverview:
        """Get mastery overview for a student.

        Calculates overall mastery, categorizes topics by mastery level,
        and provides subject-level breakdowns.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            MasteryOverview with aggregated statistics.
        """

        async def _execute(db: AsyncSession) -> MasteryOverview:
            # Get all topic-level memories
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == EntityType.TOPIC.value,
                    )
                )
            )
            memories = result.scalars().all()

            if not memories:
                return MasteryOverview(
                    student_id=student_id,
                    overall_mastery=0.0,
                    topics_mastered=0,
                    topics_learning=0,
                    topics_struggling=0,
                    total_topics=0,
                    by_subject={},
                )

            # Calculate statistics
            total = len(memories)
            mastered = sum(1 for m in memories if float(m.mastery_level) >= 0.8)
            learning = sum(
                1 for m in memories if 0.3 <= float(m.mastery_level) < 0.8
            )
            struggling = sum(1 for m in memories if float(m.mastery_level) < 0.3)

            # Calculate weighted overall mastery
            total_weight = sum(m.attempts_total for m in memories)
            if total_weight > 0:
                overall = sum(
                    float(m.mastery_level) * m.attempts_total for m in memories
                ) / total_weight
            else:
                overall = sum(float(m.mastery_level) for m in memories) / total

            return MasteryOverview(
                student_id=student_id,
                overall_mastery=round(overall, 3),
                topics_mastered=mastered,
                topics_learning=learning,
                topics_struggling=struggling,
                total_topics=total,
                by_subject={},  # Requires joining with curriculum data
            )

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def get_mastery_overview_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
    ) -> MasteryOverview:
        """Get mastery overview for a student (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Calculates overall mastery, categorizes topics by mastery level,
        and provides subject-level breakdowns.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.

        Returns:
            MasteryOverview with aggregated statistics.
        """

        def _execute(db: Session) -> MasteryOverview:
            # Get all topic-level memories
            result = db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == EntityType.TOPIC.value,
                    )
                )
            )
            memories = result.scalars().all()

            if not memories:
                return MasteryOverview(
                    student_id=student_id,
                    overall_mastery=0.0,
                    topics_mastered=0,
                    topics_learning=0,
                    topics_struggling=0,
                    total_topics=0,
                    by_subject={},
                )

            # Calculate statistics
            total = len(memories)
            mastered = sum(1 for m in memories if float(m.mastery_level) >= 0.8)
            learning = sum(
                1 for m in memories if 0.3 <= float(m.mastery_level) < 0.8
            )
            struggling = sum(1 for m in memories if float(m.mastery_level) < 0.3)

            # Calculate weighted overall mastery
            total_weight = sum(m.attempts_total for m in memories)
            if total_weight > 0:
                overall = sum(
                    float(m.mastery_level) * m.attempts_total for m in memories
                ) / total_weight
            else:
                overall = sum(float(m.mastery_level) for m in memories) / total

            return MasteryOverview(
                student_id=student_id,
                overall_mastery=round(overall, 3),
                topics_mastered=mastered,
                topics_learning=learning,
                topics_struggling=struggling,
                total_topics=total,
                by_subject={},  # Requires joining with curriculum data
            )

        with self._tenant_db.get_sync_session(tenant_code) as db:
            return _execute(db)

    async def get_by_entity_full_code(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_full_code: str,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse | None:
        """Get semantic memory by entity full code.

        This is a convenience method for getting topic mastery by full code.
        It assumes the entity is a topic (most common use case in practice).

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MATHS.Y4...").
            session: Optional database session for transaction sharing.

        Returns:
            SemanticMemoryResponse if found, None otherwise.
        """

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse | None:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def get_by_entity_full_code_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_full_code: str,
    ) -> SemanticMemoryResponse | None:
        """Get semantic memory by entity full code (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        This is a convenience method for getting topic mastery by full code.
        It assumes the entity is a topic (most common use case in practice).

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MATHS.Y4...").

        Returns:
            SemanticMemoryResponse if found, None otherwise.
        """

        def _execute(db: Session) -> SemanticMemoryResponse | None:
            result = db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            return self._to_response(memory)

        with self._tenant_db.get_sync_session(tenant_code) as db:
            return _execute(db)

    async def get_subject_average_mastery(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        subject_full_code: str,
        session: AsyncSession | None = None,
    ) -> float | None:
        """Calculate average mastery for all topics under a subject.

        This is used for subject-level practice where no specific topic
        is selected. Returns weighted average based on attempts.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            subject_full_code: Subject's full code (e.g., "UK-NC-2014.MATHS").
            session: Optional database session for transaction sharing.

        Returns:
            Average mastery level (0.0-1.0) or None if no topics found.
        """

        async def _execute(db: AsyncSession) -> float | None:
            # Find all topic memories that belong to this subject
            # Subject code is a prefix of topic full code
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == EntityType.TOPIC.value,
                        SemanticMemory.entity_full_code.startswith(subject_full_code + "."),
                    )
                )
            )
            memories = result.scalars().all()

            if not memories:
                return None

            # Calculate weighted average based on attempts
            total_weight = sum(m.attempts_total for m in memories)
            if total_weight > 0:
                average = sum(
                    float(m.mastery_level) * m.attempts_total for m in memories
                ) / total_weight
            else:
                average = sum(float(m.mastery_level) for m in memories) / len(memories)

            return round(average, 3)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_weak_areas(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        max_mastery: float = 0.5,
        limit: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SemanticMemoryResponse]:
        """Get entities where student is struggling.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            max_mastery: Maximum mastery threshold.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of weak area memories ordered by mastery (lowest first).
        """

        async def _execute(db: AsyncSession) -> list[SemanticMemoryResponse]:
            result = await db.execute(
                select(SemanticMemory)
                .where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.mastery_level < Decimal(str(max_mastery)),
                        SemanticMemory.attempts_total > 0,
                    )
                )
                .order_by(SemanticMemory.mastery_level)
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_strong_areas(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_mastery: float = 0.8,
        limit: int = 10,
        session: AsyncSession | None = None,
    ) -> list[SemanticMemoryResponse]:
        """Get entities where student has high mastery.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_mastery: Minimum mastery threshold.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of strong area memories ordered by mastery (highest first).
        """

        async def _execute(db: AsyncSession) -> list[SemanticMemoryResponse]:
            result = await db.execute(
                select(SemanticMemory)
                .where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.mastery_level >= Decimal(str(min_mastery)),
                    )
                )
                .order_by(desc(SemanticMemory.mastery_level))
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_due_for_review(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        limit: int = 20,
        session: AsyncSession | None = None,
    ) -> list[SemanticMemoryResponse]:
        """Get entities due for spaced repetition review.

        Uses FSRS due dates when available, falls back to time since
        last practice.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of memories due for review.
        """
        now = datetime.now(timezone.utc)

        async def _execute(db: AsyncSession) -> list[SemanticMemoryResponse]:
            # Prioritize by FSRS due date if available, otherwise by last_practiced_at
            result = await db.execute(
                select(SemanticMemory)
                .where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.attempts_total > 0,
                    )
                )
                .order_by(
                    # FSRS due items first (nulls last)
                    case((SemanticMemory.fsrs_due_at <= now, 0), else_=1),
                    SemanticMemory.fsrs_due_at.nullsfirst(),
                    SemanticMemory.last_practiced_at.nullsfirst(),
                )
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def update_fsrs_parameters(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        stability: float,
        difficulty: float,
        state: str,
        step: int,
        due_at: datetime,
        last_review: datetime,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse | None:
        """Update FSRS spaced repetition parameters.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            stability: FSRS stability parameter (memory strength in days).
            difficulty: FSRS difficulty parameter.
            state: FSRS card state (new, learning, review, relearning).
            step: FSRS learning step count.
            due_at: Next review due date.
            last_review: Last review timestamp.
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response, or None if not found.
        """

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse | None:
            result = await db.execute(
                select(SemanticMemory).where(SemanticMemory.id == str(memory_id))
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            memory.fsrs_stability = Decimal(str(stability))
            memory.fsrs_difficulty = Decimal(str(difficulty))
            memory.fsrs_state = state
            memory.fsrs_step = step
            memory.fsrs_due_at = ensure_utc(due_at)
            memory.fsrs_last_review = ensure_utc(last_review)

            logger.debug(
                "Updated FSRS parameters for memory %s: stability=%.4f, "
                "difficulty=%.4f, state=%s, step=%d, due=%s",
                memory_id,
                stability,
                difficulty,
                state,
                step,
                due_at.isoformat(),
            )

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def set_mastery(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str,
        entity_full_code: str,
        mastery_level: float,
        reason: str | None = None,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse:
        """Manually set mastery level (admin operation).

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Type of curriculum entity.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            mastery_level: New mastery level (0-1).
            reason: Reason for manual override.
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response.
        """
        entity_type_str = (
            entity_type.value if isinstance(entity_type, EntityType) else entity_type
        )

        async def _execute(db: AsyncSession) -> SemanticMemoryResponse:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == entity_type_str,
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                now = utc_now()
                memory = SemanticMemory(
                    id=str(uuid.uuid4()),
                    student_id=str(student_id),
                    entity_type=entity_type_str,
                    entity_full_code=entity_full_code,
                    mastery_level=Decimal(str(mastery_level)),
                    attempts_total=0,
                    attempts_correct=0,
                    total_time_seconds=0,
                    confidence=Decimal("1.00"),  # High confidence for manual set
                    current_streak=0,
                    best_streak=0,
                    created_at=now,
                    updated_at=now,
                )
                db.add(memory)
            else:
                memory.mastery_level = Decimal(str(mastery_level))
                memory.confidence = Decimal("1.00")

            logger.info(
                "Manually set mastery for student %s on %s/%s to %.2f: %s",
                student_id,
                entity_type_str,
                entity_full_code,
                mastery_level,
                reason or "no reason provided",
            )

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def reset_for_entity(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        entity_type: EntityType | str,
        entity_full_code: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Reset semantic memory for an entity.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            entity_type: Type of curriculum entity.
            entity_full_code: Entity's full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            True if reset, False if not found.
        """
        entity_type_str = (
            entity_type.value if isinstance(entity_type, EntityType) else entity_type
        )

        async def _execute(db: AsyncSession) -> bool:
            result = await db.execute(
                select(SemanticMemory).where(
                    and_(
                        SemanticMemory.student_id == str(student_id),
                        SemanticMemory.entity_type == entity_type_str,
                        SemanticMemory.entity_full_code == entity_full_code,
                    )
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return False

            await db.delete(memory)
            return True

        if session:
            deleted = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                deleted = await _execute(db)

        if deleted:
            logger.info(
                "Reset semantic memory for student %s on %s/%s",
                student_id,
                entity_type_str,
                entity_full_code,
            )
        return deleted

    def _calculate_mastery(
        self,
        attempts_correct: int,
        attempts_total: int,
        current_streak: int,
        best_streak: int,
    ) -> float:
        """Calculate mastery level from attempt statistics.

        Uses a weighted formula considering:
        - Accuracy (60% weight)
        - Current streak bonus (25% weight)
        - Best streak bonus (15% weight)

        Args:
            attempts_correct: Number of correct attempts.
            attempts_total: Total number of attempts.
            current_streak: Current correct answer streak.
            best_streak: Best ever correct answer streak.

        Returns:
            Mastery level between 0 and 1.
        """
        if attempts_total == 0:
            return 0.0

        # Base accuracy
        accuracy = attempts_correct / attempts_total

        # Streak bonuses (capped)
        streak_bonus = min(current_streak / 10, 0.25)  # Max 25% bonus
        best_streak_bonus = min(best_streak / 20, 0.15)  # Max 15% bonus

        # Weighted mastery
        mastery = (accuracy * 0.6) + (streak_bonus) + (best_streak_bonus)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, mastery))

    def _to_response(self, memory: SemanticMemory) -> SemanticMemoryResponse:
        """Convert database model to response DTO.

        Args:
            memory: Database model instance.

        Returns:
            SemanticMemoryResponse DTO.

        Note:
            entity_name is None because this layer doesn't join with
            curriculum tables. Use tools like get_my_mastery which
            join directly for name resolution.
        """
        return SemanticMemoryResponse(
            id=uuid.UUID(memory.id),
            student_id=uuid.UUID(memory.student_id),
            entity_type=EntityType(memory.entity_type),
            entity_full_code=memory.entity_full_code,
            entity_name=None,  # Requires join with curriculum tables
            mastery_level=float(memory.mastery_level),
            attempts_total=memory.attempts_total,
            attempts_correct=memory.attempts_correct,
            total_time_seconds=memory.total_time_seconds,
            confidence=float(memory.confidence),
            last_practiced_at=ensure_utc(memory.last_practiced_at),
            current_streak=memory.current_streak,
            best_streak=memory.best_streak,
            fsrs_stability=float(memory.fsrs_stability) if memory.fsrs_stability else None,
            fsrs_difficulty=float(memory.fsrs_difficulty) if memory.fsrs_difficulty else None,
            fsrs_state=memory.fsrs_state,
            fsrs_step=memory.fsrs_step,
            fsrs_due_at=ensure_utc(memory.fsrs_due_at),
            fsrs_last_review=ensure_utc(memory.fsrs_last_review),
            created_at=ensure_utc(memory.created_at),
            updated_at=ensure_utc(memory.updated_at),
        )
