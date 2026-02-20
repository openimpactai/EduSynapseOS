# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Procedural memory layer for learning strategies and patterns.

This module implements the procedural memory layer which tracks effective
learning strategies, patterns, and preferences for each student. Uses
PostgreSQL only (no vector embeddings) as procedural memories are
structured observations.

Tracks patterns such as optimal study times, preferred content formats,
hint usage patterns, and persona effectiveness.

All public methods support optional session injection to share transaction
context with the caller. If a session is provided, it will be used directly;
otherwise, a new session will be created.

Example:
    layer = ProceduralMemoryLayer(tenant_db_manager=tenant_db)

    # Record a strategy observation
    memory = await layer.record_observation(
        tenant_code="acme",
        student_id=student_uuid,
        strategy_type=StrategyType.TIME_OF_DAY,
        strategy_value="morning",
        was_effective=True,
    )

    # Get learning patterns
    patterns = await layer.get_learning_patterns(
        tenant_code="acme",
        student_id=student_uuid,
    )
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.infrastructure.database.models.base import utc_now
from src.infrastructure.database.models.tenant.memory import ProceduralMemory
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.models.memory import (
    LearningPatterns,
    ProceduralMemoryResponse,
    StrategyType,
    VARKProfile,
)

logger = logging.getLogger(__name__)


class ProceduralMemoryError(Exception):
    """Exception raised for procedural memory operations.

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


class ProceduralMemoryLayer:
    """Service layer for procedural memory operations.

    This layer manages learning strategy observations and pattern analysis.
    All data is stored in PostgreSQL without vector embeddings.

    All public methods support optional session injection to share transaction
    context with the caller.

    Attributes:
        tenant_db_manager: Manager for tenant database connections.

    Example:
        layer = ProceduralMemoryLayer(tenant_db_manager=tenant_db)

        # Record an observation with injected session
        memory = await layer.record_observation(
            tenant_code="acme",
            student_id=uuid.UUID("..."),
            strategy_type=StrategyType.CONTENT_FORMAT,
            strategy_value="visual",
            was_effective=True,
            session=db_session,  # Optional: reuse existing session
        )
    """

    def __init__(
        self,
        tenant_db_manager: TenantDatabaseManager,
    ) -> None:
        """Initialize the procedural memory layer.

        Args:
            tenant_db_manager: Manager for tenant database connections.
        """
        self._tenant_db = tenant_db_manager

    async def record_observation(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        strategy_type: StrategyType | str,
        strategy_value: str,
        was_effective: bool,
        subject_full_code: str | None = None,
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> ProceduralMemoryResponse:
        """Record a learning strategy observation.

        Updates or creates a procedural memory entry for the strategy,
        updating the effectiveness running average.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            strategy_type: Type of learning strategy.
            strategy_value: Specific strategy value (e.g., "morning", "visual").
            was_effective: Whether the strategy was effective.
            subject_full_code: Optional subject full code (e.g., "UK-NC-2014.MAT").
            topic_full_code: Optional topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            ProceduralMemoryResponse with updated effectiveness.
        """
        strategy_type_str = (
            strategy_type.value
            if isinstance(strategy_type, StrategyType)
            else strategy_type
        )

        # Parse topic_full_code into composite key parts
        topic_framework_code = None
        topic_subject_code = None
        topic_grade_code = None
        topic_unit_code = None
        topic_code = None
        if topic_full_code:
            parts = topic_full_code.split(".")
            if len(parts) == 5:
                topic_framework_code = parts[0]
                topic_subject_code = parts[1]
                topic_grade_code = parts[2]
                topic_unit_code = parts[3]
                topic_code = parts[4]

        # Parse subject_full_code into composite key parts
        subject_framework_code = None
        subject_code = None
        if subject_full_code:
            parts = subject_full_code.split(".")
            if len(parts) == 2:
                subject_framework_code = parts[0]
                subject_code = parts[1]

        async def _execute(db: AsyncSession) -> ProceduralMemoryResponse:
            # Find existing entry for this strategy
            conditions = [
                ProceduralMemory.student_id == str(student_id),
                ProceduralMemory.strategy_type == strategy_type_str,
                ProceduralMemory.strategy_value == strategy_value,
            ]

            if topic_full_code:
                conditions.extend([
                    ProceduralMemory.topic_framework_code == topic_framework_code,
                    ProceduralMemory.topic_subject_code == topic_subject_code,
                    ProceduralMemory.topic_grade_code == topic_grade_code,
                    ProceduralMemory.topic_unit_code == topic_unit_code,
                    ProceduralMemory.topic_code == topic_code,
                ])
            else:
                conditions.append(ProceduralMemory.topic_framework_code.is_(None))

            result = await db.execute(
                select(ProceduralMemory).where(and_(*conditions))
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                # Create new entry with explicit timestamps
                now = utc_now()
                memory = ProceduralMemory(
                    id=str(uuid.uuid4()),
                    student_id=str(student_id),
                    strategy_type=strategy_type_str,
                    strategy_value=strategy_value,
                    effectiveness=Decimal("1.0" if was_effective else "0.0"),
                    sample_size=1,
                    subject_framework_code=subject_framework_code,
                    subject_code=subject_code,
                    topic_framework_code=topic_framework_code,
                    topic_subject_code=topic_subject_code,
                    topic_grade_code=topic_grade_code,
                    topic_unit_code=topic_unit_code,
                    topic_code=topic_code,
                    last_observation_at=now,
                    created_at=now,
                    updated_at=now,
                )
                db.add(memory)
            else:
                # Update existing entry
                memory.record_observation(was_effective)

            logger.debug(
                "Recorded observation for student %s: %s=%s, effective=%s, "
                "new_effectiveness=%.2f",
                student_id,
                strategy_type_str,
                strategy_value,
                was_effective,
                float(memory.effectiveness),
            )

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_by_strategy(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        strategy_type: StrategyType | str,
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> list[ProceduralMemoryResponse]:
        """Get all strategies of a specific type for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            strategy_type: Type of learning strategy.
            topic_full_code: Optional topic full code filter (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            List of strategy memories ordered by effectiveness.
        """
        strategy_type_str = (
            strategy_type.value
            if isinstance(strategy_type, StrategyType)
            else strategy_type
        )

        async def _execute(db: AsyncSession) -> list[ProceduralMemoryResponse]:
            conditions = [
                ProceduralMemory.student_id == str(student_id),
                ProceduralMemory.strategy_type == strategy_type_str,
            ]

            if topic_full_code:
                parts = topic_full_code.split(".")
                if len(parts) == 5:
                    conditions.extend([
                        ProceduralMemory.topic_framework_code == parts[0],
                        ProceduralMemory.topic_subject_code == parts[1],
                        ProceduralMemory.topic_grade_code == parts[2],
                        ProceduralMemory.topic_unit_code == parts[3],
                        ProceduralMemory.topic_code == parts[4],
                    ])

            query = select(ProceduralMemory).where(and_(*conditions))
            query = query.order_by(desc(ProceduralMemory.effectiveness))

            result = await db.execute(query)
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_best_strategy(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        strategy_type: StrategyType | str,
        min_samples: int = 3,
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> ProceduralMemoryResponse | None:
        """Get the most effective strategy of a type.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            strategy_type: Type of learning strategy.
            min_samples: Minimum observations required.
            topic_full_code: Optional topic full code filter (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            Most effective strategy or None if not enough data.
        """
        strategy_type_str = (
            strategy_type.value
            if isinstance(strategy_type, StrategyType)
            else strategy_type
        )

        async def _execute(db: AsyncSession) -> ProceduralMemoryResponse | None:
            conditions = [
                ProceduralMemory.student_id == str(student_id),
                ProceduralMemory.strategy_type == strategy_type_str,
                ProceduralMemory.sample_size >= min_samples,
            ]

            if topic_full_code:
                parts = topic_full_code.split(".")
                if len(parts) == 5:
                    conditions.extend([
                        ProceduralMemory.topic_framework_code == parts[0],
                        ProceduralMemory.topic_subject_code == parts[1],
                        ProceduralMemory.topic_grade_code == parts[2],
                        ProceduralMemory.topic_unit_code == parts[3],
                        ProceduralMemory.topic_code == parts[4],
                    ])

            query = select(ProceduralMemory).where(and_(*conditions))
            query = query.order_by(desc(ProceduralMemory.effectiveness)).limit(1)

            result = await db.execute(query)
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
        session: AsyncSession | None = None,
    ) -> list[ProceduralMemoryResponse]:
        """Get all procedural memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            List of all strategy memories.
        """
        async def _execute(db: AsyncSession) -> list[ProceduralMemoryResponse]:
            result = await db.execute(
                select(ProceduralMemory)
                .where(ProceduralMemory.student_id == str(student_id))
                .order_by(
                    ProceduralMemory.strategy_type,
                    desc(ProceduralMemory.effectiveness),
                )
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_learning_patterns(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_samples: int = 5,
        session: AsyncSession | None = None,
    ) -> LearningPatterns:
        """Get aggregated learning patterns for a student.

        Analyzes all procedural memories to extract the most effective
        strategies for each category.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_samples: Minimum observations for confidence.
            session: Optional database session for transaction sharing.

        Returns:
            LearningPatterns with optimal strategies.
        """
        async def _execute(db: AsyncSession) -> LearningPatterns:
            # Get all memories with sufficient samples
            result = await db.execute(
                select(ProceduralMemory)
                .where(
                    and_(
                        ProceduralMemory.student_id == str(student_id),
                        ProceduralMemory.sample_size >= min_samples,
                    )
                )
                .order_by(
                    ProceduralMemory.strategy_type,
                    desc(ProceduralMemory.effectiveness),
                )
            )
            memories = result.scalars().all()

            # Group by strategy type and get best
            best_by_type: dict[str, ProceduralMemory] = {}
            for memory in memories:
                if memory.strategy_type not in best_by_type:
                    best_by_type[memory.strategy_type] = memory

            # Extract patterns
            time_of_day = best_by_type.get(StrategyType.TIME_OF_DAY.value)
            session_duration = best_by_type.get(StrategyType.SESSION_DURATION.value)
            content_format = best_by_type.get(StrategyType.CONTENT_FORMAT.value)
            hint_usage = best_by_type.get(StrategyType.HINT_USAGE.value)
            break_frequency = best_by_type.get(StrategyType.BREAK_FREQUENCY.value)
            difficulty_pref = best_by_type.get(StrategyType.DIFFICULTY_PREFERENCE.value)
            persona_pref = best_by_type.get(StrategyType.PERSONA_PREFERENCE.value)

            return LearningPatterns(
                student_id=student_id,
                best_time_of_day=(
                    time_of_day.strategy_value if time_of_day else None
                ),
                optimal_session_duration=(
                    int(session_duration.strategy_value)
                    if session_duration and session_duration.strategy_value.isdigit()
                    else None
                ),
                preferred_content_format=(
                    content_format.strategy_value if content_format else None
                ),
                hint_dependency=(
                    float(hint_usage.effectiveness) if hint_usage else None
                ),
                avg_break_frequency=(
                    int(break_frequency.strategy_value)
                    if break_frequency and break_frequency.strategy_value.isdigit()
                    else None
                ),
                preferred_difficulty=(
                    float(difficulty_pref.strategy_value)
                    if difficulty_pref
                    else None
                ),
                favorite_persona=(
                    persona_pref.strategy_value if persona_pref else None
                ),
                vark_profile=await self._calculate_vark_profile(
                    tenant_code, student_id, min_samples, session=db
                ),
            )

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def get_learning_patterns_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_samples: int = 5,
    ) -> LearningPatterns:
        """Get aggregated learning patterns for a student (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Analyzes all procedural memories to extract the most effective
        strategies for each category.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_samples: Minimum observations for confidence.

        Returns:
            LearningPatterns with optimal strategies.
        """
        def _execute(db: Session) -> LearningPatterns:
            # Get all memories with sufficient samples
            result = db.execute(
                select(ProceduralMemory)
                .where(
                    and_(
                        ProceduralMemory.student_id == str(student_id),
                        ProceduralMemory.sample_size >= min_samples,
                    )
                )
                .order_by(
                    ProceduralMemory.strategy_type,
                    desc(ProceduralMemory.effectiveness),
                )
            )
            memories = result.scalars().all()

            # Group by strategy type and get best
            best_by_type: dict[str, ProceduralMemory] = {}
            for memory in memories:
                if memory.strategy_type not in best_by_type:
                    best_by_type[memory.strategy_type] = memory

            # Extract patterns
            time_of_day = best_by_type.get(StrategyType.TIME_OF_DAY.value)
            session_duration = best_by_type.get(StrategyType.SESSION_DURATION.value)
            content_format = best_by_type.get(StrategyType.CONTENT_FORMAT.value)
            hint_usage = best_by_type.get(StrategyType.HINT_USAGE.value)
            break_frequency = best_by_type.get(StrategyType.BREAK_FREQUENCY.value)
            difficulty_pref = best_by_type.get(StrategyType.DIFFICULTY_PREFERENCE.value)
            persona_pref = best_by_type.get(StrategyType.PERSONA_PREFERENCE.value)

            return LearningPatterns(
                student_id=student_id,
                best_time_of_day=(
                    time_of_day.strategy_value if time_of_day else None
                ),
                optimal_session_duration=(
                    int(session_duration.strategy_value)
                    if session_duration and session_duration.strategy_value.isdigit()
                    else None
                ),
                preferred_content_format=(
                    content_format.strategy_value if content_format else None
                ),
                hint_dependency=(
                    float(hint_usage.effectiveness) if hint_usage else None
                ),
                avg_break_frequency=(
                    int(break_frequency.strategy_value)
                    if break_frequency and break_frequency.strategy_value.isdigit()
                    else None
                ),
                preferred_difficulty=(
                    float(difficulty_pref.strategy_value)
                    if difficulty_pref
                    else None
                ),
                favorite_persona=(
                    persona_pref.strategy_value if persona_pref else None
                ),
                vark_profile=self._calculate_vark_profile_sync(
                    tenant_code, student_id, min_samples, db
                ),
            )

        with self._tenant_db.get_sync_session(tenant_code) as db:
            return _execute(db)

    async def _calculate_vark_profile(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_samples: int,
        session: AsyncSession | None = None,
    ) -> VARKProfile | None:
        """Calculate VARK learning style profile.

        Based on content format effectiveness observations.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_samples: Minimum observations for each style.
            session: Optional database session for transaction sharing.

        Returns:
            VARKProfile if enough data, None otherwise.
        """
        async def _execute(db: AsyncSession) -> VARKProfile | None:
            result = await db.execute(
                select(ProceduralMemory)
                .where(
                    and_(
                        ProceduralMemory.student_id == str(student_id),
                        ProceduralMemory.strategy_type
                        == StrategyType.CONTENT_FORMAT.value,
                        ProceduralMemory.sample_size >= min_samples,
                    )
                )
            )
            memories = result.scalars().all()

            if len(memories) < 2:
                return None

            # Map content formats to VARK
            vark_mapping = {
                "visual": "visual",
                "diagram": "visual",
                "chart": "visual",
                "video": "visual",
                "audio": "auditory",
                "lecture": "auditory",
                "podcast": "auditory",
                "text": "reading_writing",
                "notes": "reading_writing",
                "reading": "reading_writing",
                "practice": "kinesthetic",
                "hands_on": "kinesthetic",
                "interactive": "kinesthetic",
                "exercise": "kinesthetic",
            }

            # Aggregate scores
            scores = {
                "visual": 0.0,
                "auditory": 0.0,
                "reading_writing": 0.0,
                "kinesthetic": 0.0,
            }
            counts = {
                "visual": 0,
                "auditory": 0,
                "reading_writing": 0,
                "kinesthetic": 0,
            }

            for memory in memories:
                vark_type = vark_mapping.get(memory.strategy_value.lower())
                if vark_type:
                    scores[vark_type] += float(memory.effectiveness) * memory.sample_size
                    counts[vark_type] += memory.sample_size

            # Calculate averages
            for key in scores:
                if counts[key] > 0:
                    scores[key] /= counts[key]

            # Normalize
            total = sum(scores.values())
            if total == 0:
                return None

            for key in scores:
                scores[key] /= total

            # Find dominant
            dominant = max(scores, key=lambda k: scores[k])

            return VARKProfile(
                visual=round(scores["visual"], 3),
                auditory=round(scores["auditory"], 3),
                reading_writing=round(scores["reading_writing"], 3),
                kinesthetic=round(scores["kinesthetic"], 3),
                dominant_style=dominant,
            )

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def _calculate_vark_profile_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_samples: int,
        session: Session,
    ) -> VARKProfile | None:
        """Calculate VARK learning style profile (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Based on content format effectiveness observations.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_samples: Minimum observations for each style.
            session: Database session (required, must be sync session).

        Returns:
            VARKProfile if enough data, None otherwise.
        """
        result = session.execute(
            select(ProceduralMemory)
            .where(
                and_(
                    ProceduralMemory.student_id == str(student_id),
                    ProceduralMemory.strategy_type
                    == StrategyType.CONTENT_FORMAT.value,
                    ProceduralMemory.sample_size >= min_samples,
                )
            )
        )
        memories = result.scalars().all()

        if len(memories) < 2:
            return None

        # Map content formats to VARK
        vark_mapping = {
            "visual": "visual",
            "diagram": "visual",
            "chart": "visual",
            "video": "visual",
            "audio": "auditory",
            "lecture": "auditory",
            "podcast": "auditory",
            "text": "reading_writing",
            "notes": "reading_writing",
            "reading": "reading_writing",
            "practice": "kinesthetic",
            "hands_on": "kinesthetic",
            "interactive": "kinesthetic",
            "exercise": "kinesthetic",
        }

        # Aggregate scores
        scores = {
            "visual": 0.0,
            "auditory": 0.0,
            "reading_writing": 0.0,
            "kinesthetic": 0.0,
        }
        counts = {
            "visual": 0,
            "auditory": 0,
            "reading_writing": 0,
            "kinesthetic": 0,
        }

        for memory in memories:
            vark_type = vark_mapping.get(memory.strategy_value.lower())
            if vark_type:
                scores[vark_type] += float(memory.effectiveness) * memory.sample_size
                counts[vark_type] += memory.sample_size

        # Calculate averages
        for key in scores:
            if counts[key] > 0:
                scores[key] /= counts[key]

        # Normalize
        total = sum(scores.values())
        if total == 0:
            return None

        for key in scores:
            scores[key] /= total

        # Find dominant
        dominant = max(scores, key=lambda k: scores[k])

        return VARKProfile(
            visual=round(scores["visual"], 3),
            auditory=round(scores["auditory"], 3),
            reading_writing=round(scores["reading_writing"], 3),
            kinesthetic=round(scores["kinesthetic"], 3),
            dominant_style=dominant,
        )

    async def get_effective_strategies_for_topic(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic_full_code: str,
        min_effectiveness: float = 0.6,
        session: AsyncSession | None = None,
    ) -> list[ProceduralMemoryResponse]:
        """Get effective strategies for a specific topic.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            min_effectiveness: Minimum effectiveness threshold.
            session: Optional database session for transaction sharing.

        Returns:
            List of effective strategies for the topic.
        """
        # Parse topic_full_code into composite key parts
        parts = topic_full_code.split(".")
        if len(parts) != 5:
            return []

        async def _execute(db: AsyncSession) -> list[ProceduralMemoryResponse]:
            result = await db.execute(
                select(ProceduralMemory)
                .where(
                    and_(
                        ProceduralMemory.student_id == str(student_id),
                        ProceduralMemory.topic_framework_code == parts[0],
                        ProceduralMemory.topic_subject_code == parts[1],
                        ProceduralMemory.topic_grade_code == parts[2],
                        ProceduralMemory.topic_unit_code == parts[3],
                        ProceduralMemory.topic_code == parts[4],
                        ProceduralMemory.effectiveness
                        >= Decimal(str(min_effectiveness)),
                    )
                )
                .order_by(desc(ProceduralMemory.effectiveness))
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def delete_for_student(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete all procedural memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            Number of memories deleted.
        """
        async def _execute(db: AsyncSession) -> int:
            result = await db.execute(
                select(ProceduralMemory).where(
                    ProceduralMemory.student_id == str(student_id)
                )
            )
            memories = result.scalars().all()

            count = len(memories)
            for memory in memories:
                await db.delete(memory)

            return count

        if session:
            count = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                count = await _execute(db)

        logger.info(
            "Deleted %d procedural memories for student %s",
            count,
            student_id,
        )
        return count

    def _to_response(self, memory: ProceduralMemory) -> ProceduralMemoryResponse:
        """Convert database model to response DTO.

        Args:
            memory: Database model instance.

        Returns:
            ProceduralMemoryResponse DTO.
        """
        # Build subject_full_code from composite key parts
        subject_full_code = None
        if memory.subject_framework_code and memory.subject_code:
            subject_full_code = f"{memory.subject_framework_code}.{memory.subject_code}"

        # Build topic_full_code from composite key parts
        topic_full_code = None
        if memory.topic_framework_code and memory.topic_code:
            topic_full_code = (
                f"{memory.topic_framework_code}."
                f"{memory.topic_subject_code}."
                f"{memory.topic_grade_code}."
                f"{memory.topic_unit_code}."
                f"{memory.topic_code}"
            )

        return ProceduralMemoryResponse(
            id=uuid.UUID(memory.id),
            student_id=uuid.UUID(memory.student_id),
            strategy_type=StrategyType(memory.strategy_type),
            strategy_value=memory.strategy_value,
            effectiveness=float(memory.effectiveness),
            sample_size=memory.sample_size,
            subject_full_code=subject_full_code,
            topic_full_code=topic_full_code,
            last_observation_at=memory.last_observation_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )
