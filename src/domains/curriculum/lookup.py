# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Curriculum lookup service for code-based entity resolution.

This module provides centralized lookup utilities for curriculum entities,
using the code-based composite key identification system.

The CurriculumLookup class provides:
- Full-code-to-entity resolution (e.g., "UK-NC-2014.MAT.Y4.NPV.001" -> Topic)
- Component-code-based lookups
- Educational context retrieval (TopicContext)

All curriculum entities use code-based composite primary keys:
- Framework: code (e.g., "UK-NC-2014")
- Stage: framework_code + code
- Grade: framework_code + stage_code + code
- Subject: framework_code + code
- Unit: framework_code + subject_code + grade_code + code
- Topic: framework_code + subject_code + grade_code + unit_code + code
- Objective: framework_code + subject_code + grade_code + unit_code + topic_code + code

Example:
    >>> lookup = CurriculumLookup(db_session)
    >>> topic = await lookup.get_topic("UK-NC-2014", "MAT", "Y4", "NPV", "001")
    >>> context = await lookup.get_topic_context("UK-NC-2014", "MAT", "Y4", "NPV", "001")
"""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.curriculum import (
    CurriculumFramework,
    CurriculumStage,
    GradeLevel,
    Subject,
    Topic,
    Unit,
    LearningObjective,
)

logger = logging.getLogger(__name__)


# Country code to language mapping for curriculum-based language detection
COUNTRY_TO_LANGUAGE: dict[str, str] = {
    "GB": "en",  # United Kingdom -> English
    "US": "en",  # United States -> English
    "AU": "en",  # Australia -> English
    "TR": "tr",  # Turkey -> Turkish
    "DE": "de",  # Germany -> German
    "FR": "fr",  # France -> French
    "ES": "es",  # Spain -> Spanish
    "RW": "en",  # Rwanda -> English (primary instruction language)
    "MW": "en",  # Malawi -> English (primary instruction language)
}

# Default language when country code is not mapped
DEFAULT_LANGUAGE: str = "en"

# Default framework when no framework can be resolved
DEFAULT_FRAMEWORK: str = "UK-NC-2014"


@dataclass
class TopicContext:
    """Educational context derived from topic's curriculum hierarchy.

    This context provides all necessary information about the topic's
    position in the curriculum hierarchy, including grade level, subject,
    and framework information needed for age-appropriate content generation.

    Attributes:
        topic_full_code: Full hierarchical topic code.
        topic_code: Topic code within unit.
        topic_name: Topic display name.
        topic_description: Optional topic description.
        unit_name: Parent unit name.
        unit_code: Parent unit code.
        subject_name: Subject display name (e.g., "Mathematics").
        subject_code: Subject code (e.g., "MAT").
        grade_name: Grade level display name (e.g., "Year 4").
        grade_code: Grade level code (e.g., "Y4").
        stage_name: Stage display name (e.g., "Key Stage 2").
        stage_code: Stage code (e.g., "KS2").
        typical_age: Typical student age for this grade.
        framework_name: Framework display name.
        framework_code: Framework code (e.g., "UK-NC-2014").
        country_code: ISO country code (e.g., "GB").
        language: Derived language code (e.g., "en" for GB).
    """

    topic_full_code: str
    topic_code: str
    topic_name: str
    topic_description: str | None
    unit_name: str
    unit_code: str
    subject_name: str
    subject_code: str
    grade_name: str
    grade_code: str
    stage_name: str
    stage_code: str
    typical_age: int | None
    framework_name: str
    framework_code: str
    country_code: str | None
    language: str

    @property
    def age_description(self) -> str | None:
        """Get age description string."""
        if self.typical_age is not None:
            return f"Age {self.typical_age}"
        return None


class CurriculumLookup:
    """Service for curriculum entity lookups using composite codes.

    Provides code-based resolution for curriculum entities using the
    composite primary key structure.

    Attributes:
        _db: Async database session.

    Example:
        >>> lookup = CurriculumLookup(db)
        >>> topic = await lookup.get_topic("UK-NC-2014", "MAT", "Y4", "NPV", "001")
        >>> framework = await lookup.get_framework("UK-NC-2014")
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the curriculum lookup service.

        Args:
            db: Async database session.
        """
        self._db = db

    # =========================================================================
    # Framework Lookups
    # =========================================================================

    async def get_framework(self, framework_code: str) -> CurriculumFramework | None:
        """Get framework by code.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").

        Returns:
            CurriculumFramework or None if not found.
        """
        stmt = select(CurriculumFramework).where(
            CurriculumFramework.code == framework_code
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_frameworks(self) -> list[CurriculumFramework]:
        """Get all active and published frameworks.

        Returns:
            List of active CurriculumFramework entities.
        """
        stmt = select(CurriculumFramework).where(
            CurriculumFramework.is_active == True,  # noqa: E712
            CurriculumFramework.is_published == True,  # noqa: E712
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_framework_by_country_code(
        self, country_code: str
    ) -> CurriculumFramework | None:
        """Get active framework for a country code.

        Looks up the active and published curriculum framework for a given
        ISO 2-letter country code. If multiple frameworks exist for a country,
        returns the first active one.

        Args:
            country_code: ISO 2-letter country code (e.g., "GB", "US", "RW", "MW").

        Returns:
            CurriculumFramework or None if not found.
        """
        stmt = (
            select(CurriculumFramework)
            .where(
                CurriculumFramework.country_code == country_code,
                CurriculumFramework.is_active == True,  # noqa: E712
                CurriculumFramework.is_published == True,  # noqa: E712
            )
            .limit(1)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def resolve_framework_code(self, country_code: str | None) -> str:
        """Resolve framework code from country code with fallback.

        Attempts to find an active framework for the given country code.
        Falls back to DEFAULT_FRAMEWORK if no framework is found.

        Args:
            country_code: ISO 2-letter country code, or None.

        Returns:
            Framework code string (never None).
        """
        if not country_code:
            logger.debug("No country code provided, using default framework")
            return DEFAULT_FRAMEWORK

        framework = await self.get_framework_by_country_code(country_code)
        if framework:
            logger.debug(
                "Resolved framework %s for country %s",
                framework.code, country_code
            )
            return framework.code

        logger.warning(
            "No framework found for country %s, using default",
            country_code
        )
        return DEFAULT_FRAMEWORK

    # =========================================================================
    # Stage Lookups
    # =========================================================================

    async def get_stage(
        self,
        framework_code: str,
        stage_code: str,
    ) -> CurriculumStage | None:
        """Get stage by composite key.

        Args:
            framework_code: Framework code.
            stage_code: Stage code within framework.

        Returns:
            CurriculumStage or None if not found.
        """
        stmt = select(CurriculumStage).where(
            CurriculumStage.framework_code == framework_code,
            CurriculumStage.code == stage_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_stages_for_framework(
        self, framework_code: str
    ) -> list[CurriculumStage]:
        """Get all stages for a framework.

        Args:
            framework_code: Framework code.

        Returns:
            List of CurriculumStage entities ordered by sequence.
        """
        stmt = (
            select(CurriculumStage)
            .where(CurriculumStage.framework_code == framework_code)
            .order_by(CurriculumStage.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Grade Lookups
    # =========================================================================

    async def get_grade(
        self,
        framework_code: str,
        stage_code: str,
        grade_code: str,
    ) -> GradeLevel | None:
        """Get grade level by composite key.

        Args:
            framework_code: Framework code.
            stage_code: Stage code.
            grade_code: Grade code within stage.

        Returns:
            GradeLevel or None if not found.
        """
        stmt = select(GradeLevel).where(
            GradeLevel.framework_code == framework_code,
            GradeLevel.stage_code == stage_code,
            GradeLevel.code == grade_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_grade_level(
        self,
        framework_code: str,
        grade_code: str,
    ) -> GradeLevel | None:
        """Get grade level by framework and grade code only.

        Simplified lookup that doesn't require stage_code.
        Useful when only framework and grade code are known.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").
            grade_code: Grade code (e.g., "Y5", "P5", "STD5").

        Returns:
            GradeLevel or None if not found.
        """
        stmt = select(GradeLevel).where(
            GradeLevel.framework_code == framework_code,
            GradeLevel.code == grade_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_grades_for_stage(
        self,
        framework_code: str,
        stage_code: str,
    ) -> list[GradeLevel]:
        """Get all grades for a stage.

        Args:
            framework_code: Framework code.
            stage_code: Stage code.

        Returns:
            List of GradeLevel entities ordered by sequence.
        """
        stmt = (
            select(GradeLevel)
            .where(
                GradeLevel.framework_code == framework_code,
                GradeLevel.stage_code == stage_code,
            )
            .order_by(GradeLevel.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Subject Lookups
    # =========================================================================

    async def get_subject(
        self,
        framework_code: str,
        subject_code: str,
    ) -> Subject | None:
        """Get subject by composite key.

        Args:
            framework_code: Framework code.
            subject_code: Subject code within framework.

        Returns:
            Subject or None if not found.
        """
        stmt = select(Subject).where(
            Subject.framework_code == framework_code,
            Subject.code == subject_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_subjects_for_framework(self, framework_code: str) -> list[Subject]:
        """Get all subjects for a framework.

        Args:
            framework_code: Framework code.

        Returns:
            List of Subject entities ordered by sequence.
        """
        stmt = (
            select(Subject)
            .where(Subject.framework_code == framework_code)
            .order_by(Subject.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Unit Lookups
    # =========================================================================

    async def get_unit(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
    ) -> Unit | None:
        """Get unit by composite key.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code within subject+grade.

        Returns:
            Unit or None if not found.
        """
        stmt = select(Unit).where(
            Unit.framework_code == framework_code,
            Unit.subject_code == subject_code,
            Unit.grade_code == grade_code,
            Unit.code == unit_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_units_for_subject_grade(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
    ) -> list[Unit]:
        """Get all units for a subject and grade combination.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.

        Returns:
            List of Unit entities ordered by sequence.
        """
        stmt = (
            select(Unit)
            .where(
                Unit.framework_code == framework_code,
                Unit.subject_code == subject_code,
                Unit.grade_code == grade_code,
            )
            .order_by(Unit.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Topic Lookups
    # =========================================================================

    async def get_topic(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
    ) -> Topic | None:
        """Get topic by composite key.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.
            topic_code: Topic code within unit.

        Returns:
            Topic or None if not found.
        """
        stmt = select(Topic).where(
            Topic.framework_code == framework_code,
            Topic.subject_code == subject_code,
            Topic.grade_code == grade_code,
            Topic.unit_code == unit_code,
            Topic.code == topic_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_topic_by_full_code(self, full_code: str) -> Topic | None:
        """Get topic by full hierarchical code.

        Args:
            full_code: Full code in format "framework.subject.grade.unit.topic"
                      (e.g., "UK-NC-2014.MAT.Y4.NPV.001")

        Returns:
            Topic or None if not found.
        """
        parts = full_code.split(".")
        if len(parts) != 5:
            logger.warning("Invalid topic full code format: %s", full_code)
            return None

        return await self.get_topic(
            framework_code=parts[0],
            subject_code=parts[1],
            grade_code=parts[2],
            unit_code=parts[3],
            topic_code=parts[4],
        )

    async def get_topics_for_unit(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
    ) -> list[Topic]:
        """Get all topics for a unit.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.

        Returns:
            List of Topic entities ordered by sequence.
        """
        stmt = (
            select(Topic)
            .where(
                Topic.framework_code == framework_code,
                Topic.subject_code == subject_code,
                Topic.grade_code == grade_code,
                Topic.unit_code == unit_code,
            )
            .order_by(Topic.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Objective Lookups
    # =========================================================================

    async def get_objective(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
        objective_code: str,
    ) -> LearningObjective | None:
        """Get learning objective by composite key.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.
            topic_code: Topic code.
            objective_code: Objective code within topic.

        Returns:
            LearningObjective or None if not found.
        """
        stmt = select(LearningObjective).where(
            LearningObjective.framework_code == framework_code,
            LearningObjective.subject_code == subject_code,
            LearningObjective.grade_code == grade_code,
            LearningObjective.unit_code == unit_code,
            LearningObjective.topic_code == topic_code,
            LearningObjective.code == objective_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_objective_by_full_code(self, full_code: str) -> LearningObjective | None:
        """Get objective by full hierarchical code.

        Args:
            full_code: Full code in format "framework.subject.grade.unit.topic.objective"
                      (e.g., "UK-NC-2014.MAT.Y4.NPV.001.LO1")

        Returns:
            LearningObjective or None if not found.
        """
        parts = full_code.split(".")
        if len(parts) != 6:
            logger.warning("Invalid objective full code format: %s", full_code)
            return None

        return await self.get_objective(
            framework_code=parts[0],
            subject_code=parts[1],
            grade_code=parts[2],
            unit_code=parts[3],
            topic_code=parts[4],
            objective_code=parts[5],
        )

    async def get_objectives_for_topic(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
    ) -> list[LearningObjective]:
        """Get all objectives for a topic.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.
            topic_code: Topic code.

        Returns:
            List of LearningObjective entities ordered by sequence.
        """
        stmt = (
            select(LearningObjective)
            .where(
                LearningObjective.framework_code == framework_code,
                LearningObjective.subject_code == subject_code,
                LearningObjective.grade_code == grade_code,
                LearningObjective.unit_code == unit_code,
                LearningObjective.topic_code == topic_code,
            )
            .order_by(LearningObjective.sequence)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Context Builders
    # =========================================================================

    async def get_topic_context(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
    ) -> TopicContext | None:
        """Get educational context for a topic from curriculum hierarchy.

        Retrieves topic details along with its parent unit, subject, grade,
        stage, and framework information. This context is essential for
        generating age-appropriate and curriculum-aligned content.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.
            topic_code: Topic code.

        Returns:
            TopicContext with full curriculum hierarchy, or None if not found.
        """
        # Get topic with relationships
        stmt = (
            select(Topic)
            .options(
                selectinload(Topic.unit).selectinload(Unit.subject).selectinload(Subject.framework),
            )
            .where(
                Topic.framework_code == framework_code,
                Topic.subject_code == subject_code,
                Topic.grade_code == grade_code,
                Topic.unit_code == unit_code,
                Topic.code == topic_code,
            )
        )
        result = await self._db.execute(stmt)
        topic = result.scalar_one_or_none()

        if not topic:
            logger.warning(
                "Topic not found: %s.%s.%s.%s.%s",
                framework_code, subject_code, grade_code, unit_code, topic_code
            )
            return None

        unit = topic.unit
        if not unit:
            logger.warning("Topic has no unit: %s", topic.full_code)
            return None

        subject = unit.subject
        if not subject:
            logger.warning("Unit has no subject: %s", unit.code)
            return None

        framework = subject.framework
        if not framework:
            logger.warning("Subject has no framework: %s", subject.code)
            return None

        # Get grade and stage info
        grade = await self.get_grade(framework_code, unit.grade_code, grade_code)

        # Find stage for this grade
        stage_code = grade.stage_code if grade else None
        stage_name = ""
        typical_age = grade.typical_age if grade else None

        if stage_code:
            stage = await self.get_stage(framework_code, stage_code)
            stage_name = stage.name if stage else ""

        # Derive language from country code
        country_code = framework.country_code
        language = (
            COUNTRY_TO_LANGUAGE.get(country_code, DEFAULT_LANGUAGE)
            if country_code
            else DEFAULT_LANGUAGE
        )

        return TopicContext(
            topic_full_code=topic.full_code,
            topic_code=topic.code,
            topic_name=topic.name,
            topic_description=topic.description,
            unit_name=unit.name,
            unit_code=unit.code,
            subject_name=subject.name,
            subject_code=subject.code,
            grade_name=grade.name if grade else grade_code,
            grade_code=grade_code,
            stage_name=stage_name,
            stage_code=stage_code or "",
            typical_age=typical_age,
            framework_name=framework.name,
            framework_code=framework.code,
            country_code=country_code,
            language=language,
        )

    async def get_topic_context_by_full_code(self, full_code: str) -> TopicContext | None:
        """Get educational context for a topic by its full code.

        Args:
            full_code: Full code in format "framework.subject.grade.unit.topic"
                      (e.g., "UK-NC-2014.MAT.Y4.NPV.001")

        Returns:
            TopicContext or None if not found.
        """
        parts = full_code.split(".")
        if len(parts) != 5:
            logger.warning("Invalid topic full code format: %s", full_code)
            return None

        return await self.get_topic_context(
            framework_code=parts[0],
            subject_code=parts[1],
            grade_code=parts[2],
            unit_code=parts[3],
            topic_code=parts[4],
        )
