# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class service for managing class/section operations.

This module provides the ClassService class for:
- Class CRUD operations
- Class activation/deactivation
- Student and teacher count tracking
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.school import (
    Class,
    School,
    AcademicYear,
    ClassStudent,
    ClassTeacher,
)
from src.infrastructure.database.models.tenant.curriculum import GradeLevel
from src.models.class_ import (
    ClassCreateRequest,
    ClassUpdateRequest,
    ClassResponse,
    ClassSummary,
)

logger = logging.getLogger(__name__)


class ClassServiceError(Exception):
    """Base exception for class service errors."""

    pass


class ClassNotFoundError(ClassServiceError):
    """Raised when class is not found."""

    pass


class ClassCodeExistsError(ClassServiceError):
    """Raised when class code already exists for school/year."""

    pass


class SchoolNotFoundError(ClassServiceError):
    """Raised when school is not found."""

    pass


class AcademicYearNotFoundError(ClassServiceError):
    """Raised when academic year is not found."""

    pass


class ClassService:
    """Service for managing classes.

    This service handles all class operations including
    creating, updating, activating/deactivating classes.

    Attributes:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize class service.

        Args:
            db: Async database session for tenant database.
        """
        self.db = db

    async def create_class(
        self,
        request: ClassCreateRequest,
        created_by: str,
    ) -> ClassResponse:
        """Create a new class.

        Args:
            request: Class creation data.
            created_by: ID of user creating the class.

        Returns:
            Created class response.

        Raises:
            SchoolNotFoundError: If school not found.
            AcademicYearNotFoundError: If academic year not found.
            ClassCodeExistsError: If class code exists for school/year.
        """
        # Verify school exists
        school = await self._get_school(request.school_id)

        # Verify academic year exists
        academic_year = await self._get_academic_year(request.academic_year_id)

        # Check for duplicate code
        existing = await self._get_by_code(
            str(request.school_id),
            str(request.academic_year_id),
            request.code,
        )
        if existing:
            raise ClassCodeExistsError(
                f"Class with code '{request.code}' already exists for this school/year"
            )

        # Verify grade level if provided (using composite key)
        grade_level = None
        if request.has_grade_level():
            grade_level = await self._get_grade_level(
                request.framework_code,
                request.stage_code,
                request.grade_code,
            )

        class_ = Class(
            school_id=str(request.school_id),
            academic_year_id=str(request.academic_year_id),
            code=request.code,
            name=request.name,
            framework_code=request.framework_code,
            stage_code=request.stage_code,
            grade_code=request.grade_code,
            max_students=request.max_students,
            is_active=True,
        )

        self.db.add(class_)
        await self.db.commit()
        await self.db.refresh(class_)

        logger.info("Created class: %s (%s) by %s", class_.name, class_.id, created_by)

        return await self._to_response(class_, school, academic_year, grade_level)

    async def list_classes(
        self,
        school_id: UUID | None = None,
        academic_year_id: UUID | None = None,
        is_active: bool | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ClassSummary], int]:
        """List classes with filtering.

        Args:
            school_id: Filter by school.
            academic_year_id: Filter by academic year.
            is_active: Filter by active status.
            search: Search in name or code.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Tuple of (list of classes, total count).
        """
        query = select(Class).options(
            selectinload(Class.school),
            selectinload(Class.grade_level),
        )

        # Apply filters
        conditions = []

        if school_id:
            conditions.append(Class.school_id == str(school_id))

        if academic_year_id:
            conditions.append(Class.academic_year_id == str(academic_year_id))

        if is_active is not None:
            conditions.append(Class.is_active == is_active)

        if search:
            search_pattern = f"%{search}%"
            conditions.append(
                (Class.name.ilike(search_pattern)) |
                (Class.code.ilike(search_pattern))
            )

        if conditions:
            query = query.where(and_(*conditions))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(Class.name).limit(limit).offset(offset)

        result = await self.db.execute(query)
        classes = result.scalars().all()

        items = []
        for class_ in classes:
            student_count = await self._get_student_count(class_.id)
            items.append(self._to_summary(class_, student_count))

        return items, total

    async def get_class(self, class_id: UUID) -> ClassResponse:
        """Get class by ID.

        Args:
            class_id: Class identifier.

        Returns:
            Class details.

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_by_id(class_id)
        school = await self._get_school(UUID(class_.school_id))
        academic_year = await self._get_academic_year(UUID(class_.academic_year_id))
        grade_level = None
        if class_.grade_full_code:
            grade_level = await self._get_grade_level(
                class_.framework_code,
                class_.stage_code,
                class_.grade_code,
            )

        return await self._to_response(class_, school, academic_year, grade_level)

    async def update_class(
        self,
        class_id: UUID,
        request: ClassUpdateRequest,
    ) -> ClassResponse:
        """Update a class.

        Args:
            class_id: Class identifier.
            request: Update data.

        Returns:
            Updated class.

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_by_id(class_id)

        # Apply updates
        if request.name is not None:
            class_.name = request.name
        if request.has_grade_level():
            # Verify grade level exists using composite key
            await self._get_grade_level(
                request.framework_code,
                request.stage_code,
                request.grade_code,
            )
            class_.framework_code = request.framework_code
            class_.stage_code = request.stage_code
            class_.grade_code = request.grade_code
        if request.max_students is not None:
            class_.max_students = request.max_students

        await self.db.commit()
        await self.db.refresh(class_)

        school = await self._get_school(UUID(class_.school_id))
        academic_year = await self._get_academic_year(UUID(class_.academic_year_id))
        grade_level = None
        if class_.grade_full_code:
            grade_level = await self._get_grade_level(
                class_.framework_code,
                class_.stage_code,
                class_.grade_code,
            )

        logger.info("Updated class: %s", class_id)

        return await self._to_response(class_, school, academic_year, grade_level)

    async def deactivate_class(self, class_id: UUID) -> None:
        """Deactivate a class.

        Args:
            class_id: Class identifier.

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_by_id(class_id)
        class_.is_active = False

        await self.db.commit()

        logger.info("Deactivated class: %s", class_id)

    async def activate_class(self, class_id: UUID) -> ClassResponse:
        """Activate a class.

        Args:
            class_id: Class identifier.

        Returns:
            Activated class.

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_by_id(class_id)
        class_.is_active = True

        await self.db.commit()
        await self.db.refresh(class_)

        school = await self._get_school(UUID(class_.school_id))
        academic_year = await self._get_academic_year(UUID(class_.academic_year_id))
        grade_level = None
        if class_.grade_full_code:
            grade_level = await self._get_grade_level(
                class_.framework_code,
                class_.stage_code,
                class_.grade_code,
            )

        logger.info("Activated class: %s", class_id)

        return await self._to_response(class_, school, academic_year, grade_level)

    async def check_school_access(self, user_id: str, class_id: str) -> bool:
        """Check if user has access to a class's school.

        Args:
            user_id: User identifier.
            class_id: Class identifier.

        Returns:
            True if user has access.
        """
        class_ = await self._get_by_id(UUID(class_id))

        from src.infrastructure.database.models.tenant.user import UserRole

        query = select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.school_id == class_.school_id,
            UserRole.role_code == "school_admin",
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none() is not None

    async def _get_by_id(self, class_id: UUID) -> Class:
        """Get class by ID.

        Args:
            class_id: Class identifier.

        Returns:
            Class model instance.

        Raises:
            ClassNotFoundError: If not found.
        """
        query = select(Class).where(Class.id == str(class_id))
        result = await self.db.execute(query)
        class_ = result.scalar_one_or_none()

        if not class_:
            raise ClassNotFoundError(f"Class {class_id} not found")

        return class_

    async def _get_by_code(
        self,
        school_id: str,
        academic_year_id: str,
        code: str,
    ) -> Class | None:
        """Get class by code within school/year.

        Args:
            school_id: School identifier.
            academic_year_id: Academic year identifier.
            code: Class code.

        Returns:
            Class if found, None otherwise.
        """
        query = select(Class).where(
            Class.school_id == school_id,
            Class.academic_year_id == academic_year_id,
            Class.code == code,
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_school(self, school_id: UUID) -> School:
        """Get school by ID.

        Args:
            school_id: School identifier.

        Returns:
            School model instance.

        Raises:
            SchoolNotFoundError: If not found.
        """
        query = select(School).where(School.id == str(school_id))
        result = await self.db.execute(query)
        school = result.scalar_one_or_none()

        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        return school

    async def _get_academic_year(self, year_id: UUID) -> AcademicYear:
        """Get academic year by ID.

        Args:
            year_id: Academic year identifier.

        Returns:
            AcademicYear model instance.

        Raises:
            AcademicYearNotFoundError: If not found.
        """
        query = select(AcademicYear).where(AcademicYear.id == str(year_id))
        result = await self.db.execute(query)
        year = result.scalar_one_or_none()

        if not year:
            raise AcademicYearNotFoundError(f"Academic year {year_id} not found")

        return year

    async def _get_grade_level(
        self,
        framework_code: str,
        stage_code: str,
        grade_code: str,
    ) -> GradeLevel:
        """Get grade level by composite key.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").
            stage_code: Stage code (e.g., "KS2").
            grade_code: Grade code (e.g., "Y4").

        Returns:
            GradeLevel model instance.

        Raises:
            ClassServiceError: If not found.
        """
        query = select(GradeLevel).where(
            GradeLevel.framework_code == framework_code,
            GradeLevel.stage_code == stage_code,
            GradeLevel.code == grade_code,
        )
        result = await self.db.execute(query)
        grade_level = result.scalar_one_or_none()

        if not grade_level:
            raise ClassServiceError(
                f"Grade level {framework_code}.{stage_code}.{grade_code} not found"
            )

        return grade_level

    async def _get_student_count(self, class_id: str) -> int:
        """Get count of active students in a class.

        Args:
            class_id: Class identifier.

        Returns:
            Number of active students.
        """
        query = select(func.count()).select_from(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.status == "active",
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def _get_teacher_count(self, class_id: str) -> int:
        """Get count of active teachers in a class.

        Args:
            class_id: Class identifier.

        Returns:
            Number of active teachers.
        """
        query = select(func.count()).select_from(ClassTeacher).where(
            ClassTeacher.class_id == class_id,
            ClassTeacher.ended_at.is_(None),
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def _to_response(
        self,
        class_: Class,
        school: School,
        academic_year: AcademicYear,
        grade_level: GradeLevel | None,
    ) -> ClassResponse:
        """Convert class model to response DTO.

        Args:
            class_: Class model instance.
            school: School model instance.
            academic_year: AcademicYear model instance.
            grade_level: Optional GradeLevel model instance.

        Returns:
            ClassResponse DTO.
        """
        student_count = await self._get_student_count(class_.id)
        teacher_count = await self._get_teacher_count(class_.id)

        return ClassResponse(
            id=UUID(class_.id),
            school_id=UUID(class_.school_id),
            school_name=school.name,
            academic_year_id=UUID(class_.academic_year_id),
            academic_year_name=academic_year.name,
            code=class_.code,
            name=class_.name,
            grade_full_code=class_.grade_full_code,
            grade_level_name=grade_level.name if grade_level else None,
            max_students=class_.max_students,
            is_active=class_.is_active,
            student_count=student_count,
            teacher_count=teacher_count,
            created_at=class_.created_at,
            updated_at=class_.updated_at,
        )

    def _to_summary(
        self,
        class_: Class,
        student_count: int,
    ) -> ClassSummary:
        """Convert class model to summary DTO.

        Args:
            class_: Class model instance.
            student_count: Number of students.

        Returns:
            ClassSummary DTO.
        """
        school_name = class_.school.name if class_.school else ""
        grade_level_name = class_.grade_level.name if class_.grade_level else None

        return ClassSummary(
            id=UUID(class_.id),
            school_id=UUID(class_.school_id),
            school_name=school_name,
            code=class_.code,
            name=class_.name,
            grade_level_name=grade_level_name,
            is_active=class_.is_active,
            student_count=student_count,
        )
