# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Provisioning service for LMS integration.

This service provides atomic provisioning of school, class, and student data
for LMS integration. All operations use upsert semantics - entities are
created if they don't exist, or their existence is verified.

NOTE: Curriculum data (frameworks, stages, grades, subjects, units, topics)
is NOT created by this service. Curriculum data is synced from the Central
Curriculum service. This service only references existing curriculum data
when creating classes.

The provisioning flow:
1. Verify GradeLevel exists (from Central Curriculum sync)
2. School (by code)
3. AcademicYear (by code)
4. Class (by school_id + academic_year_id + code)
5. Student (by email) + Class enrollment

Example:
    >>> service = ProvisioningService(db)
    >>> result = await service.provision(request)
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.curriculum import GradeLevel
from src.infrastructure.database.models.tenant.school import (
    School,
    AcademicYear,
    Class,
    ClassStudent,
)
from src.infrastructure.database.models.tenant.user import User
from src.models.provisioning import (
    ProvisioningRequest,
    ProvisioningResponse,
    SchoolProvision,
    AcademicYearProvision,
    ClassProvision,
    StudentProvision,
)

logger = logging.getLogger(__name__)


class ProvisioningError(Exception):
    """Base exception for provisioning errors."""

    pass


class GradeLevelNotFoundError(ProvisioningError):
    """Raised when referenced grade level does not exist."""

    pass


class SchoolProvisioningError(ProvisioningError):
    """Raised when school provisioning fails."""

    pass


class StudentProvisioningError(ProvisioningError):
    """Raised when student provisioning fails."""

    pass


class ProvisioningService:
    """Service for LMS data provisioning.

    Provides atomic provisioning of all entities required for a student
    to use EduSynapse. Uses upsert semantics for idempotent operations.

    NOTE: This service does not create curriculum data. It only references
    existing grade levels that have been synced from Central Curriculum.

    Attributes:
        _db: Async database session.

    Example:
        >>> service = ProvisioningService(db)
        >>> result = await service.provision(request)
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the provisioning service.

        Args:
            db: Async database session.
        """
        self._db = db

    async def provision(
        self,
        request: ProvisioningRequest,
    ) -> ProvisioningResponse:
        """Provision all entities from LMS request.

        This method is atomic - if any step fails, all changes are rolled back.

        Args:
            request: Complete provisioning request.

        Returns:
            ProvisioningResponse with results.

        Raises:
            ProvisioningError: If provisioning fails.
            GradeLevelNotFoundError: If referenced grade level doesn't exist.
        """
        try:
            # 1. Verify grade level exists (from Central Curriculum sync)
            await self._verify_grade_level(
                request.class_.framework_code,
                request.class_.stage_code,
                request.class_.grade_code,
            )
            logger.info(
                "Grade level verified: %s.%s.%s",
                request.class_.framework_code,
                request.class_.stage_code,
                request.class_.grade_code,
            )

            # 2. School upsert
            school = await self._upsert_school(request.school)
            logger.info("School provisioned: %s", school.code)

            # 3. AcademicYear upsert
            academic_year = await self._upsert_academic_year(request.academic_year)
            logger.info("AcademicYear provisioned: %s", academic_year.code)

            # 4. Class upsert
            class_ = await self._upsert_class(
                request.class_,
                school.id,
                academic_year.id,
            )
            logger.info("Class provisioned: %s", class_.code)

            # 5. Student upsert + enrollment
            student = await self._upsert_student(request.student, class_.id)
            logger.info("Student provisioned and enrolled: %s", student.email)

            # Commit all changes
            await self._db.commit()

            return ProvisioningResponse(
                success=True,
                student_id=student.id,
                school_code=school.code,
                class_code=class_.code,
                framework_code=request.class_.framework_code,
                stage_code=request.class_.stage_code,
                grade_code=request.class_.grade_code,
                message="Provisioning completed successfully",
            )

        except GradeLevelNotFoundError:
            await self._db.rollback()
            raise
        except Exception as e:
            await self._db.rollback()
            logger.error("Provisioning failed: %s", str(e))
            raise ProvisioningError(f"Provisioning failed: {str(e)}") from e

    # =========================================================================
    # Grade Level Verification
    # =========================================================================

    async def _verify_grade_level(
        self,
        framework_code: str,
        stage_code: str,
        grade_code: str,
    ) -> GradeLevel:
        """Verify that the referenced grade level exists.

        Grade levels are created by the Central Curriculum sync service.
        This method verifies that the referenced grade level exists before
        creating a class that references it.

        Args:
            framework_code: Curriculum framework code (e.g., "UK-NC-2014").
            stage_code: Curriculum stage code (e.g., "KS2").
            grade_code: Grade level code (e.g., "Y4").

        Returns:
            The existing GradeLevel.

        Raises:
            GradeLevelNotFoundError: If grade level doesn't exist.
        """
        stmt = select(GradeLevel).where(
            GradeLevel.framework_code == framework_code,
            GradeLevel.stage_code == stage_code,
            GradeLevel.code == grade_code,
        )
        result = await self._db.execute(stmt)
        grade_level = result.scalar_one_or_none()

        if grade_level is None:
            raise GradeLevelNotFoundError(
                f"Grade level not found: {framework_code}.{stage_code}.{grade_code}. "
                "Ensure curriculum data has been synced from Central Curriculum."
            )

        return grade_level

    # =========================================================================
    # School Upsert Methods
    # =========================================================================

    async def _upsert_school(self, data: SchoolProvision) -> School:
        """Upsert school by code.

        Args:
            data: School provisioning data.

        Returns:
            Created or existing School.
        """
        stmt = select(School).where(
            School.code == data.code,
            School.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing school
            existing.name = data.name
            if data.school_type is not None:
                existing.school_type = data.school_type
            existing.country_code = data.country_code
            existing.timezone = data.timezone
            return existing

        # Create new school
        school = School(
            code=data.code,
            name=data.name,
            school_type=data.school_type,
            country_code=data.country_code,
            timezone=data.timezone,
            is_active=True,
        )
        self._db.add(school)
        await self._db.flush()
        return school

    async def _upsert_academic_year(self, data: AcademicYearProvision) -> AcademicYear:
        """Upsert academic year by code.

        Args:
            data: Academic year provisioning data.

        Returns:
            Created or existing AcademicYear.
        """
        stmt = select(AcademicYear).where(AcademicYear.code == data.code)
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing academic year
            existing.name = data.name
            existing.start_date = data.start_date
            existing.end_date = data.end_date
            # Handle is_current: if setting this year as current, unset others
            if data.is_current and not existing.is_current:
                await self._unset_current_academic_year()
                existing.is_current = True
            return existing

        # If setting as current, unset others first
        if data.is_current:
            await self._unset_current_academic_year()

        # Create new academic year
        academic_year = AcademicYear(
            code=data.code,
            name=data.name,
            start_date=data.start_date,
            end_date=data.end_date,
            is_current=data.is_current,
        )
        self._db.add(academic_year)
        await self._db.flush()
        return academic_year

    async def _unset_current_academic_year(self) -> None:
        """Unset any current academic year."""
        stmt = (
            update(AcademicYear)
            .where(AcademicYear.is_current == True)  # noqa: E712
            .values(is_current=False)
        )
        await self._db.execute(stmt)

    async def _upsert_class(
        self,
        data: ClassProvision,
        school_id: str,
        academic_year_id: str,
    ) -> Class:
        """Upsert class by school_id + academic_year_id + code.

        Args:
            data: Class provisioning data.
            school_id: Parent school ID.
            academic_year_id: Academic year ID.

        Returns:
            Created or existing Class.
        """
        stmt = select(Class).where(
            Class.school_id == school_id,
            Class.academic_year_id == academic_year_id,
            Class.code == data.code,
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing class
            existing.name = data.name
            existing.framework_code = data.framework_code
            existing.stage_code = data.stage_code
            existing.grade_code = data.grade_code
            if data.max_students is not None:
                existing.max_students = data.max_students
            return existing

        # Create new class
        class_ = Class(
            school_id=school_id,
            academic_year_id=academic_year_id,
            code=data.code,
            name=data.name,
            framework_code=data.framework_code,
            stage_code=data.stage_code,
            grade_code=data.grade_code,
            max_students=data.max_students,
            is_active=True,
        )
        self._db.add(class_)
        await self._db.flush()
        return class_

    # =========================================================================
    # Student Upsert Methods
    # =========================================================================

    async def _upsert_student(
        self,
        data: StudentProvision,
        class_id: str,
    ) -> User:
        """Upsert student by email and enroll in class.

        Args:
            data: Student provisioning data.
            class_id: Class to enroll the student in.

        Returns:
            Created or existing User.
        """
        # Find or create student
        stmt = select(User).where(
            User.email == data.email,
            User.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing student
            existing.first_name = data.first_name
            existing.last_name = data.last_name
            if data.external_id:
                existing.sso_external_id = data.external_id
            existing.preferred_language = data.preferred_language
            student = existing
        else:
            # Create new student
            student = User(
                email=data.email,
                first_name=data.first_name,
                last_name=data.last_name,
                user_type="student",
                status="active",
                preferred_language=data.preferred_language,
                sso_provider="lms" if data.external_id else None,
                sso_external_id=data.external_id,
            )
            self._db.add(student)
            await self._db.flush()

        # Enroll student in class (upsert enrollment)
        await self._upsert_enrollment(student.id, class_id)

        return student

    async def _upsert_enrollment(
        self,
        student_id: str,
        class_id: str,
    ) -> ClassStudent:
        """Upsert class enrollment.

        Args:
            student_id: Student ID.
            class_id: Class ID.

        Returns:
            Created or existing ClassStudent enrollment.
        """
        stmt = select(ClassStudent).where(
            ClassStudent.student_id == student_id,
            ClassStudent.class_id == class_id,
        )
        result = await self._db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Reactivate if withdrawn
            if existing.status == "withdrawn":
                existing.status = "active"
                existing.enrolled_at = datetime.now(timezone.utc)
                existing.withdrawn_at = None
            return existing

        # Create new enrollment
        enrollment = ClassStudent(
            student_id=student_id,
            class_id=class_id,
            status="active",
        )
        self._db.add(enrollment)
        await self._db.flush()
        return enrollment
