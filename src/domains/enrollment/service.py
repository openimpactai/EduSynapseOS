# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Enrollment service for managing student class enrollments.

This module provides the EnrollmentService class for:
- Student enrollment in classes
- Enrollment withdrawal
- Bulk enrollment operations
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.school import Class, ClassStudent
from src.infrastructure.database.models.tenant.user import User
from src.models.common import StatusEnum
from src.models.enrollment import (
    EnrollStudentRequest,
    BulkEnrollRequest,
    EnrollmentResponse,
    EnrollmentSummary,
    BulkEnrollResponse,
)
from src.models.user import UserSummary

logger = logging.getLogger(__name__)


class EnrollmentServiceError(Exception):
    """Base exception for enrollment service errors."""

    pass


class ClassNotFoundError(EnrollmentServiceError):
    """Raised when class is not found."""

    pass


class StudentNotFoundError(EnrollmentServiceError):
    """Raised when student is not found."""

    pass


class AlreadyEnrolledError(EnrollmentServiceError):
    """Raised when student is already enrolled in class."""

    pass


class NotEnrolledError(EnrollmentServiceError):
    """Raised when student is not enrolled in class."""

    pass


class InvalidStudentTypeError(EnrollmentServiceError):
    """Raised when user is not a student type."""

    pass


class EnrollmentService:
    """Service for managing student enrollments.

    This service handles all enrollment operations including
    enrolling, withdrawing, and bulk enrollment operations.

    Attributes:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize enrollment service.

        Args:
            db: Async database session for tenant database.
        """
        self.db = db

    async def enroll_student(
        self,
        class_id: UUID,
        request: EnrollStudentRequest,
        enrolled_by: str,
    ) -> EnrollmentResponse:
        """Enroll a student in a class.

        Args:
            class_id: Class identifier.
            request: Enrollment request data.
            enrolled_by: ID of user performing enrollment.

        Returns:
            Enrollment response.

        Raises:
            ClassNotFoundError: If class not found.
            StudentNotFoundError: If student not found.
            InvalidStudentTypeError: If user is not a student.
            AlreadyEnrolledError: If student already enrolled.
        """
        # Verify class exists
        class_ = await self._get_class(class_id)

        # Verify student exists and is student type
        student = await self._get_student(request.student_id)

        # Check if already enrolled
        existing = await self._get_enrollment(str(class_id), str(request.student_id))
        if existing:
            if existing.status == "active":
                raise AlreadyEnrolledError(
                    f"Student is already enrolled in this class"
                )
            # Reactivate existing enrollment
            existing.status = "active"
            existing.enrolled_at = datetime.now(timezone.utc)
            existing.withdrawn_at = None
            if request.student_number:
                existing.student_number = request.student_number

            await self.db.commit()
            await self.db.refresh(existing)

            logger.info(
                "Reactivated enrollment: student=%s, class=%s, by=%s",
                request.student_id,
                class_id,
                enrolled_by,
            )

            return self._to_response(existing, student)

        # Create new enrollment
        enrollment = ClassStudent(
            class_id=str(class_id),
            student_id=str(request.student_id),
            student_number=request.student_number,
            status="active",
        )

        self.db.add(enrollment)
        await self.db.commit()
        await self.db.refresh(enrollment)

        logger.info(
            "Enrolled student: student=%s, class=%s, by=%s",
            request.student_id,
            class_id,
            enrolled_by,
        )

        return self._to_response(enrollment, student)

    async def bulk_enroll(
        self,
        class_id: UUID,
        request: BulkEnrollRequest,
        enrolled_by: str,
    ) -> BulkEnrollResponse:
        """Bulk enroll students in a class.

        Args:
            class_id: Class identifier.
            request: Bulk enrollment request.
            enrolled_by: ID of user performing enrollment.

        Returns:
            Bulk enrollment response with success/failure details.

        Raises:
            ClassNotFoundError: If class not found.
        """
        # Verify class exists
        await self._get_class(class_id)

        enrolled = []
        failed = []

        for student_id in request.student_ids:
            try:
                enroll_request = EnrollStudentRequest(student_id=student_id)
                await self.enroll_student(class_id, enroll_request, enrolled_by)
                enrolled.append(student_id)
            except StudentNotFoundError:
                failed.append({"student_id": str(student_id), "reason": "Student not found"})
            except InvalidStudentTypeError:
                failed.append({"student_id": str(student_id), "reason": "User is not a student"})
            except AlreadyEnrolledError:
                failed.append({"student_id": str(student_id), "reason": "Already enrolled"})
            except Exception as e:
                failed.append({"student_id": str(student_id), "reason": str(e)})

        logger.info(
            "Bulk enrollment: class=%s, enrolled=%d, failed=%d, by=%s",
            class_id,
            len(enrolled),
            len(failed),
            enrolled_by,
        )

        return BulkEnrollResponse(
            enrolled=enrolled,
            failed=failed,
            total_enrolled=len(enrolled),
            total_failed=len(failed),
        )

    async def list_enrollments(
        self,
        class_id: UUID,
        status: str | None = None,
    ) -> tuple[list[EnrollmentSummary], int, str]:
        """List students enrolled in a class.

        Args:
            class_id: Class identifier.
            status: Optional status filter (active, withdrawn).

        Returns:
            Tuple of (list of enrollments, total count, class name).

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_class(class_id)

        query = select(ClassStudent).options(
            selectinload(ClassStudent.student)
        ).where(ClassStudent.class_id == str(class_id))

        if status:
            query = query.where(ClassStudent.status == status)

        query = query.order_by(ClassStudent.enrolled_at.desc())

        result = await self.db.execute(query)
        enrollments = result.scalars().all()

        items = [self._to_summary(e) for e in enrollments]

        return items, len(items), class_.name

    async def get_enrollment(
        self,
        class_id: UUID,
        student_id: UUID,
    ) -> EnrollmentResponse:
        """Get specific enrollment details.

        Args:
            class_id: Class identifier.
            student_id: Student identifier.

        Returns:
            Enrollment details.

        Raises:
            NotEnrolledError: If student not enrolled.
        """
        enrollment = await self._get_enrollment(str(class_id), str(student_id))
        if not enrollment:
            raise NotEnrolledError("Student is not enrolled in this class")

        student = await self._get_student(student_id)
        return self._to_response(enrollment, student)

    async def withdraw_student(
        self,
        class_id: UUID,
        student_id: UUID,
        withdrawn_at: datetime | None = None,
        withdrawn_by: str | None = None,
    ) -> EnrollmentResponse:
        """Withdraw a student from a class.

        Args:
            class_id: Class identifier.
            student_id: Student identifier.
            withdrawn_at: Optional withdrawal date.
            withdrawn_by: ID of user performing withdrawal.

        Returns:
            Updated enrollment.

        Raises:
            NotEnrolledError: If student not enrolled.
        """
        enrollment = await self._get_enrollment(str(class_id), str(student_id))
        if not enrollment:
            raise NotEnrolledError("Student is not enrolled in this class")

        if enrollment.status == "withdrawn":
            raise NotEnrolledError("Student is already withdrawn")

        enrollment.status = "withdrawn"
        enrollment.withdrawn_at = withdrawn_at or datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(enrollment)

        student = await self._get_student(student_id)

        logger.info(
            "Withdrew student: student=%s, class=%s, by=%s",
            student_id,
            class_id,
            withdrawn_by,
        )

        return self._to_response(enrollment, student)

    async def remove_enrollment(
        self,
        class_id: UUID,
        student_id: UUID,
        removed_by: str | None = None,
    ) -> None:
        """Permanently remove an enrollment record.

        Args:
            class_id: Class identifier.
            student_id: Student identifier.
            removed_by: ID of user performing removal.

        Raises:
            NotEnrolledError: If enrollment not found.
        """
        enrollment = await self._get_enrollment(str(class_id), str(student_id))
        if not enrollment:
            raise NotEnrolledError("Enrollment not found")

        await self.db.delete(enrollment)
        await self.db.commit()

        logger.info(
            "Removed enrollment: student=%s, class=%s, by=%s",
            student_id,
            class_id,
            removed_by,
        )

    async def _get_class(self, class_id: UUID) -> Class:
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

    async def _get_student(self, student_id: UUID) -> User:
        """Get student user by ID.

        Args:
            student_id: Student identifier.

        Returns:
            User model instance.

        Raises:
            StudentNotFoundError: If not found.
            InvalidStudentTypeError: If not a student type.
        """
        query = select(User).where(
            User.id == str(student_id),
            User.deleted_at.is_(None),
        )
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise StudentNotFoundError(f"Student {student_id} not found")

        if user.user_type != "student":
            raise InvalidStudentTypeError(f"User {student_id} is not a student")

        return user

    async def _get_enrollment(
        self,
        class_id: str,
        student_id: str,
    ) -> ClassStudent | None:
        """Get enrollment record.

        Args:
            class_id: Class identifier.
            student_id: Student identifier.

        Returns:
            ClassStudent if found, None otherwise.
        """
        query = select(ClassStudent).where(
            ClassStudent.class_id == class_id,
            ClassStudent.student_id == student_id,
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    def _to_response(
        self,
        enrollment: ClassStudent,
        student: User,
    ) -> EnrollmentResponse:
        """Convert enrollment to response DTO.

        Args:
            enrollment: ClassStudent model instance.
            student: User model instance.

        Returns:
            EnrollmentResponse DTO.
        """
        return EnrollmentResponse(
            id=UUID(enrollment.id),
            class_id=UUID(enrollment.class_id),
            student=UserSummary(
                id=UUID(student.id),
                email=student.email,
                first_name=student.first_name,
                last_name=student.last_name,
                display_name=student.display_name,
                avatar_url=student.avatar_url,
                user_type=student.user_type,
                status=StatusEnum(student.status),
                sso_external_id=student.sso_external_id,
                created_at=student.created_at,
                last_login_at=student.last_login_at,
            ),
            student_number=enrollment.student_number,
            enrolled_at=enrollment.enrolled_at,
            withdrawn_at=enrollment.withdrawn_at,
            status=enrollment.status,
        )

    def _to_summary(self, enrollment: ClassStudent) -> EnrollmentSummary:
        """Convert enrollment to summary DTO.

        Args:
            enrollment: ClassStudent model instance.

        Returns:
            EnrollmentSummary DTO.
        """
        student = enrollment.student
        student_name = ""
        if student:
            student_name = f"{student.first_name} {student.last_name}".strip()

        return EnrollmentSummary(
            id=UUID(enrollment.id),
            student_id=UUID(enrollment.student_id),
            student_name=student_name,
            student_number=enrollment.student_number,
            enrolled_at=enrollment.enrolled_at,
            status=enrollment.status,
        )
