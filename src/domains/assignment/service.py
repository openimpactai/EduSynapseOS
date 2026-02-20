# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher assignment service for managing class teacher assignments.

This module provides the TeacherAssignmentService class for:
- Teacher assignment to classes
- Assignment termination
- Subject-specific assignments
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.school import Class, ClassTeacher
from src.infrastructure.database.models.tenant.curriculum import Subject
from src.infrastructure.database.models.tenant.user import User
from src.models.common import StatusEnum
from src.models.assignment import (
    AssignTeacherRequest,
    TeacherAssignmentResponse,
    TeacherAssignmentSummary,
)
from src.models.user import UserSummary

logger = logging.getLogger(__name__)


class AssignmentServiceError(Exception):
    """Base exception for assignment service errors."""

    pass


class ClassNotFoundError(AssignmentServiceError):
    """Raised when class is not found."""

    pass


class TeacherNotFoundError(AssignmentServiceError):
    """Raised when teacher is not found."""

    pass


class AlreadyAssignedError(AssignmentServiceError):
    """Raised when teacher is already assigned with same subject."""

    pass


class NotAssignedError(AssignmentServiceError):
    """Raised when teacher is not assigned to class."""

    pass


class InvalidTeacherTypeError(AssignmentServiceError):
    """Raised when user is not a teacher type."""

    pass


class TeacherAssignmentService:
    """Service for managing teacher assignments.

    This service handles all teacher assignment operations including
    assigning, ending, and listing assignments.

    Attributes:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize assignment service.

        Args:
            db: Async database session for tenant database.
        """
        self.db = db

    async def assign_teacher(
        self,
        class_id: UUID,
        request: AssignTeacherRequest,
        assigned_by: str,
    ) -> TeacherAssignmentResponse:
        """Assign a teacher to a class.

        Args:
            class_id: Class identifier.
            request: Assignment request data.
            assigned_by: ID of user performing assignment.

        Returns:
            Assignment response.

        Raises:
            ClassNotFoundError: If class not found.
            TeacherNotFoundError: If teacher not found.
            InvalidTeacherTypeError: If user is not a teacher.
            AlreadyAssignedError: If teacher already assigned.
        """
        # Verify class exists
        class_ = await self._get_class(class_id)

        # Verify teacher exists and is teacher type
        teacher = await self._get_teacher(request.teacher_id)

        # Check if already assigned with same subject
        existing = await self._get_active_assignment(
            str(class_id),
            str(request.teacher_id),
            request.subject_framework_code,
            request.subject_code,
        )
        if existing:
            raise AlreadyAssignedError(
                "Teacher is already assigned to this class with the same subject"
            )

        # Verify subject if provided
        subject = None
        if request.has_subject():
            subject = await self._get_subject(
                request.subject_framework_code,
                request.subject_code,
            )

        # Create assignment with composite subject key
        assignment = ClassTeacher(
            class_id=str(class_id),
            teacher_id=str(request.teacher_id),
            subject_framework_code=request.subject_framework_code,
            subject_code=request.subject_code,
            is_homeroom=request.is_homeroom,
        )

        self.db.add(assignment)
        await self.db.commit()
        await self.db.refresh(assignment)

        logger.info(
            "Assigned teacher: teacher=%s, class=%s, subject=%s, by=%s",
            request.teacher_id,
            class_id,
            f"{request.subject_framework_code}.{request.subject_code}" if request.has_subject() else None,
            assigned_by,
        )

        return self._to_response(assignment, teacher, subject)

    async def list_assignments(
        self,
        class_id: UUID,
        active_only: bool = True,
    ) -> tuple[list[TeacherAssignmentSummary], int, str]:
        """List teachers assigned to a class.

        Args:
            class_id: Class identifier.
            active_only: Only include active assignments.

        Returns:
            Tuple of (list of assignments, total count, class name).

        Raises:
            ClassNotFoundError: If class not found.
        """
        class_ = await self._get_class(class_id)

        query = select(ClassTeacher).options(
            selectinload(ClassTeacher.teacher),
            selectinload(ClassTeacher.subject),
        ).where(ClassTeacher.class_id == str(class_id))

        if active_only:
            query = query.where(ClassTeacher.ended_at.is_(None))

        query = query.order_by(ClassTeacher.assigned_at.desc())

        result = await self.db.execute(query)
        assignments = result.scalars().all()

        items = [self._to_summary(a) for a in assignments]

        return items, len(items), class_.name

    async def get_assignment(
        self,
        class_id: UUID,
        teacher_id: UUID,
        subject_framework_code: str | None = None,
        subject_code: str | None = None,
    ) -> TeacherAssignmentResponse:
        """Get specific assignment details.

        Args:
            class_id: Class identifier.
            teacher_id: Teacher identifier.
            subject_framework_code: Optional subject framework code.
            subject_code: Optional subject code.

        Returns:
            Assignment details.

        Raises:
            NotAssignedError: If assignment not found.
        """
        assignment = await self._get_active_assignment(
            str(class_id),
            str(teacher_id),
            subject_framework_code,
            subject_code,
        )
        if not assignment:
            raise NotAssignedError("Teacher is not assigned to this class")

        teacher = await self._get_teacher(teacher_id)
        subject = None
        if assignment.subject_framework_code and assignment.subject_code:
            subject = await self._get_subject(
                assignment.subject_framework_code,
                assignment.subject_code,
            )

        return self._to_response(assignment, teacher, subject)

    async def end_assignment(
        self,
        class_id: UUID,
        teacher_id: UUID,
        subject_framework_code: str | None = None,
        subject_code: str | None = None,
        ended_at: datetime | None = None,
        ended_by: str | None = None,
    ) -> TeacherAssignmentResponse:
        """End a teacher assignment.

        Args:
            class_id: Class identifier.
            teacher_id: Teacher identifier.
            subject_framework_code: Optional subject framework code.
            subject_code: Optional subject code.
            ended_at: Optional end date.
            ended_by: ID of user ending assignment.

        Returns:
            Updated assignment.

        Raises:
            NotAssignedError: If assignment not found.
        """
        assignment = await self._get_active_assignment(
            str(class_id),
            str(teacher_id),
            subject_framework_code,
            subject_code,
        )
        if not assignment:
            raise NotAssignedError("Teacher is not assigned to this class")

        assignment.ended_at = ended_at or datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(assignment)

        teacher = await self._get_teacher(teacher_id)
        subject = None
        if assignment.subject_framework_code and assignment.subject_code:
            subject = await self._get_subject(
                assignment.subject_framework_code,
                assignment.subject_code,
            )

        logger.info(
            "Ended teacher assignment: teacher=%s, class=%s, by=%s",
            teacher_id,
            class_id,
            ended_by,
        )

        return self._to_response(assignment, teacher, subject)

    async def remove_assignment(
        self,
        class_id: UUID,
        teacher_id: UUID,
        subject_framework_code: str | None = None,
        subject_code: str | None = None,
        removed_by: str | None = None,
    ) -> None:
        """Permanently remove an assignment record.

        Args:
            class_id: Class identifier.
            teacher_id: Teacher identifier.
            subject_framework_code: Optional subject framework code.
            subject_code: Optional subject code.
            removed_by: ID of user removing assignment.

        Raises:
            NotAssignedError: If assignment not found.
        """
        # Find any assignment (active or ended)
        query = select(ClassTeacher).where(
            ClassTeacher.class_id == str(class_id),
            ClassTeacher.teacher_id == str(teacher_id),
        )
        if subject_framework_code and subject_code:
            query = query.where(
                ClassTeacher.subject_framework_code == subject_framework_code,
                ClassTeacher.subject_code == subject_code,
            )
        else:
            query = query.where(ClassTeacher.subject_code.is_(None))

        result = await self.db.execute(query)
        assignment = result.scalar_one_or_none()

        if not assignment:
            raise NotAssignedError("Assignment not found")

        await self.db.delete(assignment)
        await self.db.commit()

        logger.info(
            "Removed teacher assignment: teacher=%s, class=%s, by=%s",
            teacher_id,
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

    async def _get_teacher(self, teacher_id: UUID) -> User:
        """Get teacher user by ID.

        Args:
            teacher_id: Teacher identifier.

        Returns:
            User model instance.

        Raises:
            TeacherNotFoundError: If not found.
            InvalidTeacherTypeError: If not a teacher type.
        """
        query = select(User).where(
            User.id == str(teacher_id),
            User.deleted_at.is_(None),
        )
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise TeacherNotFoundError(f"Teacher {teacher_id} not found")

        if user.user_type != "teacher":
            raise InvalidTeacherTypeError(f"User {teacher_id} is not a teacher")

        return user

    async def _get_subject(
        self,
        framework_code: str,
        subject_code: str,
    ) -> Subject:
        """Get subject by composite key.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").
            subject_code: Subject code (e.g., "MAT").

        Returns:
            Subject model instance.

        Raises:
            AssignmentServiceError: If not found.
        """
        query = select(Subject).where(
            Subject.framework_code == framework_code,
            Subject.code == subject_code,
        )
        result = await self.db.execute(query)
        subject = result.scalar_one_or_none()

        if not subject:
            raise AssignmentServiceError(
                f"Subject {framework_code}.{subject_code} not found"
            )

        return subject

    async def _get_active_assignment(
        self,
        class_id: str,
        teacher_id: str,
        subject_framework_code: str | None,
        subject_code: str | None,
    ) -> ClassTeacher | None:
        """Get active assignment record.

        Args:
            class_id: Class identifier.
            teacher_id: Teacher identifier.
            subject_framework_code: Subject's framework code (optional).
            subject_code: Subject code (optional).

        Returns:
            ClassTeacher if found, None otherwise.
        """
        query = select(ClassTeacher).where(
            ClassTeacher.class_id == class_id,
            ClassTeacher.teacher_id == teacher_id,
            ClassTeacher.ended_at.is_(None),
        )
        if subject_framework_code and subject_code:
            query = query.where(
                ClassTeacher.subject_framework_code == subject_framework_code,
                ClassTeacher.subject_code == subject_code,
            )
        else:
            query = query.where(ClassTeacher.subject_code.is_(None))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    def _to_response(
        self,
        assignment: ClassTeacher,
        teacher: User,
        subject: Subject | None,
    ) -> TeacherAssignmentResponse:
        """Convert assignment to response DTO.

        Args:
            assignment: ClassTeacher model instance.
            teacher: User model instance.
            subject: Optional Subject model instance.

        Returns:
            TeacherAssignmentResponse DTO.
        """
        # Build subject full code from composite keys
        subject_full_code = None
        if assignment.subject_framework_code and assignment.subject_code:
            subject_full_code = f"{assignment.subject_framework_code}.{assignment.subject_code}"

        return TeacherAssignmentResponse(
            id=UUID(assignment.id),
            class_id=UUID(assignment.class_id),
            teacher=UserSummary(
                id=UUID(teacher.id),
                email=teacher.email,
                first_name=teacher.first_name,
                last_name=teacher.last_name,
                display_name=teacher.display_name,
                avatar_url=teacher.avatar_url,
                user_type=teacher.user_type,
                status=StatusEnum(teacher.status),
                sso_external_id=teacher.sso_external_id,
                created_at=teacher.created_at,
                last_login_at=teacher.last_login_at,
            ),
            subject_full_code=subject_full_code,
            subject_name=subject.name if subject else None,
            is_homeroom=assignment.is_homeroom,
            assigned_at=assignment.assigned_at,
            ended_at=assignment.ended_at,
            is_active=assignment.ended_at is None,
        )

    def _to_summary(self, assignment: ClassTeacher) -> TeacherAssignmentSummary:
        """Convert assignment to summary DTO.

        Args:
            assignment: ClassTeacher model instance.

        Returns:
            TeacherAssignmentSummary DTO.
        """
        teacher = assignment.teacher
        teacher_name = ""
        if teacher:
            teacher_name = f"{teacher.first_name} {teacher.last_name}".strip()

        subject_name = None
        if assignment.subject:
            subject_name = assignment.subject.name

        return TeacherAssignmentSummary(
            id=UUID(assignment.id),
            teacher_id=UUID(assignment.teacher_id),
            teacher_name=teacher_name,
            subject_name=subject_name,
            is_homeroom=assignment.is_homeroom,
            assigned_at=assignment.assigned_at,
            is_active=assignment.ended_at is None,
        )
