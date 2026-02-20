# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Helper functions for teacher tools.

Provides common functionality for teacher tools including:
- Authorization verification
- Student access checks
- Class access checks
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.school import (
    Class,
    ClassStudent,
    ClassTeacher,
)


async def get_teacher_student_ids(
    session: AsyncSession,
    teacher_id: UUID,
) -> list[str]:
    """Get all student IDs that a teacher has access to through their classes.

    Args:
        session: Database session.
        teacher_id: Teacher's user ID.

    Returns:
        List of student IDs (as strings) that the teacher can access.
    """
    query = (
        select(ClassStudent.student_id)
        .join(Class, ClassStudent.class_id == Class.id)
        .join(ClassTeacher, ClassTeacher.class_id == Class.id)
        .where(ClassTeacher.teacher_id == str(teacher_id))
        .where(ClassTeacher.ended_at.is_(None))
        .where(Class.is_active == True)
        .where(ClassStudent.status == "active")
        .distinct()
    )

    result = await session.execute(query)
    return [str(row[0]) for row in result.all()]


async def verify_teacher_has_student_access(
    session: AsyncSession,
    teacher_id: UUID,
    student_id: UUID,
) -> bool:
    """Verify that a teacher has access to a specific student.

    A teacher has access to a student if the student is enrolled
    in any class that the teacher is assigned to.

    Args:
        session: Database session.
        teacher_id: Teacher's user ID.
        student_id: Student's user ID.

    Returns:
        True if teacher has access to the student, False otherwise.
    """
    query = (
        select(ClassStudent.id)
        .join(Class, ClassStudent.class_id == Class.id)
        .join(ClassTeacher, ClassTeacher.class_id == Class.id)
        .where(ClassTeacher.teacher_id == str(teacher_id))
        .where(ClassTeacher.ended_at.is_(None))
        .where(Class.is_active == True)
        .where(ClassStudent.student_id == str(student_id))
        .where(ClassStudent.status == "active")
        .limit(1)
    )

    result = await session.execute(query)
    return result.first() is not None


async def verify_teacher_has_class_access(
    session: AsyncSession,
    teacher_id: UUID,
    class_id: UUID,
) -> bool:
    """Verify that a teacher has access to a specific class.

    A teacher has access to a class if they are assigned to it
    and the assignment hasn't ended.

    Args:
        session: Database session.
        teacher_id: Teacher's user ID.
        class_id: Class ID.

    Returns:
        True if teacher has access to the class, False otherwise.
    """
    query = (
        select(ClassTeacher.id)
        .join(Class, ClassTeacher.class_id == Class.id)
        .where(ClassTeacher.teacher_id == str(teacher_id))
        .where(ClassTeacher.class_id == str(class_id))
        .where(ClassTeacher.ended_at.is_(None))
        .where(Class.is_active == True)
        .limit(1)
    )

    result = await session.execute(query)
    return result.first() is not None


async def get_teacher_class_ids(
    session: AsyncSession,
    teacher_id: UUID,
) -> list[str]:
    """Get all class IDs that a teacher is assigned to.

    Args:
        session: Database session.
        teacher_id: Teacher's user ID.

    Returns:
        List of class IDs (as strings) that the teacher is assigned to.
    """
    query = (
        select(ClassTeacher.class_id)
        .join(Class, ClassTeacher.class_id == Class.id)
        .where(ClassTeacher.teacher_id == str(teacher_id))
        .where(ClassTeacher.ended_at.is_(None))
        .where(Class.is_active == True)
    )

    result = await session.execute(query)
    return [str(row[0]) for row in result.all()]
