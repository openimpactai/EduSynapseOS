# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for Teacher Assignment service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

import pytest

from src.domains.assignment.service import (
    TeacherAssignmentService,
    NotAssignedError,
    TeacherNotFoundError,
    ClassNotFoundError,
    AlreadyAssignedError,
    InvalidTeacherTypeError,
)
from src.models.assignment import (
    AssignTeacherRequest,
)


@pytest.fixture
def mock_db():
    """Create mock database session."""
    db = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def assignment_service(mock_db):
    """Create teacher assignment service with mock database."""
    return TeacherAssignmentService(db=mock_db)


@pytest.fixture
def sample_class():
    """Create a sample class model."""
    cls = MagicMock()
    cls.id = str(uuid4())
    cls.code = "CLS001"
    cls.name = "Class 1A"
    cls.is_active = True
    return cls


@pytest.fixture
def sample_teacher():
    """Create a sample teacher model."""
    teacher = MagicMock()
    teacher.id = str(uuid4())
    teacher.email = "teacher@school.com"
    teacher.first_name = "Jane"
    teacher.last_name = "Smith"
    teacher.display_name = "Jane Smith"
    teacher.avatar_url = None
    teacher.user_type = "teacher"
    teacher.status = "active"
    teacher.deleted_at = None
    return teacher


@pytest.fixture
def sample_subject():
    """Create a sample subject model."""
    subject = MagicMock()
    subject.id = str(uuid4())
    subject.name = "Mathematics"
    subject.code = "MATH"
    return subject


@pytest.fixture
def sample_assignment(sample_class, sample_teacher, sample_subject):
    """Create a sample assignment model."""
    assignment = MagicMock()
    assignment.id = str(uuid4())
    assignment.class_id = sample_class.id
    assignment.teacher_id = sample_teacher.id
    assignment.subject_id = sample_subject.id
    assignment.is_homeroom = False
    assignment.assigned_at = datetime.now(timezone.utc)
    assignment.ended_at = None
    assignment.teacher = sample_teacher
    assignment.class_ = sample_class
    assignment.subject = sample_subject
    return assignment


class TestTeacherAssignmentServiceAssign:
    """Tests for teacher assignment."""

    @pytest.mark.asyncio
    async def test_assign_teacher_success(self, assignment_service, mock_db, sample_class, sample_teacher, sample_subject):
        """Test successful teacher assignment."""
        request = AssignTeacherRequest(
            teacher_id=UUID(sample_teacher.id),
            subject_id=UUID(sample_subject.id),
        )

        # Mock class exists
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # Mock teacher exists
        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = sample_teacher

        # Mock no existing assignment
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = None

        # Mock subject exists
        mock_subject_result = MagicMock()
        mock_subject_result.scalar_one_or_none.return_value = sample_subject

        mock_db.execute.side_effect = [
            mock_class_result,
            mock_teacher_result,
            mock_assignment_result,
            mock_subject_result,
        ]

        # Mock refresh to set proper id on the new assignment object
        async def mock_refresh(obj):
            obj.id = str(uuid4())
            obj.class_id = str(sample_class.id)
            obj.teacher_id = str(sample_teacher.id)
            obj.subject_id = str(sample_subject.id)
            obj.is_homeroom = False
            obj.assigned_at = datetime.now(timezone.utc)
            obj.ended_at = None

        mock_db.refresh.side_effect = mock_refresh

        result = await assignment_service.assign_teacher(
            class_id=UUID(sample_class.id),
            request=request,
            assigned_by=str(uuid4()),
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_assign_teacher_class_not_found(self, assignment_service, mock_db):
        """Test assignment fails when class not found."""
        request = AssignTeacherRequest(
            teacher_id=uuid4(),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ClassNotFoundError):
            await assignment_service.assign_teacher(
                class_id=uuid4(),
                request=request,
                assigned_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_assign_teacher_not_found(self, assignment_service, mock_db, sample_class):
        """Test assignment fails when teacher not found."""
        request = AssignTeacherRequest(
            teacher_id=uuid4(),
        )

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_class_result, mock_teacher_result]

        with pytest.raises(TeacherNotFoundError):
            await assignment_service.assign_teacher(
                class_id=UUID(sample_class.id),
                request=request,
                assigned_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_assign_teacher_invalid_user_type(self, assignment_service, mock_db, sample_class, sample_teacher):
        """Test assignment fails when user is not a teacher."""
        sample_teacher.user_type = "student"

        request = AssignTeacherRequest(
            teacher_id=UUID(sample_teacher.id),
        )

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = sample_teacher

        mock_db.execute.side_effect = [mock_class_result, mock_teacher_result]

        with pytest.raises(InvalidTeacherTypeError):
            await assignment_service.assign_teacher(
                class_id=UUID(sample_class.id),
                request=request,
                assigned_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_assign_teacher_already_assigned(self, assignment_service, mock_db, sample_class, sample_teacher, sample_assignment):
        """Test assignment fails when teacher already assigned."""
        request = AssignTeacherRequest(
            teacher_id=UUID(sample_teacher.id),
            subject_id=UUID(sample_assignment.subject_id),
        )

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = sample_teacher

        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = sample_assignment

        mock_db.execute.side_effect = [mock_class_result, mock_teacher_result, mock_assignment_result]

        with pytest.raises(AlreadyAssignedError):
            await assignment_service.assign_teacher(
                class_id=UUID(sample_class.id),
                request=request,
                assigned_by=str(uuid4()),
            )


class TestTeacherAssignmentServiceList:
    """Tests for listing assignments."""

    @pytest.mark.asyncio
    async def test_list_assignments_empty(self, assignment_service, mock_db, sample_class):
        """Test listing assignments when empty."""
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_assignments_result = MagicMock()
        mock_assignments_result.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_class_result, mock_assignments_result]

        items, total, class_name = await assignment_service.list_assignments(class_id=UUID(sample_class.id))

        assert items == []
        assert total == 0
        assert class_name == sample_class.name

    @pytest.mark.asyncio
    async def test_list_assignments_with_results(self, assignment_service, mock_db, sample_class, sample_assignment):
        """Test listing assignments with results."""
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_assignments_result = MagicMock()
        mock_assignments_result.scalars.return_value.all.return_value = [sample_assignment]

        mock_db.execute.side_effect = [mock_class_result, mock_assignments_result]

        items, total, class_name = await assignment_service.list_assignments(class_id=UUID(sample_class.id))

        assert len(items) == 1
        assert total == 1
        assert class_name == sample_class.name


class TestTeacherAssignmentServiceGet:
    """Tests for getting assignment."""

    @pytest.mark.asyncio
    async def test_get_assignment_success(self, assignment_service, mock_db, sample_class, sample_teacher, sample_subject, sample_assignment):
        """Test successful assignment retrieval."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = sample_assignment

        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = sample_teacher

        mock_subject_result = MagicMock()
        mock_subject_result.scalar_one_or_none.return_value = sample_subject

        mock_db.execute.side_effect = [mock_assignment_result, mock_teacher_result, mock_subject_result]

        result = await assignment_service.get_assignment(
            class_id=UUID(sample_class.id),
            teacher_id=UUID(sample_teacher.id),
            subject_id=UUID(sample_subject.id),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_assignment_not_found(self, assignment_service, mock_db, sample_class):
        """Test assignment not found error."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_assignment_result

        with pytest.raises(NotAssignedError):
            await assignment_service.get_assignment(
                class_id=UUID(sample_class.id),
                teacher_id=uuid4(),
            )


class TestTeacherAssignmentServiceEndAssignment:
    """Tests for ending teacher assignments."""

    @pytest.mark.asyncio
    async def test_end_assignment_success(self, assignment_service, mock_db, sample_class, sample_teacher, sample_subject, sample_assignment):
        """Test successful assignment end."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = sample_assignment

        mock_teacher_result = MagicMock()
        mock_teacher_result.scalar_one_or_none.return_value = sample_teacher

        mock_subject_result = MagicMock()
        mock_subject_result.scalar_one_or_none.return_value = sample_subject

        mock_db.execute.side_effect = [mock_assignment_result, mock_teacher_result, mock_subject_result]

        result = await assignment_service.end_assignment(
            class_id=UUID(sample_class.id),
            teacher_id=UUID(sample_teacher.id),
            subject_id=UUID(sample_subject.id),
            ended_at=datetime.now(timezone.utc),
            ended_by=str(uuid4()),
        )

        mock_db.commit.assert_called_once()
        assert sample_assignment.ended_at is not None

    @pytest.mark.asyncio
    async def test_end_assignment_not_found(self, assignment_service, mock_db, sample_class):
        """Test end fails when assignment not found."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_assignment_result

        with pytest.raises(NotAssignedError):
            await assignment_service.end_assignment(
                class_id=UUID(sample_class.id),
                teacher_id=uuid4(),
                ended_at=datetime.now(timezone.utc),
                ended_by=str(uuid4()),
            )


class TestTeacherAssignmentServiceRemove:
    """Tests for removing assignments."""

    @pytest.mark.asyncio
    async def test_remove_assignment_success(self, assignment_service, mock_db, sample_class, sample_teacher, sample_assignment):
        """Test successful assignment removal."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = sample_assignment

        mock_db.execute.return_value = mock_assignment_result

        await assignment_service.remove_assignment(
            class_id=UUID(sample_class.id),
            teacher_id=UUID(sample_teacher.id),
            removed_by=str(uuid4()),
        )

        mock_db.delete.assert_called_once_with(sample_assignment)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_assignment_not_found(self, assignment_service, mock_db, sample_class):
        """Test remove fails when assignment not found."""
        mock_assignment_result = MagicMock()
        mock_assignment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_assignment_result

        with pytest.raises(NotAssignedError):
            await assignment_service.remove_assignment(
                class_id=UUID(sample_class.id),
                teacher_id=uuid4(),
                removed_by=str(uuid4()),
            )
