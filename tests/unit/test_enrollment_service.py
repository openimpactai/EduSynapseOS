# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for Enrollment service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

import pytest

from src.domains.enrollment.service import (
    EnrollmentService,
    NotEnrolledError,
    StudentNotFoundError,
    ClassNotFoundError,
    AlreadyEnrolledError,
    InvalidStudentTypeError,
)
from src.models.enrollment import (
    EnrollStudentRequest,
    BulkEnrollRequest,
)


@pytest.fixture
def mock_db():
    """Create mock database session."""
    db = AsyncMock()
    db.add = MagicMock()
    db.add_all = MagicMock()
    db.delete = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def enrollment_service(mock_db):
    """Create enrollment service with mock database."""
    return EnrollmentService(db=mock_db)


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
def sample_student():
    """Create a sample student model."""
    student = MagicMock()
    student.id = str(uuid4())
    student.email = "student@school.com"
    student.first_name = "John"
    student.last_name = "Doe"
    student.display_name = "John Doe"
    student.avatar_url = None
    student.user_type = "student"
    student.status = "active"
    student.deleted_at = None
    return student


@pytest.fixture
def sample_enrollment(sample_class, sample_student):
    """Create a sample enrollment model."""
    enrollment = MagicMock()
    enrollment.id = str(uuid4())
    enrollment.class_id = sample_class.id
    enrollment.student_id = sample_student.id
    enrollment.student_number = "STU001"
    enrollment.enrolled_at = datetime.now(timezone.utc)
    enrollment.withdrawn_at = None
    enrollment.status = "active"
    enrollment.student = sample_student
    enrollment.class_ = sample_class
    return enrollment


class TestEnrollmentServiceEnroll:
    """Tests for student enrollment."""

    @pytest.mark.asyncio
    async def test_enroll_student_success(self, enrollment_service, mock_db, sample_class, sample_student):
        """Test successful student enrollment."""
        request = EnrollStudentRequest(
            student_id=UUID(sample_student.id),
        )

        # Mock class exists
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # Mock student exists
        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        # Mock no existing enrollment
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_class_result, mock_student_result, mock_enrollment_result]

        # Mock refresh to set proper id on the new enrollment object
        async def mock_refresh(obj):
            obj.id = str(uuid4())
            obj.class_id = str(sample_class.id)
            obj.student_id = str(sample_student.id)
            obj.student_number = None
            obj.status = "active"
            obj.enrolled_at = datetime.now(timezone.utc)
            obj.withdrawn_at = None

        mock_db.refresh.side_effect = mock_refresh

        result = await enrollment_service.enroll_student(
            class_id=UUID(sample_class.id),
            request=request,
            enrolled_by=str(uuid4()),
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_enroll_student_class_not_found(self, enrollment_service, mock_db):
        """Test enrollment fails when class not found."""
        request = EnrollStudentRequest(student_id=uuid4())

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ClassNotFoundError):
            await enrollment_service.enroll_student(
                class_id=uuid4(),
                request=request,
                enrolled_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_enroll_student_not_found(self, enrollment_service, mock_db, sample_class):
        """Test enrollment fails when student not found."""
        request = EnrollStudentRequest(student_id=uuid4())

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_class_result, mock_student_result]

        with pytest.raises(StudentNotFoundError):
            await enrollment_service.enroll_student(
                class_id=UUID(sample_class.id),
                request=request,
                enrolled_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_enroll_student_invalid_user_type(self, enrollment_service, mock_db, sample_class, sample_student):
        """Test enrollment fails when user is not a student."""
        sample_student.user_type = "teacher"

        request = EnrollStudentRequest(student_id=UUID(sample_student.id))

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_class_result, mock_student_result]

        with pytest.raises(InvalidStudentTypeError):
            await enrollment_service.enroll_student(
                class_id=UUID(sample_class.id),
                request=request,
                enrolled_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_enroll_student_already_enrolled(self, enrollment_service, mock_db, sample_class, sample_student, sample_enrollment):
        """Test enrollment fails when student already enrolled."""
        request = EnrollStudentRequest(student_id=UUID(sample_student.id))

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = sample_enrollment

        mock_db.execute.side_effect = [mock_class_result, mock_student_result, mock_enrollment_result]

        with pytest.raises(AlreadyEnrolledError):
            await enrollment_service.enroll_student(
                class_id=UUID(sample_class.id),
                request=request,
                enrolled_by=str(uuid4()),
            )


class TestEnrollmentServiceBulkEnroll:
    """Tests for bulk student enrollment."""

    @pytest.mark.asyncio
    async def test_bulk_enroll_success(self, enrollment_service, mock_db, sample_class):
        """Test successful bulk enrollment."""
        student1 = MagicMock()
        student1.id = str(uuid4())
        student1.email = "s1@school.com"
        student1.first_name = "Student"
        student1.last_name = "One"
        student1.display_name = "Student One"
        student1.avatar_url = None
        student1.user_type = "student"
        student1.status = "active"
        student1.deleted_at = None

        student2 = MagicMock()
        student2.id = str(uuid4())
        student2.email = "s2@school.com"
        student2.first_name = "Student"
        student2.last_name = "Two"
        student2.display_name = "Student Two"
        student2.avatar_url = None
        student2.user_type = "student"
        student2.status = "active"
        student2.deleted_at = None

        student_ids = [UUID(student1.id), UUID(student2.id)]
        request = BulkEnrollRequest(student_ids=student_ids)

        # Mock class exists (called once at start)
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # For each student: class check, student check, enrollment check
        mock_student1_result = MagicMock()
        mock_student1_result.scalar_one_or_none.return_value = student1
        mock_enrollment1_result = MagicMock()
        mock_enrollment1_result.scalar_one_or_none.return_value = None

        mock_student2_result = MagicMock()
        mock_student2_result.scalar_one_or_none.return_value = student2
        mock_enrollment2_result = MagicMock()
        mock_enrollment2_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [
            mock_class_result,  # Initial class check
            mock_class_result, mock_student1_result, mock_enrollment1_result,  # First student
            mock_class_result, mock_student2_result, mock_enrollment2_result,  # Second student
        ]

        # Mock refresh to set proper id on new enrollment objects
        call_count = [0]
        students = [student1, student2]

        async def mock_refresh(obj):
            idx = call_count[0]
            call_count[0] += 1
            obj.id = str(uuid4())
            obj.class_id = str(sample_class.id)
            obj.student_id = students[idx].id if idx < len(students) else str(uuid4())
            obj.student_number = None
            obj.status = "active"
            obj.enrolled_at = datetime.now(timezone.utc)
            obj.withdrawn_at = None

        mock_db.refresh.side_effect = mock_refresh

        result = await enrollment_service.bulk_enroll(
            class_id=UUID(sample_class.id),
            request=request,
            enrolled_by=str(uuid4()),
        )

        assert result.total_enrolled == 2
        assert result.total_failed == 0


class TestEnrollmentServiceList:
    """Tests for listing enrollments."""

    @pytest.mark.asyncio
    async def test_list_enrollments_empty(self, enrollment_service, mock_db, sample_class):
        """Test listing enrollments when empty."""
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_enrollments_result = MagicMock()
        mock_enrollments_result.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_class_result, mock_enrollments_result]

        items, total, class_name = await enrollment_service.list_enrollments(class_id=UUID(sample_class.id))

        assert items == []
        assert total == 0
        assert class_name == sample_class.name

    @pytest.mark.asyncio
    async def test_list_enrollments_with_results(self, enrollment_service, mock_db, sample_class, sample_enrollment):
        """Test listing enrollments with results."""
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_enrollments_result = MagicMock()
        mock_enrollments_result.scalars.return_value.all.return_value = [sample_enrollment]

        mock_db.execute.side_effect = [mock_class_result, mock_enrollments_result]

        items, total, class_name = await enrollment_service.list_enrollments(class_id=UUID(sample_class.id))

        assert len(items) == 1
        assert total == 1
        assert class_name == sample_class.name


class TestEnrollmentServiceGet:
    """Tests for getting enrollment."""

    @pytest.mark.asyncio
    async def test_get_enrollment_success(self, enrollment_service, mock_db, sample_class, sample_student, sample_enrollment):
        """Test successful enrollment retrieval."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = sample_enrollment

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_enrollment_result, mock_student_result]

        result = await enrollment_service.get_enrollment(
            class_id=UUID(sample_class.id),
            student_id=UUID(sample_student.id),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_enrollment_not_found(self, enrollment_service, mock_db, sample_class):
        """Test enrollment not found error."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_enrollment_result

        with pytest.raises(NotEnrolledError):
            await enrollment_service.get_enrollment(
                class_id=UUID(sample_class.id),
                student_id=uuid4(),
            )


class TestEnrollmentServiceWithdraw:
    """Tests for withdrawing students."""

    @pytest.mark.asyncio
    async def test_withdraw_student_success(self, enrollment_service, mock_db, sample_class, sample_student, sample_enrollment):
        """Test successful student withdrawal."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = sample_enrollment

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_enrollment_result, mock_student_result]

        result = await enrollment_service.withdraw_student(
            class_id=UUID(sample_class.id),
            student_id=UUID(sample_student.id),
            withdrawn_by=str(uuid4()),
        )

        mock_db.commit.assert_called_once()
        assert sample_enrollment.status == "withdrawn"

    @pytest.mark.asyncio
    async def test_withdraw_student_not_found(self, enrollment_service, mock_db, sample_class):
        """Test withdraw fails when enrollment not found."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_enrollment_result

        with pytest.raises(NotEnrolledError):
            await enrollment_service.withdraw_student(
                class_id=UUID(sample_class.id),
                student_id=uuid4(),
                withdrawn_by=str(uuid4()),
            )


class TestEnrollmentServiceRemove:
    """Tests for removing enrollments."""

    @pytest.mark.asyncio
    async def test_remove_enrollment_success(self, enrollment_service, mock_db, sample_class, sample_student, sample_enrollment):
        """Test successful enrollment removal."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = sample_enrollment

        mock_db.execute.return_value = mock_enrollment_result

        await enrollment_service.remove_enrollment(
            class_id=UUID(sample_class.id),
            student_id=UUID(sample_student.id),
            removed_by=str(uuid4()),
        )

        mock_db.delete.assert_called_once_with(sample_enrollment)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_enrollment_not_found(self, enrollment_service, mock_db, sample_class):
        """Test remove fails when enrollment not found."""
        mock_enrollment_result = MagicMock()
        mock_enrollment_result.scalar_one_or_none.return_value = None

        mock_db.execute.return_value = mock_enrollment_result

        with pytest.raises(NotEnrolledError):
            await enrollment_service.remove_enrollment(
                class_id=UUID(sample_class.id),
                student_id=uuid4(),
                removed_by=str(uuid4()),
            )
