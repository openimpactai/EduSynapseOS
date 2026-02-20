# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for Class service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

import pytest

from src.domains.class_.service import (
    ClassService,
    ClassNotFoundError,
    ClassCodeExistsError,
    SchoolNotFoundError,
    AcademicYearNotFoundError,
)
from src.models.class_ import (
    ClassCreateRequest,
    ClassUpdateRequest,
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
def class_service(mock_db):
    """Create class service with mock database."""
    return ClassService(db=mock_db)


@pytest.fixture
def sample_school():
    """Create a sample school model."""
    school = MagicMock()
    school.id = str(uuid4())
    school.code = "SCH001"
    school.name = "Test School"
    school.is_active = True
    return school


@pytest.fixture
def sample_academic_year():
    """Create a sample academic year model."""
    year = MagicMock()
    year.id = str(uuid4())
    year.name = "2024-2025"
    year.is_current = True
    return year


@pytest.fixture
def sample_grade_level():
    """Create a sample grade level model."""
    grade = MagicMock()
    grade.id = str(uuid4())
    grade.name = "Grade 1"
    grade.code = "G1"
    return grade


@pytest.fixture
def sample_class(sample_school, sample_academic_year, sample_grade_level):
    """Create a sample class model."""
    cls = MagicMock()
    cls.id = str(uuid4())
    cls.code = "CLS001"
    cls.name = "Class 1A"
    cls.school_id = sample_school.id
    cls.academic_year_id = sample_academic_year.id
    cls.grade_level_id = sample_grade_level.id
    cls.max_students = 30
    cls.is_active = True
    cls.created_at = datetime.now(timezone.utc)
    cls.updated_at = datetime.now(timezone.utc)
    cls.school = sample_school
    cls.academic_year = sample_academic_year
    cls.grade_level = sample_grade_level
    return cls


class TestClassServiceCreate:
    """Tests for class creation."""

    @pytest.mark.asyncio
    async def test_create_class_success(self, class_service, mock_db, sample_school, sample_academic_year):
        """Test successful class creation."""
        request = ClassCreateRequest(
            code="NEW001",
            name="New Class",
            school_id=UUID(sample_school.id),
            academic_year_id=UUID(sample_academic_year.id),
        )

        # Mock school exists
        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        # Mock academic year exists
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock no existing class with same code
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = None

        # Mock student count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        # Mock teacher count query
        mock_teacher_count_result = MagicMock()
        mock_teacher_count_result.scalar.return_value = 0

        mock_db.execute.side_effect = [
            mock_school_result,
            mock_year_result,
            mock_class_result,
            mock_count_result,
            mock_teacher_count_result,
        ]

        # Mock refresh to set values
        async def mock_refresh(obj):
            obj.id = str(uuid4())
            obj.code = request.code
            obj.name = request.name
            obj.school_id = str(request.school_id)
            obj.academic_year_id = str(request.academic_year_id)
            obj.is_active = True
            obj.created_at = datetime.now(timezone.utc)
            obj.updated_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = mock_refresh

        result = await class_service.create_class(
            request=request,
            created_by=str(uuid4()),
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert result.code == "NEW001"
        assert result.name == "New Class"

    @pytest.mark.asyncio
    async def test_create_class_school_not_found(self, class_service, mock_db):
        """Test creation fails when school not found."""
        request = ClassCreateRequest(
            code="NEW001",
            name="New Class",
            school_id=uuid4(),
            academic_year_id=uuid4(),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await class_service.create_class(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_class_academic_year_not_found(self, class_service, mock_db, sample_school):
        """Test creation fails when academic year not found."""
        request = ClassCreateRequest(
            code="NEW001",
            name="New Class",
            school_id=UUID(sample_school.id),
            academic_year_id=uuid4(),
        )

        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_school_result, mock_year_result]

        with pytest.raises(AcademicYearNotFoundError):
            await class_service.create_class(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_class_code_exists(self, class_service, mock_db, sample_school, sample_academic_year, sample_class):
        """Test creation fails when code exists."""
        request = ClassCreateRequest(
            code="CLS001",
            name="Another Class",
            school_id=UUID(sample_school.id),
            academic_year_id=UUID(sample_academic_year.id),
        )

        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        mock_db.execute.side_effect = [mock_school_result, mock_year_result, mock_class_result]

        with pytest.raises(ClassCodeExistsError):
            await class_service.create_class(
                request=request,
                created_by=str(uuid4()),
            )


class TestClassServiceGet:
    """Tests for getting classes."""

    @pytest.mark.asyncio
    async def test_get_class_success(self, class_service, mock_db, sample_class, sample_school, sample_academic_year, sample_grade_level):
        """Test successful class retrieval."""
        # Mock class query
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # Mock school query
        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        # Mock academic year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock grade level query
        mock_grade_result = MagicMock()
        mock_grade_result.scalar_one_or_none.return_value = sample_grade_level

        # Mock student count query
        mock_student_count = MagicMock()
        mock_student_count.scalar.return_value = 5

        # Mock teacher count query
        mock_teacher_count = MagicMock()
        mock_teacher_count.scalar.return_value = 2

        mock_db.execute.side_effect = [
            mock_class_result,
            mock_school_result,
            mock_year_result,
            mock_grade_result,
            mock_student_count,
            mock_teacher_count,
        ]

        result = await class_service.get_class(UUID(sample_class.id))

        assert str(result.id) == sample_class.id
        assert result.code == sample_class.code

    @pytest.mark.asyncio
    async def test_get_class_not_found(self, class_service, mock_db):
        """Test class not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ClassNotFoundError):
            await class_service.get_class(uuid4())


class TestClassServiceList:
    """Tests for listing classes."""

    @pytest.mark.asyncio
    async def test_list_classes_empty(self, class_service, mock_db):
        """Test listing classes when empty."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_classes_result = MagicMock()
        mock_classes_result.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_count_result, mock_classes_result]

        items, total = await class_service.list_classes()

        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_classes_with_results(self, class_service, mock_db, sample_class):
        """Test listing classes with results."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_classes_result = MagicMock()
        mock_classes_result.scalars.return_value.all.return_value = [sample_class]

        # Mock student count query for each class
        mock_student_count = MagicMock()
        mock_student_count.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_count_result, mock_classes_result, mock_student_count]

        items, total = await class_service.list_classes()

        assert len(items) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_classes_with_school_filter(self, class_service, mock_db, sample_class):
        """Test listing classes with school filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_classes_result = MagicMock()
        mock_classes_result.scalars.return_value.all.return_value = [sample_class]

        # Mock student count query for each class
        mock_student_count = MagicMock()
        mock_student_count.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_count_result, mock_classes_result, mock_student_count]

        items, total = await class_service.list_classes(school_id=uuid4())

        assert len(items) == 1


class TestClassServiceUpdate:
    """Tests for updating classes."""

    @pytest.mark.asyncio
    async def test_update_class_success(self, class_service, mock_db, sample_class, sample_school, sample_academic_year, sample_grade_level):
        """Test successful class update."""
        # Mock class query
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # Mock school query
        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        # Mock academic year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock grade level query
        mock_grade_result = MagicMock()
        mock_grade_result.scalar_one_or_none.return_value = sample_grade_level

        # Mock student count query
        mock_student_count = MagicMock()
        mock_student_count.scalar.return_value = 5

        # Mock teacher count query
        mock_teacher_count = MagicMock()
        mock_teacher_count.scalar.return_value = 2

        mock_db.execute.side_effect = [
            mock_class_result,
            mock_school_result,
            mock_year_result,
            mock_grade_result,
            mock_student_count,
            mock_teacher_count,
        ]

        request = ClassUpdateRequest(name="Updated Class Name")

        result = await class_service.update_class(
            class_id=UUID(sample_class.id),
            request=request,
        )

        mock_db.commit.assert_called_once()
        assert sample_class.name == "Updated Class Name"

    @pytest.mark.asyncio
    async def test_update_class_not_found(self, class_service, mock_db):
        """Test update class not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        request = ClassUpdateRequest(name="Updated Class")

        with pytest.raises(ClassNotFoundError):
            await class_service.update_class(
                class_id=uuid4(),
                request=request,
            )


class TestClassServiceDeactivate:
    """Tests for deactivating classes."""

    @pytest.mark.asyncio
    async def test_deactivate_class_success(self, class_service, mock_db, sample_class):
        """Test successful class deactivation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_class
        mock_db.execute.return_value = mock_result

        await class_service.deactivate_class(class_id=UUID(sample_class.id))

        mock_db.commit.assert_called_once()
        assert sample_class.is_active is False

    @pytest.mark.asyncio
    async def test_deactivate_class_not_found(self, class_service, mock_db):
        """Test deactivate class not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ClassNotFoundError):
            await class_service.deactivate_class(class_id=uuid4())


class TestClassServiceActivate:
    """Tests for activating classes."""

    @pytest.mark.asyncio
    async def test_activate_class_success(self, class_service, mock_db, sample_class, sample_school, sample_academic_year, sample_grade_level):
        """Test successful class activation."""
        sample_class.is_active = False

        # Mock class query
        mock_class_result = MagicMock()
        mock_class_result.scalar_one_or_none.return_value = sample_class

        # Mock school query
        mock_school_result = MagicMock()
        mock_school_result.scalar_one_or_none.return_value = sample_school

        # Mock academic year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock grade level query
        mock_grade_result = MagicMock()
        mock_grade_result.scalar_one_or_none.return_value = sample_grade_level

        # Mock student count query
        mock_student_count = MagicMock()
        mock_student_count.scalar.return_value = 5

        # Mock teacher count query
        mock_teacher_count = MagicMock()
        mock_teacher_count.scalar.return_value = 2

        mock_db.execute.side_effect = [
            mock_class_result,
            mock_school_result,
            mock_year_result,
            mock_grade_result,
            mock_student_count,
            mock_teacher_count,
        ]

        result = await class_service.activate_class(class_id=UUID(sample_class.id))

        mock_db.commit.assert_called_once()
        assert sample_class.is_active is True

    @pytest.mark.asyncio
    async def test_activate_class_not_found(self, class_service, mock_db):
        """Test activate class not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ClassNotFoundError):
            await class_service.activate_class(class_id=uuid4())
