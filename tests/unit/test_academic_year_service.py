# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for Academic Year service."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

import pytest

from src.domains.academic_year.service import (
    AcademicYearService,
    AcademicYearNotFoundError,
    AcademicYearOverlapError,
    AcademicYearServiceError,
)
from src.models.academic_year import (
    AcademicYearCreateRequest,
    AcademicYearUpdateRequest,
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
def academic_year_service(mock_db):
    """Create academic year service with mock database."""
    return AcademicYearService(db=mock_db)


@pytest.fixture
def sample_academic_year():
    """Create a sample academic year model with proper field values."""
    year = MagicMock()
    year.id = str(uuid4())
    year.name = "2024-2025"
    year.start_date = date(2024, 9, 1)
    year.end_date = date(2025, 6, 30)
    year.is_current = False
    year.created_at = datetime.now(timezone.utc)
    return year


class TestAcademicYearServiceCreate:
    """Tests for academic year creation."""

    @pytest.mark.asyncio
    async def test_create_academic_year_success(self, academic_year_service, mock_db):
        """Test successful academic year creation."""
        request = AcademicYearCreateRequest(
            name="2024-2025",
            start_date=date(2024, 9, 1),
            end_date=date(2025, 6, 30),
        )

        # Mock no overlapping years
        mock_overlap_result = MagicMock()
        mock_overlap_result.scalar_one_or_none.return_value = None

        # Mock class count for response
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_db.execute.side_effect = [mock_overlap_result, mock_count_result]

        # Mock refresh to set values on the new object
        async def mock_refresh(obj):
            obj.id = str(uuid4())
            obj.name = request.name
            obj.start_date = request.start_date
            obj.end_date = request.end_date
            obj.is_current = False
            obj.created_at = datetime.now(timezone.utc)

        mock_db.refresh.side_effect = mock_refresh

        result = await academic_year_service.create_academic_year(request=request)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert result.name == "2024-2025"

    @pytest.mark.asyncio
    async def test_create_academic_year_overlap(self, academic_year_service, mock_db, sample_academic_year):
        """Test creation fails with overlapping dates."""
        request = AcademicYearCreateRequest(
            name="2024-2025 Overlap",
            start_date=date(2024, 10, 1),
            end_date=date(2025, 5, 30),
        )

        # Mock overlapping year exists
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_academic_year
        mock_db.execute.return_value = mock_result

        with pytest.raises(AcademicYearOverlapError):
            await academic_year_service.create_academic_year(request=request)


class TestAcademicYearServiceGet:
    """Tests for getting academic years."""

    @pytest.mark.asyncio
    async def test_get_academic_year_success(self, academic_year_service, mock_db, sample_academic_year):
        """Test successful academic year retrieval."""
        # Mock year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock class count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_year_result, mock_count_result]

        result = await academic_year_service.get_academic_year(UUID(sample_academic_year.id))

        assert str(result.id) == sample_academic_year.id
        assert result.name == sample_academic_year.name

    @pytest.mark.asyncio
    async def test_get_academic_year_not_found(self, academic_year_service, mock_db):
        """Test academic year not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(AcademicYearNotFoundError):
            await academic_year_service.get_academic_year(uuid4())

    @pytest.mark.asyncio
    async def test_get_current_year_success(self, academic_year_service, mock_db, sample_academic_year):
        """Test getting current academic year."""
        sample_academic_year.is_current = True

        # Mock current year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock class count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_year_result, mock_count_result]

        result = await academic_year_service.get_current_year()

        assert result is not None
        assert result.is_current is True

    @pytest.mark.asyncio
    async def test_get_current_year_none(self, academic_year_service, mock_db):
        """Test no current academic year returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await academic_year_service.get_current_year()

        assert result is None


class TestAcademicYearServiceList:
    """Tests for listing academic years."""

    @pytest.mark.asyncio
    async def test_list_academic_years_empty(self, academic_year_service, mock_db):
        """Test listing academic years when empty."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_years_result = MagicMock()
        mock_years_result.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_count_result, mock_years_result]

        items, total = await academic_year_service.list_academic_years()

        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_academic_years_with_results(self, academic_year_service, mock_db, sample_academic_year):
        """Test listing academic years with results."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_years_result = MagicMock()
        mock_years_result.scalars.return_value.all.return_value = [sample_academic_year]

        mock_db.execute.side_effect = [mock_count_result, mock_years_result]

        items, total = await academic_year_service.list_academic_years()

        assert len(items) == 1
        assert total == 1
        assert items[0].name == sample_academic_year.name


class TestAcademicYearServiceUpdate:
    """Tests for updating academic years."""

    @pytest.mark.asyncio
    async def test_update_academic_year_success(self, academic_year_service, mock_db, sample_academic_year):
        """Test successful academic year update."""
        # Mock year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock class count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_year_result, mock_count_result]

        request = AcademicYearUpdateRequest(name="Updated Name")

        result = await academic_year_service.update_academic_year(
            year_id=UUID(sample_academic_year.id),
            request=request,
        )

        mock_db.commit.assert_called_once()
        assert sample_academic_year.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_academic_year_not_found(self, academic_year_service, mock_db):
        """Test update academic year not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        request = AcademicYearUpdateRequest(name="Updated")

        with pytest.raises(AcademicYearNotFoundError):
            await academic_year_service.update_academic_year(
                year_id=uuid4(),
                request=request,
            )


class TestAcademicYearServiceDelete:
    """Tests for deleting academic years."""

    @pytest.mark.asyncio
    async def test_delete_academic_year_success(self, academic_year_service, mock_db, sample_academic_year):
        """Test successful academic year deletion."""
        # Mock year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock class count - no associated classes
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_db.execute.side_effect = [mock_year_result, mock_count_result]

        await academic_year_service.delete_academic_year(year_id=UUID(sample_academic_year.id))

        mock_db.delete.assert_called_once_with(sample_academic_year)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_academic_year_not_found(self, academic_year_service, mock_db):
        """Test delete academic year not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(AcademicYearNotFoundError):
            await academic_year_service.delete_academic_year(year_id=uuid4())

    @pytest.mark.asyncio
    async def test_delete_academic_year_has_classes(self, academic_year_service, mock_db, sample_academic_year):
        """Test delete fails when academic year has associated classes."""
        # Mock year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock class count - has associated classes
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_year_result, mock_count_result]

        with pytest.raises(AcademicYearServiceError):
            await academic_year_service.delete_academic_year(year_id=UUID(sample_academic_year.id))


class TestAcademicYearServiceSetCurrent:
    """Tests for setting current academic year."""

    @pytest.mark.asyncio
    async def test_set_current_year_success(self, academic_year_service, mock_db, sample_academic_year):
        """Test setting current academic year."""
        # Mock year query
        mock_year_result = MagicMock()
        mock_year_result.scalar_one_or_none.return_value = sample_academic_year

        # Mock update (unset current year)
        mock_update_result = MagicMock()

        # Mock class count query for _to_response
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5

        mock_db.execute.side_effect = [mock_year_result, mock_update_result, mock_count_result]

        result = await academic_year_service.set_current_year(year_id=UUID(sample_academic_year.id))

        mock_db.commit.assert_called()
        assert sample_academic_year.is_current is True

    @pytest.mark.asyncio
    async def test_set_current_year_not_found(self, academic_year_service, mock_db):
        """Test set current year fails when year not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(AcademicYearNotFoundError):
            await academic_year_service.set_current_year(year_id=uuid4())
