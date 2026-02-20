# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for School service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from uuid import uuid4, UUID

import pytest

from src.domains.school.service import (
    SchoolService,
    SchoolNotFoundError,
    SchoolCodeExistsError,
    SchoolAdminError,
)
from src.models.school import (
    SchoolCreateRequest,
    SchoolUpdateRequest,
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
def school_service(mock_db):
    """Create school service with mock database."""
    return SchoolService(db=mock_db)


def create_mock_result(value):
    """Create a mock result with scalar_one_or_none."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    result.scalar.return_value = value if isinstance(value, int) else 0
    return result


def create_mock_stats_results():
    """Create mock results for school stats (3 queries)."""
    class_count = MagicMock()
    class_count.scalar.return_value = 5
    student_count = MagicMock()
    student_count.scalar.return_value = 50
    teacher_count = MagicMock()
    teacher_count.scalar.return_value = 10
    return [class_count, student_count, teacher_count]


@pytest.fixture
def sample_school():
    """Create a sample school model with proper field values."""
    school = MagicMock()
    school.id = str(uuid4())
    school.code = "test001"
    school.name = "Test School"
    school.school_type = "primary"
    school.address_line1 = "123 Test St"
    school.address_line2 = None
    school.city = "Istanbul"
    school.state_province = "Istanbul"
    school.postal_code = "34000"
    school.country_code = "TR"
    school.phone = "+1234567890"
    school.email = "test@school.com"
    school.website = "https://test.school.com"
    school.timezone = "Europe/Istanbul"
    school.settings = {}
    school.is_active = True
    school.deleted_at = None
    school.created_at = datetime.now(timezone.utc)
    school.updated_at = datetime.now(timezone.utc)
    return school


@pytest.fixture
def sample_user():
    """Create a sample user model with proper field values."""
    user = MagicMock()
    user.id = str(uuid4())
    user.email = "admin@school.com"
    user.first_name = "John"
    user.last_name = "Doe"
    user.display_name = "John Doe"
    user.avatar_url = None
    user.user_type = "school_admin"
    user.status = "active"
    user.deleted_at = None
    return user


@pytest.fixture
def sample_role():
    """Create a sample role model."""
    role = MagicMock()
    role.id = str(uuid4())
    role.code = "school_admin"
    role.name = "School Admin"
    return role


class TestSchoolServiceCreate:
    """Tests for school creation."""

    @pytest.mark.asyncio
    async def test_create_school_success(self, school_service, mock_db, sample_school):
        """Test successful school creation."""
        request = SchoolCreateRequest(
            code="NEW001",
            name="New School",
            school_type="primary",
        )

        # Mock: no existing school with same code
        mock_code_check = create_mock_result(None)

        # Mock refresh to set values on the new school object
        async def mock_refresh(obj):
            obj.id = str(uuid4())
            obj.code = request.code.lower()  # Service may lowercase
            obj.name = request.name
            obj.school_type = request.school_type
            obj.is_active = True
            obj.created_at = datetime.now(timezone.utc)
            obj.updated_at = datetime.now(timezone.utc)
            obj.deleted_at = None
            obj.address_line1 = None
            obj.address_line2 = None
            obj.city = None
            obj.state_province = None
            obj.postal_code = None
            obj.country_code = "TR"  # Required field
            obj.phone = None
            obj.email = None
            obj.website = None
            obj.timezone = "Europe/Istanbul"  # Required field

        mock_db.refresh.side_effect = mock_refresh

        # Setup execute side_effect: code check + 3 stats queries
        mock_db.execute.side_effect = [mock_code_check] + create_mock_stats_results()

        result = await school_service.create_school(request=request)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert result.name == "New School"

    @pytest.mark.asyncio
    async def test_create_school_code_exists(self, school_service, mock_db, sample_school):
        """Test school creation fails when code exists."""
        request = SchoolCreateRequest(
            code="TEST001",
            name="Another School",
            school_type="primary",
        )

        # Mock existing school with same code
        mock_result = create_mock_result(sample_school)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolCodeExistsError):
            await school_service.create_school(request=request)


class TestSchoolServiceGet:
    """Tests for getting schools."""

    @pytest.mark.asyncio
    async def test_get_school_success(self, school_service, mock_db, sample_school):
        """Test successful school retrieval."""
        # Mock: school query + 3 stats queries
        mock_school = create_mock_result(sample_school)
        mock_db.execute.side_effect = [mock_school] + create_mock_stats_results()

        result = await school_service.get_school(sample_school.id)

        assert str(result.id) == sample_school.id
        assert result.code == sample_school.code

    @pytest.mark.asyncio
    async def test_get_school_not_found(self, school_service, mock_db):
        """Test school not found error."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.get_school(uuid4())


class TestSchoolServiceList:
    """Tests for listing schools."""

    @pytest.mark.asyncio
    async def test_list_schools_empty(self, school_service, mock_db):
        """Test listing schools when empty."""
        # Mock count query
        mock_count = MagicMock()
        mock_count.scalar.return_value = 0

        # Mock schools query
        mock_schools = MagicMock()
        mock_schools.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_count, mock_schools]

        items, total = await school_service.list_schools()

        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_schools_with_results(self, school_service, mock_db, sample_school):
        """Test listing schools with results."""
        # Mock count query
        mock_count = MagicMock()
        mock_count.scalar.return_value = 1

        # Mock schools query
        mock_schools = MagicMock()
        mock_schools.scalars.return_value.all.return_value = [sample_school]

        # For each school: 3 stats queries
        mock_db.execute.side_effect = [mock_count, mock_schools] + create_mock_stats_results()

        items, total = await school_service.list_schools()

        assert len(items) == 1
        assert total == 1
        assert items[0].code == sample_school.code


class TestSchoolServiceUpdate:
    """Tests for updating schools."""

    @pytest.mark.asyncio
    async def test_update_school_success(self, school_service, mock_db, sample_school):
        """Test successful school update."""
        # Mock: school query + refresh + 3 stats queries
        mock_school = create_mock_result(sample_school)
        mock_db.execute.side_effect = [mock_school] + create_mock_stats_results()

        request = SchoolUpdateRequest(name="Updated School Name")

        result = await school_service.update_school(
            school_id=sample_school.id,
            request=request,
        )

        mock_db.commit.assert_called_once()
        assert sample_school.name == "Updated School Name"

    @pytest.mark.asyncio
    async def test_update_school_not_found(self, school_service, mock_db):
        """Test update school not found error."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        request = SchoolUpdateRequest(name="Updated School")

        with pytest.raises(SchoolNotFoundError):
            await school_service.update_school(
                school_id=uuid4(),
                request=request,
            )


class TestSchoolServiceDeactivate:
    """Tests for deactivating schools."""

    @pytest.mark.asyncio
    async def test_deactivate_school_success(self, school_service, mock_db, sample_school):
        """Test successful school deactivation."""
        mock_result = create_mock_result(sample_school)
        mock_db.execute.return_value = mock_result

        await school_service.deactivate_school(school_id=sample_school.id)

        mock_db.commit.assert_called_once()
        assert sample_school.is_active is False

    @pytest.mark.asyncio
    async def test_deactivate_school_not_found(self, school_service, mock_db):
        """Test deactivate school not found error."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.deactivate_school(school_id=uuid4())


class TestSchoolServiceActivate:
    """Tests for activating schools."""

    @pytest.mark.asyncio
    async def test_activate_school_success(self, school_service, mock_db, sample_school):
        """Test successful school activation."""
        sample_school.is_active = False
        sample_school.deleted_at = datetime.now(timezone.utc)

        # Mock: school query + refresh + 3 stats queries
        mock_school = create_mock_result(sample_school)
        mock_db.execute.side_effect = [mock_school] + create_mock_stats_results()

        result = await school_service.activate_school(school_id=sample_school.id)

        mock_db.commit.assert_called_once()
        assert sample_school.is_active is True
        assert sample_school.deleted_at is None

    @pytest.mark.asyncio
    async def test_activate_school_not_found(self, school_service, mock_db):
        """Test activate school not found error."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.activate_school(school_id=uuid4())


class TestSchoolServiceAdmins:
    """Tests for school admin management."""

    @pytest.mark.asyncio
    async def test_assign_admin_success(self, school_service, mock_db, sample_school, sample_user, sample_role):
        """Test successful admin assignment."""
        # Mock: school query, user query, role query, existing assignment check, role query again
        mock_school = create_mock_result(sample_school)
        mock_user = create_mock_result(sample_user)
        mock_role = create_mock_result(sample_role)
        mock_no_existing = create_mock_result(None)

        mock_db.execute.side_effect = [
            mock_school,  # _get_by_id
            mock_user,    # _get_user_by_id
            mock_role,    # _get_role_by_code (for assignment check)
            mock_no_existing,  # existing assignment check
            mock_role,    # _get_role_by_code (for creating assignment)
        ]

        await school_service.assign_admin(
            school_id=sample_school.id,
            user_id=sample_user.id,
            assigned_by=str(uuid4()),
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_assign_admin_school_not_found(self, school_service, mock_db):
        """Test assign admin fails when school not found."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.assign_admin(
                school_id=uuid4(),
                user_id=uuid4(),
                assigned_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_assign_admin_user_not_found(self, school_service, mock_db, sample_school):
        """Test assign admin fails when user not found."""
        mock_school = create_mock_result(sample_school)
        mock_no_user = create_mock_result(None)

        mock_db.execute.side_effect = [mock_school, mock_no_user]

        with pytest.raises(SchoolAdminError):
            await school_service.assign_admin(
                school_id=sample_school.id,
                user_id=uuid4(),
                assigned_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_list_admins_success(self, school_service, mock_db, sample_school, sample_user, sample_role):
        """Test listing school admins."""
        # Create mock assignment with user relationship
        mock_assignment = MagicMock()
        mock_assignment.user = sample_user
        mock_assignment.created_at = datetime.now(timezone.utc)
        mock_assignment.granted_by = str(uuid4())

        mock_school = create_mock_result(sample_school)
        mock_role_result = create_mock_result(sample_role)
        mock_assignments = MagicMock()
        mock_assignments.scalars.return_value.all.return_value = [mock_assignment]

        mock_db.execute.side_effect = [mock_school, mock_role_result, mock_assignments]

        result = await school_service.list_admins(sample_school.id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_admins_school_not_found(self, school_service, mock_db):
        """Test list admins fails when school not found."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.list_admins(uuid4())

    @pytest.mark.asyncio
    async def test_remove_admin_success(self, school_service, mock_db, sample_school, sample_role, sample_user):
        """Test removing school admin."""
        mock_assignment = MagicMock()
        mock_assignment.id = str(uuid4())

        mock_school = create_mock_result(sample_school)
        mock_role = create_mock_result(sample_role)
        mock_existing = create_mock_result(mock_assignment)
        mock_count = MagicMock()
        mock_count.scalar.return_value = 0
        mock_user = create_mock_result(sample_user)

        mock_db.execute.side_effect = [
            mock_school,   # _get_by_id
            mock_role,     # _get_role_by_code for assignment check
            mock_existing, # get assignment
            mock_role,     # _get_role_by_code for count
            mock_count,    # count other assignments
            mock_user,     # _get_user_by_id to update user_type
        ]

        await school_service.remove_admin(
            school_id=sample_school.id,
            user_id=sample_user.id,
        )

        mock_db.delete.assert_called_once_with(mock_assignment)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_admin_school_not_found(self, school_service, mock_db):
        """Test remove admin fails when school not found."""
        mock_result = create_mock_result(None)
        mock_db.execute.return_value = mock_result

        with pytest.raises(SchoolNotFoundError):
            await school_service.remove_admin(
                school_id=uuid4(),
                user_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_remove_admin_not_assigned(self, school_service, mock_db, sample_school, sample_role):
        """Test remove admin fails when user is not an admin."""
        mock_school = create_mock_result(sample_school)
        mock_role = create_mock_result(sample_role)
        mock_no_assignment = create_mock_result(None)

        mock_db.execute.side_effect = [mock_school, mock_role, mock_no_assignment]

        with pytest.raises(SchoolAdminError):
            await school_service.remove_admin(
                school_id=sample_school.id,
                user_id=uuid4(),
            )
