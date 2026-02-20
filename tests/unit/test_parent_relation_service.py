# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for Parent Relation service."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.domains.parent_relation.service import (
    ParentRelationService,
    RelationNotFoundError,
    ParentNotFoundError,
    StudentNotFoundError,
    RelationExistsError,
    InvalidUserTypeError,
)
from src.models.parent_relation import (
    CreateParentRelationRequest,
    UpdateParentRelationRequest,
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
def parent_relation_service(mock_db):
    """Create parent relation service with mock database."""
    return ParentRelationService(db=mock_db)


@pytest.fixture
def sample_parent():
    """Create a sample parent model."""
    parent = MagicMock()
    parent.id = str(uuid4())
    parent.email = "parent@example.com"
    parent.first_name = "John"
    parent.last_name = "Doe"
    parent.user_type = "parent"
    parent.is_active = True
    parent.external_id = "ext_parent_123"
    parent.deleted_at = None
    return parent


@pytest.fixture
def sample_student():
    """Create a sample student model."""
    student = MagicMock()
    student.id = str(uuid4())
    student.email = "student@school.com"
    student.first_name = "Jane"
    student.last_name = "Doe"
    student.user_type = "student"
    student.is_active = True
    student.external_id = "ext_student_123"
    student.deleted_at = None
    return student


@pytest.fixture
def sample_relation(sample_parent, sample_student):
    """Create a sample relation model."""
    relation = MagicMock()
    relation.id = str(uuid4())
    relation.parent_id = sample_parent.id
    relation.student_id = sample_student.id
    relation.relationship_type = "parent"
    relation.can_view_progress = True
    relation.can_view_conversations = False
    relation.can_receive_notifications = True
    relation.can_chat_with_ai = False
    relation.is_primary = True
    relation.verified_at = None
    relation.verified_by = None
    relation.created_at = datetime.now(timezone.utc)
    relation.parent = sample_parent
    relation.student = sample_student
    return relation


class TestParentRelationServiceCreate:
    """Tests for parent relation creation."""

    @pytest.mark.asyncio
    async def test_create_relation_success(self, parent_relation_service, mock_db, sample_parent, sample_student):
        """Test successful relation creation."""
        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        # Mock parent exists
        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        # Mock student exists
        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        # Mock no existing relation
        mock_relation_result = MagicMock()
        mock_relation_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_parent_result, mock_student_result, mock_relation_result]

        result = await parent_relation_service.create_relation(
            request=request,
            created_by=str(uuid4()),
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relation_parent_not_found(self, parent_relation_service, mock_db):
        """Test creation fails when parent not found."""
        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(ParentNotFoundError):
            await parent_relation_service.create_relation(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_relation_student_not_found(self, parent_relation_service, mock_db, sample_parent):
        """Test creation fails when student not found."""
        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [mock_parent_result, mock_student_result]

        with pytest.raises(StudentNotFoundError):
            await parent_relation_service.create_relation(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_relation_invalid_parent_type(self, parent_relation_service, mock_db, sample_parent):
        """Test creation fails when parent user type is invalid."""
        sample_parent.user_type = "student"

        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_parent
        mock_db.execute.return_value = mock_result

        with pytest.raises(InvalidUserTypeError):
            await parent_relation_service.create_relation(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_relation_invalid_student_type(self, parent_relation_service, mock_db, sample_parent, sample_student):
        """Test creation fails when student user type is invalid."""
        sample_student.user_type = "teacher"

        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_parent_result, mock_student_result]

        with pytest.raises(InvalidUserTypeError):
            await parent_relation_service.create_relation(
                request=request,
                created_by=str(uuid4()),
            )

    @pytest.mark.asyncio
    async def test_create_relation_already_exists(self, parent_relation_service, mock_db, sample_parent, sample_student, sample_relation):
        """Test creation fails when relation already exists."""
        request = CreateParentRelationRequest(
            parent_id=uuid4(),
            student_id=uuid4(),
            relationship_type="parent",
        )

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_relation_result = MagicMock()
        mock_relation_result.scalar_one_or_none.return_value = sample_relation

        mock_db.execute.side_effect = [mock_parent_result, mock_student_result, mock_relation_result]

        with pytest.raises(RelationExistsError):
            await parent_relation_service.create_relation(
                request=request,
                created_by=str(uuid4()),
            )


class TestParentRelationServiceList:
    """Tests for listing relations."""

    @pytest.mark.asyncio
    async def test_list_relations_empty(self, parent_relation_service, mock_db):
        """Test listing relations when empty."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_relations_result = MagicMock()
        mock_relations_result.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [mock_count_result, mock_relations_result]

        items, total = await parent_relation_service.list_relations()

        assert items == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_relations_with_results(self, parent_relation_service, mock_db, sample_relation):
        """Test listing relations with results."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_relations_result = MagicMock()
        mock_relations_result.scalars.return_value.all.return_value = [sample_relation]

        mock_db.execute.side_effect = [mock_count_result, mock_relations_result]

        items, total = await parent_relation_service.list_relations()

        assert len(items) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_relations_with_parent_filter(self, parent_relation_service, mock_db, sample_relation):
        """Test listing relations with parent filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_relations_result = MagicMock()
        mock_relations_result.scalars.return_value.all.return_value = [sample_relation]

        mock_db.execute.side_effect = [mock_count_result, mock_relations_result]

        items, total = await parent_relation_service.list_relations(parent_id=uuid4())

        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_list_relations_with_student_filter(self, parent_relation_service, mock_db, sample_relation):
        """Test listing relations with student filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_relations_result = MagicMock()
        mock_relations_result.scalars.return_value.all.return_value = [sample_relation]

        mock_db.execute.side_effect = [mock_count_result, mock_relations_result]

        items, total = await parent_relation_service.list_relations(student_id=uuid4())

        assert len(items) == 1


class TestParentRelationServiceGet:
    """Tests for getting relation."""

    @pytest.mark.asyncio
    async def test_get_relation_success(self, parent_relation_service, mock_db, sample_relation, sample_parent, sample_student):
        """Test successful relation retrieval."""
        mock_relation_result = MagicMock()
        mock_relation_result.scalar_one_or_none.return_value = sample_relation

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_relation_result, mock_parent_result, mock_student_result]

        result = await parent_relation_service.get_relation(uuid4())

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_relation_not_found(self, parent_relation_service, mock_db):
        """Test relation not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(RelationNotFoundError):
            await parent_relation_service.get_relation(uuid4())


class TestParentRelationServiceUpdate:
    """Tests for updating relations."""

    @pytest.mark.asyncio
    async def test_update_relation_success(self, parent_relation_service, mock_db, sample_relation, sample_parent, sample_student):
        """Test successful relation update."""
        mock_relation_result = MagicMock()
        mock_relation_result.scalar_one_or_none.return_value = sample_relation

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_relation_result, mock_parent_result, mock_student_result]

        request = UpdateParentRelationRequest(can_view_progress=False)

        result = await parent_relation_service.update_relation(
            relation_id=uuid4(),
            request=request,
            updated_by=str(uuid4()),
        )

        mock_db.commit.assert_called_once()
        assert sample_relation.can_view_progress is False

    @pytest.mark.asyncio
    async def test_update_relation_not_found(self, parent_relation_service, mock_db):
        """Test update relation not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        request = UpdateParentRelationRequest(can_view_progress=False)

        with pytest.raises(RelationNotFoundError):
            await parent_relation_service.update_relation(
                relation_id=uuid4(),
                request=request,
                updated_by=str(uuid4()),
            )


class TestParentRelationServiceVerify:
    """Tests for verifying relations."""

    @pytest.mark.asyncio
    async def test_verify_relation_success(self, parent_relation_service, mock_db, sample_relation, sample_parent, sample_student):
        """Test successful relation verification."""
        mock_relation_result = MagicMock()
        mock_relation_result.scalar_one_or_none.return_value = sample_relation

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = sample_parent

        mock_student_result = MagicMock()
        mock_student_result.scalar_one_or_none.return_value = sample_student

        mock_db.execute.side_effect = [mock_relation_result, mock_parent_result, mock_student_result]

        verified_by = str(uuid4())
        result = await parent_relation_service.verify_relation(
            relation_id=uuid4(),
            verified_by=verified_by,
        )

        mock_db.commit.assert_called_once()
        assert sample_relation.verified_at is not None
        assert sample_relation.verified_by == verified_by

    @pytest.mark.asyncio
    async def test_verify_relation_not_found(self, parent_relation_service, mock_db):
        """Test verify relation not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(RelationNotFoundError):
            await parent_relation_service.verify_relation(
                relation_id=uuid4(),
                verified_by=str(uuid4()),
            )


class TestParentRelationServiceDelete:
    """Tests for deleting relations."""

    @pytest.mark.asyncio
    async def test_delete_relation_success(self, parent_relation_service, mock_db, sample_relation):
        """Test successful relation deletion."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_relation
        mock_db.execute.return_value = mock_result

        await parent_relation_service.delete_relation(
            relation_id=uuid4(),
            deleted_by=str(uuid4()),
        )

        mock_db.delete.assert_called_once_with(sample_relation)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_relation_not_found(self, parent_relation_service, mock_db):
        """Test delete relation not found error."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(RelationNotFoundError):
            await parent_relation_service.delete_relation(
                relation_id=uuid4(),
                deleted_by=str(uuid4()),
            )
