# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parent-student relation service for managing family relationships.

This module provides the ParentRelationService class for:
- Creating parent-student relationships
- Managing relationship permissions
- Verifying relationships
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.school import ParentStudentRelation
from src.infrastructure.database.models.tenant.user import User
from src.models.common import StatusEnum
from src.models.parent_relation import (
    CreateParentRelationRequest,
    UpdateParentRelationRequest,
    ParentRelationResponse,
    ParentRelationSummary,
)
from src.models.user import UserSummary

logger = logging.getLogger(__name__)


class ParentRelationServiceError(Exception):
    """Base exception for parent relation service errors."""

    pass


class RelationNotFoundError(ParentRelationServiceError):
    """Raised when relation is not found."""

    pass


class ParentNotFoundError(ParentRelationServiceError):
    """Raised when parent is not found."""

    pass


class StudentNotFoundError(ParentRelationServiceError):
    """Raised when student is not found."""

    pass


class RelationExistsError(ParentRelationServiceError):
    """Raised when relation already exists."""

    pass


class InvalidUserTypeError(ParentRelationServiceError):
    """Raised when user type is invalid for the role."""

    pass


class ParentRelationService:
    """Service for managing parent-student relationships.

    This service handles all parent-student relation operations including
    creating, updating, verifying, and deleting relationships.

    Attributes:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize parent relation service.

        Args:
            db: Async database session for tenant database.
        """
        self.db = db

    async def create_relation(
        self,
        request: CreateParentRelationRequest,
        created_by: str,
    ) -> ParentRelationResponse:
        """Create a parent-student relationship.

        Args:
            request: Relation creation data.
            created_by: ID of user creating the relation.

        Returns:
            Created relation response.

        Raises:
            ParentNotFoundError: If parent not found.
            StudentNotFoundError: If student not found.
            InvalidUserTypeError: If user types are invalid.
            RelationExistsError: If relation already exists.
        """
        # Verify parent exists and is parent type
        parent = await self._get_parent(request.parent_id)

        # Verify student exists and is student type
        student = await self._get_student(request.student_id)

        # Check if relation already exists
        existing = await self._get_relation_by_users(
            str(request.parent_id),
            str(request.student_id),
        )
        if existing:
            raise RelationExistsError(
                "Relationship already exists between this parent and student"
            )

        # Create relation
        relation = ParentStudentRelation(
            parent_id=str(request.parent_id),
            student_id=str(request.student_id),
            relationship_type=request.relationship_type,
            can_view_progress=request.can_view_progress,
            can_view_conversations=request.can_view_conversations,
            can_receive_notifications=request.can_receive_notifications,
            can_chat_with_ai=request.can_chat_with_ai,
            is_primary=request.is_primary,
        )

        self.db.add(relation)
        await self.db.commit()
        await self.db.refresh(relation)

        logger.info(
            "Created parent-student relation: parent=%s, student=%s, by=%s",
            request.parent_id,
            request.student_id,
            created_by,
        )

        return self._to_response(relation, parent, student)

    async def list_relations(
        self,
        parent_id: UUID | None = None,
        student_id: UUID | None = None,
        is_verified: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ParentRelationSummary], int]:
        """List parent-student relationships.

        Args:
            parent_id: Filter by parent.
            student_id: Filter by student.
            is_verified: Filter by verification status.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Tuple of (list of relations, total count).
        """
        query = select(ParentStudentRelation).options(
            selectinload(ParentStudentRelation.parent),
            selectinload(ParentStudentRelation.student),
        )

        # Apply filters
        conditions = []

        if parent_id:
            conditions.append(ParentStudentRelation.parent_id == str(parent_id))

        if student_id:
            conditions.append(ParentStudentRelation.student_id == str(student_id))

        if is_verified is not None:
            if is_verified:
                conditions.append(ParentStudentRelation.verified_at.isnot(None))
            else:
                conditions.append(ParentStudentRelation.verified_at.is_(None))

        if conditions:
            query = query.where(and_(*conditions))

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(ParentStudentRelation.created_at.desc()).limit(limit).offset(offset)

        result = await self.db.execute(query)
        relations = result.scalars().all()

        items = [self._to_summary(r) for r in relations]

        return items, total

    async def get_relation(self, relation_id: UUID) -> ParentRelationResponse:
        """Get relation by ID.

        Args:
            relation_id: Relation identifier.

        Returns:
            Relation details.

        Raises:
            RelationNotFoundError: If relation not found.
        """
        relation = await self._get_by_id(relation_id)
        parent = await self._get_parent(UUID(relation.parent_id))
        student = await self._get_student(UUID(relation.student_id))

        return self._to_response(relation, parent, student)

    async def update_relation(
        self,
        relation_id: UUID,
        request: UpdateParentRelationRequest,
        updated_by: str,
    ) -> ParentRelationResponse:
        """Update a parent-student relationship.

        Args:
            relation_id: Relation identifier.
            request: Update data.
            updated_by: ID of user performing update.

        Returns:
            Updated relation.

        Raises:
            RelationNotFoundError: If relation not found.
        """
        relation = await self._get_by_id(relation_id)

        # Apply updates
        if request.relationship_type is not None:
            relation.relationship_type = request.relationship_type
        if request.can_view_progress is not None:
            relation.can_view_progress = request.can_view_progress
        if request.can_view_conversations is not None:
            relation.can_view_conversations = request.can_view_conversations
        if request.can_receive_notifications is not None:
            relation.can_receive_notifications = request.can_receive_notifications
        if request.can_chat_with_ai is not None:
            relation.can_chat_with_ai = request.can_chat_with_ai
        if request.is_primary is not None:
            relation.is_primary = request.is_primary

        await self.db.commit()
        await self.db.refresh(relation)

        parent = await self._get_parent(UUID(relation.parent_id))
        student = await self._get_student(UUID(relation.student_id))

        logger.info("Updated parent-student relation: %s by %s", relation_id, updated_by)

        return self._to_response(relation, parent, student)

    async def verify_relation(
        self,
        relation_id: UUID,
        verified_by: str,
    ) -> ParentRelationResponse:
        """Verify a parent-student relationship.

        Args:
            relation_id: Relation identifier.
            verified_by: ID of user verifying the relation.

        Returns:
            Updated relation.

        Raises:
            RelationNotFoundError: If relation not found.
        """
        relation = await self._get_by_id(relation_id)

        relation.verified_at = datetime.now(timezone.utc)
        relation.verified_by = verified_by

        await self.db.commit()
        await self.db.refresh(relation)

        parent = await self._get_parent(UUID(relation.parent_id))
        student = await self._get_student(UUID(relation.student_id))

        logger.info("Verified parent-student relation: %s by %s", relation_id, verified_by)

        return self._to_response(relation, parent, student)

    async def delete_relation(
        self,
        relation_id: UUID,
        deleted_by: str,
    ) -> None:
        """Delete a parent-student relationship.

        Args:
            relation_id: Relation identifier.
            deleted_by: ID of user deleting the relation.

        Raises:
            RelationNotFoundError: If relation not found.
        """
        relation = await self._get_by_id(relation_id)

        await self.db.delete(relation)
        await self.db.commit()

        logger.info("Deleted parent-student relation: %s by %s", relation_id, deleted_by)

    async def _get_by_id(self, relation_id: UUID) -> ParentStudentRelation:
        """Get relation by ID.

        Args:
            relation_id: Relation identifier.

        Returns:
            ParentStudentRelation model instance.

        Raises:
            RelationNotFoundError: If not found.
        """
        query = select(ParentStudentRelation).where(
            ParentStudentRelation.id == str(relation_id)
        )
        result = await self.db.execute(query)
        relation = result.scalar_one_or_none()

        if not relation:
            raise RelationNotFoundError(f"Relation {relation_id} not found")

        return relation

    async def _get_relation_by_users(
        self,
        parent_id: str,
        student_id: str,
    ) -> ParentStudentRelation | None:
        """Get relation by parent and student IDs.

        Args:
            parent_id: Parent identifier.
            student_id: Student identifier.

        Returns:
            ParentStudentRelation if found, None otherwise.
        """
        query = select(ParentStudentRelation).where(
            ParentStudentRelation.parent_id == parent_id,
            ParentStudentRelation.student_id == student_id,
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_parent(self, parent_id: UUID) -> User:
        """Get parent user by ID.

        Args:
            parent_id: Parent identifier.

        Returns:
            User model instance.

        Raises:
            ParentNotFoundError: If not found.
            InvalidUserTypeError: If not a parent type.
        """
        query = select(User).where(
            User.id == str(parent_id),
            User.deleted_at.is_(None),
        )
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            raise ParentNotFoundError(f"Parent {parent_id} not found")

        if user.user_type != "parent":
            raise InvalidUserTypeError(f"User {parent_id} is not a parent")

        return user

    async def _get_student(self, student_id: UUID) -> User:
        """Get student user by ID.

        Args:
            student_id: Student identifier.

        Returns:
            User model instance.

        Raises:
            StudentNotFoundError: If not found.
            InvalidUserTypeError: If not a student type.
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
            raise InvalidUserTypeError(f"User {student_id} is not a student")

        return user

    def _to_response(
        self,
        relation: ParentStudentRelation,
        parent: User,
        student: User,
    ) -> ParentRelationResponse:
        """Convert relation to response DTO.

        Args:
            relation: ParentStudentRelation model instance.
            parent: Parent user instance.
            student: Student user instance.

        Returns:
            ParentRelationResponse DTO.
        """
        return ParentRelationResponse(
            id=UUID(relation.id),
            parent=UserSummary(
                id=UUID(parent.id),
                email=parent.email,
                first_name=parent.first_name,
                last_name=parent.last_name,
                display_name=parent.display_name,
                avatar_url=parent.avatar_url,
                user_type=parent.user_type,
                status=StatusEnum(parent.status),
                sso_external_id=parent.sso_external_id,
                created_at=parent.created_at,
                last_login_at=parent.last_login_at,
            ),
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
            relationship_type=relation.relationship_type,
            can_view_progress=relation.can_view_progress,
            can_view_conversations=relation.can_view_conversations,
            can_receive_notifications=relation.can_receive_notifications,
            can_chat_with_ai=relation.can_chat_with_ai,
            is_primary=relation.is_primary,
            is_verified=relation.verified_at is not None,
            verified_at=relation.verified_at,
            verified_by=UUID(relation.verified_by) if relation.verified_by else None,
            created_at=relation.created_at,
        )

    def _to_summary(self, relation: ParentStudentRelation) -> ParentRelationSummary:
        """Convert relation to summary DTO.

        Args:
            relation: ParentStudentRelation model instance.

        Returns:
            ParentRelationSummary DTO.
        """
        parent = relation.parent
        student = relation.student

        parent_name = ""
        if parent:
            parent_name = f"{parent.first_name} {parent.last_name}".strip()

        student_name = ""
        if student:
            student_name = f"{student.first_name} {student.last_name}".strip()

        return ParentRelationSummary(
            id=UUID(relation.id),
            parent_id=UUID(relation.parent_id),
            parent_name=parent_name,
            student_id=UUID(relation.student_id),
            student_name=student_name,
            relationship_type=relation.relationship_type,
            is_primary=relation.is_primary,
            is_verified=relation.verified_at is not None,
        )
