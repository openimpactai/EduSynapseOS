# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""User service for tenant user management.

This module provides the UserService that handles:
- User CRUD operations
- User status management (activate, suspend)
- Role assignments
- User search and filtering

Users are authenticated via LMS integration - the LMS authenticates users
and asserts their identity to EduSynapseOS using API credentials.

Example:
    >>> user_service = UserService(db_session)
    >>> user = await user_service.create_user(request)
    >>> users = await user_service.list_users(user_type="student", limit=20)
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.user import User, UserRole, Role
from src.models.user import (
    UserCreateRequest,
    UserUpdateRequest,
    UserResponse,
    UserSummary,
)
from src.models.common import StatusEnum

logger = logging.getLogger(__name__)


class UserServiceError(Exception):
    """Base exception for user service errors."""

    pass


class UserNotFoundError(UserServiceError):
    """Raised when a user is not found."""

    pass


class UserAlreadyExistsError(UserServiceError):
    """Raised when trying to create a user with existing email."""

    pass


class UserOperationError(UserServiceError):
    """Raised when a user operation fails."""

    pass


class UserService:
    """Service for managing tenant users.

    Handles all user CRUD operations, status management, and role assignments.
    Each user belongs to exactly one tenant (database isolation).

    Users are authenticated via LMS integration - the LMS authenticates users
    and asserts their identity to EduSynapseOS using API credentials.

    Attributes:
        _db: Async database session.

    Example:
        >>> service = UserService(db)
        >>> user = await service.create_user(create_request)
        >>> await service.activate_user(user.id)
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the user service.

        Args:
            db: Async database session.
        """
        self._db = db

    async def create_user(
        self,
        request: UserCreateRequest,
        created_by: str | None = None,
        auto_activate: bool = False,
    ) -> UserResponse:
        """Create a new user.

        Users are created without passwords since authentication is handled
        via LMS integration. The LMS authenticates users and asserts their
        identity to EduSynapseOS using API credentials.

        Args:
            request: User creation request.
            created_by: ID of user who is creating this user.
            auto_activate: If True, set status to active immediately.

        Returns:
            Created user response.

        Raises:
            UserAlreadyExistsError: If email already exists.
        """
        # Check if email already exists
        existing = await self._get_by_email(request.email)
        if existing:
            raise UserAlreadyExistsError(f"User with email {request.email} already exists")

        # Build extra_data - merge request extra_data with service data
        extra_data: dict = request.extra_data.copy() if request.extra_data else {}
        if created_by:
            extra_data["created_by"] = created_by

        # Create user (no password - auth is handled via LMS)
        user = User(
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            user_type=request.user_type,
            status="active" if auto_activate else "pending",
            preferred_language=request.preferred_language.value,
            timezone=request.timezone,
            sso_provider="lms" if request.external_id else None,
            sso_external_id=request.external_id,
            extra_data=extra_data,
        )

        self._db.add(user)
        await self._db.flush()

        # Assign default role based on user_type
        await self._assign_default_role(user)

        await self._db.commit()
        await self._db.refresh(user)

        logger.info("User created: %s (type=%s)", user.id, user.user_type)

        return self._to_response(user)

    async def get_user(self, user_id: str | UUID) -> UserResponse:
        """Get a user by ID.

        Args:
            user_id: User identifier.

        Returns:
            User response.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._get_by_id(str(user_id))
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        return self._to_response(user)

    async def get_user_by_email(self, email: str) -> UserResponse | None:
        """Get a user by email.

        Args:
            email: User email address.

        Returns:
            User response or None if not found.
        """
        user = await self._get_by_email(email)
        if not user:
            return None

        return self._to_response(user)

    async def list_users(
        self,
        user_type: str | None = None,
        status: str | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[UserSummary], int]:
        """List users with optional filtering.

        Args:
            user_type: Filter by user type.
            status: Filter by status.
            search: Search by name or email.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Tuple of (user summaries, total count).
        """
        # Build base query (deleted_at.is_(None) for non-deleted users)
        stmt = select(User).where(User.deleted_at.is_(None))

        # Apply filters
        if user_type:
            stmt = stmt.where(User.user_type == user_type)

        if status:
            stmt = stmt.where(User.status == status)

        if search:
            search_pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    User.email.ilike(search_pattern),
                    User.first_name.ilike(search_pattern),
                    User.last_name.ilike(search_pattern),
                )
            )

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self._db.execute(count_stmt)
        total = count_result.scalar() or 0

        # Apply pagination and ordering
        stmt = stmt.order_by(User.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._db.execute(stmt)
        users = result.scalars().all()

        return [self._to_summary(u) for u in users], total

    async def update_user(
        self,
        user_id: str | UUID,
        request: UserUpdateRequest,
    ) -> UserResponse:
        """Update a user.

        Args:
            user_id: User identifier.
            request: Update request with fields to change.

        Returns:
            Updated user response.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._get_by_id(str(user_id))
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        # Update fields if provided
        if request.first_name is not None:
            user.first_name = request.first_name
        if request.last_name is not None:
            user.last_name = request.last_name
        if request.display_name is not None:
            user.display_name = request.display_name
        if request.avatar_url is not None:
            user.avatar_url = request.avatar_url
        if request.preferred_language is not None:
            user.preferred_language = request.preferred_language.value
        if request.timezone is not None:
            user.timezone = request.timezone
        if request.extra_data is not None:
            # Merge with existing extra_data
            current_extra = user.extra_data or {}
            current_extra.update(request.extra_data)
            user.extra_data = current_extra

        await self._db.commit()
        await self._db.refresh(user)

        logger.info("User updated: %s", user.id)

        return self._to_response(user)

    async def delete_user(self, user_id: str | UUID) -> None:
        """Soft delete a user.

        Args:
            user_id: User identifier.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._get_by_id(str(user_id))
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        user.deleted_at = datetime.now(timezone.utc)
        user.status = "archived"

        await self._db.commit()

        logger.info("User deleted: %s", user.id)

    async def activate_user(self, user_id: str | UUID) -> UserResponse:
        """Activate a user account.

        Args:
            user_id: User identifier.

        Returns:
            Updated user response.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._get_by_id(str(user_id))
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        user.status = "active"
        await self._db.commit()
        await self._db.refresh(user)

        logger.info("User activated: %s", user.id)

        return self._to_response(user)

    async def suspend_user(
        self,
        user_id: str | UUID,
        reason: str | None = None,
    ) -> UserResponse:
        """Suspend a user account.

        Args:
            user_id: User identifier.
            reason: Optional suspension reason.

        Returns:
            Updated user response.

        Raises:
            UserNotFoundError: If user not found.
        """
        user = await self._get_by_id(str(user_id))
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        user.status = "suspended"
        if reason:
            extra_data = user.extra_data or {}
            extra_data["suspension_reason"] = reason
            extra_data["suspended_at"] = datetime.now(timezone.utc).isoformat()
            user.extra_data = extra_data

        await self._db.commit()
        await self._db.refresh(user)

        logger.info("User suspended: %s (reason=%s)", user.id, reason)

        return self._to_response(user)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        stmt = select(User).where(
            User.id == user_id,
            User.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        stmt = select(User).where(
            User.email == email,
            User.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _assign_default_role(self, user: User) -> None:
        """Assign default role based on user type."""
        role_code = user.user_type  # student, teacher, parent, school_admin, tenant_admin

        # Find the role
        stmt = select(Role).where(Role.code == role_code)
        result = await self._db.execute(stmt)
        role = result.scalar_one_or_none()

        if role:
            user_role = UserRole(
                user_id=user.id,
                role_id=role.id,
            )
            self._db.add(user_role)

    def _to_response(self, user: User) -> UserResponse:
        """Convert User model to UserResponse."""
        return UserResponse(
            id=UUID(user.id),
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            user_type=user.user_type,
            status=StatusEnum(user.status),
            email_verified=user.email_verified,
            mfa_enabled=user.mfa_enabled,
            preferred_language=user.preferred_language,
            timezone=user.timezone,
            last_login_at=user.last_login_at,
            last_activity_at=user.last_activity_at,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    def _to_summary(self, user: User) -> UserSummary:
        """Convert User model to UserSummary."""
        return UserSummary(
            id=UUID(user.id),
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            user_type=user.user_type,
            status=StatusEnum(user.status),
            sso_external_id=user.sso_external_id,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
