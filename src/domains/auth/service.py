# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Authentication service for session management.

This module provides the main AuthService that orchestrates:
- Token refresh with rotation
- Session management
- User logout

Users are authenticated via LMS integration - the LMS authenticates users
and asserts their identity to EduSynapseOS using API credentials.
Token exchange happens via the /auth/exchange endpoint.

Example:
    >>> auth_service = AuthService(db_session, jwt_manager)
    >>> tokens = await auth_service.refresh_tokens(refresh_token, tenant_id, tenant_code)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.domains.auth.jwt import JWTManager, TokenPair, TokenExpiredError, InvalidTokenError
from src.infrastructure.database.models.tenant.user import (
    User,
    UserRole,
    Role,
    RolePermission,
)
from src.infrastructure.database.models.tenant.session import (
    UserSession,
    RefreshToken,
)
from src.models.auth import SessionInfo

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Base exception for authentication errors."""

    pass


class AccountInactiveError(AuthenticationError):
    """Raised when account is not active."""

    pass


class SessionNotFoundError(AuthenticationError):
    """Raised when session is not found or invalid."""

    pass


class TokenRefreshError(AuthenticationError):
    """Raised when token refresh fails."""

    pass


class DeviceInfo(NamedTuple):
    """Device information for session tracking."""

    device_type: str | None = None
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None


class AuthService:
    """Authentication service for session management.

    Handles session-related operations including:
    - Token refresh with rotation
    - Session management
    - User logout

    Users are authenticated via the /auth/exchange endpoint which
    validates API credentials and creates sessions.

    Attributes:
        _db: Database session for queries.
        _jwt_manager: JWT token manager.

    Example:
        >>> auth_service = AuthService(db, jwt_manager)
        >>> tokens = await auth_service.refresh_tokens(
        ...     refresh_token=token,
        ...     tenant_id=tenant.id,
        ...     tenant_code=tenant.code,
        ... )
    """

    def __init__(
        self,
        db: AsyncSession,
        jwt_manager: JWTManager,
    ) -> None:
        """Initialize the authentication service.

        Args:
            db: Async database session.
            jwt_manager: JWT token manager.
        """
        self._db = db
        self._jwt_manager = jwt_manager

    async def refresh_tokens(
        self,
        refresh_token: str,
        tenant_id: str | UUID,
        tenant_code: str,
    ) -> TokenPair:
        """Refresh access token using a refresh token.

        Implements refresh token rotation - the old refresh token is marked
        as used and a new one is created.

        Args:
            refresh_token: Current refresh token.
            tenant_id: Tenant identifier.
            tenant_code: Tenant code.

        Returns:
            New TokenPair with fresh access and refresh tokens.

        Raises:
            TokenRefreshError: If refresh token is invalid or expired.
        """
        try:
            # Decode and validate refresh token
            payload = self._jwt_manager.decode_token(refresh_token, expected_type="refresh")
        except (TokenExpiredError, InvalidTokenError) as e:
            raise TokenRefreshError(f"Invalid refresh token: {str(e)}")

        # Find the stored refresh token
        token_hash = self._jwt_manager.hash_token(refresh_token)
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.user_id == payload.sub,
        )
        result = await self._db.execute(stmt)
        stored_token = result.scalar_one_or_none()

        if not stored_token:
            raise TokenRefreshError("Refresh token not found")

        if not stored_token.is_valid:
            # Token was already used or revoked - possible token theft
            # Revoke all tokens in this family
            await self._revoke_token_family(stored_token.family_id)
            raise TokenRefreshError("Refresh token has been used or revoked")

        # Get user
        stmt = select(User).where(User.id == payload.sub)
        result = await self._db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise TokenRefreshError("User not found or inactive")

        # Mark old token as used
        stored_token.mark_used()

        # Get user roles and permissions
        roles, permissions, school_ids = await self._get_user_roles_permissions(user.id)

        # Create new tokens
        new_tokens = self._jwt_manager.create_token_pair(
            user_id=user.id,
            tenant_id=tenant_id,
            tenant_code=tenant_code,
            user_type=user.user_type,
            roles=roles,
            permissions=permissions,
            school_ids=school_ids,
            preferred_language=user.preferred_language,
        )

        # Store new refresh token in same family
        await self._store_refresh_token(
            user_id=user.id,
            session_id=stored_token.session_id,
            refresh_token=new_tokens.refresh_token,
            expires_days=self._jwt_manager._settings.refresh_token_expire_days,
            family_id=stored_token.family_id,
            generation=stored_token.generation + 1,
        )

        await self._db.commit()

        logger.info("Tokens refreshed for user: %s", user.id)

        return new_tokens

    async def logout(
        self,
        session_id: str | UUID,
        revoke_all: bool = False,
        user_id: str | UUID | None = None,
    ) -> None:
        """Logout a user session.

        Args:
            session_id: Session to logout.
            revoke_all: If True, revoke all sessions for the user.
            user_id: User ID (required if revoke_all is True).
        """
        if revoke_all and user_id:
            # Revoke all sessions
            stmt = select(UserSession).where(
                UserSession.user_id == str(user_id),
                UserSession.is_active == True,  # noqa: E712
            )
            result = await self._db.execute(stmt)
            sessions = result.scalars().all()

            for session in sessions:
                session.revoke("logout_all")

            # Revoke all refresh tokens
            stmt = update(RefreshToken).where(
                RefreshToken.user_id == str(user_id),
                RefreshToken.revoked_at == None,  # noqa: E711
            ).values(revoked_at=datetime.now(timezone.utc))
            await self._db.execute(stmt)

            logger.info("All sessions revoked for user: %s", user_id)
        else:
            # Revoke single session
            stmt = select(UserSession).where(UserSession.id == str(session_id))
            result = await self._db.execute(stmt)
            session = result.scalar_one_or_none()

            if session:
                session.revoke("logout")

                # Revoke associated refresh tokens
                stmt = update(RefreshToken).where(
                    RefreshToken.session_id == str(session_id),
                    RefreshToken.revoked_at == None,  # noqa: E711
                ).values(revoked_at=datetime.now(timezone.utc))
                await self._db.execute(stmt)

                logger.info("Session revoked: %s", session_id)

        await self._db.commit()

    async def get_active_sessions(self, user_id: str | UUID) -> list[SessionInfo]:
        """Get all active sessions for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of SessionInfo objects.
        """
        stmt = select(UserSession).where(
            UserSession.user_id == str(user_id),
            UserSession.is_active == True,  # noqa: E712
            UserSession.expires_at > datetime.now(timezone.utc),
        ).order_by(UserSession.created_at.desc())

        result = await self._db.execute(stmt)
        sessions = result.scalars().all()

        return [
            SessionInfo(
                id=UUID(session.id),
                device_type=session.device_type,
                device_name=session.device_name,
                ip_address=session.ip_address,
                city=session.city,
                country_code=session.country_code,
                created_at=session.created_at,
                last_activity_at=session.last_activity_at,
                is_current=False,  # Caller should set this based on current session
            )
            for session in sessions
        ]

    async def revoke_session(
        self,
        session_id: str | UUID,
        user_id: str | UUID,
    ) -> bool:
        """Revoke a specific session.

        Args:
            session_id: Session to revoke.
            user_id: User who owns the session (for verification).

        Returns:
            True if session was revoked, False if not found.
        """
        stmt = select(UserSession).where(
            UserSession.id == str(session_id),
            UserSession.user_id == str(user_id),
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return False

        session.revoke("user_revoked")

        # Revoke associated refresh tokens
        stmt = update(RefreshToken).where(
            RefreshToken.session_id == str(session_id),
            RefreshToken.revoked_at == None,  # noqa: E711
        ).values(revoked_at=datetime.now(timezone.utc))
        await self._db.execute(stmt)

        await self._db.commit()

        return True

    async def validate_session(self, access_token: str) -> UserSession | None:
        """Validate an access token and return the session.

        Args:
            access_token: JWT access token.

        Returns:
            UserSession if valid, None otherwise.
        """
        token_hash = self._jwt_manager.hash_token(access_token)

        stmt = select(UserSession).where(
            UserSession.access_token_hash == token_hash,
            UserSession.is_active == True,  # noqa: E712
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if session and session.is_valid:
            session.update_activity()
            await self._db.commit()
            return session

        return None

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _get_user_roles_permissions(
        self,
        user_id: str,
    ) -> tuple[list[str], list[str], list[str]]:
        """Get user's roles, permissions, and accessible school IDs.

        Uses eager loading to avoid N+1 queries and lazy loading issues
        in async context.

        Returns:
            Tuple of (role_codes, permission_codes, school_ids).
        """
        # Single query with eager loading for all relationships
        stmt = (
            select(UserRole)
            .options(
                selectinload(UserRole.role)
                .selectinload(Role.role_permissions)
                .selectinload(RolePermission.permission)
            )
            .where(UserRole.user_id == user_id)
        )
        result = await self._db.execute(stmt)
        user_roles = result.scalars().all()

        roles: list[str] = []
        permissions: list[str] = []
        school_ids: list[str] = []

        for user_role in user_roles:
            if not user_role.is_active:
                continue

            role = user_role.role
            if role:
                roles.append(role.code)

                for rp in role.role_permissions:
                    if rp.permission.code not in permissions:
                        permissions.append(rp.permission.code)

            if user_role.school_id and user_role.school_id not in school_ids:
                school_ids.append(user_role.school_id)

        return roles, permissions, school_ids

    async def _create_session(
        self,
        user: User,
        access_token: str,
        device_info: DeviceInfo,
        expires_minutes: int,
    ) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user.id,
            access_token_hash=self._jwt_manager.hash_token(access_token),
            device_type=device_info.device_type,
            device_name=device_info.device_name,
            ip_address=device_info.ip_address,
            user_agent=device_info.user_agent,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
        )

        self._db.add(session)
        await self._db.flush()

        return session

    async def _store_refresh_token(
        self,
        user_id: str,
        session_id: str | None,
        refresh_token: str,
        expires_days: int,
        family_id: str | None = None,
        generation: int = 1,
    ) -> RefreshToken:
        """Store a refresh token in the database."""
        token = RefreshToken(
            user_id=user_id,
            session_id=session_id,
            token_hash=self._jwt_manager.hash_token(refresh_token),
            family_id=family_id or str(uuid4()),
            generation=generation,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days),
        )

        self._db.add(token)
        await self._db.flush()

        return token

    async def _revoke_token_family(self, family_id: str) -> None:
        """Revoke all tokens in a token family (for security)."""
        stmt = update(RefreshToken).where(
            RefreshToken.family_id == family_id,
            RefreshToken.revoked_at == None,  # noqa: E711
        ).values(revoked_at=datetime.now(timezone.utc))
        await self._db.execute(stmt)

        logger.warning("Token family revoked due to potential token theft: %s", family_id)


# Keep old exceptions for backward compatibility with existing code
class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid.

    Note: Password-based login has been removed. Use API key authentication instead.
    This exception is kept for backward compatibility.
    """

    pass


class AccountLockedError(AuthenticationError):
    """Raised when account is locked.

    Note: Account locking has been removed with password-based authentication.
    This exception is kept for backward compatibility.
    """

    pass
