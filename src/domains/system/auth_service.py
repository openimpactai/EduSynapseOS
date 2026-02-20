# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System admin authentication service.

This module provides authentication for system administrators who manage
tenants and platform configuration. System admins use the Central DB
for authentication, separate from tenant users.

Example:
    >>> auth_service = SystemAuthService(db, jwt_manager, password_hasher)
    >>> result = await auth_service.login("admin@edusynapse.io", "password", device_info)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domains.auth.jwt import JWTManager, TokenPair, TokenExpiredError, InvalidTokenError
from src.domains.auth.password import PasswordHasher
from src.infrastructure.database.models.central.system_user import SystemUser
from src.infrastructure.database.models.central.system_session import SystemSession

logger = logging.getLogger(__name__)


class SystemAuthError(Exception):
    """Base exception for system authentication errors."""

    pass


class SystemInvalidCredentialsError(SystemAuthError):
    """Raised when login credentials are invalid."""

    pass


class SystemAccountLockedError(SystemAuthError):
    """Raised when account is locked due to failed login attempts."""

    pass


class SystemAccountInactiveError(SystemAuthError):
    """Raised when account is not active."""

    pass


class SystemSessionNotFoundError(SystemAuthError):
    """Raised when session is not found or invalid."""

    pass


class SystemTokenRefreshError(SystemAuthError):
    """Raised when token refresh fails."""

    pass


class DeviceInfo(NamedTuple):
    """Device information for session tracking."""

    ip_address: str | None = None
    user_agent: str | None = None


class SystemLoginResult(NamedTuple):
    """Result of a successful system admin login."""

    user: SystemUser
    tokens: TokenPair
    session_id: str


class SystemAuthService:
    """Authentication service for system administrators.

    Handles authentication using the Central database for platform-level
    admins who manage tenants and system configuration.

    Attributes:
        _db: Central database session.
        _jwt_manager: JWT token manager.
        _password_hasher: Password hashing utility.
        _max_failed_attempts: Maximum failed login attempts before lockout.
        _lockout_duration_minutes: Duration of account lockout.

    Example:
        >>> auth_service = SystemAuthService(db, jwt_manager, password_hasher)
        >>> result = await auth_service.login(
        ...     email="admin@edusynapse.io",
        ...     password="secure_password",
        ...     device_info=DeviceInfo(ip_address="192.168.1.1"),
        ... )
        >>> print(result.tokens.access_token)
    """

    def __init__(
        self,
        db: AsyncSession,
        jwt_manager: JWTManager,
        password_hasher: PasswordHasher | None = None,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30,
    ) -> None:
        """Initialize the system authentication service.

        Args:
            db: Central database async session.
            jwt_manager: JWT token manager.
            password_hasher: Password hasher (uses default if not provided).
            max_failed_attempts: Max failed login attempts before lockout.
            lockout_duration_minutes: Duration of account lockout in minutes.
        """
        self._db = db
        self._jwt_manager = jwt_manager
        self._password_hasher = password_hasher or PasswordHasher()
        self._max_failed_attempts = max_failed_attempts
        self._lockout_duration_minutes = lockout_duration_minutes

    async def login(
        self,
        email: str,
        password: str,
        device_info: DeviceInfo | None = None,
    ) -> SystemLoginResult:
        """Authenticate a system administrator.

        Args:
            email: Admin email address.
            password: Plain text password.
            device_info: Optional device information.

        Returns:
            SystemLoginResult with user, tokens, and session ID.

        Raises:
            SystemInvalidCredentialsError: If email or password is incorrect.
            SystemAccountLockedError: If account is locked.
            SystemAccountInactiveError: If account is not active.
        """
        device_info = device_info or DeviceInfo()

        stmt = select(SystemUser).where(SystemUser.email == email)
        result = await self._db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            logger.warning("System login failed: user not found for email %s", email)
            raise SystemInvalidCredentialsError("Invalid email or password")

        if user.is_locked:
            logger.warning("System login failed: account locked for user %s", user.id)
            raise SystemAccountLockedError(
                f"Account locked due to too many failed attempts. "
                f"Try again after {user.locked_until}"
            )

        if not user.is_active:
            logger.warning("System login failed: inactive account for user %s", user.id)
            raise SystemAccountInactiveError("Account is not active")

        if not self._password_hasher.verify(password, user.password_hash):
            await self._handle_failed_login(user)
            raise SystemInvalidCredentialsError("Invalid email or password")

        user.reset_failed_attempts()
        user.record_login(device_info.ip_address)

        tokens = self._jwt_manager.create_token_pair(
            user_id=user.id,
            tenant_id=None,
            tenant_code=None,
            user_type="system_admin",
            roles=[user.role],
            permissions=self._get_system_permissions(user.role),
        )

        session = await self._create_session(
            user=user,
            tokens=tokens,
            device_info=device_info,
        )

        await self._db.commit()

        logger.info("System admin logged in: %s", user.id)

        return SystemLoginResult(user=user, tokens=tokens, session_id=session.id)

    async def refresh_tokens(self, refresh_token: str) -> TokenPair:
        """Refresh access token using a refresh token.

        Args:
            refresh_token: Current refresh token.

        Returns:
            New TokenPair with fresh access and refresh tokens.

        Raises:
            SystemTokenRefreshError: If refresh token is invalid or expired.
        """
        try:
            payload = self._jwt_manager.decode_token(refresh_token, expected_type="refresh")
        except (TokenExpiredError, InvalidTokenError) as e:
            raise SystemTokenRefreshError(f"Invalid refresh token: {str(e)}")

        token_hash = self._jwt_manager.hash_token(refresh_token)
        stmt = select(SystemSession).where(
            SystemSession.refresh_token_hash == token_hash,
            SystemSession.user_id == payload.sub,
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise SystemTokenRefreshError("Session not found")

        if not session.can_refresh():
            raise SystemTokenRefreshError("Session cannot be refreshed")

        stmt = select(SystemUser).where(SystemUser.id == payload.sub)
        result = await self._db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise SystemTokenRefreshError("User not found or inactive")

        new_tokens = self._jwt_manager.create_token_pair(
            user_id=user.id,
            tenant_id=None,
            tenant_code=None,
            user_type="system_admin",
            roles=[user.role],
            permissions=self._get_system_permissions(user.role),
        )

        session.access_token_hash = self._jwt_manager.hash_token(new_tokens.access_token)
        session.refresh_token_hash = self._jwt_manager.hash_token(new_tokens.refresh_token)
        session.access_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self._jwt_manager._settings.access_token_expire_minutes
        )
        session.refresh_expires_at = datetime.now(timezone.utc) + timedelta(
            days=self._jwt_manager._settings.refresh_token_expire_days
        )

        await self._db.commit()

        logger.info("System tokens refreshed for user: %s", user.id)

        return new_tokens

    async def logout(self, session_id: str | UUID) -> None:
        """Logout a system admin session.

        Args:
            session_id: Session to logout.
        """
        stmt = select(SystemSession).where(SystemSession.id == str(session_id))
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if session:
            session.revoke()
            await self._db.commit()
            logger.info("System session revoked: %s", session_id)

    async def logout_all(self, user_id: str | UUID) -> None:
        """Logout all sessions for a system admin.

        Args:
            user_id: User whose sessions to logout.
        """
        stmt = update(SystemSession).where(
            SystemSession.user_id == str(user_id),
            SystemSession.revoked_at == None,  # noqa: E711
        ).values(revoked_at=datetime.now(timezone.utc))
        await self._db.execute(stmt)
        await self._db.commit()
        logger.info("All system sessions revoked for user: %s", user_id)

    async def get_current_user(self, access_token: str) -> SystemUser | None:
        """Get the current user from an access token.

        Args:
            access_token: JWT access token.

        Returns:
            SystemUser if valid, None otherwise.
        """
        try:
            payload = self._jwt_manager.decode_token(access_token, expected_type="access")
        except (TokenExpiredError, InvalidTokenError):
            return None

        token_hash = self._jwt_manager.hash_token(access_token)
        stmt = select(SystemSession).where(
            SystemSession.access_token_hash == token_hash,
            SystemSession.revoked_at == None,  # noqa: E711
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session or not session.is_active:
            return None

        stmt = select(SystemUser).where(SystemUser.id == payload.sub)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def validate_session(self, access_token: str) -> SystemSession | None:
        """Validate an access token and return the session.

        Args:
            access_token: JWT access token.

        Returns:
            SystemSession if valid, None otherwise.
        """
        token_hash = self._jwt_manager.hash_token(access_token)

        stmt = select(SystemSession).where(
            SystemSession.access_token_hash == token_hash,
            SystemSession.revoked_at == None,  # noqa: E711
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if session and session.is_active:
            return session

        return None

    def _get_system_permissions(self, role: str) -> list[str]:
        """Get permissions for a system admin role.

        Args:
            role: The admin role (admin, super_admin).

        Returns:
            List of permission codes.
        """
        base_permissions = [
            "tenants.view",
            "tenants.create",
            "tenants.update",
            "licenses.view",
            "system.audit.view",
        ]

        if role == "super_admin":
            return base_permissions + [
                "tenants.delete",
                "licenses.create",
                "licenses.update",
                "licenses.delete",
                "system.users.manage",
                "system.config.manage",
            ]

        return base_permissions

    async def _handle_failed_login(self, user: SystemUser) -> None:
        """Handle a failed login attempt."""
        user.increment_failed_attempts()

        if user.failed_login_attempts >= self._max_failed_attempts:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=self._lockout_duration_minutes
            )
            logger.warning(
                "System account locked for user %s after %d failed attempts",
                user.id,
                user.failed_login_attempts,
            )

        await self._db.commit()

    async def _create_session(
        self,
        user: SystemUser,
        tokens: TokenPair,
        device_info: DeviceInfo,
    ) -> SystemSession:
        """Create a new system session."""
        now = datetime.now(timezone.utc)
        session = SystemSession(
            user_id=user.id,
            access_token_hash=self._jwt_manager.hash_token(tokens.access_token),
            refresh_token_hash=self._jwt_manager.hash_token(tokens.refresh_token),
            ip_address=device_info.ip_address,
            user_agent=device_info.user_agent,
            access_expires_at=now + timedelta(
                minutes=self._jwt_manager._settings.access_token_expire_minutes
            ),
            refresh_expires_at=now + timedelta(
                days=self._jwt_manager._settings.refresh_token_expire_days
            ),
        )

        self._db.add(session)
        await self._db.flush()

        return session
