# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""JWT token management utilities.

This module provides JWT token creation and validation using python-jose.
Supports access tokens and refresh tokens with configurable expiration.

Example:
    >>> from src.core.config import get_settings
    >>> jwt_manager = JWTManager(get_settings().jwt)
    >>> token = jwt_manager.create_access_token(user_id="user-123", ...)
    >>> claims = jwt_manager.decode_token(token.access_token)
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from jose import ExpiredSignatureError, JWTError as JoseJWTError, jwt
from pydantic import BaseModel

from src.core.config.settings import JWTSettings

logger = logging.getLogger(__name__)


class TokenPayload(BaseModel):
    """JWT token payload structure.

    Attributes:
        sub: Subject (user ID).
        type: Token type (access or refresh).
        tenant_id: Tenant ID for multi-tenant context.
        tenant_code: Tenant code for routing.
        user_type: User type (student, teacher, etc.).
        roles: List of role codes.
        permissions: List of permission codes.
        school_ids: List of accessible school IDs.
        preferred_language: User's preferred language code.
        exp: Expiration timestamp.
        iat: Issued at timestamp.
        jti: JWT ID for token tracking.
    """

    sub: str
    type: Literal["access", "refresh"]
    tenant_id: str | None = None
    tenant_code: str | None = None
    user_type: str | None = None
    roles: list[str] = []
    permissions: list[str] = []
    school_ids: list[str] = []
    preferred_language: str = "en"
    exp: int
    iat: int
    jti: str


class TokenPair(BaseModel):
    """Access and refresh token pair.

    Attributes:
        access_token: JWT access token string.
        refresh_token: JWT refresh token string.
        token_type: Token type (always "Bearer").
        expires_in: Access token expiration in seconds.
        refresh_expires_in: Refresh token expiration in seconds.
    """

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_expires_in: int


class JWTError(Exception):
    """Base exception for JWT operations."""

    pass


class TokenExpiredError(JWTError):
    """Raised when a token has expired."""

    pass


class InvalidTokenError(JWTError):
    """Raised when a token is invalid."""

    pass


class JWTManager:
    """JWT token creation and validation manager.

    Handles creation and validation of JWT tokens for authentication.
    Supports both access tokens (short-lived) and refresh tokens (long-lived).

    Attributes:
        _settings: JWT configuration settings.

    Example:
        >>> jwt_manager = JWTManager(settings)
        >>> tokens = jwt_manager.create_token_pair(
        ...     user_id="user-123",
        ...     tenant_id="tenant-456",
        ...     tenant_code="school_abc",
        ...     user_type="student",
        ...     roles=["student"],
        ...     permissions=["practice.create"],
        ... )
        >>> claims = jwt_manager.decode_token(tokens.access_token)
    """

    def __init__(self, settings: JWTSettings) -> None:
        """Initialize the JWT manager.

        Args:
            settings: JWT configuration settings.
        """
        self._settings = settings

    def create_token_pair(
        self,
        user_id: str | UUID,
        tenant_id: str | UUID | None = None,
        tenant_code: str | None = None,
        user_type: str | None = None,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        school_ids: list[str | UUID] | None = None,
        preferred_language: str = "en",
    ) -> TokenPair:
        """Create an access and refresh token pair.

        Args:
            user_id: User identifier.
            tenant_id: Tenant identifier.
            tenant_code: Tenant code for routing.
            user_type: Type of user (student, teacher, etc.).
            roles: List of role codes.
            permissions: List of permission codes.
            school_ids: List of accessible school IDs.
            preferred_language: User's preferred language code.

        Returns:
            TokenPair with access and refresh tokens.
        """
        now = datetime.now(timezone.utc)
        access_exp = now + timedelta(minutes=self._settings.access_token_expire_minutes)
        refresh_exp = now + timedelta(days=self._settings.refresh_token_expire_days)

        # Convert UUIDs to strings
        user_id_str = str(user_id)
        tenant_id_str = str(tenant_id) if tenant_id else None
        school_ids_str = [str(s) for s in (school_ids or [])]

        # Create access token
        access_payload = {
            "sub": user_id_str,
            "type": "access",
            "tenant_id": tenant_id_str,
            "tenant_code": tenant_code,
            "user_type": user_type,
            "roles": roles or [],
            "permissions": permissions or [],
            "school_ids": school_ids_str,
            "preferred_language": preferred_language,
            "exp": int(access_exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(16),
        }

        access_token = jwt.encode(
            access_payload,
            self._settings.secret_key.get_secret_value(),
            algorithm=self._settings.algorithm,
        )

        # Create refresh token
        refresh_payload = {
            "sub": user_id_str,
            "type": "refresh",
            "tenant_id": tenant_id_str,
            "tenant_code": tenant_code,
            "exp": int(refresh_exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(16),
        }

        refresh_token = jwt.encode(
            refresh_payload,
            self._settings.secret_key.get_secret_value(),
            algorithm=self._settings.algorithm,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=self._settings.access_token_expire_minutes * 60,
            refresh_expires_in=self._settings.refresh_token_expire_days * 24 * 60 * 60,
        )

    def create_access_token(
        self,
        user_id: str | UUID,
        tenant_id: str | UUID | None = None,
        tenant_code: str | None = None,
        user_type: str | None = None,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        school_ids: list[str | UUID] | None = None,
    ) -> str:
        """Create an access token.

        Args:
            user_id: User identifier.
            tenant_id: Tenant identifier.
            tenant_code: Tenant code for routing.
            user_type: Type of user.
            roles: List of role codes.
            permissions: List of permission codes.
            school_ids: List of accessible school IDs.

        Returns:
            JWT access token string.
        """
        now = datetime.now(timezone.utc)
        exp = now + timedelta(minutes=self._settings.access_token_expire_minutes)

        payload = {
            "sub": str(user_id),
            "type": "access",
            "tenant_id": str(tenant_id) if tenant_id else None,
            "tenant_code": tenant_code,
            "user_type": user_type,
            "roles": roles or [],
            "permissions": permissions or [],
            "school_ids": [str(s) for s in (school_ids or [])],
            "exp": int(exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(16),
        }

        return jwt.encode(
            payload,
            self._settings.secret_key.get_secret_value(),
            algorithm=self._settings.algorithm,
        )

    def decode_token(
        self,
        token: str,
        expected_type: Literal["access", "refresh"] | None = None,
    ) -> TokenPayload:
        """Decode and validate a JWT token.

        Args:
            token: JWT token string.
            expected_type: Expected token type (access or refresh).

        Returns:
            TokenPayload with decoded claims.

        Raises:
            TokenExpiredError: If the token has expired.
            InvalidTokenError: If the token is invalid or wrong type.
        """
        try:
            payload = jwt.decode(
                token,
                self._settings.secret_key.get_secret_value(),
                algorithms=[self._settings.algorithm],
            )

            # Validate token type if specified
            if expected_type and payload.get("type") != expected_type:
                raise InvalidTokenError(
                    f"Expected {expected_type} token, got {payload.get('type')}"
                )

            return TokenPayload(
                sub=payload["sub"],
                type=payload["type"],
                tenant_id=payload.get("tenant_id"),
                tenant_code=payload.get("tenant_code"),
                user_type=payload.get("user_type"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                school_ids=payload.get("school_ids", []),
                exp=payload["exp"],
                iat=payload["iat"],
                jti=payload["jti"],
            )

        except ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except Exception as e:
            logger.warning("Token decode failed: %s", str(e))
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    def verify_token(
        self,
        token: str,
        expected_type: Literal["access", "refresh"] | None = None,
    ) -> bool:
        """Verify if a token is valid.

        Args:
            token: JWT token string.
            expected_type: Expected token type (access or refresh).

        Returns:
            True if token is valid, False otherwise.
        """
        try:
            self.decode_token(token, expected_type)
            return True
        except (TokenExpiredError, InvalidTokenError):
            return False

    @staticmethod
    def hash_token(token: str) -> str:
        """Create a SHA-256 hash of a token.

        Used for storing token hashes in the database instead of
        the actual token for security.

        Args:
            token: Token string to hash.

        Returns:
            SHA-256 hash of the token as hex string.
        """
        return hashlib.sha256(token.encode()).hexdigest()
