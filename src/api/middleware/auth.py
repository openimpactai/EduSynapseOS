# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""JWT authentication middleware.

This middleware validates JWT tokens from the Authorization header
and populates request.state with user information.

Example:
    # Request with Bearer token
    GET /api/v1/students
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
"""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.config import get_settings
from src.domains.auth.jwt import JWTManager, TokenPayload, TokenExpiredError, InvalidTokenError

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = frozenset({
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/password/reset",
    "/api/v1/auth/password/reset/confirm",
    "/api/v1/auth/verify-email",
})

# Path prefixes that don't require authentication
PUBLIC_PATH_PREFIXES = (
    "/api/v1/public/",
)


class CurrentUser:
    """Current authenticated user from JWT token.

    Attributes:
        id: User UUID.
        tenant_id: Tenant UUID.
        tenant_code: Tenant code.
        user_type: Type of user.
        roles: List of role codes.
        permissions: List of permission codes.
        school_ids: List of accessible school IDs.
        preferred_language: User's preferred language code.
    """

    def __init__(self, payload: TokenPayload) -> None:
        """Initialize from token payload.

        Args:
            payload: Decoded JWT token payload.
        """
        self.id = payload.sub
        self.tenant_id = payload.tenant_id
        self.tenant_code = payload.tenant_code
        self.user_type = payload.user_type
        self.roles = payload.roles
        self.permissions = payload.permissions
        self.school_ids = payload.school_ids
        self.preferred_language = payload.preferred_language

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role.

        Args:
            role: Role code to check.

        Returns:
            True if user has the role.
        """
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission.

        Args:
            permission: Permission code to check.

        Returns:
            True if user has the permission.
        """
        return permission in self.permissions

    def has_any_role(self, *roles: str) -> bool:
        """Check if user has any of the specified roles.

        Args:
            roles: Role codes to check.

        Returns:
            True if user has any of the roles.
        """
        return any(role in self.roles for role in roles)

    def has_any_permission(self, *permissions: str) -> bool:
        """Check if user has any of the specified permissions.

        Args:
            permissions: Permission codes to check.

        Returns:
            True if user has any of the permissions.
        """
        return any(perm in self.permissions for perm in permissions)

    def has_all_permissions(self, *permissions: str) -> bool:
        """Check if user has all of the specified permissions.

        Args:
            permissions: Permission codes to check.

        Returns:
            True if user has all of the permissions.
        """
        return all(perm in self.permissions for perm in permissions)

    def can_access_school(self, school_id: str) -> bool:
        """Check if user can access a specific school.

        Args:
            school_id: School ID to check.

        Returns:
            True if user can access the school.
        """
        # Tenant admins can access all schools
        if self.user_type == "tenant_admin":
            return True
        return school_id in self.school_ids

    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.user_type in ("tenant_admin", "school_admin")

    @property
    def is_teacher(self) -> bool:
        """Check if user is a teacher."""
        return self.user_type == "teacher"

    @property
    def is_student(self) -> bool:
        """Check if user is a student."""
        return self.user_type == "student"

    @property
    def is_parent(self) -> bool:
        """Check if user is a parent."""
        return self.user_type == "parent"


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication.

    Validates Bearer tokens from the Authorization header and
    populates request.state.user with CurrentUser.

    For public paths, authentication is skipped.
    For protected paths without valid token, request continues
    with request.state.user = None (let endpoint handle auth requirements).

    Attributes:
        _jwt_manager: JWT token manager.

    Example:
        >>> middleware = AuthMiddleware(app)
        >>> # After processing, request.state.user is set
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the auth middleware.

        Args:
            app: ASGI application.
        """
        super().__init__(app)
        settings = get_settings()
        self._jwt_manager = JWTManager(settings.jwt)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request and validate authentication.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler.

        Returns:
            HTTP response.
        """
        # Initialize user state as None
        request.state.user = None

        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Try to extract and validate token
        try:
            token = self._extract_token(request)
            if token:
                payload = self._jwt_manager.decode_token(token, expected_type="access")
                request.state.user = CurrentUser(payload)
                logger.debug("User authenticated: %s", payload.sub)

        except TokenExpiredError:
            logger.debug("Token expired")
            # Continue with user=None, let endpoint handle

        except InvalidTokenError as e:
            logger.debug("Invalid token: %s", str(e))
            # Continue with user=None, let endpoint handle

        except Exception as e:
            logger.warning("Auth middleware error: %s", str(e))
            # Continue with user=None

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required).

        Args:
            path: Request path.

        Returns:
            True if path is public.
        """
        if path in PUBLIC_PATHS:
            return True

        for prefix in PUBLIC_PATH_PREFIXES:
            if path.startswith(prefix):
                return True

        return False

    def _extract_token(self, request: Request) -> str | None:
        """Extract JWT token from Authorization header.

        Expects format: Bearer <token>

        Args:
            request: HTTP request.

        Returns:
            Token string or None if not found.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        return parts[1]


def get_current_user(request: Request) -> CurrentUser | None:
    """Get current user from request state.

    Helper function for accessing the authenticated user.

    Args:
        request: HTTP request with state.

    Returns:
        CurrentUser or None if not authenticated.
    """
    return getattr(request.state, "user", None)


def require_user(request: Request) -> CurrentUser:
    """Get current user, raising if not authenticated.

    Args:
        request: HTTP request with state.

    Returns:
        CurrentUser.

    Raises:
        ValueError: If no user in request (should return 401).
    """
    user = get_current_user(request)
    if not user:
        raise ValueError("Not authenticated")
    return user
