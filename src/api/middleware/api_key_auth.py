# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API Key authentication middleware.

This middleware validates API key credentials from LMS systems.
Used for tenant LMS integration where the LMS authenticates users
internally and asserts their identity to EduSynapseOS.

Headers:
    X-API-Key: The tenant's API key (required for API auth)
    X-API-Secret: The API secret (required for API auth)
    X-User-Id: User ID from the LMS (user assertion)
    X-User-Email: User email (for user creation/lookup)
    X-User-First-Name: User's first name (optional)
    X-User-Last-Name: User's last name (optional)
    X-User-Type: User type - student, teacher, parent (optional, default: student)

Example:
    # Direct API authentication with user assertion
    POST /api/v1/practice/start
    X-API-Key: tk_abc123def456789012345678901234
    X-API-Secret: ts_xyz789abc123def456789012345678901234567890123456
    X-User-Id: lms_user_12345
    X-User-Email: student@school.com
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.middleware.tenant import TenantContext
from src.domains.auth.api_key_service import (
    APIKeyService,
    APIKeyError,
    InvalidAPIKeyError,
    InvalidAPISecretError,
    APIKeyRevokedError,
    APIKeyInactiveError,
    IPNotAllowedError,
)
from src.infrastructure.database.models.central.api_credential import TenantAPICredential
from src.infrastructure.database.models.central.tenant import Tenant

logger = logging.getLogger(__name__)

# Header names
API_KEY_HEADER = "X-API-Key"
API_SECRET_HEADER = "X-API-Secret"
USER_ID_HEADER = "X-User-Id"
USER_EMAIL_HEADER = "X-User-Email"
USER_FIRST_NAME_HEADER = "X-User-First-Name"
USER_LAST_NAME_HEADER = "X-User-Last-Name"
USER_TYPE_HEADER = "X-User-Type"

# Paths that skip API key authentication
SKIP_PATHS = frozenset({
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
})

# Path prefixes that skip API key authentication
SKIP_PATH_PREFIXES = (
    "/api/v1/system/",  # System admin endpoints use their own auth
    "/api/v1/auth/refresh",  # Token refresh uses JWT not API key
    "/api/v1/auth/logout",  # Logout uses JWT not API key
    "/api/v1/auth/sessions",  # Session management uses JWT
    "/api/v1/auth/me",  # Get current user uses JWT
)


@dataclass
class UserAssertion:
    """User identity asserted by the LMS.

    Attributes:
        external_id: User ID from the LMS system.
        email: User's email address.
        first_name: User's first name.
        last_name: User's last name.
        user_type: Type of user (student, teacher, parent).
    """

    external_id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    user_type: str = "student"


@dataclass
class APIAuthContext:
    """Context for API key authenticated requests.

    Attributes:
        credential: The validated API credential.
        tenant: The tenant associated with the credential.
        user_assertion: User identity asserted by the LMS (if provided).
    """

    credential: TenantAPICredential
    tenant: Tenant
    user_assertion: UserAssertion | None = None


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.

    Validates API credentials from request headers and sets up
    the authentication context for downstream handlers.

    When API credentials are provided:
    1. Validates the API key and secret
    2. Checks IP whitelist if configured
    3. Sets request.state.api_auth with APIAuthContext
    4. Sets request.state.tenant with tenant from credential

    When no API credentials are provided:
    - Continues without API auth context (allows JWT auth to work)

    Attributes:
        _get_central_db: Callable that returns central database session.

    Example:
        >>> middleware = APIKeyAuthMiddleware(app, get_central_session)
    """

    def __init__(
        self,
        app: ASGIApp,
        get_central_db: Callable,
    ) -> None:
        """Initialize the API key auth middleware.

        Args:
            app: ASGI application.
            get_central_db: Callable returning central database session context manager.
        """
        super().__init__(app)
        self._get_central_db = get_central_db

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request and validate API credentials.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler.

        Returns:
            HTTP response.
        """
        # Initialize API auth state as None
        request.state.api_auth = None
        start_time = time.time()

        # Skip API key auth for certain paths
        if self._should_skip(request.url.path):
            return await call_next(request)

        # Check if API credentials are provided
        api_key = request.headers.get(API_KEY_HEADER)
        api_secret = request.headers.get(API_SECRET_HEADER)

        if not api_key or not api_secret:
            # No API credentials - continue without API auth
            # (JWT auth middleware will handle authentication)
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        try:
            async with self._get_central_db() as db:
                service = APIKeyService(db)

                # Validate credentials
                result = await service.validate_credential(
                    api_key=api_key,
                    api_secret=api_secret,
                    client_ip=client_ip,
                )

                # Extract user assertion if provided
                user_assertion = self._extract_user_assertion(request)

                # Create API auth context
                api_auth = APIAuthContext(
                    credential=result.credential,
                    tenant=result.tenant,
                    user_assertion=user_assertion,
                )
                request.state.api_auth = api_auth

                # Also set tenant context for compatibility
                request.state.tenant = TenantContext(
                    tenant_id=result.tenant.id,
                    code=result.tenant.code,
                    name=result.tenant.name,
                    status=result.tenant.status,
                    tier=result.tenant.tier,
                )
                request.state.tenant_code = result.tenant.code

                # Record usage
                await service.record_usage(result.credential)

                # Log successful auth (don't await to not block request)
                response_time_ms = int((time.time() - start_time) * 1000)
                await service.log_audit_event(
                    credential=result.credential,
                    action="authenticate",
                    success=True,
                    endpoint=request.url.path,
                    method=request.method,
                    user_id_asserted=user_assertion.external_id if user_assertion else None,
                    ip_address=client_ip,
                    user_agent=request.headers.get("user-agent"),
                    response_time_ms=response_time_ms,
                )

                await db.commit()

                logger.debug(
                    "API key authenticated: %s (tenant: %s, user: %s)",
                    result.credential.api_key_prefix,
                    result.tenant.code,
                    user_assertion.external_id if user_assertion else "none",
                )

        except InvalidAPIKeyError as e:
            return await self._handle_auth_error(
                request, "INVALID_API_KEY", str(e), 401
            )

        except InvalidAPISecretError as e:
            return await self._handle_auth_error(
                request, "INVALID_API_SECRET", str(e), 401
            )

        except APIKeyRevokedError as e:
            return await self._handle_auth_error(
                request, "API_KEY_REVOKED", str(e), 401
            )

        except APIKeyInactiveError as e:
            return await self._handle_auth_error(
                request, "API_KEY_INACTIVE", str(e), 401
            )

        except IPNotAllowedError as e:
            return await self._handle_auth_error(
                request, "IP_NOT_ALLOWED", str(e), 403
            )

        except APIKeyError as e:
            return await self._handle_auth_error(
                request, "API_KEY_ERROR", str(e), 401
            )

        except Exception as e:
            logger.exception("API key auth error: %s", str(e))
            return await self._handle_auth_error(
                request, "AUTH_ERROR", "Authentication failed", 500
            )

        return await call_next(request)

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip API key authentication.

        Args:
            path: Request path.

        Returns:
            True if should skip.
        """
        if path in SKIP_PATHS:
            return True

        for prefix in SKIP_PATH_PREFIXES:
            if path.startswith(prefix):
                return True

        return False

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract client IP from request.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: HTTP request.

        Returns:
            Client IP address or None.
        """
        # Check X-Forwarded-For header first (for proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct client
        if request.client:
            return request.client.host

        return None

    def _extract_user_assertion(self, request: Request) -> UserAssertion | None:
        """Extract user assertion from request headers.

        Args:
            request: HTTP request.

        Returns:
            UserAssertion if X-User-Id is provided, None otherwise.
        """
        user_id = request.headers.get(USER_ID_HEADER)
        if not user_id:
            return None

        return UserAssertion(
            external_id=user_id,
            email=request.headers.get(USER_EMAIL_HEADER),
            first_name=request.headers.get(USER_FIRST_NAME_HEADER),
            last_name=request.headers.get(USER_LAST_NAME_HEADER),
            user_type=request.headers.get(USER_TYPE_HEADER, "student"),
        )

    async def _handle_auth_error(
        self,
        request: Request,
        error_code: str,
        error_message: str,
        status_code: int,
    ) -> Response:
        """Handle authentication errors.

        Args:
            request: HTTP request.
            error_code: Error code for response.
            error_message: Error message for response.
            status_code: HTTP status code.

        Returns:
            JSON error response.
        """
        import json

        logger.warning(
            "API key auth failed: %s - %s (path: %s)",
            error_code,
            error_message,
            request.url.path,
        )

        return Response(
            content=json.dumps({
                "error": error_code,
                "message": error_message,
            }),
            status_code=status_code,
            media_type="application/json",
        )


def get_api_auth(request: Request) -> APIAuthContext | None:
    """Get API auth context from request state.

    Helper function for accessing the API authentication context.

    Args:
        request: HTTP request with state.

    Returns:
        APIAuthContext or None if not API-authenticated.
    """
    return getattr(request.state, "api_auth", None)


def require_api_auth(request: Request) -> APIAuthContext:
    """Require API authentication context.

    Args:
        request: HTTP request with state.

    Returns:
        APIAuthContext.

    Raises:
        ValueError: If no API auth context in request.
    """
    api_auth = get_api_auth(request)
    if not api_auth:
        raise ValueError("API authentication required")
    return api_auth


def get_user_assertion(request: Request) -> UserAssertion | None:
    """Get user assertion from API auth context.

    Args:
        request: HTTP request with state.

    Returns:
        UserAssertion or None if not provided.
    """
    api_auth = get_api_auth(request)
    if api_auth:
        return api_auth.user_assertion
    return None
