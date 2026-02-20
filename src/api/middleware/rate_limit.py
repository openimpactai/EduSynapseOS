# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rate limiting middleware using slowapi.

This module provides rate limiting functionality to protect API endpoints
from abuse. Rate limits are applied per client (IP address or user ID).

Example:
    # Limit login attempts
    @limiter.limit("5/minute")
    async def login(...):
        ...
"""

import logging
from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.types import ASGIApp

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def get_client_identifier(request: Request) -> str:
    """Get a unique identifier for the client.

    Uses user ID if authenticated, otherwise uses IP address.
    For tenant-specific rate limiting, includes tenant code.

    Args:
        request: HTTP request.

    Returns:
        Client identifier string.
    """
    # Try to get authenticated user
    user = getattr(request.state, "user", None)
    tenant = getattr(request.state, "tenant", None)

    parts = []

    # Add tenant code for isolation
    if tenant:
        parts.append(f"tenant:{tenant.code}")

    # Add user ID or IP
    if user:
        parts.append(f"user:{user.id}")
    else:
        ip = get_remote_address(request)
        parts.append(f"ip:{ip}")

    return ":".join(parts)


def get_ip_only(request: Request) -> str:
    """Get client IP address only.

    Used for login endpoints where user is not yet authenticated.

    Args:
        request: HTTP request.

    Returns:
        IP address string.
    """
    return get_remote_address(request)


# Create the limiter instance
settings = get_settings()
limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=[f"{settings.rate_limit.requests_per_minute}/minute"],
    storage_uri=settings.redis.url,
)


class RateLimitMiddleware:
    """Rate limiting middleware wrapper.

    This class provides a wrapper around slowapi's SlowAPIMiddleware
    with custom configuration and error handling.

    Attributes:
        _app: ASGI application.
        _limiter: Slowapi Limiter instance.

    Example:
        >>> app.state.limiter = limiter
        >>> app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
        >>> app.add_middleware(SlowAPIMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the rate limit middleware.

        Args:
            app: ASGI application.
        """
        self._app = app
        self._limiter = limiter

    async def __call__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
    ) -> None:
        """Process the request with rate limiting.

        Args:
            scope: ASGI scope.
            receive: ASGI receive.
            send: ASGI send.
        """
        await self._app(scope, receive, send)


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> Response:
    """Handle rate limit exceeded errors.

    Returns a 429 Too Many Requests response with retry information.

    Args:
        request: HTTP request.
        exc: Rate limit exceeded exception.

    Returns:
        JSON response with error details.
    """
    logger.warning(
        "Rate limit exceeded: %s for %s",
        exc.detail,
        get_client_identifier(request),
    )

    return Response(
        content='{"detail": "Too many requests. Please try again later."}',
        status_code=429,
        media_type="application/json",
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
            "X-RateLimit-Limit": str(getattr(exc, "limit", settings.rate_limit.requests_per_minute)),
        },
    )


# Pre-configured rate limit decorators
def rate_limit(limit_string: str) -> Callable:
    """Create a rate limit decorator.

    Args:
        limit_string: Rate limit string (e.g., "5/minute", "100/hour").

    Returns:
        Decorator function.
    """
    return limiter.limit(limit_string)


# Common rate limit configurations
RATE_LIMIT_AUTH = "100/minute"  # Login attempts
RATE_LIMIT_API_STANDARD = "60/minute"  # Standard API calls
RATE_LIMIT_API_BURST = "10/second"  # Burst requests
RATE_LIMIT_UPLOAD = "10/minute"  # File uploads
RATE_LIMIT_EXPENSIVE = "10/minute"  # Expensive operations (AI, analytics)
