# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant resolution middleware.

This middleware resolves the tenant context from:
1. X-Tenant-Code header
2. Subdomain (e.g., school_abc.edusynapse.com)
3. JWT token claims (if authenticated)

The resolved tenant is stored in request.state for use by dependencies.

Example:
    # Request with header
    GET /api/v1/students
    X-Tenant-Code: school_abc

    # Request with subdomain
    GET https://school_abc.edusynapse.com/api/v1/students
"""

import logging
from typing import Callable
from uuid import UUID

from fastapi import Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.infrastructure.database.models.central.tenant import Tenant

logger = logging.getLogger(__name__)

# Header name for tenant code
TENANT_HEADER = "X-Tenant-Code"

# Paths that don't require tenant context
PUBLIC_PATHS = frozenset({
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
})

# Path prefixes that don't require tenant context
PUBLIC_PATH_PREFIXES = (
    "/api/v1/system/",  # System admin endpoints
)


class TenantContext:
    """Tenant context resolved from request.

    Attributes:
        id: Tenant UUID.
        code: Tenant code string.
        name: Tenant display name.
        status: Tenant status.
        tier: Subscription tier.
    """

    def __init__(
        self,
        tenant_id: str | UUID,
        code: str,
        name: str,
        status: str,
        tier: str,
    ) -> None:
        """Initialize tenant context.

        Args:
            tenant_id: Tenant identifier.
            code: Tenant code.
            name: Tenant name.
            status: Tenant status.
            tier: Subscription tier.
        """
        self.id = str(tenant_id) if isinstance(tenant_id, UUID) else tenant_id
        self.code = code
        self.name = name
        self.status = status
        self.tier = tier

    @property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == "active"


class TenantResolutionError(Exception):
    """Raised when tenant resolution fails."""

    pass


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware for resolving tenant context from requests.

    Resolves tenant from:
    1. X-Tenant-Code header (highest priority)
    2. Subdomain extraction from Host header
    3. JWT token claims (for authenticated requests)

    The resolved tenant is stored in request.state.tenant.

    Attributes:
        _get_central_db: Callable that returns central database session.
        _base_domain: Base domain for subdomain extraction.

    Example:
        >>> middleware = TenantMiddleware(app, get_central_db, "edusynapse.com")
        >>> # Adds request.state.tenant with TenantContext
    """

    def __init__(
        self,
        app: ASGIApp,
        get_central_db: Callable[[], AsyncSession],
        base_domain: str = "edusynapse.com",
    ) -> None:
        """Initialize the tenant middleware.

        Args:
            app: ASGI application.
            get_central_db: Callable returning central database session.
            base_domain: Base domain for subdomain extraction.
        """
        super().__init__(app)
        self._get_central_db = get_central_db
        self._base_domain = base_domain.lower()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Process the request and resolve tenant.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler.

        Returns:
            HTTP response.
        """
        # Initialize tenant state as None
        request.state.tenant = None
        request.state.tenant_code = None

        # Skip tenant resolution for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        try:
            tenant_code = self._extract_tenant_code(request)

            if tenant_code:
                tenant = await self._resolve_tenant(tenant_code)
                if tenant:
                    request.state.tenant = tenant
                    request.state.tenant_code = tenant.code
                    logger.debug("Tenant resolved: %s", tenant.code)
                else:
                    logger.warning("Tenant not found: %s", tenant_code)

        except Exception as e:
            logger.warning("Tenant resolution error: %s", str(e))

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no tenant required).

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

    def _extract_tenant_code(self, request: Request) -> str | None:
        """Extract tenant code from request.

        Priority:
        1. X-Tenant-Code header
        2. Subdomain from Host header

        Args:
            request: HTTP request.

        Returns:
            Tenant code or None if not found.
        """
        # 1. Try header first
        tenant_code = request.headers.get(TENANT_HEADER)
        if tenant_code:
            return tenant_code.strip().lower()

        # 2. Try subdomain
        host = request.headers.get("host", "")
        tenant_code = self._extract_subdomain(host)
        if tenant_code:
            return tenant_code

        return None

    def _extract_subdomain(self, host: str) -> str | None:
        """Extract subdomain from host.

        Examples:
            school_abc.edusynapse.com -> school_abc
            edusynapse.com -> None
            localhost:8000 -> None

        Args:
            host: Host header value.

        Returns:
            Subdomain or None.
        """
        if not host:
            return None

        # Remove port if present
        host = host.split(":")[0].lower()

        # Skip localhost
        if host in ("localhost", "127.0.0.1"):
            return None

        # Check if host ends with base domain
        if not host.endswith(self._base_domain):
            return None

        # Extract subdomain
        subdomain = host[: -(len(self._base_domain) + 1)]  # +1 for the dot
        if subdomain and subdomain != "www":
            return subdomain

        return None

    async def _resolve_tenant(self, tenant_code: str) -> TenantContext | None:
        """Resolve tenant from database.

        Args:
            tenant_code: Tenant code to look up.

        Returns:
            TenantContext or None if not found.
        """
        async with self._get_central_db() as db:
            stmt = select(Tenant).where(
                Tenant.code == tenant_code,
                Tenant.deleted_at == None,  # noqa: E711
            )
            result = await db.execute(stmt)
            tenant = result.scalar_one_or_none()

            if tenant:
                return TenantContext(
                    tenant_id=tenant.id,
                    code=tenant.code,
                    name=tenant.name,
                    status=tenant.status,
                    tier=tenant.tier,
                )

            return None


def get_tenant_from_request(request: Request) -> TenantContext | None:
    """Get tenant context from request state.

    Helper function for accessing the resolved tenant.

    Args:
        request: HTTP request with state.

    Returns:
        TenantContext or None.
    """
    return getattr(request.state, "tenant", None)


def require_tenant(request: Request) -> TenantContext:
    """Get tenant context, raising if not present.

    Args:
        request: HTTP request with state.

    Returns:
        TenantContext.

    Raises:
        TenantResolutionError: If no tenant in request.
    """
    tenant = get_tenant_from_request(request)
    if not tenant:
        raise TenantResolutionError("No tenant context in request")
    return tenant
