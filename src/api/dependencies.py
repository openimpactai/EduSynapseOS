# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""FastAPI dependency injection definitions.

This module provides dependency functions for FastAPI endpoints.
Dependencies are used to:
- Get database sessions (central and tenant)
- Get authenticated users
- Get tenant context
- Get service instances

Example:
    @router.get("/students")
    async def list_students(
        db: AsyncSession = Depends(get_tenant_db),
        current_user: CurrentUser = Depends(require_auth),
    ):
        ...
"""

import logging
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import CurrentUser, get_current_user
from src.api.middleware.tenant import TenantContext, get_tenant_from_request
from src.core.config import get_settings
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.orchestration.checkpointer import (
    init_checkpointer,
    get_checkpointer_instance,
    close_checkpointer,
)
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.auth.service import AuthService
from src.infrastructure.database.connection import (
    init_central_database,
    close_central_database,
    get_central_session,
)
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.vectors import init_qdrant, close_qdrant

logger = logging.getLogger(__name__)

# Tenant database manager singleton
_tenant_db_manager: TenantDatabaseManager | None = None


async def init_db() -> None:
    """Initialize all database connections, vector store, and checkpointer."""
    global _tenant_db_manager
    settings = get_settings()

    # Initialize central database
    await init_central_database(settings)

    # Initialize tenant database manager
    _tenant_db_manager = TenantDatabaseManager(settings)

    # Initialize Qdrant vector store
    await init_qdrant(settings)

    # Initialize checkpointer for workflow state persistence
    await init_checkpointer(settings.central_db.url)


async def close_db() -> None:
    """Close all database connections, vector store, and checkpointer."""
    global _tenant_db_manager

    # Close checkpointer
    await close_checkpointer()

    # Close Qdrant vector store
    await close_qdrant()

    # Close central database
    await close_central_database()

    # Close tenant connections
    if _tenant_db_manager:
        await _tenant_db_manager.close_all()
        _tenant_db_manager = None


async def get_central_db() -> AsyncGenerator[AsyncSession, None]:
    """Get central database session.

    Dependency for endpoints that need central database access.

    Yields:
        AsyncSession for central database.
    """
    async with get_central_session() as session:
        yield session


async def get_tenant_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Get tenant database session.

    Resolves tenant from request and returns a session for that tenant's database.

    Args:
        request: HTTP request with tenant context.

    Yields:
        AsyncSession for tenant database.

    Raises:
        HTTPException: If no tenant context in request.
    """
    global _tenant_db_manager

    tenant = get_tenant_from_request(request)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required",
        )

    if not _tenant_db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database manager not initialized",
        )

    async with _tenant_db_manager.get_session(tenant.code) as session:
        yield session


# =========================================================================
# Authentication Dependencies
# =========================================================================


def get_optional_user(request: Request) -> CurrentUser | None:
    """Get current user if authenticated, None otherwise.

    Args:
        request: HTTP request.

    Returns:
        CurrentUser or None.
    """
    return get_current_user(request)


def require_auth(request: Request) -> CurrentUser:
    """Require authenticated user.

    Args:
        request: HTTP request.

    Returns:
        CurrentUser.

    Raises:
        HTTPException: If not authenticated.
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(request: Request) -> CurrentUser:
    """Require admin user (tenant_admin or school_admin).

    Args:
        request: HTTP request.

    Returns:
        CurrentUser.

    Raises:
        HTTPException: If not authenticated or not admin.
    """
    user = require_auth(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_tenant_admin(request: Request) -> CurrentUser:
    """Require tenant admin user.

    Args:
        request: HTTP request.

    Returns:
        CurrentUser.

    Raises:
        HTTPException: If not tenant admin.
    """
    user = require_auth(request)
    if user.user_type != "tenant_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant admin access required",
        )
    return user


def require_teacher(request: Request) -> CurrentUser:
    """Require teacher user.

    Args:
        request: HTTP request.

    Returns:
        CurrentUser.

    Raises:
        HTTPException: If not a teacher.
    """
    user = require_auth(request)
    if not user.is_teacher:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher access required",
        )
    return user


def require_teacher_or_admin(request: Request) -> CurrentUser:
    """Require teacher or admin user.

    Args:
        request: HTTP request.

    Returns:
        CurrentUser.

    Raises:
        HTTPException: If not teacher or admin.
    """
    user = require_auth(request)
    if not (user.is_teacher or user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher or admin access required",
        )
    return user


class RequirePermission:
    """Dependency for requiring specific permissions.

    Example:
        @router.get("/students")
        async def list_students(
            user: CurrentUser = Depends(RequirePermission("students.view")),
        ):
            ...
    """

    def __init__(self, *permissions: str, require_all: bool = False) -> None:
        """Initialize permission requirement.

        Args:
            permissions: Required permission codes.
            require_all: If True, require all permissions. If False, any.
        """
        self.permissions = permissions
        self.require_all = require_all

    def __call__(self, request: Request) -> CurrentUser:
        """Check permissions and return user.

        Args:
            request: HTTP request.

        Returns:
            CurrentUser.

        Raises:
            HTTPException: If missing required permissions.
        """
        user = require_auth(request)

        if self.require_all:
            if not user.has_all_permissions(*self.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permissions: {', '.join(self.permissions)}",
                )
        else:
            if not user.has_any_permission(*self.permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of: {', '.join(self.permissions)}",
                )

        return user


class RequireRole:
    """Dependency for requiring specific roles.

    Example:
        @router.get("/admin")
        async def admin_only(
            user: CurrentUser = Depends(RequireRole("tenant_admin")),
        ):
            ...
    """

    def __init__(self, *roles: str) -> None:
        """Initialize role requirement.

        Args:
            roles: Required role codes (any of these).
        """
        self.roles = roles

    def __call__(self, request: Request) -> CurrentUser:
        """Check roles and return user.

        Args:
            request: HTTP request.

        Returns:
            CurrentUser.

        Raises:
            HTTPException: If missing required roles.
        """
        user = require_auth(request)

        if not user.has_any_role(*self.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(self.roles)}",
            )

        return user


# =========================================================================
# Tenant Dependencies
# =========================================================================


def require_api_credentials(request: Request) -> TenantContext:
    """Require API credentials authentication (no JWT needed).

    This dependency validates that the request has valid API credentials
    (X-API-Key and X-API-Secret headers) and returns the tenant context.
    Unlike require_auth, this does NOT require a JWT token.

    Used for public-facing endpoints that need tenant context but not user auth.

    Args:
        request: HTTP request.

    Returns:
        TenantContext from API credentials.

    Raises:
        HTTPException: If no valid API credentials.
    """
    from src.api.middleware.api_key_auth import get_api_auth

    api_auth = get_api_auth(request)
    if not api_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API credentials required. Provide X-API-Key and X-API-Secret headers.",
        )

    tenant = get_tenant_from_request(request)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context not established from API credentials.",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is not active",
        )

    return tenant


def get_tenant(request: Request) -> TenantContext | None:
    """Get tenant context if available.

    Args:
        request: HTTP request.

    Returns:
        TenantContext or None.
    """
    return get_tenant_from_request(request)


def require_tenant(request: Request) -> TenantContext:
    """Require tenant context.

    Args:
        request: HTTP request.

    Returns:
        TenantContext.

    Raises:
        HTTPException: If no tenant context.
    """
    tenant = get_tenant_from_request(request)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required. Provide X-Tenant-Code header.",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is not active",
        )

    return tenant


# =========================================================================
# Service Dependencies
# =========================================================================


def get_jwt_manager() -> JWTManager:
    """Get JWT manager instance.

    Returns:
        JWTManager.
    """
    settings = get_settings()
    return JWTManager(settings.jwt)


def get_password_hasher() -> PasswordHasher:
    """Get password hasher instance.

    Returns:
        PasswordHasher.
    """
    return PasswordHasher()


async def get_auth_service(
    db: AsyncSession = Depends(get_tenant_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> AuthService:
    """Get AuthService instance.

    Args:
        db: Tenant database session.
        jwt_manager: JWT manager.

    Returns:
        AuthService.
    """
    return AuthService(db, jwt_manager)


def get_checkpointer() -> BaseCheckpointSaver | None:
    """Get the checkpointer instance for workflow state persistence.

    Returns the singleton checkpointer initialized at application startup.
    Returns None if checkpointer is not yet initialized.

    Returns:
        BaseCheckpointSaver instance or None.
    """
    return get_checkpointer_instance()


def get_tenant_db_manager() -> TenantDatabaseManager:
    """Get the tenant database manager singleton.

    Returns the TenantDatabaseManager initialized at application startup.
    This is needed for services that require direct access to the manager
    rather than just a session (e.g., MemoryManager).

    Returns:
        TenantDatabaseManager instance.

    Raises:
        HTTPException: If not initialized.
    """
    global _tenant_db_manager
    if _tenant_db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database manager not initialized",
        )
    return _tenant_db_manager


# =========================================================================
# Type Aliases for Cleaner Endpoint Signatures
# =========================================================================

CentralDB = Annotated[AsyncSession, Depends(get_central_db)]
TenantDB = Annotated[AsyncSession, Depends(get_tenant_db)]
OptionalUser = Annotated[CurrentUser | None, Depends(get_optional_user)]
AuthenticatedUser = Annotated[CurrentUser, Depends(require_auth)]
AdminUser = Annotated[CurrentUser, Depends(require_admin)]
TenantAdminUser = Annotated[CurrentUser, Depends(require_tenant_admin)]
TeacherUser = Annotated[CurrentUser, Depends(require_teacher)]
TeacherOrAdmin = Annotated[CurrentUser, Depends(require_teacher_or_admin)]
Tenant = Annotated[TenantContext, Depends(require_tenant)]
Checkpointer = Annotated[BaseCheckpointSaver | None, Depends(get_checkpointer)]
OptionalTenant = Annotated[TenantContext | None, Depends(get_tenant)]
TenantDBManager = Annotated[TenantDatabaseManager, Depends(get_tenant_db_manager)]
