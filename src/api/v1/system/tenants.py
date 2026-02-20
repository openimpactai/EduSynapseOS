# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant management endpoints.

This module provides CRUD endpoints for tenant lifecycle management:
- GET / - List tenants
- POST / - Create tenant
- GET /{id} - Get tenant
- PUT /{id} - Update tenant
- DELETE /{id} - Delete tenant
- POST /{id}/suspend - Suspend tenant
- POST /{id}/activate - Activate tenant
- GET /{id}/container - Get container status
- GET /{id}/credentials - List API credentials
- POST /{id}/credentials - Create API credential
- DELETE /{id}/credentials/{credential_id} - Revoke API credential
- POST /{id}/credentials/{credential_id}/regenerate - Regenerate API secret
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_central_db, get_jwt_manager, get_password_hasher
from src.domains.auth.api_key_service import APIKeyService
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.system.auth_service import SystemAuthService
from src.domains.system.tenant_service import (
    TenantService,
    TenantAlreadyExistsError,
    TenantNotFoundError,
    TenantProvisioningError,
)
from src.infrastructure.docker.tenant_container import ContainerStatus

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""

    code: str = Field(
        ...,
        min_length=2,
        max_length=50,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique tenant code (lowercase, alphanumeric with underscores)",
    )
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    admin_email: EmailStr = Field(..., description="Primary admin email")
    admin_name: str | None = Field(None, max_length=255, description="Primary admin name")
    tier: str = Field(
        default="standard",
        pattern=r"^(free|standard|premium|enterprise)$",
        description="Subscription tier",
    )
    license_id: str | None = Field(None, description="License ID to associate")
    settings: dict | None = Field(None, description="Tenant settings")


class UpdateTenantRequest(BaseModel):
    """Request to update tenant details."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    admin_email: EmailStr | None = Field(None, description="Primary admin email")
    admin_name: str | None = Field(None, max_length=255, description="Primary admin name")
    tier: str | None = Field(
        None,
        pattern=r"^(free|standard|premium|enterprise)$",
        description="Subscription tier",
    )
    settings: dict | None = Field(None, description="Tenant settings")


class TenantResponse(BaseModel):
    """Tenant details response."""

    id: str = Field(..., description="Tenant ID")
    code: str = Field(..., description="Tenant code")
    name: str = Field(..., description="Display name")
    status: str = Field(..., description="Current status")
    tier: str = Field(..., description="Subscription tier")
    admin_email: str = Field(..., description="Admin email")
    admin_name: str | None = Field(None, description="Admin name")
    db_host: str = Field(..., description="Database container hostname")
    db_port: int = Field(..., description="Database port")
    db_name: str = Field(..., description="Database name")
    created_at: str = Field(..., description="Creation timestamp")
    provisioned_at: str | None = Field(None, description="Provisioning timestamp")
    suspended_at: str | None = Field(None, description="Suspension timestamp")

    class Config:
        from_attributes = True


class TenantListResponse(BaseModel):
    """Paginated tenant list response."""

    items: list[TenantResponse] = Field(..., description="Tenant list")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class ContainerStatusResponse(BaseModel):
    """Tenant container status response."""

    tenant_id: str = Field(..., description="Tenant ID")
    container_name: str | None = Field(None, description="Container name")
    container_id: str | None = Field(None, description="Container ID")
    status: str = Field(..., description="Container status")
    host_port: int | None = Field(None, description="Host port mapping")


class CreateCredentialRequest(BaseModel):
    """Request to create an API credential for a tenant."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Credential name for identification",
    )
    allowed_ips: list[str] | None = Field(
        None,
        description="IP addresses allowed to use this credential (empty = all IPs)",
    )
    allowed_origins: list[str] | None = Field(
        None,
        description="Origins allowed for CORS (optional)",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute",
    )


class CredentialResponse(BaseModel):
    """API credential details (without secret)."""

    id: str = Field(..., description="Credential ID")
    tenant_id: str = Field(..., description="Tenant ID")
    name: str = Field(..., description="Credential name")
    api_key: str = Field(..., description="API key (tk_...)")
    status: str = Field(..., description="active or revoked")
    allowed_ips: list[str] | None = Field(None, description="Allowed IP addresses")
    allowed_origins: list[str] | None = Field(None, description="Allowed origins")
    rate_limit_per_minute: int = Field(..., description="Rate limit")
    last_used_at: str | None = Field(None, description="Last usage timestamp")
    usage_count: int = Field(..., description="Total usage count")
    created_at: str = Field(..., description="Creation timestamp")
    revoked_at: str | None = Field(None, description="Revocation timestamp")


class CredentialCreatedResponse(CredentialResponse):
    """Response when creating a credential (includes secret - shown only once)."""

    api_secret: str = Field(
        ...,
        description="API secret (ts_...) - SAVE THIS! Only shown once.",
    )


class CredentialListResponse(BaseModel):
    """List of API credentials for a tenant."""

    items: list[CredentialResponse] = Field(..., description="Credentials list")
    total: int = Field(..., description="Total count")


async def require_system_admin(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> None:
    """Require authenticated system admin.

    Args:
        request: HTTP request.
        db: Central database session.
        jwt_manager: JWT manager.
        password_hasher: Password hasher.

    Raises:
        HTTPException: If not authenticated as system admin.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ")[1]
    auth_service = SystemAuthService(db, jwt_manager, password_hasher)

    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_tenant_service(
    db: AsyncSession = Depends(get_central_db),
) -> TenantService:
    """Get TenantService instance."""
    return TenantService(db)


def _tenant_to_response(tenant) -> TenantResponse:
    """Convert Tenant model to response."""
    return TenantResponse(
        id=tenant.id,
        code=tenant.code,
        name=tenant.name,
        status=tenant.status,
        tier=tenant.tier,
        admin_email=tenant.admin_email,
        admin_name=tenant.admin_name,
        db_host=tenant.db_host,
        db_port=tenant.db_port,
        db_name=tenant.db_name,
        created_at=tenant.created_at.isoformat(),
        provisioned_at=tenant.provisioned_at.isoformat() if tenant.provisioned_at else None,
        suspended_at=tenant.suspended_at.isoformat() if tenant.suspended_at else None,
    )


@router.get(
    "",
    response_model=TenantListResponse,
    summary="List tenants",
    description="Get paginated list of tenants with optional filters.",
    dependencies=[Depends(require_system_admin)],
)
async def list_tenants(
    status: str | None = Query(None, description="Filter by status"),
    tier: str | None = Query(None, description="Filter by tier"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantListResponse:
    """List tenants with optional filters.

    Args:
        status: Filter by status.
        tier: Filter by tier.
        limit: Page size.
        offset: Page offset.
        tenant_service: Tenant service.

    Returns:
        TenantListResponse with paginated results.
    """
    tenants, total = await tenant_service.list_tenants(
        status=status,
        tier=tier,
        limit=limit,
        offset=offset,
    )

    return TenantListResponse(
        items=[_tenant_to_response(t) for t in tenants],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create tenant",
    description="Create a new tenant with Docker container provisioning.",
    dependencies=[Depends(require_system_admin)],
)
async def create_tenant(
    data: CreateTenantRequest,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Create a new tenant.

    This will:
    1. Create tenant record in Central DB
    2. Create PostgreSQL container for tenant
    3. Wait for container to become healthy
    4. Run database initialization script

    Args:
        data: Tenant creation request.
        tenant_service: Tenant service.

    Returns:
        TenantResponse with created tenant.

    Raises:
        HTTPException: If creation fails.
    """
    try:
        tenant = await tenant_service.create_tenant(
            code=data.code,
            name=data.name,
            admin_email=data.admin_email,
            admin_name=data.admin_name,
            tier=data.tier,
            license_id=data.license_id,
            settings=data.settings,
        )

        return _tenant_to_response(tenant)

    except TenantAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except TenantProvisioningError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Provisioning failed: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant",
    description="Get tenant details by ID.",
    dependencies=[Depends(require_system_admin)],
)
async def get_tenant(
    tenant_id: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Get tenant by ID.

    Args:
        tenant_id: Tenant identifier.
        tenant_service: Tenant service.

    Returns:
        TenantResponse with tenant details.

    Raises:
        HTTPException: If tenant not found.
    """
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    return _tenant_to_response(tenant)


@router.put(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Update tenant",
    description="Update tenant details.",
    dependencies=[Depends(require_system_admin)],
)
async def update_tenant(
    tenant_id: str,
    data: UpdateTenantRequest,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Update tenant details.

    Args:
        tenant_id: Tenant identifier.
        data: Update request.
        tenant_service: Tenant service.

    Returns:
        TenantResponse with updated tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await tenant_service.update_tenant(
            tenant_id=tenant_id,
            name=data.name,
            admin_email=data.admin_email,
            admin_name=data.admin_name,
            tier=data.tier,
            settings=data.settings,
        )

        return _tenant_to_response(tenant)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{tenant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete tenant",
    description="Soft delete a tenant. Use ?hard=true for permanent deletion.",
    dependencies=[Depends(require_system_admin)],
)
async def delete_tenant(
    tenant_id: str,
    hard: bool = Query(False, description="Permanent deletion with data removal"),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> None:
    """Delete a tenant.

    Args:
        tenant_id: Tenant identifier.
        hard: If True, permanently deletes tenant and data.
        tenant_service: Tenant service.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        await tenant_service.delete_tenant(tenant_id, hard_delete=hard)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/{tenant_id}/suspend",
    response_model=TenantResponse,
    summary="Suspend tenant",
    description="Suspend a tenant and stop its container.",
    dependencies=[Depends(require_system_admin)],
)
async def suspend_tenant(
    tenant_id: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Suspend a tenant.

    Args:
        tenant_id: Tenant identifier.
        tenant_service: Tenant service.

    Returns:
        TenantResponse with suspended tenant.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        tenant = await tenant_service.suspend_tenant(tenant_id)
        return _tenant_to_response(tenant)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/{tenant_id}/activate",
    response_model=TenantResponse,
    summary="Activate tenant",
    description="Activate a suspended tenant and start its container.",
    dependencies=[Depends(require_system_admin)],
)
async def activate_tenant(
    tenant_id: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Activate a suspended tenant.

    Args:
        tenant_id: Tenant identifier.
        tenant_service: Tenant service.

    Returns:
        TenantResponse with activated tenant.

    Raises:
        HTTPException: If tenant not found or activation fails.
    """
    try:
        tenant = await tenant_service.activate_tenant(tenant_id)
        return _tenant_to_response(tenant)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except TenantProvisioningError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get(
    "/{tenant_id}/container",
    response_model=ContainerStatusResponse,
    summary="Get container status",
    description="Get Docker container status for a tenant.",
    dependencies=[Depends(require_system_admin)],
)
async def get_container_status(
    tenant_id: str,
    tenant_service: TenantService = Depends(get_tenant_service),
) -> ContainerStatusResponse:
    """Get tenant container status.

    Args:
        tenant_id: Tenant identifier.
        tenant_service: Tenant service.

    Returns:
        ContainerStatusResponse with container details.

    Raises:
        HTTPException: If tenant not found.
    """
    try:
        container_info = await tenant_service.get_tenant_container_info(tenant_id)

        if container_info:
            return ContainerStatusResponse(
                tenant_id=tenant_id,
                container_name=container_info.container_name,
                container_id=container_info.container_id,
                status=container_info.status.value,
                host_port=container_info.host_port,
            )
        else:
            return ContainerStatusResponse(
                tenant_id=tenant_id,
                container_name=None,
                container_id=None,
                status=ContainerStatus.NOT_FOUND.value,
                host_port=None,
            )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ==============================================================================
# API Credential Management Endpoints
# ==============================================================================


def _credential_to_response(cred) -> CredentialResponse:
    """Convert TenantAPICredential model to response."""
    # Compute status from model flags
    if cred.is_revoked:
        status = "revoked"
    elif cred.is_active:
        status = "active"
    else:
        status = "inactive"

    return CredentialResponse(
        id=cred.id,
        tenant_id=cred.tenant_id,
        name=cred.name,
        api_key=cred.api_key,
        status=status,
        allowed_ips=cred.allowed_ips,
        allowed_origins=cred.allowed_origins,
        rate_limit_per_minute=cred.rate_limit_per_minute,
        last_used_at=cred.last_used_at.isoformat() if cred.last_used_at else None,
        usage_count=cred.usage_count,
        created_at=cred.created_at.isoformat(),
        revoked_at=cred.revoked_at.isoformat() if cred.revoked_at else None,
    )


@router.get(
    "/{tenant_id}/credentials",
    response_model=CredentialListResponse,
    summary="List API credentials",
    description="Get all API credentials for a tenant.",
    dependencies=[Depends(require_system_admin)],
)
async def list_credentials(
    tenant_id: str,
    include_revoked: bool = Query(False, description="Include revoked credentials"),
    db: AsyncSession = Depends(get_central_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> CredentialListResponse:
    """List API credentials for a tenant.

    Args:
        tenant_id: Tenant identifier.
        include_revoked: Whether to include revoked credentials.
        db: Central database session.
        tenant_service: Tenant service.

    Returns:
        CredentialListResponse with credentials list.

    Raises:
        HTTPException: If tenant not found.
    """
    # Verify tenant exists
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    api_key_service = APIKeyService(db)
    credentials = await api_key_service.list_credentials(
        tenant_id=tenant_id,
        include_revoked=include_revoked,
    )

    return CredentialListResponse(
        items=[_credential_to_response(c) for c in credentials],
        total=len(credentials),
    )


@router.post(
    "/{tenant_id}/credentials",
    response_model=CredentialCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API credential",
    description="Create a new API credential for a tenant. The secret is only returned once!",
    dependencies=[Depends(require_system_admin)],
)
async def create_credential(
    tenant_id: str,
    data: CreateCredentialRequest,
    db: AsyncSession = Depends(get_central_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> CredentialCreatedResponse:
    """Create a new API credential.

    IMPORTANT: The API secret is only returned in this response.
    It cannot be retrieved later - store it securely!

    Args:
        tenant_id: Tenant identifier.
        data: Credential creation request.
        db: Central database session.
        tenant_service: Tenant service.

    Returns:
        CredentialCreatedResponse with credential and secret.

    Raises:
        HTTPException: If tenant not found.
    """
    # Verify tenant exists
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    api_key_service = APIKeyService(db)
    result = await api_key_service.create_credential(
        tenant_id=tenant_id,
        name=data.name,
        allowed_ips=data.allowed_ips,
        allowed_origins=data.allowed_origins,
        rate_limit_per_minute=data.rate_limit_per_minute,
    )

    logger.info(
        "Created API credential for tenant %s: %s (key=%s)",
        tenant_id,
        data.name,
        result.api_key,
    )

    # New credentials are always active
    return CredentialCreatedResponse(
        id=result.credential.id,
        tenant_id=result.credential.tenant_id,
        name=result.credential.name,
        api_key=result.api_key,
        api_secret=result.api_secret,
        status="active",
        allowed_ips=result.credential.allowed_ips,
        allowed_origins=result.credential.allowed_origins,
        rate_limit_per_minute=result.credential.rate_limit_per_minute,
        last_used_at=None,
        usage_count=0,
        created_at=result.credential.created_at.isoformat(),
        revoked_at=None,
    )


@router.delete(
    "/{tenant_id}/credentials/{credential_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API credential",
    description="Revoke an API credential. This is permanent - the credential cannot be reactivated.",
    dependencies=[Depends(require_system_admin)],
)
async def revoke_credential(
    tenant_id: str,
    credential_id: str,
    db: AsyncSession = Depends(get_central_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> None:
    """Revoke an API credential.

    Args:
        tenant_id: Tenant identifier.
        credential_id: Credential identifier.
        db: Central database session.
        tenant_service: Tenant service.

    Raises:
        HTTPException: If tenant or credential not found.
    """
    # Verify tenant exists
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    api_key_service = APIKeyService(db)
    try:
        await api_key_service.revoke_credential(
            credential_id=credential_id,
            reason="Revoked via admin API",
        )
        logger.info("Revoked API credential: %s for tenant %s", credential_id, tenant_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/{tenant_id}/credentials/{credential_id}/regenerate",
    response_model=CredentialCreatedResponse,
    summary="Regenerate API secret",
    description="Generate a new secret for an API credential. The old secret is invalidated.",
    dependencies=[Depends(require_system_admin)],
)
async def regenerate_secret(
    tenant_id: str,
    credential_id: str,
    db: AsyncSession = Depends(get_central_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> CredentialCreatedResponse:
    """Regenerate the API secret for a credential.

    The old secret is immediately invalidated and a new one is generated.
    IMPORTANT: The new secret is only returned once!

    Args:
        tenant_id: Tenant identifier.
        credential_id: Credential identifier.
        db: Central database session.
        tenant_service: Tenant service.

    Returns:
        CredentialCreatedResponse with new secret.

    Raises:
        HTTPException: If tenant or credential not found, or credential is revoked.
    """
    # Verify tenant exists
    tenant = await tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant not found: {tenant_id}",
        )

    api_key_service = APIKeyService(db)
    try:
        result = await api_key_service.regenerate_secret(credential_id)
        logger.info("Regenerated secret for credential: %s", credential_id)

        # Compute status from model flags
        cred = result.credential
        if cred.is_revoked:
            cred_status = "revoked"
        elif cred.is_active:
            cred_status = "active"
        else:
            cred_status = "inactive"

        return CredentialCreatedResponse(
            id=cred.id,
            tenant_id=cred.tenant_id,
            name=cred.name,
            api_key=cred.api_key,
            api_secret=result.api_secret,
            status=cred_status,
            allowed_ips=cred.allowed_ips,
            allowed_origins=cred.allowed_origins,
            rate_limit_per_minute=cred.rate_limit_per_minute,
            last_used_at=cred.last_used_at.isoformat() if cred.last_used_at else None,
            usage_count=cred.usage_count,
            created_at=cred.created_at.isoformat(),
            revoked_at=cred.revoked_at.isoformat() if cred.revoked_at else None,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
