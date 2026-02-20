# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant admin API endpoints.

This module provides endpoints for tenant administrators:
- GET /overview - Get tenant overview statistics
- GET /credentials - List API credentials
- POST /credentials - Create API credential
- DELETE /credentials/{id} - Revoke API credential
- POST /credentials/{id}/rotate - Rotate API credential secret
- GET /profile - Get tenant profile
- GET /settings - Get tenant settings
- PUT /settings - Update tenant settings

Example:
    GET /api/v1/tenant/overview
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    get_central_db,
    require_tenant_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.infrastructure.database.models.tenant.user import User
from src.infrastructure.database.models.tenant.school import School, Class
from src.infrastructure.database.models.tenant.practice import PracticeSession
from src.infrastructure.database.models.tenant.learning import LearningSession
from src.infrastructure.database.models.central.tenant import Tenant
from src.infrastructure.database.models.central.api_credential import TenantAPICredential
from src.domains.auth.api_key_service import APIKeyService

logger = logging.getLogger(__name__)

router = APIRouter()


class TenantOverviewStats(BaseModel):
    """Tenant overview statistics."""

    total_students: int = Field(description="Total number of students")
    total_teachers: int = Field(description="Total number of teachers")
    total_schools: int = Field(description="Total number of schools")
    total_classes: int = Field(description="Total number of classes")
    sessions_today: int = Field(description="Practice and learning sessions today")
    generated_at: datetime = Field(description="When stats were generated")


@router.get(
    "/overview",
    response_model=TenantOverviewStats,
    summary="Get tenant overview",
    description="Get overview statistics for the tenant dashboard.",
)
async def get_tenant_overview(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TenantOverviewStats:
    """Get tenant overview statistics.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        TenantOverviewStats with counts and session stats.
    """
    logger.info("Getting tenant overview for: %s", tenant.code)

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Count students
    students_result = await db.execute(
        select(func.count(User.id)).where(
            User.user_type == "student",
            User.status == "active",
            User.deleted_at.is_(None),
        )
    )
    total_students = students_result.scalar() or 0

    # Count teachers
    teachers_result = await db.execute(
        select(func.count(User.id)).where(
            User.user_type == "teacher",
            User.status == "active",
            User.deleted_at.is_(None),
        )
    )
    total_teachers = teachers_result.scalar() or 0

    # Count schools
    schools_result = await db.execute(
        select(func.count(School.id)).where(
            School.is_active == True,
            School.deleted_at.is_(None),
        )
    )
    total_schools = schools_result.scalar() or 0

    # Count classes
    classes_result = await db.execute(
        select(func.count(Class.id)).where(
            Class.is_active == True,
        )
    )
    total_classes = classes_result.scalar() or 0

    # Count sessions today (practice + learning)
    practice_today_result = await db.execute(
        select(func.count(PracticeSession.id)).where(
            PracticeSession.started_at >= today_start,
        )
    )
    practice_today = practice_today_result.scalar() or 0

    learning_today_result = await db.execute(
        select(func.count(LearningSession.id)).where(
            LearningSession.started_at >= today_start,
        )
    )
    learning_today = learning_today_result.scalar() or 0

    sessions_today = practice_today + learning_today

    return TenantOverviewStats(
        total_students=total_students,
        total_teachers=total_teachers,
        total_schools=total_schools,
        total_classes=total_classes,
        sessions_today=sessions_today,
        generated_at=now,
    )


# ============================================================================
# API Credentials Endpoints
# ============================================================================


class CredentialResponse(BaseModel):
    """API credential response."""

    id: str = Field(description="Credential ID")
    name: str = Field(description="Credential name")
    api_key: str = Field(description="API key (public)")
    status: str = Field(description="Credential status")
    last_used_at: datetime | None = Field(description="Last used timestamp")
    usage_count: int = Field(description="Total usage count")
    created_at: datetime = Field(description="Created timestamp")


class CredentialWithSecretResponse(CredentialResponse):
    """API credential with secret (only shown on create/rotate)."""

    api_secret: str = Field(description="API secret (only shown once)")


class CreateCredentialRequest(BaseModel):
    """Request to create new API credential."""

    name: str = Field(min_length=1, max_length=100, description="Credential name")


def _credential_to_response(cred: TenantAPICredential) -> CredentialResponse:
    """Convert credential model to response."""
    status_value = "active" if cred.is_active and not cred.is_revoked else "revoked"
    return CredentialResponse(
        id=str(cred.id),
        name=cred.name,
        api_key=cred.api_key,
        status=status_value,
        last_used_at=cred.last_used_at,
        usage_count=cred.usage_count,
        created_at=cred.created_at,
    )


@router.get(
    "/credentials",
    response_model=list[CredentialResponse],
    summary="List API credentials",
    description="List all API credentials for the tenant.",
)
async def list_credentials(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> list[CredentialResponse]:
    """List all API credentials for the tenant.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        List of API credentials.
    """
    logger.info("Listing credentials for tenant: %s", tenant.code)

    service = APIKeyService(central_db)
    credentials = await service.list_credentials(tenant.id, include_revoked=False)

    return [_credential_to_response(cred) for cred in credentials]


@router.post(
    "/credentials",
    response_model=CredentialWithSecretResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API credential",
    description="Create a new API credential. The secret is only shown once.",
)
async def create_credential(
    data: CreateCredentialRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> CredentialWithSecretResponse:
    """Create a new API credential.

    Args:
        data: Credential creation request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Created credential with secret (only shown once).
    """
    logger.info(
        "Creating credential for tenant: %s, name: %s, by: %s",
        tenant.code,
        data.name,
        current_user.id,
    )

    service = APIKeyService(central_db)

    try:
        result = await service.create_credential(
            tenant_id=tenant.id,
            name=data.name,
            created_by_id=None,  # Tenant admin, not system user
        )
        await central_db.commit()

        cred = result.credential
        return CredentialWithSecretResponse(
            id=str(cred.id),
            name=cred.name,
            api_key=result.api_key,
            api_secret=result.api_secret,
            status="active",
            last_used_at=None,
            usage_count=0,
            created_at=cred.created_at,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/credentials/{credential_id}",
    status_code=status.HTTP_200_OK,
    summary="Revoke API credential",
    description="Revoke an API credential. This action cannot be undone.",
)
async def revoke_credential(
    credential_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> dict:
    """Revoke an API credential.

    Args:
        credential_id: Credential ID to revoke.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Confirmation message.
    """
    logger.info(
        "Revoking credential %s for tenant: %s, by: %s",
        credential_id,
        tenant.code,
        current_user.id,
    )

    service = APIKeyService(central_db)

    # Verify credential belongs to this tenant
    credential = await service.get_credential_by_id(credential_id)
    if not credential or credential.tenant_id != tenant.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found",
        )

    result = await service.revoke_credential(
        credential_id=credential_id,
        reason="Revoked by tenant admin",
    )
    await central_db.commit()

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found",
        )

    return {"status": "revoked", "credential_id": str(credential_id)}


@router.post(
    "/credentials/{credential_id}/rotate",
    response_model=CredentialWithSecretResponse,
    summary="Rotate API credential secret",
    description="Generate a new secret for the credential. Old secret becomes invalid.",
)
async def rotate_credential(
    credential_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> CredentialWithSecretResponse:
    """Rotate the API secret for a credential.

    Args:
        credential_id: Credential ID to rotate.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Credential with new secret (only shown once).
    """
    logger.info(
        "Rotating credential %s for tenant: %s, by: %s",
        credential_id,
        tenant.code,
        current_user.id,
    )

    service = APIKeyService(central_db)

    # Verify credential belongs to this tenant
    credential = await service.get_credential_by_id(credential_id)
    if not credential or credential.tenant_id != tenant.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found",
        )

    if credential.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot rotate revoked credential",
        )

    try:
        result = await service.regenerate_secret(credential_id)
        await central_db.commit()

        cred = result.credential
        return CredentialWithSecretResponse(
            id=str(cred.id),
            name=cred.name,
            api_key=result.api_key,
            api_secret=result.api_secret,
            status="active" if cred.is_active else "inactive",
            last_used_at=cred.last_used_at,
            usage_count=cred.usage_count,
            created_at=cred.created_at,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ============================================================================
# Tenant Profile and Settings Endpoints
# ============================================================================


# Feature tiers configuration
TIER_FEATURES = {
    "free": {
        "ai_tutoring": True,
        "practice_mode": True,
        "adaptive_assessment": False,
        "diagnostics": False,
        "parent_dashboard": False,
        "teacher_tools": False,
        "content_generation": False,
        "emotional_detection": False,
    },
    "standard": {
        "ai_tutoring": True,
        "practice_mode": True,
        "adaptive_assessment": True,
        "diagnostics": False,
        "parent_dashboard": True,
        "teacher_tools": True,
        "content_generation": False,
        "emotional_detection": False,
    },
    "premium": {
        "ai_tutoring": True,
        "practice_mode": True,
        "adaptive_assessment": True,
        "diagnostics": True,
        "parent_dashboard": True,
        "teacher_tools": True,
        "content_generation": True,
        "emotional_detection": False,
    },
    "enterprise": {
        "ai_tutoring": True,
        "practice_mode": True,
        "adaptive_assessment": True,
        "diagnostics": True,
        "parent_dashboard": True,
        "teacher_tools": True,
        "content_generation": True,
        "emotional_detection": True,
    },
}


def get_tier_features(tier: str) -> dict[str, bool]:
    """Get features available for a tier."""
    return TIER_FEATURES.get(tier, TIER_FEATURES["free"])


class TenantProfileResponse(BaseModel):
    """Tenant profile information."""

    id: str = Field(description="Tenant ID")
    code: str = Field(description="Tenant code")
    name: str = Field(description="Tenant name")
    status: str = Field(description="Tenant status")
    tier: str = Field(description="Subscription tier")
    admin_email: str | None = Field(description="Admin email")
    admin_name: str | None = Field(description="Admin name")
    created_at: datetime = Field(description="Created timestamp")
    features: dict[str, bool] = Field(description="Available features")


class TenantSettingsResponse(BaseModel):
    """Tenant settings."""

    organization_name: str = Field(description="Organization name")
    admin_email: str | None = Field(description="Admin email")
    timezone: str = Field(description="Default timezone")
    language: str = Field(description="Default language")
    features: dict[str, bool] = Field(description="Available features")


class UpdateSettingsRequest(BaseModel):
    """Update settings request."""

    organization_name: str | None = Field(default=None, max_length=255)
    admin_email: str | None = Field(default=None, max_length=255)
    timezone: str | None = Field(default=None, max_length=50)
    language: str | None = Field(default=None, max_length=10)


@router.get(
    "/profile",
    response_model=TenantProfileResponse,
    summary="Get tenant profile",
    description="Get tenant profile information.",
)
async def get_tenant_profile(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> TenantProfileResponse:
    """Get tenant profile information.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Tenant profile with features.
    """
    logger.info("Getting profile for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    features = get_tier_features(tenant_record.tier)

    return TenantProfileResponse(
        id=str(tenant_record.id),
        code=tenant_record.code,
        name=tenant_record.name,
        status=tenant_record.status,
        tier=tenant_record.tier,
        admin_email=tenant_record.admin_email,
        admin_name=tenant_record.admin_name,
        created_at=tenant_record.created_at,
        features=features,
    )


@router.get(
    "/settings",
    response_model=TenantSettingsResponse,
    summary="Get tenant settings",
    description="Get tenant settings.",
)
async def get_tenant_settings(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> TenantSettingsResponse:
    """Get tenant settings.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Tenant settings.
    """
    logger.info("Getting settings for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Get settings from tenant settings JSON field
    settings = tenant_record.settings or {}
    features = get_tier_features(tenant_record.tier)

    return TenantSettingsResponse(
        organization_name=tenant_record.name,
        admin_email=tenant_record.admin_email,
        timezone=settings.get("timezone", "Europe/Istanbul"),
        language=settings.get("language", "en"),
        features=features,
    )


@router.put(
    "/settings",
    response_model=TenantSettingsResponse,
    summary="Update tenant settings",
    description="Update tenant settings.",
)
async def update_tenant_settings(
    data: UpdateSettingsRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> TenantSettingsResponse:
    """Update tenant settings.

    Args:
        data: Settings update request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Updated tenant settings.
    """
    logger.info("Updating settings for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Update fields if provided
    if data.organization_name is not None:
        tenant_record.name = data.organization_name
    if data.admin_email is not None:
        tenant_record.admin_email = data.admin_email

    # Update settings JSON
    settings = tenant_record.settings or {}
    if data.timezone is not None:
        settings["timezone"] = data.timezone
    if data.language is not None:
        settings["language"] = data.language
    tenant_record.settings = settings

    await central_db.commit()
    await central_db.refresh(tenant_record)

    features = get_tier_features(tenant_record.tier)

    return TenantSettingsResponse(
        organization_name=tenant_record.name,
        admin_email=tenant_record.admin_email,
        timezone=settings.get("timezone", "Europe/Istanbul"),
        language=settings.get("language", "en"),
        features=features,
    )


# ============================================================================
# Central Curriculum Credentials Endpoints
# ============================================================================


class CCCredentialsStatus(BaseModel):
    """Central Curriculum credentials status (without exposing secrets)."""

    is_configured: bool = Field(description="Whether CC credentials are configured")
    api_key: str | None = Field(default=None, description="API key (masked)")
    last_sync_at: datetime | None = Field(default=None, description="Last sync timestamp")


class CCCredentialsRequest(BaseModel):
    """Request to set CC credentials."""

    api_key: str = Field(min_length=10, max_length=100, description="CC API key")
    api_secret: str = Field(min_length=10, max_length=100, description="CC API secret")


@router.get(
    "/cc-credentials",
    response_model=CCCredentialsStatus,
    summary="Get CC credentials status",
    description="Check if Central Curriculum credentials are configured.",
)
async def get_cc_credentials_status(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> CCCredentialsStatus:
    """Get CC credentials status.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        CC credentials status (whether configured, masked key).
    """
    logger.info("Getting CC credentials status for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    settings = tenant_record.settings or {}
    cc_settings = settings.get("central_curriculum", {})

    api_key = cc_settings.get("api_key")
    is_configured = bool(api_key and cc_settings.get("api_secret"))

    # Mask the API key for display
    masked_key = None
    if api_key:
        masked_key = api_key[:12] + "..." + api_key[-4:] if len(api_key) > 20 else api_key[:8] + "..."

    return CCCredentialsStatus(
        is_configured=is_configured,
        api_key=masked_key,
        last_sync_at=cc_settings.get("last_sync_at"),
    )


@router.put(
    "/cc-credentials",
    response_model=CCCredentialsStatus,
    summary="Set CC credentials",
    description="Configure Central Curriculum API credentials for this tenant.",
)
async def set_cc_credentials(
    data: CCCredentialsRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> CCCredentialsStatus:
    """Set CC credentials for the tenant.

    Args:
        data: CC credentials to set.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Updated CC credentials status.
    """
    logger.info("Setting CC credentials for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Update settings with CC credentials
    settings = tenant_record.settings or {}
    settings["central_curriculum"] = {
        "api_key": data.api_key,
        "api_secret": data.api_secret,
    }
    tenant_record.settings = settings

    await central_db.commit()
    await central_db.refresh(tenant_record)

    logger.info("CC credentials set for tenant: %s", tenant.code)

    # Return masked status
    masked_key = data.api_key[:12] + "..." + data.api_key[-4:] if len(data.api_key) > 20 else data.api_key[:8] + "..."

    return CCCredentialsStatus(
        is_configured=True,
        api_key=masked_key,
        last_sync_at=None,
    )


@router.delete(
    "/cc-credentials",
    summary="Remove CC credentials",
    description="Remove Central Curriculum API credentials for this tenant.",
)
async def delete_cc_credentials(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    central_db: AsyncSession = Depends(get_central_db),
) -> dict:
    """Remove CC credentials for the tenant.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        central_db: Central database session.

    Returns:
        Confirmation message.
    """
    logger.info("Removing CC credentials for tenant: %s", tenant.code)

    result = await central_db.execute(
        select(Tenant).where(Tenant.code == tenant.code)
    )
    tenant_record = result.scalar_one_or_none()

    if not tenant_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Remove CC credentials from settings
    settings = tenant_record.settings or {}
    if "central_curriculum" in settings:
        del settings["central_curriculum"]
        tenant_record.settings = settings

    await central_db.commit()

    logger.info("CC credentials removed for tenant: %s", tenant.code)

    return {"status": "removed", "message": "CC credentials removed successfully"}
