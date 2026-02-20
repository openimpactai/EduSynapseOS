# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Curriculum API endpoints.

This module provides endpoints for curriculum data access and management:
- GET /frameworks - List all curriculum frameworks
- GET /frameworks/{code} - Get a specific framework
- GET /frameworks/{code}/stages - List stages in a framework
- GET /frameworks/{code}/grades - List all grades in a framework
- GET /frameworks/{code}/subjects - List subjects in a framework
- GET /frameworks/{code}/subjects/{subject}/grades/{grade}/units - List units
- GET /frameworks/{code}/subjects/{subject}/grades/{grade}/units/{unit}/topics - List topics
- POST /sync - Trigger manual curriculum sync (admin only)

Curriculum Hierarchy:
    Framework -> Stage -> Grade (independent of subject)
    Framework -> Subject -> Units (per grade) -> Topics

Curriculum data is synced from the Central Curriculum service.
This endpoint provides read-only access to the synced data.

Authentication:
    Requires authenticated user and tenant context.

Example:
    GET /api/v1/curriculum/frameworks
    Headers:
        Authorization: Bearer <token>
        X-Tenant-Code: test_school
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_tenant_db, require_auth, require_tenant_admin
from src.api.middleware.auth import CurrentUser
from src.infrastructure.database.models.tenant.curriculum import (
    CurriculumFramework,
    CurriculumStage,
    GradeLevel,
    Subject,
    Unit,
    Topic,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Response DTOs
# =============================================================================


class CurriculumFrameworkResponse(BaseModel):
    """Curriculum framework response."""

    code: str = Field(description="Framework unique code")
    name: str = Field(description="Framework display name")
    description: str | None = Field(default=None, description="Framework description")
    framework_type: str = Field(description="Type: national, regional, international")
    country_code: str | None = Field(default=None, description="ISO country code")
    organization: str | None = Field(default=None, description="Publishing organization")
    version: str | None = Field(default=None, description="Framework version")
    language: str = Field(default="en", description="Primary language")
    is_active: bool = Field(description="Whether framework is active")


class CurriculumStageResponse(BaseModel):
    """Curriculum stage response."""

    framework_code: str = Field(description="Parent framework code")
    code: str = Field(description="Stage unique code")
    name: str = Field(description="Stage display name")
    description: str | None = Field(default=None, description="Stage description")
    age_start: int | None = Field(default=None, description="Starting age")
    age_end: int | None = Field(default=None, description="Ending age")
    sequence: int = Field(description="Display order")


class GradeLevelResponse(BaseModel):
    """Grade level response."""

    full_code: str = Field(description="Full composite code (framework.stage.grade)")
    framework_code: str = Field(description="Framework code")
    stage_code: str = Field(description="Stage code")
    code: str = Field(description="Grade code")
    name: str = Field(description="Grade display name")
    typical_age: int | None = Field(default=None, description="Typical student age")
    sequence: int = Field(description="Display order")


class SubjectResponse(BaseModel):
    """Subject response."""

    full_code: str = Field(description="Full composite code (framework.subject)")
    framework_code: str = Field(description="Framework code")
    code: str = Field(description="Subject code")
    name: str = Field(description="Subject display name")
    description: str | None = Field(default=None, description="Subject description")
    icon: str | None = Field(default=None, description="Icon identifier")
    color: str | None = Field(default=None, description="Display color")
    is_core: bool = Field(description="Whether subject is core")
    sequence: int = Field(description="Display order")


class FrameworkListResponse(BaseModel):
    """List of frameworks response."""

    frameworks: list[CurriculumFrameworkResponse]
    total: int = Field(description="Total number of frameworks")


class StageListResponse(BaseModel):
    """List of stages response."""

    stages: list[CurriculumStageResponse]
    total: int = Field(description="Total number of stages")


class GradeListResponse(BaseModel):
    """List of grades response."""

    grades: list[GradeLevelResponse]
    total: int = Field(description="Total number of grades")


class SubjectListResponse(BaseModel):
    """List of subjects response."""

    subjects: list[SubjectResponse]
    total: int = Field(description="Total number of subjects")


class UnitResponse(BaseModel):
    """Unit response."""

    full_code: str = Field(description="Full composite code (framework.subject.grade.unit)")
    framework_code: str = Field(description="Framework code")
    subject_code: str = Field(description="Subject code")
    grade_code: str = Field(description="Grade code")
    code: str = Field(description="Unit code")
    name: str = Field(description="Unit display name")
    description: str | None = Field(default=None, description="Unit description")
    estimated_hours: int | None = Field(default=None, description="Estimated teaching hours")
    sequence: int = Field(description="Display order")


class UnitListResponse(BaseModel):
    """List of units response."""

    units: list[UnitResponse]
    total: int = Field(description="Total number of units")


class TopicResponse(BaseModel):
    """Topic response."""

    full_code: str = Field(description="Full composite code (framework.subject.grade.unit.topic)")
    framework_code: str = Field(description="Framework code")
    subject_code: str = Field(description="Subject code")
    grade_code: str = Field(description="Grade code")
    unit_code: str = Field(description="Unit code")
    code: str = Field(description="Topic code")
    name: str = Field(description="Topic display name")
    description: str | None = Field(default=None, description="Topic description")
    base_difficulty: float = Field(description="Base difficulty (0.0-1.0)")
    estimated_minutes: int | None = Field(default=None, description="Estimated learning time")
    sequence: int = Field(description="Display order")


class TopicListResponse(BaseModel):
    """List of topics response."""

    topics: list[TopicResponse]
    total: int = Field(description="Total number of topics")


class SyncTriggerResponse(BaseModel):
    """Response for sync trigger."""

    status: str = Field(description="Sync status: scheduled, disabled")
    message: str = Field(description="Status message")


class SSOUrlResponse(BaseModel):
    """Response containing SSO URL for Central Curriculum Portal."""

    url: str = Field(description="SSO URL to open in new tab")
    expires_in: int = Field(description="Token expiration in seconds")


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/frameworks",
    response_model=FrameworkListResponse,
    summary="List curriculum frameworks",
    description="Get all available curriculum frameworks in the tenant database.",
)
async def list_frameworks(
    active_only: bool = Query(True, description="Only return active frameworks"),
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> FrameworkListResponse:
    """List all curriculum frameworks.

    Args:
        active_only: Whether to only return active frameworks.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of curriculum frameworks.
    """
    query = select(CurriculumFramework)
    if active_only:
        query = query.where(CurriculumFramework.is_active.is_(True))
    query = query.order_by(CurriculumFramework.name)

    result = await db.execute(query)
    frameworks = result.scalars().all()

    return FrameworkListResponse(
        frameworks=[
            CurriculumFrameworkResponse(
                code=fw.code,
                name=fw.name,
                description=fw.description,
                framework_type=fw.framework_type,
                country_code=fw.country_code,
                organization=fw.organization,
                version=fw.version,
                language=fw.language,
                is_active=fw.is_active,
            )
            for fw in frameworks
        ],
        total=len(frameworks),
    )


@router.get(
    "/frameworks/{framework_code}",
    response_model=CurriculumFrameworkResponse,
    summary="Get curriculum framework",
    description="Get a specific curriculum framework by code.",
)
async def get_framework(
    framework_code: str,
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> CurriculumFrameworkResponse:
    """Get a specific curriculum framework.

    Args:
        framework_code: Framework code to retrieve.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        Curriculum framework details.

    Raises:
        HTTPException: If framework not found.
    """
    result = await db.execute(
        select(CurriculumFramework).where(CurriculumFramework.code == framework_code)
    )
    fw = result.scalar_one_or_none()

    if not fw:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Framework not found: {framework_code}",
        )

    return CurriculumFrameworkResponse(
        code=fw.code,
        name=fw.name,
        description=fw.description,
        framework_type=fw.framework_type,
        country_code=fw.country_code,
        organization=fw.organization,
        version=fw.version,
        language=fw.language,
        is_active=fw.is_active,
    )


@router.get(
    "/frameworks/{framework_code}/stages",
    response_model=StageListResponse,
    summary="List framework stages",
    description="Get all stages within a curriculum framework.",
)
async def list_stages(
    framework_code: str,
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> StageListResponse:
    """List stages in a framework.

    Args:
        framework_code: Framework code.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of stages in the framework.
    """
    result = await db.execute(
        select(CurriculumStage)
        .where(CurriculumStage.framework_code == framework_code)
        .order_by(CurriculumStage.sequence)
    )
    stages = result.scalars().all()

    return StageListResponse(
        stages=[
            CurriculumStageResponse(
                framework_code=s.framework_code,
                code=s.code,
                name=s.name,
                description=s.description,
                age_start=s.age_start,
                age_end=s.age_end,
                sequence=s.sequence,
            )
            for s in stages
        ],
        total=len(stages),
    )


@router.get(
    "/frameworks/{framework_code}/grades",
    response_model=GradeListResponse,
    summary="List framework grades",
    description="Get all grade levels within a curriculum framework.",
)
async def list_grades(
    framework_code: str,
    stage_code: str | None = Query(None, description="Filter by stage code"),
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> GradeListResponse:
    """List grade levels in a framework.

    Args:
        framework_code: Framework code.
        stage_code: Optional stage code filter.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of grade levels in the framework.
    """
    query = select(GradeLevel).where(GradeLevel.framework_code == framework_code)
    if stage_code:
        query = query.where(GradeLevel.stage_code == stage_code)
    query = query.order_by(GradeLevel.stage_code, GradeLevel.sequence)

    result = await db.execute(query)
    grades = result.scalars().all()

    return GradeListResponse(
        grades=[
            GradeLevelResponse(
                full_code=f"{g.framework_code}.{g.stage_code}.{g.code}",
                framework_code=g.framework_code,
                stage_code=g.stage_code,
                code=g.code,
                name=g.name,
                typical_age=g.typical_age,
                sequence=g.sequence,
            )
            for g in grades
        ],
        total=len(grades),
    )


@router.get(
    "/frameworks/{framework_code}/subjects",
    response_model=SubjectListResponse,
    summary="List framework subjects",
    description="""Get subjects within a curriculum framework.

    When grade_code is provided, only returns subjects that have units for that grade.
    This is useful for showing only relevant subjects for a student's grade level.
    """,
)
async def list_subjects(
    framework_code: str,
    grade_code: str | None = Query(None, description="Filter by grade code (only subjects with units for this grade)"),
    core_only: bool = Query(False, description="Only return core subjects"),
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> SubjectListResponse:
    """List subjects in a framework.

    Args:
        framework_code: Framework code.
        grade_code: Optional grade code filter. When provided, only returns subjects
            that have units for this grade level.
        core_only: Whether to only return core subjects.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of subjects in the framework (optionally filtered by grade).
    """
    if grade_code:
        # Filter subjects that have units for this grade
        # Join with Unit to find subjects that have content for this grade
        query = (
            select(Subject)
            .join(
                Unit,
                (Subject.framework_code == Unit.framework_code)
                & (Subject.code == Unit.subject_code),
            )
            .where(
                Subject.framework_code == framework_code,
                Unit.grade_code == grade_code,
            )
            .distinct()
        )
    else:
        # Return all subjects for the framework
        query = select(Subject).where(Subject.framework_code == framework_code)

    if core_only:
        query = query.where(Subject.is_core.is_(True))
    query = query.order_by(Subject.sequence)

    result = await db.execute(query)
    subjects = result.scalars().all()

    return SubjectListResponse(
        subjects=[
            SubjectResponse(
                full_code=f"{s.framework_code}.{s.code}",
                framework_code=s.framework_code,
                code=s.code,
                name=s.name,
                description=s.description,
                icon=s.icon,
                color=s.color,
                is_core=s.is_core,
                sequence=s.sequence,
            )
            for s in subjects
        ],
        total=len(subjects),
    )


@router.get(
    "/frameworks/{framework_code}/subjects/{subject_code}/grades/{grade_code}/units",
    response_model=UnitListResponse,
    summary="List units for a subject and grade",
    description="Get all units for a specific subject and grade within a framework.",
)
async def list_units(
    framework_code: str,
    subject_code: str,
    grade_code: str,
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> UnitListResponse:
    """List units for a subject and grade.

    Args:
        framework_code: Framework code.
        subject_code: Subject code.
        grade_code: Grade code.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of units for the subject and grade.
    """
    query = (
        select(Unit)
        .where(
            Unit.framework_code == framework_code,
            Unit.subject_code == subject_code,
            Unit.grade_code == grade_code,
        )
        .order_by(Unit.sequence)
    )

    result = await db.execute(query)
    units = result.scalars().all()

    return UnitListResponse(
        units=[
            UnitResponse(
                full_code=f"{u.framework_code}.{u.subject_code}.{u.grade_code}.{u.code}",
                framework_code=u.framework_code,
                subject_code=u.subject_code,
                grade_code=u.grade_code,
                code=u.code,
                name=u.name,
                description=u.description,
                estimated_hours=u.estimated_hours,
                sequence=u.sequence,
            )
            for u in units
        ],
        total=len(units),
    )


@router.get(
    "/frameworks/{framework_code}/subjects/{subject_code}/grades/{grade_code}/units/{unit_code}/topics",
    response_model=TopicListResponse,
    summary="List topics for a unit",
    description="Get all topics for a specific unit within a framework, subject, and grade.",
)
async def list_topics(
    framework_code: str,
    subject_code: str,
    grade_code: str,
    unit_code: str,
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> TopicListResponse:
    """List topics for a unit.

    Args:
        framework_code: Framework code.
        subject_code: Subject code.
        grade_code: Grade code.
        unit_code: Unit code.
        db: Tenant database session.
        current_user: Authenticated user.

    Returns:
        List of topics for the unit.
    """
    query = (
        select(Topic)
        .where(
            Topic.framework_code == framework_code,
            Topic.subject_code == subject_code,
            Topic.grade_code == grade_code,
            Topic.unit_code == unit_code,
        )
        .order_by(Topic.sequence)
    )

    result = await db.execute(query)
    topics = result.scalars().all()

    return TopicListResponse(
        topics=[
            TopicResponse(
                full_code=f"{t.framework_code}.{t.subject_code}.{t.grade_code}.{t.unit_code}.{t.code}",
                framework_code=t.framework_code,
                subject_code=t.subject_code,
                grade_code=t.grade_code,
                unit_code=t.unit_code,
                code=t.code,
                name=t.name,
                description=t.description,
                base_difficulty=float(t.base_difficulty) if t.base_difficulty else 0.5,
                estimated_minutes=t.estimated_minutes,
                sequence=t.sequence,
            )
            for t in topics
        ],
        total=len(topics),
    )


@router.post(
    "/sync",
    response_model=SyncTriggerResponse,
    summary="Trigger curriculum sync",
    description="Manually trigger curriculum sync from Central Curriculum service.",
)
async def trigger_sync(
    framework_code: str | None = Query(None, description="Specific framework to sync"),
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_tenant_admin),
) -> SyncTriggerResponse:
    """Trigger manual curriculum sync.

    This endpoint dispatches a background task to sync curriculum data
    from the Central Curriculum service.

    Args:
        framework_code: Optional specific framework to sync.
        db: Tenant database session.
        current_user: Authenticated admin user.

    Returns:
        Sync trigger status.
    """
    from src.core.config import get_settings
    from src.infrastructure.background.tasks import sync_curriculum_for_tenant

    settings = get_settings()

    if not settings.central_curriculum.sync_enabled:
        return SyncTriggerResponse(
            status="disabled",
            message="Curriculum sync is disabled in settings",
        )

    # Get tenant code from current context
    tenant_code = current_user.tenant_code

    # Dispatch sync task
    sync_curriculum_for_tenant.send(tenant_code, framework_code)

    message = (
        f"Curriculum sync scheduled for framework: {framework_code}"
        if framework_code
        else "Full curriculum sync scheduled"
    )

    logger.info(
        "Curriculum sync triggered: tenant=%s, framework=%s, user=%s",
        tenant_code,
        framework_code or "all",
        current_user.id,
    )

    return SyncTriggerResponse(
        status="scheduled",
        message=message,
    )


@router.get(
    "/sso-url",
    response_model=SSOUrlResponse,
    summary="Get SSO URL for Central Curriculum Portal",
    description="Generate an SSO URL to open Central Curriculum Portal with automatic authentication.",
)
async def get_sso_url(
    current_user: CurrentUser = Depends(require_tenant_admin),
) -> SSOUrlResponse:
    """Get SSO URL for Central Curriculum Portal.

    This endpoint calls Central Curriculum's SSO generate endpoint
    to get a short-lived token, then returns the redirect URL.

    Args:
        current_user: Authenticated tenant admin.

    Returns:
        SSO URL to open in a new tab.

    Raises:
        HTTPException: If SSO token generation fails.
    """
    import httpx

    from src.core.config import get_settings

    settings = get_settings()
    tenant_code = current_user.tenant_code

    # Call Central Curriculum SSO generate endpoint
    url = f"{settings.central_curriculum.base_url}/auth/sso/generate"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                headers=settings.central_curriculum.sso_headers,
                json={"tenant_code": tenant_code},
            )

            if response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tenant '{tenant_code}' not found in Central Curriculum. Please ensure the tenant is registered.",
                )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="SSO authentication failed. Please contact support.",
                )

            response.raise_for_status()
            data = response.json()

            logger.info(
                "SSO URL generated: tenant=%s, user=%s",
                tenant_code,
                current_user.id,
            )

            return SSOUrlResponse(
                url=data["redirect_url"],
                expires_in=data["expires_in"],
            )

    except httpx.TimeoutException:
        logger.error("SSO request timeout: tenant=%s", tenant_code)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Central Curriculum service timeout",
        )
    except httpx.HTTPStatusError as e:
        logger.error("SSO request failed: tenant=%s, status=%s", tenant_code, e.response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate SSO URL",
        )
    except httpx.RequestError as e:
        logger.error("SSO request error: tenant=%s, error=%s", tenant_code, str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Central Curriculum service unavailable",
        )
