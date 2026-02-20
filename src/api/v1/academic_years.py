# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Academic year management API endpoints.

This module provides endpoints for academic year management:
- POST / - Create a new academic year
- GET / - List academic years
- GET /current - Get current academic year
- GET /{year_id} - Get academic year details
- PUT /{year_id} - Update academic year
- DELETE /{year_id} - Delete academic year
- POST /{year_id}/set-current - Set as current year

Academic year management requires tenant admin access.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    require_tenant_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.domains.academic_year.service import (
    AcademicYearService,
    AcademicYearNotFoundError,
    AcademicYearOverlapError,
    AcademicYearServiceError,
)
from src.models.academic_year import (
    AcademicYearCreateRequest,
    AcademicYearUpdateRequest,
    AcademicYearResponse,
    AcademicYearListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service(db: AsyncSession) -> AcademicYearService:
    """Get academic year service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured AcademicYearService instance.
    """
    return AcademicYearService(db=db)


@router.post(
    "",
    response_model=AcademicYearResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create academic year",
    description="Create a new academic year. Requires tenant admin access.",
)
async def create_academic_year(
    data: AcademicYearCreateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearResponse:
    """Create a new academic year.

    Only tenant admins can create academic years.

    Args:
        data: Academic year creation request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created academic year response.

    Raises:
        HTTPException: If dates overlap with existing year.
    """
    logger.info(
        "Creating academic year: %s (%s to %s) by %s",
        data.name,
        data.start_date,
        data.end_date,
        current_user.id,
    )

    service = _get_service(db)

    try:
        return await service.create_academic_year(request=data)
    except AcademicYearOverlapError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.get(
    "",
    response_model=AcademicYearListResponse,
    summary="List academic years",
    description="List all academic years. Requires tenant admin access.",
)
async def list_academic_years(
    include_past: Annotated[
        bool, Query(description="Include past academic years")
    ] = True,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearListResponse:
    """List all academic years.

    Args:
        include_past: Whether to include past years.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of academic years.
    """
    service = _get_service(db)

    items, total = await service.list_academic_years(include_past=include_past)

    return AcademicYearListResponse(items=items, total=total)


@router.get(
    "/current",
    response_model=AcademicYearResponse | None,
    summary="Get current academic year",
    description="Get the current academic year. Requires tenant admin access.",
)
async def get_current_academic_year(
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearResponse | None:
    """Get the current academic year.

    Args:
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Current academic year or None.
    """
    service = _get_service(db)
    return await service.get_current_year()


@router.get(
    "/{year_id}",
    response_model=AcademicYearResponse,
    summary="Get academic year",
    description="Get academic year details by ID. Requires tenant admin access.",
)
async def get_academic_year(
    year_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearResponse:
    """Get academic year details.

    Args:
        year_id: Academic year identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Academic year details.

    Raises:
        HTTPException: If academic year not found.
    """
    service = _get_service(db)

    try:
        return await service.get_academic_year(year_id)
    except AcademicYearNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Academic year not found",
        )


@router.put(
    "/{year_id}",
    response_model=AcademicYearResponse,
    summary="Update academic year",
    description="Update academic year information. Requires tenant admin access.",
)
async def update_academic_year(
    year_id: UUID,
    data: AcademicYearUpdateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearResponse:
    """Update academic year information.

    Args:
        year_id: Academic year identifier.
        data: Update request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated academic year.

    Raises:
        HTTPException: If year not found or dates overlap.
    """
    logger.info("Updating academic year: %s by %s", year_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.update_academic_year(year_id, data)
    except AcademicYearNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Academic year not found",
        )
    except AcademicYearOverlapError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.delete(
    "/{year_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete academic year",
    description="Delete an academic year. Requires tenant admin access.",
)
async def delete_academic_year(
    year_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Delete an academic year.

    Cannot delete if there are associated classes.

    Args:
        year_id: Academic year identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If year not found or has classes.
    """
    logger.info("Deleting academic year: %s by %s", year_id, current_user.id)

    service = _get_service(db)

    try:
        await service.delete_academic_year(year_id)
    except AcademicYearNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Academic year not found",
        )
    except AcademicYearServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/{year_id}/set-current",
    response_model=AcademicYearResponse,
    summary="Set current academic year",
    description="Set an academic year as the current year. Requires tenant admin access.",
)
async def set_current_academic_year(
    year_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> AcademicYearResponse:
    """Set an academic year as the current year.

    This will unset any previous current year.

    Args:
        year_id: Academic year identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated academic year.

    Raises:
        HTTPException: If year not found.
    """
    logger.info("Setting current academic year: %s by %s", year_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.set_current_year(year_id)
    except AcademicYearNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Academic year not found",
        )
