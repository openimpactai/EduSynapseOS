# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""School management API endpoints.

This module provides endpoints for school management:
- POST / - Create a new school
- GET / - List schools with filtering
- GET /{school_id} - Get school details
- PUT /{school_id} - Update school
- DELETE /{school_id} - Deactivate school (soft delete)
- POST /{school_id}/activate - Reactivate school
- POST /{school_id}/admins - Assign school admin
- DELETE /{school_id}/admins/{user_id} - Remove school admin
- GET /{school_id}/admins - List school admins

School management requires tenant context and appropriate permissions:
- tenant_admin: Full access to all schools
- school_admin: Read access to assigned schools only

Example:
    POST /api/v1/schools
    {
        "code": "primary_01",
        "name": "Primary School One",
        "school_type": "primary",
        "city": "Istanbul"
    }
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    require_auth,
    require_tenant_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.domains.school.service import (
    SchoolService,
    SchoolNotFoundError,
    SchoolCodeExistsError,
    SchoolAdminError,
)
from src.models.school import (
    SchoolCreateRequest,
    SchoolUpdateRequest,
    SchoolResponse,
    SchoolListResponse,
    SchoolAdminAssignRequest,
    SchoolAdminListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_school_service(db: AsyncSession) -> SchoolService:
    """Get school service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured SchoolService instance.
    """
    return SchoolService(db=db)


async def _check_school_access(
    service: SchoolService,
    current_user: CurrentUser,
    school_id: str,
) -> bool:
    """Check if current user can access a school.

    Tenant admins can access all schools.
    School admins can only access their assigned schools.

    Args:
        service: School service.
        current_user: Current authenticated user.
        school_id: School ID to check access for.

    Returns:
        True if user has access.
    """
    if current_user.user_type == "tenant_admin":
        return True

    if current_user.user_type == "school_admin":
        return await service.check_user_access(current_user.id, school_id)

    return False


@router.post(
    "",
    response_model=SchoolResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create school",
    description="Create a new school. Requires tenant admin access.",
)
async def create_school(
    data: SchoolCreateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolResponse:
    """Create a new school.

    Only tenant admins can create schools.

    Args:
        data: School creation request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created school response.

    Raises:
        HTTPException: If school code already exists.
    """
    logger.info(
        "Creating school: code=%s, name=%s, by=%s",
        data.code,
        data.name,
        current_user.id,
    )

    service = _get_school_service(db)

    try:
        return await service.create_school(
            request=data,
            created_by=current_user.id,
        )
    except SchoolCodeExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"School with code '{data.code}' already exists",
        )


@router.get(
    "",
    response_model=SchoolListResponse,
    summary="List schools",
    description="List schools with optional filtering. Access based on user role.",
)
async def list_schools(
    is_active: Annotated[bool | None, Query(description="Filter by active status")] = None,
    search: Annotated[str | None, Query(description="Search by name, code, or city")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 20,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolListResponse:
    """List schools with filtering.

    Tenant admins see all schools.
    School admins see only their assigned schools.
    Other users cannot access this endpoint.

    Args:
        is_active: Optional active status filter.
        search: Optional search query.
        limit: Maximum results.
        offset: Pagination offset.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of schools with pagination info.

    Raises:
        HTTPException: If user doesn't have admin access.
    """
    # Check authorization
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to list schools",
        )

    service = _get_school_service(db)

    # Filter by assigned schools for school_admin
    school_ids = None
    if current_user.user_type == "school_admin":
        school_ids = current_user.school_ids

    schools, total = await service.list_schools(
        is_active=is_active,
        search=search,
        school_ids=school_ids,
        limit=limit,
        offset=offset,
    )

    return SchoolListResponse(
        items=schools,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{school_id}",
    response_model=SchoolResponse,
    summary="Get school",
    description="Get school details by ID. Access based on user role.",
)
async def get_school(
    school_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolResponse:
    """Get school details.

    Tenant admins can access any school.
    School admins can only access their assigned schools.

    Args:
        school_id: School identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        School details.

    Raises:
        HTTPException: If school not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    service = _get_school_service(db)

    # Check access
    has_access = await _check_school_access(service, current_user, str(school_id))
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this school",
        )

    try:
        return await service.get_school(school_id)
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )


@router.put(
    "/{school_id}",
    response_model=SchoolResponse,
    summary="Update school",
    description="Update school information. Requires tenant admin access.",
)
async def update_school(
    school_id: UUID,
    data: SchoolUpdateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolResponse:
    """Update school information.

    Only tenant admins can update schools.

    Args:
        school_id: School identifier.
        data: Update request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated school.

    Raises:
        HTTPException: If school not found.
    """
    logger.info("Updating school: %s by %s", school_id, current_user.id)

    service = _get_school_service(db)

    try:
        return await service.update_school(school_id, data)
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )


@router.delete(
    "/{school_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate school",
    description="Soft delete a school. Requires tenant admin access.",
)
async def deactivate_school(
    school_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Deactivate a school (soft delete).

    Only tenant admins can deactivate schools.
    The school can be reactivated later.

    Args:
        school_id: School identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If school not found.
    """
    logger.info("Deactivating school: %s by %s", school_id, current_user.id)

    service = _get_school_service(db)

    try:
        await service.deactivate_school(school_id)
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )


@router.post(
    "/{school_id}/activate",
    response_model=SchoolResponse,
    summary="Activate school",
    description="Reactivate a deactivated school. Requires tenant admin access.",
)
async def activate_school(
    school_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolResponse:
    """Reactivate a deactivated school.

    Only tenant admins can reactivate schools.

    Args:
        school_id: School identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Activated school.

    Raises:
        HTTPException: If school not found.
    """
    logger.info("Activating school: %s by %s", school_id, current_user.id)

    service = _get_school_service(db)

    try:
        return await service.activate_school(school_id)
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )


@router.post(
    "/{school_id}/admins",
    status_code=status.HTTP_201_CREATED,
    summary="Assign school admin",
    description="Assign a user as school admin. Requires tenant admin access.",
)
async def assign_school_admin(
    school_id: UUID,
    data: SchoolAdminAssignRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> dict:
    """Assign a user as school admin.

    Only tenant admins can assign school admins.
    The user will get the school_admin role scoped to this school.

    Args:
        school_id: School identifier.
        data: Admin assignment request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Success message.

    Raises:
        HTTPException: If school/user not found or already admin.
    """
    logger.info(
        "Assigning school admin: school=%s, user=%s, by=%s",
        school_id,
        data.user_id,
        current_user.id,
    )

    service = _get_school_service(db)

    try:
        await service.assign_admin(
            school_id=school_id,
            user_id=data.user_id,
            assigned_by=current_user.id,
        )
        return {"message": "Admin assigned successfully"}
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )
    except SchoolAdminError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/{school_id}/admins/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove school admin",
    description="Remove a user as school admin. Requires tenant admin access.",
)
async def remove_school_admin(
    school_id: UUID,
    user_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Remove a user as school admin.

    Only tenant admins can remove school admins.

    Args:
        school_id: School identifier.
        user_id: User identifier to remove.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If school not found or user is not admin.
    """
    logger.info(
        "Removing school admin: school=%s, user=%s, by=%s",
        school_id,
        user_id,
        current_user.id,
    )

    service = _get_school_service(db)

    try:
        await service.remove_admin(school_id=school_id, user_id=user_id)
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )
    except SchoolAdminError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/{school_id}/admins",
    response_model=SchoolAdminListResponse,
    summary="List school admins",
    description="List all admins for a school. Requires admin access.",
)
async def list_school_admins(
    school_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> SchoolAdminListResponse:
    """List all admins for a school.

    Tenant admins can view any school's admins.
    School admins can only view their own school's admins.

    Args:
        school_id: School identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of school admins.

    Raises:
        HTTPException: If school not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    service = _get_school_service(db)

    # Check access
    has_access = await _check_school_access(service, current_user, str(school_id))
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this school",
        )

    try:
        admins = await service.list_admins(school_id)
        return SchoolAdminListResponse(
            items=admins,
            total=len(admins),
        )
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )
