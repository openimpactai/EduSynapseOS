# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parent-student relation management API endpoints.

This module provides endpoints for parent-student relationship management:
- POST / - Create a new relationship
- GET / - List relationships with filtering
- GET /{relation_id} - Get relationship details
- PUT /{relation_id} - Update relationship
- DELETE /{relation_id} - Delete relationship
- POST /{relation_id}/verify - Verify relationship

Parent relation management requires tenant admin or school admin access.
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
from src.domains.parent_relation.service import (
    ParentRelationService,
    RelationNotFoundError,
    ParentNotFoundError,
    StudentNotFoundError,
    RelationExistsError,
    InvalidUserTypeError,
)
from src.models.parent_relation import (
    CreateParentRelationRequest,
    UpdateParentRelationRequest,
    ParentRelationResponse,
    ParentRelationListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service(db: AsyncSession) -> ParentRelationService:
    """Get parent relation service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured ParentRelationService instance.
    """
    return ParentRelationService(db=db)


@router.post(
    "",
    response_model=ParentRelationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create parent-student relationship",
    description="Create a new parent-student relationship. Requires tenant admin access.",
)
async def create_relation(
    data: CreateParentRelationRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ParentRelationResponse:
    """Create a parent-student relationship.

    Only tenant admins can create relationships.

    Args:
        data: Relation creation request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created relation response.

    Raises:
        HTTPException: If parent/student not found or relation exists.
    """
    logger.info(
        "Creating parent-student relation: parent=%s, student=%s, by=%s",
        data.parent_id,
        data.student_id,
        current_user.id,
    )

    service = _get_service(db)

    try:
        return await service.create_relation(
            request=data,
            created_by=current_user.id,
        )
    except ParentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent not found",
        )
    except StudentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )
    except InvalidUserTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RelationExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Relationship already exists between this parent and student",
        )


@router.get(
    "",
    response_model=ParentRelationListResponse,
    summary="List parent-student relationships",
    description="List relationships with optional filtering. Access based on user role.",
)
async def list_relations(
    parent_id: Annotated[UUID | None, Query(description="Filter by parent")] = None,
    student_id: Annotated[UUID | None, Query(description="Filter by student")] = None,
    is_verified: Annotated[bool | None, Query(description="Filter by verification status")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 20,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ParentRelationListResponse:
    """List parent-student relationships.

    Tenant admins can list all relationships.
    School admins can list relationships for students in their schools.

    Args:
        parent_id: Filter by parent.
        student_id: Filter by student.
        is_verified: Filter by verification status.
        limit: Maximum results.
        offset: Pagination offset.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of relationships with pagination info.

    Raises:
        HTTPException: If user doesn't have admin access.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to list relationships",
        )

    service = _get_service(db)

    items, total = await service.list_relations(
        parent_id=parent_id,
        student_id=student_id,
        is_verified=is_verified,
        limit=limit,
        offset=offset,
    )

    return ParentRelationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{relation_id}",
    response_model=ParentRelationResponse,
    summary="Get relationship details",
    description="Get relationship details by ID. Access based on user role.",
)
async def get_relation(
    relation_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ParentRelationResponse:
    """Get relationship details.

    Args:
        relation_id: Relation identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Relationship details.

    Raises:
        HTTPException: If relation not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    service = _get_service(db)

    try:
        return await service.get_relation(relation_id)
    except RelationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Relationship not found",
        )


@router.put(
    "/{relation_id}",
    response_model=ParentRelationResponse,
    summary="Update relationship",
    description="Update relationship permissions. Requires tenant admin access.",
)
async def update_relation(
    relation_id: UUID,
    data: UpdateParentRelationRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ParentRelationResponse:
    """Update relationship permissions.

    Only tenant admins can update relationships.

    Args:
        relation_id: Relation identifier.
        data: Update request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated relationship.

    Raises:
        HTTPException: If relation not found.
    """
    logger.info("Updating parent-student relation: %s by %s", relation_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.update_relation(
            relation_id=relation_id,
            request=data,
            updated_by=current_user.id,
        )
    except RelationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Relationship not found",
        )


@router.post(
    "/{relation_id}/verify",
    response_model=ParentRelationResponse,
    summary="Verify relationship",
    description="Verify a parent-student relationship. Requires tenant admin access.",
)
async def verify_relation(
    relation_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ParentRelationResponse:
    """Verify a parent-student relationship.

    Only tenant admins can verify relationships.

    Args:
        relation_id: Relation identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Verified relationship.

    Raises:
        HTTPException: If relation not found.
    """
    logger.info("Verifying parent-student relation: %s by %s", relation_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.verify_relation(
            relation_id=relation_id,
            verified_by=current_user.id,
        )
    except RelationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Relationship not found",
        )


@router.delete(
    "/{relation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete relationship",
    description="Delete a parent-student relationship. Requires tenant admin access.",
)
async def delete_relation(
    relation_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Delete a parent-student relationship.

    Only tenant admins can delete relationships.

    Args:
        relation_id: Relation identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If relation not found.
    """
    logger.info("Deleting parent-student relation: %s by %s", relation_id, current_user.id)

    service = _get_service(db)

    try:
        await service.delete_relation(
            relation_id=relation_id,
            deleted_by=current_user.id,
        )
    except RelationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Relationship not found",
        )
