# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""User management API endpoints.

This module provides endpoints for user management:
- POST / - Create a new user
- GET / - List users with filtering
- GET /{user_id} - Get user details
- PUT /{user_id} - Update user
- DELETE /{user_id} - Delete user (soft delete)
- POST /{user_id}/activate - Activate user
- POST /{user_id}/suspend - Suspend user
- GET /{user_id}/classes - Get user's classes (for students/teachers)
- GET /{user_id}/children - Get user's children (for parents)
- GET /{user_id}/roles - Get user's roles

Users are authenticated via LMS integration - the LMS authenticates users
and asserts their identity to EduSynapseOS using API credentials.

Example:
    POST /api/v1/users
    {
        "email": "student@school.com",
        "first_name": "John",
        "last_name": "Doe",
        "user_type": "student",
        "external_id": "lms_user_123"
    }
"""

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_tenant_db,
    require_auth,
    require_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.domains.user.service import (
    UserService,
    UserNotFoundError,
    UserAlreadyExistsError,
)
from src.models.user import (
    UserCreateRequest,
    UserUpdateRequest,
    UserResponse,
    UserSummary,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from src.infrastructure.database.models.tenant.user import User, UserRole, Role
from src.infrastructure.database.models.tenant.school import (
    ClassStudent,
    ClassTeacher,
    ParentStudentRelation,
    Class,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_user_service(db: AsyncSession) -> UserService:
    """Get user service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured UserService instance.
    """
    return UserService(db=db)


class UserListResponse(BaseModel):
    """Response for user list endpoint."""

    users: list[UserSummary]
    total: int
    limit: int
    offset: int


class SuspendUserRequest(BaseModel):
    """Request to suspend a user."""

    reason: str | None = None


class UserClassInfo(BaseModel):
    """Class information for a user."""

    class_id: UUID
    class_name: str
    class_code: str
    school_id: UUID
    school_name: str
    role: str  # "student" or "teacher"
    is_homeroom: bool | None = None  # Only for teachers
    subject_name: str | None = None  # Only for teachers
    enrolled_at: datetime | None = None  # Only for students
    assigned_at: datetime | None = None  # Only for teachers


class UserClassesResponse(BaseModel):
    """Response for user's classes."""

    user_id: UUID
    user_name: str
    classes: list[UserClassInfo]
    total: int


class UserChildInfo(BaseModel):
    """Child information for a parent."""

    student_id: UUID
    student_name: str
    student_email: str
    relationship_type: str
    is_primary: bool
    is_verified: bool
    can_view_progress: bool
    can_view_conversations: bool


class UserChildrenResponse(BaseModel):
    """Response for parent's children."""

    parent_id: UUID
    parent_name: str
    children: list[UserChildInfo]
    total: int


class UserRoleInfo(BaseModel):
    """Role information for a user."""

    role_id: UUID
    role_code: str
    role_name: str
    school_id: UUID | None = None
    school_name: str | None = None
    class_id: UUID | None = None
    class_name: str | None = None


class UserRolesResponse(BaseModel):
    """Response for user's roles."""

    user_id: UUID
    user_name: str
    user_type: str
    roles: list[UserRoleInfo]
    total: int


from datetime import datetime


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
    description="Create a new user in the tenant. Requires admin access.",
)
async def create_user(
    data: UserCreateRequest,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserResponse:
    """Create a new user.

    Users are created without passwords since authentication is handled
    via LMS integration. The LMS authenticates users and asserts their
    identity to EduSynapseOS using API credentials.

    Args:
        data: User creation request.
        current_user: Authenticated admin user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created user response.

    Raises:
        HTTPException: If email already exists.
    """
    logger.info(
        "Creating user: email=%s, type=%s, by=%s",
        data.email,
        data.user_type,
        current_user.id,
    )

    service = _get_user_service(db)

    try:
        return await service.create_user(
            request=data,
            created_by=current_user.id,
            auto_activate=True,  # Admin-created users are auto-activated
        )
    except UserAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with email {data.email} already exists",
        )


@router.get(
    "",
    response_model=UserListResponse,
    summary="List users",
    description="List users with optional filtering. Requires admin access.",
)
async def list_users(
    user_type: Annotated[str | None, Query(description="Filter by user type")] = None,
    user_status: Annotated[str | None, Query(alias="status", description="Filter by status")] = None,
    search: Annotated[str | None, Query(description="Search by name or email")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 20,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserListResponse:
    """List users with filtering.

    Args:
        user_type: Optional user type filter.
        user_status: Optional status filter.
        search: Optional search query.
        limit: Maximum results.
        offset: Pagination offset.
        current_user: Authenticated admin user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of users with pagination info.
    """
    service = _get_user_service(db)

    users, total = await service.list_users(
        user_type=user_type,
        status=user_status,
        search=search,
        limit=limit,
        offset=offset,
    )

    return UserListResponse(
        users=users,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
    description="Get user details by ID. Requires authentication.",
)
async def get_user(
    user_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserResponse:
    """Get user details.

    Args:
        user_id: User identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        User details.

    Raises:
        HTTPException: If user not found.
    """
    # Users can view their own profile, admins can view anyone
    if str(user_id) != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other users' profiles",
        )

    service = _get_user_service(db)

    try:
        return await service.get_user(user_id)
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description="Update user information. Users can update themselves, admins can update anyone.",
)
async def update_user(
    user_id: UUID,
    data: UserUpdateRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserResponse:
    """Update user information.

    Args:
        user_id: User identifier.
        data: Update request.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated user.

    Raises:
        HTTPException: If user not found or not authorized.
    """
    # Users can update their own profile, admins can update anyone
    if str(user_id) != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot update other users' profiles",
        )

    service = _get_user_service(db)

    try:
        return await service.update_user(user_id, data)
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Soft delete a user. Requires admin access.",
)
async def delete_user(
    user_id: UUID,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Delete a user (soft delete).

    Args:
        user_id: User identifier.
        current_user: Authenticated admin user.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If user not found.
    """
    # Cannot delete yourself
    if str(user_id) == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    logger.info("Deleting user: %s by %s", user_id, current_user.id)

    service = _get_user_service(db)

    try:
        await service.delete_user(user_id)
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.post(
    "/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user",
    description="Activate a pending or suspended user. Requires admin access.",
)
async def activate_user(
    user_id: UUID,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserResponse:
    """Activate a user account.

    Args:
        user_id: User identifier.
        current_user: Authenticated admin user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Activated user.

    Raises:
        HTTPException: If user not found.
    """
    logger.info("Activating user: %s by %s", user_id, current_user.id)

    service = _get_user_service(db)

    try:
        return await service.activate_user(user_id)
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.post(
    "/{user_id}/suspend",
    response_model=UserResponse,
    summary="Suspend user",
    description="Suspend a user account. Requires admin access.",
)
async def suspend_user(
    user_id: UUID,
    data: SuspendUserRequest | None = None,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserResponse:
    """Suspend a user account.

    Args:
        user_id: User identifier.
        data: Optional suspension details.
        current_user: Authenticated admin user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Suspended user.

    Raises:
        HTTPException: If user not found or trying to suspend self.
    """
    # Cannot suspend yourself
    if str(user_id) == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend your own account",
        )

    logger.info("Suspending user: %s by %s", user_id, current_user.id)

    service = _get_user_service(db)

    try:
        return await service.suspend_user(
            user_id,
            reason=data.reason if data else None,
        )
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


# ============================================================================
# User Context Endpoints
# ============================================================================


@router.get(
    "/{user_id}/classes",
    response_model=UserClassesResponse,
    summary="Get user's classes",
    description="Get classes a user is enrolled in (students) or assigned to (teachers).",
)
async def get_user_classes(
    user_id: UUID,
    active_only: Annotated[bool, Query(description="Only active enrollments/assignments")] = True,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserClassesResponse:
    """Get classes for a user.

    Students see classes they are enrolled in.
    Teachers see classes they are assigned to.

    Args:
        user_id: User identifier.
        active_only: Only include active enrollments/assignments.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        User's classes.

    Raises:
        HTTPException: If user not found or access denied.
    """
    # Users can view their own classes, admins can view anyone's
    if str(user_id) != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other users' classes",
        )

    # Get user
    user_query = select(User).where(
        User.id == str(user_id),
        User.deleted_at.is_(None),
    )
    result = await db.execute(user_query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user_name = f"{user.first_name} {user.last_name}".strip()
    classes = []

    if user.user_type == "student":
        # Get student enrollments
        query = select(ClassStudent).options(
            selectinload(ClassStudent.class_).selectinload(Class.school),
        ).where(ClassStudent.student_id == str(user_id))

        if active_only:
            query = query.where(ClassStudent.status == "active")

        result = await db.execute(query)
        enrollments = result.scalars().all()

        for enrollment in enrollments:
            class_ = enrollment.class_
            if class_:
                classes.append(UserClassInfo(
                    class_id=UUID(class_.id),
                    class_name=class_.name,
                    class_code=class_.code,
                    school_id=UUID(class_.school_id),
                    school_name=class_.school.name if class_.school else "",
                    role="student",
                    enrolled_at=enrollment.enrolled_at,
                ))

    elif user.user_type == "teacher":
        # Get teacher assignments
        query = select(ClassTeacher).options(
            selectinload(ClassTeacher.class_).selectinload(Class.school),
            selectinload(ClassTeacher.subject),
        ).where(ClassTeacher.teacher_id == str(user_id))

        if active_only:
            query = query.where(ClassTeacher.ended_at.is_(None))

        result = await db.execute(query)
        assignments = result.scalars().all()

        for assignment in assignments:
            class_ = assignment.class_
            if class_:
                classes.append(UserClassInfo(
                    class_id=UUID(class_.id),
                    class_name=class_.name,
                    class_code=class_.code,
                    school_id=UUID(class_.school_id),
                    school_name=class_.school.name if class_.school else "",
                    role="teacher",
                    is_homeroom=assignment.is_homeroom,
                    subject_name=assignment.subject.name if assignment.subject else None,
                    assigned_at=assignment.assigned_at,
                ))

    return UserClassesResponse(
        user_id=user_id,
        user_name=user_name,
        classes=classes,
        total=len(classes),
    )


@router.get(
    "/{user_id}/children",
    response_model=UserChildrenResponse,
    summary="Get parent's children",
    description="Get children linked to a parent account.",
)
async def get_user_children(
    user_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserChildrenResponse:
    """Get children for a parent user.

    Args:
        user_id: Parent user identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Parent's children.

    Raises:
        HTTPException: If user not found, not a parent, or access denied.
    """
    # Users can view their own children, admins can view anyone's
    if str(user_id) != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other users' children",
        )

    # Get user
    user_query = select(User).where(
        User.id == str(user_id),
        User.deleted_at.is_(None),
    )
    result = await db.execute(user_query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.user_type != "parent":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a parent",
        )

    parent_name = f"{user.first_name} {user.last_name}".strip()

    # Get parent-student relations
    query = select(ParentStudentRelation).options(
        selectinload(ParentStudentRelation.student),
    ).where(ParentStudentRelation.parent_id == str(user_id))

    result = await db.execute(query)
    relations = result.scalars().all()

    children = []
    for relation in relations:
        student = relation.student
        if student:
            children.append(UserChildInfo(
                student_id=UUID(student.id),
                student_name=f"{student.first_name} {student.last_name}".strip(),
                student_email=student.email,
                relationship_type=relation.relationship_type,
                is_primary=relation.is_primary,
                is_verified=relation.verified_at is not None,
                can_view_progress=relation.can_view_progress,
                can_view_conversations=relation.can_view_conversations,
            ))

    return UserChildrenResponse(
        parent_id=user_id,
        parent_name=parent_name,
        children=children,
        total=len(children),
    )


@router.get(
    "/{user_id}/roles",
    response_model=UserRolesResponse,
    summary="Get user's roles",
    description="Get roles assigned to a user. Requires admin access.",
)
async def get_user_roles(
    user_id: UUID,
    current_user: CurrentUser = Depends(require_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> UserRolesResponse:
    """Get roles assigned to a user.

    Args:
        user_id: User identifier.
        current_user: Authenticated admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        User's roles.

    Raises:
        HTTPException: If user not found.
    """
    # Get user
    user_query = select(User).where(
        User.id == str(user_id),
        User.deleted_at.is_(None),
    )
    result = await db.execute(user_query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user_name = f"{user.first_name} {user.last_name}".strip()

    # Get user roles
    from src.infrastructure.database.models.tenant.school import School

    query = select(UserRole).options(
        selectinload(UserRole.role),
    ).where(UserRole.user_id == str(user_id))

    result = await db.execute(query)
    user_roles = result.scalars().all()

    roles = []
    for user_role in user_roles:
        role = user_role.role
        if role:
            role_info = UserRoleInfo(
                role_id=UUID(role.id),
                role_code=role.code,
                role_name=role.name,
            )

            # Add school info if scoped
            if user_role.school_id:
                school_query = select(School).where(School.id == user_role.school_id)
                school_result = await db.execute(school_query)
                school = school_result.scalar_one_or_none()
                if school:
                    role_info.school_id = UUID(school.id)
                    role_info.school_name = school.name

            # Add class info if scoped
            if user_role.class_id:
                class_query = select(Class).where(Class.id == user_role.class_id)
                class_result = await db.execute(class_query)
                class_ = class_result.scalar_one_or_none()
                if class_:
                    role_info.class_id = UUID(class_.id)
                    role_info.class_name = class_.name

            roles.append(role_info)

    return UserRolesResponse(
        user_id=user_id,
        user_name=user_name,
        user_type=user.user_type,
        roles=roles,
        total=len(roles),
    )
