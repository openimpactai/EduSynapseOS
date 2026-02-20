# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class management API endpoints.

This module provides endpoints for class/section management:
- POST / - Create a new class
- GET / - List classes with filtering
- GET /{class_id} - Get class details
- PUT /{class_id} - Update class
- DELETE /{class_id} - Deactivate class
- POST /{class_id}/activate - Reactivate class

Student enrollment endpoints:
- POST /{class_id}/students - Enroll a student
- GET /{class_id}/students - List enrolled students
- POST /{class_id}/students/bulk - Bulk enroll students
- GET /{class_id}/students/{student_id} - Get enrollment details
- POST /{class_id}/students/{student_id}/withdraw - Withdraw student
- DELETE /{class_id}/students/{student_id} - Remove enrollment

Teacher assignment endpoints:
- POST /{class_id}/teachers - Assign a teacher
- GET /{class_id}/teachers - List assigned teachers
- GET /{class_id}/teachers/{teacher_id} - Get assignment details
- POST /{class_id}/teachers/{teacher_id}/end - End assignment
- DELETE /{class_id}/teachers/{teacher_id} - Remove assignment

Class management requires tenant admin or school admin access.
School admins can only manage classes in their assigned schools.
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
from src.domains.class_.service import (
    ClassService,
    ClassNotFoundError,
    ClassCodeExistsError,
    SchoolNotFoundError,
    AcademicYearNotFoundError,
    ClassServiceError,
)
from src.models.class_ import (
    ClassCreateRequest,
    ClassUpdateRequest,
    ClassResponse,
    ClassListResponse,
)
from src.domains.enrollment.service import (
    EnrollmentService,
    ClassNotFoundError as EnrollmentClassNotFoundError,
    StudentNotFoundError,
    AlreadyEnrolledError,
    NotEnrolledError,
    InvalidStudentTypeError,
)
from src.models.enrollment import (
    EnrollStudentRequest,
    BulkEnrollRequest,
    WithdrawStudentRequest,
    EnrollmentResponse,
    EnrollmentListResponse,
    BulkEnrollResponse,
)
from src.domains.assignment.service import (
    TeacherAssignmentService,
    ClassNotFoundError as AssignmentClassNotFoundError,
    TeacherNotFoundError,
    AlreadyAssignedError,
    NotAssignedError,
    InvalidTeacherTypeError,
)
from src.models.assignment import (
    AssignTeacherRequest,
    EndAssignmentRequest,
    TeacherAssignmentResponse,
    TeacherAssignmentListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service(db: AsyncSession) -> ClassService:
    """Get class service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured ClassService instance.
    """
    return ClassService(db=db)


def _get_enrollment_service(db: AsyncSession) -> EnrollmentService:
    """Get enrollment service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured EnrollmentService instance.
    """
    return EnrollmentService(db=db)


def _get_assignment_service(db: AsyncSession) -> TeacherAssignmentService:
    """Get teacher assignment service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured TeacherAssignmentService instance.
    """
    return TeacherAssignmentService(db=db)


async def _check_class_access(
    service: ClassService,
    current_user: CurrentUser,
    class_id: str,
) -> bool:
    """Check if current user can access a class.

    Tenant admins can access all classes.
    School admins can only access classes in their schools.

    Args:
        service: Class service.
        current_user: Current authenticated user.
        class_id: Class ID to check access for.

    Returns:
        True if user has access.
    """
    if current_user.user_type == "tenant_admin":
        return True

    if current_user.user_type == "school_admin":
        return await service.check_school_access(current_user.id, class_id)

    return False


@router.post(
    "",
    response_model=ClassResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create class",
    description="Create a new class. Requires tenant admin access.",
)
async def create_class(
    data: ClassCreateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassResponse:
    """Create a new class.

    Only tenant admins can create classes.

    Args:
        data: Class creation request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created class response.

    Raises:
        HTTPException: If school/year not found or code exists.
    """
    logger.info(
        "Creating class: %s in school %s by %s",
        data.code,
        data.school_id,
        current_user.id,
    )

    service = _get_service(db)

    try:
        return await service.create_class(
            request=data,
            created_by=current_user.id,
        )
    except SchoolNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="School not found",
        )
    except AcademicYearNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Academic year not found",
        )
    except ClassCodeExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ClassServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "",
    response_model=ClassListResponse,
    summary="List classes",
    description="List classes with optional filtering. Access based on user role.",
)
async def list_classes(
    school_id: Annotated[UUID | None, Query(description="Filter by school")] = None,
    academic_year_id: Annotated[UUID | None, Query(description="Filter by academic year")] = None,
    is_active: Annotated[bool | None, Query(description="Filter by active status")] = None,
    search: Annotated[str | None, Query(description="Search by name or code")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 20,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassListResponse:
    """List classes with filtering.

    Tenant admins see all classes.
    School admins see only classes in their schools.

    Args:
        school_id: Optional school filter.
        academic_year_id: Optional academic year filter.
        is_active: Optional active status filter.
        search: Optional search query.
        limit: Maximum results.
        offset: Pagination offset.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of classes with pagination info.

    Raises:
        HTTPException: If user doesn't have admin access.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to list classes",
        )

    service = _get_service(db)

    # For school admins, limit to their schools
    effective_school_id = school_id
    if current_user.user_type == "school_admin":
        if school_id:
            # Verify they have access to this school
            if str(school_id) not in current_user.school_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this school",
                )
        elif len(current_user.school_ids) == 1:
            effective_school_id = UUID(current_user.school_ids[0])

    classes, total = await service.list_classes(
        school_id=effective_school_id,
        academic_year_id=academic_year_id,
        is_active=is_active,
        search=search,
        limit=limit,
        offset=offset,
    )

    return ClassListResponse(
        items=classes,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{class_id}",
    response_model=ClassResponse,
    summary="Get class",
    description="Get class details by ID. Access based on user role.",
)
async def get_class(
    class_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassResponse:
    """Get class details.

    Tenant admins can access any class.
    School admins can only access classes in their schools.

    Args:
        class_id: Class identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Class details.

    Raises:
        HTTPException: If class not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    service = _get_service(db)

    # Check access for school admins
    if current_user.user_type == "school_admin":
        has_access = await _check_class_access(service, current_user, str(class_id))
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this class",
            )

    try:
        return await service.get_class(class_id)
    except ClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


@router.put(
    "/{class_id}",
    response_model=ClassResponse,
    summary="Update class",
    description="Update class information. Requires tenant admin access.",
)
async def update_class(
    class_id: UUID,
    data: ClassUpdateRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassResponse:
    """Update class information.

    Only tenant admins can update classes.

    Args:
        class_id: Class identifier.
        data: Update request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated class.

    Raises:
        HTTPException: If class not found.
    """
    logger.info("Updating class: %s by %s", class_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.update_class(class_id, data)
    except ClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )
    except ClassServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/{class_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate class",
    description="Deactivate a class. Requires tenant admin access.",
)
async def deactivate_class(
    class_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Deactivate a class.

    Only tenant admins can deactivate classes.

    Args:
        class_id: Class identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If class not found.
    """
    logger.info("Deactivating class: %s by %s", class_id, current_user.id)

    service = _get_service(db)

    try:
        await service.deactivate_class(class_id)
    except ClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


@router.post(
    "/{class_id}/activate",
    response_model=ClassResponse,
    summary="Activate class",
    description="Reactivate a deactivated class. Requires tenant admin access.",
)
async def activate_class(
    class_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ClassResponse:
    """Reactivate a deactivated class.

    Only tenant admins can reactivate classes.

    Args:
        class_id: Class identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Activated class.

    Raises:
        HTTPException: If class not found.
    """
    logger.info("Activating class: %s by %s", class_id, current_user.id)

    service = _get_service(db)

    try:
        return await service.activate_class(class_id)
    except ClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


# ============================================================================
# Student Enrollment Endpoints
# ============================================================================


@router.post(
    "/{class_id}/students",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll student",
    description="Enroll a student in a class. Requires tenant admin access.",
)
async def enroll_student(
    class_id: UUID,
    data: EnrollStudentRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> EnrollmentResponse:
    """Enroll a student in a class.

    Only tenant admins can enroll students.

    Args:
        class_id: Class identifier.
        data: Enrollment request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Enrollment response.

    Raises:
        HTTPException: If class/student not found or already enrolled.
    """
    logger.info(
        "Enrolling student: student=%s, class=%s, by=%s",
        data.student_id,
        class_id,
        current_user.id,
    )

    service = _get_enrollment_service(db)

    try:
        return await service.enroll_student(
            class_id=class_id,
            request=data,
            enrolled_by=current_user.id,
        )
    except EnrollmentClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )
    except StudentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )
    except InvalidStudentTypeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a student type",
        )
    except AlreadyEnrolledError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Student is already enrolled in this class",
        )


@router.post(
    "/{class_id}/students/bulk",
    response_model=BulkEnrollResponse,
    summary="Bulk enroll students",
    description="Enroll multiple students in a class. Requires tenant admin access.",
)
async def bulk_enroll_students(
    class_id: UUID,
    data: BulkEnrollRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> BulkEnrollResponse:
    """Bulk enroll students in a class.

    Only tenant admins can perform bulk enrollment.

    Args:
        class_id: Class identifier.
        data: Bulk enrollment request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Bulk enrollment response with success/failure details.

    Raises:
        HTTPException: If class not found.
    """
    logger.info(
        "Bulk enrolling students: class=%s, count=%d, by=%s",
        class_id,
        len(data.student_ids),
        current_user.id,
    )

    service = _get_enrollment_service(db)

    try:
        return await service.bulk_enroll(
            class_id=class_id,
            request=data,
            enrolled_by=current_user.id,
        )
    except EnrollmentClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


@router.get(
    "/{class_id}/students",
    response_model=EnrollmentListResponse,
    summary="List enrolled students",
    description="List students enrolled in a class. Access based on user role.",
)
async def list_enrolled_students(
    class_id: UUID,
    enrollment_status: Annotated[
        str | None, Query(alias="status", description="Filter by status (active, withdrawn)")
    ] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> EnrollmentListResponse:
    """List students enrolled in a class.

    Tenant admins can list any class.
    School admins can only list their school's classes.

    Args:
        class_id: Class identifier.
        enrollment_status: Optional status filter.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of enrolled students.

    Raises:
        HTTPException: If class not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    class_service = _get_service(db)

    # Check access for school admins
    if current_user.user_type == "school_admin":
        has_access = await _check_class_access(class_service, current_user, str(class_id))
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this class",
            )

    enrollment_service = _get_enrollment_service(db)

    try:
        items, total, class_name = await enrollment_service.list_enrollments(
            class_id=class_id,
            status=enrollment_status,
        )
        return EnrollmentListResponse(
            items=items,
            total=total,
            class_id=class_id,
            class_name=class_name,
        )
    except EnrollmentClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


@router.get(
    "/{class_id}/students/{student_id}",
    response_model=EnrollmentResponse,
    summary="Get enrollment details",
    description="Get specific enrollment details. Access based on user role.",
)
async def get_enrollment(
    class_id: UUID,
    student_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> EnrollmentResponse:
    """Get specific enrollment details.

    Args:
        class_id: Class identifier.
        student_id: Student identifier.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Enrollment details.

    Raises:
        HTTPException: If not enrolled or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    class_service = _get_service(db)

    # Check access for school admins
    if current_user.user_type == "school_admin":
        has_access = await _check_class_access(class_service, current_user, str(class_id))
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this class",
            )

    enrollment_service = _get_enrollment_service(db)

    try:
        return await enrollment_service.get_enrollment(class_id, student_id)
    except NotEnrolledError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student is not enrolled in this class",
        )


@router.post(
    "/{class_id}/students/{student_id}/withdraw",
    response_model=EnrollmentResponse,
    summary="Withdraw student",
    description="Withdraw a student from a class. Requires tenant admin access.",
)
async def withdraw_student(
    class_id: UUID,
    student_id: UUID,
    data: WithdrawStudentRequest | None = None,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> EnrollmentResponse:
    """Withdraw a student from a class.

    Only tenant admins can withdraw students.

    Args:
        class_id: Class identifier.
        student_id: Student identifier.
        data: Optional withdrawal request with date.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated enrollment.

    Raises:
        HTTPException: If not enrolled.
    """
    logger.info(
        "Withdrawing student: student=%s, class=%s, by=%s",
        student_id,
        class_id,
        current_user.id,
    )

    service = _get_enrollment_service(db)

    try:
        return await service.withdraw_student(
            class_id=class_id,
            student_id=student_id,
            withdrawn_at=data.withdrawn_at if data else None,
            withdrawn_by=current_user.id,
        )
    except NotEnrolledError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{class_id}/students/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove enrollment",
    description="Permanently remove an enrollment record. Requires tenant admin access.",
)
async def remove_enrollment(
    class_id: UUID,
    student_id: UUID,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Permanently remove an enrollment record.

    Only tenant admins can remove enrollments.

    Args:
        class_id: Class identifier.
        student_id: Student identifier.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If enrollment not found.
    """
    logger.info(
        "Removing enrollment: student=%s, class=%s, by=%s",
        student_id,
        class_id,
        current_user.id,
    )

    service = _get_enrollment_service(db)

    try:
        await service.remove_enrollment(
            class_id=class_id,
            student_id=student_id,
            removed_by=current_user.id,
        )
    except NotEnrolledError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found",
        )


# ============================================================================
# Teacher Assignment Endpoints
# ============================================================================


@router.post(
    "/{class_id}/teachers",
    response_model=TeacherAssignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assign teacher",
    description="Assign a teacher to a class. Requires tenant admin access.",
)
async def assign_teacher(
    class_id: UUID,
    data: AssignTeacherRequest,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TeacherAssignmentResponse:
    """Assign a teacher to a class.

    Only tenant admins can assign teachers.

    Args:
        class_id: Class identifier.
        data: Assignment request.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Assignment response.

    Raises:
        HTTPException: If class/teacher not found or already assigned.
    """
    logger.info(
        "Assigning teacher: teacher=%s, class=%s, subject=%s, by=%s",
        data.teacher_id,
        class_id,
        data.subject_full_code if data.has_subject() else None,
        current_user.id,
    )

    service = _get_assignment_service(db)

    try:
        return await service.assign_teacher(
            class_id=class_id,
            request=data,
            assigned_by=current_user.id,
        )
    except AssignmentClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )
    except TeacherNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Teacher not found",
        )
    except InvalidTeacherTypeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a teacher type",
        )
    except AlreadyAssignedError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Teacher is already assigned to this class with the same subject",
        )


@router.get(
    "/{class_id}/teachers",
    response_model=TeacherAssignmentListResponse,
    summary="List assigned teachers",
    description="List teachers assigned to a class. Access based on user role.",
)
async def list_assigned_teachers(
    class_id: UUID,
    active_only: Annotated[
        bool, Query(description="Only include active assignments")
    ] = True,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TeacherAssignmentListResponse:
    """List teachers assigned to a class.

    Tenant admins can list any class.
    School admins can only list their school's classes.

    Args:
        class_id: Class identifier.
        active_only: Only include active assignments.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of assigned teachers.

    Raises:
        HTTPException: If class not found or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    class_service = _get_service(db)

    # Check access for school admins
    if current_user.user_type == "school_admin":
        has_access = await _check_class_access(class_service, current_user, str(class_id))
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this class",
            )

    assignment_service = _get_assignment_service(db)

    try:
        items, total, class_name = await assignment_service.list_assignments(
            class_id=class_id,
            active_only=active_only,
        )
        return TeacherAssignmentListResponse(
            items=items,
            total=total,
            class_id=class_id,
            class_name=class_name,
        )
    except AssignmentClassNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )


@router.get(
    "/{class_id}/teachers/{teacher_id}",
    response_model=TeacherAssignmentResponse,
    summary="Get assignment details",
    description="Get specific assignment details. Access based on user role.",
)
async def get_teacher_assignment(
    class_id: UUID,
    teacher_id: UUID,
    subject_framework_code: Annotated[str | None, Query(description="Subject framework code")] = None,
    subject_code: Annotated[str | None, Query(description="Subject code")] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TeacherAssignmentResponse:
    """Get specific assignment details.

    Args:
        class_id: Class identifier.
        teacher_id: Teacher identifier.
        subject_framework_code: Optional subject framework code (e.g., "UK-NC-2014").
        subject_code: Optional subject code (e.g., "MAT").
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Assignment details.

    Raises:
        HTTPException: If not assigned or access denied.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    class_service = _get_service(db)

    # Check access for school admins
    if current_user.user_type == "school_admin":
        has_access = await _check_class_access(class_service, current_user, str(class_id))
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this class",
            )

    assignment_service = _get_assignment_service(db)

    try:
        return await assignment_service.get_assignment(
            class_id, teacher_id, subject_framework_code, subject_code
        )
    except NotAssignedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Teacher is not assigned to this class",
        )


@router.post(
    "/{class_id}/teachers/{teacher_id}/end",
    response_model=TeacherAssignmentResponse,
    summary="End teacher assignment",
    description="End a teacher assignment. Requires tenant admin access.",
)
async def end_teacher_assignment(
    class_id: UUID,
    teacher_id: UUID,
    subject_framework_code: Annotated[str | None, Query(description="Subject framework code")] = None,
    subject_code: Annotated[str | None, Query(description="Subject code")] = None,
    data: EndAssignmentRequest | None = None,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> TeacherAssignmentResponse:
    """End a teacher assignment.

    Only tenant admins can end assignments.

    Args:
        class_id: Class identifier.
        teacher_id: Teacher identifier.
        subject_framework_code: Optional subject framework code (e.g., "UK-NC-2014").
        subject_code: Optional subject code (e.g., "MAT").
        data: Optional end request with date.
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated assignment.

    Raises:
        HTTPException: If not assigned.
    """
    logger.info(
        "Ending teacher assignment: teacher=%s, class=%s, by=%s",
        teacher_id,
        class_id,
        current_user.id,
    )

    service = _get_assignment_service(db)

    try:
        return await service.end_assignment(
            class_id=class_id,
            teacher_id=teacher_id,
            subject_framework_code=subject_framework_code,
            subject_code=subject_code,
            ended_at=data.ended_at if data else None,
            ended_by=current_user.id,
        )
    except NotAssignedError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/{class_id}/teachers/{teacher_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove assignment",
    description="Permanently remove an assignment record. Requires tenant admin access.",
)
async def remove_teacher_assignment(
    class_id: UUID,
    teacher_id: UUID,
    subject_framework_code: Annotated[str | None, Query(description="Subject framework code")] = None,
    subject_code: Annotated[str | None, Query(description="Subject code")] = None,
    current_user: CurrentUser = Depends(require_tenant_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> None:
    """Permanently remove an assignment record.

    Only tenant admins can remove assignments.

    Args:
        class_id: Class identifier.
        teacher_id: Teacher identifier.
        subject_framework_code: Optional subject framework code (e.g., "UK-NC-2014").
        subject_code: Optional subject code (e.g., "MAT").
        current_user: Authenticated tenant admin.
        tenant: Tenant context.
        db: Database session.

    Raises:
        HTTPException: If assignment not found.
    """
    logger.info(
        "Removing teacher assignment: teacher=%s, class=%s, by=%s",
        teacher_id,
        class_id,
        current_user.id,
    )

    service = _get_assignment_service(db)

    try:
        await service.remove_assignment(
            class_id=class_id,
            teacher_id=teacher_id,
            subject_framework_code=subject_framework_code,
            subject_code=subject_code,
            removed_by=current_user.id,
        )
    except NotAssignedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found",
        )
