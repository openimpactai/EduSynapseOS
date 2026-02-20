# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""School service for tenant school management.

This module provides the SchoolService that handles:
- School CRUD operations
- School admin assignment
- School statistics and counts

Example:
    >>> school_service = SchoolService(db_session)
    >>> school = await school_service.create_school(request)
    >>> schools = await school_service.list_schools(limit=20)
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.tenant.school import (
    School,
    Class,
    ClassStudent,
    ClassTeacher,
)
from src.infrastructure.database.models.tenant.user import User, UserRole, Role
from src.models.school import (
    SchoolCreateRequest,
    SchoolUpdateRequest,
    SchoolResponse,
    SchoolSummary,
    SchoolAdminResponse,
)
from src.models.user import UserSummary
from src.models.common import StatusEnum

logger = logging.getLogger(__name__)


class SchoolServiceError(Exception):
    """Base exception for school service errors."""

    pass


class SchoolNotFoundError(SchoolServiceError):
    """Raised when a school is not found."""

    pass


class SchoolCodeExistsError(SchoolServiceError):
    """Raised when trying to create a school with existing code."""

    pass


class SchoolAdminError(SchoolServiceError):
    """Raised when school admin operations fail."""

    pass


class SchoolService:
    """Service for managing tenant schools.

    Handles all school CRUD operations, admin assignment, and statistics.
    Each school belongs to exactly one tenant (database isolation).

    Attributes:
        _db: Async database session.

    Example:
        >>> service = SchoolService(db)
        >>> school = await service.create_school(create_request)
        >>> await service.assign_admin(school.id, user_id)
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the school service.

        Args:
            db: Async database session.
        """
        self._db = db

    async def create_school(
        self,
        request: SchoolCreateRequest,
        created_by: str | None = None,
    ) -> SchoolResponse:
        """Create a new school.

        Args:
            request: School creation request.
            created_by: ID of user creating the school.

        Returns:
            Created school response.

        Raises:
            SchoolCodeExistsError: If school code already exists.
        """
        # Check if code already exists
        existing = await self._get_by_code(request.code)
        if existing:
            raise SchoolCodeExistsError(f"School with code '{request.code}' already exists")

        # Build settings
        settings: dict = {}
        if created_by:
            settings["created_by"] = created_by

        # Create school
        school = School(
            code=request.code,
            name=request.name,
            school_type=request.school_type,
            address_line1=request.address_line1,
            address_line2=request.address_line2,
            city=request.city,
            state_province=request.state_province,
            postal_code=request.postal_code,
            country_code=request.country_code,
            phone=request.phone,
            email=request.email,
            website=request.website,
            timezone=request.timezone,
            settings=settings,
            is_active=True,
        )

        self._db.add(school)
        await self._db.commit()
        await self._db.refresh(school)

        logger.info("School created: %s (code=%s)", school.id, school.code)

        return await self._to_response(school)

    async def list_schools(
        self,
        is_active: bool | None = None,
        search: str | None = None,
        school_ids: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[SchoolSummary], int]:
        """List schools with optional filtering.

        Args:
            is_active: Filter by active status.
            search: Search by name or code.
            school_ids: Filter to specific school IDs (for school_admin access).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Tuple of (school summaries, total count).
        """
        # Build base query (exclude soft-deleted)
        stmt = select(School).where(School.deleted_at.is_(None))

        # Apply filters
        if is_active is not None:
            stmt = stmt.where(School.is_active == is_active)

        if search:
            search_pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    School.name.ilike(search_pattern),
                    School.code.ilike(search_pattern),
                    School.city.ilike(search_pattern),
                )
            )

        if school_ids:
            stmt = stmt.where(School.id.in_(school_ids))

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self._db.execute(count_stmt)
        total = count_result.scalar() or 0

        # Apply pagination and ordering
        stmt = stmt.order_by(School.name.asc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._db.execute(stmt)
        schools = result.scalars().all()

        summaries = []
        for school in schools:
            summary = await self._to_summary(school)
            summaries.append(summary)

        return summaries, total

    async def get_school(
        self,
        school_id: str | UUID,
        include_stats: bool = True,
    ) -> SchoolResponse:
        """Get a school by ID.

        Args:
            school_id: School identifier.
            include_stats: Whether to include class/student/teacher counts.

        Returns:
            School response.

        Raises:
            SchoolNotFoundError: If school not found.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        return await self._to_response(school, include_stats=include_stats)

    async def update_school(
        self,
        school_id: str | UUID,
        request: SchoolUpdateRequest,
    ) -> SchoolResponse:
        """Update a school.

        Args:
            school_id: School identifier.
            request: Update request with fields to change.

        Returns:
            Updated school response.

        Raises:
            SchoolNotFoundError: If school not found.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        # Update fields if provided
        if request.name is not None:
            school.name = request.name
        if request.school_type is not None:
            school.school_type = request.school_type
        if request.address_line1 is not None:
            school.address_line1 = request.address_line1
        if request.address_line2 is not None:
            school.address_line2 = request.address_line2
        if request.city is not None:
            school.city = request.city
        if request.state_province is not None:
            school.state_province = request.state_province
        if request.postal_code is not None:
            school.postal_code = request.postal_code
        if request.country_code is not None:
            school.country_code = request.country_code
        if request.phone is not None:
            school.phone = request.phone
        if request.email is not None:
            school.email = request.email
        if request.website is not None:
            school.website = request.website
        if request.timezone is not None:
            school.timezone = request.timezone

        await self._db.commit()
        await self._db.refresh(school)

        logger.info("School updated: %s", school.id)

        return await self._to_response(school)

    async def deactivate_school(self, school_id: str | UUID) -> None:
        """Deactivate a school (soft delete).

        Args:
            school_id: School identifier.

        Raises:
            SchoolNotFoundError: If school not found.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        school.is_active = False
        school.deleted_at = datetime.now(timezone.utc)

        await self._db.commit()

        logger.info("School deactivated: %s", school.id)

    async def activate_school(self, school_id: str | UUID) -> SchoolResponse:
        """Reactivate a deactivated school.

        Args:
            school_id: School identifier.

        Returns:
            Activated school response.

        Raises:
            SchoolNotFoundError: If school not found.
        """
        # Get including soft-deleted
        stmt = select(School).where(School.id == str(school_id))
        result = await self._db.execute(stmt)
        school = result.scalar_one_or_none()

        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        school.is_active = True
        school.deleted_at = None

        await self._db.commit()
        await self._db.refresh(school)

        logger.info("School activated: %s", school.id)

        return await self._to_response(school)

    async def assign_admin(
        self,
        school_id: str | UUID,
        user_id: str | UUID,
        assigned_by: str | None = None,
    ) -> None:
        """Assign a user as school admin.

        Creates a UserRole with school_admin role scoped to this school.

        Args:
            school_id: School identifier.
            user_id: User identifier to assign as admin.
            assigned_by: ID of user making the assignment.

        Raises:
            SchoolNotFoundError: If school not found.
            SchoolAdminError: If user not found or already admin.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        # Verify user exists and is not already a school_admin for this school
        user = await self._get_user_by_id(str(user_id))
        if not user:
            raise SchoolAdminError(f"User {user_id} not found")

        # Check if already assigned
        existing = await self._get_admin_assignment(str(school_id), str(user_id))
        if existing:
            raise SchoolAdminError(f"User {user_id} is already admin for school {school_id}")

        # Find school_admin role
        role = await self._get_role_by_code("school_admin")
        if not role:
            raise SchoolAdminError("school_admin role not found in database")

        # Create role assignment
        user_role = UserRole(
            user_id=str(user_id),
            role_id=role.id,
            school_id=str(school_id),
            granted_by=assigned_by,
        )

        # Update user's user_type if not already an admin
        if user.user_type not in ("school_admin", "tenant_admin"):
            user.user_type = "school_admin"

        self._db.add(user_role)
        await self._db.commit()

        logger.info(
            "School admin assigned: user=%s, school=%s, by=%s",
            user_id, school_id, assigned_by,
        )

    async def remove_admin(
        self,
        school_id: str | UUID,
        user_id: str | UUID,
    ) -> None:
        """Remove a user as school admin.

        Args:
            school_id: School identifier.
            user_id: User identifier to remove as admin.

        Raises:
            SchoolNotFoundError: If school not found.
            SchoolAdminError: If user is not admin for this school.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        # Find the assignment
        assignment = await self._get_admin_assignment(str(school_id), str(user_id))
        if not assignment:
            raise SchoolAdminError(f"User {user_id} is not admin for school {school_id}")

        await self._db.delete(assignment)

        # Check if user has other school_admin assignments
        other_assignments = await self._count_admin_assignments(str(user_id))
        if other_assignments == 0:
            # Revert user_type to teacher or original type
            user = await self._get_user_by_id(str(user_id))
            if user and user.user_type == "school_admin":
                user.user_type = "teacher"

        await self._db.commit()

        logger.info("School admin removed: user=%s, school=%s", user_id, school_id)

    async def list_admins(
        self,
        school_id: str | UUID,
    ) -> list[SchoolAdminResponse]:
        """List all admins for a school.

        Args:
            school_id: School identifier.

        Returns:
            List of school admin responses.

        Raises:
            SchoolNotFoundError: If school not found.
        """
        school = await self._get_by_id(str(school_id))
        if not school:
            raise SchoolNotFoundError(f"School {school_id} not found")

        # Find school_admin role
        role = await self._get_role_by_code("school_admin")
        if not role:
            return []

        # Get all user_role assignments for this school with school_admin role
        stmt = (
            select(UserRole)
            .options(selectinload(UserRole.user))
            .where(
                and_(
                    UserRole.school_id == str(school_id),
                    UserRole.role_id == role.id,
                )
            )
            .order_by(UserRole.created_at.desc())
        )

        result = await self._db.execute(stmt)
        assignments = result.scalars().all()

        admins = []
        for assignment in assignments:
            if assignment.user and assignment.user.deleted_at is None:
                admins.append(
                    SchoolAdminResponse(
                        user=self._user_to_summary(assignment.user),
                        assigned_at=assignment.created_at,
                        assigned_by=UUID(assignment.granted_by) if assignment.granted_by else None,
                    )
                )

        return admins

    async def check_user_access(
        self,
        user_id: str,
        school_id: str,
    ) -> bool:
        """Check if a user has access to a school.

        Tenant admins have access to all schools.
        School admins have access to their assigned schools.

        Args:
            user_id: User identifier.
            school_id: School identifier.

        Returns:
            True if user has access.
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            return False

        # Tenant admins have access to all schools
        if user.user_type == "tenant_admin":
            return True

        # Check school_admin assignment
        assignment = await self._get_admin_assignment(school_id, user_id)
        return assignment is not None

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _get_by_id(self, school_id: str) -> School | None:
        """Get school by ID (excluding soft-deleted)."""
        stmt = select(School).where(
            School.id == school_id,
            School.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_by_code(self, code: str) -> School | None:
        """Get school by code (excluding soft-deleted)."""
        stmt = select(School).where(
            School.code == code.lower(),
            School.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        stmt = select(User).where(
            User.id == user_id,
            User.deleted_at.is_(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_role_by_code(self, code: str) -> Role | None:
        """Get role by code."""
        stmt = select(Role).where(Role.code == code)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_admin_assignment(
        self,
        school_id: str,
        user_id: str,
    ) -> UserRole | None:
        """Get school admin assignment."""
        role = await self._get_role_by_code("school_admin")
        if not role:
            return None

        stmt = select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.school_id == school_id,
            UserRole.role_id == role.id,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _count_admin_assignments(self, user_id: str) -> int:
        """Count school_admin assignments for a user."""
        role = await self._get_role_by_code("school_admin")
        if not role:
            return 0

        stmt = select(func.count()).select_from(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role.id,
            UserRole.school_id.isnot(None),
        )
        result = await self._db.execute(stmt)
        return result.scalar() or 0

    async def _get_school_stats(self, school_id: str) -> dict:
        """Get school statistics (class, student, teacher counts)."""
        # Count active classes
        class_stmt = select(func.count()).select_from(Class).where(
            Class.school_id == school_id,
            Class.is_active == True,
        )
        class_result = await self._db.execute(class_stmt)
        class_count = class_result.scalar() or 0

        # Count students enrolled in active classes
        student_stmt = (
            select(func.count(func.distinct(ClassStudent.student_id)))
            .select_from(ClassStudent)
            .join(Class, ClassStudent.class_id == Class.id)
            .where(
                Class.school_id == school_id,
                Class.is_active == True,
                ClassStudent.status == "active",
            )
        )
        student_result = await self._db.execute(student_stmt)
        student_count = student_result.scalar() or 0

        # Count teachers assigned to active classes
        teacher_stmt = (
            select(func.count(func.distinct(ClassTeacher.teacher_id)))
            .select_from(ClassTeacher)
            .join(Class, ClassTeacher.class_id == Class.id)
            .where(
                Class.school_id == school_id,
                Class.is_active == True,
                ClassTeacher.ended_at.is_(None),
            )
        )
        teacher_result = await self._db.execute(teacher_stmt)
        teacher_count = teacher_result.scalar() or 0

        return {
            "class_count": class_count,
            "student_count": student_count,
            "teacher_count": teacher_count,
        }

    async def _to_response(
        self,
        school: School,
        include_stats: bool = True,
    ) -> SchoolResponse:
        """Convert School model to SchoolResponse."""
        stats = {"class_count": 0, "student_count": 0, "teacher_count": 0}
        if include_stats:
            stats = await self._get_school_stats(school.id)

        return SchoolResponse(
            id=UUID(school.id),
            code=school.code,
            name=school.name,
            school_type=school.school_type,
            address_line1=school.address_line1,
            address_line2=school.address_line2,
            city=school.city,
            state_province=school.state_province,
            postal_code=school.postal_code,
            country_code=school.country_code,
            phone=school.phone,
            email=school.email,
            website=school.website,
            timezone=school.timezone,
            is_active=school.is_active,
            class_count=stats["class_count"],
            student_count=stats["student_count"],
            teacher_count=stats["teacher_count"],
            created_at=school.created_at,
            updated_at=school.updated_at,
        )

    async def _to_summary(self, school: School) -> SchoolSummary:
        """Convert School model to SchoolSummary."""
        stats = await self._get_school_stats(school.id)

        return SchoolSummary(
            id=UUID(school.id),
            code=school.code,
            name=school.name,
            school_type=school.school_type,
            city=school.city,
            is_active=school.is_active,
            class_count=stats["class_count"],
            student_count=stats["student_count"],
        )

    def _user_to_summary(self, user: User) -> UserSummary:
        """Convert User model to UserSummary."""
        return UserSummary(
            id=UUID(user.id),
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
            user_type=user.user_type,
            status=StatusEnum(user.status),
            sso_external_id=user.sso_external_id,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
