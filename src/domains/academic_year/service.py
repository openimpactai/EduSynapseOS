# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Academic year service for managing academic year operations.

This module provides the AcademicYearService class for:
- Academic year CRUD operations
- Setting current academic year
- Academic year validation
"""

from __future__ import annotations

import logging
from datetime import date
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.school import AcademicYear, Class
from src.models.academic_year import (
    AcademicYearCreateRequest,
    AcademicYearUpdateRequest,
    AcademicYearResponse,
    AcademicYearSummary,
)

logger = logging.getLogger(__name__)


class AcademicYearServiceError(Exception):
    """Base exception for academic year service errors."""

    pass


class AcademicYearNotFoundError(AcademicYearServiceError):
    """Raised when academic year is not found."""

    pass


class AcademicYearOverlapError(AcademicYearServiceError):
    """Raised when academic year dates overlap with existing year."""

    pass


class AcademicYearService:
    """Service for managing academic years.

    This service handles all academic year operations including
    creating, updating, deleting, and setting the current year.

    Attributes:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize academic year service.

        Args:
            db: Async database session for tenant database.
        """
        self.db = db

    async def create_academic_year(
        self,
        request: AcademicYearCreateRequest,
    ) -> AcademicYearResponse:
        """Create a new academic year.

        Args:
            request: Academic year creation data.

        Returns:
            Created academic year response.

        Raises:
            AcademicYearOverlapError: If dates overlap with existing year.
        """
        # Check for overlapping years
        overlap_query = select(AcademicYear).where(
            AcademicYear.start_date <= request.end_date,
            AcademicYear.end_date >= request.start_date,
        )
        result = await self.db.execute(overlap_query)
        if result.scalar_one_or_none():
            raise AcademicYearOverlapError(
                f"Academic year dates overlap with an existing year"
            )

        # If setting as current, unset other current years
        if request.is_current:
            await self._unset_current_year()

        academic_year = AcademicYear(
            code=request.name,
            name=request.name,
            start_date=request.start_date,
            end_date=request.end_date,
            is_current=request.is_current,
        )

        self.db.add(academic_year)
        await self.db.commit()
        await self.db.refresh(academic_year)

        logger.info("Created academic year: %s (%s)", academic_year.name, academic_year.id)

        return await self._to_response(academic_year)

    async def list_academic_years(
        self,
        include_past: bool = True,
    ) -> tuple[list[AcademicYearSummary], int]:
        """List all academic years.

        Args:
            include_past: Whether to include past years.

        Returns:
            Tuple of (list of academic years, total count).
        """
        query = select(AcademicYear)

        if not include_past:
            today = date.today()
            query = query.where(AcademicYear.end_date >= today)

        query = query.order_by(AcademicYear.start_date.desc())

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Execute main query
        result = await self.db.execute(query)
        years = result.scalars().all()

        items = [self._to_summary(year) for year in years]

        return items, total

    async def get_academic_year(self, year_id: UUID) -> AcademicYearResponse:
        """Get academic year by ID.

        Args:
            year_id: Academic year identifier.

        Returns:
            Academic year details.

        Raises:
            AcademicYearNotFoundError: If year not found.
        """
        academic_year = await self._get_by_id(year_id)
        return await self._to_response(academic_year)

    async def get_current_year(self) -> AcademicYearResponse | None:
        """Get the current academic year.

        Returns:
            Current academic year or None if not set.
        """
        query = select(AcademicYear).where(AcademicYear.is_current == True)
        result = await self.db.execute(query)
        academic_year = result.scalar_one_or_none()

        if not academic_year:
            return None

        return await self._to_response(academic_year)

    async def update_academic_year(
        self,
        year_id: UUID,
        request: AcademicYearUpdateRequest,
    ) -> AcademicYearResponse:
        """Update an academic year.

        Args:
            year_id: Academic year identifier.
            request: Update data.

        Returns:
            Updated academic year.

        Raises:
            AcademicYearNotFoundError: If year not found.
            AcademicYearOverlapError: If new dates overlap.
        """
        academic_year = await self._get_by_id(year_id)

        # Determine new dates
        new_start = request.start_date or academic_year.start_date
        new_end = request.end_date or academic_year.end_date

        # Validate date order
        if new_end <= new_start:
            raise AcademicYearOverlapError("End date must be after start date")

        # Check for overlaps with other years (exclude current year)
        if request.start_date or request.end_date:
            overlap_query = select(AcademicYear).where(
                AcademicYear.id != str(year_id),
                AcademicYear.start_date <= new_end,
                AcademicYear.end_date >= new_start,
            )
            result = await self.db.execute(overlap_query)
            if result.scalar_one_or_none():
                raise AcademicYearOverlapError(
                    "Academic year dates overlap with an existing year"
                )

        # Apply updates
        if request.name is not None:
            academic_year.name = request.name
        if request.start_date is not None:
            academic_year.start_date = request.start_date
        if request.end_date is not None:
            academic_year.end_date = request.end_date

        await self.db.commit()
        await self.db.refresh(academic_year)

        logger.info("Updated academic year: %s", year_id)

        return await self._to_response(academic_year)

    async def delete_academic_year(self, year_id: UUID) -> None:
        """Delete an academic year.

        Args:
            year_id: Academic year identifier.

        Raises:
            AcademicYearNotFoundError: If year not found.
            AcademicYearServiceError: If year has associated classes.
        """
        academic_year = await self._get_by_id(year_id)

        # Check for associated classes
        class_count = await self._get_class_count(str(year_id))
        if class_count > 0:
            raise AcademicYearServiceError(
                f"Cannot delete academic year with {class_count} associated classes"
            )

        await self.db.delete(academic_year)
        await self.db.commit()

        logger.info("Deleted academic year: %s", year_id)

    async def set_current_year(self, year_id: UUID) -> AcademicYearResponse:
        """Set an academic year as current.

        Args:
            year_id: Academic year identifier.

        Returns:
            Updated academic year.

        Raises:
            AcademicYearNotFoundError: If year not found.
        """
        academic_year = await self._get_by_id(year_id)

        # Unset other current years
        await self._unset_current_year()

        # Set this year as current
        academic_year.is_current = True

        await self.db.commit()
        await self.db.refresh(academic_year)

        logger.info("Set academic year %s as current", year_id)

        return await self._to_response(academic_year)

    async def _get_by_id(self, year_id: UUID) -> AcademicYear:
        """Get academic year by ID.

        Args:
            year_id: Academic year identifier.

        Returns:
            AcademicYear model instance.

        Raises:
            AcademicYearNotFoundError: If not found.
        """
        query = select(AcademicYear).where(AcademicYear.id == str(year_id))
        result = await self.db.execute(query)
        academic_year = result.scalar_one_or_none()

        if not academic_year:
            raise AcademicYearNotFoundError(f"Academic year {year_id} not found")

        return academic_year

    async def _unset_current_year(self) -> None:
        """Unset any current academic year."""
        stmt = update(AcademicYear).where(
            AcademicYear.is_current == True
        ).values(is_current=False)
        await self.db.execute(stmt)

    async def _get_class_count(self, year_id: str) -> int:
        """Get count of classes in an academic year.

        Args:
            year_id: Academic year identifier.

        Returns:
            Number of classes.
        """
        query = select(func.count()).select_from(Class).where(
            Class.academic_year_id == year_id
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def _to_response(self, academic_year: AcademicYear) -> AcademicYearResponse:
        """Convert academic year model to response DTO.

        Args:
            academic_year: AcademicYear model instance.

        Returns:
            AcademicYearResponse DTO.
        """
        class_count = await self._get_class_count(academic_year.id)
        today = date.today()

        return AcademicYearResponse(
            id=UUID(academic_year.id),
            name=academic_year.name,
            start_date=academic_year.start_date,
            end_date=academic_year.end_date,
            is_current=academic_year.is_current,
            is_active=academic_year.start_date <= today <= academic_year.end_date,
            created_at=academic_year.created_at,
            class_count=class_count,
        )

    def _to_summary(self, academic_year: AcademicYear) -> AcademicYearSummary:
        """Convert academic year model to summary DTO.

        Args:
            academic_year: AcademicYear model instance.

        Returns:
            AcademicYearSummary DTO.
        """
        today = date.today()

        return AcademicYearSummary(
            id=UUID(academic_year.id),
            name=academic_year.name,
            start_date=academic_year.start_date,
            end_date=academic_year.end_date,
            is_current=academic_year.is_current,
            is_active=academic_year.start_date <= today <= academic_year.end_date,
        )
