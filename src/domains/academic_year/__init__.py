# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Academic year domain package.

This package provides academic year management functionality including:
- Academic year CRUD operations
- Setting current academic year
- Year overlap validation
"""

from src.domains.academic_year.service import (
    AcademicYearService,
    AcademicYearServiceError,
    AcademicYearNotFoundError,
    AcademicYearOverlapError,
)

__all__ = [
    "AcademicYearService",
    "AcademicYearServiceError",
    "AcademicYearNotFoundError",
    "AcademicYearOverlapError",
]
