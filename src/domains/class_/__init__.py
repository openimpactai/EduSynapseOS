# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class domain package.

This package provides class/section management functionality including:
- Class CRUD operations
- Class activation/deactivation
- Student and teacher count tracking
"""

from src.domains.class_.service import (
    ClassService,
    ClassServiceError,
    ClassNotFoundError,
    ClassCodeExistsError,
    SchoolNotFoundError,
    AcademicYearNotFoundError,
)

__all__ = [
    "ClassService",
    "ClassServiceError",
    "ClassNotFoundError",
    "ClassCodeExistsError",
    "SchoolNotFoundError",
    "AcademicYearNotFoundError",
]
