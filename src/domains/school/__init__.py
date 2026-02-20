# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""School domain package.

This package provides school management functionality including:
- School CRUD operations
- School admin assignment
- School statistics
"""

from src.domains.school.service import (
    SchoolService,
    SchoolServiceError,
    SchoolNotFoundError,
    SchoolCodeExistsError,
    SchoolAdminError,
)

__all__ = [
    "SchoolService",
    "SchoolServiceError",
    "SchoolNotFoundError",
    "SchoolCodeExistsError",
    "SchoolAdminError",
]
