# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher assignment domain package.

This package provides teacher assignment management functionality including:
- Teacher assignment to classes
- Subject-specific assignments
- Assignment termination
"""

from src.domains.assignment.service import (
    TeacherAssignmentService,
    AssignmentServiceError,
    ClassNotFoundError,
    TeacherNotFoundError,
    AlreadyAssignedError,
    NotAssignedError,
    InvalidTeacherTypeError,
)

__all__ = [
    "TeacherAssignmentService",
    "AssignmentServiceError",
    "ClassNotFoundError",
    "TeacherNotFoundError",
    "AlreadyAssignedError",
    "NotAssignedError",
    "InvalidTeacherTypeError",
]
