# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Enrollment domain package.

This package provides student enrollment management functionality including:
- Student enrollment in classes
- Enrollment withdrawal
- Bulk enrollment operations
"""

from src.domains.enrollment.service import (
    EnrollmentService,
    EnrollmentServiceError,
    ClassNotFoundError,
    StudentNotFoundError,
    AlreadyEnrolledError,
    NotEnrolledError,
    InvalidStudentTypeError,
)

__all__ = [
    "EnrollmentService",
    "EnrollmentServiceError",
    "ClassNotFoundError",
    "StudentNotFoundError",
    "AlreadyEnrolledError",
    "NotEnrolledError",
    "InvalidStudentTypeError",
]
