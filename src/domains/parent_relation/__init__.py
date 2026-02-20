# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parent-student relation domain package.

This package provides parent-student relationship management functionality:
- Creating parent-student relationships
- Managing relationship permissions
- Verifying relationships
"""

from src.domains.parent_relation.service import (
    ParentRelationService,
    ParentRelationServiceError,
    RelationNotFoundError,
    ParentNotFoundError,
    StudentNotFoundError,
    RelationExistsError,
    InvalidUserTypeError,
)

__all__ = [
    "ParentRelationService",
    "ParentRelationServiceError",
    "RelationNotFoundError",
    "ParentNotFoundError",
    "StudentNotFoundError",
    "RelationExistsError",
    "InvalidUserTypeError",
]
