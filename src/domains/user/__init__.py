# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""User domain package.

This package provides user management functionality:
- UserService: CRUD operations for tenant users
- Exceptions: User-related error types

Example:
    >>> from src.domains.user import UserService
    >>> service = UserService(db, password_hasher)
    >>> user = await service.create_user(request)
"""

from src.domains.user.service import (
    UserService,
    UserServiceError,
    UserNotFoundError,
    UserAlreadyExistsError,
    UserOperationError,
)

__all__ = [
    "UserService",
    "UserServiceError",
    "UserNotFoundError",
    "UserAlreadyExistsError",
    "UserOperationError",
]
