# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher domain package.

This package provides teacher-specific services and schemas:
- TeacherCompanionService: Teacher assistant conversation service
- Schemas: Request/Response models for teacher API
"""

from src.domains.teacher.schemas import (
    TeacherChatRequest,
    TeacherChatResponse,
    TeacherSessionResponse,
)
from src.domains.teacher.service import (
    TeacherCompanionService,
    TeacherServiceError,
    TeacherSessionNotActiveError,
    TeacherSessionNotFoundError,
)

__all__ = [
    "TeacherChatRequest",
    "TeacherChatResponse",
    "TeacherSessionResponse",
    "TeacherCompanionService",
    "TeacherServiceError",
    "TeacherSessionNotActiveError",
    "TeacherSessionNotFoundError",
]
