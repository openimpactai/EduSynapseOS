# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Curriculum domain services.

This package provides curriculum services:
- CurriculumLookup: Code-based lookups for curriculum entities
- TopicContext: Educational context derived from curriculum hierarchy
- CurriculumSyncService: Sync curriculum data from Central Curriculum

Curriculum data is synced from the Central Curriculum service and stored
locally using code-based composite primary keys for consistent identification
across environments.
"""

from src.domains.curriculum.lookup import (
    CurriculumLookup,
    TopicContext,
    COUNTRY_TO_LANGUAGE,
    DEFAULT_LANGUAGE,
)
from src.domains.curriculum.sync_service import (
    CurriculumSyncService,
    CurriculumSyncError,
    SyncResult,
)

__all__ = [
    # Lookup
    "CurriculumLookup",
    "TopicContext",
    "COUNTRY_TO_LANGUAGE",
    "DEFAULT_LANGUAGE",
    # Sync
    "CurriculumSyncService",
    "CurriculumSyncError",
    "SyncResult",
]
