# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation Workflow States.

This module defines state structures for content creation workflows.

All curriculum-related data (framework_code, subject_code, topic_code,
grade_level, learning_objectives, country_code) is populated dynamically
via curriculum tools - never hardcoded.
"""

from src.core.orchestration.states.content.content_creation import (
    ContentCreationState,
    ContentTurn,
    GeneratedContent,
    HandoffContext,
    ReviewResult,
    UserAction,
    UserMessageInference,
    create_initial_content_state,
)

__all__ = [
    "ContentCreationState",
    "ContentTurn",
    "GeneratedContent",
    "HandoffContext",
    "ReviewResult",
    "UserAction",
    "UserMessageInference",
    "create_initial_content_state",
]
