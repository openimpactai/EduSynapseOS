# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion domain - emotional support and activity guidance.

This domain provides the Companion Agent functionality through
a unified chat interface with LLM tool calling:
- Proactive greetings
- Emotional support
- Activity suggestions
- Navigation guidance
- Tutor handoff

The companion uses CompanionWorkflow with interrupt_before pattern
for reliable multi-turn conversations.

ProactiveService integration is handled via companion_system_alert_handler,
which is registered at application startup.

Note: CompanionService should be imported directly from
src.domains.companion.service to avoid circular imports with
CompanionWorkflow.
"""

from src.domains.companion.proactive_handler import companion_system_alert_handler
from src.domains.companion.schemas import (
    CompanionChatRequest,
    CompanionChatResponse,
    CompanionMessageResponse,
    CompanionMessagesResponse,
    CompanionSessionResponse,
    EmotionalState,
)

# Note: CompanionService is NOT imported here to avoid circular imports.
# Import it directly: from src.domains.companion.service import CompanionService

__all__ = [
    # Proactive Handler
    "companion_system_alert_handler",
    # Schemas
    "CompanionChatRequest",
    "CompanionChatResponse",
    "CompanionSessionResponse",
    "CompanionMessageResponse",
    "CompanionMessagesResponse",
    "EmotionalState",
]


def __getattr__(name: str):
    """Lazy import for CompanionService to avoid circular imports."""
    if name == "CompanionService":
        from src.domains.companion.service import CompanionService
        return CompanionService
    if name == "CompanionServiceError":
        from src.domains.companion.service import CompanionServiceError
        return CompanionServiceError
    if name == "SessionNotFoundError":
        from src.domains.companion.service import SessionNotFoundError
        return SessionNotFoundError
    if name == "SessionNotActiveError":
        from src.domains.companion.service import SessionNotActiveError
        return SessionNotActiveError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
