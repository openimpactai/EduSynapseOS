# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System Explainer domain package.

This domain provides an AI agent that explains EduSynapseOS to various
audiences - investors, educators, technical teams, and general users.

The agent uses comprehensive knowledge about the platform's architecture,
educational theories, memory system, and business model to provide
engaging, audience-appropriate explanations.

Example:
    from src.domains.system_explainer import (
        SystemExplainerService,
        ExplainerChatRequest,
        AudienceType,
    )

    service = SystemExplainerService()
    response = await service.chat(
        ExplainerChatRequest(
            message="What is EduSynapseOS?",
            audience=AudienceType.INVESTOR,
        )
    )
"""

from src.domains.system_explainer.schemas import (
    AudienceType,
    ConversationMessage,
    ExplainerChatRequest,
    ExplainerChatResponse,
    ExplainerSessionResponse,
    ExplainerTopicsResponse,
    MessageRole,
    QuickExplainRequest,
    QuickExplainResponse,
)
from src.domains.system_explainer.service import (
    AVAILABLE_TOPICS,
    SystemExplainerService,
    get_system_explainer_service,
)

__all__ = [
    # Schemas
    "AudienceType",
    "MessageRole",
    "ConversationMessage",
    "ExplainerChatRequest",
    "ExplainerChatResponse",
    "ExplainerSessionResponse",
    "ExplainerTopicsResponse",
    "QuickExplainRequest",
    "QuickExplainResponse",
    # Service
    "SystemExplainerService",
    "get_system_explainer_service",
    "AVAILABLE_TOPICS",
]
