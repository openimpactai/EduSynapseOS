# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation API endpoints.

This module provides endpoints for AI-powered H5P content creation:
- POST /chat - Interactive content creation conversation
- GET /sessions/{session_id} - Get session info
- GET /sessions/{session_id}/messages - Get message history
- POST /sessions/{session_id}/end - End a session
- GET /content-types - List available content types
- POST /export - Export generated content to H5P

The content creation system uses LangGraph workflows with
specialized agents for different content types.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_checkpointer,
    get_tenant_db,
    require_auth,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.config import get_settings
from src.core.intelligence.llm import LLMClient
from src.domains.content_creation.schemas import (
    ContentChatRequest,
    ContentChatResponse,
    ContentMessagesResponse,
    ContentSessionResponse,
    ContentTypesResponse,
)
from src.domains.content_creation.service import (
    ContentCreationService,
    ContentSessionNotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_content_creation_service(
    db: AsyncSession,
    checkpointer: BaseCheckpointSaver | None,
) -> ContentCreationService:
    """Create a ContentCreationService instance with dependencies.

    Args:
        db: Database session.
        checkpointer: LangGraph checkpointer.

    Returns:
        Configured ContentCreationService.
    """
    settings = get_settings()
    llm_client = LLMClient(llm_settings=settings.llm)

    return ContentCreationService(
        db=db,
        llm_client=llm_client,
        checkpointer=checkpointer,
    )


@router.post("/chat", response_model=ContentChatResponse)
async def chat(
    request: ContentChatRequest,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Interactive content creation chat endpoint.

    This endpoint handles all content creation interactions:
    - New sessions: Omit session_id to start a new conversation
    - Continue conversation: Provide session_id and message

    All curriculum-related values (framework_code, subject_code, topic_code,
    grade_level, country_code) come from:
    - User profile (user.preferences)
    - Tenant settings (tenant.settings)
    - Request context (request.context)

    NO HARDCODED DEFAULTS - curriculum data is always external.

    The system uses specialized agents for different content types:
    - Quiz Generator: Multiple choice, true/false, fill blanks
    - Vocabulary Generator: Flashcards, dialog cards, crossword
    - Game Generator: Memory game, timeline
    - Learning Generator: Course presentation, interactive book

    Args:
        request: Chat request with message and optional session_id.
        user: Authenticated user (teacher or admin).
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        ContentChatResponse with assistant message and generated content.

    Raises:
        HTTPException: On session not found or server error.
    """
    service = _get_content_creation_service(db, checkpointer)

    # Extract curriculum data from external sources - NO HARDCODED VALUES
    # Priority: request.context > user.preferences > tenant.settings
    context = request.context or {}
    user_prefs = getattr(user, "preferences", {}) or {}
    tenant_settings = getattr(tenant, "settings", {}) or {}

    # User role from user profile or context
    user_role = context.get("user_role") or user_prefs.get("role")

    # Language priority: request > user > tenant > None (will be detected)
    language = context.get("language") or user_prefs.get("language") or tenant_settings.get("default_language")

    # Country code from user or tenant
    country_code = context.get("country_code") or user_prefs.get("country_code") or tenant_settings.get("country_code")

    # Curriculum framework from tenant settings (each tenant has their curriculum)
    framework_code = context.get("framework_code") or tenant_settings.get("framework_code")

    # Subject, topic, grade from context (user selection in UI)
    subject_code = context.get("subject_code")
    topic_code = context.get("topic_code")
    grade_level = context.get("grade_level") or user_prefs.get("default_grade_level")

    try:
        response = await service.chat(
            user_id=user.id,
            tenant_id=tenant.id,
            tenant_code=tenant.code,
            request=request,
            user_role=user_role,
            language=language,
            country_code=country_code,
            framework_code=framework_code,
            subject_code=subject_code,
            topic_code=topic_code,
            grade_level=grade_level,
        )
        return response

    except ContentSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Content creation chat failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content creation failed",
        )


@router.get("/sessions/{session_id}", response_model=ContentSessionResponse)
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Get content creation session information.

    Args:
        session_id: Session identifier.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Session information including status and counts.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_content_creation_service(db, checkpointer)

    try:
        return await service.get_session(
            session_id=session_id,
            user_id=user.id,
        )
    except ContentSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Get session failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session",
        )


@router.get("/sessions/{session_id}/messages", response_model=ContentMessagesResponse)
async def get_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    before_id: UUID | None = None,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Get message history for a content creation session.

    Args:
        session_id: Session identifier.
        limit: Maximum messages to return (1-100, default 50).
        before_id: Get messages before this ID for pagination.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Message history with pagination info.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_content_creation_service(db, checkpointer)

    try:
        return await service.get_messages(
            session_id=session_id,
            user_id=user.id,
            limit=limit,
            before_id=before_id,
        )
    except ContentSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Get messages failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get messages",
        )


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """End a content creation session.

    Args:
        session_id: Session identifier.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Session summary with counts.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_content_creation_service(db, checkpointer)

    try:
        result = await service.end_session(
            session_id=session_id,
            user_id=user.id,
        )
        return {"success": True, **result}

    except ContentSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("End session failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session",
        )


@router.get("/content-types", response_model=ContentTypesResponse)
async def list_content_types(
    category: str | None = Query(
        default=None,
        description="Filter by category (assessment, vocabulary, learning, game, media)",
    ),
    ai_support: str | None = Query(
        default=None,
        description="Filter by AI support level (full, partial, none)",
    ),
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """List available H5P content types.

    Returns content types that can be generated by the AI system,
    with information about their capabilities and requirements.

    Args:
        category: Filter by category.
        ai_support: Filter by AI support level.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        List of content types with metadata.
    """
    service = _get_content_creation_service(db, checkpointer)

    try:
        return await service.get_content_types(
            category=category,
            ai_support=ai_support,
        )
    except Exception as e:
        logger.exception("List content types failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list content types",
        )
