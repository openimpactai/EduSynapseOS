# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher API endpoints.

This module provides endpoints for teacher assistant interactions:
- POST /chat - Unified chat endpoint for all interactions
- GET /sessions/{session_id} - Get session info
- GET /sessions/{session_id}/messages - Get message history
- POST /sessions/{session_id}/end - End a session

The teacher assistant uses a unified chat endpoint with tool calling.
All interactions (class viewing, student monitoring, analytics)
flow through the same endpoint with the LLM deciding what actions to take.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    Checkpointer,
    TenantDB,
    get_checkpointer,
    get_tenant_db,
    require_teacher,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.config import get_settings
from src.core.intelligence.llm import LLMClient
from src.core.personas.manager import get_persona_manager
from src.domains.teacher.schemas import (
    TeacherChatRequest,
    TeacherChatResponse,
    TeacherMessagesResponse,
    TeacherSessionResponse,
)
from src.domains.teacher.service import (
    TeacherCompanionService,
    TeacherSessionNotActiveError,
    TeacherSessionNotFoundError,
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_LANGUAGE = "en"

router = APIRouter()


def _get_teacher_service(
    db: AsyncSession,
    checkpointer: BaseCheckpointSaver | None,
) -> TeacherCompanionService:
    """Create a TeacherCompanionService instance with all dependencies.

    Args:
        db: Tenant database session.
        checkpointer: LangGraph checkpointer for state persistence.

    Returns:
        Configured TeacherCompanionService instance.
    """
    settings = get_settings()

    # Create LLM client
    llm_client = LLMClient(llm_settings=settings.llm)

    # Get persona manager
    persona_manager = get_persona_manager()

    return TeacherCompanionService(
        db=db,
        llm_client=llm_client,
        persona_manager=persona_manager,
        checkpointer=checkpointer,
    )


@router.post("/chat", response_model=TeacherChatResponse)
async def chat(
    request: TeacherChatRequest,
    user: CurrentUser = Depends(require_teacher),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Unified chat endpoint for teacher assistant interactions.

    This endpoint handles all teacher assistant interactions through a single interface:
    - New sessions: Omit session_id to start a new conversation with greeting
    - Continue conversation: Provide session_id and message to continue

    The assistant uses tool calling to:
    - Get teacher's classes (get_my_classes)
    - Get students in a class (get_class_students)
    - Get student progress and mastery (get_student_progress, get_student_mastery)
    - Get class analytics (get_class_analytics)
    - Identify struggling students (get_struggling_students)
    - Get topic performance (get_topic_performance)
    - Get student notes (get_student_notes)
    - Get alerts (get_alerts)
    - Get emotional history (get_emotional_history)

    Args:
        request: Chat request with optional session_id and message.
        user: Authenticated teacher user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        TeacherChatResponse with message, suggestions, and tool data.

    Raises:
        HTTPException: On session not found, not active, or server error.
    """
    service = _get_teacher_service(db, checkpointer)

    try:
        response = await service.chat(
            user_id=user.id,
            tenant_id=tenant.id,
            tenant_code=tenant.code,
            request=request,
            language=user.preferred_language,
        )
        return response

    except TeacherSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except TeacherSessionNotActiveError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Teacher chat failed: %s", e)
        # Ensure transaction is rolled back to prevent connection pollution
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat operation failed",
        )


@router.get("/sessions/{session_id}", response_model=TeacherSessionResponse)
async def get_session(
    session_id: UUID,
    user: CurrentUser = Depends(require_teacher),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Get session information.

    Args:
        session_id: Session identifier.
        user: Authenticated teacher user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Session information including status and message count.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_teacher_service(db, checkpointer)

    try:
        return await service.get_session(
            session_id=session_id,
            user_id=user.id,
        )
    except TeacherSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Get teacher session failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session",
        )


@router.get("/sessions/{session_id}/messages", response_model=TeacherMessagesResponse)
async def get_messages(
    session_id: UUID,
    limit: int = 50,
    before_id: UUID | None = None,
    user: CurrentUser = Depends(require_teacher),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """Get message history for a session.

    Args:
        session_id: Session identifier.
        limit: Maximum messages to return (default 50, max 100).
        before_id: Get messages before this message ID (for pagination).
        user: Authenticated teacher user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Message history with pagination info.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_teacher_service(db, checkpointer)

    # Clamp limit
    limit = min(max(1, limit), 100)

    try:
        return await service.get_messages(
            session_id=session_id,
            user_id=user.id,
            limit=limit,
            before_id=before_id,
        )
    except TeacherSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Get teacher messages failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get messages",
        )


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: UUID,
    user: CurrentUser = Depends(require_teacher),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
):
    """End a teacher assistant session.

    Args:
        session_id: Session identifier.
        user: Authenticated teacher user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.

    Returns:
        Success message.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_teacher_service(db, checkpointer)

    try:
        await service.end_session(
            session_id=session_id,
            user_id=user.id,
        )
        return {"success": True, "message": "Session ended"}

    except TeacherSessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("End teacher session failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session",
        )
