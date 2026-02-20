# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice Helper Tutor API endpoints.

This module provides endpoints for practice helper tutoring sessions:
- POST /start - Start a tutoring session when student needs help
- POST /{session_id}/message - Send a message in the session
- GET /{session_id} - Get session details
- GET /{session_id}/history - Get conversation history
- POST /{session_id}/complete - Complete the session

The practice helper is triggered when a student answers incorrectly
during practice and clicks "Get Help" to understand the concept.

Example:
    POST /api/v1/practice-helper/start
    {
        "practice_session_id": "uuid",
        "question_id": "uuid",
        "student_answer": "B"
    }
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_checkpointer,
    get_tenant_db,
    get_tenant_db_manager,
    require_auth,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.domains.practice_helper.service import PracticeHelperService
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.models.practice_helper import (
    CompleteSessionRequest,
    MessageResponse,
    PracticeHelperSessionResponse,
    SendMessageRequest,
    SessionCompletionResponse,
    SessionHistoryResponse,
    StartPracticeHelperRequest,
    StartPracticeHelperResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_practice_helper_service(
    tenant_db_manager: TenantDatabaseManager,
    checkpointer: BaseCheckpointSaver | None,
    tenant: TenantContext,
) -> PracticeHelperService:
    """Get practice helper service instance.

    Args:
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.
        tenant: Tenant context for event tracking.

    Returns:
        Configured PracticeHelperService instance.
    """
    from src.core.agents.capabilities.registry import get_default_registry
    from src.core.intelligence.embeddings.service import EmbeddingService
    from src.core.intelligence.llm.client import LLMClient
    from src.core.memory.manager import MemoryManager
    from src.core.orchestration.workflows import PracticeHelperWorkflow
    from src.core.personas.manager import PersonaManager
    from src.core.agents import AgentFactory
    from src.domains.analytics.events import EventTracker
    from src.infrastructure.vectors import get_qdrant

    # Get global Qdrant client
    qdrant_client = get_qdrant()

    # Create per-request services
    embedding_service = EmbeddingService()
    llm_client = LLMClient()

    # Create managers
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    persona_manager = PersonaManager()

    # Create agent factory
    capability_registry = get_default_registry()
    agent_factory = AgentFactory(
        llm_client=llm_client,
        capability_registry=capability_registry,
        persona_manager=persona_manager,
    )

    # Create event tracker for analytics events
    event_tracker = EventTracker(tenant_code=tenant.code)

    # Create workflow
    workflow = PracticeHelperWorkflow(
        agent_factory=agent_factory,
        memory_manager=memory_manager,
        persona_manager=persona_manager,
        checkpointer=checkpointer,
        event_tracker=event_tracker,
    )

    return PracticeHelperService(workflow=workflow)


@router.post(
    "/start",
    response_model=StartPracticeHelperResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start tutoring session",
    description="Start a practice helper tutoring session when student clicks 'Get Help' after answering incorrectly.",
)
async def start_session(
    request: StartPracticeHelperRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> StartPracticeHelperResponse:
    """Start a practice helper tutoring session.

    This endpoint is called when a student answers incorrectly during practice
    and clicks "Get Help" to understand the concept better.

    The system will:
    1. Load the question context from the practice session
    2. Determine the best tutoring mode based on emotional state and mastery
    3. Select a subject-specific tutor agent
    4. Generate the first tutoring message

    Args:
        request: Request with practice session and question IDs.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        StartPracticeHelperResponse with session ID and first message.

    Raises:
        HTTPException: 400 if practice session or question not found.
        HTTPException: 403 if user type is not student.
    """
    # Verify user is a student
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can use practice helper",
        )

    service = _get_practice_helper_service(tenant_db_manager, checkpointer, tenant)

    try:
        return await service.start_session(
            db=db,
            student_id=current_user.id,
            tenant_id=tenant.id,
            tenant_code=tenant.code,
            request=request,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/{session_id}/message",
    response_model=MessageResponse,
    summary="Send message",
    description="Send a message or action in the tutoring session.",
)
async def send_message(
    session_id: UUID,
    request: SendMessageRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> MessageResponse:
    """Send a message in the tutoring session.

    Students can send messages or actions:
    - "respond": Regular message response
    - "next_step": Request next step (in STEP_BY_STEP mode)
    - "show_me": Request step-by-step explanation
    - "i_understand": Indicate understanding (ends session)
    - "end": End the session

    Args:
        session_id: Practice helper session ID.
        request: Request with message and action.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        MessageResponse with tutor's response.

    Raises:
        HTTPException: 400 if session not found or not active.
        HTTPException: 403 if session doesn't belong to user.
    """
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can use practice helper",
        )

    service = _get_practice_helper_service(tenant_db_manager, checkpointer, tenant)

    try:
        return await service.send_message(
            db=db,
            session_id=session_id,
            student_id=current_user.id,
            request=request,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/{session_id}",
    response_model=PracticeHelperSessionResponse,
    summary="Get session",
    description="Get practice helper session details.",
)
async def get_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> PracticeHelperSessionResponse:
    """Get practice helper session details.

    Args:
        session_id: Practice helper session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        PracticeHelperSessionResponse with session details.

    Raises:
        HTTPException: 404 if session not found.
        HTTPException: 403 if session doesn't belong to user.
    """
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access practice helper sessions",
        )

    service = _get_practice_helper_service(tenant_db_manager, checkpointer, tenant)

    try:
        return await service.get_session(
            db=db,
            session_id=session_id,
            student_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get history",
    description="Get conversation history for the session.",
)
async def get_session_history(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> SessionHistoryResponse:
    """Get conversation history for the session.

    Returns all messages exchanged in the tutoring session,
    ordered by sequence.

    Args:
        session_id: Practice helper session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        SessionHistoryResponse with message list.

    Raises:
        HTTPException: 404 if session not found.
        HTTPException: 403 if session doesn't belong to user.
    """
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can access practice helper sessions",
        )

    service = _get_practice_helper_service(tenant_db_manager, checkpointer, tenant)

    try:
        return await service.get_session_history(
            db=db,
            session_id=session_id,
            student_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/{session_id}/complete",
    response_model=SessionCompletionResponse,
    summary="Complete session",
    description="Complete the tutoring session and return to practice.",
)
async def complete_session(
    session_id: UUID,
    request: CompleteSessionRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> SessionCompletionResponse:
    """Complete the tutoring session.

    Called when the student is done with the tutoring session
    and wants to return to practice. The student indicates whether
    they understood the concept and if they want to retry the question.

    Args:
        session_id: Practice helper session ID.
        request: Request with completion details.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        SessionCompletionResponse with completion details and return info.

    Raises:
        HTTPException: 404 if session not found.
        HTTPException: 403 if session doesn't belong to user.
    """
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can use practice helper",
        )

    service = _get_practice_helper_service(tenant_db_manager, checkpointer, tenant)

    try:
        return await service.complete_session(
            db=db,
            session_id=session_id,
            student_id=current_user.id,
            request=request,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
