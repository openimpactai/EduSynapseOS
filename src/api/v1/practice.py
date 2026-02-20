# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice session API endpoints.

This module provides endpoints for practice sessions:
- POST /start - Start a new practice session
- POST /{session_id}/answer - Submit an answer
- GET /{session_id} - Get session status
- GET /{session_id}/question - Get current question
- POST /{session_id}/complete - Complete the session
- POST /{session_id}/pause - Pause the session
- POST /{session_id}/resume - Resume the session

Example:
    POST /api/v1/practice/start
    {
        "topic_framework_code": "UK-NC-2014",
        "topic_subject_code": "MAT",
        "topic_grade_code": "Y4",
        "topic_unit_code": "NPV",
        "topic_code": "001",
        "session_type": "quick",
        "persona_id": "tutor"
    }
"""

import logging
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
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.agents import AgentFactory
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.personas.manager import PersonaManager
from src.domains.practice.service import (
    PracticeService,
    SessionNotFoundError,
    SessionNotActiveError,
)
from src.models.practice import (
    StartPracticeRequest,
    PracticeSessionResponse,
    QuestionResponse,
    SubmitAnswerRequest,
    AnswerResultResponse,
    CompleteSessionRequest,
    SessionCompletionResponse,
    ResumeSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_practice_service(
    db: AsyncSession,
    tenant: TenantContext,
    tenant_db_manager: TenantDatabaseManager,
    checkpointer: BaseCheckpointSaver | None,
) -> PracticeService:
    """Get practice service instance.

    Args:
        db: Tenant database session (for PracticeService DB operations).
        tenant: Tenant context.
        tenant_db_manager: Tenant database manager (for MemoryManager).
        checkpointer: Workflow checkpointer.

    Returns:
        Configured PracticeService instance.
    """
    from src.core.agents.capabilities.registry import get_default_registry
    from src.core.intelligence.embeddings.service import EmbeddingService
    from src.core.intelligence.llm.client import LLMClient
    from src.infrastructure.vectors import get_qdrant

    # Get global Qdrant client (initialized at startup)
    qdrant_client = get_qdrant()

    # Create per-request services
    embedding_service = EmbeddingService()
    llm_client = LLMClient()

    # Create managers with correct parameters
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    rag_retriever = RAGRetriever(
        memory_manager=memory_manager,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
    )

    theory_orchestrator = TheoryOrchestrator()
    persona_manager = PersonaManager()

    # Create agent factory with correct parameters
    capability_registry = get_default_registry()
    agent_factory = AgentFactory(
        llm_client=llm_client,
        capability_registry=capability_registry,
        persona_manager=persona_manager,
    )

    return PracticeService(
        db=db,
        agent_factory=agent_factory,
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        theory_orchestrator=theory_orchestrator,
        persona_manager=persona_manager,
        checkpointer=checkpointer,
    )


class StartPracticeResponse(PracticeSessionResponse):
    """Response when starting a practice session."""

    first_question: QuestionResponse | None = None


@router.post(
    "/start",
    response_model=StartPracticeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start practice session",
    description="Start a new practice session. Returns the session details and first question.",
)
async def start_practice(
    data: StartPracticeRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> StartPracticeResponse:
    """Start a new practice session.

    Args:
        data: Practice session configuration.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        StartPracticeResponse with session details and first question.
    """
    logger.info(
        "Starting practice session: user=%s, topic=%s, type=%s",
        current_user.id,
        data.topic_code,
        data.session_type.value,
    )

    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    session, first_question = await service.start_session(
        student_id=UUID(current_user.id),
        tenant_id=UUID(tenant.id),
        tenant_code=tenant.code,
        request=data,
    )

    # Build response with first question included
    response_data = session.model_dump()
    response_data["first_question"] = first_question

    return StartPracticeResponse(**response_data)


@router.get(
    "/{session_id}",
    response_model=PracticeSessionResponse,
    summary="Get session status",
    description="Get the current status and progress of a practice session.",
)
async def get_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    db: AsyncSession = Depends(get_tenant_db),
    tenant: TenantContext = Depends(require_tenant),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> PracticeSessionResponse:
    """Get practice session status.

    Args:
        session_id: The session ID.
        current_user: Authenticated user.
        db: Database session.
        tenant: Tenant context.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        PracticeSessionResponse with session details.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.get_session(
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice session not found",
        )


@router.get(
    "/{session_id}/question",
    response_model=QuestionResponse | None,
    summary="Get current question",
    description="Get the current question for an active practice session.",
)
async def get_current_question(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    db: AsyncSession = Depends(get_tenant_db),
    tenant: TenantContext = Depends(require_tenant),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> QuestionResponse | None:
    """Get current question for a session.

    Args:
        session_id: The session ID.
        current_user: Authenticated user.
        db: Database session.
        tenant: Tenant context.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        QuestionResponse or None if session is complete.

    Raises:
        HTTPException: If session not found or not active.
    """
    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.get_current_question(
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice session not found",
        )
    except SessionNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Practice session is not active",
        )


@router.post(
    "/{session_id}/answer",
    response_model=AnswerResultResponse,
    summary="Submit answer",
    description="Submit an answer for the current question and get evaluation.",
)
async def submit_answer(
    session_id: UUID,
    data: SubmitAnswerRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> AnswerResultResponse:
    """Submit an answer for evaluation.

    Args:
        session_id: The session ID.
        data: Answer submission request.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        AnswerResultResponse with evaluation and next question.

    Raises:
        HTTPException: If session not found or not active.
    """
    logger.info(
        "Submitting answer: session=%s, question=%s",
        session_id,
        data.question_id,
    )

    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.submit_answer(
            session_id=session_id,
            student_id=UUID(current_user.id),
            tenant_id=UUID(tenant.id),
            tenant_code=tenant.code,
            request=data,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice session not found",
        )
    except SessionNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Practice session is not active",
        )


@router.post(
    "/{session_id}/complete",
    response_model=SessionCompletionResponse,
    summary="Complete session",
    description="Complete the practice session and get a summary.",
)
async def complete_session(
    session_id: UUID,
    data: CompleteSessionRequest | None = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> SessionCompletionResponse:
    """Complete a practice session.

    Args:
        session_id: The session ID.
        data: Optional completion request.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        SessionCompletionResponse with summary.

    Raises:
        HTTPException: If session not found.
    """
    logger.info("Completing practice session: session=%s", session_id)

    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.complete_session(
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice session not found",
        )


@router.post(
    "/{session_id}/pause",
    response_model=PracticeSessionResponse,
    summary="Pause session",
    description="Pause an active practice session to resume later.",
)
async def pause_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> PracticeSessionResponse:
    """Pause a practice session.

    Args:
        session_id: The session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        Updated session response.

    Raises:
        HTTPException: If session not found or not active.
    """
    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.pause_session(
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active practice session not found",
        )


@router.post(
    "/{session_id}/resume",
    response_model=ResumeSessionResponse,
    summary="Resume session",
    description="Resume a paused practice session.",
)
async def resume_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> ResumeSessionResponse:
    """Resume a paused practice session.

    Args:
        session_id: The session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        ResumeSessionResponse with session and current question.

    Raises:
        HTTPException: If session not found or not paused.
    """
    service = _get_practice_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        session, current_question = await service.resume_session(
            session_id=session_id,
            student_id=UUID(current_user.id),
        )

        return ResumeSessionResponse(
            session=session,
            current_question=current_question,
            checkpoint_data=None,
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paused practice session not found",
        )
