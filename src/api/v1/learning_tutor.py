# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning Tutor API endpoints.

This module provides endpoints for learning tutor sessions:
- POST /start - Start a learning session for a topic
- POST /{session_id}/message - Send a message in the session
- GET /{session_id} - Get session details
- GET /{session_id}/history - Get conversation history
- POST /{session_id}/complete - Complete the session

The learning tutor proactively teaches students new concepts.
It can be triggered from:
- Companion handoff ("I want to learn about...")
- Practice help ("I need to learn this")
- Direct access (learning menu)
- LMS deep link
- Spaced repetition review
- Weakness suggestions

Example:
    POST /api/v1/learning-tutor/start
    {
        "topic_framework_code": "UK-NC-2014",
        "topic_subject_code": "MAT",
        "topic_grade_code": "Y4",
        "topic_unit_code": "NPV",
        "topic_code": "001",
        "topic_name": "Place Value",
        "entry_point": "direct"
    }
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
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
from src.domains.learning.service import LearningService
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.models.learning import (
    CompleteSessionRequest,
    LearningSessionResponse,
    MessageResponse,
    SendMessageRequest,
    SessionCompletionResponse,
    SessionHistoryResponse,
    StartLearningRequest,
    StartLearningResponse,
)

logger = logging.getLogger(__name__)

# Default values if grade level cannot be determined
DEFAULT_GRADE_LEVEL = 5
DEFAULT_LANGUAGE = "en"


async def _get_student_grade_level(db: AsyncSession, user_id: UUID) -> int:
    """Get student's grade level from their class enrollment.

    Queries class_students -> classes -> grade_levels to find the
    student's current grade level sequence number.

    Args:
        db: Database session.
        user_id: Student's user ID.

    Returns:
        Grade level sequence number (e.g., 4 for Year 4).
        Returns DEFAULT_GRADE_LEVEL if not found.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from src.infrastructure.database.models.tenant.school import Class, ClassStudent

    try:
        result = await db.execute(
            select(ClassStudent)
            .options(
                selectinload(ClassStudent.class_).selectinload(Class.grade_level)
            )
            .where(ClassStudent.student_id == str(user_id))
            .limit(1)
        )
        class_student = result.scalars().first()

        if class_student and class_student.class_ and class_student.class_.grade_level:
            grade = class_student.class_.grade_level.sequence
            logger.debug(
                "Student grade level found: user_id=%s, grade=%s, class=%s",
                user_id, grade, class_student.class_.name
            )
            return grade
        else:
            logger.debug(
                "Student grade level not found: user_id=%s",
                user_id,
            )

    except Exception as e:
        logger.warning("Failed to get student grade level: %s", e)

    return DEFAULT_GRADE_LEVEL


router = APIRouter()


def _get_learning_service(
    db: AsyncSession,
    tenant_db_manager: TenantDatabaseManager,
    checkpointer: BaseCheckpointSaver | None,
    tenant: TenantContext,
) -> LearningService:
    """Get learning service instance.

    Args:
        db: Tenant database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.
        tenant: Tenant context for event tracking.

    Returns:
        Configured LearningService instance.
    """
    from src.core.agents import AgentFactory
    from src.core.agents.capabilities.registry import get_default_registry
    from src.core.educational.orchestrator import TheoryOrchestrator
    from src.core.intelligence.embeddings.service import EmbeddingService
    from src.core.intelligence.llm.client import LLMClient
    from src.domains.analytics.events import EventTracker
    from src.core.memory.manager import MemoryManager
    from src.core.orchestration.workflows.learning_tutor import LearningTutorWorkflow
    from src.core.personas.manager import PersonaManager
    from src.core.emotional import EmotionalStateService
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
    theory_orchestrator = TheoryOrchestrator()
    emotional_service = EmotionalStateService(db=db)

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
    workflow = LearningTutorWorkflow(
        agent_factory=agent_factory,
        memory_manager=memory_manager,
        persona_manager=persona_manager,
        db_session=None,
        checkpointer=checkpointer,
        emotional_service=emotional_service,
        theory_orchestrator=theory_orchestrator,
        event_tracker=event_tracker,
    )

    return LearningService(workflow=workflow)


@router.post(
    "/start",
    response_model=StartLearningResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start learning session",
    description="Start a learning tutor session to learn a new topic.",
)
async def start_session(
    request: StartLearningRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> StartLearningResponse:
    """Start a learning tutor session.

    This endpoint starts a proactive teaching session for a specific topic.
    The tutor will adapt its teaching approach based on:
    - Student's emotional state
    - Topic mastery level
    - Subject matter (math, science, history, etc.)
    - Entry point (companion handoff, direct access, etc.)

    The system will:
    1. Determine the best learning mode based on student context
    2. Select a subject-specific tutor agent
    3. Generate an engaging opening message
    4. Track understanding progress throughout

    Note: If the student already has an active session, it will be
    auto-completed with reason "new_session_started" before starting
    the new session. This ensures proper memory recording.

    Args:
        request: Request with topic details and entry point.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        StartLearningResponse with session ID, mode, and first message.

    Raises:
        HTTPException: 400 if topic not found or invalid parameters.
        HTTPException: 403 if user type is not student.
    """
    # Verify user is a student
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can use learning tutor",
        )

    service = _get_learning_service(db, tenant_db_manager, checkpointer, tenant)

    # Get student's grade level from their class enrollment
    grade_level = await _get_student_grade_level(db, UUID(current_user.id))

    try:
        return await service.start_session(
            db=db,
            student_id=UUID(current_user.id),
            tenant_id=UUID(tenant.id),
            tenant_code=tenant.code,
            request=request,
            grade_level=grade_level,
            language=current_user.preferred_language,
        )
    except ValueError as e:
        error_msg = str(e)
        if "active learning session" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.post(
    "/{session_id}/message",
    response_model=MessageResponse,
    summary="Send message",
    description="Send a message or action in the learning session.",
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
    """Send a message in the learning session.

    Students can send messages or actions based on the current learning mode:

    All modes:
    - "respond": Regular message response
    - "i_understand": Indicate understanding
    - "end": End the session

    DISCOVERY mode:
    - "give_hint": Request a hint
    - "show_me": Request direct explanation

    EXPLANATION mode:
    - "more_examples": Request more examples
    - "let_me_try": Request practice
    - "simpler": Request simpler explanation

    WORKED_EXAMPLE mode:
    - "another_example": Request another example
    - "let_me_try": Request practice

    GUIDED_PRACTICE mode:
    - "hint": Request hint for current question
    - "show_solution": Show the solution

    ASSESSMENT mode:
    - "hint": Request hint for assessment question

    Args:
        session_id: The learning session ID.
        request: Message request with content and action.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        MessageResponse with tutor's response and session status.

    Raises:
        HTTPException: 400 if session not active.
        HTTPException: 404 if session not found.
    """
    service = _get_learning_service(db, tenant_db_manager, checkpointer, tenant)

    try:
        return await service.send_message(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
            request=request,
        )
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        if "not active" in error_msg or "does not belong" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.get(
    "/{session_id}",
    response_model=LearningSessionResponse,
    summary="Get session details",
    description="Get learning session details by ID.",
)
async def get_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> LearningSessionResponse:
    """Get learning session details.

    Returns the current state of a learning session including:
    - Topic information
    - Current learning mode
    - Progress metrics
    - Turn count
    - Practice statistics (if applicable)

    Args:
        session_id: The learning session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        LearningSessionResponse with session details.

    Raises:
        HTTPException: 404 if session not found.
    """
    service = _get_learning_service(db, tenant_db_manager, checkpointer, tenant)

    try:
        return await service.get_session(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get message history",
    description="Get conversation history for a learning session.",
)
async def get_history(
    session_id: UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> SessionHistoryResponse:
    """Get conversation history for a learning session.

    Returns the message history with pagination support.
    Messages are ordered by sequence (oldest first).

    Args:
        session_id: The learning session ID.
        limit: Maximum messages to return (default: 50, max: 100).
        offset: Offset for pagination (default: 0).
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        SessionHistoryResponse with message list and pagination info.

    Raises:
        HTTPException: 404 if session not found.
    """
    service = _get_learning_service(db, tenant_db_manager, checkpointer, tenant)

    try:
        return await service.get_history(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
            limit=limit,
            offset=offset,
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
    description="Manually complete a learning session.",
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
    """Complete a learning session.

    Marks the session as completed and records completion details.
    This can be called when the student indicates they're done learning,
    or when they want to end the session early.

    Completion reasons:
    - "user_ended": Student ended the session
    - "mastery_achieved": Student demonstrated understanding
    - "max_turns": Maximum turn limit reached
    - "timeout": Session timed out

    Args:
        session_id: The learning session ID.
        request: Request with completion reason and understanding flag.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        SessionCompletionResponse with final summary.

    Raises:
        HTTPException: 404 if session not found.
    """
    service = _get_learning_service(db, tenant_db_manager, checkpointer, tenant)

    try:
        return await service.complete_session(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
            request=request,
            tenant_code=tenant.code,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
