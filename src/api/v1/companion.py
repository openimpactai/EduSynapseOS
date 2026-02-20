# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion API endpoints.

This module provides endpoints for companion agent interactions:
- POST /chat - Unified chat endpoint for all interactions
- GET /sessions/{session_id} - Get session info
- GET /sessions/{session_id}/messages - Get message history
- POST /sessions/{session_id}/end - End a session

The companion uses a unified chat endpoint with tool calling.
All interactions (greetings, emotional support, activity guidance)
flow through the same endpoint with the LLM deciding what actions to take.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import (
    Checkpointer,
    TenantDB,
    TenantDBManager,
    get_checkpointer,
    get_tenant_db,
    get_tenant_db_manager,
    require_auth,
    require_tenant,
)
from src.domains.analytics.events import EventTracker
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.config import get_settings
from src.core.emotional import EmotionalStateService
from src.core.intelligence.embeddings import EmbeddingService
from src.core.intelligence.llm import LLMClient
from src.core.memory.manager import MemoryManager
from src.core.personas.manager import get_persona_manager
from src.core.proactive import get_proactive_service
from src.domains.companion.schemas import (
    CompanionChatRequest,
    CompanionChatResponse,
    CompanionMessagesResponse,
    CompanionSessionResponse,
)
from src.domains.companion.service import (
    CompanionService,
    SessionNotActiveError,
    SessionNotFoundError,
)
from src.infrastructure.database.models.tenant.school import ClassStudent
from src.infrastructure.database.models.tenant.user import User
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.vectors import get_qdrant
from src.domains.curriculum.lookup import CurriculumLookup, DEFAULT_FRAMEWORK
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default values if grade level cannot be determined
DEFAULT_GRADE_LEVEL = 5
DEFAULT_LANGUAGE = "en"


@dataclass
class StudentCurriculumContext:
    """Student's curriculum context resolved from class/school/user data.

    Attributes:
        framework_code: Curriculum framework code (e.g., "UK-NC-2014").
        grade_code: Grade code within framework (e.g., "Y5", "G5").
        grade_level: Grade sequence number for approximate comparisons.
        country_code: ISO 2-letter country code (e.g., "GB", "US").
    """
    framework_code: str
    grade_code: str | None
    grade_level: int
    country_code: str | None


async def _get_student_curriculum_context(
    db: AsyncSession, user_id: UUID
) -> StudentCurriculumContext:
    """Get student's full curriculum context from class enrollment or user profile.

    Resolution priority:
    1. Class enrollment: class.framework_code + class.grade_code + grade_level.sequence
    2. User extra_data: country_code + level (grade_code)
    3. School country_code: Fallback for framework resolution

    This provides:
    - framework_code: For filtering subjects/topics to student's curriculum
    - grade_code: For exact grade matching (e.g., "Y5", "G5", "P5", "STD5")
    - grade_level: Sequence number for approximate grade comparisons

    Args:
        db: Database session.
        user_id: Student's user ID.

    Returns:
        StudentCurriculumContext with all resolved values.
    """
    from src.infrastructure.database.models.tenant.school import Class, School

    framework_code: str | None = None
    grade_code: str | None = None
    grade_level: int = DEFAULT_GRADE_LEVEL
    country_code: str | None = None

    try:
        # Step 1: Try to get from class enrollment (most reliable)
        result = await db.execute(
            select(ClassStudent)
            .options(
                selectinload(ClassStudent.class_)
                .selectinload(Class.grade_level),
                selectinload(ClassStudent.class_)
                .selectinload(Class.school),
            )
            .where(ClassStudent.student_id == str(user_id))
            .limit(1)
        )
        class_student = result.scalars().first()

        if class_student and class_student.class_:
            cls = class_student.class_

            # Get framework_code from class (if explicitly set)
            if cls.framework_code:
                framework_code = cls.framework_code
                logger.debug(
                    "Framework from class: user_id=%s, framework=%s",
                    user_id, framework_code
                )

            # Get grade_code from class
            if cls.grade_code:
                grade_code = cls.grade_code
                logger.debug(
                    "Grade code from class: user_id=%s, grade_code=%s",
                    user_id, grade_code
                )

            # Get grade_level sequence
            if cls.grade_level:
                grade_level = cls.grade_level.sequence
                logger.debug(
                    "Grade level from class: user_id=%s, level=%s",
                    user_id, grade_level
                )

            # Get country_code from school (for framework resolution fallback)
            if cls.school and cls.school.country_code:
                country_code = cls.school.country_code
                logger.debug(
                    "Country code from school: user_id=%s, country=%s",
                    user_id, country_code
                )

        # Step 2: Get user's extra_data for fallback/override
        user_result = await db.execute(
            select(User.extra_data).where(User.id == str(user_id))
        )
        extra_data = user_result.scalar_one_or_none()

        if extra_data:
            # Override country_code from user if present
            if extra_data.get("country_code"):
                country_code = extra_data["country_code"]
                logger.debug(
                    "Country code from user: user_id=%s, country=%s",
                    user_id, country_code
                )

            # Override grade_code from user's level if present
            if extra_data.get("level") and not grade_code:
                grade_code = extra_data["level"]
                logger.debug(
                    "Grade code from user level: user_id=%s, grade_code=%s",
                    user_id, grade_code
                )

        # Step 3: Resolve framework_code from country_code if not set
        lookup = CurriculumLookup(db)
        if not framework_code and country_code:
            framework_code = await lookup.resolve_framework_code(country_code)
            logger.debug(
                "Framework resolved from country: user_id=%s, country=%s, framework=%s",
                user_id, country_code, framework_code
            )

        # Step 4: Apply defaults
        if not framework_code:
            framework_code = DEFAULT_FRAMEWORK
            logger.debug(
                "Using default framework: user_id=%s, framework=%s",
                user_id, framework_code
            )

        # Step 5: Resolve grade_level from grade_code if we have grade_code but no class enrollment
        # This handles playground users who only have extra_data.level without class enrollment
        if grade_code and grade_level == DEFAULT_GRADE_LEVEL:
            resolved_grade = await lookup.get_grade_level(framework_code, grade_code)
            if resolved_grade:
                grade_level = resolved_grade.sequence
                logger.debug(
                    "Grade level resolved from grade_code: user_id=%s, grade_code=%s, sequence=%s",
                    user_id, grade_code, grade_level
                )

    except Exception as e:
        logger.warning("Failed to get student curriculum context: %s", e)
        framework_code = DEFAULT_FRAMEWORK

    return StudentCurriculumContext(
        framework_code=framework_code,
        grade_code=grade_code,
        grade_level=grade_level,
        country_code=country_code,
    )


async def _get_student_grade_level(db: AsyncSession, user_id: UUID) -> int:
    """Get student's grade level from their class enrollment.

    DEPRECATED: Use _get_student_curriculum_context for full context.

    Queries class_students -> classes -> grade_levels to find the
    student's current grade level sequence number.

    Args:
        db: Database session.
        user_id: Student's user ID.

    Returns:
        Grade level sequence number (e.g., 4 for Year 4).
        Returns DEFAULT_GRADE_LEVEL if not found.
    """
    context = await _get_student_curriculum_context(db, user_id)
    return context.grade_level


async def _get_student_name(db: AsyncSession, user_id: UUID) -> str:
    """Get student's first name from users table.

    Args:
        db: Database session.
        user_id: Student's user ID.

    Returns:
        Student's first name, or "there" as fallback.
    """
    from src.infrastructure.database.models.tenant.user import User

    try:
        result = await db.execute(
            select(User.first_name).where(User.id == str(user_id))
        )
        first_name = result.scalar_one_or_none()

        if first_name:
            logger.debug("Student name found: user_id=%s, name=%s", user_id, first_name)
            return first_name

    except Exception as e:
        logger.warning("Failed to get student name: %s", e)

    return "there"  # Fallback for "Hello there!"


router = APIRouter()


def _get_companion_service(
    db: AsyncSession,
    tenant: TenantContext,
    checkpointer: BaseCheckpointSaver | None,
    tenant_db_manager: TenantDatabaseManager,
) -> CompanionService:
    """Create a CompanionService instance with all dependencies.

    Args:
        db: Tenant database session.
        tenant: Tenant context.
        checkpointer: LangGraph checkpointer for state persistence.
        tenant_db_manager: Tenant database manager for MemoryManager.

    Returns:
        Configured CompanionService instance.
    """
    settings = get_settings()

    # Create LLM client
    llm_client = LLMClient(llm_settings=settings.llm)

    # Get Qdrant client and create embedding service
    qdrant_client = get_qdrant()
    embedding_service = EmbeddingService()

    # Create memory manager
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    # Get persona manager
    persona_manager = get_persona_manager()

    # Create emotional service
    emotional_service = EmotionalStateService(db=db)

    # Get proactive service (optional)
    proactive_service = None
    try:
        proactive_service = get_proactive_service(
            memory_manager=memory_manager,
            tenant_db_manager=tenant_db_manager,
        )
    except Exception as e:
        logger.warning("ProactiveService not available: %s", e)

    # Create event tracker for analytics events
    event_tracker = EventTracker(tenant_code=tenant.code)

    return CompanionService(
        db=db,
        llm_client=llm_client,
        memory_manager=memory_manager,
        persona_manager=persona_manager,
        checkpointer=checkpointer,
        emotional_service=emotional_service,
        proactive_service=proactive_service,
        event_tracker=event_tracker,
    )


@router.post("/chat", response_model=CompanionChatResponse)
async def chat(
    request: CompanionChatRequest,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
):
    """Unified chat endpoint for companion interactions.

    This endpoint handles all companion interactions through a single interface:
    - New sessions: Omit session_id to start a new conversation with greeting
    - Continue conversation: Provide session_id and message to continue

    The companion uses tool calling to:
    - Suggest activities (get_activities)
    - Navigate the student (navigate)
    - Record emotional signals (record_emotion)
    - Hand off to tutor (handoff_to_tutor)
    - Get personalization context (get_student_context, get_parent_notes)
    - Check review schedule (get_review_schedule)

    Args:
        request: Chat request with optional session_id and message.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.
        tenant_db_manager: Tenant database manager.

    Returns:
        CompanionChatResponse with message, actions, and emotional state.

    Raises:
        HTTPException: On session not found, not active, or server error.
    """
    service = _get_companion_service(db, tenant, checkpointer, tenant_db_manager)

    # Get student's curriculum context and name from their profile
    curriculum_context = await _get_student_curriculum_context(db, user.id)
    student_name = await _get_student_name(db, user.id)

    try:
        response = await service.chat(
            user_id=user.id,
            tenant_id=tenant.id,
            tenant_code=tenant.code,
            request=request,
            grade_level=curriculum_context.grade_level,
            framework_code=curriculum_context.framework_code,
            grade_code=curriculum_context.grade_code,
            language=user.preferred_language,
            student_name=student_name,
        )
        return response

    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except SessionNotActiveError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat operation failed",
        )


@router.get("/sessions/{session_id}", response_model=CompanionSessionResponse)
async def get_session(
    session_id: UUID,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
):
    """Get session information.

    Args:
        session_id: Session identifier.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.
        tenant_db_manager: Tenant database manager.

    Returns:
        Session information including status and message count.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_companion_service(db, tenant, checkpointer, tenant_db_manager)

    try:
        return await service.get_session(
            session_id=session_id,
            user_id=user.id,
        )
    except SessionNotFoundError as e:
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


@router.get("/sessions/{session_id}/messages", response_model=CompanionMessagesResponse)
async def get_messages(
    session_id: UUID,
    limit: int = 50,
    before_id: UUID | None = None,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
):
    """Get message history for a session.

    Args:
        session_id: Session identifier.
        limit: Maximum messages to return (default 50, max 100).
        before_id: Get messages before this message ID (for pagination).
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.
        tenant_db_manager: Tenant database manager.

    Returns:
        Message history with pagination info.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_companion_service(db, tenant, checkpointer, tenant_db_manager)

    # Clamp limit
    limit = min(max(1, limit), 100)

    try:
        return await service.get_messages(
            session_id=session_id,
            user_id=user.id,
            limit=limit,
            before_id=before_id,
        )
    except SessionNotFoundError as e:
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
    session_id: UUID,
    final_mood: str | None = None,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
):
    """End a companion session.

    Args:
        session_id: Session identifier.
        final_mood: Optional final emotional state.
        user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        checkpointer: Workflow state checkpointer.
        tenant_db_manager: Tenant database manager.

    Returns:
        Success message.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_companion_service(db, tenant, checkpointer, tenant_db_manager)

    try:
        await service.end_session(
            session_id=session_id,
            user_id=user.id,
            final_mood=final_mood,
        )
        return {"success": True, "message": "Session ended"}

    except SessionNotFoundError as e:
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
