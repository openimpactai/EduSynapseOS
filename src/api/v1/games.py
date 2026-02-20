# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Coach API endpoints.

This module provides endpoints for educational games:
- POST /start - Start a new game session
- GET /available - List available games (MUST be before /{session_id})
- GET /sessions - List game sessions (MUST be before /{session_id})
- POST /{session_id}/move - Submit a move
- POST /{session_id}/hint - Request a hint
- GET /{session_id} - Get game status
- POST /{session_id}/resign - Resign the game
- POST /{session_id}/analyze - Get full game analysis

IMPORTANT: Static routes (/available, /sessions) MUST be defined before
parameterized routes (/{session_id}) to prevent route matching issues.

All coach messages, hints, and analyses are generated dynamically via LLM
to provide personalized, age-appropriate feedback for students.

Example:
    POST /api/v1/games/start
    {
        "game_type": "chess",
        "game_mode": "practice",
        "difficulty": "medium",
        "player_color": "white"
    }
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select
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
from src.core.config import get_settings
from src.core.intelligence.llm import LLMClient
from src.domains.gaming.models import GameType
from src.domains.gaming.schemas import (
    AnalyzeGameResponse,
    AvailableGamesResponse,
    GameStatusResponse,
    GetHintRequest,
    GetHintResponse,
    ListGamesResponse,
    MakeMoveRequest,
    MakeMoveResponse,
    ResignResponse,
    StartGameRequest,
    StartGameResponse,
)
from src.domains.gaming.service import (
    GameCoachService,
    GameNotFoundError,
    GameNotActiveError,
    InvalidMoveError,
)
from src.infrastructure.database.models.tenant.user import User
from src.infrastructure.database.models.tenant.school import ClassStudent, Class
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

DEFAULT_GRADE_LEVEL = 5
DEFAULT_LANGUAGE = "en"


async def _get_student_grade_level(db: AsyncSession, user_id: UUID) -> int:
    """Get student's grade level from their class enrollment.

    Args:
        db: Database session.
        user_id: Student's user ID.

    Returns:
        Grade level sequence number (e.g., 4 for Year 4).
    """
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
            return class_student.class_.grade_level.sequence

    except Exception as e:
        logger.warning("Failed to get student grade level: %s", e)

    return DEFAULT_GRADE_LEVEL

router = APIRouter()


def _get_game_service(
    tenant: TenantContext,
    tenant_db_manager: TenantDatabaseManager,
) -> GameCoachService:
    """Get game coach service instance with LLM client.

    Creates LLMClient for personalized coach messages and MemoryManager
    for recording game-related memory.

    Args:
        tenant: Tenant context for event tracking.
        tenant_db_manager: Tenant database manager for memory operations.

    Returns:
        GameCoachService instance.
    """
    from src.core.intelligence.embeddings.service import EmbeddingService
    from src.core.memory.manager import MemoryManager
    from src.domains.analytics.events import EventTracker
    from src.infrastructure.vectors import get_qdrant

    settings = get_settings()
    llm_client = LLMClient(llm_settings=settings.llm)

    # Create event tracker for analytics events
    event_tracker = EventTracker(tenant_code=tenant.code)

    # Create memory manager for game memory recording
    qdrant_client = get_qdrant()
    embedding_service = EmbeddingService()
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    return GameCoachService(
        llm_client=llm_client,
        event_tracker=event_tracker,
        tenant_code=tenant.code,
        memory_manager=memory_manager,
    )


# =============================================================================
# STATIC ROUTES (must be defined before parameterized routes)
# =============================================================================


@router.post(
    "/start",
    response_model=StartGameResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new game",
    description="Start a new game session with the specified game type, mode, and difficulty.",
)
async def start_game(
    data: StartGameRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> StartGameResponse:
    """Start a new game session.

    Args:
        data: Game configuration.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        StartGameResponse with session ID, board state, and coach greeting.
    """
    logger.info(
        "Starting game: user=%s, type=%s, mode=%s, difficulty=%s",
        current_user.id,
        data.game_type.value,
        data.game_mode.value,
        data.difficulty.value,
    )

    service = _get_game_service(tenant, tenant_db_manager)

    # Fetch user to get first_name for personalized coaching
    result = await db.execute(select(User).where(User.id == current_user.id))
    user = result.scalar_one_or_none()
    student_name = user.first_name if user else "there"

    # Get student's grade level for age-appropriate coaching
    grade_level = await _get_student_grade_level(db, UUID(current_user.id))

    return await service.start_game(
        db=db,
        student_id=UUID(current_user.id),
        request=data,
        student_name=student_name,
        language=current_user.preferred_language,
        grade_level=grade_level,
    )


@router.get(
    "/available",
    response_model=AvailableGamesResponse,
    summary="Get available games",
    description="Get list of available games and active session if any.",
)
async def get_available_games(
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> AvailableGamesResponse:
    """Get available games.

    Returns:
        AvailableGamesResponse with game list and active session.
    """
    service = _get_game_service(tenant, tenant_db_manager)

    return await service.get_available_games(
        db=db,
        student_id=UUID(current_user.id),
    )


@router.get(
    "/sessions",
    response_model=ListGamesResponse,
    summary="List game sessions",
    description="List game sessions for the current user, optionally filtered by game type.",
)
async def list_sessions(
    game_type: GameType | None = None,
    limit: int = 20,
    offset: int = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> ListGamesResponse:
    """List game sessions.

    Args:
        game_type: Optional filter by game type.
        limit: Maximum number of sessions to return.
        offset: Offset for pagination.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        ListGamesResponse with session list.
    """
    service = _get_game_service(tenant, tenant_db_manager)

    return await service.list_sessions(
        db=db,
        student_id=UUID(current_user.id),
        game_type=game_type,
        limit=limit,
        offset=offset,
    )


# =============================================================================
# PARAMETERIZED ROUTES (must be defined after static routes)
# =============================================================================


@router.get(
    "/{session_id}",
    response_model=GameStatusResponse,
    summary="Get game status",
    description="Get the current status and board state of a game session.",
)
async def get_game_status(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> GameStatusResponse:
    """Get game session status.

    Args:
        session_id: The game session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        GameStatusResponse with current game state.

    Raises:
        HTTPException: If session not found.
    """
    service = _get_game_service(tenant, tenant_db_manager)

    try:
        return await service.get_status(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except GameNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found",
        )


@router.post(
    "/{session_id}/move",
    response_model=MakeMoveResponse,
    summary="Submit a move",
    description="Submit a move in the game. Returns the move result, AI response, and coach feedback.",
)
async def submit_move(
    session_id: UUID,
    data: MakeMoveRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> MakeMoveResponse:
    """Submit a move in the game.

    Args:
        session_id: The game session ID.
        data: Move data.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        MakeMoveResponse with move result and coach message.

    Raises:
        HTTPException: If session not found, not active, or invalid move.
    """
    logger.info(
        "Processing move: session=%s, move=%s",
        session_id,
        data.move,
    )

    service = _get_game_service(tenant, tenant_db_manager)

    try:
        return await service.process_move(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
            request=data,
        )
    except GameNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found",
        )
    except GameNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Game session is not active",
        )
    except InvalidMoveError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/{session_id}/hint",
    response_model=GetHintResponse,
    summary="Request a hint",
    description="Request a hint for the current position. Hints become more specific at higher levels.",
)
async def get_hint(
    session_id: UUID,
    data: GetHintRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> GetHintResponse:
    """Get a hint for the current position.

    Hint levels:
    - Level 1: General strategic hint
    - Level 2: More specific guidance
    - Level 3: Reveals the best move

    Args:
        session_id: The game session ID.
        data: Hint request with level.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        GetHintResponse with hint text and remaining hints.

    Raises:
        HTTPException: If session not found or not active.
    """
    logger.info(
        "Getting hint: session=%s, level=%d",
        session_id,
        data.level,
    )

    service = _get_game_service(tenant, tenant_db_manager)

    try:
        return await service.get_hint(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
            request=data,
        )
    except GameNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found",
        )
    except GameNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Game session is not active",
        )


@router.post(
    "/{session_id}/resign",
    response_model=ResignResponse,
    summary="Resign the game",
    description="Resign the current game. Returns a summary and learning points.",
)
async def resign_game(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> ResignResponse:
    """Resign the game.

    Args:
        session_id: The game session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        ResignResponse with game summary.

    Raises:
        HTTPException: If session not found or not active.
    """
    logger.info("Resigning game: session=%s", session_id)

    service = _get_game_service(tenant, tenant_db_manager)

    try:
        return await service.resign(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except GameNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found",
        )
    except GameNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Game session is not active",
        )


@router.post(
    "/{session_id}/analyze",
    response_model=AnalyzeGameResponse,
    summary="Analyze the game",
    description="Get detailed analysis of a completed game with learning points and tips.",
)
async def analyze_game(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> AnalyzeGameResponse:
    """Get detailed game analysis.

    Provides:
    - Overall statistics and accuracy
    - Critical moments in the game
    - Learning points
    - Improvement tips

    Args:
        session_id: The game session ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        AnalyzeGameResponse with detailed analysis.

    Raises:
        HTTPException: If session not found.
    """
    logger.info("Analyzing game: session=%s", session_id)

    service = _get_game_service(tenant, tenant_db_manager)

    try:
        return await service.analyze_game(
            db=db,
            session_id=session_id,
            student_id=UUID(current_user.id),
        )
    except GameNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found",
        )
