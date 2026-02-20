# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Coach Service.

This service manages game coaching sessions:
- Start a game session (chess, connect4, etc.)
- Process player moves
- Generate AI moves
- Provide coaching feedback (via LLM through GameCoachAgent)
- Get hints (via LLM through GameCoachAgent)
- Analyze games (via LLM through GameCoachAgent)

The service integrates with:
- Game engines (Stockfish, Minimax) for move validation and AI play
- LLM (via GameCoachAgent using DynamicAgent) for personalized coaching messages
- Database (for persistence)

All coach messages, hints, and analyses are generated dynamically
via the GameCoachAgent which uses the DynamicAgent infrastructure
with proper YAML configuration and context-aware prompts.

v2.0 Changes:
- Uses GameCoachAgent for all LLM interactions
- Sends full board state to LLM via GameCoachContext
- AI moves are explained in tutorial mode
- Uses YAML-driven prompts from game_coach.yaml
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agents.capabilities.registry import CapabilityRegistry
from src.core.intelligence.llm import LLMClient
from src.domains.gaming.coach_agent import GameCoachAgent

if TYPE_CHECKING:
    from src.core.memory.manager import MemoryManager
    from src.domains.analytics.events import EventTracker
from src.domains.gaming.context import StudentContext
from src.domains.gaming.context_builder import GameContextBuilder
from src.domains.gaming.engines import get_engine_registry
from src.domains.gaming.models import (
    GameDifficulty,
    GameMode,
    GameStatus,
    GameState,
    GameType,
    Move,
    MoveResult,
    Position,
)
from src.domains.gaming.schemas import (
    AIMoveInfo,
    AnalyzeGameResponse,
    AvailableGame,
    AvailableGamesResponse,
    BoardDisplay,
    CriticalMoment,
    GameSessionSummary,
    GetHintRequest,
    GetHintResponse,
    LearningPointInfo,
    ListGamesResponse,
    MakeMoveRequest,
    MakeMoveResponse,
    MoveQualityInfo,
    ResignResponse,
    SessionStats,
    StartGameRequest,
    StartGameResponse,
    GameStatusResponse,
)
from src.infrastructure.database.models.tenant import (
    GameSession,
    GameMove,
    GameAnalysis,
    User,
)

logger = logging.getLogger(__name__)


class GameServiceError(Exception):
    """Base exception for game service errors."""

    pass


class GameNotFoundError(GameServiceError):
    """Raised when a game session is not found."""

    pass


class GameNotActiveError(GameServiceError):
    """Raised when trying to operate on an inactive game."""

    pass


class InvalidMoveError(GameServiceError):
    """Raised when a move is invalid."""

    pass


class GameCoachService:
    """Service for managing game coaching sessions.

    Manages the lifecycle of game coaching sessions:
    1. Start session - creates game and returns initial board
    2. Process move - validates move, gets AI response, returns coach message
    3. Get hint - provides hints for current position
    4. Get status - retrieves current game state
    5. Resign - ends game with resignation
    6. Analyze - provides detailed game analysis

    The service uses LLM for generating all coach messages, making feedback
    personalized and age-appropriate. Game engines handle move validation
    and AI play, while the LLM provides educational coaching.

    The service is stateless - all state is stored in the database
    and game engines are instantiated per request.

    Example:
        >>> llm_client = LLMClient()
        >>> service = GameCoachService(llm_client=llm_client)
        >>> response = await service.start_game(db, student_id, request)
        >>> move_response = await service.process_move(db, session_id, move_request)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        capability_registry: CapabilityRegistry | None = None,
        event_tracker: "EventTracker | None" = None,
        tenant_code: str | None = None,
        memory_manager: "MemoryManager | None" = None,
    ) -> None:
        """Initialize the service.

        Args:
            llm_client: LLM client for generating coach messages.
            capability_registry: Capability registry (uses default if None).
            event_tracker: Tracker for publishing analytics events.
            tenant_code: Tenant code for fallback event tracking.
            memory_manager: Memory manager for recording game-related memory.
        """
        self._llm_client = llm_client
        self._capability_registry = capability_registry or CapabilityRegistry.default()
        self._engine_registry = get_engine_registry()
        self._event_tracker = event_tracker
        self._tenant_code = tenant_code
        self._memory_manager = memory_manager
        # Cache for game-specific coach agents (lazy initialization)
        self._coach_agents: dict[GameType, GameCoachAgent] = {}

    def _get_coach_agent(self, game_type: GameType) -> GameCoachAgent:
        """Get or create a coach agent for the specified game type.

        Uses lazy initialization with caching for efficiency.

        Args:
            game_type: The type of game (chess, connect4, etc.)

        Returns:
            GameCoachAgent configured for the specified game type.
        """
        if game_type not in self._coach_agents:
            self._coach_agents[game_type] = GameCoachAgent(
                llm_client=self._llm_client,
                game_type=game_type,
                capability_registry=self._capability_registry,
            )
        return self._coach_agents[game_type]

    async def start_game(
        self,
        db: AsyncSession,
        student_id: UUID,
        request: StartGameRequest,
        student_name: str = "there",
        language: str = "en",
        grade_level: int = 5,
    ) -> StartGameResponse:
        """Start a new game session.

        Creates a new game session with the specified configuration.
        Returns the initial board state and coach greeting.

        Args:
            db: Database session.
            student_id: Student's ID.
            request: Game configuration.
            student_name: Student's first name for personalization.
            language: Student's language preference.
            grade_level: Student's grade level for age-appropriate coaching.

        Returns:
            StartGameResponse with session info and initial board.

        Raises:
            ValueError: If student has an active session for this game type.
            ValueError: If game type is not supported.
        """
        logger.info(
            "Starting game: student=%s, type=%s, mode=%s, difficulty=%s",
            student_id,
            request.game_type.value,
            request.game_mode.value,
            request.difficulty.value,
        )

        # Auto-abandon any existing active session of same game type
        existing = await self._get_active_session(db, student_id, request.game_type)
        if existing:
            logger.info(
                "Auto-abandoning existing %s session %s for student %s",
                request.game_type.value,
                existing.id,
                student_id,
            )
            existing.status = "abandoned"
            existing.ended_at = datetime.now(timezone.utc)
            await db.flush()

        # Get engine
        engine = self._engine_registry.get(request.game_type)

        # Determine player color
        player_color = request.player_color
        if player_color == "random":
            player_color = random.choice(["white", "black"])
        if request.game_type == GameType.CONNECT4:
            player_color = "player1" if player_color in ("white", "player1") else "player2"

        # Get initial position
        if request.initial_position:
            initial_position = Position(notation=request.initial_position)
        else:
            initial_position = engine.get_initial_position()

        # Create game state
        game_state = GameState(
            game_type=request.game_type,
            position=initial_position,
            move_history=[],
            current_player="white" if request.game_type == GameType.CHESS else "player1",
            status=GameStatus.ACTIVE,
            metadata={"player_color": player_color},
        )

        # Calculate hints available
        hints_available = self._get_hints_for_mode(request.game_mode)

        # Create session record
        session = GameSession(
            student_id=str(student_id),
            game_type=request.game_type.value,
            game_mode=request.game_mode.value,
            difficulty=request.difficulty.value,
            player_color=player_color,
            status=GameStatus.ACTIVE.value,
            total_moves=0,
            hints_used=0,
            mistakes_count=0,
            excellent_moves_count=0,
            started_at=datetime.now(timezone.utc),
            initial_position=initial_position.model_dump(),
            game_state=game_state.to_storage_dict(),
            learning_points=[],
        )

        db.add(session)
        await db.flush()

        # Generate personalized greeting via LLM
        greeting = await self._generate_greeting(
            student_name=student_name,
            game_type=request.game_type,
            game_mode=request.game_mode,
            difficulty=request.difficulty,
            player_color=player_color,
            language=language,
            grade_level=grade_level,
        )

        # Determine if it's student's turn
        your_turn = self._is_player_turn(player_color, game_state.current_player)

        # If AI moves first, make AI move
        ai_first_move = None
        if not your_turn:
            ai_move = engine.get_ai_move(
                initial_position,
                request.difficulty,
                time_limit_ms=1000,
            )

            # Apply AI move
            new_state, validation = engine.apply_move(game_state, Move(notation=ai_move.move))

            # Save AI move
            await self._save_move(
                db=db,
                session=session,
                move_number=1,
                player="ai",
                notation=ai_move.move,
                position_before=initial_position,
                position_after=new_state.position,
            )

            # Update game state
            game_state = new_state
            session.game_state = game_state.to_storage_dict()
            session.total_moves = 1
            your_turn = True

        await db.commit()

        # Publish session started event
        await self._publish_session_started(session, student_id)

        # Build display data
        display = self._build_display(engine, game_state.position, request.game_type)

        return StartGameResponse(
            session_id=str(session.id),
            game_type=request.game_type,
            game_mode=request.game_mode,
            difficulty=request.difficulty,
            player_color=player_color,
            display=display,
            your_turn=your_turn,
            hints_available=hints_available,
            coach_greeting=greeting,
        )

    async def process_move(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: MakeMoveRequest,
    ) -> MakeMoveResponse:
        """Process a player's move.

        Validates the move, updates the game state, gets AI response,
        and generates coaching feedback.

        Args:
            db: Database session.
            session_id: Game session ID.
            student_id: Student's ID for verification.
            request: Move request with notation.

        Returns:
            MakeMoveResponse with updated board and coach message.

        Raises:
            ValueError: If session not found or not owned by student.
            ValueError: If game is not active.
        """
        logger.info(
            "Processing move: session=%s, move=%s",
            session_id,
            request.move,
        )

        # Get session
        session = await self._get_session(db, session_id)
        if not session:
            raise GameNotFoundError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise GameNotFoundError("Session does not belong to this student")

        if session.status != GameStatus.ACTIVE.value:
            raise GameNotActiveError(f"Game is not active: {session.status}")

        # Get engine and game state
        game_type = GameType(session.game_type)
        engine = self._engine_registry.get(game_type)
        game_state = GameState.from_storage_dict(session.game_state)

        # Validate move
        validation = engine.validate_move(
            game_state.position,
            Move(notation=request.move),
        )

        if not validation.is_valid:
            display = self._build_display(engine, game_state.position, game_type)
            # For invalid moves, return a simple error without LLM-generated message
            # This prevents hallucination about old moves
            error_msg = validation.error_message or f"Invalid move: {request.move}"
            return MakeMoveResponse(
                valid=False,
                error_message=error_msg,
                display=display,
                coach_message=error_msg,  # Use same error message as coach message
                game_over=False,
                your_turn=True,
                move_number=session.total_moves,
                hints_remaining=self._get_hints_remaining(session),
            )

        # Apply player's move
        position_before = game_state.position
        new_state, _ = engine.apply_move(game_state, Move(notation=request.move))
        session.total_moves += 1

        # Analyze position before the move (to get best_move and eval_before)
        analysis_before = engine.analyze_position(position_before, depth=10)

        # Analyze position after the move (to get eval_after for centipawn loss)
        analysis_after = engine.analyze_position(new_state.position, depth=10)

        # Determine if player is white (for centipawn loss calculation)
        player_is_white = session.player_color in ("white", "player1")

        # Assess move quality using centipawn loss
        move_quality = self._assess_move_quality(
            played_move=request.move,
            best_move=analysis_before.best_move,
            eval_before=analysis_before.evaluation,
            eval_after=analysis_after.evaluation,
            player_is_white=player_is_white,
        )

        # Use analysis_before for coach message context
        analysis = analysis_before

        # Update session stats
        self._update_session_stats(session, move_quality.quality)

        # Save player's move
        await self._save_move(
            db=db,
            session=session,
            move_number=session.total_moves,
            player="player",
            notation=request.move,
            position_before=position_before,
            position_after=new_state.position,
            quality=move_quality.quality,
            is_best_move=move_quality.is_best_move,
            best_move=analysis.best_move,
            time_spent_seconds=(request.time_spent_ms or 0) // 1000,
        )

        # Check if game ended after player's move
        is_over, result_type, winner = engine.is_game_over(new_state.position)

        if is_over:
            return await self._handle_game_end(
                db=db,
                session=session,
                engine=engine,
                game_state=new_state,
                game_type=game_type,
                result_type=result_type,
                winner=winner,
                player_color=session.player_color,
                move_quality=move_quality,
            )

        # Get AI's move
        ai_move = engine.get_ai_move(
            new_state.position,
            GameDifficulty(session.difficulty),
            time_limit_ms=1000,
        )

        # Apply AI's move
        ai_state, ai_validation = engine.apply_move(new_state, Move(notation=ai_move.move))
        session.total_moves += 1

        # Save AI's move
        await self._save_move(
            db=db,
            session=session,
            move_number=session.total_moves,
            player="ai",
            notation=ai_move.move,
            position_before=new_state.position,
            position_after=ai_state.position,
        )

        # Update game state
        session.game_state = ai_state.to_storage_dict()

        # Check if game ended after AI's move
        is_over, result_type, winner = engine.is_game_over(ai_state.position)

        if is_over:
            return await self._handle_game_end(
                db=db,
                session=session,
                engine=engine,
                game_state=ai_state,
                game_type=game_type,
                result_type=result_type,
                winner=winner,
                player_color=session.player_color,
                move_quality=move_quality,
                ai_move_notation=ai_move.move,
            )

        # Generate personalized coach message via LLM with position context
        # Get student info for personalization
        student = await db.get(User, session.student_id)
        student_name = student.first_name if student else "there"

        coach_message = await self._generate_coach_message(
            student_name=student_name,
            game_type=game_type,
            game_mode=GameMode(session.game_mode),
            position_before=position_before,
            position_after=new_state.position,
            move_notation=request.move,
            move_quality=move_quality,
            analysis=analysis,
        )

        # Generate AI move explanation for tutorial mode
        ai_explanation = None
        if GameMode(session.game_mode) == GameMode.TUTORIAL:
            ai_explanation = await self._generate_ai_move_explanation(
                student_name=student_name,
                game_type=game_type,
                game_mode=GameMode(session.game_mode),
                position_before=new_state.position,
                position_after=ai_state.position,
                ai_move=ai_move.move,
            )

        await db.commit()

        # Publish player's move event
        await self._publish_move_made(
            session=session,
            student_id=student_id,
            move_notation=request.move,
            move_quality=move_quality.quality,
            is_player_move=True,
        )

        # Record memory for significant moves
        asyncio.create_task(self._record_game_move_memory(
            session=session,
            student_id=student_id,
            move_notation=request.move,
            move_quality=move_quality.quality,
            is_best_move=move_quality.is_best_move,
        ))
        asyncio.create_task(self._record_procedural_observation(
            session=session,
            student_id=student_id,
            move_quality=move_quality.quality,
        ))

        # Build last_move info for frontend highlighting
        ai_player = "black" if session.player_color in ("white", "player1") else "white"
        ai_last_move = self._build_last_move(ai_move.move, ai_player, game_type)

        # Build response
        display = self._build_display(engine, ai_state.position, game_type, last_move=ai_last_move)

        return MakeMoveResponse(
            valid=True,
            display=display,
            move_quality=move_quality,
            ai_move=AIMoveInfo(
                move=ai_move.move,
                display=self._get_move_display(ai_move.move),
                explanation=ai_explanation,
            ),
            coach_message=coach_message,
            game_over=False,
            your_turn=True,
            move_number=session.total_moves,
            hints_remaining=self._get_hints_remaining(session),
        )

    async def get_hint(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: GetHintRequest,
    ) -> GetHintResponse:
        """Get a hint for the current position.

        Args:
            db: Database session.
            session_id: Game session ID.
            student_id: Student's ID.
            request: Hint request with level.

        Returns:
            GetHintResponse with hint text.

        Raises:
            ValueError: If session not found or no hints remaining.
        """
        session = await self._get_session(db, session_id)
        if not session:
            raise GameNotFoundError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise GameNotFoundError("Session does not belong to this student")

        # Check hints remaining
        hints_remaining = self._get_hints_remaining(session)
        if hints_remaining == 0:
            raise GameNotActiveError("No hints remaining")

        # Get engine and position
        game_type = GameType(session.game_type)
        engine = self._engine_registry.get(game_type)
        game_state = GameState.from_storage_dict(session.game_state)

        # Get hint from engine
        hint_response = engine.get_hint(
            game_state.position,
            hint_level=request.level,
        )

        # Get student info for personalization
        student = await db.get(User, session.student_id)
        student_name = student.first_name if student else "there"

        # Enhance hint with LLM using agent (analysis done inside)
        enhanced_hint = await self._generate_enhanced_hint(
            student_name=student_name,
            game_type=game_type,
            game_mode=GameMode(session.game_mode),
            position=game_state.position,
            hint_level=request.level,
        )

        # Increment hints used
        session.hints_used += 1
        await db.commit()

        # Get squares to highlight
        suggested_squares = None
        if hint_response.hint_level >= 2 and hint_response.reveals_move:
            analysis = engine.analyze_position(game_state.position)
            if analysis.best_move:
                suggested_squares = self._get_hint_squares(analysis.best_move)

        return GetHintResponse(
            hint_text=enhanced_hint,
            hint_level=hint_response.hint_level,
            hint_type=hint_response.hint_type,
            reveals_move=hint_response.reveals_move,
            suggested_squares=suggested_squares,
            hints_remaining=self._get_hints_remaining(session),
        )

    async def get_status(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> GameStatusResponse:
        """Get current game status.

        Args:
            db: Database session.
            session_id: Game session ID.
            student_id: Student's ID.

        Returns:
            GameStatusResponse with current state.
        """
        session = await self._get_session(db, session_id)
        if not session:
            raise GameNotFoundError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise GameNotFoundError("Session does not belong to this student")

        game_type = GameType(session.game_type)
        engine = self._engine_registry.get(game_type)
        game_state = GameState.from_storage_dict(session.game_state)

        display = self._build_display(engine, game_state.position, game_type)

        elapsed = int((datetime.now(timezone.utc) - session.started_at).total_seconds())

        return GameStatusResponse(
            session_id=str(session.id),
            game_type=game_type,
            game_mode=GameMode(session.game_mode),
            difficulty=GameDifficulty(session.difficulty),
            status=GameStatus(session.status),
            display=display,
            your_turn=self._is_player_turn(session.player_color, game_state.current_player),
            move_count=session.total_moves,
            stats=self._build_stats(session),
            hints_remaining=self._get_hints_remaining(session),
            time_elapsed_seconds=elapsed,
            game_result=session.result,
            result_reason=session.winner,
        )

    async def resign(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> ResignResponse:
        """Resign from the current game.

        Args:
            db: Database session.
            session_id: Game session ID.
            student_id: Student's ID.

        Returns:
            ResignResponse with summary.
        """
        session = await self._get_session(db, session_id)
        if not session:
            raise GameNotFoundError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise GameNotFoundError("Session does not belong to this student")

        if session.status != GameStatus.ACTIVE.value:
            raise GameNotActiveError(f"Game is not active: {session.status}")

        # Update session
        session.status = GameStatus.COMPLETED.value
        session.result = "loss"
        session.winner = "ai"
        session.ended_at = datetime.now(timezone.utc)

        # Store final position
        game_state = GameState.from_storage_dict(session.game_state)
        session.final_position = game_state.position.model_dump()

        # Generate personalized resign message via LLM
        game_type = GameType(session.game_type)
        student = await db.get(User, session.student_id)
        student_name = student.first_name if student else "there"

        coach_message = await self._generate_game_end_message(
            student_name=student_name,
            game_type=game_type,
            game_mode=GameMode(session.game_mode),
            position=game_state.position,
            result="loss",
            result_type="resignation",
            total_moves=session.total_moves,
            excellent_moves=session.excellent_moves_count,
            mistakes=session.mistakes_count,
        )

        await db.commit()

        # Publish session completed event
        await self._publish_session_completed(
            session=session,
            student_id=student_id,
            result="loss",
            result_reason="resignation",
        )

        # Record game completion to memory
        asyncio.create_task(self._record_game_completion_memory(
            session=session,
            student_id=student_id,
            result="loss",
            result_reason="resignation",
        ))

        return ResignResponse(
            session_id=str(session.id),
            game_result="loss",
            result_reason="resignation",
            coach_message=coach_message,
            stats=self._build_stats(session),
            learning_points=[],
        )

    async def analyze_game(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> AnalyzeGameResponse:
        """Get full game analysis.

        Args:
            db: Database session.
            session_id: Game session ID.
            student_id: Student's ID.

        Returns:
            AnalyzeGameResponse with detailed analysis.
        """
        session = await self._get_session(db, session_id)
        if not session:
            raise GameNotFoundError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise GameNotFoundError("Session does not belong to this student")

        # Get all moves
        moves = await self._get_moves(db, session_id)

        # Get student info for personalization
        student = await db.get(User, session.student_id)
        student_name = student.first_name if student else "there"

        # Find critical moments
        critical_moments = self._identify_critical_moments(moves)

        # Generate personalized summary via LLM
        coach_summary = await self._generate_analysis_summary(
            session, moves, critical_moments, student_name=student_name
        )

        # Generate personalized improvement tips via LLM
        improvement_tips = await self._generate_improvement_tips(
            session, moves, student_name=student_name
        )

        # Identify strengths and weaknesses
        strength_areas, weakness_areas = self._identify_strength_weakness(moves)

        stats = self._build_stats(session)

        return AnalyzeGameResponse(
            session_id=str(session.id),
            game_type=GameType(session.game_type),
            game_result=session.result or "unknown",
            result_reason=session.winner,
            overall_stats=stats,
            accuracy_percentage=stats.accuracy_percentage,
            phase_analysis=None,  # Chess-specific, implement later
            critical_moments=critical_moments,
            coach_summary=coach_summary,
            learning_points=[
                LearningPointInfo(point=lp, category="general")
                for lp in session.learning_points
            ],
            improvement_tips=improvement_tips,
            strength_areas=strength_areas,
            weakness_areas=weakness_areas,
            performance_rating=None,
        )

    async def get_available_games(
        self,
        db: AsyncSession,
        student_id: UUID,
    ) -> AvailableGamesResponse:
        """Get list of available games.

        Args:
            db: Database session.
            student_id: Student's ID.

        Returns:
            AvailableGamesResponse with games and active session.
        """
        games = [
            AvailableGame(
                game_type=GameType.CHESS,
                name="Chess",
                description="Play chess against AI with coaching",
                difficulty_levels=["beginner", "easy", "medium", "hard", "expert"],
                modes=["tutorial", "practice", "challenge"],
                icon="chess",
            ),
            AvailableGame(
                game_type=GameType.CONNECT4,
                name="Connect 4",
                description="Connect four pieces in a row",
                difficulty_levels=["beginner", "easy", "medium", "hard", "expert"],
                modes=["practice", "challenge"],
                icon="connect4",
            ),
            AvailableGame(
                game_type=GameType.GOMOKU,
                name="Gomoku",
                description="Get five stones in a row to win",
                difficulty_levels=["beginner", "easy", "medium", "hard", "expert"],
                modes=["practice", "challenge"],
                icon="gomoku",
            ),
            AvailableGame(
                game_type=GameType.OTHELLO,
                name="Othello",
                description="Flip your opponent's pieces to win",
                difficulty_levels=["beginner", "easy", "medium", "hard", "expert"],
                modes=["practice", "challenge"],
                icon="othello",
            ),
            AvailableGame(
                game_type=GameType.CHECKERS,
                name="Checkers",
                description="Jump and capture to win",
                difficulty_levels=["beginner", "easy", "medium", "hard", "expert"],
                modes=["practice", "challenge"],
                icon="checkers",
            ),
        ]

        # Check for active session
        active_session = None
        for game_type in [GameType.CHESS, GameType.CONNECT4, GameType.GOMOKU, GameType.OTHELLO, GameType.CHECKERS]:
            session = await self._get_active_session(db, student_id, game_type)
            if session:
                active_session = GameSessionSummary(
                    session_id=str(session.id),
                    game_type=GameType(session.game_type),
                    game_mode=GameMode(session.game_mode),
                    difficulty=GameDifficulty(session.difficulty),
                    status=GameStatus(session.status),
                    result=session.result,
                    total_moves=session.total_moves,
                    started_at=session.started_at,
                    ended_at=session.ended_at,
                )
                break

        return AvailableGamesResponse(
            games=games,
            active_session=active_session,
        )

    async def list_sessions(
        self,
        db: AsyncSession,
        student_id: UUID,
        game_type: GameType | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ListGamesResponse:
        """List game sessions for a student.

        Args:
            db: Database session.
            student_id: Student's ID.
            game_type: Optional filter by game type.
            limit: Maximum sessions to return.
            offset: Offset for pagination.

        Returns:
            ListGamesResponse with sessions.
        """
        query = select(GameSession).where(
            GameSession.student_id == str(student_id),
        )

        if game_type:
            query = query.where(GameSession.game_type == game_type.value)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Get sessions
        query = query.order_by(desc(GameSession.started_at)).offset(offset).limit(limit + 1)
        result = await db.execute(query)
        sessions = list(result.scalars().all())

        has_more = len(sessions) > limit
        if has_more:
            sessions = sessions[:limit]

        return ListGamesResponse(
            sessions=[
                GameSessionSummary(
                    session_id=str(s.id),
                    game_type=GameType(s.game_type),
                    game_mode=GameMode(s.game_mode),
                    difficulty=GameDifficulty(s.difficulty),
                    status=GameStatus(s.status),
                    result=s.result,
                    total_moves=s.total_moves,
                    started_at=s.started_at,
                    ended_at=s.ended_at,
                )
                for s in sessions
            ],
            total=total,
            has_more=has_more,
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_context_builder(self, game_type: GameType) -> GameContextBuilder:
        """Get a context builder for the specified game type.

        Args:
            game_type: Type of game.

        Returns:
            GameContextBuilder initialized with the appropriate engine.
        """
        engine = self._engine_registry.get(game_type)
        return GameContextBuilder(engine=engine)

    async def _get_session(
        self,
        db: AsyncSession,
        session_id: UUID,
    ) -> GameSession | None:
        """Fetch session from database."""
        return await db.get(GameSession, str(session_id))

    async def _get_active_session(
        self,
        db: AsyncSession,
        student_id: UUID,
        game_type: GameType,
    ) -> GameSession | None:
        """Get active session for student and game type."""
        query = (
            select(GameSession)
            .where(
                GameSession.student_id == str(student_id),
                GameSession.game_type == game_type.value,
                GameSession.status.in_([GameStatus.ACTIVE.value, GameStatus.PAUSED.value]),
            )
            .limit(1)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def _get_moves(
        self,
        db: AsyncSession,
        session_id: UUID,
    ) -> list[GameMove]:
        """Get all moves for a session."""
        query = (
            select(GameMove)
            .where(GameMove.session_id == str(session_id))
            .order_by(GameMove.move_number)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def _save_move(
        self,
        db: AsyncSession,
        session: GameSession,
        move_number: int,
        player: str,
        notation: str,
        position_before: Position,
        position_after: Position,
        quality: str | None = None,
        is_best_move: bool = False,
        best_move: str | None = None,
        time_spent_seconds: int = 0,
    ) -> GameMove:
        """Save a move to database."""
        move = GameMove(
            session_id=str(session.id),
            move_number=move_number,
            player=player,
            notation=notation,
            position_before=position_before.model_dump(),
            position_after=position_after.model_dump(),
            time_spent_seconds=time_spent_seconds,
            quality=quality,
            is_best_move=is_best_move,
            best_move=best_move,
        )
        db.add(move)
        return move

    def _get_hints_for_mode(self, mode: GameMode) -> int:
        """Get hints available for game mode."""
        hints_map = {
            GameMode.TUTORIAL: -1,  # Unlimited
            GameMode.PRACTICE: 5,
            GameMode.CHALLENGE: 0,
            GameMode.PUZZLE: 3,
            GameMode.ANALYSIS: -1,
        }
        return hints_map.get(mode, 3)

    def _get_hints_remaining(self, session: GameSession) -> int:
        """Calculate remaining hints."""
        mode = GameMode(session.game_mode)
        total = self._get_hints_for_mode(mode)
        if total == -1:
            return -1
        return max(0, total - session.hints_used)

    def _is_player_turn(self, player_color: str, current_player: str) -> bool:
        """Check if it's player's turn."""
        if player_color in ("white", "player1"):
            return current_player in ("white", "player1")
        else:
            return current_player in ("black", "player2")

    def _build_display(
        self,
        engine: Any,
        position: Position,
        game_type: GameType,
        last_move: dict[str, str] | None = None,
    ) -> BoardDisplay:
        """Build display data for frontend.

        Args:
            engine: Game engine instance.
            position: Current board position.
            game_type: Type of game.
            last_move: Last move info for highlighting (optional).
                Format: {"from": "e2", "to": "e4", "player": "white"}
                For drop games (Gomoku, Connect4): {"to": "h8", "player": "white"}

        Returns:
            BoardDisplay with current position and last move info.
        """
        display_data = engine.position_to_display(position)

        return BoardDisplay(
            fen=display_data.get("fen"),
            grid=display_data.get("grid"),
            pieces=display_data.get("pieces"),
            turn=display_data.get("turn", "white"),
            legal_moves=None,  # Can be computed if needed
            in_check=display_data.get("in_check", False),
            last_move=last_move,
        )

    def _build_last_move(
        self,
        move_notation: str,
        player: str,
        game_type: GameType,
    ) -> dict[str, str]:
        """Build last_move dict for frontend highlighting.

        Args:
            move_notation: Move in game notation (e2e4, h8, 3, etc.)
            player: Player who made the move (white/black, player1/player2)
            game_type: Type of game.

        Returns:
            Dict with move info: {"from": "e2", "to": "e4", "player": "white"}
            For drop games: {"to": "h8", "player": "white"}
        """
        if game_type == GameType.CHESS:
            # UCI notation: e2e4
            if len(move_notation) >= 4:
                return {
                    "from": move_notation[:2],
                    "to": move_notation[2:4],
                    "player": player,
                }
        elif game_type == GameType.CONNECT4:
            # Column number: 3
            return {
                "to": f"col{move_notation}",
                "player": player,
            }
        elif game_type in (GameType.GOMOKU, GameType.OTHELLO):
            # Coordinate notation: h8, d3
            return {
                "to": move_notation,
                "player": player,
            }
        elif game_type == GameType.CHECKERS:
            # From-to notation: c3-d4 or c3xd4
            parts = move_notation.replace("x", "-").split("-")
            if len(parts) == 2:
                return {
                    "from": parts[0],
                    "to": parts[1],
                    "player": player,
                }

        # Fallback
        return {"to": move_notation, "player": player}

    def _build_stats(self, session: GameSession) -> SessionStats:
        """Build session statistics."""
        return SessionStats(
            total_moves=session.total_moves,
            excellent_moves=session.excellent_moves_count,
            mistakes=session.mistakes_count,
            hints_used=session.hints_used,
            time_spent_seconds=session.time_spent_seconds,
        )

    def _assess_move_quality(
        self,
        played_move: str,
        best_move: str | None,
        eval_before: float | None,
        eval_after: float | None,
        player_is_white: bool = True,
    ) -> MoveQualityInfo:
        """Assess the quality of a move using centipawn loss.

        Quality categories based on centipawn loss:
        - excellent: best move or <10cp loss
        - good: 10-50cp loss
        - inaccuracy: 50-100cp loss
        - mistake: 100-300cp loss
        - blunder: >300cp loss

        Args:
            played_move: The move that was played.
            best_move: Engine's best move.
            eval_before: Position evaluation before the move (white perspective).
            eval_after: Position evaluation after the move (white perspective).
            player_is_white: True if player is white/player1.

        Returns:
            MoveQualityInfo with quality assessment.
        """
        is_best = played_move == best_move if best_move else False

        # If it's the best move, it's excellent
        if is_best:
            return MoveQualityInfo(
                quality="excellent",
                is_best_move=True,
                evaluation_change=0.0,
            )

        # Calculate centipawn loss
        # Evaluation is from white's perspective (positive = white better)
        # For white player: loss = eval_before - eval_after (if positive, position got worse)
        # For black player: loss = eval_after - eval_before (if positive, position got worse for black = better for white)
        if eval_before is not None and eval_after is not None:
            if player_is_white:
                # White wants higher eval, so loss = before - after
                eval_change = eval_after - eval_before  # Negative means worse for white
                cp_loss = -eval_change * 100  # Convert to centipawns, positive = loss
            else:
                # Black wants lower eval, so loss = after - before
                eval_change = eval_after - eval_before  # Positive means worse for black
                cp_loss = eval_change * 100  # Convert to centipawns, positive = loss

            # Determine quality based on centipawn loss
            if cp_loss < 10:
                quality = "excellent"
            elif cp_loss < 50:
                quality = "good"
            elif cp_loss < 100:
                quality = "inaccuracy"
            elif cp_loss < 300:
                quality = "mistake"
            else:
                quality = "blunder"

            return MoveQualityInfo(
                quality=quality,
                is_best_move=False,
                evaluation_change=round(eval_change, 2) if eval_change else None,
            )

        # Fallback if no evaluation available
        return MoveQualityInfo(
            quality="good",
            is_best_move=False,
            evaluation_change=None,
        )

    def _update_session_stats(self, session: GameSession, quality: str | None) -> None:
        """Update session statistics based on move quality.

        Quality categories:
        - excellent: Counts towards excellent_moves_count
        - good: No special tracking
        - inaccuracy: Counts towards mistakes_count (minor)
        - mistake: Counts towards mistakes_count
        - blunder: Counts towards mistakes_count (major)
        """
        if quality == "excellent":
            session.excellent_moves_count += 1
        elif quality in ("inaccuracy", "mistake", "blunder"):
            session.mistakes_count += 1

    async def _generate_greeting(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        difficulty: GameDifficulty,
        player_color: str,
        language: str,
        grade_level: int = 5,
    ) -> str:
        """Generate personalized coach greeting via LLM.

        Uses GameCoachAgent with YAML-driven prompts for consistent,
        persona-aware greetings.

        Args:
            student_name: Student's first name.
            game_type: Type of game (chess, connect4).
            game_mode: Game mode (tutorial, practice, challenge).
            difficulty: Difficulty level.
            player_color: Player's color/side.
            language: Language for the message.
            grade_level: Student's grade level for age-appropriate language.

        Returns:
            Personalized greeting message.
        """
        try:
            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_greeting_context(
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                    player_color=player_color,
                ),
                game_mode=game_mode,
                difficulty=difficulty,
            )

            greeting = await self._get_coach_agent(game_type).generate_greeting(context)
            if greeting:
                return greeting

        except Exception as e:
            logger.warning("Failed to generate greeting via agent: %s", str(e))

        # Fallback
        game_name = "Chess" if game_type == GameType.CHESS else "Connect 4"
        return f"Hi {student_name}! Ready to play {game_name}? Let's have fun and learn together!"

    async def _generate_enhanced_hint(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        position: Position,
        hint_level: int,
        language: str = "en",
        grade_level: int = 5,
    ) -> str:
        """Generate personalized hint via LLM.

        Uses GameCoachAgent with position context for accurate,
        age-appropriate hints. Analysis is performed inside this method.

        Args:
            student_name: Student's first name.
            game_type: Type of game.
            game_mode: Current game mode.
            position: Current game position.
            hint_level: Hint level (1=strategic, 2=tactical, 3=solution).
            language: Language for the hint.
            grade_level: Student's grade level.

        Returns:
            Personalized hint message.
        """
        try:
            context_builder = self._get_context_builder(game_type)
            engine = self._engine_registry.get(game_type)
            analysis = engine.analyze_position(position, depth=10)

            context = context_builder.build_hint_context(
                position=position,
                analysis=analysis,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=game_mode,
                hint_level=hint_level,
            )

            hint = await self._get_coach_agent(game_type).generate_hint(context)
            if hint:
                return hint

        except Exception as e:
            logger.warning("Failed to generate hint via agent: %s", str(e))

        # Fallback
        return "Think about controlling the center."

    async def _generate_invalid_move_message(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        position: Position,
        invalid_move: str,
        invalid_reason: str | None,
        language: str = "en",
        grade_level: int = 5,
    ) -> str:
        """Generate encouraging message for invalid move via LLM.

        Uses GameCoachAgent with position context for helpful,
        educational feedback.

        Args:
            student_name: Student's first name.
            game_type: Type of game.
            game_mode: Current game mode.
            position: Current game position.
            invalid_move: The invalid move attempted.
            invalid_reason: Error message from engine.
            language: Language for the message.
            grade_level: Student's grade level.

        Returns:
            Encouraging message about the invalid move.
        """
        try:
            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_invalid_move_context(
                position=position,
                invalid_move=invalid_move,
                invalid_reason=invalid_reason or "not allowed",
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
            )

            message = await self._get_coach_agent(game_type).generate_invalid_move_message(context)
            if message:
                return message

        except Exception as e:
            logger.warning("Failed to generate invalid move message via agent: %s", str(e))

        # Fallback
        return "That move isn't allowed here. Take another look and try again!"

    async def _generate_coach_message(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        position_before: Position,
        position_after: Position,
        move_notation: str,
        move_quality: MoveQualityInfo,
        analysis: Any,
        language: str = "en",
        grade_level: int = 5,
    ) -> str:
        """Generate personalized coach message via LLM.

        Uses GameCoachAgent with full position context for accurate,
        educational feedback that can see the board state.

        Args:
            student_name: Student's first name.
            game_type: Type of game.
            game_mode: Current game mode.
            position_before: Position before the move.
            position_after: Position after the move.
            move_notation: The move that was played.
            move_quality: Quality assessment of the move.
            analysis: Position analysis from engine.
            language: Language for the message.
            grade_level: Student's grade level.

        Returns:
            Personalized coach feedback message.
        """
        # Challenge mode: minimal feedback
        if game_mode == GameMode.CHALLENGE:
            return "Your move."

        try:
            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_move_context(
                position_before=position_before,
                position_after=position_after,
                move=move_notation,
                player="player",
                analysis=analysis,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=game_mode,
                move_quality=move_quality.quality,
                is_best_move=move_quality.is_best_move,
            )

            message = await self._get_coach_agent(game_type).generate_move_comment(context)
            if message:
                return message

        except Exception as e:
            logger.warning("Failed to generate coach message via agent: %s", str(e))

        # Fallback messages
        quality = move_quality.quality or "good"
        if quality == "excellent":
            return "Excellent move! Well played!"
        elif quality in ("mistake", "blunder"):
            return "Interesting choice. Keep thinking about what your opponent might do next!"
        return "Good move! Your turn."

    async def _generate_ai_move_explanation(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        position_before: Position,
        position_after: Position,
        ai_move: str,
        language: str = "en",
        grade_level: int = 5,
    ) -> str | None:
        """Generate explanation for AI's move (tutorial mode only).

        Uses GameCoachAgent to explain AI's reasoning in first person,
        helping students understand strategic thinking.

        Args:
            student_name: Student's first name.
            game_type: Type of game.
            game_mode: Current game mode (should be tutorial).
            position_before: Position before AI's move.
            position_after: Position after AI's move.
            ai_move: The move AI played.
            language: Language for the explanation.
            grade_level: Student's grade level.

        Returns:
            AI move explanation or None if not tutorial mode.
        """
        # Only explain in tutorial mode
        if game_mode != GameMode.TUTORIAL:
            return None

        try:
            context_builder = self._get_context_builder(game_type)
            engine = self._engine_registry.get(game_type)

            # Get analysis for AI's move
            analysis = engine.analyze_position(position_before, depth=10)

            context = context_builder.build_ai_move_context(
                position_before=position_before,
                position_after=position_after,
                ai_move=ai_move,
                analysis=analysis,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=game_mode,
            )

            logger.info("Generating AI move explanation for %s mode", game_mode.value)
            explanation = await self._get_coach_agent(game_type).generate_ai_move_explanation(context)
            logger.info("AI move explanation result: %s", explanation[:50] if explanation else "None/Empty")
            if explanation:
                return explanation
            else:
                logger.warning("AI explanation was empty, using fallback")

        except Exception as e:
            logger.warning("Failed to generate AI move explanation via agent: %s", str(e))

        # Fallback
        return "I made that move to improve my position."

    async def _generate_game_end_message(
        self,
        student_name: str,
        game_type: GameType,
        game_mode: GameMode,
        position: Position,
        result: str,
        result_type: str,
        total_moves: int,
        excellent_moves: int = 0,
        mistakes: int = 0,
        language: str = "en",
        grade_level: int = 5,
    ) -> str:
        """Generate personalized game-end message via LLM.

        Uses GameCoachAgent with game context for personalized,
        encouraging feedback.

        Args:
            student_name: Student's first name.
            game_type: Type of game.
            game_mode: Current game mode.
            position: Final game position.
            result: Game result (win, loss, draw).
            result_type: How the game ended (checkmate, resignation, etc.).
            total_moves: Total moves played.
            excellent_moves: Number of excellent moves.
            mistakes: Number of mistakes.
            language: Language for the message.
            grade_level: Student's grade level.

        Returns:
            Personalized game-end message.
        """
        try:
            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_game_end_context(
                position=position,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=game_mode,
                game_result=result,
                result_reason=result_type,
                session_stats={
                    "total_moves": total_moves,
                    "excellent_moves": excellent_moves,
                    "mistakes": mistakes,
                },
            )

            message = await self._get_coach_agent(game_type).generate_game_end_message(context)
            if message:
                return message

        except Exception as e:
            logger.warning("Failed to generate game end message via agent: %s", str(e))

        # Fallback messages
        if result == "win":
            return f"Congratulations, {student_name}! You won by {result_type}! Great game!"
        elif result == "loss":
            return f"Good game, {student_name}! You played well. Every game is a chance to learn. Want to play again?"
        return f"It's a draw by {result_type}. Well played, {student_name}!"

    async def _handle_game_end(
        self,
        db: AsyncSession,
        session: GameSession,
        engine: Any,
        game_state: GameState,
        game_type: GameType,
        result_type: str | None,
        winner: str | None,
        player_color: str,
        move_quality: MoveQualityInfo,
        ai_move_notation: str | None = None,
    ) -> MakeMoveResponse:
        """Handle game ending."""
        # Determine result from player's perspective
        if winner is None:
            result = "draw"
        elif self._is_player_turn(player_color, winner):
            result = "win"
        else:
            result = "loss"

        # Update session
        session.status = GameStatus.COMPLETED.value
        session.result = result
        session.winner = winner
        session.ended_at = datetime.now(timezone.utc)
        session.final_position = game_state.position.model_dump()
        session.game_state = game_state.to_storage_dict()

        # Generate personalized game-end message via LLM
        student = await db.get(User, session.student_id)
        student_name = student.first_name if student else "there"

        coach_message = await self._generate_game_end_message(
            student_name=student_name,
            game_type=game_type,
            game_mode=GameMode(session.game_mode),
            position=game_state.position,
            result=result,
            result_type=result_type or "completion",
            total_moves=session.total_moves,
            excellent_moves=session.excellent_moves_count,
            mistakes=session.mistakes_count,
        )

        await db.commit()

        # Publish session completed event
        await self._publish_session_completed(
            session=session,
            student_id=UUID(session.student_id),
            result=result,
            result_reason=result_type,
        )

        # Record game completion to memory
        asyncio.create_task(self._record_game_completion_memory(
            session=session,
            student_id=UUID(session.student_id),
            result=result,
            result_reason=result_type,
        ))

        # Build last_move if AI made the final move
        last_move = None
        if ai_move_notation:
            ai_player = "black" if player_color in ("white", "player1") else "white"
            last_move = self._build_last_move(ai_move_notation, ai_player, game_type)

        display = self._build_display(engine, game_state.position, game_type, last_move=last_move)

        return MakeMoveResponse(
            valid=True,
            display=display,
            move_quality=move_quality,
            ai_move=AIMoveInfo(move=ai_move_notation) if ai_move_notation else None,
            coach_message=coach_message,
            game_over=True,
            game_result=result,
            result_reason=result_type,
            your_turn=False,
            move_number=session.total_moves,
            hints_remaining=0,
        )

    def _get_move_display(self, move: str) -> dict[str, str] | None:
        """Get move display info for animation."""
        if len(move) >= 4:
            return {
                "from_square": move[:2],
                "to_square": move[2:4],
            }
        return None

    def _get_hint_squares(self, move: str) -> list[str]:
        """Extract squares to highlight from a move."""
        if len(move) >= 4:
            return [move[:2], move[2:4]]
        return []

    def _identify_critical_moments(self, moves: list[GameMove]) -> list[CriticalMoment]:
        """Identify critical moments in the game."""
        critical = []
        for move in moves:
            if move.player == "player" and move.quality in ("mistake", "blunder"):
                critical.append(CriticalMoment(
                    move_number=move.move_number,
                    position_fen=move.position_before.get("notation"),
                    player_move=move.notation,
                    best_move=move.best_move,
                    explanation=f"This was a {move.quality}. Consider {move.best_move} instead.",
                    category=move.quality or "mistake",
                ))
        return critical

    async def _generate_analysis_summary(
        self,
        session: GameSession,
        moves: list[GameMove],
        critical_moments: list[CriticalMoment],
        student_name: str = "there",
        language: str = "en",
        grade_level: int = 5,
    ) -> str:
        """Generate personalized game analysis summary via LLM.

        Uses GameCoachAgent for consistent, persona-aware analysis.

        Args:
            session: Game session.
            moves: List of game moves.
            critical_moments: List of critical moments.
            student_name: Student's first name.
            language: Language for the message.
            grade_level: Student's grade level.

        Returns:
            Personalized analysis summary.
        """
        player_moves = [m for m in moves if m.player == "player"]
        total = len(player_moves)
        excellent = sum(1 for m in player_moves if m.quality == "excellent")
        mistakes = len(critical_moments)
        game_type = GameType(session.game_type)

        try:
            # Get final position for context
            final_position = Position(notation=session.final_position.get("notation", ""))

            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_game_end_context(
                position=final_position,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=GameMode(session.game_mode),
                game_result=session.result or "unknown",
                result_reason=session.winner or "",
                session_stats={
                    "total_moves": total,
                    "excellent_moves": excellent,
                    "mistakes": mistakes,
                    "hints_used": session.hints_used,
                },
            )

            summary = await self._get_coach_agent(game_type).generate_analysis_summary(context)
            if summary:
                return summary

        except Exception as e:
            logger.warning("Failed to generate analysis summary via agent: %s", str(e))

        # Fallback
        if session.result == "win":
            return f"Great win, {student_name}! You played {total} moves with {excellent} excellent moves. Keep practicing!"
        elif session.result == "loss":
            return f"Good effort, {student_name}! You played {total} moves. Review the {mistakes} critical moments to improve."
        return f"Solid game ending in a draw. {total} moves played. Well done!"

    async def _generate_improvement_tips(
        self,
        session: GameSession,
        moves: list[GameMove],
        student_name: str = "there",
        language: str = "en",
        grade_level: int = 5,
    ) -> list[str]:
        """Generate personalized improvement tips via LLM.

        Uses GameCoachAgent for consistent, persona-aware tips.

        Args:
            session: Game session.
            moves: List of game moves.
            student_name: Student's first name.
            language: Language for the tips.
            grade_level: Student's grade level.

        Returns:
            List of personalized improvement tips.
        """
        player_moves = [m for m in moves if m.player == "player"]
        excellent = sum(1 for m in player_moves if m.quality == "excellent")
        mistakes = sum(1 for m in player_moves if m.quality in ("mistake", "blunder"))
        game_type = GameType(session.game_type)

        try:
            # Get final position for context
            final_position = Position(notation=session.final_position.get("notation", ""))

            context_builder = self._get_context_builder(game_type)
            context = context_builder.build_game_end_context(
                position=final_position,
                student=StudentContext(
                    name=student_name,
                    grade_level=grade_level,
                    language=language,
                ),
                game_mode=GameMode(session.game_mode),
                game_result=session.result or "unknown",
                result_reason=session.winner or "",
                session_stats={
                    "total_moves": len(player_moves),
                    "excellent_moves": excellent,
                    "mistakes": mistakes,
                    "hints_used": session.hints_used,
                },
            )

            tips = await self._get_coach_agent(game_type).generate_improvement_tips(context)
            if tips:
                return tips

        except Exception as e:
            logger.warning("Failed to generate improvement tips via agent: %s", str(e))

        # Fallback
        tips = []
        if session.hints_used > 3:
            tips.append("Try to rely less on hints and think through positions yourself.")
        if mistakes > 2:
            tips.append("Take your time before each move to check for threats.")
        if session.game_type == "chess":
            tips.append("Study common opening principles to get better positions.")
        if not tips:
            tips.append("Keep practicing to improve your skills!")
        return tips

    def _identify_strength_weakness(
        self,
        moves: list[GameMove],
    ) -> tuple[list[str], list[str]]:
        """Identify strength and weakness areas."""
        player_moves = [m for m in moves if m.player == "player"]

        excellent = sum(1 for m in player_moves if m.quality == "excellent")
        mistakes = sum(1 for m in player_moves if m.quality in ("mistake", "blunder"))

        strengths = []
        weaknesses = []

        if excellent > len(player_moves) * 0.5:
            strengths.append("Finding strong moves")
        if mistakes < len(player_moves) * 0.1:
            strengths.append("Avoiding blunders")

        if mistakes > len(player_moves) * 0.3:
            weaknesses.append("Move accuracy needs improvement")

        return strengths, weaknesses

    # =========================================================================
    # Event Publishing Methods
    # =========================================================================

    async def _publish_session_started(
        self,
        session: GameSession,
        student_id: UUID,
    ) -> None:
        """Publish gaming.session.started event.

        Args:
            session: The created game session.
            student_id: Student's ID.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                if not self._tenant_code:
                    logger.debug("No tenant_code available, skipping event publish")
                    return
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=self._tenant_code)

            await tracker.track_event(
                event_type="gaming.session.started",
                student_id=student_id,
                session_id=UUID(session.id),
                data={
                    "game_type": session.game_type,
                    "game_mode": session.game_mode,
                    "difficulty": session.difficulty,
                    "player_color": session.player_color,
                },
            )
            logger.debug(
                "Published gaming.session.started: session_id=%s, game_type=%s",
                session.id, session.game_type,
            )
        except Exception as e:
            logger.warning("Failed to publish gaming.session.started: %s", e)

    async def _publish_move_made(
        self,
        session: GameSession,
        student_id: UUID,
        move_notation: str,
        move_quality: str | None,
        is_player_move: bool,
    ) -> None:
        """Publish gaming.move.made event.

        Args:
            session: The game session.
            student_id: Student's ID.
            move_notation: The move in notation.
            move_quality: Quality assessment of the move.
            is_player_move: True if this is the player's move.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                if not self._tenant_code:
                    logger.debug("No tenant_code available, skipping event publish")
                    return
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=self._tenant_code)

            await tracker.track_event(
                event_type="gaming.move.made",
                student_id=student_id,
                session_id=UUID(session.id),
                data={
                    "game_type": session.game_type,
                    "move": move_notation,
                    "move_number": session.total_moves,
                    "move_quality": move_quality,
                    "is_player_move": is_player_move,
                },
            )
            logger.debug(
                "Published gaming.move.made: session_id=%s, move=%s, quality=%s",
                session.id, move_notation, move_quality,
            )
        except Exception as e:
            logger.warning("Failed to publish gaming.move.made: %s", e)

    async def _publish_session_completed(
        self,
        session: GameSession,
        student_id: UUID,
        result: str,
        result_reason: str | None,
    ) -> None:
        """Publish gaming.session.completed event.

        Args:
            session: The completed game session.
            student_id: Student's ID.
            result: Game result (win, loss, draw).
            result_reason: How the game ended.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                if not self._tenant_code:
                    logger.debug("No tenant_code available, skipping event publish")
                    return
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=self._tenant_code)

            await tracker.track_event(
                event_type="gaming.session.completed",
                student_id=student_id,
                session_id=UUID(session.id),
                data={
                    "game_type": session.game_type,
                    "game_mode": session.game_mode,
                    "difficulty": session.difficulty,
                    "result": result,
                    "result_reason": result_reason,
                    "total_moves": session.total_moves,
                    "excellent_moves": session.excellent_moves_count,
                    "mistakes": session.mistakes_count,
                    "hints_used": session.hints_used,
                },
            )
            logger.debug(
                "Published gaming.session.completed: session_id=%s, result=%s",
                session.id, result,
            )
        except Exception as e:
            logger.warning("Failed to publish gaming.session.completed: %s", e)

    # =========================================================================
    # Memory Recording Methods
    # =========================================================================

    async def _record_game_move_memory(
        self,
        session: GameSession,
        student_id: UUID,
        move_notation: str,
        move_quality: str | None,
        is_best_move: bool,
    ) -> None:
        """Record significant game moves to episodic memory.

        Only records excellent, mistake, or blunder moves to avoid
        overwhelming memory with routine moves.

        Args:
            session: The game session.
            student_id: Student's ID.
            move_notation: The move in notation.
            move_quality: Quality assessment (excellent, good, mistake, blunder).
            is_best_move: Whether this was the best available move.
        """
        if self._memory_manager is None:
            return

        # Only record significant moves
        if move_quality not in ("excellent", "mistake", "blunder"):
            return

        try:
            importance = 0.7 if move_quality == "excellent" else 0.5
            game_topic = f"game_{session.game_type}"

            await self._memory_manager.record_learning_event(
                tenant_code=self._tenant_code or "default",
                student_id=student_id,
                event_type="game_move",
                topic=game_topic,
                data={
                    "session_id": session.id,
                    "game_type": session.game_type,
                    "move_notation": move_notation,
                    "move_quality": move_quality,
                    "move_number": session.total_moves,
                    "is_best_move": is_best_move,
                },
                importance=importance,
            )
            logger.debug(
                "Recorded game move memory: session_id=%s, move=%s, quality=%s",
                session.id, move_notation, move_quality,
            )
        except Exception as e:
            logger.warning("Failed to record game move memory: %s", e)

    async def _record_game_completion_memory(
        self,
        session: GameSession,
        student_id: UUID,
        result: str,
        result_reason: str | None,
    ) -> None:
        """Record game completion to episodic memory.

        Args:
            session: The completed game session.
            student_id: Student's ID.
            result: Game result (win, loss, draw).
            result_reason: How the game ended.
        """
        if self._memory_manager is None:
            return

        try:
            game_topic = f"game_{session.game_type}"

            await self._memory_manager.record_learning_event(
                tenant_code=self._tenant_code or "default",
                student_id=student_id,
                event_type="game_completed",
                topic=game_topic,
                data={
                    "session_id": session.id,
                    "game_type": session.game_type,
                    "game_mode": session.game_mode,
                    "difficulty": session.difficulty,
                    "result": result,
                    "result_reason": result_reason,
                    "total_moves": session.total_moves,
                    "excellent_moves": session.excellent_moves_count,
                    "mistakes": session.mistakes_count,
                    "hints_used": session.hints_used,
                },
                importance=0.7,
            )
            logger.debug(
                "Recorded game completion memory: session_id=%s, result=%s",
                session.id, result,
            )
        except Exception as e:
            logger.warning("Failed to record game completion memory: %s", e)

    async def _record_procedural_observation(
        self,
        session: GameSession,
        student_id: UUID,
        move_quality: str | None,
        thinking_time_seconds: float | None = None,
    ) -> None:
        """Record gaming patterns to procedural memory.

        Records patterns about how the student plays games, including
        thinking time and move quality distribution.

        Args:
            session: The game session.
            student_id: Student's ID.
            move_quality: Quality of the move.
            thinking_time_seconds: Time spent thinking on the move.
        """
        if self._memory_manager is None:
            return

        try:
            # Determine time of day bucket (0-3 for quarters of day)
            hour = datetime.now().hour
            time_of_day = hour // 6

            await self._memory_manager.record_procedural_observation(
                tenant_code=self._tenant_code or "default",
                student_id=student_id,
                observation={
                    "game_type": session.game_type,
                    "difficulty": session.difficulty,
                    "thinking_time_seconds": thinking_time_seconds or 0,
                    "move_quality": move_quality,
                    "time_of_day": time_of_day,
                },
            )
            logger.debug(
                "Recorded procedural observation: session_id=%s, quality=%s",
                session.id, move_quality,
            )
        except Exception as e:
            logger.warning("Failed to record procedural observation: %s", e)
