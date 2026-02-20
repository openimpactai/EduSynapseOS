# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Coach workflow state.

This module defines the state structure for game coaching workflows.
The state tracks:
- Game session configuration (game type, mode, difficulty)
- Board position and move history
- Engine analysis data for coach responses
- Performance metrics and learning points
- Coach messages and hints

The Game Coach workflow uses interrupt points to wait for player moves,
similar to Practice workflow waiting for student answers.
"""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class MoveRecord(TypedDict, total=False):
    """Record of a move in the game."""

    move_number: int
    player: Literal["player", "ai"]
    notation: str
    position_before: dict[str, Any]
    position_after: dict[str, Any]
    evaluation_before: float | None
    evaluation_after: float | None
    quality: str | None
    is_best_move: bool
    best_move: str | None
    coach_comment: str | None
    time_spent_seconds: int


class GameStats(TypedDict, total=False):
    """Statistics for the game session."""

    total_moves: int
    excellent_moves: int
    good_moves: int
    inaccuracies: int
    mistakes: int
    blunders: int
    hints_used: int
    time_spent_seconds: int


class GameCoachState(TypedDict, total=False):
    """State for game coach workflow.

    Manages the state of a game coaching session where a student
    plays strategy games (chess, connect4) with AI coaching.

    Attributes:
        # Session Identity
        session_id: Unique session identifier (game session ID).
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.
        student_id: Student identifier.
        student_name: Student's name for personalization.
        grade_level: Student's grade level for age-appropriate language.
        language: Language for coach messages.

        # Game Configuration
        game_type: Type of game (chess, connect4).
        game_mode: Game mode (tutorial, practice, challenge, puzzle, analysis).
        difficulty: AI difficulty level.
        player_color: Student's color/side.
        hints_available: Hints available (-1 = unlimited).

        # Persona
        persona_id: Selected persona for coaching.
        persona_name: Persona display name.

        # Game State
        current_position: Current board position (FEN for chess, grid for connect4).
        move_history: List of all moves played.
        current_turn: Whose turn it is (player or ai).
        move_number: Current move number.

        # Engine Analysis
        last_analysis: Analysis from game engine for current position.
        last_move_quality: Quality of the last player move.
        last_best_move: Best move according to engine.

        # Session Status
        status: Current session status.
        your_turn: Whether it's the player's turn.
        game_over: Whether the game has ended.
        game_result: Result if game over (win, loss, draw).
        result_reason: Reason for game end (checkmate, resignation, etc).

        # Pending move data (set via aupdate_state during send_move)
        _pending_move: Move injected during resume.
        _pending_time_spent: Time spent on the move.

        # Coach Responses
        first_greeting: Initial greeting from coach.
        last_coach_message: Most recent coach message.
        pending_hint: Pending hint if requested.

        # Performance Tracking
        stats: Game statistics.
        hints_remaining: Hints remaining.

        # Learning
        learning_points: Key learning points from the game.
        critical_moments: Critical moments identified.

        # Context
        memory_context: Memory context for personalization.
        messages: LangGraph messages for context.

        # Timestamps
        started_at: Session start time.
        last_activity_at: Last interaction time.
        ended_at: Session end time.

        # Error handling
        error: Error message if workflow failed.
    """

    # Session Identity
    session_id: str
    tenant_id: str
    tenant_code: str
    student_id: str
    student_name: str
    grade_level: int
    language: str

    # Game Configuration
    game_type: str
    game_mode: str
    difficulty: str
    player_color: str
    hints_available: int

    # Persona
    persona_id: str
    persona_name: str | None

    # Game State
    current_position: dict[str, Any]
    move_history: list[MoveRecord]
    current_turn: Literal["player", "ai"]
    move_number: int

    # Engine Analysis
    last_analysis: dict[str, Any] | None
    last_move_quality: str | None
    last_best_move: str | None

    # Session Status
    status: Literal["pending", "active", "paused", "completed", "error"]
    your_turn: bool
    game_over: bool
    game_result: str | None
    result_reason: str | None

    # Pending move data
    _pending_move: str | None
    _pending_time_spent: int | None

    # Coach Responses
    first_greeting: str | None
    last_coach_message: str | None
    pending_hint: dict[str, Any] | None

    # Performance Tracking
    stats: GameStats
    hints_remaining: int

    # Learning
    learning_points: list[str]
    critical_moments: list[dict[str, Any]]

    # Context
    memory_context: dict[str, Any]
    messages: Annotated[list[dict[str, Any]], add_messages]

    # Timestamps
    started_at: str
    last_activity_at: str
    ended_at: str | None

    # Error handling
    error: str | None


def create_initial_game_coach_state(
    session_id: str,
    tenant_id: str,
    tenant_code: str,
    student_id: str,
    student_name: str,
    grade_level: int,
    game_type: str,
    game_mode: str = "practice",
    difficulty: str = "medium",
    player_color: str = "white",
    language: str = "en",
    persona_id: str = "game_coach",
    initial_position: dict[str, Any] | None = None,
    hints_available: int = -1,
) -> GameCoachState:
    """Create initial state for a game coaching session.

    Args:
        session_id: Unique session identifier.
        tenant_id: Tenant UUID as string.
        tenant_code: Tenant code for memory operations.
        student_id: Student identifier.
        student_name: Student's name for personalization.
        grade_level: Student's grade level.
        game_type: Type of game (chess, connect4).
        game_mode: Game mode (tutorial, practice, challenge).
        difficulty: AI difficulty level.
        player_color: Student's color/side.
        language: Language for coach messages.
        persona_id: Persona to use.
        initial_position: Custom starting position (for puzzles).
        hints_available: Hints available (-1 = unlimited).

    Returns:
        Initial GameCoachState.
    """
    now = datetime.now().isoformat()

    # Determine initial turn based on color
    # For chess: white moves first
    # For connect4: player1 moves first
    your_turn = True
    if game_type == "chess":
        your_turn = player_color.lower() in ["white", "w"]
    elif game_type == "connect4":
        your_turn = player_color.lower() in ["player1", "p1", "1"]

    return GameCoachState(
        # Session Identity
        session_id=session_id,
        tenant_id=tenant_id,
        tenant_code=tenant_code,
        student_id=student_id,
        student_name=student_name,
        grade_level=grade_level,
        language=language,
        # Game Configuration
        game_type=game_type,
        game_mode=game_mode,
        difficulty=difficulty,
        player_color=player_color,
        hints_available=hints_available,
        # Persona
        persona_id=persona_id,
        persona_name=None,
        # Game State
        current_position=initial_position or {},
        move_history=[],
        current_turn="player" if your_turn else "ai",
        move_number=1,
        # Engine Analysis
        last_analysis=None,
        last_move_quality=None,
        last_best_move=None,
        # Session Status
        status="pending",
        your_turn=your_turn,
        game_over=False,
        game_result=None,
        result_reason=None,
        # Pending move
        _pending_move=None,
        _pending_time_spent=None,
        # Coach Responses
        first_greeting=None,
        last_coach_message=None,
        pending_hint=None,
        # Performance
        stats=GameStats(
            total_moves=0,
            excellent_moves=0,
            good_moves=0,
            inaccuracies=0,
            mistakes=0,
            blunders=0,
            hints_used=0,
            time_spent_seconds=0,
        ),
        hints_remaining=hints_available,
        # Learning
        learning_points=[],
        critical_moments=[],
        # Context
        memory_context={},
        messages=[],
        # Timestamps
        started_at=now,
        last_activity_at=now,
        ended_at=None,
        # Error
        error=None,
    )
