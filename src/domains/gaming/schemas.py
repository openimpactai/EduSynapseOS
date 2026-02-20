# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API schemas for gaming domain.

This module defines Pydantic models for API request/response:
- StartGameRequest/Response: Start a new game
- MakeMoveRequest/Response: Submit a move
- GetHintRequest/Response: Request a hint
- GameStatusResponse: Current game status
- EndGameResponse: Game completion details
- GameAnalysisResponse: Detailed game analysis

These schemas are used by the API endpoints and service layer.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.domains.gaming.models import (
    GameDifficulty,
    GameMode,
    GameStatus,
    GameType,
    OpponentSource,
)


# =============================================================================
# Common Response Models
# =============================================================================


class BoardDisplay(BaseModel):
    """Board display data for frontend rendering.

    Contains all information needed to render the game board.
    """

    fen: str | None = Field(
        default=None,
        description="FEN notation (chess only)",
    )
    grid: list[list[str | None]] | None = Field(
        default=None,
        description="Grid representation (connect4)",
    )
    pieces: list[dict[str, Any]] | None = Field(
        default=None,
        description="Piece positions for rendering",
    )
    turn: str = Field(description="Current turn (white/black, player1/player2)")
    legal_moves: list[str] | None = Field(
        default=None,
        description="Legal moves in current position",
    )
    in_check: bool = Field(
        default=False,
        description="Whether king is in check (chess)",
    )
    last_move: dict[str, str] | None = Field(
        default=None,
        description="Last move for highlighting",
    )


class MoveQualityInfo(BaseModel):
    """Move quality assessment."""

    quality: str | None = Field(
        default=None,
        description="Move quality (excellent, good, inaccuracy, mistake, blunder)",
    )
    is_best_move: bool = Field(
        default=False,
        description="Whether this was the best move",
    )
    evaluation_change: float | None = Field(
        default=None,
        description="Change in position evaluation",
    )


class SessionStats(BaseModel):
    """Game session statistics."""

    total_moves: int = Field(default=0, description="Total moves played")
    excellent_moves: int = Field(default=0, description="Excellent moves count")
    good_moves: int = Field(default=0, description="Good moves count")
    inaccuracies: int = Field(default=0, description="Inaccuracies count")
    mistakes: int = Field(default=0, description="Mistakes count")
    blunders: int = Field(default=0, description="Blunders count")
    hints_used: int = Field(default=0, description="Hints used")
    time_spent_seconds: int = Field(default=0, description="Time spent in seconds")

    @property
    def accuracy_percentage(self) -> float | None:
        """Calculate accuracy percentage."""
        if self.total_moves == 0:
            return None
        good = self.excellent_moves + self.good_moves
        bad = self.inaccuracies + self.mistakes + self.blunders
        total = good + bad
        if total == 0:
            return None
        return (good / total) * 100


class LearningPointInfo(BaseModel):
    """Learning point extracted from the game."""

    point: str = Field(description="The learning point")
    move_number: int | None = Field(
        default=None,
        description="Move number this relates to",
    )
    category: str | None = Field(
        default=None,
        description="Category (tactical, strategic, endgame, etc.)",
    )


# =============================================================================
# Start Game
# =============================================================================


class StartGameRequest(BaseModel):
    """Request to start a new game."""

    game_type: GameType = Field(description="Type of game to play")
    game_mode: GameMode = Field(
        default=GameMode.PRACTICE,
        description="Game mode",
    )
    difficulty: GameDifficulty = Field(
        default=GameDifficulty.MEDIUM,
        description="AI difficulty level",
    )
    player_color: str = Field(
        default="white",
        description="Player's color/side (white, black, player1, player2, random)",
    )
    initial_position: str | None = Field(
        default=None,
        description="Custom starting position (for puzzles)",
    )
    opponent_source: OpponentSource = Field(
        default=OpponentSource.ENGINE,
        description="Source of opponent moves (engine=Stockfish/Minimax, llm=AI-generated)",
    )


class StartGameResponse(BaseModel):
    """Response after starting a game."""

    session_id: str = Field(description="Game session ID")
    game_type: GameType = Field(description="Type of game")
    game_mode: GameMode = Field(description="Game mode")
    difficulty: GameDifficulty = Field(description="AI difficulty")
    player_color: str = Field(description="Player's color/side")
    display: BoardDisplay = Field(description="Board display data")
    your_turn: bool = Field(description="Whether it's player's turn")
    hints_available: int = Field(description="Number of hints available (-1 = unlimited)")
    coach_greeting: str = Field(description="Coach's greeting message")


# =============================================================================
# Make Move
# =============================================================================


class MakeMoveRequest(BaseModel):
    """Request to make a move."""

    move: str = Field(
        description="Move notation (e2e4 for chess, column number for connect4)",
    )
    time_spent_ms: int | None = Field(
        default=None,
        description="Time spent thinking in milliseconds",
    )


class AIMoveInfo(BaseModel):
    """Information about AI's move."""

    move: str = Field(description="AI's move notation")
    display: dict[str, str] | None = Field(
        default=None,
        description="Display info for animation (from_square, to_square)",
    )
    explanation: str | None = Field(
        default=None,
        description="AI's explanation of the move (tutorial mode only)",
    )


class MakeMoveResponse(BaseModel):
    """Response after making a move."""

    valid: bool = Field(description="Whether the move was valid")
    error_message: str | None = Field(
        default=None,
        description="Error message if move was invalid",
    )
    display: BoardDisplay = Field(description="Updated board display")
    move_quality: MoveQualityInfo | None = Field(
        default=None,
        description="Quality assessment of player's move",
    )
    ai_move: AIMoveInfo | None = Field(
        default=None,
        description="AI's response move (if game continues)",
    )
    coach_message: str = Field(description="Coach's message about the move")
    game_over: bool = Field(
        default=False,
        description="Whether the game ended",
    )
    game_result: str | None = Field(
        default=None,
        description="Game result if over (win, loss, draw)",
    )
    result_reason: str | None = Field(
        default=None,
        description="Reason for game end (checkmate, resignation, etc.)",
    )
    your_turn: bool = Field(description="Whether it's player's turn")
    move_number: int = Field(description="Current move number")
    hints_remaining: int = Field(description="Hints remaining (-1 = unlimited)")


# =============================================================================
# Get Hint
# =============================================================================


class GetHintRequest(BaseModel):
    """Request for a hint."""

    level: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Hint level (1=general, 2=specific, 3=reveals move)",
    )


class GetHintResponse(BaseModel):
    """Response with hint."""

    hint_text: str = Field(description="The hint message")
    hint_level: int = Field(description="Hint level provided")
    hint_type: str = Field(description="Type of hint (strategic, tactical, solution)")
    reveals_move: bool = Field(description="Whether this reveals the best move")
    suggested_squares: list[str] | None = Field(
        default=None,
        description="Squares to highlight on the board",
    )
    hints_remaining: int = Field(description="Hints remaining (-1 = unlimited)")


# =============================================================================
# Game Status
# =============================================================================


class GameStatusResponse(BaseModel):
    """Current game status."""

    session_id: str = Field(description="Game session ID")
    game_type: GameType = Field(description="Type of game")
    game_mode: GameMode = Field(description="Game mode")
    difficulty: GameDifficulty = Field(description="AI difficulty")
    status: GameStatus = Field(description="Session status")
    display: BoardDisplay = Field(description="Current board display")
    your_turn: bool = Field(description="Whether it's player's turn")
    move_count: int = Field(description="Total moves played")
    stats: SessionStats = Field(description="Session statistics")
    hints_remaining: int = Field(description="Hints remaining")
    time_elapsed_seconds: int = Field(description="Time elapsed in seconds")
    game_result: str | None = Field(
        default=None,
        description="Game result if completed",
    )
    result_reason: str | None = Field(
        default=None,
        description="Reason for game end",
    )


# =============================================================================
# End Game (Resign)
# =============================================================================


class ResignResponse(BaseModel):
    """Response after resigning."""

    session_id: str = Field(description="Game session ID")
    game_result: str = Field(description="Game result (loss)")
    result_reason: str = Field(default="resignation", description="Resignation")
    coach_message: str = Field(description="Coach's message about the game")
    stats: SessionStats = Field(description="Final session statistics")
    learning_points: list[LearningPointInfo] = Field(
        default_factory=list,
        description="Learning points from the game",
    )


# =============================================================================
# Game Analysis
# =============================================================================


class CriticalMoment(BaseModel):
    """A critical moment in the game."""

    move_number: int = Field(description="Move number")
    position_fen: str | None = Field(default=None, description="Position FEN")
    player_move: str = Field(description="Move played")
    best_move: str | None = Field(default=None, description="Best move")
    evaluation_loss: float | None = Field(
        default=None,
        description="Evaluation loss from this move",
    )
    explanation: str = Field(description="Explanation of why this was critical")
    category: str = Field(description="Category (blunder, missed_win, etc.)")


class PhaseAnalysis(BaseModel):
    """Analysis of a game phase (chess only)."""

    phase: str = Field(description="Phase name (opening, middlegame, endgame)")
    accuracy: float | None = Field(default=None, description="Accuracy in this phase")
    key_moments: list[str] = Field(
        default_factory=list,
        description="Key moments in this phase",
    )
    evaluation_trend: str | None = Field(
        default=None,
        description="How evaluation changed in this phase",
    )


class AnalyzeGameResponse(BaseModel):
    """Full game analysis response."""

    session_id: str = Field(description="Game session ID")
    game_type: GameType = Field(description="Type of game")
    game_result: str = Field(description="Game result")
    result_reason: str | None = Field(default=None, description="Reason for result")

    # Overall statistics
    overall_stats: SessionStats = Field(description="Overall statistics")
    accuracy_percentage: float | None = Field(
        default=None,
        description="Overall accuracy",
    )

    # Phase analysis (chess)
    phase_analysis: list[PhaseAnalysis] | None = Field(
        default=None,
        description="Analysis by game phase",
    )

    # Critical moments
    critical_moments: list[CriticalMoment] = Field(
        default_factory=list,
        description="Critical moments in the game",
    )

    # Learning
    coach_summary: str = Field(description="Coach's summary of the game")
    learning_points: list[LearningPointInfo] = Field(
        default_factory=list,
        description="Key learning points",
    )
    improvement_tips: list[str] = Field(
        default_factory=list,
        description="Tips for improvement",
    )
    strength_areas: list[str] = Field(
        default_factory=list,
        description="Areas where player performed well",
    )
    weakness_areas: list[str] = Field(
        default_factory=list,
        description="Areas needing improvement",
    )

    # Rating
    performance_rating: int | None = Field(
        default=None,
        description="Estimated performance rating",
    )


# =============================================================================
# List Games
# =============================================================================


class GameSessionSummary(BaseModel):
    """Summary of a game session for listing."""

    session_id: str = Field(description="Session ID")
    game_type: GameType = Field(description="Game type")
    game_mode: GameMode = Field(description="Game mode")
    difficulty: GameDifficulty = Field(description="Difficulty")
    status: GameStatus = Field(description="Status")
    result: str | None = Field(default=None, description="Result if completed")
    total_moves: int = Field(default=0, description="Total moves")
    started_at: datetime = Field(description="When started")
    ended_at: datetime | None = Field(default=None, description="When ended")


class ListGamesResponse(BaseModel):
    """Response listing game sessions."""

    sessions: list[GameSessionSummary] = Field(
        default_factory=list,
        description="List of sessions",
    )
    total: int = Field(description="Total count")
    has_more: bool = Field(description="Whether there are more results")


# =============================================================================
# Available Games (for Companion)
# =============================================================================


class AvailableGame(BaseModel):
    """Information about an available game."""

    game_type: GameType = Field(description="Game type")
    name: str = Field(description="Display name")
    description: str = Field(description="Game description")
    difficulty_levels: list[str] = Field(description="Available difficulty levels")
    modes: list[str] = Field(description="Available game modes")
    icon: str | None = Field(default=None, description="Icon name for UI")


class AvailableGamesResponse(BaseModel):
    """Response listing available games."""

    games: list[AvailableGame] = Field(description="Available games")
    active_session: GameSessionSummary | None = Field(
        default=None,
        description="Currently active game session if any",
    )
