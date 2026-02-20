# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data models for the gaming domain.

This module defines Pydantic models and enums for:
- Game types and modes
- Game state representation
- Move validation and analysis
- AI responses and hints

These models are used throughout the gaming system for
type-safe data transfer between components.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GameType(str, Enum):
    """Supported game types."""

    CHESS = "chess"
    CONNECT4 = "connect4"
    GOMOKU = "gomoku"
    OTHELLO = "othello"
    CHECKERS = "checkers"


class GameMode(str, Enum):
    """Game session modes.

    Each mode provides different educational experiences:
    - TUTORIAL: Step-by-step learning with heavy guidance
    - PRACTICE: Regular play with coaching feedback
    - CHALLENGE: Competitive play with minimal hints
    - PUZZLE: Solve specific positions (e.g., mate in 2)
    - ANALYSIS: Review and analyze completed games
    """

    TUTORIAL = "tutorial"
    PRACTICE = "practice"
    CHALLENGE = "challenge"
    PUZZLE = "puzzle"
    ANALYSIS = "analysis"


class GameDifficulty(str, Enum):
    """AI opponent difficulty levels.

    Maps to engine-specific configurations:
    - BEGINNER: Very easy, makes intentional mistakes
    - EASY: Light challenge, occasional mistakes
    - MEDIUM: Balanced play, appropriate for learning
    - HARD: Strong play, minimal mistakes
    - EXPERT: Maximum strength, tournament-level
    """

    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class OpponentSource(str, Enum):
    """Source of opponent moves.

    Determines who calculates the opponent's moves:
    - ENGINE: Game engine (Stockfish, Minimax) - optimal play at given difficulty
    - LLM: LLM generates moves - conversational, creative play (future)

    Note: Regardless of source, the AI Coach always explains moves as if
    IT made the decision ("I played this move because..."). This creates
    a consistent experience where students always feel they're playing
    against an intelligent AI coach.
    """

    ENGINE = "engine"  # Default: Stockfish/Minimax calculates moves
    LLM = "llm"  # Future: LLM generates moves conversationally


class GameStatus(str, Enum):
    """Game session status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MoveResult(str, Enum):
    """Result of a move attempt.

    Used to categorize moves for coaching feedback.
    """

    VALID = "valid"
    INVALID = "invalid"
    WINNING = "winning"
    LOSING = "losing"
    DRAW = "draw"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class Position(BaseModel):
    """Game position representation.

    A serializable representation of the current game state
    that can be stored in the database and passed to engines.

    Attributes:
        notation: Position in standard notation (FEN for chess, custom for others).
        board_state: 2D array representation for visualization.
        metadata: Additional position data (castling rights, en passant, etc.).
    """

    notation: str = Field(description="Position in standard notation")
    board_state: list[list[str | None]] = Field(
        default_factory=list,
        description="2D board representation for visualization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional position metadata",
    )


class Move(BaseModel):
    """A game move.

    Represents a single move in any supported game type.

    Attributes:
        notation: Move in standard notation (e.g., "e2e4", "Nf3", "col3").
        from_pos: Source position (if applicable).
        to_pos: Target position.
        piece: Piece being moved (if applicable).
        promotion: Promotion piece (chess pawn promotion).
        metadata: Additional move data.
    """

    notation: str = Field(description="Move in standard notation")
    from_pos: str | None = Field(default=None, description="Source position")
    to_pos: str | None = Field(default=None, description="Target position")
    piece: str | None = Field(default=None, description="Piece being moved")
    promotion: str | None = Field(default=None, description="Promotion piece")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional move metadata",
    )


class MoveValidation(BaseModel):
    """Result of move validation.

    Returned by game engines when validating a move.

    Attributes:
        is_valid: Whether the move is legal.
        result: Categorization of the move result.
        new_position: Position after the move (if valid).
        error_message: Explanation if move is invalid.
        is_game_over: Whether the game ended with this move.
        winner: Winner if game is over ("white", "black", "player1", "player2", None for draw).
    """

    is_valid: bool = Field(description="Whether the move is legal")
    result: MoveResult = Field(description="Move result category")
    new_position: Position | None = Field(
        default=None,
        description="Position after the move",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if move is invalid",
    )
    is_game_over: bool = Field(
        default=False,
        description="Whether the game ended",
    )
    winner: str | None = Field(
        default=None,
        description="Winner if game is over",
    )


class SuggestedMove(BaseModel):
    """A suggested move with evaluation and explanation.

    Used by both local engines and remote game-engine service.

    Attributes:
        move: Move in standard notation.
        evaluation: Numeric evaluation (from engine analysis).
        explanation: Human-readable explanation of the move.
    """

    move: str = Field(description="Move in standard notation")
    evaluation: float | None = Field(
        default=None,
        description="Numeric evaluation (positive = player advantage)",
    )
    explanation: str | None = Field(
        default=None,
        description="Human-readable explanation of the move",
    )


class PositionAnalysis(BaseModel):
    """Analysis of a game position.

    Provides evaluation and insights for coaching.

    Attributes:
        evaluation: Numeric evaluation (positive = player advantage).
        evaluation_text: Human-readable evaluation description.
        best_move: Best move in the position.
        best_move_explanation: Why this is the best move.
        threats: Current threats on the board.
        opportunities: Tactical opportunities available.
        strategic_themes: Strategic concepts present in the position.
        suggested_moves: Top moves with evaluations and explanations.
        mistakes_to_avoid: Moves that would be mistakes.
    """

    evaluation: float = Field(
        default=0.0,
        description="Numeric evaluation (positive = player advantage)",
    )
    evaluation_text: str = Field(
        default="",
        description="Human-readable evaluation",
    )
    best_move: str | None = Field(
        default=None,
        description="Best move in notation",
    )
    best_move_explanation: str | None = Field(
        default=None,
        description="Why this is the best move",
    )
    threats: list[str] = Field(
        default_factory=list,
        description="Current threats on the board",
    )
    opportunities: list[str] = Field(
        default_factory=list,
        description="Tactical opportunities",
    )
    strategic_themes: list[str] = Field(
        default_factory=list,
        description="Strategic concepts in the position",
    )
    suggested_moves: list[SuggestedMove] = Field(
        default_factory=list,
        description="Top moves with evaluations and explanations",
    )
    mistakes_to_avoid: list[dict[str, str]] = Field(
        default_factory=list,
        description="Moves that would be mistakes",
    )


class HintResponse(BaseModel):
    """Hint for the current position.

    Provides graduated hints based on hint level.

    Attributes:
        hint_level: Current hint level (1-3, higher = more specific).
        hint_text: The hint message.
        hint_type: Type of hint (tactical, strategic, general).
        reveals_move: Whether this hint reveals the best move.
    """

    hint_level: int = Field(ge=1, le=3, description="Hint specificity level")
    hint_text: str = Field(description="The hint message")
    hint_type: str = Field(
        default="general",
        description="Type of hint",
    )
    reveals_move: bool = Field(
        default=False,
        description="Whether this reveals the best move",
    )


class AIMove(BaseModel):
    """AI opponent's move response.

    Returned by game engines for the AI's turn.

    Attributes:
        move: The move in standard notation.
        thinking_time_ms: Time spent calculating.
        evaluation: Position evaluation after the move.
        move_quality: Quality category of the move.
        commentary: Optional commentary about the move.
    """

    move: str = Field(description="Move in standard notation")
    thinking_time_ms: int = Field(
        default=0,
        description="Time spent calculating in milliseconds",
    )
    evaluation: float = Field(
        default=0.0,
        description="Position evaluation after move",
    )
    move_quality: str = Field(
        default="normal",
        description="Quality category of the move",
    )
    commentary: str | None = Field(
        default=None,
        description="Optional commentary about the move",
    )


class GameState(BaseModel):
    """Complete game state.

    Represents the full state of a game session,
    including position, history, and metadata.

    Attributes:
        game_type: Type of game.
        position: Current position.
        move_history: List of moves played.
        current_player: Who's turn it is.
        status: Game status.
        result: Game result if completed.
        move_count: Total moves played.
        time_remaining: Time remaining for each player (if timed).
        metadata: Additional game data.
    """

    game_type: GameType = Field(description="Type of game")
    position: Position = Field(description="Current position")
    move_history: list[Move] = Field(
        default_factory=list,
        description="Moves played in order",
    )
    current_player: str = Field(
        default="white",
        description="Who's turn it is",
    )
    status: GameStatus = Field(
        default=GameStatus.ACTIVE,
        description="Game status",
    )
    result: str | None = Field(
        default=None,
        description="Game result if completed",
    )
    move_count: int = Field(
        default=0,
        description="Total moves played",
    )
    time_remaining: dict[str, int] | None = Field(
        default=None,
        description="Time remaining per player in seconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional game metadata",
    )

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage.

        Returns:
            Dictionary representation suitable for JSONB storage.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "GameState":
        """Create from database storage dictionary.

        Args:
            data: Dictionary from JSONB storage.

        Returns:
            GameState instance.
        """
        return cls.model_validate(data)
