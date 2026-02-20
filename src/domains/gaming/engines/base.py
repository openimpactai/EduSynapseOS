# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Abstract base class for game engines.

This module defines the GameEngine ABC that all game engines must implement.
The interface provides a consistent API for:
- Position initialization and management
- Move validation and execution
- AI move generation
- Position analysis
- Hint generation

Each game engine implementation (Chess, Connect4, etc.) inherits from this
base and provides game-specific logic while maintaining interface consistency.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.domains.gaming.models import (
    AIMove,
    GameDifficulty,
    GameState,
    GameType,
    HintResponse,
    Move,
    MoveValidation,
    Position,
    PositionAnalysis,
)

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Base exception for game engine errors.

    Attributes:
        message: Error description.
        game_type: Type of game that raised the error.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        game_type: GameType | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.game_type = game_type
        self.details = details or {}
        super().__init__(self.message)


class EngineNotAvailableError(EngineError):
    """Raised when an engine is not available (e.g., Stockfish not installed)."""

    pass


class InvalidPositionError(EngineError):
    """Raised when a position is invalid or cannot be parsed."""

    pass


class InvalidMoveError(EngineError):
    """Raised when a move is invalid."""

    pass


class GameEngine(ABC):
    """Abstract base class for all game engines.

    Game engines provide the core game logic for different game types.
    They handle move validation, AI move generation, and position analysis.

    The engine is stateless - all state is passed via Position/GameState objects.
    This allows engines to be shared across multiple game sessions.

    Attributes:
        game_type: The type of game this engine handles.
        name: Human-readable engine name.

    Example:
        class ChessEngine(GameEngine):
            @property
            def game_type(self) -> GameType:
                return GameType.CHESS

            def get_initial_position(self) -> Position:
                return Position(
                    notation="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    board_state=self._fen_to_board(STARTING_FEN),
                )

            # ... implement other abstract methods
    """

    @property
    @abstractmethod
    def game_type(self) -> GameType:
        """Get the game type this engine handles.

        Returns:
            GameType enum value.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the human-readable engine name.

        Returns:
            Engine name (e.g., "Stockfish Chess Engine").
        """
        pass

    @abstractmethod
    def get_initial_position(self) -> Position:
        """Get the starting position for a new game.

        Returns:
            Position object representing the initial game state.
        """
        pass

    @abstractmethod
    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a move and return the result.

        Checks if the move is legal in the current position.
        If valid, returns the new position after the move.

        Args:
            position: Current game position.
            move: Move to validate.

        Returns:
            MoveValidation with result and new position if valid.

        Raises:
            InvalidPositionError: If the position is invalid.
        """
        pass

    @abstractmethod
    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves in the current position.

        Args:
            position: Current game position.

        Returns:
            List of all legal Move objects.

        Raises:
            InvalidPositionError: If the position is invalid.
        """
        pass

    @abstractmethod
    def get_ai_move(
        self,
        position: Position,
        difficulty: GameDifficulty,
        time_limit_ms: int = 1000,
    ) -> AIMove:
        """Get the AI's move for the current position.

        Args:
            position: Current game position.
            difficulty: AI difficulty level.
            time_limit_ms: Maximum time for move calculation.

        Returns:
            AIMove with the chosen move and metadata.

        Raises:
            InvalidPositionError: If the position is invalid.
            EngineError: If move generation fails.
        """
        pass

    @abstractmethod
    def analyze_position(
        self,
        position: Position,
        depth: int = 10,
    ) -> PositionAnalysis:
        """Analyze the current position.

        Provides evaluation and insights for coaching purposes.

        Args:
            position: Current game position.
            depth: Analysis depth (engine-specific).

        Returns:
            PositionAnalysis with evaluation and insights.

        Raises:
            InvalidPositionError: If the position is invalid.
        """
        pass

    @abstractmethod
    def get_hint(
        self,
        position: Position,
        hint_level: int,
        previous_hints: list[str] | None = None,
    ) -> HintResponse:
        """Get a hint for the current position.

        Hints are graduated:
        - Level 1: General strategic hint
        - Level 2: More specific tactical hint
        - Level 3: Reveals the best move

        Args:
            position: Current game position.
            hint_level: Hint specificity level (1-3).
            previous_hints: Previously given hints to avoid repetition.

        Returns:
            HintResponse with the hint text.

        Raises:
            InvalidPositionError: If the position is invalid.
        """
        pass

    @abstractmethod
    def is_game_over(self, position: Position) -> tuple[bool, str | None, str | None]:
        """Check if the game is over.

        Args:
            position: Current game position.

        Returns:
            Tuple of (is_over, result_type, winner).
            result_type: "checkmate", "stalemate", "draw", "resignation", etc.
            winner: "white", "black", "player1", "player2", or None for draw.
        """
        pass

    @abstractmethod
    def position_to_display(self, position: Position) -> dict[str, Any]:
        """Convert position to display format for frontend.

        Args:
            position: Current game position.

        Returns:
            Dictionary with display data (board, captured pieces, etc.).
        """
        pass

    def create_game_state(
        self,
        player_color: str = "white",
        time_control: dict[str, int] | None = None,
    ) -> GameState:
        """Create a new game state with initial position.

        Convenience method for starting a new game.

        Args:
            player_color: Color the player will play ("white" or "black").
            time_control: Optional time control settings.

        Returns:
            GameState with initial position.
        """
        position = self.get_initial_position()
        return GameState(
            game_type=self.game_type,
            position=position,
            move_history=[],
            current_player="white",  # White always moves first in chess-like games
            time_remaining=time_control,
            metadata={
                "player_color": player_color,
            },
        )

    def apply_move(
        self,
        state: GameState,
        move: Move,
    ) -> tuple[GameState, MoveValidation]:
        """Apply a move to a game state.

        Convenience method that validates and applies a move,
        returning the updated game state.

        Args:
            state: Current game state.
            move: Move to apply.

        Returns:
            Tuple of (new_state, validation_result).
        """
        validation = self.validate_move(state.position, move)

        if not validation.is_valid or validation.new_position is None:
            return state, validation

        # Create new state with updated position
        new_state = GameState(
            game_type=state.game_type,
            position=validation.new_position,
            move_history=state.move_history + [move],
            current_player=self._next_player(state.current_player),
            status=state.status,
            move_count=state.move_count + 1,
            time_remaining=state.time_remaining,
            metadata=state.metadata.copy(),
        )

        # Check if game is over
        if validation.is_game_over:
            new_state.status = "completed"
            new_state.result = validation.winner or "draw"

        return new_state, validation

    def _next_player(self, current: str) -> str:
        """Get the next player to move.

        Args:
            current: Current player.

        Returns:
            Next player identifier.
        """
        player_cycle = {
            "white": "black",
            "black": "white",
            "player1": "player2",
            "player2": "player1",
        }
        return player_cycle.get(current, "player1")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(game_type={self.game_type.value})"
