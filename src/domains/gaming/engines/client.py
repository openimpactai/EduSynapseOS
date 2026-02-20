# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Engine HTTP Client for external game engine microservice.

This client communicates with the edusynapse-game-engine container via HTTP,
providing access to professional-level game engines:
- Chess (Stockfish)
- Gomoku (Rapfi)
- Othello (Edax)
- Checkers (Raven)
- Connect4 (Built-in Minimax)

The client follows the same interface as local engines but delegates
to the external service for move calculation and analysis.
"""

import logging
from typing import Any

import chess  # For parsing chess FEN to get check/castling info
import httpx

from src.core.config.settings import get_settings
from src.domains.gaming.models import (
    AIMove,
    GameDifficulty,
    GameType,
    HintResponse,
    Move,
    MoveResult,
    MoveValidation,
    Position,
    PositionAnalysis,
    SuggestedMove,
)
from src.domains.gaming.engines.base import (
    EngineError,
    EngineNotAvailableError,
    GameEngine,
    InvalidMoveError,
    InvalidPositionError,
)

logger = logging.getLogger(__name__)


# Difficulty to 1-10 mapping
DIFFICULTY_LEVEL_MAP = {
    GameDifficulty.BEGINNER: 1,
    GameDifficulty.EASY: 3,
    GameDifficulty.MEDIUM: 5,
    GameDifficulty.HARD: 7,
    GameDifficulty.EXPERT: 10,
}


class GameEngineClient(GameEngine):
    """HTTP client for the Game Engine microservice.

    This client communicates with the external game engine container
    via REST API, providing access to professional-level game engines.

    Attributes:
        game_type: The type of game this client handles.

    Example:
        client = GameEngineClient(GameType.CHESS)
        position = client.get_initial_position()
        ai_move = client.get_ai_move(position, GameDifficulty.MEDIUM)
    """

    def __init__(self, game_type: GameType) -> None:
        """Initialize the game engine client.

        Args:
            game_type: The type of game to handle.
        """
        self._game_type = game_type
        self._settings = get_settings().game_engine
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return self._game_type

    @property
    def name(self) -> str:
        """Get the engine name."""
        return f"Remote {self._game_type.value.title()} Engine"

    @property
    def base_url(self) -> str:
        """Get the base URL for the game engine service."""
        return self._settings.url

    @property
    def api_url(self) -> str:
        """Get the API URL for the game type."""
        return f"{self.base_url}/api/v1/games/{self._game_type.value}"

    def _get_sync_client(self) -> httpx.Client:
        """Get or create a synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self._settings.timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._sync_client

    def _handle_response(self, response: httpx.Response, operation: str) -> dict:
        """Handle HTTP response and raise appropriate errors."""
        if response.status_code == 200:
            return response.json()

        error_detail = "Unknown error"
        try:
            error_data = response.json()
            error_detail = error_data.get("detail", str(error_data))
        except Exception:
            error_detail = response.text

        if response.status_code == 400:
            raise InvalidMoveError(
                message=error_detail,
                game_type=self._game_type,
            )
        elif response.status_code == 404:
            raise EngineNotAvailableError(
                message=f"Engine not available for {self._game_type.value}",
                game_type=self._game_type,
            )
        else:
            raise EngineError(
                message=f"{operation} failed: {error_detail}",
                game_type=self._game_type,
            )

    def get_initial_position(self) -> Position:
        """Get the initial position for the game."""
        try:
            client = self._get_sync_client()
            response = client.get(f"{self.api_url}/init")
            data = self._handle_response(response, "Get initial position")

            display_info = data.get("display", {})
            board_state = display_info.get("grid") if display_info else None

            return Position(
                notation=data["position"],
                board_state=board_state or [],
                metadata={
                    "current_player": data.get("current_player"),
                    "display": display_info,
                },
            )
        except httpx.RequestError as e:
            logger.error("Connection error to game engine: %s", e)
            raise EngineNotAvailableError(
                message=f"Game engine service not available: {e}",
                game_type=self._game_type,
            ) from e

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a move via the game engine service."""
        try:
            client = self._get_sync_client()
            response = client.post(
                f"{self.api_url}/validate",
                json={
                    "position": position.notation,
                    "move": move.notation,
                    "player": position.metadata.get("current_player", "white")
                    if position.metadata else "white",
                },
            )
            data = self._handle_response(response, "Validate move")

            if not data["is_valid"]:
                return MoveValidation(
                    is_valid=False,
                    result=MoveResult.INVALID,
                    error_message=data.get("error_message", "Invalid move"),
                )

            # Determine move result
            result = MoveResult.VALID
            if data.get("is_game_over"):
                if data.get("winner"):
                    result = MoveResult.CHECKMATE
                else:
                    result = MoveResult.DRAW

            # Build new position
            new_position = None
            if data.get("new_position"):
                display_info = data.get("display", {})
                board_state = display_info.get("grid") if display_info else None

                new_position = Position(
                    notation=data["new_position"],
                    board_state=board_state or [],
                    metadata={
                        "move_quality": data.get("move_quality"),
                        "current_player": self._get_next_player(position),
                        "display": display_info,
                    },
                )

            return MoveValidation(
                is_valid=True,
                result=result,
                new_position=new_position,
                is_game_over=data.get("is_game_over", False),
                winner=data.get("winner"),
            )

        except httpx.RequestError as e:
            logger.error("Connection error validating move: %s", e)
            raise EngineNotAvailableError(
                message=f"Game engine service not available: {e}",
                game_type=self._game_type,
            ) from e

    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves in the position."""
        try:
            client = self._get_sync_client()
            response = client.get(
                f"{self.api_url}/legal",
                params={"position": position.notation},
            )
            data = self._handle_response(response, "Get legal moves")

            moves = []
            for move_str in data.get("moves", []):
                moves.append(Move(notation=move_str))

            return moves

        except httpx.RequestError as e:
            logger.error("Connection error getting legal moves: %s", e)
            raise EngineNotAvailableError(
                message=f"Game engine service not available: {e}",
                game_type=self._game_type,
            ) from e

    def get_ai_move(
        self,
        position: Position,
        difficulty: GameDifficulty,
        time_limit_ms: int = 1000,
    ) -> AIMove:
        """Get AI's move from the game engine service."""
        try:
            client = self._get_sync_client()
            response = client.post(
                f"{self.api_url}/move",
                json={
                    "position": position.notation,
                    "difficulty": DIFFICULTY_LEVEL_MAP.get(difficulty, 5),
                    "time_limit_ms": time_limit_ms,
                    "player_color": self._get_ai_color(position),
                },
            )
            data = self._handle_response(response, "Get AI move")

            if not data.get("success", True):
                raise EngineError(
                    message=data.get("error", "Failed to get AI move"),
                    game_type=self._game_type,
                )

            return AIMove(
                move=data["move"],
                thinking_time_ms=data.get("thinking_time_ms", 0),
                evaluation=data.get("evaluation", 0.0),
                move_quality=self._quality_from_score(data.get("evaluation", 0)),
                commentary=data.get("move_reason"),
            )

        except httpx.RequestError as e:
            logger.error("Connection error getting AI move: %s", e)
            raise EngineNotAvailableError(
                message=f"Game engine service not available: {e}",
                game_type=self._game_type,
            ) from e

    def analyze_position(
        self,
        position: Position,
        depth: int = 10,
    ) -> PositionAnalysis:
        """Analyze the position via the game engine service."""
        try:
            client = self._get_sync_client()
            response = client.post(
                f"{self.api_url}/analyze",
                json={
                    "position": position.notation,
                    "depth": depth,
                },
            )
            data = self._handle_response(response, "Analyze position")

            # Convert top_moves from game-engine to SuggestedMove models
            suggested_moves = []
            for move_data in data.get("top_moves", []):
                suggested_moves.append(SuggestedMove(
                    move=move_data.get("move", ""),
                    evaluation=move_data.get("evaluation"),
                    explanation=move_data.get("reason"),
                ))

            return PositionAnalysis(
                evaluation=data.get("evaluation", 0.0),
                evaluation_text=data.get("evaluation_text", "Unknown"),
                best_move=data.get("best_move"),
                best_move_explanation=data.get("best_move_reason"),
                suggested_moves=suggested_moves,
                threats=data.get("threats", []),
                opportunities=data.get("opportunities", []),
            )

        except httpx.RequestError as e:
            logger.error("Connection error analyzing position: %s", e)
            raise EngineNotAvailableError(
                message=f"Game engine service not available: {e}",
                game_type=self._game_type,
            ) from e

    def get_hint(
        self,
        position: Position,
        hint_level: int,
        previous_hints: list[str] | None = None,
    ) -> HintResponse:
        """Get a hint for the position.

        Uses position analysis to generate hints at different levels.
        """
        analysis = self.analyze_position(position, depth=12)

        if hint_level == 1:
            # General strategic hint
            if analysis.threats:
                hint_text = f"Watch out for threats: {analysis.threats[0]}"
            elif analysis.opportunities:
                hint_text = f"Consider: {analysis.opportunities[0]}"
            else:
                hint_text = "Look for ways to improve your position."

            return HintResponse(
                hint_level=1,
                hint_text=hint_text,
                hint_type="strategic",
                reveals_move=False,
            )

        elif hint_level == 2:
            # More specific hint
            if analysis.best_move:
                hint_text = f"There's a strong move available. Think about your options."
            else:
                hint_text = "Consider the position carefully."

            return HintResponse(
                hint_level=2,
                hint_text=hint_text,
                hint_type="tactical",
                reveals_move=False,
            )

        else:  # Level 3
            # Reveal the best move
            if analysis.best_move:
                explanation = analysis.best_move_explanation or "This is the best move."
                hint_text = f"The best move is {analysis.best_move}. {explanation}"
            else:
                hint_text = "No clear best move found."

            return HintResponse(
                hint_level=3,
                hint_text=hint_text,
                hint_type="solution",
                reveals_move=True,
            )

    def is_game_over(self, position: Position) -> tuple[bool, str | None, str | None]:
        """Check if the game is over.

        Uses validation with a dummy move to check game state.
        """
        # Get legal moves to check if game is over
        try:
            moves = self.get_legal_moves(position)
            if not moves:
                # No legal moves - game is over
                return True, "no_moves", None
            return False, None, None
        except Exception:
            return False, None, None

    def position_to_display(self, position: Position) -> dict[str, Any]:
        """Convert position to display format.

        Returns display info from game-engine when available,
        otherwise builds a fallback dict from position data.
        """
        if position.metadata and position.metadata.get("display"):
            display = position.metadata["display"]
            return {
                "fen": display.get("fen"),
                "grid": display.get("grid"),
                "pieces": display.get("pieces"),
                "turn": display.get("turn", "white"),
                "in_check": display.get("in_check", False),
                "castling_rights": display.get("castling_rights"),
                "legal_moves": display.get("legal_moves"),
                "last_move": display.get("last_move"),
            }

        turn = "white"
        if position.metadata and position.metadata.get("current_player"):
            turn = position.metadata["current_player"]
        if turn == "w":
            turn = "white"
        elif turn == "b":
            turn = "black"

        return {
            "fen": position.notation if self._game_type == GameType.CHESS else None,
            "grid": position.board_state,
            "pieces": None,
            "turn": turn,
            "in_check": False,
            "legal_moves": None,
            "last_move": None,
        }

    def is_available(self) -> bool:
        """Check if the game engine service is available."""
        try:
            client = self._get_sync_client()
            response = client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def get_engine_info(self) -> dict[str, Any]:
        """Get information about the engine from the service."""
        try:
            client = self._get_sync_client()
            response = client.get(f"{self.base_url}/api/v1/engines/{self._game_type.value}")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"game_type": self._game_type.value, "available": False}

    def _get_next_player(self, position: Position) -> str:
        """Determine the next player based on current position."""
        current = "white"
        if position.metadata and position.metadata.get("current_player"):
            current = position.metadata["current_player"]

        # Simple toggle
        return "black" if current == "white" else "white"

    def _get_ai_color(self, position: Position) -> str:
        """Get the AI's color (opposite of human player's turn)."""
        current = "white"
        if position.metadata and position.metadata.get("current_player"):
            current = position.metadata["current_player"]
        return current

    def _quality_from_score(self, score: float) -> str:
        """Convert evaluation score to move quality string."""
        abs_score = abs(score)
        if abs_score > 5:
            return "excellent"
        elif abs_score > 2:
            return "good"
        elif abs_score > 0.5:
            return "normal"
        else:
            return "normal"

    def __del__(self):
        """Cleanup HTTP clients."""
        if self._sync_client:
            try:
                self._sync_client.close()
            except Exception:
                pass


def get_game_engine_client(game_type: GameType) -> GameEngineClient:
    """Factory function to get a game engine client.

    Args:
        game_type: The type of game.

    Returns:
        A GameEngineClient configured for the game type.
    """
    return GameEngineClient(game_type)
