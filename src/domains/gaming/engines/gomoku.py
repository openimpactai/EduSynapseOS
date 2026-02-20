# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gomoku (Five in a Row) engine implementation using Minimax algorithm.

This module provides a Gomoku engine that:
- Implements standard Gomoku rules (15x15 board, 5 in a row to win)
- Uses Minimax with alpha-beta pruning for AI moves
- Provides educational hints and analysis for coaching

The engine is stateless - all game state is passed via Position/GameState objects.
"""

import logging
import random
from typing import Any

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
)
from src.domains.gaming.engines.base import (
    GameEngine,
    InvalidPositionError,
)

logger = logging.getLogger(__name__)

# Board dimensions
BOARD_SIZE = 15
CONNECT = 5

# Player symbols
EMPTY = "."
BLACK = "X"  # Black plays first
WHITE = "O"

# Difficulty settings (search depth)
DIFFICULTY_DEPTHS = {
    GameDifficulty.BEGINNER: 1,
    GameDifficulty.EASY: 2,
    GameDifficulty.MEDIUM: 3,
    GameDifficulty.HARD: 4,
    GameDifficulty.EXPERT: 5,
}


class GomokuEngine(GameEngine):
    """Gomoku engine using Minimax algorithm.

    This engine provides full Gomoku functionality with
    Minimax-based AI for move generation and analysis.

    The board is represented as a 2D list where:
    - Row 0 is the top row
    - Row 14 is the bottom row
    - Columns are 0-14 (a-o)

    Notation: "h8" means column h (7), row 8 (7 from top)

    Example:
        engine = GomokuEngine()
        position = engine.get_initial_position()
        validation = engine.validate_move(position, Move(notation="h8"))
        if validation.is_valid:
            ai_response = engine.get_ai_move(validation.new_position, GameDifficulty.MEDIUM)
    """

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return GameType.GOMOKU

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Minimax Gomoku Engine"

    def get_initial_position(self) -> Position:
        """Get the starting Gomoku position (empty board)."""
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        return self._board_to_position(board, BLACK)

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a Gomoku move.

        Args:
            position: Current board position.
            move: Move to validate (e.g., "h8", "a1").

        Returns:
            MoveValidation with result and new position if valid.
        """
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.GOMOKU,
            ) from e

        # Parse move notation (e.g., "h8" -> col=7, row=7)
        try:
            col, row = self._parse_move(move.notation)
        except ValueError as e:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=str(e),
            )

        # Check bounds
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Position {move.notation} is out of bounds.",
            )

        # Check if cell is empty
        if board[row][col] != EMPTY:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Position {move.notation} is already occupied.",
            )

        # Make the move
        new_board = [row_list[:] for row_list in board]
        new_board[row][col] = current_player

        # Check for win
        is_winner = self._check_win(new_board, row, col, current_player)
        is_draw = self._check_draw(new_board)

        # Determine result
        if is_winner:
            result = MoveResult.WINNING
            is_game_over = True
            winner = "black" if current_player == BLACK else "white"
        elif is_draw:
            result = MoveResult.DRAW
            is_game_over = True
            winner = None
        else:
            result = MoveResult.VALID
            is_game_over = False
            winner = None

        # Next player
        next_player = WHITE if current_player == BLACK else BLACK
        new_position = self._board_to_position(new_board, next_player)

        return MoveValidation(
            is_valid=True,
            result=result,
            new_position=new_position,
            is_game_over=is_game_over,
            winner=winner,
        )

    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves (empty cells near existing stones)."""
        try:
            board, _ = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.GOMOKU,
            ) from e

        moves = []
        # For efficiency, only consider moves near existing stones
        candidates = self._get_candidate_moves(board)

        for row, col in candidates:
            if board[row][col] == EMPTY:
                notation = self._to_notation(row, col)
                moves.append(Move(
                    notation=notation,
                    to_pos=notation,
                ))

        return moves

    def get_ai_move(
        self,
        position: Position,
        difficulty: GameDifficulty,
        time_limit_ms: int = 1000,
    ) -> AIMove:
        """Get AI's move using Minimax algorithm."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.GOMOKU,
            ) from e

        depth = DIFFICULTY_DEPTHS[difficulty]

        # For beginner, sometimes make random moves
        if difficulty == GameDifficulty.BEGINNER and random.random() < 0.4:
            candidates = self._get_candidate_moves(board)
            empty_candidates = [(r, c) for r, c in candidates if board[r][c] == EMPTY]
            if empty_candidates:
                row, col = random.choice(empty_candidates)
                return AIMove(
                    move=self._to_notation(row, col),
                    thinking_time_ms=50,
                    evaluation=0.0,
                    move_quality="normal",
                )

        # Use Minimax
        best_move, best_score = self._minimax_root(board, depth, current_player)

        if best_move is None:
            # Fallback: first empty cell near center
            candidates = self._get_candidate_moves(board)
            for row, col in candidates:
                if board[row][col] == EMPTY:
                    best_move = (row, col)
                    break

        if best_move is None:
            best_move = (BOARD_SIZE // 2, BOARD_SIZE // 2)

        row, col = best_move
        notation = self._to_notation(row, col)

        # Determine move quality
        move_quality = "normal"
        if abs(best_score) > 100000:
            move_quality = "winning" if best_score > 0 else "losing"

        return AIMove(
            move=notation,
            thinking_time_ms=100,
            evaluation=best_score / 1000,
            move_quality=move_quality,
        )

    def analyze_position(
        self,
        position: Position,
        depth: int = 4,
    ) -> PositionAnalysis:
        """Analyze the current position."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.GOMOKU,
            ) from e

        # Get best move
        best_move, best_score = self._minimax_root(board, min(depth, 3), current_player)

        evaluation = best_score / 1000 if best_score else 0.0

        if abs(best_score) > 100000:
            evaluation_text = "Winning position!" if best_score > 0 else "Losing position"
        elif abs(best_score) > 1000:
            evaluation_text = "Good advantage" if best_score > 0 else "Slight disadvantage"
        else:
            evaluation_text = "Even position"

        # Find threats
        threats = self._find_threats(board, current_player)
        opportunities = self._find_opportunities(board, current_player)

        best_notation = self._to_notation(*best_move) if best_move else None

        return PositionAnalysis(
            evaluation=evaluation,
            evaluation_text=evaluation_text,
            best_move=best_notation,
            best_move_explanation="This move gives the best position.",
            threats=threats,
            opportunities=opportunities,
        )

    def get_hint(
        self,
        position: Position,
        hint_level: int,
        previous_hints: list[str] | None = None,
    ) -> HintResponse:
        """Get a hint for the current position."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.GOMOKU,
            ) from e

        analysis = self.analyze_position(position, depth=3)

        if hint_level == 1:
            if analysis.threats:
                hint_text = "Watch out! Your opponent might have a winning threat."
            elif analysis.opportunities:
                hint_text = "Look for ways to build your line toward 5."
            else:
                hint_text = "Try to build lines while blocking your opponent."

            return HintResponse(
                hint_level=1,
                hint_text=hint_text,
                hint_type="strategic",
                reveals_move=False,
            )

        elif hint_level == 2:
            if analysis.threats:
                hint_text = f"Threat: {analysis.threats[0]}"
            elif analysis.opportunities:
                hint_text = f"Opportunity: {analysis.opportunities[0]}"
            elif analysis.best_move:
                # Give area hint
                col, row = self._parse_move(analysis.best_move)
                if row < 5:
                    area = "top"
                elif row > 9:
                    area = "bottom"
                else:
                    area = "center"
                hint_text = f"Look at the {area} area of the board."
            else:
                hint_text = "Build multiple threats at once."

            return HintResponse(
                hint_level=2,
                hint_text=hint_text,
                hint_type="tactical",
                reveals_move=False,
            )

        else:  # Level 3
            if analysis.best_move:
                hint_text = f"Play {analysis.best_move}. {analysis.best_move_explanation or ''}"
            else:
                hint_text = "Any move near existing stones is fine."

            return HintResponse(
                hint_level=3,
                hint_text=hint_text,
                hint_type="solution",
                reveals_move=True,
            )

    def is_game_over(self, position: Position) -> tuple[bool, str | None, str | None]:
        """Check if the game is over."""
        try:
            board, _ = self._position_to_board(position)
        except Exception:
            return False, None, None

        # Check for wins
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece != EMPTY:
                    if self._check_win(board, row, col, piece):
                        winner = "black" if piece == BLACK else "white"
                        return True, "five_in_row", winner

        # Check for draw
        if self._check_draw(board):
            return True, "draw", None

        return False, None, None

    def position_to_display(self, position: Position) -> dict[str, Any]:
        """Convert position to display format."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception:
            return {"error": "Invalid position"}

        display_grid = []
        for row in range(BOARD_SIZE):
            display_row = []
            for col in range(BOARD_SIZE):
                cell = board[row][col]
                if cell == BLACK:
                    display_row.append("black")
                elif cell == WHITE:
                    display_row.append("white")
                else:
                    display_row.append(None)
            display_grid.append(display_row)

        return {
            "grid": display_grid,
            "size": BOARD_SIZE,
            "current_player": "black" if current_player == BLACK else "white",
        }

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _board_to_position(
        self,
        board: list[list[str]],
        current_player: str,
    ) -> Position:
        """Convert board array to Position."""
        rows_str = []
        for row in board:
            rows_str.append("".join(row))
        notation = "/".join(rows_str) + f" {current_player}"

        return Position(
            notation=notation,
            board_state=board,
            metadata={
                "current_player": current_player,
                "size": BOARD_SIZE,
            },
        )

    def _position_to_board(self, position: Position) -> tuple[list[list[str]], str]:
        """Convert Position to board array and current player."""
        parts = position.notation.split(" ")
        rows_str = parts[0].split("/")
        current_player = parts[1] if len(parts) > 1 else BLACK

        board = []
        for row_str in rows_str:
            row = list(row_str)
            board.append(row)

        return board, current_player

    def _parse_move(self, notation: str) -> tuple[int, int]:
        """Parse move notation (e.g., 'h8') to (col, row)."""
        notation = notation.lower().strip()
        if len(notation) < 2:
            raise ValueError(f"Invalid notation: {notation}")

        col_char = notation[0]
        row_str = notation[1:]

        if not col_char.isalpha():
            raise ValueError(f"Invalid column: {col_char}")

        col = ord(col_char) - ord('a')
        if col < 0 or col >= BOARD_SIZE:
            raise ValueError(f"Column out of range: {col_char}")

        try:
            row = int(row_str) - 1  # 1-indexed to 0-indexed
        except ValueError:
            raise ValueError(f"Invalid row: {row_str}")

        if row < 0 or row >= BOARD_SIZE:
            raise ValueError(f"Row out of range: {row_str}")

        return col, row

    def _to_notation(self, row: int, col: int) -> str:
        """Convert (row, col) to notation (e.g., 'h8')."""
        col_char = chr(ord('a') + col)
        row_num = row + 1
        return f"{col_char}{row_num}"

    def _get_candidate_moves(self, board: list[list[str]]) -> list[tuple[int, int]]:
        """Get candidate moves (empty cells near existing stones)."""
        candidates = set()
        has_stones = False

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != EMPTY:
                    has_stones = True
                    # Add all empty cells within 2 squares
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if board[nr][nc] == EMPTY:
                                    candidates.add((nr, nc))

        if not has_stones:
            # First move: center area
            center = BOARD_SIZE // 2
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    candidates.add((center + dr, center + dc))

        # Sort by distance from center (prefer center moves)
        center = BOARD_SIZE // 2
        result = sorted(candidates, key=lambda p: abs(p[0] - center) + abs(p[1] - center))
        return result

    def _check_win(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        player: str,
    ) -> bool:
        """Check if the player has won after placing at (row, col)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= CONNECT:
                return True

        return False

    def _check_draw(self, board: list[list[str]]) -> bool:
        """Check if the game is a draw (board is full)."""
        for row in board:
            for cell in row:
                if cell == EMPTY:
                    return False
        return True

    def _evaluate_board(self, board: list[list[str]], player: str) -> int:
        """Evaluate the board position for the given player."""
        opponent = WHITE if player == BLACK else BLACK
        score = 0

        # Check all lines of 5
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # Horizontal
                if col <= BOARD_SIZE - CONNECT:
                    window = [board[row][col + i] for i in range(CONNECT)]
                    score += self._evaluate_window(window, player, opponent)

                # Vertical
                if row <= BOARD_SIZE - CONNECT:
                    window = [board[row + i][col] for i in range(CONNECT)]
                    score += self._evaluate_window(window, player, opponent)

                # Diagonal down-right
                if row <= BOARD_SIZE - CONNECT and col <= BOARD_SIZE - CONNECT:
                    window = [board[row + i][col + i] for i in range(CONNECT)]
                    score += self._evaluate_window(window, player, opponent)

                # Diagonal down-left
                if row <= BOARD_SIZE - CONNECT and col >= CONNECT - 1:
                    window = [board[row + i][col - i] for i in range(CONNECT)]
                    score += self._evaluate_window(window, player, opponent)

        # Center preference
        center = BOARD_SIZE // 2
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == player:
                    distance = abs(row - center) + abs(col - center)
                    score += max(0, 10 - distance)

        return score

    def _evaluate_window(
        self,
        window: list[str],
        player: str,
        opponent: str,
    ) -> int:
        """Evaluate a window of 5 cells."""
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(EMPTY)

        # Can't score if both players in window
        if player_count > 0 and opponent_count > 0:
            return 0

        if player_count == 5:
            return 1000000
        elif player_count == 4 and empty_count == 1:
            return 10000
        elif player_count == 3 and empty_count == 2:
            return 500
        elif player_count == 2 and empty_count == 3:
            return 50

        if opponent_count == 5:
            return -1000000
        elif opponent_count == 4 and empty_count == 1:
            return -50000  # Must block!
        elif opponent_count == 3 and empty_count == 2:
            return -1000

        return 0

    def _minimax_root(
        self,
        board: list[list[str]],
        depth: int,
        player: str,
    ) -> tuple[tuple[int, int] | None, int]:
        """Minimax root with alpha-beta pruning."""
        best_move = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        candidates = self._get_candidate_moves(board)

        for row, col in candidates:
            if board[row][col] != EMPTY:
                continue

            new_board = [r[:] for r in board]
            new_board[row][col] = player

            opponent = WHITE if player == BLACK else BLACK
            score = -self._minimax(new_board, depth - 1, opponent, -beta, -alpha, player)

            if score > best_score:
                best_score = score
                best_move = (row, col)

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_move, int(best_score)

    def _minimax(
        self,
        board: list[list[str]],
        depth: int,
        current_player: str,
        alpha: float,
        beta: float,
        maximizing_player: str,
    ) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        # Check terminal states
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece != EMPTY and self._check_win(board, row, col, piece):
                    if piece == maximizing_player:
                        return 1000000 + depth
                    else:
                        return -1000000 - depth

        if depth == 0:
            return self._evaluate_board(board, maximizing_player)

        candidates = self._get_candidate_moves(board)
        if not candidates:
            return 0  # Draw

        best_score = float("-inf")
        for row, col in candidates:
            if board[row][col] != EMPTY:
                continue

            new_board = [r[:] for r in board]
            new_board[row][col] = current_player

            opponent = WHITE if current_player == BLACK else BLACK
            score = -self._minimax(
                new_board, depth - 1, opponent, -beta, -alpha, maximizing_player
            )

            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score

    def _find_threats(self, board: list[list[str]], player: str) -> list[str]:
        """Find threats in the position."""
        threats = []
        opponent = WHITE if player == BLACK else BLACK

        candidates = self._get_candidate_moves(board)
        for row, col in candidates:
            if board[row][col] != EMPTY:
                continue

            test_board = [r[:] for r in board]
            test_board[row][col] = opponent
            if self._check_win(test_board, row, col, opponent):
                threats.append(f"Opponent can win at {self._to_notation(row, col)}!")

        return threats[:3]

    def _find_opportunities(self, board: list[list[str]], player: str) -> list[str]:
        """Find opportunities in the position."""
        opportunities = []

        candidates = self._get_candidate_moves(board)
        for row, col in candidates:
            if board[row][col] != EMPTY:
                continue

            test_board = [r[:] for r in board]
            test_board[row][col] = player
            if self._check_win(test_board, row, col, player):
                opportunities.append(f"You can win at {self._to_notation(row, col)}!")

        return opportunities[:3]
