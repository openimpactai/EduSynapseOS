# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Connect 4 engine implementation using Minimax algorithm.

This module provides a Connect 4 engine that:
- Implements standard Connect 4 rules (7 columns, 6 rows)
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
    SuggestedMove,
)
from src.domains.gaming.engines.base import (
    EngineError,
    GameEngine,
    InvalidMoveError,
    InvalidPositionError,
)

logger = logging.getLogger(__name__)

# Board dimensions
ROWS = 6
COLS = 7
CONNECT = 4

# Player symbols
EMPTY = "."
PLAYER1 = "X"  # Red (human player typically)
PLAYER2 = "O"  # Yellow (AI typically)

# Difficulty settings (search depth)
DIFFICULTY_DEPTHS = {
    GameDifficulty.BEGINNER: 1,
    GameDifficulty.EASY: 2,
    GameDifficulty.MEDIUM: 4,
    GameDifficulty.HARD: 6,
    GameDifficulty.EXPERT: 8,
}

# Position weights for evaluation (center columns are better)
COLUMN_WEIGHTS = [1, 2, 3, 4, 3, 2, 1]


class Connect4Engine(GameEngine):
    """Connect 4 engine using Minimax algorithm.

    This engine provides full Connect 4 functionality with
    Minimax-based AI for move generation and analysis.

    The board is represented as a 2D list where:
    - Row 0 is the bottom row
    - Row 5 is the top row
    - Columns are 0-6 (left to right)

    Example:
        engine = Connect4Engine()
        position = engine.get_initial_position()
        validation = engine.validate_move(position, Move(notation="3"))
        if validation.is_valid:
            ai_response = engine.get_ai_move(validation.new_position, GameDifficulty.MEDIUM)
    """

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return GameType.CONNECT4

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Minimax Connect 4 Engine"

    def get_initial_position(self) -> Position:
        """Get the starting Connect 4 position (empty board)."""
        board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        return self._board_to_position(board, PLAYER1)

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a Connect 4 move.

        Args:
            position: Current board position.
            move: Move to validate (column number 0-6 or 1-7).

        Returns:
            MoveValidation with result and new position if valid.
        """
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CONNECT4,
            ) from e

        # Parse column number
        try:
            # Accept both 0-indexed (0-6) and 1-indexed (1-7)
            col = int(move.notation)
            if col >= 1 and col <= 7:
                col -= 1  # Convert to 0-indexed
        except ValueError:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Invalid move notation: {move.notation}. Use column number 1-7.",
            )

        # Check column is valid
        if col < 0 or col >= COLS:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Column {col + 1} is out of bounds. Use 1-7.",
            )

        # Check column is not full
        if board[ROWS - 1][col] != EMPTY:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Column {col + 1} is full.",
            )

        # Find the row where the piece will land
        row = self._find_landing_row(board, col)

        # Make the move
        new_board = [row[:] for row in board]
        new_board[row][col] = current_player

        # Check for win
        is_winner = self._check_win(new_board, row, col, current_player)
        is_draw = self._check_draw(new_board)

        # Determine result
        if is_winner:
            result = MoveResult.WINNING
            is_game_over = True
            winner = "player1" if current_player == PLAYER1 else "player2"
        elif is_draw:
            result = MoveResult.DRAW
            is_game_over = True
            winner = None
        else:
            result = MoveResult.VALID
            is_game_over = False
            winner = None

        # Next player
        next_player = PLAYER2 if current_player == PLAYER1 else PLAYER1

        new_position = self._board_to_position(new_board, next_player)

        return MoveValidation(
            is_valid=True,
            result=result,
            new_position=new_position,
            is_game_over=is_game_over,
            winner=winner,
        )

    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves (non-full columns)."""
        try:
            board, _ = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CONNECT4,
            ) from e

        moves = []
        for col in range(COLS):
            if board[ROWS - 1][col] == EMPTY:
                moves.append(Move(
                    notation=str(col + 1),
                    to_pos=f"col{col + 1}",
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
                game_type=GameType.CONNECT4,
            ) from e

        depth = DIFFICULTY_DEPTHS[difficulty]

        # Add randomness for lower difficulties
        if difficulty == GameDifficulty.BEGINNER:
            # Sometimes make random moves
            if random.random() < 0.5:
                legal_cols = [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]
                if legal_cols:
                    col = random.choice(legal_cols)
                    return AIMove(
                        move=str(col + 1),
                        thinking_time_ms=50,
                        evaluation=0.0,
                        move_quality="normal",
                    )

        # Use Minimax
        best_col, best_score = self._minimax_root(board, depth, current_player)

        if best_col is None:
            # No valid move found (shouldn't happen in normal play)
            legal_cols = [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]
            best_col = random.choice(legal_cols) if legal_cols else 0

        # Determine move quality
        move_quality = "normal"
        if abs(best_score) > 10000:
            move_quality = "winning" if best_score > 0 else "losing"

        return AIMove(
            move=str(best_col + 1),
            thinking_time_ms=100,
            evaluation=best_score / 100,  # Normalize
            move_quality=move_quality,
        )

    def analyze_position(
        self,
        position: Position,
        depth: int = 6,
    ) -> PositionAnalysis:
        """Analyze the current position."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CONNECT4,
            ) from e

        # Get best move
        best_col, best_score = self._minimax_root(board, depth, current_player)

        # Evaluate position
        evaluation = best_score / 100 if best_score else 0.0

        if abs(evaluation) > 100:
            if current_player == PLAYER1:
                evaluation_text = "Winning position!" if evaluation > 0 else "Losing position"
            else:
                evaluation_text = "Winning position!" if evaluation < 0 else "Losing position"
        elif abs(evaluation) > 5:
            evaluation_text = "Slight advantage"
        else:
            evaluation_text = "Even position"

        # Find threats
        threats = self._find_threats(board, current_player)

        # Find opportunities
        opportunities = self._find_opportunities(board, current_player)

        # Get suggested moves
        suggested_moves = []
        legal_cols = [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]
        for col in legal_cols:
            row = self._find_landing_row(board, col)
            new_board = [r[:] for r in board]
            new_board[row][col] = current_player

            if self._check_win(new_board, row, col, current_player):
                suggested_moves.append(SuggestedMove(
                    move=str(col + 1),
                    explanation="Winning move!",
                ))
                break

        if not suggested_moves and best_col is not None:
            suggested_moves.append(SuggestedMove(
                move=str(best_col + 1),
                explanation="Best move based on analysis.",
            ))

        return PositionAnalysis(
            evaluation=evaluation,
            evaluation_text=evaluation_text,
            best_move=str(best_col + 1) if best_col is not None else None,
            best_move_explanation="This move gives the best position.",
            threats=threats,
            opportunities=opportunities,
            suggested_moves=suggested_moves,
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
                game_type=GameType.CONNECT4,
            ) from e

        analysis = self.analyze_position(position, depth=6)

        if hint_level == 1:
            # General hint
            if analysis.opportunities:
                hint_text = "Look for ways to connect your pieces."
            elif analysis.threats:
                hint_text = "Be careful - your opponent might have a winning move."
            else:
                hint_text = "Try to control the center columns - they give more winning options."

            return HintResponse(
                hint_level=1,
                hint_text=hint_text,
                hint_type="strategic",
                reveals_move=False,
            )

        elif hint_level == 2:
            # More specific hint
            if analysis.threats:
                hint_text = f"Watch out: {analysis.threats[0]}"
            elif analysis.opportunities:
                hint_text = f"Consider: {analysis.opportunities[0]}"
            elif analysis.best_move:
                # Hint about which side of the board
                col = int(analysis.best_move) - 1
                if col < 3:
                    hint_text = "Consider the left side of the board."
                elif col > 3:
                    hint_text = "Consider the right side of the board."
                else:
                    hint_text = "The center column is often a strong choice."
            else:
                hint_text = "Build multiple threats at once to create winning opportunities."

            return HintResponse(
                hint_level=2,
                hint_text=hint_text,
                hint_type="tactical",
                reveals_move=False,
            )

        else:  # Level 3
            # Reveal the best move
            if analysis.best_move:
                hint_text = f"Play column {analysis.best_move}. {analysis.best_move_explanation or ''}"
            else:
                hint_text = "Any move is fine in this position."

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
        for row in range(ROWS):
            for col in range(COLS):
                piece = board[row][col]
                if piece != EMPTY:
                    if self._check_win(board, row, col, piece):
                        winner = "player1" if piece == PLAYER1 else "player2"
                        return True, "connect4", winner

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

        # Build display grid (row 0 at bottom, displayed at top for visual)
        display_grid = []
        for row in range(ROWS - 1, -1, -1):
            display_row = []
            for col in range(COLS):
                cell = board[row][col]
                if cell == PLAYER1:
                    display_row.append("red")
                elif cell == PLAYER2:
                    display_row.append("yellow")
                else:
                    display_row.append(None)
            display_grid.append(display_row)

        return {
            "grid": display_grid,
            "rows": ROWS,
            "cols": COLS,
            "current_player": "player1" if current_player == PLAYER1 else "player2",
            "legal_columns": [
                col + 1 for col in range(COLS)
                if board[ROWS - 1][col] == EMPTY
            ],
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
        # Create notation: rows from bottom to top, current player indicator
        rows_str = []
        for row in board:
            rows_str.append("".join(row))
        notation = "/".join(rows_str) + f" {current_player}"

        return Position(
            notation=notation,
            board_state=board,
            metadata={
                "current_player": current_player,
                "rows": ROWS,
                "cols": COLS,
            },
        )

    def _position_to_board(self, position: Position) -> tuple[list[list[str]], str]:
        """Convert Position to board array and current player."""
        parts = position.notation.split(" ")
        rows_str = parts[0].split("/")
        current_player = parts[1] if len(parts) > 1 else PLAYER1

        board = []
        for row_str in rows_str:
            row = list(row_str)
            board.append(row)

        return board, current_player

    def _find_landing_row(self, board: list[list[str]], col: int) -> int:
        """Find the row where a piece will land in the given column."""
        for row in range(ROWS):
            if board[row][col] == EMPTY:
                return row
        return -1  # Column is full

    def _check_win(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        player: str,
    ) -> bool:
        """Check if the player has won after placing at (row, col)."""
        # Directions: horizontal, vertical, diagonal up-right, diagonal up-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= CONNECT:
                return True

        return False

    def _check_draw(self, board: list[list[str]]) -> bool:
        """Check if the game is a draw (board is full)."""
        return all(board[ROWS - 1][col] != EMPTY for col in range(COLS))

    def _evaluate_board(self, board: list[list[str]], player: str) -> int:
        """Evaluate the board position for the given player."""
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        score = 0

        # Check all windows of 4
        # Horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                window = [board[row][col + i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        # Vertical
        for row in range(ROWS - 3):
            for col in range(COLS):
                window = [board[row + i][col] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        # Diagonal up-right
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        # Diagonal up-left
        for row in range(ROWS - 3):
            for col in range(3, COLS):
                window = [board[row + i][col - i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        # Center column preference
        center_col = COLS // 2
        for row in range(ROWS):
            if board[row][center_col] == player:
                score += 3

        return score

    def _evaluate_window(
        self,
        window: list[str],
        player: str,
        opponent: str,
    ) -> int:
        """Evaluate a window of 4 cells."""
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(EMPTY)

        if player_count == 4:
            return 100000
        elif player_count == 3 and empty_count == 1:
            return 100
        elif player_count == 2 and empty_count == 2:
            return 10

        if opponent_count == 4:
            return -100000
        elif opponent_count == 3 and empty_count == 1:
            return -80  # Block threats

        return 0

    def _minimax_root(
        self,
        board: list[list[str]],
        depth: int,
        player: str,
    ) -> tuple[int | None, int]:
        """Minimax root with alpha-beta pruning."""
        best_col = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        legal_cols = [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]

        # Order moves: center first
        legal_cols.sort(key=lambda c: abs(c - COLS // 2))

        for col in legal_cols:
            row = self._find_landing_row(board, col)
            new_board = [r[:] for r in board]
            new_board[row][col] = player

            opponent = PLAYER2 if player == PLAYER1 else PLAYER1
            score = -self._minimax(new_board, depth - 1, opponent, -beta, -alpha, player)

            if score > best_score:
                best_score = score
                best_col = col

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_col, int(best_score)

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
        for row in range(ROWS):
            for col in range(COLS):
                piece = board[row][col]
                if piece != EMPTY and self._check_win(board, row, col, piece):
                    if piece == maximizing_player:
                        return 100000 + depth  # Prefer faster wins
                    else:
                        return -100000 - depth  # Avoid losses

        if self._check_draw(board):
            return 0

        if depth == 0:
            return self._evaluate_board(board, maximizing_player)

        legal_cols = [c for c in range(COLS) if board[ROWS - 1][c] == EMPTY]

        # Order moves: center first
        legal_cols.sort(key=lambda c: abs(c - COLS // 2))

        best_score = float("-inf")
        for col in legal_cols:
            row = self._find_landing_row(board, col)
            new_board = [r[:] for r in board]
            new_board[row][col] = current_player

            opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
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
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1

        # Check for opponent's winning threats
        for col in range(COLS):
            if board[ROWS - 1][col] != EMPTY:
                continue
            row = self._find_landing_row(board, col)
            if row < 0:
                continue

            # Simulate opponent's move
            test_board = [r[:] for r in board]
            test_board[row][col] = opponent
            if self._check_win(test_board, row, col, opponent):
                threats.append(f"Opponent can win by playing column {col + 1}!")

        return threats[:3]

    def _find_opportunities(self, board: list[list[str]], player: str) -> list[str]:
        """Find opportunities in the position."""
        opportunities = []

        # Check for winning moves
        for col in range(COLS):
            if board[ROWS - 1][col] != EMPTY:
                continue
            row = self._find_landing_row(board, col)
            if row < 0:
                continue

            test_board = [r[:] for r in board]
            test_board[row][col] = player
            if self._check_win(test_board, row, col, player):
                opportunities.append(f"You can win by playing column {col + 1}!")

        # Check for building threats
        if not opportunities:
            for col in range(COLS):
                if board[ROWS - 1][col] != EMPTY:
                    continue
                row = self._find_landing_row(board, col)
                if row < 0:
                    continue

                test_board = [r[:] for r in board]
                test_board[row][col] = player

                # Check if this creates a threat
                if row + 1 < ROWS:
                    threat_board = [r[:] for r in test_board]
                    threat_board[row + 1][col] = player
                    if self._check_win(threat_board, row + 1, col, player):
                        opportunities.append(
                            f"Playing column {col + 1} creates a winning threat."
                        )

        return opportunities[:3]
