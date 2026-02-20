# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Othello (Reversi) engine implementation using Minimax algorithm.

This module provides an Othello engine that:
- Implements standard Othello rules (8x8 board, flipping mechanic)
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
BOARD_SIZE = 8

# Player symbols
EMPTY = "."
BLACK = "B"  # Black plays first
WHITE = "W"

# Difficulty settings (search depth)
DIFFICULTY_DEPTHS = {
    GameDifficulty.BEGINNER: 1,
    GameDifficulty.EASY: 2,
    GameDifficulty.MEDIUM: 4,
    GameDifficulty.HARD: 6,
    GameDifficulty.EXPERT: 8,
}

# Position weights for evaluation (corners are very valuable)
POSITION_WEIGHTS = [
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   1,   1,   1,   1,  -2,  10],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [ 10,  -2,   1,   1,   1,   1,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
]

# All eight directions
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloEngine(GameEngine):
    """Othello engine using Minimax algorithm.

    This engine provides full Othello functionality with
    Minimax-based AI for move generation and analysis.

    The board is represented as a 2D list (8x8).
    Notation: "d3" means column d (3), row 3 (2 from top)

    Example:
        engine = OthelloEngine()
        position = engine.get_initial_position()
        validation = engine.validate_move(position, Move(notation="d3"))
    """

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return GameType.OTHELLO

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Minimax Othello Engine"

    def get_initial_position(self) -> Position:
        """Get the starting Othello position (center 4 pieces)."""
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        # Initial setup: 2 white, 2 black in center
        board[3][3] = WHITE
        board[3][4] = BLACK
        board[4][3] = BLACK
        board[4][4] = WHITE
        return self._board_to_position(board, BLACK)

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate an Othello move.

        Args:
            position: Current board position.
            move: Move to validate (e.g., "d3").

        Returns:
            MoveValidation with result and new position if valid.
        """
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.OTHELLO,
            ) from e

        # Check for pass move
        if move.notation.lower() == "pass":
            legal_moves = self._get_legal_moves_internal(board, current_player)
            if legal_moves:
                return MoveValidation(
                    is_valid=False,
                    result=MoveResult.INVALID,
                    error_message="Cannot pass when you have legal moves.",
                )
            # Valid pass
            next_player = WHITE if current_player == BLACK else BLACK
            new_position = self._board_to_position(board, next_player)

            # Check if game is over (both players must pass)
            opponent_moves = self._get_legal_moves_internal(board, next_player)
            if not opponent_moves:
                # Game over
                black_count, white_count = self._count_pieces(board)
                if black_count > white_count:
                    winner = "black"
                elif white_count > black_count:
                    winner = "white"
                else:
                    winner = None
                return MoveValidation(
                    is_valid=True,
                    result=MoveResult.VALID,
                    new_position=new_position,
                    is_game_over=True,
                    winner=winner,
                )

            return MoveValidation(
                is_valid=True,
                result=MoveResult.VALID,
                new_position=new_position,
                is_game_over=False,
            )

        # Parse move notation
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

        # Find pieces to flip
        flips = self._get_flips(board, row, col, current_player)
        if not flips:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Move {move.notation} doesn't flip any pieces.",
            )

        # Make the move
        new_board = [row_list[:] for row_list in board]
        new_board[row][col] = current_player
        for flip_row, flip_col in flips:
            new_board[flip_row][flip_col] = current_player

        # Next player
        next_player = WHITE if current_player == BLACK else BLACK

        # Check if next player has moves
        next_player_moves = self._get_legal_moves_internal(new_board, next_player)
        if not next_player_moves:
            # Check if current player has moves
            current_player_moves = self._get_legal_moves_internal(new_board, current_player)
            if not current_player_moves:
                # Game over
                black_count, white_count = self._count_pieces(new_board)
                if black_count > white_count:
                    winner = "black"
                    result = MoveResult.WINNING if current_player == BLACK else MoveResult.LOSING
                elif white_count > black_count:
                    winner = "white"
                    result = MoveResult.WINNING if current_player == WHITE else MoveResult.LOSING
                else:
                    winner = None
                    result = MoveResult.DRAW

                new_position = self._board_to_position(new_board, next_player)
                return MoveValidation(
                    is_valid=True,
                    result=result,
                    new_position=new_position,
                    is_game_over=True,
                    winner=winner,
                )
            else:
                # Next player must pass, current player continues
                next_player = current_player

        new_position = self._board_to_position(new_board, next_player)

        return MoveValidation(
            is_valid=True,
            result=MoveResult.VALID,
            new_position=new_position,
            is_game_over=False,
        )

    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.OTHELLO,
            ) from e

        internal_moves = self._get_legal_moves_internal(board, current_player)

        if not internal_moves:
            # Must pass
            return [Move(notation="pass", to_pos="pass")]

        moves = []
        for row, col in internal_moves:
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
                game_type=GameType.OTHELLO,
            ) from e

        legal_moves = self._get_legal_moves_internal(board, current_player)

        if not legal_moves:
            return AIMove(
                move="pass",
                thinking_time_ms=10,
                evaluation=0.0,
                move_quality="forced",
            )

        depth = DIFFICULTY_DEPTHS[difficulty]

        # For beginner, sometimes make random moves
        if difficulty == GameDifficulty.BEGINNER and random.random() < 0.4:
            row, col = random.choice(legal_moves)
            return AIMove(
                move=self._to_notation(row, col),
                thinking_time_ms=50,
                evaluation=0.0,
                move_quality="normal",
            )

        # Use Minimax
        best_move, best_score = self._minimax_root(board, depth, current_player)

        if best_move is None:
            best_move = random.choice(legal_moves)

        row, col = best_move
        notation = self._to_notation(row, col)

        # Determine move quality
        move_quality = "normal"
        # Corner moves are excellent
        if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            move_quality = "excellent"

        return AIMove(
            move=notation,
            thinking_time_ms=100,
            evaluation=best_score / 100,
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
                game_type=GameType.OTHELLO,
            ) from e

        legal_moves = self._get_legal_moves_internal(board, current_player)

        if not legal_moves:
            return PositionAnalysis(
                evaluation=0.0,
                evaluation_text="No moves available - must pass",
                best_move="pass",
            )

        # Get best move
        best_move, best_score = self._minimax_root(board, min(depth, 4), current_player)

        evaluation = best_score / 100 if best_score else 0.0

        black_count, white_count = self._count_pieces(board)
        total = black_count + white_count

        if total > 55:  # Endgame
            piece_diff = black_count - white_count if current_player == BLACK else white_count - black_count
            if piece_diff > 10:
                evaluation_text = "Winning - ahead in pieces"
            elif piece_diff < -10:
                evaluation_text = "Behind in pieces"
            else:
                evaluation_text = "Close game"
        else:
            if evaluation > 10:
                evaluation_text = "Good position"
            elif evaluation < -10:
                evaluation_text = "Difficult position"
            else:
                evaluation_text = "Even position"

        # Find threats and opportunities
        threats = self._find_threats(board, current_player)
        opportunities = self._find_opportunities(board, current_player, legal_moves)

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
                game_type=GameType.OTHELLO,
            ) from e

        analysis = self.analyze_position(position, depth=4)

        if hint_level == 1:
            if analysis.opportunities:
                hint_text = "Look for corner moves or moves that flip many pieces."
            elif analysis.threats:
                hint_text = "Be careful not to give your opponent corner access."
            else:
                hint_text = "Try to get corners and edges - they can't be flipped!"

            return HintResponse(
                hint_level=1,
                hint_text=hint_text,
                hint_type="strategic",
                reveals_move=False,
            )

        elif hint_level == 2:
            if analysis.opportunities:
                hint_text = analysis.opportunities[0]
            elif analysis.threats:
                hint_text = analysis.threats[0]
            elif analysis.best_move:
                col, row = self._parse_move(analysis.best_move)
                # Give area hint
                if row < 3 and col < 3:
                    area = "top-left"
                elif row < 3 and col > 4:
                    area = "top-right"
                elif row > 4 and col < 3:
                    area = "bottom-left"
                elif row > 4 and col > 4:
                    area = "bottom-right"
                else:
                    area = "center"
                hint_text = f"Consider the {area} area of the board."
            else:
                hint_text = "Focus on controlling the edges."

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
                hint_text = "Pass this turn."

            return HintResponse(
                hint_level=3,
                hint_text=hint_text,
                hint_type="solution",
                reveals_move=True,
            )

    def is_game_over(self, position: Position) -> tuple[bool, str | None, str | None]:
        """Check if the game is over."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception:
            return False, None, None

        opponent = WHITE if current_player == BLACK else BLACK

        current_moves = self._get_legal_moves_internal(board, current_player)
        opponent_moves = self._get_legal_moves_internal(board, opponent)

        if current_moves or opponent_moves:
            return False, None, None

        # Game over
        black_count, white_count = self._count_pieces(board)
        if black_count > white_count:
            return True, "more_pieces", "black"
        elif white_count > black_count:
            return True, "more_pieces", "white"
        else:
            return True, "draw", None

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

        black_count, white_count = self._count_pieces(board)
        legal_moves = self._get_legal_moves_internal(board, current_player)

        return {
            "grid": display_grid,
            "size": BOARD_SIZE,
            "current_player": "black" if current_player == BLACK else "white",
            "black_count": black_count,
            "white_count": white_count,
            "legal_moves": [self._to_notation(r, c) for r, c in legal_moves],
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
        """Parse move notation (e.g., 'd3') to (col, row)."""
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
        """Convert (row, col) to notation (e.g., 'd3')."""
        col_char = chr(ord('a') + col)
        row_num = row + 1
        return f"{col_char}{row_num}"

    def _get_flips(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        player: str,
    ) -> list[tuple[int, int]]:
        """Get all pieces that would be flipped by playing at (row, col)."""
        opponent = WHITE if player == BLACK else BLACK
        all_flips = []

        for dr, dc in DIRECTIONS:
            flips = []
            r, c = row + dr, col + dc

            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opponent:
                flips.append((r, c))
                r += dr
                c += dc

            # Check if we found our piece at the end
            if flips and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                all_flips.extend(flips)

        return all_flips

    def _get_legal_moves_internal(
        self,
        board: list[list[str]],
        player: str,
    ) -> list[tuple[int, int]]:
        """Get all legal moves as (row, col) tuples."""
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == EMPTY:
                    if self._get_flips(board, row, col, player):
                        moves.append((row, col))
        return moves

    def _count_pieces(self, board: list[list[str]]) -> tuple[int, int]:
        """Count black and white pieces."""
        black = 0
        white = 0
        for row in board:
            for cell in row:
                if cell == BLACK:
                    black += 1
                elif cell == WHITE:
                    white += 1
        return black, white

    def _evaluate_board(self, board: list[list[str]], player: str) -> int:
        """Evaluate the board position for the given player."""
        opponent = WHITE if player == BLACK else BLACK
        score = 0

        # Position weights
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == player:
                    score += POSITION_WEIGHTS[row][col]
                elif board[row][col] == opponent:
                    score -= POSITION_WEIGHTS[row][col]

        # Mobility (number of legal moves)
        player_moves = len(self._get_legal_moves_internal(board, player))
        opponent_moves = len(self._get_legal_moves_internal(board, opponent))
        score += (player_moves - opponent_moves) * 5

        # Piece count (more important in endgame)
        black_count, white_count = self._count_pieces(board)
        total = black_count + white_count

        if total > 55:  # Endgame - piece count matters more
            if player == BLACK:
                score += (black_count - white_count) * 10
            else:
                score += (white_count - black_count) * 10

        return score

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

        legal_moves = self._get_legal_moves_internal(board, player)

        # Order moves: corners first, then edges
        def move_priority(move: tuple[int, int]) -> int:
            r, c = move
            if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                return 0  # Corners first
            if r == 0 or r == 7 or c == 0 or c == 7:
                return 1  # Edges second
            return 2  # Rest

        legal_moves.sort(key=move_priority)

        for row, col in legal_moves:
            new_board = self._make_move(board, row, col, player)

            opponent = WHITE if player == BLACK else BLACK
            score = -self._minimax(new_board, depth - 1, opponent, -beta, -alpha, player)

            if score > best_score:
                best_score = score
                best_move = (row, col)

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_move, int(best_score)

    def _make_move(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        player: str,
    ) -> list[list[str]]:
        """Make a move and return new board."""
        new_board = [r[:] for r in board]
        new_board[row][col] = player
        for flip_row, flip_col in self._get_flips(board, row, col, player):
            new_board[flip_row][flip_col] = player
        return new_board

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
        opponent = WHITE if current_player == BLACK else BLACK

        legal_moves = self._get_legal_moves_internal(board, current_player)

        if not legal_moves:
            # Check if opponent can move
            opponent_moves = self._get_legal_moves_internal(board, opponent)
            if not opponent_moves:
                # Game over
                black_count, white_count = self._count_pieces(board)
                if maximizing_player == BLACK:
                    diff = black_count - white_count
                else:
                    diff = white_count - black_count
                return diff * 1000  # Winning is very valuable

            # Must pass - opponent plays
            return -self._minimax(
                board, depth, opponent, -beta, -alpha, maximizing_player
            )

        if depth == 0:
            return self._evaluate_board(board, maximizing_player)

        best_score = float("-inf")
        for row, col in legal_moves:
            new_board = self._make_move(board, row, col, current_player)

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

        opponent_moves = self._get_legal_moves_internal(board, opponent)
        for row, col in opponent_moves:
            if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                threats.append(f"Opponent can take corner at {self._to_notation(row, col)}!")

        return threats[:3]

    def _find_opportunities(
        self,
        board: list[list[str]],
        player: str,
        legal_moves: list[tuple[int, int]],
    ) -> list[str]:
        """Find opportunities in the position."""
        opportunities = []

        for row, col in legal_moves:
            if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                opportunities.append(f"Corner available at {self._to_notation(row, col)}!")

            flips = self._get_flips(board, row, col, player)
            if len(flips) >= 4:
                opportunities.append(
                    f"Big flip ({len(flips)} pieces) at {self._to_notation(row, col)}!"
                )

        return opportunities[:3]
