# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkers (Draughts) engine implementation using Minimax algorithm.

This module provides a Checkers engine that:
- Implements standard American Checkers rules (8x8 board)
- Uses Minimax with alpha-beta pruning for AI moves
- Supports kings (crowned pieces) and mandatory captures
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
BLACK = "b"  # Black pieces (moves down, starts at top)
BLACK_KING = "B"
WHITE = "w"  # White pieces (moves up, starts at bottom)
WHITE_KING = "W"

# Difficulty settings (search depth)
DIFFICULTY_DEPTHS = {
    GameDifficulty.BEGINNER: 2,
    GameDifficulty.EASY: 3,
    GameDifficulty.MEDIUM: 5,
    GameDifficulty.HARD: 7,
    GameDifficulty.EXPERT: 9,
}

# Position weights for evaluation
POSITION_WEIGHTS = [
    [0, 4, 0, 4, 0, 4, 0, 4],
    [4, 0, 3, 0, 3, 0, 3, 0],
    [0, 3, 0, 2, 0, 2, 0, 4],
    [4, 0, 2, 0, 1, 0, 3, 0],
    [0, 3, 0, 1, 0, 2, 0, 4],
    [4, 0, 2, 0, 2, 0, 3, 0],
    [0, 3, 0, 3, 0, 3, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0],
]


class CheckersEngine(GameEngine):
    """Checkers engine using Minimax algorithm.

    This engine provides full Checkers functionality with
    Minimax-based AI for move generation and analysis.

    The board is represented as a 2D list (8x8).
    - Only dark squares are used (where row + col is odd)
    - Black starts at top, White starts at bottom
    - Notation: "b6-a5" for simple move, "b6xd4" for capture

    Example:
        engine = CheckersEngine()
        position = engine.get_initial_position()
        validation = engine.validate_move(position, Move(notation="c3-d4"))
    """

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return GameType.CHECKERS

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Minimax Checkers Engine"

    def get_initial_position(self) -> Position:
        """Get the starting Checkers position."""
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        # Place black pieces (top 3 rows)
        for row in range(3):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:  # Dark squares
                    board[row][col] = BLACK

        # Place white pieces (bottom 3 rows)
        for row in range(5, 8):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:  # Dark squares
                    board[row][col] = WHITE

        return self._board_to_position(board, BLACK)

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a Checkers move.

        Args:
            position: Current board position.
            move: Move notation (e.g., "c3-d4" for move, "c3xe5" for capture).

        Returns:
            MoveValidation with result and new position if valid.
        """
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHECKERS,
            ) from e

        # Parse move notation
        try:
            from_pos, to_pos, is_capture = self._parse_move(move.notation)
        except ValueError as e:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=str(e),
            )

        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Check bounds
        if not self._in_bounds(from_row, from_col) or not self._in_bounds(to_row, to_col):
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message="Position out of bounds.",
            )

        # Check if there's a piece at from position
        piece = board[from_row][from_col]
        if piece == EMPTY:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message="No piece at starting position.",
            )

        # Check if piece belongs to current player
        if not self._is_player_piece(piece, current_player):
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message="That's not your piece.",
            )

        # Check if destination is empty
        if board[to_row][to_col] != EMPTY:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message="Destination is occupied.",
            )

        # Get all legal moves
        all_captures = self._get_all_captures(board, current_player)
        all_simple = self._get_all_simple_moves(board, current_player)

        # Mandatory capture rule
        if all_captures and not is_capture:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message="You must capture when possible!",
            )

        # Validate the specific move
        if is_capture:
            valid_captures = self._get_captures_for_piece(board, from_row, from_col, piece)
            if (to_row, to_col) not in [c[-1] for c in valid_captures]:
                return MoveValidation(
                    is_valid=False,
                    result=MoveResult.INVALID,
                    error_message="Invalid capture move.",
                )

            # Find the capture sequence
            capture_path = None
            for path in valid_captures:
                if path[-1] == (to_row, to_col):
                    capture_path = path
                    break

            if not capture_path:
                return MoveValidation(
                    is_valid=False,
                    result=MoveResult.INVALID,
                    error_message="Invalid capture path.",
                )

            # Execute the capture
            new_board = self._execute_capture(board, from_row, from_col, capture_path)
        else:
            # Simple move
            valid_moves = self._get_simple_moves_for_piece(board, from_row, from_col, piece)
            if (to_row, to_col) not in valid_moves:
                return MoveValidation(
                    is_valid=False,
                    result=MoveResult.INVALID,
                    error_message="Invalid move.",
                )

            # Execute simple move
            new_board = [row[:] for row in board]
            new_board[from_row][from_col] = EMPTY
            new_board[to_row][to_col] = piece

            # Check for promotion
            new_board = self._check_promotion(new_board, to_row, to_col)

        # Next player
        next_player = WHITE if current_player == BLACK else BLACK

        # Check for game over
        next_moves = self._get_all_captures(new_board, next_player) or \
                     self._get_all_simple_moves(new_board, next_player)

        if not next_moves:
            # Current player wins (opponent has no moves)
            winner = "black" if current_player == BLACK else "white"
            new_position = self._board_to_position(new_board, next_player)
            return MoveValidation(
                is_valid=True,
                result=MoveResult.WINNING,
                new_position=new_position,
                is_game_over=True,
                winner=winner,
            )

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
                game_type=GameType.CHECKERS,
            ) from e

        moves = []

        # Check for captures first (mandatory)
        captures = self._get_all_captures(board, current_player)
        if captures:
            for from_pos, path in captures:
                from_row, from_col = from_pos
                to_row, to_col = path[-1]
                notation = f"{self._to_notation(from_row, from_col)}x{self._to_notation(to_row, to_col)}"
                moves.append(Move(
                    notation=notation,
                    from_pos=self._to_notation(from_row, from_col),
                    to_pos=self._to_notation(to_row, to_col),
                ))
            return moves

        # Simple moves
        simple_moves = self._get_all_simple_moves(board, current_player)
        for from_pos, to_pos in simple_moves:
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            notation = f"{self._to_notation(from_row, from_col)}-{self._to_notation(to_row, to_col)}"
            moves.append(Move(
                notation=notation,
                from_pos=self._to_notation(from_row, from_col),
                to_pos=self._to_notation(to_row, to_col),
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
                game_type=GameType.CHECKERS,
            ) from e

        legal_moves = self.get_legal_moves(position)

        if not legal_moves:
            return AIMove(
                move="",
                thinking_time_ms=0,
                evaluation=0.0,
                move_quality="no_moves",
            )

        depth = DIFFICULTY_DEPTHS[difficulty]

        # For beginner, sometimes make random moves
        if difficulty == GameDifficulty.BEGINNER and random.random() < 0.4:
            chosen = random.choice(legal_moves)
            return AIMove(
                move=chosen.notation,
                thinking_time_ms=50,
                evaluation=0.0,
                move_quality="normal",
            )

        # Use Minimax
        best_move, best_score = self._minimax_root(board, depth, current_player)

        if best_move is None:
            best_move = legal_moves[0].notation
        else:
            # Convert to notation
            from_pos, to_pos, is_capture = best_move
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            sep = "x" if is_capture else "-"
            best_move = f"{self._to_notation(from_row, from_col)}{sep}{self._to_notation(to_row, to_col)}"

        move_quality = "normal"
        if "x" in best_move:
            move_quality = "capture"

        return AIMove(
            move=best_move,
            thinking_time_ms=100,
            evaluation=best_score / 100 if best_score else 0.0,
            move_quality=move_quality,
        )

    def analyze_position(
        self,
        position: Position,
        depth: int = 5,
    ) -> PositionAnalysis:
        """Analyze the current position."""
        try:
            board, current_player = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHECKERS,
            ) from e

        legal_moves = self.get_legal_moves(position)

        if not legal_moves:
            return PositionAnalysis(
                evaluation=-1000.0,
                evaluation_text="No moves - game over",
            )

        # Get best move
        best_move, best_score = self._minimax_root(board, min(depth, 4), current_player)

        evaluation = best_score / 100 if best_score else 0.0

        # Count pieces
        black_pieces, black_kings = self._count_pieces(board, BLACK)
        white_pieces, white_kings = self._count_pieces(board, WHITE)

        if current_player == BLACK:
            piece_diff = (black_pieces + black_kings * 1.5) - (white_pieces + white_kings * 1.5)
        else:
            piece_diff = (white_pieces + white_kings * 1.5) - (black_pieces + black_kings * 1.5)

        if piece_diff > 2:
            evaluation_text = "Good advantage - more pieces"
        elif piece_diff < -2:
            evaluation_text = "Behind in pieces"
        elif evaluation > 50:
            evaluation_text = "Good position"
        elif evaluation < -50:
            evaluation_text = "Difficult position"
        else:
            evaluation_text = "Even position"

        # Find threats and opportunities
        threats = self._find_threats(board, current_player)
        opportunities = self._find_opportunities(board, current_player)

        best_notation = None
        if best_move:
            from_pos, to_pos, is_capture = best_move
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            sep = "x" if is_capture else "-"
            best_notation = f"{self._to_notation(from_row, from_col)}{sep}{self._to_notation(to_row, to_col)}"

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
                game_type=GameType.CHECKERS,
            ) from e

        analysis = self.analyze_position(position, depth=4)

        if hint_level == 1:
            captures = self._get_all_captures(board, current_player)
            if captures:
                hint_text = "Look for a capture - you must take when you can!"
            elif analysis.opportunities:
                hint_text = "Look for ways to get a king or set up a capture."
            else:
                hint_text = "Try to advance your pieces toward the king row."

            return HintResponse(
                hint_level=1,
                hint_text=hint_text,
                hint_type="strategic",
                reveals_move=False,
            )

        elif hint_level == 2:
            captures = self._get_all_captures(board, current_player)
            if captures:
                hint_text = "You have a capture available - find it!"
            elif analysis.threats:
                hint_text = analysis.threats[0]
            elif analysis.opportunities:
                hint_text = analysis.opportunities[0]
            elif analysis.best_move:
                # Give piece hint
                parts = analysis.best_move.split("-")
                if not parts or len(parts) < 1:
                    parts = analysis.best_move.split("x")
                if parts:
                    hint_text = f"Consider moving the piece at {parts[0]}."
                else:
                    hint_text = "Look for an advancing move."
            else:
                hint_text = "Try to control the center of the board."

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
                hint_text = "Make any available move."

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

        # Check if current player has any moves
        captures = self._get_all_captures(board, current_player)
        simple_moves = self._get_all_simple_moves(board, current_player)

        if not captures and not simple_moves:
            # Current player loses (no moves)
            winner = "white" if current_player == BLACK else "black"
            return True, "no_moves", winner

        # Check for piece count
        black_pieces, black_kings = self._count_pieces(board, BLACK)
        white_pieces, white_kings = self._count_pieces(board, WHITE)

        if black_pieces + black_kings == 0:
            return True, "no_pieces", "white"
        if white_pieces + white_kings == 0:
            return True, "no_pieces", "black"

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
                elif cell == BLACK_KING:
                    display_row.append("black_king")
                elif cell == WHITE:
                    display_row.append("white")
                elif cell == WHITE_KING:
                    display_row.append("white_king")
                else:
                    display_row.append(None)
            display_grid.append(display_row)

        black_pieces, black_kings = self._count_pieces(board, BLACK)
        white_pieces, white_kings = self._count_pieces(board, WHITE)

        return {
            "grid": display_grid,
            "size": BOARD_SIZE,
            "current_player": "black" if current_player == BLACK else "white",
            "black_pieces": black_pieces,
            "black_kings": black_kings,
            "white_pieces": white_pieces,
            "white_kings": white_kings,
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

    def _parse_move(self, notation: str) -> tuple[tuple[int, int], tuple[int, int], bool]:
        """Parse move notation (e.g., 'c3-d4' or 'c3xd4')."""
        notation = notation.lower().strip()

        is_capture = "x" in notation
        sep = "x" if is_capture else "-"

        parts = notation.split(sep)
        if len(parts) != 2:
            raise ValueError(f"Invalid notation: {notation}")

        from_col, from_row = self._parse_square(parts[0])
        to_col, to_row = self._parse_square(parts[1])

        return (from_row, from_col), (to_row, to_col), is_capture

    def _parse_square(self, square: str) -> tuple[int, int]:
        """Parse square notation (e.g., 'c3') to (col, row)."""
        if len(square) < 2:
            raise ValueError(f"Invalid square: {square}")

        col_char = square[0]
        row_str = square[1:]

        col = ord(col_char) - ord('a')
        row = int(row_str) - 1

        return col, row

    def _to_notation(self, row: int, col: int) -> str:
        """Convert (row, col) to notation (e.g., 'c3')."""
        col_char = chr(ord('a') + col)
        row_num = row + 1
        return f"{col_char}{row_num}"

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _is_player_piece(self, piece: str, player: str) -> bool:
        """Check if piece belongs to player."""
        if player == BLACK:
            return piece in (BLACK, BLACK_KING)
        else:
            return piece in (WHITE, WHITE_KING)

    def _is_king(self, piece: str) -> bool:
        """Check if piece is a king."""
        return piece in (BLACK_KING, WHITE_KING)

    def _get_move_directions(self, piece: str) -> list[tuple[int, int]]:
        """Get valid move directions for a piece."""
        if piece == BLACK:
            return [(1, -1), (1, 1)]  # Black moves down
        elif piece == WHITE:
            return [(-1, -1), (-1, 1)]  # White moves up
        else:  # Kings
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def _get_simple_moves_for_piece(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        piece: str,
    ) -> list[tuple[int, int]]:
        """Get simple moves for a specific piece."""
        moves = []
        directions = self._get_move_directions(piece)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._in_bounds(new_row, new_col) and board[new_row][new_col] == EMPTY:
                moves.append((new_row, new_col))

        return moves

    def _get_captures_for_piece(
        self,
        board: list[list[str]],
        row: int,
        col: int,
        piece: str,
        path: list[tuple[int, int]] | None = None,
    ) -> list[list[tuple[int, int]]]:
        """Get all capture paths for a specific piece."""
        if path is None:
            path = []

        captures = []
        directions = self._get_move_directions(piece)
        opponent = WHITE if piece in (BLACK, BLACK_KING) else BLACK

        for dr, dc in directions:
            mid_row, mid_col = row + dr, col + dc
            end_row, end_col = row + 2 * dr, col + 2 * dc

            if not self._in_bounds(end_row, end_col):
                continue

            mid_piece = board[mid_row][mid_col]
            if mid_piece == EMPTY or self._is_player_piece(mid_piece, piece[0].lower() if piece.isupper() else piece):
                continue

            # Check if opponent piece and destination empty
            if self._is_player_piece(mid_piece, opponent) and board[end_row][end_col] == EMPTY:
                # Found a capture
                new_path = path + [(end_row, end_col)]

                # Check for multi-capture
                temp_board = [r[:] for r in board]
                temp_board[row][col] = EMPTY
                temp_board[mid_row][mid_col] = EMPTY
                temp_board[end_row][end_col] = piece

                # Check for promotion
                temp_board = self._check_promotion(temp_board, end_row, end_col)
                new_piece = temp_board[end_row][end_col]

                further_captures = self._get_captures_for_piece(
                    temp_board, end_row, end_col, new_piece, new_path
                )

                if further_captures:
                    captures.extend(further_captures)
                else:
                    captures.append(new_path)

        return captures

    def _get_all_captures(
        self,
        board: list[list[str]],
        player: str,
    ) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
        """Get all captures for a player."""
        captures = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece != EMPTY and self._is_player_piece(piece, player):
                    piece_captures = self._get_captures_for_piece(board, row, col, piece)
                    for path in piece_captures:
                        captures.append(((row, col), path))

        return captures

    def _get_all_simple_moves(
        self,
        board: list[list[str]],
        player: str,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Get all simple moves for a player."""
        moves = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece != EMPTY and self._is_player_piece(piece, player):
                    piece_moves = self._get_simple_moves_for_piece(board, row, col, piece)
                    for to_pos in piece_moves:
                        moves.append(((row, col), to_pos))

        return moves

    def _execute_capture(
        self,
        board: list[list[str]],
        from_row: int,
        from_col: int,
        path: list[tuple[int, int]],
    ) -> list[list[str]]:
        """Execute a capture sequence."""
        new_board = [row[:] for row in board]
        piece = new_board[from_row][from_col]
        new_board[from_row][from_col] = EMPTY

        current_row, current_col = from_row, from_col

        for to_row, to_col in path:
            # Find and remove captured piece
            dr = 1 if to_row > current_row else -1
            dc = 1 if to_col > current_col else -1
            mid_row, mid_col = current_row + dr, current_col + dc
            new_board[mid_row][mid_col] = EMPTY

            current_row, current_col = to_row, to_col

        # Place piece at final position
        new_board[current_row][current_col] = piece

        # Check for promotion
        new_board = self._check_promotion(new_board, current_row, current_col)

        return new_board

    def _check_promotion(
        self,
        board: list[list[str]],
        row: int,
        col: int,
    ) -> list[list[str]]:
        """Check and apply promotion to king."""
        piece = board[row][col]

        if piece == BLACK and row == BOARD_SIZE - 1:
            board[row][col] = BLACK_KING
        elif piece == WHITE and row == 0:
            board[row][col] = WHITE_KING

        return board

    def _count_pieces(
        self,
        board: list[list[str]],
        player: str,
    ) -> tuple[int, int]:
        """Count regular pieces and kings for a player."""
        pieces = 0
        kings = 0

        regular = BLACK if player == BLACK else WHITE
        king = BLACK_KING if player == BLACK else WHITE_KING

        for row in board:
            for cell in row:
                if cell == regular:
                    pieces += 1
                elif cell == king:
                    kings += 1

        return pieces, kings

    def _evaluate_board(self, board: list[list[str]], player: str) -> int:
        """Evaluate the board position for the given player."""
        opponent = WHITE if player == BLACK else BLACK
        score = 0

        # Piece count
        player_pieces, player_kings = self._count_pieces(board, player)
        opp_pieces, opp_kings = self._count_pieces(board, opponent)

        score += (player_pieces - opp_pieces) * 100
        score += (player_kings - opp_kings) * 150  # Kings are more valuable

        # Position weights
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece == EMPTY:
                    continue

                weight = POSITION_WEIGHTS[row][col]
                if self._is_player_piece(piece, player):
                    score += weight
                elif self._is_player_piece(piece, opponent):
                    score -= weight

        # Mobility
        player_captures = len(self._get_all_captures(board, player))
        player_moves = len(self._get_all_simple_moves(board, player))
        opp_captures = len(self._get_all_captures(board, opponent))
        opp_moves = len(self._get_all_simple_moves(board, opponent))

        score += (player_captures + player_moves - opp_captures - opp_moves) * 5

        return score

    def _minimax_root(
        self,
        board: list[list[str]],
        depth: int,
        player: str,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int], bool] | None, int]:
        """Minimax root with alpha-beta pruning."""
        best_move = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        # Get moves (captures take priority)
        captures = self._get_all_captures(board, player)
        if captures:
            moves = [(from_pos, path[-1], True, path) for from_pos, path in captures]
        else:
            simple = self._get_all_simple_moves(board, player)
            moves = [(from_pos, to_pos, False, None) for from_pos, to_pos in simple]

        for from_pos, to_pos, is_capture, path in moves:
            if is_capture:
                new_board = self._execute_capture(board, from_pos[0], from_pos[1], path)
            else:
                new_board = [row[:] for row in board]
                piece = new_board[from_pos[0]][from_pos[1]]
                new_board[from_pos[0]][from_pos[1]] = EMPTY
                new_board[to_pos[0]][to_pos[1]] = piece
                new_board = self._check_promotion(new_board, to_pos[0], to_pos[1])

            opponent = WHITE if player == BLACK else BLACK
            score = -self._minimax(new_board, depth - 1, opponent, -beta, -alpha, player)

            if score > best_score:
                best_score = score
                best_move = (from_pos, to_pos, is_capture)

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
        # Check for game over
        captures = self._get_all_captures(board, current_player)
        simple_moves = self._get_all_simple_moves(board, current_player)

        if not captures and not simple_moves:
            # Current player loses
            if current_player == maximizing_player:
                return -100000 - depth
            else:
                return 100000 + depth

        if depth == 0:
            return self._evaluate_board(board, maximizing_player)

        # Get moves
        if captures:
            moves = [(from_pos, path[-1], True, path) for from_pos, path in captures]
        else:
            moves = [(from_pos, to_pos, False, None) for from_pos, to_pos in simple_moves]

        best_score = float("-inf")
        opponent = WHITE if current_player == BLACK else BLACK

        for from_pos, to_pos, is_capture, path in moves:
            if is_capture:
                new_board = self._execute_capture(board, from_pos[0], from_pos[1], path)
            else:
                new_board = [row[:] for row in board]
                piece = new_board[from_pos[0]][from_pos[1]]
                new_board[from_pos[0]][from_pos[1]] = EMPTY
                new_board[to_pos[0]][to_pos[1]] = piece
                new_board = self._check_promotion(new_board, to_pos[0], to_pos[1])

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

        opp_captures = self._get_all_captures(board, opponent)
        if opp_captures:
            threats.append("Opponent can capture your piece!")

        return threats[:3]

    def _find_opportunities(self, board: list[list[str]], player: str) -> list[str]:
        """Find opportunities in the position."""
        opportunities = []

        captures = self._get_all_captures(board, player)
        if captures:
            if len(captures) > 1:
                opportunities.append(f"You have {len(captures)} capture options!")
            else:
                opportunities.append("You can capture an opponent's piece!")

        # Check for potential kings
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece == BLACK and row == BOARD_SIZE - 2:
                    opportunities.append("A piece is close to becoming a king!")
                    break
                if piece == WHITE and row == 1:
                    opportunities.append("A piece is close to becoming a king!")
                    break

        return opportunities[:3]
