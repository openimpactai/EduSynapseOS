# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Chess engine implementation using python-chess and Stockfish.

This module provides a chess engine that:
- Uses python-chess for move validation and game logic
- Integrates with Stockfish for AI moves and analysis
- Provides educational hints and analysis for coaching

Stockfish Integration:
- If Stockfish is available, uses it for strong AI play and deep analysis
- If not available, falls back to simple heuristics for basic functionality
- Stockfish path can be configured via STOCKFISH_PATH environment variable

The engine is stateless - all game state is passed via Position/GameState objects.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import chess
import chess.engine

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

# Default Stockfish paths to check
DEFAULT_STOCKFISH_PATHS = [
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",  # macOS Homebrew
    "stockfish",  # In PATH
]

# Difficulty settings for Stockfish
DIFFICULTY_SETTINGS = {
    GameDifficulty.BEGINNER: {"skill_level": 0, "depth": 1, "time_limit": 0.1},
    GameDifficulty.EASY: {"skill_level": 3, "depth": 3, "time_limit": 0.3},
    GameDifficulty.MEDIUM: {"skill_level": 8, "depth": 8, "time_limit": 0.5},
    GameDifficulty.HARD: {"skill_level": 15, "depth": 15, "time_limit": 1.0},
    GameDifficulty.EXPERT: {"skill_level": 20, "depth": 20, "time_limit": 2.0},
}

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ChessEngine(GameEngine):
    """Chess engine using python-chess and Stockfish.

    This engine provides full chess functionality with optional
    Stockfish integration for AI moves and deep analysis.

    Attributes:
        stockfish_path: Path to Stockfish binary.
        stockfish_available: Whether Stockfish is available.

    Example:
        engine = ChessEngine()
        position = engine.get_initial_position()
        validation = engine.validate_move(position, Move(notation="e2e4"))
        if validation.is_valid:
            ai_response = engine.get_ai_move(validation.new_position, GameDifficulty.MEDIUM)
    """

    def __init__(self, stockfish_path: str | None = None) -> None:
        """Initialize the chess engine.

        Args:
            stockfish_path: Optional path to Stockfish binary.
                If not provided, searches common locations.
        """
        self._stockfish_path = self._find_stockfish(stockfish_path)
        self._stockfish_available = self._stockfish_path is not None

        if self._stockfish_available:
            logger.info("ChessEngine initialized with Stockfish: %s", self._stockfish_path)
        else:
            logger.warning(
                "ChessEngine initialized without Stockfish. "
                "AI moves will use simple heuristics."
            )

    def _find_stockfish(self, explicit_path: str | None) -> str | None:
        """Find Stockfish binary.

        Args:
            explicit_path: Explicitly provided path.

        Returns:
            Path to Stockfish binary or None if not found.
        """
        # Check explicit path
        if explicit_path and Path(explicit_path).exists():
            return explicit_path

        # Check environment variable
        env_path = os.environ.get("STOCKFISH_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # Check default locations
        for path in DEFAULT_STOCKFISH_PATHS:
            try:
                if Path(path).exists():
                    return path
            except Exception:
                continue

        return None

    @property
    def game_type(self) -> GameType:
        """Get the game type."""
        return GameType.CHESS

    @property
    def name(self) -> str:
        """Get the engine name."""
        if self._stockfish_available:
            return "Stockfish Chess Engine"
        return "Basic Chess Engine"

    @property
    def stockfish_available(self) -> bool:
        """Check if Stockfish is available."""
        return self._stockfish_available

    def get_initial_position(self) -> Position:
        """Get the starting chess position."""
        board = chess.Board()
        return self._board_to_position(board)

    def validate_move(
        self,
        position: Position,
        move: Move,
    ) -> MoveValidation:
        """Validate a chess move.

        Args:
            position: Current board position.
            move: Move to validate (in UCI or SAN notation).

        Returns:
            MoveValidation with result and new position if valid.
        """
        try:
            board = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHESS,
            ) from e

        # Parse move - try UCI first, then SAN
        chess_move = self._parse_move(board, move.notation)

        if chess_move is None:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Invalid move notation: {move.notation}",
            )

        # Check if move is legal
        if chess_move not in board.legal_moves:
            return MoveValidation(
                is_valid=False,
                result=MoveResult.INVALID,
                error_message=f"Illegal move: {move.notation}",
            )

        # Make the move
        board.push(chess_move)

        # Determine result
        result = self._determine_move_result(board, chess_move)
        is_game_over, winner = self._check_game_over(board)

        new_position = self._board_to_position(board)

        return MoveValidation(
            is_valid=True,
            result=result,
            new_position=new_position,
            is_game_over=is_game_over,
            winner=winner,
        )

    def get_legal_moves(self, position: Position) -> list[Move]:
        """Get all legal moves in the position."""
        try:
            board = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHESS,
            ) from e

        moves = []
        for chess_move in board.legal_moves:
            moves.append(Move(
                notation=chess_move.uci(),
                from_pos=chess.square_name(chess_move.from_square),
                to_pos=chess.square_name(chess_move.to_square),
                piece=self._get_piece_name(board, chess_move.from_square),
                promotion=chess.piece_name(chess_move.promotion) if chess_move.promotion else None,
            ))

        return moves

    def get_ai_move(
        self,
        position: Position,
        difficulty: GameDifficulty,
        time_limit_ms: int = 1000,
    ) -> AIMove:
        """Get AI's move for the position.

        Uses Stockfish if available, otherwise falls back to simple heuristics.
        """
        try:
            board = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHESS,
            ) from e

        if self._stockfish_available:
            return self._get_stockfish_move(board, difficulty, time_limit_ms)
        else:
            return self._get_fallback_move(board, difficulty)

    def _get_stockfish_move(
        self,
        board: chess.Board,
        difficulty: GameDifficulty,
        time_limit_ms: int,
    ) -> AIMove:
        """Get move from Stockfish."""
        settings = DIFFICULTY_SETTINGS[difficulty]

        try:
            with chess.engine.SimpleEngine.popen_uci(self._stockfish_path) as engine:
                # Set skill level for difficulty
                engine.configure({"Skill Level": settings["skill_level"]})

                # Calculate time limit
                time_limit = min(settings["time_limit"], time_limit_ms / 1000)

                # Get best move
                result = engine.play(
                    board,
                    chess.engine.Limit(
                        time=time_limit,
                        depth=settings["depth"],
                    ),
                )

                if result.move is None:
                    raise EngineError(
                        message="Stockfish returned no move",
                        game_type=GameType.CHESS,
                    )

                # Get evaluation
                info = engine.analyse(board, chess.engine.Limit(depth=settings["depth"]))
                score = info.get("score")
                evaluation = 0.0
                if score:
                    pov_score = score.relative
                    if pov_score.is_mate():
                        evaluation = 100.0 if pov_score.mate() > 0 else -100.0
                    else:
                        cp = pov_score.score()
                        evaluation = cp / 100 if cp else 0.0

                return AIMove(
                    move=result.move.uci(),
                    thinking_time_ms=int(time_limit * 1000),
                    evaluation=evaluation,
                    move_quality="normal",
                )

        except chess.engine.EngineTerminatedError as e:
            logger.error("Stockfish engine terminated: %s", e)
            return self._get_fallback_move(board, difficulty)
        except Exception as e:
            logger.error("Stockfish error: %s", e)
            return self._get_fallback_move(board, difficulty)

    def _get_fallback_move(
        self,
        board: chess.Board,
        difficulty: GameDifficulty,
    ) -> AIMove:
        """Get move using simple heuristics when Stockfish unavailable."""
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            raise EngineError(
                message="No legal moves available",
                game_type=GameType.CHESS,
            )

        # Simple move selection based on difficulty
        if difficulty == GameDifficulty.BEGINNER:
            # Random move
            move = random.choice(legal_moves)
        elif difficulty == GameDifficulty.EASY:
            # Prefer captures
            captures = [m for m in legal_moves if board.is_capture(m)]
            move = random.choice(captures) if captures else random.choice(legal_moves)
        else:
            # Prefer checks and captures
            checks = [m for m in legal_moves if board.gives_check(m)]
            captures = [m for m in legal_moves if board.is_capture(m)]
            preferred = checks + captures
            move = random.choice(preferred) if preferred else random.choice(legal_moves)

        return AIMove(
            move=move.uci(),
            thinking_time_ms=100,
            evaluation=0.0,
            move_quality="normal",
            commentary="(Basic engine - Stockfish not available)",
        )

    def analyze_position(
        self,
        position: Position,
        depth: int = 10,
    ) -> PositionAnalysis:
        """Analyze the current position."""
        try:
            board = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHESS,
            ) from e

        if self._stockfish_available:
            return self._analyze_with_stockfish(board, depth)
        else:
            return self._analyze_basic(board)

    def _analyze_with_stockfish(
        self,
        board: chess.Board,
        depth: int,
    ) -> PositionAnalysis:
        """Analyze position with Stockfish."""
        try:
            with chess.engine.SimpleEngine.popen_uci(self._stockfish_path) as engine:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))

                # Get evaluation
                score = info.get("score")
                evaluation = 0.0
                evaluation_text = "Equal position"

                if score:
                    pov_score = score.white()
                    if pov_score.is_mate():
                        mate_in = pov_score.mate()
                        evaluation = 100.0 if mate_in > 0 else -100.0
                        evaluation_text = f"Mate in {abs(mate_in)}"
                    else:
                        cp = pov_score.score()
                        if cp:
                            evaluation = cp / 100
                            if abs(evaluation) < 0.3:
                                evaluation_text = "Equal position"
                            elif evaluation > 0:
                                evaluation_text = f"White is better (+{evaluation:.1f})"
                            else:
                                evaluation_text = f"Black is better ({evaluation:.1f})"

                # Get best move
                best_move = None
                best_move_explanation = None
                pv = info.get("pv", [])
                if pv:
                    best_move = pv[0].uci()
                    best_move_explanation = self._explain_move(board, pv[0])

                # Get top moves
                suggested_moves = []
                try:
                    multi_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=depth),
                        multipv=3,
                    )
                    if isinstance(multi_info, list):
                        for line_info in multi_info[:3]:
                            line_pv = line_info.get("pv", [])
                            if line_pv:
                                move = line_pv[0]
                                suggested_moves.append(SuggestedMove(
                                    move=move.uci(),
                                    explanation=self._explain_move(board, move),
                                ))
                except Exception:
                    pass

                return PositionAnalysis(
                    evaluation=evaluation,
                    evaluation_text=evaluation_text,
                    best_move=best_move,
                    best_move_explanation=best_move_explanation,
                    suggested_moves=suggested_moves,
                    threats=self._find_threats(board),
                    opportunities=self._find_opportunities(board),
                    strategic_themes=self._identify_themes(board),
                )

        except Exception as e:
            logger.error("Stockfish analysis error: %s", e)
            return self._analyze_basic(board)

    def _analyze_basic(self, board: chess.Board) -> PositionAnalysis:
        """Basic position analysis without Stockfish."""
        # Simple material count
        material = self._count_material(board)
        evaluation = material / 100  # Rough centipawn to pawn conversion

        if abs(evaluation) < 1:
            evaluation_text = "Roughly equal position"
        elif evaluation > 0:
            evaluation_text = f"White has an advantage (+{evaluation:.1f})"
        else:
            evaluation_text = f"Black has an advantage ({evaluation:.1f})"

        return PositionAnalysis(
            evaluation=evaluation,
            evaluation_text=evaluation_text,
            threats=self._find_threats(board),
            opportunities=self._find_opportunities(board),
            strategic_themes=self._identify_themes(board),
        )

    def get_hint(
        self,
        position: Position,
        hint_level: int,
        previous_hints: list[str] | None = None,
    ) -> HintResponse:
        """Get a hint for the current position.

        Level 1: General strategic hint
        Level 2: More specific tactical hint
        Level 3: Reveals the best move
        """
        try:
            board = self._position_to_board(position)
        except Exception as e:
            raise InvalidPositionError(
                message=f"Invalid position: {e}",
                game_type=GameType.CHESS,
            ) from e

        analysis = self.analyze_position(position, depth=12)

        if hint_level == 1:
            # General hint based on position characteristics
            hints = self._generate_strategic_hints(board, analysis)
            hint_text = hints[0] if hints else "Look for ways to improve your pieces."
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
                # Hint about the piece to move
                best_move = self._parse_move(board, analysis.best_move)
                if best_move:
                    piece = board.piece_at(best_move.from_square)
                    if piece:
                        hint_text = f"Consider moving your {chess.piece_name(piece.piece_type)}."
                    else:
                        hint_text = "There's a strong move available."
                else:
                    hint_text = "There's a strong move available."
            else:
                hint_text = "Develop your pieces and control the center."

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
                hint_text = "No clear best move found. Choose based on your strategy."

            return HintResponse(
                hint_level=3,
                hint_text=hint_text,
                hint_type="solution",
                reveals_move=True,
            )

    def is_game_over(self, position: Position) -> tuple[bool, str | None, str | None]:
        """Check if the game is over."""
        try:
            board = self._position_to_board(position)
        except Exception:
            return False, None, None

        if board.is_checkmate():
            winner = "black" if board.turn == chess.WHITE else "white"
            return True, "checkmate", winner

        if board.is_stalemate():
            return True, "stalemate", None

        if board.is_insufficient_material():
            return True, "insufficient_material", None

        if board.is_fifty_moves():
            return True, "fifty_moves", None

        if board.is_repetition(3):
            return True, "repetition", None

        return False, None, None

    def position_to_display(self, position: Position) -> dict[str, Any]:
        """Convert position to display format."""
        try:
            board = self._position_to_board(position)
        except Exception:
            return {"error": "Invalid position"}

        # Build display data
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces.append({
                    "square": chess.square_name(square),
                    "piece": piece.symbol(),
                    "color": "white" if piece.color == chess.WHITE else "black",
                    "type": chess.piece_name(piece.piece_type),
                })

        return {
            "fen": board.fen(),
            "pieces": pieces,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "in_check": board.is_check(),
            "castling_rights": {
                "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
                "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
                "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
                "black_queenside": board.has_queenside_castling_rights(chess.BLACK),
            },
            "fullmove_number": board.fullmove_number,
            "halfmove_clock": board.halfmove_clock,
        }

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _board_to_position(self, board: chess.Board) -> Position:
        """Convert chess.Board to Position."""
        board_state = []
        for rank in range(7, -1, -1):  # 8 to 1
            row = []
            for file in range(8):  # a to h
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                row.append(piece.symbol() if piece else None)
            board_state.append(row)

        return Position(
            notation=board.fen(),
            board_state=board_state,
            metadata={
                "turn": "white" if board.turn == chess.WHITE else "black",
                "castling": board.castling_xfen(),
                "en_passant": chess.square_name(board.ep_square) if board.ep_square else None,
                "halfmove_clock": board.halfmove_clock,
                "fullmove_number": board.fullmove_number,
            },
        )

    def _position_to_board(self, position: Position) -> chess.Board:
        """Convert Position to chess.Board."""
        return chess.Board(position.notation)

    def _parse_move(self, board: chess.Board, notation: str) -> chess.Move | None:
        """Parse move notation to chess.Move."""
        # Try UCI notation first (e2e4)
        try:
            return chess.Move.from_uci(notation)
        except ValueError:
            pass

        # Try SAN notation (e4, Nf3, O-O)
        try:
            return board.parse_san(notation)
        except ValueError:
            pass

        return None

    def _determine_move_result(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> MoveResult:
        """Determine the result category of a move."""
        if board.is_checkmate():
            return MoveResult.CHECKMATE
        if board.is_stalemate():
            return MoveResult.STALEMATE
        if board.is_check():
            return MoveResult.CHECK
        if board.is_game_over():
            return MoveResult.DRAW
        return MoveResult.VALID

    def _check_game_over(self, board: chess.Board) -> tuple[bool, str | None]:
        """Check if game is over and determine winner."""
        if board.is_checkmate():
            winner = "black" if board.turn == chess.WHITE else "white"
            return True, winner
        if board.is_game_over():
            return True, None  # Draw
        return False, None

    def _get_piece_name(self, board: chess.Board, square: int) -> str | None:
        """Get the name of the piece on a square."""
        piece = board.piece_at(square)
        if piece:
            return chess.piece_name(piece.piece_type)
        return None

    def _count_material(self, board: chess.Board) -> int:
        """Count material balance in centipawns."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    total += value
                else:
                    total -= value

        return total

    def _find_threats(self, board: chess.Board) -> list[str]:
        """Find threats in the position."""
        threats = []

        # Check for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                attackers = board.attackers(board.turn, square)
                defenders = board.attackers(not board.turn, square)
                if attackers and len(attackers) > len(defenders):
                    threats.append(
                        f"The {chess.piece_name(piece.piece_type)} on "
                        f"{chess.square_name(square)} is under attack."
                    )

        return threats[:3]  # Limit to 3 threats

    def _find_opportunities(self, board: chess.Board) -> list[str]:
        """Find opportunities in the position."""
        opportunities = []

        # Check for checks
        for move in board.legal_moves:
            if board.gives_check(move):
                opportunities.append("You can give check.")
                break

        # Check for captures
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if captures:
            opportunities.append("There are pieces you can capture.")

        return opportunities[:3]

    def _identify_themes(self, board: chess.Board) -> list[str]:
        """Identify strategic themes in the position."""
        themes = []

        # Check center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        white_center = sum(
            1 for sq in center_squares
            if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
        )
        black_center = sum(
            1 for sq in center_squares
            if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
        )

        if white_center > black_center:
            themes.append("White controls the center.")
        elif black_center > white_center:
            themes.append("Black controls the center.")

        # Check king safety
        if board.fullmove_number > 10:
            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)

            if white_king_sq and chess.square_file(white_king_sq) in [0, 1, 6, 7]:
                themes.append("White king has castled.")
            if black_king_sq and chess.square_file(black_king_sq) in [0, 1, 6, 7]:
                themes.append("Black king has castled.")

        return themes[:3]

    def _generate_strategic_hints(
        self,
        board: chess.Board,
        analysis: PositionAnalysis,
    ) -> list[str]:
        """Generate strategic hints based on position."""
        hints = []

        # Opening hints
        if board.fullmove_number <= 10:
            hints.append("Focus on developing your pieces and controlling the center.")
            hints.append("Try to castle early to protect your king.")

        # Based on position characteristics
        if board.is_check():
            hints.append("You're in check! Find a way to escape.")

        if analysis.threats:
            hints.append("Watch out for threats against your pieces.")

        if not hints:
            hints.append("Look for ways to improve your worst-placed piece.")

        return hints

    def _explain_move(self, board: chess.Board, move: chess.Move) -> str:
        """Generate a brief explanation for a move."""
        explanations = []

        # Check if capture
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                explanations.append(f"captures the {chess.piece_name(captured.piece_type)}")

        # Check if gives check
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            explanations.append("gives check")
        if board_copy.is_checkmate():
            explanations.append("checkmate!")

        # Check if castling
        if board.is_castling(move):
            explanations.append("castles to safety")

        if not explanations:
            piece = board.piece_at(move.from_square)
            if piece:
                explanations.append(
                    f"moves the {chess.piece_name(piece.piece_type)} "
                    f"to {chess.square_name(move.to_square)}"
                )

        return ", ".join(explanations) if explanations else "A solid move."
