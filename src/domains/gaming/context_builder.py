# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game context builder for converting engine data to LLM context.

This module provides the GameContextBuilder class that converts raw engine
analysis and position data into structured GameCoachContext objects.
The builder handles game-specific position rendering for chess, connect4,
and future game types.

Example:
    builder = GameContextBuilder(chess_engine)
    context = builder.build_move_context(
        position_before=pos1,
        position_after=pos2,
        move="e2e4",
        player="player",
        analysis=engine_analysis,
        student=StudentContext(name="Oliver", grade_level=4),
        game_mode=GameMode.TUTORIAL,
    )
"""

import logging
from typing import Any

from src.domains.gaming.context import (
    AnalysisContext,
    CoachIntent,
    GameCoachContext,
    MoveContext,
    PositionContext,
    StudentContext,
)
from src.domains.gaming.engines.base import GameEngine
from src.domains.gaming.models import (
    GameDifficulty,
    GameMode,
    GameType,
    Position,
    PositionAnalysis,
)

logger = logging.getLogger(__name__)


class GameContextBuilder:
    """Builds GameCoachContext from engine data.

    This builder is responsible for converting raw game engine data
    (positions, analysis) into structured context for LLM coaching.
    It handles game-specific rendering of positions and moves.

    Attributes:
        engine: The game engine for this builder.

    Example:
        builder = GameContextBuilder(chess_engine)
        context = builder.build_move_context(
            position_before=pos1,
            position_after=pos2,
            move="e2e4",
            player="player",
            analysis=engine_analysis,
            student=StudentContext(name="Oliver", grade_level=4),
            game_mode=GameMode.TUTORIAL,
        )
    """

    def __init__(self, engine: GameEngine):
        """Initialize the context builder.

        Args:
            engine: Game engine for position/move operations.
        """
        self._engine = engine

    @property
    def game_type(self) -> GameType:
        """Get the game type for this builder."""
        return self._engine.game_type

    def build_move_context(
        self,
        position_before: Position,
        position_after: Position,
        move: str,
        player: str,
        analysis: PositionAnalysis,
        student: StudentContext,
        game_mode: GameMode,
        difficulty: GameDifficulty = GameDifficulty.MEDIUM,
        move_number: int = 1,
        move_quality: str | None = None,
        is_best_move: bool = False,
        session_stats: dict[str, int] | None = None,
    ) -> GameCoachContext:
        """Build context for move commentary.

        Creates a complete GameCoachContext for generating coach commentary
        about a player's move. Includes position before/after, analysis,
        and all relevant context.

        Args:
            position_before: Position before the move.
            position_after: Position after the move.
            move: Move notation.
            player: Who made the move ('player' or 'ai').
            analysis: Engine analysis of position.
            student: Student context.
            game_mode: Current game mode.
            difficulty: AI difficulty level.
            move_number: Move number in game.
            move_quality: Quality assessment (excellent/good/etc).
            is_best_move: Whether this was the best move.
            session_stats: Session statistics dict.

        Returns:
            GameCoachContext ready for LLM.
        """
        stats = session_stats or {}

        return GameCoachContext(
            intent=CoachIntent.MOVE_COMMENT,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            difficulty=difficulty.value,
            move_number=move_number,
            position_before=self._build_position_context(position_before),
            position_after=self._build_position_context(
                position_after,
                evaluation=analysis.evaluation,
                evaluation_text=analysis.evaluation_text,
            ),
            last_move=MoveContext(
                notation=move,
                player=player,
                description=self._describe_move(position_before, move),
                quality=move_quality,
                is_best=is_best_move,
            ),
            analysis=self._build_analysis_context(analysis),
            total_moves=stats.get("total_moves", 0),
            excellent_moves_count=stats.get("excellent_moves", 0),
            mistakes_count=stats.get("mistakes", 0),
            hints_used=stats.get("hints_used", 0),
        )

    def build_ai_move_context(
        self,
        position_before: Position,
        position_after: Position,
        ai_move: str,
        analysis: PositionAnalysis,
        student: StudentContext,
        game_mode: GameMode,
        difficulty: GameDifficulty = GameDifficulty.MEDIUM,
        move_number: int = 1,
    ) -> GameCoachContext:
        """Build context for AI move explanation.

        Creates context specifically for explaining why the AI made
        a particular move. Used in tutorial mode to help students
        understand AI strategy.

        Args:
            position_before: Position before AI's move.
            position_after: Position after AI's move.
            ai_move: AI's move notation.
            analysis: Engine analysis.
            student: Student context.
            game_mode: Current game mode.
            difficulty: AI difficulty level.
            move_number: Move number in game.

        Returns:
            GameCoachContext for AI move explanation.
        """
        ai_move_reason = self._explain_ai_move(position_before, ai_move, analysis)

        return GameCoachContext(
            intent=CoachIntent.AI_MOVE_EXPLAIN,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            difficulty=difficulty.value,
            move_number=move_number,
            position_before=self._build_position_context(position_before),
            position_after=self._build_position_context(
                position_after,
                evaluation=analysis.evaluation,
                evaluation_text=analysis.evaluation_text,
            ),
            ai_move=MoveContext(
                notation=ai_move,
                player="ai",
                description=self._describe_move(position_before, ai_move),
            ),
            ai_move_reason=ai_move_reason,
            analysis=self._build_analysis_context(analysis),
        )

    def build_greeting_context(
        self,
        student: StudentContext,
        game_mode: GameMode,
        difficulty: GameDifficulty,
    ) -> GameCoachContext:
        """Build context for game start greeting.

        Args:
            student: Student context.
            game_mode: Game mode.
            difficulty: AI difficulty.

        Returns:
            GameCoachContext for greeting.
        """
        return GameCoachContext(
            intent=CoachIntent.GREETING,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            difficulty=difficulty.value,
            move_number=0,
        )

    def build_hint_context(
        self,
        position: Position,
        analysis: PositionAnalysis,
        student: StudentContext,
        game_mode: GameMode,
        hint_level: int,
        previous_hints: list[str] | None = None,
    ) -> GameCoachContext:
        """Build context for hint generation.

        Args:
            position: Current position.
            analysis: Engine analysis.
            student: Student context.
            game_mode: Game mode.
            hint_level: Hint level (1-3).
            previous_hints: Previous hints this turn.

        Returns:
            GameCoachContext for hint.
        """
        return GameCoachContext(
            intent=CoachIntent.HINT,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            position_after=self._build_position_context(
                position,
                evaluation=analysis.evaluation,
                evaluation_text=analysis.evaluation_text,
            ),
            analysis=self._build_analysis_context(analysis),
            hint_level=hint_level,
            previous_hints=previous_hints or [],
        )

    def build_game_end_context(
        self,
        position: Position,
        student: StudentContext,
        game_mode: GameMode,
        game_result: str,
        result_reason: str,
        session_stats: dict[str, int],
    ) -> GameCoachContext:
        """Build context for game end message.

        Args:
            position: Final position.
            student: Student context.
            game_mode: Game mode.
            game_result: Result (win/loss/draw).
            result_reason: How game ended.
            session_stats: Session statistics.

        Returns:
            GameCoachContext for game end.
        """
        return GameCoachContext(
            intent=CoachIntent.GAME_END,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            position_after=self._build_position_context(position),
            game_result=game_result,
            result_reason=result_reason,
            total_moves=session_stats.get("total_moves", 0),
            excellent_moves_count=session_stats.get("excellent_moves", 0),
            mistakes_count=session_stats.get("mistakes", 0),
            hints_used=session_stats.get("hints_used", 0),
        )

    def build_invalid_move_context(
        self,
        position: Position,
        invalid_move: str,
        invalid_reason: str,
        student: StudentContext,
    ) -> GameCoachContext:
        """Build context for invalid move feedback.

        Args:
            position: Current position.
            invalid_move: The invalid move attempted.
            invalid_reason: Why it's invalid.
            student: Student context.

        Returns:
            GameCoachContext for invalid move.
        """
        return GameCoachContext(
            intent=CoachIntent.INVALID_MOVE,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode="practice",
            position_after=self._build_position_context(position),
            invalid_move=invalid_move,
            invalid_reason=invalid_reason,
        )

    def build_analysis_context(
        self,
        position: Position,
        analysis: PositionAnalysis,
        student: StudentContext,
        game_mode: GameMode,
        session_stats: dict[str, int],
        critical_moments: list[dict[str, Any]] | None = None,
    ) -> GameCoachContext:
        """Build context for full game analysis.

        Args:
            position: Final position.
            analysis: Position analysis.
            student: Student context.
            game_mode: Game mode.
            session_stats: Session statistics.
            critical_moments: Notable moments in the game.

        Returns:
            GameCoachContext for analysis.
        """
        return GameCoachContext(
            intent=CoachIntent.ANALYSIS,
            student=student,
            game_type=self._engine.game_type.value,
            game_mode=game_mode.value,
            position_after=self._build_position_context(
                position,
                evaluation=analysis.evaluation,
                evaluation_text=analysis.evaluation_text,
            ),
            analysis=self._build_analysis_context(analysis),
            total_moves=session_stats.get("total_moves", 0),
            excellent_moves_count=session_stats.get("excellent_moves", 0),
            mistakes_count=session_stats.get("mistakes", 0),
            hints_used=session_stats.get("hints_used", 0),
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _build_position_context(
        self,
        position: Position,
        evaluation: float | None = None,
        evaluation_text: str | None = None,
    ) -> PositionContext:
        """Build position context from Position.

        Args:
            position: The position.
            evaluation: Optional evaluation.
            evaluation_text: Optional evaluation text.

        Returns:
            PositionContext.
        """
        display = self._engine.position_to_display(position)

        return PositionContext(
            text_representation=self._position_to_text(position),
            description=self._describe_position(position),
            evaluation=evaluation,
            evaluation_text=evaluation_text,
            is_check=display.get("in_check", False),
            is_game_over=False,
        )

    def _build_analysis_context(
        self,
        analysis: PositionAnalysis,
    ) -> AnalysisContext:
        """Build analysis context from PositionAnalysis.

        Args:
            analysis: Engine analysis.

        Returns:
            AnalysisContext.
        """
        alternatives = []
        for suggested in analysis.suggested_moves or []:
            alternatives.append({
                "move": suggested.move,
                "reason": suggested.explanation or "",
            })

        return AnalysisContext(
            best_move=analysis.best_move,
            best_move_reason=analysis.best_move_explanation,
            threats=analysis.threats or [],
            opportunities=analysis.opportunities or [],
            strategic_themes=analysis.strategic_themes or [],
            alternative_moves=alternatives,
        )

    def _position_to_text(self, position: Position) -> str:
        """Convert position to text for LLM.

        Delegates to game-specific method based on game type.

        Args:
            position: The position.

        Returns:
            Text representation.
        """
        if self._engine.game_type == GameType.CHESS:
            return self._chess_position_to_text(position)
        elif self._engine.game_type == GameType.CONNECT4:
            return self._connect4_position_to_text(position)
        else:
            return str(position.notation)

    def _chess_position_to_text(self, position: Position) -> str:
        """Convert chess position to text.

        Args:
            position: Chess position.

        Returns:
            FEN with ASCII board representation.
        """
        board_state = position.board_state
        if not board_state:
            return f"FEN: {position.notation}"

        lines = [f"FEN: {position.notation}", "", "Board:"]
        lines.append("  a b c d e f g h")

        for rank_idx, rank in enumerate(board_state):
            rank_num = 8 - rank_idx
            row = []
            for piece in rank:
                if piece is None:
                    row.append(".")
                else:
                    row.append(piece)
            lines.append(f"{rank_num} {' '.join(row)}")

        return "\n".join(lines)

    def _connect4_position_to_text(self, position: Position) -> str:
        """Convert Connect4 position to text.

        Args:
            position: Connect4 position.

        Returns:
            ASCII grid representation.
        """
        grid = position.board_state
        if not grid:
            return "Empty board"

        lines = ["Grid (R=Red, Y=Yellow, .=Empty):", ""]

        for row in grid:
            row_chars = []
            for cell in row:
                if cell == "red":
                    row_chars.append("R")
                elif cell == "yellow":
                    row_chars.append("Y")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))

        lines.append("")
        lines.append("1 2 3 4 5 6 7")

        return "\n".join(lines)

    def _describe_position(self, position: Position) -> str:
        """Generate human-readable position description.

        Args:
            position: The position.

        Returns:
            Description string.
        """
        if self._engine.game_type == GameType.CHESS:
            return self._describe_chess_position(position)
        elif self._engine.game_type == GameType.CONNECT4:
            return self._describe_connect4_position(position)
        else:
            return ""

    def _describe_chess_position(self, position: Position) -> str:
        """Describe chess position in natural language."""
        display = self._engine.position_to_display(position)

        parts = []

        turn = display.get("turn", "white")
        parts.append(f"{turn.title()} to move.")

        if display.get("in_check"):
            parts.append("King is in check!")

        castling = display.get("castling_rights", {})
        castling_str = []
        if castling.get("white_kingside"):
            castling_str.append("White can castle kingside")
        if castling.get("white_queenside"):
            castling_str.append("White can castle queenside")
        if castling.get("black_kingside"):
            castling_str.append("Black can castle kingside")
        if castling.get("black_queenside"):
            castling_str.append("Black can castle queenside")

        if castling_str:
            parts.append("; ".join(castling_str) + ".")

        return " ".join(parts)

    def _describe_connect4_position(self, position: Position) -> str:
        """Describe Connect4 position in natural language."""
        grid = position.board_state
        if not grid:
            return "Empty board"

        red_count = 0
        yellow_count = 0

        for row in grid:
            for cell in row:
                if cell == "red":
                    red_count += 1
                elif cell == "yellow":
                    yellow_count += 1

        return f"Red has {red_count} pieces, Yellow has {yellow_count} pieces."

    def _describe_move(self, position: Position, move: str) -> str:
        """Generate human-readable move description.

        Args:
            position: Position before move.
            move: Move notation.

        Returns:
            Move description.
        """
        if self._engine.game_type == GameType.CHESS:
            return self._describe_chess_move(position, move)
        elif self._engine.game_type == GameType.CONNECT4:
            return self._describe_connect4_move(position, move)
        else:
            return f"Move: {move}"

    def _describe_chess_move(self, position: Position, move: str) -> str:
        """Describe chess move in natural language.

        Args:
            position: Position before move.
            move: UCI notation move.

        Returns:
            Description like "Knight from g1 to f3".
        """
        if len(move) < 4:
            return f"Move: {move}"

        from_sq = move[:2]
        to_sq = move[2:4]

        piece_name = self._get_piece_at(position, from_sq)
        captured = self._get_piece_at(position, to_sq)

        parts = [f"{piece_name} from {from_sq} to {to_sq}"]

        if captured:
            parts.append(f"capturing the {captured}")

        if len(move) > 4:
            promo_pieces = {"q": "Queen", "r": "Rook", "b": "Bishop", "n": "Knight"}
            promo = promo_pieces.get(move[4].lower(), move[4])
            parts.append(f"promoting to {promo}")

        return ", ".join(parts)

    def _describe_connect4_move(self, position: Position, move: str) -> str:
        """Describe Connect4 move.

        Args:
            position: Position before move.
            move: Column number.

        Returns:
            Description.
        """
        try:
            col = int(move)
            return f"Piece dropped in column {col}"
        except ValueError:
            return f"Move: {move}"

    def _get_piece_at(self, position: Position, square: str) -> str:
        """Get piece name at a square.

        Args:
            position: The position.
            square: Square notation (e.g., 'e4').

        Returns:
            Piece name or "Piece".
        """
        board_state = position.board_state
        if not board_state:
            return "Piece"

        try:
            file_idx = ord(square[0]) - ord('a')
            rank_idx = 8 - int(square[1])

            if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                piece = board_state[rank_idx][file_idx]
                if piece:
                    piece_names = {
                        'K': 'King', 'Q': 'Queen', 'R': 'Rook',
                        'B': 'Bishop', 'N': 'Knight', 'P': 'Pawn',
                        'k': 'King', 'q': 'Queen', 'r': 'Rook',
                        'b': 'Bishop', 'n': 'Knight', 'p': 'Pawn',
                    }
                    return piece_names.get(piece, "Piece")
        except (IndexError, ValueError):
            pass

        return "Piece"

    def _explain_ai_move(
        self,
        position: Position,
        move: str,
        analysis: PositionAnalysis,
    ) -> str:
        """Generate explanation for why AI made this move.

        Args:
            position: Position before move.
            move: AI's move.
            analysis: Engine analysis.

        Returns:
            Explanation string.
        """
        parts = []

        if analysis.best_move_explanation:
            parts.append(analysis.best_move_explanation)

        if analysis.threats:
            parts.append(f"This addresses: {analysis.threats[0]}")

        if analysis.strategic_themes:
            parts.append(f"Theme: {analysis.strategic_themes[0]}")

        if parts:
            return " ".join(parts)

        return "This develops my position."
