# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game coach context model for LLM interactions.

This module provides the GameCoachContext model that captures all game state
information needed for intelligent LLM-based coaching. The design is
game-agnostic, supporting chess, connect4, and future game types.

The context model is used by GameCoachAgent to build prompts that include:
- Full board state (before and after moves)
- Engine analysis (threats, opportunities, best moves)
- Student personalization (name, grade, language)
- Game mode awareness (tutorial/practice/challenge)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CoachIntent(str, Enum):
    """Intent for coach message generation.

    Determines what type of coaching message is needed.
    """

    GREETING = "greeting"
    MOVE_COMMENT = "move_comment"
    AI_MOVE_EXPLAIN = "ai_move_explain"
    HINT = "hint"
    INVALID_MOVE = "invalid_move"
    GAME_END = "game_end"
    ANALYSIS = "analysis"
    ENCOURAGEMENT = "encouragement"


class StudentContext(BaseModel):
    """Student information for personalization.

    Attributes:
        name: Student's first name.
        grade_level: Student's grade level (1-12).
        language: Preferred language code.
        player_color: Player's side in the game.
    """

    name: str = Field(description="Student's first name")
    grade_level: int = Field(description="Student's grade level (1-12)")
    language: str = Field(default="en", description="Preferred language")
    player_color: str = Field(default="white", description="Player's side in the game")


class MoveContext(BaseModel):
    """Context about a single move.

    Attributes:
        notation: Move in standard notation (e.g., 'e2e4', '3').
        player: Who made the move: 'player' or 'ai'.
        description: Human-readable move description.
        quality: Move quality assessment.
        is_best: Whether this was the best move.
    """

    notation: str = Field(description="Move in standard notation")
    player: str = Field(description="Who made the move: 'player' or 'ai'")
    description: str | None = Field(
        default=None,
        description="Human-readable move description",
    )
    quality: str | None = Field(
        default=None,
        description="Move quality: excellent, good, inaccuracy, mistake, blunder",
    )
    is_best: bool = Field(default=False, description="Whether this was the best move")


class PositionContext(BaseModel):
    """Context about a board position.

    Attributes:
        text_representation: Text representation of position.
        description: Human-readable position description.
        evaluation: Position evaluation score.
        evaluation_text: Human-readable evaluation.
        is_check: King in check (chess).
        is_game_over: Game has ended.
    """

    text_representation: str = Field(
        description="Text representation of position (FEN for chess, grid for connect4)",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable position description",
    )
    evaluation: float | None = Field(
        default=None,
        description="Position evaluation (positive = student advantage)",
    )
    evaluation_text: str | None = Field(
        default=None,
        description="Human-readable evaluation",
    )
    is_check: bool = Field(default=False, description="King in check (chess)")
    is_game_over: bool = Field(default=False, description="Game has ended")


class AnalysisContext(BaseModel):
    """Strategic analysis of the position.

    Attributes:
        best_move: Engine's best move.
        best_move_reason: Why best_move is best.
        threats: Current threats in position.
        opportunities: Opportunities available.
        strategic_themes: Strategic themes.
        alternative_moves: Alternative good moves.
        material_balance: Material count.
    """

    best_move: str | None = Field(default=None, description="Engine's best move")
    best_move_reason: str | None = Field(
        default=None,
        description="Why best_move is best",
    )
    threats: list[str] = Field(
        default_factory=list,
        description="Current threats in position",
    )
    opportunities: list[str] = Field(
        default_factory=list,
        description="Opportunities available",
    )
    strategic_themes: list[str] = Field(
        default_factory=list,
        description="Strategic themes",
    )
    alternative_moves: list[dict[str, str]] = Field(
        default_factory=list,
        description="Alternative good moves: [{'move': 'Bc4', 'reason': '...'}]",
    )
    material_balance: str | None = Field(
        default=None,
        description="Material count",
    )


class GameCoachContext(BaseModel):
    """Complete context for game coaching LLM interactions.

    This is the primary context model passed to the GameCoachAgent.
    It aggregates all information needed for intelligent coaching.

    Design Principles:
    - Game-agnostic: Works for chess, connect4, or any future game
    - Position-aware: Includes before/after positions
    - Analysis-rich: Includes engine analysis data
    - Mode-aware: Different handling for tutorial/practice/challenge

    Attributes:
        intent: What type of coach response is needed.
        student: Student information.
        game_type: Type of game.
        game_mode: Game mode.
        difficulty: AI difficulty level.
        move_number: Current move number.
        position_before: Position before the move.
        position_after: Position after the move.
        last_move: The most recent move.
        ai_move: AI's move to explain.
        ai_move_reason: Strategic reason for AI's move.
        analysis: Engine analysis of the position.
        game_result: Result if game ended.
        result_reason: How game ended.
        total_moves: Total moves in session.
        excellent_moves_count: Count of excellent moves.
        mistakes_count: Count of mistakes/blunders.
        hints_used: Number of hints used.
        hint_level: Hint specificity.
        previous_hints: Previous hints given this turn.
        invalid_move: The invalid move attempted.
        invalid_reason: Why the move is invalid.

    Example:
        context = GameCoachContext(
            intent=CoachIntent.MOVE_COMMENT,
            student=StudentContext(name="Oliver", grade_level=4),
            game_type="chess",
            game_mode="tutorial",
            move_number=5,
            last_move=MoveContext(notation="e2e4", player="player"),
            position_before=PositionContext(text_representation="..."),
            position_after=PositionContext(text_representation="..."),
            analysis=AnalysisContext(best_move="e2e4", threats=[...]),
        )
    """

    # === INTENT ===
    intent: CoachIntent = Field(
        description="What type of coach response is needed",
    )

    # === STUDENT CONTEXT ===
    student: StudentContext = Field(description="Student information")

    # === GAME CONTEXT ===
    game_type: str = Field(description="Type of game: 'chess', 'connect4', etc.")
    game_mode: str = Field(
        description="Game mode: 'tutorial', 'practice', 'challenge'"
    )
    difficulty: str = Field(default="medium", description="AI difficulty level")
    move_number: int = Field(default=1, description="Current move number")

    # === POSITION CONTEXT ===
    position_before: PositionContext | None = Field(
        default=None,
        description="Position before the move",
    )
    position_after: PositionContext | None = Field(
        default=None,
        description="Position after the move",
    )

    # === MOVE CONTEXT ===
    last_move: MoveContext | None = Field(
        default=None,
        description="The most recent move",
    )

    # === AI MOVE CONTEXT (for ai_move_explain intent) ===
    ai_move: MoveContext | None = Field(
        default=None,
        description="AI's move to explain",
    )
    ai_move_reason: str | None = Field(
        default=None,
        description="Strategic reason for AI's move",
    )

    # === ANALYSIS CONTEXT ===
    analysis: AnalysisContext | None = Field(
        default=None,
        description="Engine analysis of the position",
    )

    # === GAME STATE ===
    game_result: str | None = Field(
        default=None,
        description="Result if game ended: 'win', 'loss', 'draw'",
    )
    result_reason: str | None = Field(
        default=None,
        description="How game ended: 'checkmate', 'resignation', etc.",
    )

    # === SESSION STATS ===
    total_moves: int = Field(default=0, description="Total moves in session")
    excellent_moves_count: int = Field(
        default=0, description="Count of excellent moves"
    )
    mistakes_count: int = Field(default=0, description="Count of mistakes/blunders")
    hints_used: int = Field(default=0, description="Number of hints used")

    # === HINT CONTEXT (for hint intent) ===
    hint_level: int | None = Field(
        default=None,
        description="Hint specificity: 1=vague, 2=specific, 3=solution",
    )
    previous_hints: list[str] = Field(
        default_factory=list,
        description="Previous hints given this turn",
    )

    # === INVALID MOVE CONTEXT ===
    invalid_move: str | None = Field(
        default=None,
        description="The invalid move attempted",
    )
    invalid_reason: str | None = Field(
        default=None,
        description="Why the move is invalid",
    )

    def to_runtime_context(self) -> dict[str, Any]:
        """Convert to runtime context dict for YAML prompt interpolation.

        This method transforms the structured GameCoachContext into a flat
        dictionary suitable for SystemPromptBuilder's variable interpolation.
        It includes 'has_*' boolean flags for conditional section rendering.

        Returns:
            Dictionary with all context values for prompt building.
        """
        ctx: dict[str, Any] = {
            # Student
            "student_name": self.student.name,
            "grade_level": self.student.grade_level,
            "language": self.student.language,
            "player_color": self.student.player_color,
            # Game
            "game_type": self.game_type,
            "game_mode": self.game_mode,
            "difficulty": self.difficulty,
            "move_number": self.move_number,
            # Stats
            "total_moves": self.total_moves,
            "excellent_moves_count": self.excellent_moves_count,
            "mistakes_count": self.mistakes_count,
            "hints_used": self.hints_used,
        }

        # Position context
        if self.position_before:
            ctx["position_before"] = self.position_before.text_representation
            ctx["has_position_before"] = True
        else:
            ctx["has_position_before"] = False

        if self.position_after:
            ctx["position_after"] = self.position_after.text_representation
            ctx["position_description"] = self.position_after.description
            ctx["evaluation"] = self.position_after.evaluation
            ctx["evaluation_text"] = self.position_after.evaluation_text
            ctx["is_check"] = self.position_after.is_check

        # Move context
        if self.last_move:
            ctx["last_move"] = self.last_move.notation
            ctx["last_move_player"] = self.last_move.player
            ctx["last_move_description"] = self.last_move.description
            ctx["move_quality"] = self.last_move.quality
            ctx["has_last_move"] = True
        else:
            ctx["has_last_move"] = False

        # AI move context
        if self.ai_move:
            ctx["ai_move"] = self.ai_move.notation
            ctx["ai_move_description"] = self.ai_move.description
            ctx["ai_move_reason"] = self.ai_move_reason
            ctx["has_ai_move"] = True
        else:
            ctx["has_ai_move"] = False

        # Analysis context
        if self.analysis:
            ctx["best_move"] = self.analysis.best_move
            ctx["best_move_reason"] = self.analysis.best_move_reason
            ctx["threats"] = self.analysis.threats
            ctx["opportunities"] = self.analysis.opportunities
            ctx["strategic_themes"] = self.analysis.strategic_themes
            ctx["alternative_moves"] = self.analysis.alternative_moves
            ctx["material_balance"] = self.analysis.material_balance
            ctx["has_threats"] = len(self.analysis.threats) > 0
            ctx["has_opportunities"] = len(self.analysis.opportunities) > 0
            ctx["has_alternatives"] = len(self.analysis.alternative_moves) > 0

            # Check if best move differs from played move
            if self.last_move and self.analysis.best_move:
                ctx["has_best_move_different"] = (
                    self.last_move.notation != self.analysis.best_move
                )
            else:
                ctx["has_best_move_different"] = False
        else:
            ctx["has_threats"] = False
            ctx["has_opportunities"] = False
            ctx["has_alternatives"] = False
            ctx["has_best_move_different"] = False

        # Game end context
        if self.game_result:
            ctx["game_result"] = self.game_result
            ctx["result_reason"] = self.result_reason
            ctx["has_game_result"] = True
        else:
            ctx["has_game_result"] = False

        # Hint context
        if self.hint_level:
            ctx["hint_level"] = self.hint_level
            ctx["previous_hints"] = self.previous_hints

        # Invalid move context
        if self.invalid_move:
            ctx["invalid_move"] = self.invalid_move
            ctx["invalid_reason"] = self.invalid_reason

        return ctx

    def get_capability_params(self) -> dict[str, Any]:
        """Convert to capability parameters dictionary.

        Returns:
            Dictionary suitable for capability parameter validation.
        """
        params: dict[str, Any] = {
            "game_type": self.game_type,
            "game_mode": self.game_mode,
            "student_name": self.student.name,
            "grade_level": self.student.grade_level,
            "player_color": self.student.player_color,
            "language": self.student.language,
            "move_number": self.move_number,
        }

        if self.last_move:
            params["move_notation"] = self.last_move.notation
            params["move_player"] = self.last_move.player
            params["move_quality"] = self.last_move.quality
        elif self.ai_move:
            # For AI move explanation, use ai_move as the move
            params["move_notation"] = self.ai_move.notation
            params["move_player"] = self.ai_move.player

        if self.position_before:
            params["position_before"] = self.position_before.text_representation

        if self.position_after:
            params["position_after"] = self.position_after.text_representation
            params["position_description"] = self.position_after.description
            params["evaluation"] = self.position_after.evaluation

        if self.analysis:
            params["best_move"] = self.analysis.best_move
            params["best_move_reason"] = self.analysis.best_move_reason
            params["threats"] = self.analysis.threats
            params["opportunities"] = self.analysis.opportunities
            params["strategic_themes"] = self.analysis.strategic_themes
            params["alternative_moves"] = self.analysis.alternative_moves

        if self.ai_move:
            params["ai_move"] = self.ai_move.notation
            params["ai_move_description"] = self.ai_move.description
            params["ai_move_reason"] = self.ai_move_reason

        return params
