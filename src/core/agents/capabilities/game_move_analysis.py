# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game move analysis capability for coach feedback generation.

This capability generates coach commentary about a game move:
- Explains move quality and implications
- Provides age-appropriate feedback
- Adapts to game mode (tutorial/practice/challenge)
- Follows encouraging coaching style

The actual move analysis (evaluation, quality scoring) is done by the
game engine. This capability focuses on generating appropriate LLM
responses based on that analysis data.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class GameMode(str, Enum):
    """Game mode affects how verbose the coach is."""

    TUTORIAL = "tutorial"
    PRACTICE = "practice"
    CHALLENGE = "challenge"
    PUZZLE = "puzzle"
    ANALYSIS = "analysis"


class MoveQuality(str, Enum):
    """Quality classification of a move."""

    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class MoveAnalysisParams(BaseModel):
    """Parameters for move analysis commentary generation.

    Attributes:
        game_type: Type of game (chess, connect4).
        game_mode: Current game mode.
        move_player: Who made the move (player or ai).
        move_notation: The move in standard notation.
        move_number: Move number in the game.
        move_quality: Quality assessment from engine.
        evaluation: Position evaluation after move.
        best_move: What the best move would have been.
        best_move_reason: Why best_move is best.
        threats: Current threats in the position.
        opportunities: Opportunities available.
        strategic_themes: Strategic themes in the position.
        student_name: Student's name for personalization.
        grade_level: Student's grade for age-appropriate language.
        player_color: What color/side the student plays.
        is_winning: Whether the move leads to a winning sequence.
        is_game_over: Whether this move ends the game.
        game_result: Result if game ended (win/loss/draw).
        language: Language for the response.
        position_before: Text representation of position before move.
        position_after: Text representation of position after move.
        position_description: Human-readable position description.
        move_description: Human-readable move description.
        ai_move: AI's move notation (when explaining AI move).
        ai_move_reason: Reason for AI's move.
        ai_move_description: Human-readable AI move description.
        alternative_moves: Alternative good moves with reasons.
    """

    game_type: str = Field(description="Type of game (chess, connect4)")
    game_mode: GameMode = Field(description="Current game mode")
    move_player: str = Field(description="Who made the move (player or ai)")
    move_notation: str = Field(description="The move in standard notation")
    move_number: int = Field(description="Move number in the game")
    move_quality: MoveQuality | None = Field(
        default=None,
        description="Quality assessment from engine",
    )
    evaluation: float | None = Field(
        default=None,
        description="Position evaluation (positive = student advantage)",
    )
    best_move: str | None = Field(
        default=None,
        description="What the best move would have been",
    )
    best_move_reason: str | None = Field(
        default=None,
        description="Why best_move is the best option",
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
        description="Strategic themes in position",
    )
    student_name: str = Field(description="Student's name")
    grade_level: int = Field(description="Student's grade level")
    player_color: str = Field(description="Student's color/side")
    is_winning: bool = Field(default=False, description="Move leads to win")
    is_game_over: bool = Field(default=False, description="Game ended with this move")
    game_result: str | None = Field(default=None, description="Result if game ended")
    language: str = Field(default="en", description="Response language")

    # Position context (new fields for v2.0)
    position_before: str | None = Field(
        default=None,
        description="Text representation of position before move",
    )
    position_after: str | None = Field(
        default=None,
        description="Text representation of position after move",
    )
    position_description: str | None = Field(
        default=None,
        description="Human-readable position description",
    )
    move_description: str | None = Field(
        default=None,
        description="Human-readable move description (e.g., 'Knight from g1 to f3')",
    )

    # AI move context (for ai_move_explain intent)
    ai_move: str | None = Field(
        default=None,
        description="AI's move notation (when explaining AI move)",
    )
    ai_move_reason: str | None = Field(
        default=None,
        description="Strategic reason for AI's move",
    )
    ai_move_description: str | None = Field(
        default=None,
        description="Human-readable AI move description",
    )

    # Alternative moves
    alternative_moves: list[dict[str, str]] = Field(
        default_factory=list,
        description="Alternative good moves: [{'move': 'Bc4', 'reason': '...'}]",
    )


class MoveAnalysisResult(CapabilityResult):
    """Result of move analysis commentary generation.

    Attributes:
        coach_message: The coach's commentary about the move.
        move_quality_explained: Plain-language quality explanation.
        tactical_tip: Optional tactical tip for the student.
        encouragement: Optional encouragement message.
        learning_point: Optional learning point from the move.
    """

    coach_message: str = Field(description="Coach's commentary about the move")
    move_quality_explained: str | None = Field(
        default=None,
        description="Plain-language quality explanation",
    )
    tactical_tip: str | None = Field(
        default=None,
        description="Optional tactical tip",
    )
    encouragement: str | None = Field(
        default=None,
        description="Optional encouragement message",
    )
    learning_point: str | None = Field(
        default=None,
        description="Optional learning point from the move",
    )


class GameMoveAnalysisCapability(Capability):
    """Capability for generating coach commentary about game moves.

    Generates age-appropriate, encouraging feedback based on move
    quality and game context. Uses engine analysis data to provide
    accurate but friendly coaching messages.

    Example:
        capability = GameMoveAnalysisCapability()
        params = MoveAnalysisParams(
            game_type="chess",
            game_mode=GameMode.PRACTICE,
            move_player="player",
            move_notation="e2e4",
            move_number=1,
            move_quality=MoveQuality.GOOD,
            student_name="Ali",
            grade_level=5,
            player_color="white",
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        result = capability.parse_response(llm_response)
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "game_move_analysis"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates coach commentary about game moves"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate move analysis parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            MoveAnalysisParams(**params)
        except Exception as e:
            raise CapabilityError(
                message=f"Invalid parameters: {e}",
                capability_name=self.name,
                original_error=e,
            ) from e

    def build_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> list[dict[str, str]]:
        """Build prompt for move commentary generation.

        Args:
            params: Move analysis parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = MoveAnalysisParams(**params)

        system_parts = []

        system_parts.append(self._get_base_instruction(p))
        system_parts.append(self._get_mode_instruction(p.game_mode))
        system_parts.append(self._get_age_adaptation(p.grade_level))

        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        system_message = "\n\n".join(filter(None, system_parts))

        user_parts = []

        # Check if this is an AI move explanation
        if p.ai_move:
            user_parts.append(
                f"Explain why you (the AI) made this {p.game_type} move:\n"
            )
            user_parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")
            user_parts.append(f"- AI move: {p.ai_move}")
            if p.ai_move_description:
                user_parts.append(f"- Move description: {p.ai_move_description}")
            if p.ai_move_reason:
                user_parts.append(f"- Strategic reason: {p.ai_move_reason}")
            user_parts.append(
                "\nSpeak as if you ARE the opponent explaining your thinking."
            )
            user_parts.append("Use first person: 'I played...', 'My plan is...'")
        else:
            user_parts.append(
                f"Generate a coach comment for this {p.game_type} move:\n"
            )
            user_parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")
            user_parts.append(f"- Playing as: {p.player_color}")
            user_parts.append(f"- Move #{p.move_number}: {p.move_notation}")
            if p.move_description:
                user_parts.append(f"- Move description: {p.move_description}")
            user_parts.append(f"- Move by: {p.move_player}")

            if p.move_quality:
                user_parts.append(f"- Move quality: {p.move_quality.value}")

        # Position context
        if p.position_before:
            user_parts.append(f"\n**Position Before:**\n{p.position_before}")

        if p.position_after:
            user_parts.append(f"\n**Position After:**\n{p.position_after}")

        if p.position_description:
            user_parts.append(f"- Description: {p.position_description}")

        if p.evaluation is not None:
            eval_desc = self._describe_evaluation(p.evaluation)
            user_parts.append(f"- Position: {eval_desc}")

        # Best move context
        if p.best_move and p.move_quality in [MoveQuality.MISTAKE, MoveQuality.BLUNDER]:
            user_parts.append(f"\n**Better Option:**")
            user_parts.append(f"- Best move: {p.best_move}")
            if p.best_move_reason:
                user_parts.append(f"- Reason: {p.best_move_reason}")

        # Analysis context
        if p.threats:
            user_parts.append(f"- Threats: {', '.join(p.threats[:3])}")

        if p.opportunities:
            user_parts.append(f"- Opportunities: {', '.join(p.opportunities[:3])}")

        if p.strategic_themes:
            user_parts.append(f"- Themes: {', '.join(p.strategic_themes[:3])}")

        # Alternatives (for tutorial mode)
        if p.alternative_moves and p.game_mode == GameMode.TUTORIAL:
            alts = []
            for alt in p.alternative_moves[:2]:
                if isinstance(alt, dict):
                    move = alt.get("move", "")
                    reason = alt.get("reason", "")
                    alts.append(f"{move}: {reason}")
            if alts:
                user_parts.append(f"\n**Alternatives:** {'; '.join(alts)}")

        if p.is_game_over:
            user_parts.append(f"- Game ended: {p.game_result}")

        user_parts.append(f"\nLanguage: {p.language}")
        user_parts.append("Keep the comment SHORT (2-3 sentences max).")

        user_parts.append(self._get_output_format())

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def build_user_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> str:
        """Build user prompt for YAML-driven system prompt mode.

        Args:
            params: Move analysis parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            User prompt string.
        """
        self.validate_params(params)
        p = MoveAnalysisParams(**params)

        parts = []

        # Check if this is an AI move explanation
        if p.ai_move:
            parts.append(f"Explain why you (the AI) made this {p.game_type} move:\n")
            parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")
            parts.append(f"- AI move: {p.ai_move}")
            if p.ai_move_description:
                parts.append(f"- Move description: {p.ai_move_description}")
            if p.ai_move_reason:
                parts.append(f"- Strategic reason: {p.ai_move_reason}")
            parts.append("\nSpeak as if you ARE the opponent explaining your thinking.")
            parts.append("Use first person: 'I played...', 'My plan is...'")
        else:
            parts.append(f"Generate a coach comment for this {p.game_type} move:\n")
            parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")
            parts.append(f"- Playing as: {p.player_color}")
            parts.append(f"- Move #{p.move_number}: {p.move_notation}")
            if p.move_description:
                parts.append(f"- Move description: {p.move_description}")
            parts.append(f"- Move by: {p.move_player}")

            if p.move_quality:
                parts.append(f"- Move quality: {p.move_quality.value}")

        # Position context
        if p.position_before:
            parts.append(f"\n**Position Before:**\n{p.position_before}")

        if p.position_after:
            parts.append(f"\n**Position After:**\n{p.position_after}")

        if p.position_description:
            parts.append(f"- Description: {p.position_description}")

        if p.evaluation is not None:
            eval_desc = self._describe_evaluation(p.evaluation)
            parts.append(f"- Position: {eval_desc}")

        # Best move context
        if p.best_move and p.move_quality in [MoveQuality.MISTAKE, MoveQuality.BLUNDER]:
            parts.append(f"\n**Better Option:**")
            parts.append(f"- Best move: {p.best_move}")
            if p.best_move_reason:
                parts.append(f"- Reason: {p.best_move_reason}")

        # Analysis context
        if p.threats:
            parts.append(f"- Threats: {', '.join(p.threats[:3])}")

        if p.opportunities:
            parts.append(f"- Opportunities: {', '.join(p.opportunities[:3])}")

        if p.strategic_themes:
            parts.append(f"- Themes: {', '.join(p.strategic_themes[:3])}")

        # Alternatives (for tutorial mode)
        if p.alternative_moves and p.game_mode == GameMode.TUTORIAL:
            alts = []
            for alt in p.alternative_moves[:2]:
                if isinstance(alt, dict):
                    move = alt.get("move", "")
                    reason = alt.get("reason", "")
                    alts.append(f"{move}: {reason}")
            if alts:
                parts.append(f"\n**Alternatives:** {'; '.join(alts)}")

        if p.is_game_over:
            parts.append(f"- Game ended: {p.game_result}")

        parts.append("\nKeep the comment SHORT (2-3 sentences max).")
        parts.append(self._get_output_format())

        return "\n".join(parts)

    def _get_base_instruction(self, params: MoveAnalysisParams) -> str:
        """Get base instruction for the coach."""
        return f"""You are a friendly game coach teaching {params.game_type} to a student.
Your role is to:
- Comment on moves in simple, age-appropriate language
- Be encouraging even when explaining mistakes
- Never be discouraging, critical, or condescending
- Help the student learn strategy through the game

NEVER say directly negative things like:
- "That was a bad move"
- "You blundered"
- "Wrong!"

INSTEAD, use encouraging reframes:
- "Interesting choice! But there's an even better option..."
- "I see what you were trying! Let me show you another idea..."
- "That's a common trap - now you'll recognize it next time!"
"""

    def _get_mode_instruction(self, mode: GameMode) -> str:
        """Get mode-specific instruction."""
        mode_instructions = {
            GameMode.TUTORIAL: (
                "TUTORIAL MODE: Be very explanatory. Explain every move, "
                "give strategic tips, point out threats and opportunities, "
                "and highlight the 'why' behind moves."
            ),
            GameMode.PRACTICE: (
                "PRACTICE MODE: Be supportive but concise. Only explain "
                "mistakes in detail. Praise good moves briefly. Let the "
                "student think independently."
            ),
            GameMode.CHALLENGE: (
                "CHALLENGE MODE: Minimal intervention. Very brief comments "
                "only. Respect the competitive nature. Just acknowledge moves."
            ),
            GameMode.PUZZLE: (
                "PUZZLE MODE: Focus on the solution. Guide without giving "
                "away the answer immediately. Encourage analytical thinking."
            ),
            GameMode.ANALYSIS: (
                "ANALYSIS MODE: Be detailed and educational. Explain "
                "strategic concepts thoroughly. Discuss alternatives."
            ),
        }
        return mode_instructions.get(mode, "")

    def _get_age_adaptation(self, grade_level: int) -> str:
        """Get age-appropriate language guidance."""
        if grade_level <= 3:
            return (
                "Language for ages 6-8: Use very simple words. "
                "Use fun analogies. Short sentences. Lots of encouragement."
            )
        elif grade_level <= 6:
            return (
                "Language for ages 9-11: Can use basic game terms. "
                "Explain strategy simply. Ask guiding questions."
            )
        else:
            return (
                "Language for ages 12+: Can use standard terminology. "
                "Discuss deeper strategic ideas. More analytical explanations."
            )

    def _describe_evaluation(self, evaluation: float) -> str:
        """Convert numeric evaluation to description."""
        if evaluation > 3.0:
            return "Student has a big advantage"
        elif evaluation > 1.0:
            return "Student is better"
        elif evaluation > 0.3:
            return "Student has a slight edge"
        elif evaluation > -0.3:
            return "Position is equal"
        elif evaluation > -1.0:
            return "AI has a slight edge"
        elif evaluation > -3.0:
            return "AI is better"
        else:
            return "AI has a big advantage"

    def _get_output_format(self) -> str:
        """Get the output format instruction."""
        return """
Respond with valid JSON:
```json
{
  "coach_message": "Your encouraging 2-3 sentence comment",
  "move_quality_explained": "Brief quality explanation if relevant",
  "tactical_tip": "Optional tactical tip",
  "encouragement": "Optional encouragement",
  "learning_point": "Optional learning point"
}
```
Only include optional fields when appropriate.
"""

    def parse_response(self, response: str) -> MoveAnalysisResult:
        """Parse LLM response into MoveAnalysisResult.

        Args:
            response: Raw LLM response text.

        Returns:
            MoveAnalysisResult.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        return MoveAnalysisResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            coach_message=data.get("coach_message", response.strip()),
            move_quality_explained=data.get("move_quality_explained"),
            tactical_tip=data.get("tactical_tip"),
            encouragement=data.get("encouragement"),
            learning_point=data.get("learning_point"),
        )

    def _parse_plain_text_response(self, response: str) -> MoveAnalysisResult:
        """Parse plain text when JSON fails."""
        return MoveAnalysisResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            coach_message=response.strip(),
            metadata={"parse_method": "plain_text_fallback"},
        )
