# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game hint generation capability for coaching hints.

This capability generates hints at different levels:
- Level 1: General strategic hint (vague)
- Level 2: More specific directional hint
- Level 3: Reveals the best move

Uses engine analysis to generate accurate hints while
keeping them age-appropriate and encouraging learning.
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


class HintLevel(int, Enum):
    """Hint specificity levels."""

    GENERAL = 1
    SPECIFIC = 2
    SOLUTION = 3


class HintGenerationParams(BaseModel):
    """Parameters for hint generation.

    Attributes:
        game_type: Type of game (chess, connect4).
        hint_level: How specific the hint should be.
        best_move: The best move from engine analysis.
        best_move_reason: Why this move is best.
        threats: Current threats to consider.
        opportunities: Opportunities available.
        strategic_themes: Strategic themes in position.
        student_name: Student's name.
        grade_level: Student's grade level.
        previous_hints: Previous hints given this turn.
        position_evaluation: Current position evaluation.
        language: Language for the hint.
    """

    game_type: str = Field(description="Type of game (chess, connect4)")
    hint_level: HintLevel = Field(description="How specific the hint should be")
    best_move: str = Field(description="The best move from engine analysis")
    best_move_reason: str | None = Field(
        default=None,
        description="Why this move is best",
    )
    threats: list[str] = Field(
        default_factory=list,
        description="Current threats to consider",
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
    previous_hints: list[str] = Field(
        default_factory=list,
        description="Previous hints given this turn",
    )
    position_evaluation: float | None = Field(
        default=None,
        description="Current position evaluation",
    )
    language: str = Field(default="en", description="Language for hint")


class HintGenerationResult(CapabilityResult):
    """Result of hint generation.

    Attributes:
        hint_text: The generated hint message.
        hint_type: Type of hint (strategic, tactical, solution).
        suggested_squares: Squares to highlight on board.
        reveals_move: Whether this reveals the solution.
        encouragement: Optional encouragement with the hint.
    """

    hint_text: str = Field(description="The generated hint message")
    hint_type: str = Field(description="Type of hint")
    suggested_squares: list[str] = Field(
        default_factory=list,
        description="Squares to highlight",
    )
    reveals_move: bool = Field(
        default=False,
        description="Whether this reveals the solution",
    )
    encouragement: str | None = Field(
        default=None,
        description="Optional encouragement",
    )


class GameHintGenerationCapability(Capability):
    """Capability for generating game hints at various levels.

    Generates age-appropriate hints that guide without giving
    away the answer (unless at level 3). Uses engine analysis
    for accuracy while maintaining an encouraging tone.

    Example:
        capability = GameHintGenerationCapability()
        params = HintGenerationParams(
            game_type="chess",
            hint_level=HintLevel.GENERAL,
            best_move="Nf3",
            student_name="Ali",
            grade_level=5,
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        result = capability.parse_response(llm_response)
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "game_hint_generation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates coaching hints at various specificity levels"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate hint generation parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            HintGenerationParams(**params)
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
        """Build prompt for hint generation.

        Args:
            params: Hint generation parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = HintGenerationParams(**params)

        system_parts = []

        system_parts.append(self._get_base_instruction(p))
        system_parts.append(self._get_level_instruction(p.hint_level))
        system_parts.append(self._get_age_adaptation(p.grade_level))

        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        system_message = "\n\n".join(filter(None, system_parts))

        user_message = self._build_user_message(p)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def build_user_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> str:
        """Build user prompt for YAML-driven mode.

        Args:
            params: Hint generation parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            User prompt string.
        """
        self.validate_params(params)
        p = HintGenerationParams(**params)
        return self._build_user_message(p)

    def _build_user_message(self, p: HintGenerationParams) -> str:
        """Build the user message content."""
        parts = []

        parts.append(
            f"Generate a level {p.hint_level.value} hint for {p.game_type}.\n"
        )
        parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")

        parts.append(f"\n**Analysis Data (for your reference, not for the student):**")
        parts.append(f"- Best move: {p.best_move}")

        if p.best_move_reason:
            parts.append(f"- Reason: {p.best_move_reason}")

        if p.threats:
            parts.append(f"- Threats: {', '.join(p.threats[:3])}")

        if p.opportunities:
            parts.append(f"- Opportunities: {', '.join(p.opportunities[:3])}")

        if p.strategic_themes:
            parts.append(f"- Strategic themes: {', '.join(p.strategic_themes[:3])}")

        if p.position_evaluation is not None:
            parts.append(f"- Position eval: {p.position_evaluation:+.2f}")

        if p.previous_hints:
            parts.append(f"\n**Previous hints given:**")
            for hint in p.previous_hints:
                parts.append(f"- {hint}")
            parts.append("(Make this hint progressively more specific)")

        parts.append(f"\nLanguage: {p.language}")
        parts.append(self._get_output_format(p.hint_level))

        return "\n".join(parts)

    def _get_base_instruction(self, params: HintGenerationParams) -> str:
        """Get base instruction for hint generation."""
        return f"""You are a friendly game coach giving hints to help a student.
The student is playing {params.game_type} and needs guidance.

Your hints should:
- Be encouraging and supportive
- Guide thinking, not just give answers (unless level 3)
- Be age-appropriate for grade {params.grade_level}
- Use simple language the student can understand
- Make them feel like they're figuring it out themselves

Never be discouraging or make the student feel bad for needing help.
"""

    def _get_level_instruction(self, level: HintLevel) -> str:
        """Get level-specific instruction."""
        level_instructions = {
            HintLevel.GENERAL: (
                "LEVEL 1 (GENERAL): Give a vague, strategic hint. "
                "Point to a general idea or area to think about. "
                "DO NOT mention specific pieces or squares. "
                "Examples: 'Think about controlling the center' or "
                "'Look for ways to develop your pieces' or "
                "'Something in the middle of the board looks interesting'"
            ),
            HintLevel.SPECIFIC: (
                "LEVEL 2 (SPECIFIC): Give a more directional hint. "
                "You can mention which piece or area to focus on, "
                "but DON'T give the exact move. "
                "Examples: 'Your knight could be more active' or "
                "'Look at what your bishop can do' or "
                "'The e-file has some interesting possibilities'"
            ),
            HintLevel.SOLUTION: (
                "LEVEL 3 (SOLUTION): Reveal the best move clearly. "
                "Explain what makes it the best choice. "
                "Be friendly about it, don't make them feel bad for needing help. "
                "Examples: 'The best move here is Nf3, developing your knight "
                "while eyeing the center' or 'Try column 4 - you can start a "
                "connection there'"
            ),
        }
        return level_instructions.get(level, "")

    def _get_age_adaptation(self, grade_level: int) -> str:
        """Get age-appropriate language guidance."""
        if grade_level <= 3:
            return (
                "Language for ages 6-8: Very simple words. "
                "Use fun comparisons. Be extra encouraging. "
                "Examples: 'Where can your horsey jump to?' or "
                "'Look for pieces that need friends to help them!'"
            )
        elif grade_level <= 6:
            return (
                "Language for ages 9-11: Can use basic game terms. "
                "Ask guiding questions. "
                "Examples: 'Think about piece development' or "
                "'What threats could you create?'"
            )
        else:
            return (
                "Language for ages 12+: Can use standard terminology. "
                "More analytical hints. "
                "Examples: 'Consider the tension in the center' or "
                "'Look for tactical opportunities with your pieces.'"
            )

    def _get_output_format(self, level: HintLevel) -> str:
        """Get output format based on hint level."""
        reveals = "true" if level == HintLevel.SOLUTION else "false"
        hint_type = (
            "solution" if level == HintLevel.SOLUTION
            else "specific" if level == HintLevel.SPECIFIC
            else "strategic"
        )

        return f"""
Respond with valid JSON:
```json
{{
  "hint_text": "Your helpful hint message",
  "hint_type": "{hint_type}",
  "suggested_squares": ["square1", "square2"],
  "reveals_move": {reveals},
  "encouragement": "Optional encouraging message"
}}
```
suggested_squares should be empty for level 1, may include hints for level 2,
and should include the target square for level 3.
"""

    def parse_response(self, response: str) -> HintGenerationResult:
        """Parse LLM response into HintGenerationResult.

        Args:
            response: Raw LLM response text.

        Returns:
            HintGenerationResult.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        return HintGenerationResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            hint_text=data.get("hint_text", response.strip()),
            hint_type=data.get("hint_type", "strategic"),
            suggested_squares=data.get("suggested_squares", []),
            reveals_move=data.get("reveals_move", False),
            encouragement=data.get("encouragement"),
        )

    def _parse_plain_text_response(self, response: str) -> HintGenerationResult:
        """Parse plain text when JSON fails."""
        return HintGenerationResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            hint_text=response.strip(),
            hint_type="strategic",
            metadata={"parse_method": "plain_text_fallback"},
        )
