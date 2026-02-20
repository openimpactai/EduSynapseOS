# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game coach response capability for general coaching messages.

This capability generates various coach messages:
- Game greetings when starting a game
- Game end summaries (win/loss/draw)
- Encouragement during the game
- Progress updates and milestone celebrations

Works with engine analysis data to provide accurate,
age-appropriate, and encouraging coach responses.
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


class ResponseType(str, Enum):
    """Type of coach response to generate."""

    GREETING = "greeting"
    GAME_END = "game_end"
    ENCOURAGEMENT = "encouragement"
    MILESTONE = "milestone"
    PROGRESS_UPDATE = "progress_update"
    TIMEOUT_WARNING = "timeout_warning"
    RETURN_WELCOME = "return_welcome"


class GameResult(str, Enum):
    """Game result for end-of-game responses."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    RESIGNATION = "resignation"
    TIMEOUT = "timeout"


class SessionStats(BaseModel):
    """Session statistics for summary generation.

    Attributes:
        total_moves: Total moves in the game.
        excellent_moves: Count of excellent moves.
        good_moves: Count of good moves.
        inaccuracies: Count of inaccuracies.
        mistakes: Count of mistakes.
        blunders: Count of blunders.
        hints_used: Number of hints requested.
        time_spent_seconds: Total time spent.
    """

    total_moves: int = Field(default=0, description="Total moves")
    excellent_moves: int = Field(default=0, description="Excellent moves")
    good_moves: int = Field(default=0, description="Good moves")
    inaccuracies: int = Field(default=0, description="Inaccuracies")
    mistakes: int = Field(default=0, description="Mistakes")
    blunders: int = Field(default=0, description="Blunders")
    hints_used: int = Field(default=0, description="Hints used")
    time_spent_seconds: int = Field(default=0, description="Time spent")


class CoachResponseParams(BaseModel):
    """Parameters for coach response generation.

    Attributes:
        response_type: Type of response to generate.
        game_type: Type of game (chess, connect4).
        game_mode: Current game mode.
        student_name: Student's name.
        grade_level: Student's grade level.
        player_color: Student's color/side.
        difficulty: AI difficulty level.
        game_result: Result if game ended.
        result_reason: Reason for game end (checkmate, etc).
        session_stats: Session statistics for summary.
        learning_points: Key learning points from the game.
        critical_moments: Notable moments in the game.
        milestone_type: Type of milestone achieved.
        games_played_total: Total games played by student.
        language: Language for response.
    """

    response_type: ResponseType | None = Field(
        default=None,
        description="Type of response (optional, inferred from context)",
    )
    game_type: str = Field(description="Type of game (chess, connect4)")
    game_mode: str = Field(description="Current game mode")
    student_name: str = Field(description="Student's name")
    grade_level: int = Field(description="Student's grade level")
    player_color: str = Field(description="Student's color/side")
    difficulty: str = Field(default="medium", description="AI difficulty")
    game_result: GameResult | None = Field(
        default=None,
        description="Result if game ended",
    )
    result_reason: str | None = Field(
        default=None,
        description="Reason for game end",
    )
    session_stats: SessionStats | None = Field(
        default=None,
        description="Session statistics",
    )
    learning_points: list[str] = Field(
        default_factory=list,
        description="Key learning points",
    )
    critical_moments: list[str] = Field(
        default_factory=list,
        description="Notable moments",
    )
    milestone_type: str | None = Field(
        default=None,
        description="Type of milestone achieved",
    )
    games_played_total: int = Field(
        default=0,
        description="Total games played by student",
    )
    language: str = Field(default="en", description="Response language")


class CoachResponseResult(CapabilityResult):
    """Result of coach response generation.

    Attributes:
        message: The main coach message.
        encouragement: Additional encouragement.
        summary: Brief summary (for game end).
        learning_highlights: Key learning highlights.
        next_suggestion: Suggestion for next steps.
        celebration: Celebration message for milestones/wins.
    """

    message: str = Field(description="Main coach message")
    encouragement: str | None = Field(
        default=None,
        description="Additional encouragement",
    )
    summary: str | None = Field(
        default=None,
        description="Brief summary for game end",
    )
    learning_highlights: list[str] = Field(
        default_factory=list,
        description="Key learning highlights",
    )
    next_suggestion: str | None = Field(
        default=None,
        description="Suggestion for next steps",
    )
    celebration: str | None = Field(
        default=None,
        description="Celebration message",
    )


class GameCoachResponseCapability(Capability):
    """Capability for generating general coach messages.

    Generates greetings, game summaries, encouragement, and
    milestone celebrations. All messages are age-appropriate
    and encouraging.

    Example:
        capability = GameCoachResponseCapability()
        params = CoachResponseParams(
            response_type=ResponseType.GREETING,
            game_type="chess",
            game_mode="tutorial",
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
        return "game_coach_response"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates general coach messages for games"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate coach response parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            CoachResponseParams(**params)
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
        """Build prompt for coach response generation.

        Args:
            params: Coach response parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = CoachResponseParams(**params)

        system_parts = []

        system_parts.append(self._get_base_instruction(p))
        system_parts.append(self._get_response_type_instruction(p.response_type))
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
            params: Coach response parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            User prompt string.
        """
        self.validate_params(params)
        p = CoachResponseParams(**params)
        return self._build_user_message(p)

    def _build_user_message(self, p: CoachResponseParams) -> str:
        """Build the user message content."""
        parts = []

        # Handle optional response_type
        response_type_str = (
            p.response_type.value.replace("_", " ")
            if p.response_type
            else "coach"
        )
        parts.append(
            f"Generate a {response_type_str} message "
            f"for a {p.game_type} game.\n"
        )
        parts.append(f"- Student: {p.student_name} (Grade {p.grade_level})")
        parts.append(f"- Game mode: {p.game_mode}")
        parts.append(f"- Difficulty: {p.difficulty}")
        parts.append(f"- Playing as: {p.player_color}")

        if p.games_played_total > 0:
            parts.append(f"- Games played: {p.games_played_total}")

        if p.response_type == ResponseType.GAME_END and p.game_result:
            parts.append(f"\n**Game Result:**")
            parts.append(f"- Result: {p.game_result.value}")
            if p.result_reason:
                parts.append(f"- Reason: {p.result_reason}")

            if p.session_stats:
                stats = p.session_stats
                parts.append(f"\n**Statistics:**")
                parts.append(f"- Moves: {stats.total_moves}")
                parts.append(f"- Excellent moves: {stats.excellent_moves}")
                parts.append(f"- Good moves: {stats.good_moves}")
                if stats.mistakes > 0:
                    parts.append(f"- Mistakes: {stats.mistakes}")
                if stats.blunders > 0:
                    parts.append(f"- Blunders: {stats.blunders}")
                if stats.hints_used > 0:
                    parts.append(f"- Hints used: {stats.hints_used}")
                minutes = stats.time_spent_seconds // 60
                if minutes > 0:
                    parts.append(f"- Time: {minutes} minutes")

            if p.learning_points:
                parts.append(f"\n**Learning Points:**")
                for point in p.learning_points[:3]:
                    parts.append(f"- {point}")

            if p.critical_moments:
                parts.append(f"\n**Key Moments:**")
                for moment in p.critical_moments[:3]:
                    parts.append(f"- {moment}")

        if p.response_type == ResponseType.MILESTONE and p.milestone_type:
            parts.append(f"\n**Milestone:** {p.milestone_type}")

        parts.append(f"\nLanguage: {p.language}")
        parts.append(self._get_output_format(p.response_type))

        return "\n".join(parts)

    def _get_base_instruction(self, params: CoachResponseParams) -> str:
        """Get base instruction for coach."""
        return f"""You are a friendly and encouraging game coach on EduSynapseOS.
You are coaching a student named {params.student_name} in {params.game_type}.

Your personality:
- Warm, friendly, and supportive
- Always encouraging, never critical
- Use simple, age-appropriate language
- Make learning fun through games
- Celebrate progress, big or small

Address the student by name occasionally, but not in every sentence.
"""

    def _get_response_type_instruction(self, response_type: ResponseType) -> str:
        """Get type-specific instruction."""
        type_instructions = {
            ResponseType.GREETING: (
                "GREETING: Welcome the student warmly. Be excited about "
                "playing together. Briefly explain what to expect based on "
                "the game mode. Keep it short and friendly."
            ),
            ResponseType.GAME_END: (
                "GAME END: Summarize the game positively. If they won, "
                "celebrate! If they lost, be supportive and highlight what "
                "they did well. Mention key learning points. Suggest playing again."
            ),
            ResponseType.ENCOURAGEMENT: (
                "ENCOURAGEMENT: Provide a motivating boost. Acknowledge "
                "their effort. Help them stay positive and focused."
            ),
            ResponseType.MILESTONE: (
                "MILESTONE: Celebrate the achievement! Make it special. "
                "Acknowledge their progress and dedication."
            ),
            ResponseType.PROGRESS_UPDATE: (
                "PROGRESS UPDATE: Summarize how they're doing. Highlight "
                "improvements. Keep it positive and forward-looking."
            ),
            ResponseType.TIMEOUT_WARNING: (
                "TIMEOUT WARNING: Gently remind them about time. Be helpful, "
                "not stressful. Suggest they can save and continue later."
            ),
            ResponseType.RETURN_WELCOME: (
                "RETURN WELCOME: Welcome them back warmly. Remind them "
                "where they left off. Express excitement to continue."
            ),
        }
        return type_instructions.get(response_type, "")

    def _get_age_adaptation(self, grade_level: int) -> str:
        """Get age-appropriate language guidance."""
        if grade_level <= 3:
            return (
                "Language for ages 6-8: Very simple words. Fun expressions. "
                "Short sentences. Extra enthusiasm and encouragement."
            )
        elif grade_level <= 6:
            return (
                "Language for ages 9-11: Simple but can be slightly more "
                "sophisticated. Can mention game concepts by name."
            )
        else:
            return (
                "Language for ages 12+: Can use standard game terminology. "
                "More conversational and peer-like tone."
            )

    def _get_output_format(self, response_type: ResponseType) -> str:
        """Get output format based on response type."""
        if response_type == ResponseType.GAME_END:
            return """
Respond with valid JSON:
```json
{
  "message": "Main encouraging message about the game",
  "summary": "Brief 1-sentence summary of the game",
  "learning_highlights": ["Key learning point 1", "Key learning point 2"],
  "encouragement": "Additional encouragement",
  "next_suggestion": "Suggestion for what to try next",
  "celebration": "Celebration if they won or achieved something"
}
```
"""
        elif response_type == ResponseType.MILESTONE:
            return """
Respond with valid JSON:
```json
{
  "message": "Celebratory message for the achievement",
  "celebration": "Special celebration text",
  "encouragement": "Keep going motivation"
}
```
"""
        else:
            return """
Respond with valid JSON:
```json
{
  "message": "Your friendly coach message",
  "encouragement": "Optional additional encouragement"
}
```
"""

    def parse_response(self, response: str) -> CoachResponseResult:
        """Parse LLM response into CoachResponseResult.

        Args:
            response: Raw LLM response text.

        Returns:
            CoachResponseResult.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        return CoachResponseResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            message=data.get("message", response.strip()),
            encouragement=data.get("encouragement"),
            summary=data.get("summary"),
            learning_highlights=data.get("learning_highlights", []),
            next_suggestion=data.get("next_suggestion"),
            celebration=data.get("celebration"),
        )

    def _parse_plain_text_response(self, response: str) -> CoachResponseResult:
        """Parse plain text when JSON fails."""
        return CoachResponseResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            message=response.strip(),
            metadata={"parse_method": "plain_text_fallback"},
        )
