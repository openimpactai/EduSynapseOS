# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wellbeing check capability for Companion Agent.

This capability generates check-in messages and mood options for students.
Used at login, mid-session, or logout to understand student's emotional state.

Usage:
    capability = WellbeingCheckCapability()
    prompt = capability.build_prompt(
        {"check_type": "login", "student_name": "Alex"},
        context
    )
    # Agent sends prompt to LLM
    result = capability.parse_response(llm_response)
    # result.greeting_message = "Hey Alex! How are you feeling today?"
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class WellbeingCheckParams(BaseModel):
    """Parameters for wellbeing check.

    Attributes:
        check_type: Type of check-in (login, mid_session, logout).
        student_name: Student's name for personalization.
        previous_mood: Yesterday's reported mood if available.
        last_session_summary: Brief summary of last session.
        emotional_context: Current emotional indicators.
        parent_notes: Active parent notes to consider.
    """

    check_type: str = Field(
        default="login",
        description="Type of check-in: login, mid_session, logout",
    )
    student_name: str | None = Field(
        default=None,
        description="Student's name for personalization",
    )
    previous_mood: str | None = Field(
        default=None,
        description="Yesterday's reported mood",
    )
    last_session_summary: str | None = Field(
        default=None,
        description="Brief summary of last session",
    )
    emotional_context: dict | None = Field(
        default=None,
        description="Current emotional indicators",
    )
    parent_notes: list[str] = Field(
        default_factory=list,
        description="Active parent notes",
    )


class WellbeingCheckResult(CapabilityResult):
    """Result of wellbeing check.

    Attributes:
        greeting_message: The warm greeting message for the student.
        mood_options: List of mood options to display.
        follow_up_question: Optional follow-up question.
        suggested_activity: Optional activity suggestion based on context.
    """

    greeting_message: str = Field(
        default="Hey there! How are you feeling today?",
        description="Warm greeting message",
    )
    mood_options: list[dict] = Field(
        default_factory=list,
        description="Mood selection options",
    )
    follow_up_question: str | None = Field(
        default=None,
        description="Optional follow-up question",
    )
    suggested_activity: str | None = Field(
        default=None,
        description="Optional activity suggestion",
    )


# Default mood options (always included)
DEFAULT_MOOD_OPTIONS = [
    {"emoji": "😊", "label": "Great", "value": "great"},
    {"emoji": "🙂", "label": "Good", "value": "good"},
    {"emoji": "😐", "label": "Okay", "value": "okay"},
    {"emoji": "😔", "label": "Not great", "value": "not_great"},
    {"emoji": "😰", "label": "Stressed", "value": "stressed"},
]


class WellbeingCheckCapability(Capability):
    """Capability for checking student wellbeing.

    Generates warm, personalized check-in messages based on context.
    Used at login (daily check-in), mid-session (if extended practice),
    or logout (session wrap-up).

    Example:
        capability = WellbeingCheckCapability()
        params = {"check_type": "login", "student_name": "Alex"}
        prompt = capability.build_prompt(params, context)
        # Agent handles LLM call
        result = capability.parse_response(llm_response)
        # result.greeting_message = "Hey Alex! How are you feeling today?"
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "wellbeing_check"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates warm check-in messages and mood prompts for students"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate check parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            WellbeingCheckParams(**params)
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
        """Build prompt for wellbeing check.

        Args:
            params: Check parameters including check_type and student info.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = WellbeingCheckParams(**params)

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are a warm and supportive learning companion. "
            "Your task is to give a friendly greeting to understand the student's emotional state. "
            "Be brief and friendly (1-2 sentences). Make the student feel comfortable."
        )

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(f"Persona:\n{persona_prompt}")

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(f"Check-in type: {p.check_type}")

        if p.student_name:
            user_parts.append(f"Student name: {p.student_name}")

        if p.previous_mood:
            user_parts.append(f"Yesterday's mood: {p.previous_mood}")

        if p.last_session_summary:
            user_parts.append(f"Last session: {p.last_session_summary}")

        if p.emotional_context:
            user_parts.append(f"Current emotional indicators: {p.emotional_context}")

        if p.parent_notes:
            notes_text = "; ".join(p.parent_notes[:3])
            user_parts.append(f"Parent notes: {notes_text}")

        user_parts.append(self._get_output_format_instruction())

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_output_format_instruction(self) -> str:
        """Get the JSON output format instruction."""
        return """
Respond in the following JSON format:
```json
{
  "greeting": "Your warm greeting message",
  "follow_up": "Optional follow-up question (can be null)",
  "suggested_activity": "Optional activity suggestion (can be null)"
}
```

Rules:
- greeting: Friendly, short (1-2 sentences), use student's name (if available)
- For login: "How are you feeling today?" style
- For mid_session: "How's it going?" style
- For logout: "Great work today!" style
"""

    def parse_response(self, response: str) -> WellbeingCheckResult:
        """Parse LLM response into WellbeingCheckResult.

        Args:
            response: Raw LLM response text.

        Returns:
            WellbeingCheckResult with greeting and options.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        data = self._extract_json_from_response(response)

        return WellbeingCheckResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            greeting_message=data.get("greeting", "Hey there! How are you feeling today?"),
            mood_options=DEFAULT_MOOD_OPTIONS,
            follow_up_question=data.get("follow_up"),
            suggested_activity=data.get("suggested_activity"),
        )
