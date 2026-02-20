# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emotional support capability for Companion Agent.

This capability generates supportive messages for students experiencing
negative emotions like frustration, anxiety, or confusion.

Usage:
    capability = EmotionalSupportCapability()
    prompt = capability.build_prompt(
        {"emotional_state": "frustrated", "intensity": "high"},
        context
    )
    # Agent sends prompt to LLM
    result = capability.parse_response(llm_response)
    # result.support_message = "I understand, you're going through a tough moment..."
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class EmotionalSupportParams(BaseModel):
    """Parameters for emotional support.

    Attributes:
        emotional_state: Current emotional state (frustrated, anxious, confused, etc.).
        intensity: Intensity level (low, moderate, high).
        triggers: Factors that triggered this emotional state.
        student_message: What the student said (if any).
        consecutive_errors: Number of consecutive errors (if applicable).
        session_duration_minutes: How long the session has been.
    """

    emotional_state: str = Field(
        default="neutral",
        description="Current emotional state",
    )
    intensity: str = Field(
        default="moderate",
        description="Emotional intensity: low, moderate, high",
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="Emotional triggers identified",
    )
    student_message: str | None = Field(
        default=None,
        description="What the student said",
    )
    consecutive_errors: int = Field(
        default=0,
        description="Number of consecutive errors",
    )
    session_duration_minutes: int = Field(
        default=0,
        description="Session duration in minutes",
    )


class EmotionalSupportResult(CapabilityResult):
    """Result of emotional support generation.

    Attributes:
        support_message: The supportive message for the student.
        suggested_actions: List of suggested actions.
        offer_break: Whether to offer a break.
        offer_easier_activity: Whether to offer an easier activity.
    """

    support_message: str = Field(
        default="",
        description="Supportive message for the student",
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Suggested actions for the student",
    )
    offer_break: bool = Field(
        default=False,
        description="Whether to offer a break",
    )
    offer_easier_activity: bool = Field(
        default=False,
        description="Whether to offer an easier activity",
    )


class EmotionalSupportCapability(Capability):
    """Capability for providing emotional support.

    Generates warm, empathetic support messages for students
    experiencing negative emotions. Considers emotional state,
    intensity, and context to provide appropriate support.

    Example:
        capability = EmotionalSupportCapability()
        params = {"emotional_state": "frustrated", "intensity": "high"}
        prompt = capability.build_prompt(params, context)
        # Agent handles LLM call
        result = capability.parse_response(llm_response)
        # result.support_message = "I understand, this is a tough moment..."
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "emotional_support"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates supportive messages for students experiencing negative emotions"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate support parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            EmotionalSupportParams(**params)
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
        """Build prompt for emotional support.

        Args:
            params: Support parameters including emotional state.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = EmotionalSupportParams(**params)

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are a warm and empathetic learning companion. "
            f"The student is currently feeling {p.emotional_state}, intensity: {p.intensity}. "
            "Your task is to give a supportive and caring message."
        )

        system_parts.append(
            "Rules:\n"
            "- Don't minimize or dismiss their feelings\n"
            "- Be empathetic and understanding\n"
            "- Suggest practical help (break, easier activity)\n"
            "- Be brief but sincere (2-3 sentences max)\n"
            "- Don't lecture or preach"
        )

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(f"Persona:\n{persona_prompt}")

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(f"Emotional state: {p.emotional_state}")
        user_parts.append(f"Intensity: {p.intensity}")

        if p.triggers:
            user_parts.append(f"Triggers: {', '.join(p.triggers)}")

        if p.student_message:
            user_parts.append(f'Student said: "{p.student_message}"')

        if p.consecutive_errors > 0:
            user_parts.append(f"Consecutive errors: {p.consecutive_errors}")

        if p.session_duration_minutes > 25:
            user_parts.append(
                f"Session duration: {p.session_duration_minutes} minutes (long session)"
            )

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
  "support_message": "Your supportive message",
  "offer_break": true/false,
  "offer_easier_activity": true/false,
  "suggested_actions": ["action1", "action2"]
}
```

Rules:
- support_message: Friendly, empathetic, 2-3 sentences max
- offer_break: true for long sessions or high intensity
- offer_easier_activity: true for consecutive errors or frustration
- suggested_actions: Practical, doable suggestions (max 2)
"""

    def parse_response(self, response: str) -> EmotionalSupportResult:
        """Parse LLM response into EmotionalSupportResult.

        Args:
            response: Raw LLM response text.

        Returns:
            EmotionalSupportResult with support message and options.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        data = self._extract_json_from_response(response)

        return EmotionalSupportResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            support_message=data.get("support_message", ""),
            suggested_actions=data.get("suggested_actions", []),
            offer_break=data.get("offer_break", False),
            offer_easier_activity=data.get("offer_easier_activity", False),
        )
