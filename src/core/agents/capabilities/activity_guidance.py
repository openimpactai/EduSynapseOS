# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Activity guidance capability for Companion Agent.

This capability suggests appropriate learning activities based on
the student's emotional state, interests, and learning needs.

Usage:
    capability = ActivityGuidanceCapability()
    prompt = capability.build_prompt(
        {"emotional_state": "good", "weak_topics": ["fractions"]},
        context
    )
    # Agent sends prompt to LLM
    result = capability.parse_response(llm_response)
    # result.activities = [ActivityOption(...), ...]
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class ActivityGuidanceParams(BaseModel):
    """Parameters for activity guidance.

    Attributes:
        emotional_state: Current emotional state.
        available_activities: List of available activity types.
        weak_topics: Topics where student is struggling.
        due_review_count: Number of items due for review.
        interests: Student's interests.
        time_available_minutes: Available time for activities.
        parent_preferences: Parent preferences for activities.
    """

    emotional_state: str = Field(
        default="neutral",
        description="Current emotional state",
    )
    available_activities: list[str] = Field(
        default_factory=lambda: ["practice", "review", "game"],
        description="Available activity types",
    )
    weak_topics: list[str] = Field(
        default_factory=list,
        description="Topics needing practice",
    )
    due_review_count: int = Field(
        default=0,
        description="Number of reviews due",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Student interests",
    )
    time_available_minutes: int | None = Field(
        default=None,
        description="Available time in minutes",
    )
    parent_preferences: dict | None = Field(
        default=None,
        description="Parent preferences",
    )


class ActivityOption(BaseModel):
    """A suggested activity.

    Attributes:
        activity_type: Type of activity (practice, game, review, etc.).
        title: Activity title.
        description: Brief description.
        topic_name: Topic name if applicable.
        estimated_minutes: Estimated duration.
        difficulty: Difficulty level (easy, medium, hard).
        priority: Priority ranking (1 = highest).
    """

    activity_type: str = Field(description="Activity type")
    title: str = Field(description="Activity title")
    description: str = Field(description="Brief description")
    topic_name: str | None = Field(default=None, description="Topic if applicable")
    estimated_minutes: int = Field(default=10, description="Estimated duration")
    difficulty: str = Field(default="medium", description="Difficulty level")
    priority: int = Field(default=1, description="Priority ranking")


class ActivityGuidanceResult(CapabilityResult):
    """Result of activity guidance.

    Attributes:
        introduction_message: Friendly intro to suggestions.
        activities: List of suggested activities.
        personalization_reason: Why these activities were chosen.
    """

    introduction_message: str = Field(
        default="",
        description="Friendly introduction to suggestions",
    )
    activities: list[ActivityOption] = Field(
        default_factory=list,
        description="Suggested activities",
    )
    personalization_reason: str = Field(
        default="",
        description="Reason for these suggestions",
    )


class ActivityGuidanceCapability(Capability):
    """Capability for suggesting learning activities.

    Generates personalized activity suggestions based on emotional state,
    learning needs, and interests. Considers parent preferences when available.

    Example:
        capability = ActivityGuidanceCapability()
        params = {"emotional_state": "good", "weak_topics": ["fractions"]}
        prompt = capability.build_prompt(params, context)
        # Agent handles LLM call
        result = capability.parse_response(llm_response)
        # result.activities = [ActivityOption(type="practice", title="...")]
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "activity_guidance"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Suggests appropriate learning activities based on context"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate guidance parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            ActivityGuidanceParams(**params)
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
        """Build prompt for activity guidance.

        Args:
            params: Guidance parameters including emotional state.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = ActivityGuidanceParams(**params)

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are a learning companion. "
            "Your task is to suggest 2-3 activities appropriate for the student's situation. "
            "Your suggestions should consider the student's emotional state, interests, "
            "and learning needs."
        )

        system_parts.append(
            "Rules:\n"
            "- Don't suggest difficult activities for negative emotional states\n"
            "- Include student interests in activities\n"
            "- Provide variety (not just practice)\n"
            "- Explain each suggestion briefly"
        )

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(f"Student context:\n{student_summary}")

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(f"Persona:\n{persona_prompt}")

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(f"Emotional state: {p.emotional_state}")

        if p.available_activities:
            user_parts.append(
                f"Available activity types: {', '.join(p.available_activities)}"
            )

        if p.weak_topics:
            user_parts.append(
                f"Topics needing practice: {', '.join(p.weak_topics[:3])}"
            )

        if p.due_review_count > 0:
            user_parts.append(f"Pending reviews: {p.due_review_count} items")

        if p.interests:
            user_parts.append(f"Student interests: {', '.join(p.interests[:3])}")

        if p.time_available_minutes:
            user_parts.append(f"Available time: {p.time_available_minutes} minutes")

        if p.parent_preferences:
            user_parts.append(f"Parent preferences: {p.parent_preferences}")

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
  "introduction": "Friendly introduction message",
  "activities": [
    {
      "type": "practice|game|review|creative|break",
      "title": "Activity title",
      "description": "Brief description",
      "topic": "Topic name (if any)",
      "minutes": 10,
      "difficulty": "easy|medium|hard",
      "priority": 1
    }
  ],
  "reason": "Why you chose these activities"
}
```

Rules:
- Suggest 2-3 activities
- introduction: Friendly, brief intro
- type: Choose from available activity types
- difficulty: Adjust based on emotional state (negative = easy)
- priority: 1 = most important
"""

    def parse_response(self, response: str) -> ActivityGuidanceResult:
        """Parse LLM response into ActivityGuidanceResult.

        Args:
            response: Raw LLM response text.

        Returns:
            ActivityGuidanceResult with activities.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        data = self._extract_json_from_response(response)

        activities = []
        for act in data.get("activities", []):
            activities.append(
                ActivityOption(
                    activity_type=act.get("type", "practice"),
                    title=act.get("title", ""),
                    description=act.get("description", ""),
                    topic_name=act.get("topic"),
                    estimated_minutes=act.get("minutes", 10),
                    difficulty=act.get("difficulty", "medium"),
                    priority=act.get("priority", 1),
                )
            )

        return ActivityGuidanceResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            introduction_message=data.get("introduction", ""),
            activities=activities,
            personalization_reason=data.get("reason", ""),
        )
