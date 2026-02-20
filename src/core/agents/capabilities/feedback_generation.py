# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Feedback generation capability for personalized learning feedback.

This capability generates personalized feedback based on:
- Session performance and progress
- Historical learning patterns
- Emotional state and motivation level
- Educational theory recommendations

The feedback includes:
- Performance summary
- Motivational messaging
- Specific improvement suggestions
- Next steps recommendations
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


class FeedbackType(str, Enum):
    """Types of feedback."""

    SESSION_COMPLETE = "session_complete"
    MILESTONE_REACHED = "milestone_reached"
    STRUGGLE_SUPPORT = "struggle_support"
    ENCOURAGEMENT = "encouragement"
    PROGRESS_UPDATE = "progress_update"
    RECOMMENDATION = "recommendation"


class FeedbackTone(str, Enum):
    """Tone for feedback delivery."""

    CELEBRATORY = "celebratory"
    SUPPORTIVE = "supportive"
    MOTIVATIONAL = "motivational"
    NEUTRAL = "neutral"
    ENCOURAGING = "encouraging"


class PerformanceData(BaseModel):
    """Performance data for feedback generation.

    Attributes:
        questions_total: Total questions attempted.
        questions_correct: Correct answers.
        score: Overall score (0.0-1.0).
        time_spent_minutes: Time spent on session.
        topics_practiced: Topics covered.
        improvement_areas: Areas needing improvement.
        strengths: Areas of strength.
        streak: Current correct streak.
        mastery_changes: Changes in mastery levels.
    """

    questions_total: int = Field(default=0, description="Total questions")
    questions_correct: int = Field(default=0, description="Correct answers")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score 0.0-1.0")
    time_spent_minutes: int = Field(default=0, description="Time spent")
    topics_practiced: list[str] = Field(
        default_factory=list,
        description="Topics covered",
    )
    improvement_areas: list[str] = Field(
        default_factory=list,
        description="Areas needing work",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Areas of strength",
    )
    streak: int = Field(default=0, description="Current streak")
    mastery_changes: dict[str, float] = Field(
        default_factory=dict,
        description="Topic -> mastery change",
    )


class FeedbackGenerationParams(BaseModel):
    """Parameters for feedback generation.

    Attributes:
        feedback_type: Type of feedback to generate.
        performance: Performance data for the session.
        context_message: Additional context for feedback.
        preferred_tone: Desired tone for feedback.
        include_stats: Whether to include statistics.
        include_next_steps: Whether to suggest next steps.
        include_encouragement: Whether to add encouragement.
        language: Language for feedback.
    """

    feedback_type: FeedbackType = Field(
        default=FeedbackType.SESSION_COMPLETE,
        description="Type of feedback",
    )
    performance: PerformanceData = Field(
        default_factory=PerformanceData,
        description="Performance data",
    )
    context_message: str | None = Field(
        default=None,
        description="Additional context",
    )
    preferred_tone: FeedbackTone = Field(
        default=FeedbackTone.SUPPORTIVE,
        description="Desired feedback tone",
    )
    include_stats: bool = Field(
        default=True,
        description="Include statistics",
    )
    include_next_steps: bool = Field(
        default=True,
        description="Include next steps",
    )
    include_encouragement: bool = Field(
        default=True,
        description="Include encouragement",
    )
    language: str = Field(
        default="en",
        description="Language for feedback",
    )


class NextStep(BaseModel):
    """A suggested next step for the student.

    Attributes:
        action: What to do.
        reason: Why this is recommended.
        priority: Priority level (high, medium, low).
    """

    action: str = Field(description="What to do")
    reason: str = Field(description="Why this is recommended")
    priority: str = Field(default="medium", description="Priority level")


class GeneratedFeedback(CapabilityResult):
    """Result of feedback generation.

    Attributes:
        feedback_type: Type of feedback generated.
        main_message: The primary feedback message.
        summary: Brief performance summary.
        encouragement: Motivational message.
        statistics: Performance statistics.
        strengths_highlighted: Highlighted strengths.
        improvement_suggestions: Specific suggestions.
        next_steps: Recommended next actions.
        celebration_message: Special celebration if milestone.
        emotional_support: Support message if struggling.
        tone_used: The tone that was used.
        language: Language of the feedback.
    """

    feedback_type: FeedbackType = Field(description="Type of feedback")
    main_message: str = Field(description="Primary feedback message")
    summary: str = Field(description="Brief summary")
    encouragement: str | None = Field(
        default=None,
        description="Motivational message",
    )
    statistics: dict[str, Any] | None = Field(
        default=None,
        description="Performance stats",
    )
    strengths_highlighted: list[str] = Field(
        default_factory=list,
        description="Highlighted strengths",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions",
    )
    next_steps: list[NextStep] = Field(
        default_factory=list,
        description="Recommended next actions",
    )
    celebration_message: str | None = Field(
        default=None,
        description="Celebration for milestones",
    )
    emotional_support: str | None = Field(
        default=None,
        description="Support message if struggling",
    )
    tone_used: FeedbackTone = Field(description="Tone used")
    language: str = Field(default="en", description="Language")


class FeedbackGenerationCapability(Capability):
    """Capability for generating personalized learning feedback.

    Generates feedback tailored to student's performance, emotional state,
    and learning context. Uses persona for consistent communication style.

    Example:
        capability = FeedbackGenerationCapability()
        params = FeedbackGenerationParams(
            feedback_type=FeedbackType.SESSION_COMPLETE,
            performance=PerformanceData(
                questions_total=10,
                questions_correct=8,
                score=0.8,
            ),
            preferred_tone=FeedbackTone.CELEBRATORY,
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        # Agent sends prompt to LLM, gets response
        result = capability.parse_response(llm_response)
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "feedback_generation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates personalized motivational feedback based on performance"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate feedback parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            FeedbackGenerationParams(**params)
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
        """Build prompt for feedback generation.

        Args:
            params: Feedback parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = FeedbackGenerationParams(**params)

        # Determine appropriate tone based on performance
        tone = self._determine_tone(p)

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are an encouraging educational mentor. "
            "Provide personalized, constructive feedback that motivates "
            "students while being honest about areas for improvement."
        )

        # Add tone guidance
        tone_guidance = self._get_tone_guidance(tone)
        if tone_guidance:
            system_parts.append(tone_guidance)

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        # Add student context
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(student_summary)

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(
            f"Generate {p.feedback_type.value.replace('_', ' ')} feedback "
            f"for a student with the following performance:\n"
        )

        # Performance data
        perf = p.performance
        if perf.questions_total > 0:
            accuracy = (perf.questions_correct / perf.questions_total) * 100
            user_parts.append(f"- Score: {perf.score:.0%} ({perf.questions_correct}/{perf.questions_total} correct)")
            user_parts.append(f"- Accuracy: {accuracy:.1f}%")
        else:
            user_parts.append(f"- Score: {perf.score:.0%}")

        if perf.time_spent_minutes > 0:
            user_parts.append(f"- Time spent: {perf.time_spent_minutes} minutes")

        if perf.streak > 0:
            user_parts.append(f"- Current streak: {perf.streak} correct in a row")

        if perf.topics_practiced:
            user_parts.append(f"- Topics: {', '.join(perf.topics_practiced)}")

        if perf.strengths:
            user_parts.append(f"- Strengths: {', '.join(perf.strengths)}")

        if perf.improvement_areas:
            user_parts.append(f"- Areas to improve: {', '.join(perf.improvement_areas)}")

        if perf.mastery_changes:
            changes = [
                f"{topic}: {change:+.0%}"
                for topic, change in perf.mastery_changes.items()
            ]
            user_parts.append(f"- Mastery changes: {', '.join(changes)}")

        # Requirements
        user_parts.append(f"\n**Requirements:**")
        user_parts.append(f"- Tone: {tone.value}")
        user_parts.append(f"- Language: {p.language}")

        if p.include_stats:
            user_parts.append("- Include relevant statistics in the feedback")

        if p.include_next_steps:
            user_parts.append("- Suggest specific next steps for improvement")

        if p.include_encouragement:
            user_parts.append("- Include motivational encouragement")

        if p.context_message:
            user_parts.append(f"\nAdditional context: {p.context_message}")

        # Output format
        user_parts.append(self._get_output_format_instruction(p))

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _determine_tone(self, params: FeedbackGenerationParams) -> FeedbackTone:
        """Determine appropriate tone based on performance.

        Args:
            params: Feedback parameters.

        Returns:
            Appropriate feedback tone.
        """
        perf = params.performance

        # Use preferred tone if explicitly set for certain types
        if params.feedback_type == FeedbackType.MILESTONE_REACHED:
            return FeedbackTone.CELEBRATORY

        if params.feedback_type == FeedbackType.STRUGGLE_SUPPORT:
            return FeedbackTone.SUPPORTIVE

        # Determine based on score
        if perf.score >= 0.9:
            return FeedbackTone.CELEBRATORY
        elif perf.score >= 0.7:
            return FeedbackTone.ENCOURAGING
        elif perf.score >= 0.5:
            return FeedbackTone.MOTIVATIONAL
        else:
            return FeedbackTone.SUPPORTIVE

    def _get_tone_guidance(self, tone: FeedbackTone) -> str:
        """Get guidance based on feedback tone.

        Args:
            tone: The desired feedback tone.

        Returns:
            Tone-specific guidance text.
        """
        guidance_map = {
            FeedbackTone.CELEBRATORY: (
                "Be enthusiastic and celebratory! Highlight achievements "
                "and make the student feel proud of their accomplishment."
            ),
            FeedbackTone.SUPPORTIVE: (
                "Be warm and understanding. Acknowledge challenges while "
                "providing reassurance that improvement is possible with practice."
            ),
            FeedbackTone.MOTIVATIONAL: (
                "Be energizing and forward-looking. Focus on potential "
                "and the exciting progress ahead. Use action-oriented language."
            ),
            FeedbackTone.NEUTRAL: (
                "Be balanced and objective. Present facts clearly while "
                "maintaining a professional and respectful tone."
            ),
            FeedbackTone.ENCOURAGING: (
                "Be positive and hopeful. Emphasize progress made and "
                "express confidence in the student's abilities."
            ),
        }
        return guidance_map.get(tone, "")

    def _get_output_format_instruction(self, params: FeedbackGenerationParams) -> str:
        """Get the JSON output format instruction.

        Args:
            params: Feedback parameters.

        Returns:
            Output format instruction string.
        """
        return """
Respond with a valid JSON object in this exact format:
```json
{
  "main_message": "The primary personalized feedback message",
  "summary": "Brief 1-sentence performance summary",
  "encouragement": "Motivational message",
  "statistics": {
    "accuracy": "80%",
    "improvement": "+5%"
  },
  "strengths_highlighted": ["Strength 1", "Strength 2"],
  "improvement_suggestions": ["Specific suggestion 1", "Specific suggestion 2"],
  "next_steps": [
    {
      "action": "Practice more fraction problems",
      "reason": "To solidify understanding",
      "priority": "high"
    }
  ],
  "celebration_message": "Special celebration message if milestone",
  "emotional_support": "Support message if struggling"
}
```
Note: celebration_message and emotional_support should only be included when appropriate.
"""

    def parse_response(self, response: str) -> GeneratedFeedback:
        """Parse LLM response into GeneratedFeedback.

        Args:
            response: Raw LLM response text.

        Returns:
            GeneratedFeedback result.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        # Parse next steps
        next_steps = []
        if "next_steps" in data and data["next_steps"]:
            for ns in data["next_steps"]:
                next_steps.append(
                    NextStep(
                        action=ns.get("action", ""),
                        reason=ns.get("reason", ""),
                        priority=ns.get("priority", "medium"),
                    )
                )

        return GeneratedFeedback(
            success=True,
            capability_name=self.name,
            raw_response=response,
            feedback_type=FeedbackType.SESSION_COMPLETE,
            main_message=data.get("main_message", ""),
            summary=data.get("summary", ""),
            encouragement=data.get("encouragement"),
            statistics=data.get("statistics"),
            strengths_highlighted=data.get("strengths_highlighted", []),
            improvement_suggestions=data.get("improvement_suggestions", []),
            next_steps=next_steps,
            celebration_message=data.get("celebration_message"),
            emotional_support=data.get("emotional_support"),
            tone_used=FeedbackTone.SUPPORTIVE,
            language=data.get("language", "tr"),
        )

    def _parse_plain_text_response(self, response: str) -> GeneratedFeedback:
        """Parse a plain text response when JSON parsing fails.

        Args:
            response: Plain text response.

        Returns:
            GeneratedFeedback with content extracted from text.
        """
        lines = response.strip().split("\n")

        # Use entire response as main message
        main_message = response.strip()

        # First sentence as summary
        summary = ""
        if lines:
            first_line = lines[0].strip()
            if "." in first_line:
                summary = first_line.split(".")[0] + "."
            else:
                summary = first_line

        return GeneratedFeedback(
            success=True,
            capability_name=self.name,
            raw_response=response,
            feedback_type=FeedbackType.SESSION_COMPLETE,
            main_message=main_message,
            summary=summary,
            encouragement=None,
            statistics=None,
            strengths_highlighted=[],
            improvement_suggestions=[],
            next_steps=[],
            celebration_message=None,
            emotional_support=None,
            tone_used=FeedbackTone.SUPPORTIVE,
            language="tr",
            metadata={"parse_method": "plain_text_fallback"},
        )
