# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion decision capability for Ambient Companion.

This capability determines WHEN and WHAT the companion should say.
It's the "brain" of the ambient companion system.

The decision is LLM-based, not rule-based, allowing for nuanced,
context-aware decisions about proactive engagement.

Usage:
    capability = CompanionDecisionCapability()
    prompt = capability.build_prompt(
        {
            "pending_alerts": [...],
            "emotional_state": "stressed",
            "days_inactive": 3,
        },
        context
    )
    # Agent sends prompt to LLM
    result = capability.parse_response(llm_response)
    # result.should_speak = True
    # result.intention = "greet_with_support"
    # result.target_capability = "wellbeing_check"
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)


class PendingAlert(BaseModel):
    """Pending alert from ProactiveService.

    Attributes:
        alert_type: Type of alert (struggle, milestone, inactivity, etc.).
        severity: Alert severity (info, warning, critical).
        title: Short title.
        topic: Related topic if any.
    """

    alert_type: str = Field(description="Type of alert")
    severity: str = Field(default="info", description="Alert severity")
    title: str = Field(default="", description="Short title")
    topic: str | None = Field(default=None, description="Related topic")


class CompanionDecisionParams(BaseModel):
    """Parameters for companion decision.

    Attributes:
        pending_alerts: Active alerts from ProactiveService.
        emotional_state: Current emotional state (if known).
        emotional_intensity: Intensity of emotional state.
        days_inactive: Days since last activity.
        last_interaction_minutes: Minutes since last companion interaction.
        student_activity: Current student activity (dashboard, practice, idle).
        session_duration_minutes: Current session duration.
        student_name: Student name for context.
        has_milestone: Whether there's a pending milestone to celebrate.
        milestone_details: Details of milestone if any.
    """

    pending_alerts: list[PendingAlert] = Field(
        default_factory=list,
        description="Active alerts from ProactiveService",
    )
    emotional_state: str | None = Field(
        default=None,
        description="Current emotional state",
    )
    emotional_intensity: str | None = Field(
        default=None,
        description="Emotional intensity: low, moderate, high",
    )
    days_inactive: int = Field(
        default=0,
        description="Days since last activity",
    )
    last_interaction_minutes: int | None = Field(
        default=None,
        description="Minutes since last companion interaction",
    )
    student_activity: str = Field(
        default="unknown",
        description="Current activity: dashboard, practice, idle, unknown",
    )
    session_duration_minutes: int = Field(
        default=0,
        description="Current session duration in minutes",
    )
    student_name: str | None = Field(
        default=None,
        description="Student name",
    )
    has_milestone: bool = Field(
        default=False,
        description="Whether there's a milestone to celebrate",
    )
    milestone_details: str | None = Field(
        default=None,
        description="Milestone details if any",
    )


class CompanionDecisionResult(CapabilityResult):
    """Result of companion decision.

    Attributes:
        should_speak: Whether companion should initiate conversation.
        intention: The intention behind speaking (greet, support, celebrate, etc.).
        priority: Priority level 1-5 (5 is highest).
        approach: Guidance for how to approach the conversation.
        target_capability: Which capability to use for response generation.
        reasoning: Brief reasoning for the decision.
    """

    should_speak: bool = Field(
        default=False,
        description="Whether companion should speak",
    )
    intention: str = Field(
        default="idle",
        description="Intention: greet, support, celebrate, guide, remind, intervene, idle",
    )
    priority: int = Field(
        default=1,
        description="Priority 1-5, 5 is highest",
    )
    approach: str | None = Field(
        default=None,
        description="Guidance for response generation",
    )
    target_capability: str | None = Field(
        default=None,
        description="Which capability to use: wellbeing_check, emotional_support, activity_guidance",
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief reasoning for the decision",
    )


# Intention to capability mapping
INTENTION_TO_CAPABILITY = {
    "greet": "wellbeing_check",
    "greet_with_support": "wellbeing_check",
    "support": "emotional_support",
    "celebrate": "wellbeing_check",
    "guide": "activity_guidance",
    "remind": "wellbeing_check",
    "intervene": "emotional_support",
    "idle": None,
}


class CompanionDecisionCapability(Capability):
    """Capability for deciding when and what companion should say.

    This is the decision engine for the ambient companion. It analyzes
    the current context (alerts, emotional state, activity, etc.) and
    decides whether the companion should proactively engage.

    All decisions are LLM-based for nuanced, context-aware behavior.

    Example:
        capability = CompanionDecisionCapability()
        params = {
            "pending_alerts": [{"alert_type": "struggle", "severity": "warning"}],
            "emotional_state": "frustrated",
            "days_inactive": 0,
        }
        prompt = capability.build_prompt(params, context)
        # Agent handles LLM call
        result = capability.parse_response(llm_response)
        # result.should_speak = True
        # result.intention = "support"
        # result.target_capability = "emotional_support"
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "companion_decision"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Decides when and what the companion should proactively say"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate decision parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            CompanionDecisionParams(**params)
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
        """Build prompt for companion decision.

        Args:
            params: Decision parameters including alerts, emotional state, etc.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = CompanionDecisionParams(**params)

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are the decision engine for a learning companion. "
            "Your task is to decide whether the companion should speak to the student "
            "based on the given context.\n\n"
            "Your decision should be balanced:\n"
            "- Don't speak too often (annoying)\n"
            "- Don't stay silent too long (uncaring)\n"
            "- Always speak at important moments (support, celebration, reminder)\n"
            "- Don't disturb when student is busy"
        )

        # Add persona context if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(f"Persona context:\n{persona_prompt}")

        system_message = "\n\n".join(system_parts)

        # Build user message with context
        user_parts = []

        user_parts.append("=== CURRENT STATE ===")

        # Student info
        if p.student_name:
            user_parts.append(f"Student: {p.student_name}")

        # Activity
        user_parts.append(f"Current activity: {p.student_activity}")
        user_parts.append(f"Session duration: {p.session_duration_minutes} minutes")

        # Inactivity
        if p.days_inactive > 0:
            user_parts.append(f"Inactivity: {p.days_inactive} days since last login")

        # Last interaction
        if p.last_interaction_minutes is not None:
            if p.last_interaction_minutes == 0:
                user_parts.append("Last conversation: Haven't spoken yet this session")
            else:
                user_parts.append(
                    f"Last conversation: {p.last_interaction_minutes} minutes ago"
                )

        # Emotional state
        if p.emotional_state:
            intensity = p.emotional_intensity or "unknown"
            user_parts.append(f"Emotional state: {p.emotional_state} ({intensity})")

        # Pending alerts
        if p.pending_alerts:
            user_parts.append("\n=== PENDING ALERTS ===")
            for alert in p.pending_alerts:
                alert_line = f"- [{alert.severity.upper()}] {alert.alert_type}"
                if alert.title:
                    alert_line += f": {alert.title}"
                if alert.topic:
                    alert_line += f" (topic: {alert.topic})"
                user_parts.append(alert_line)

        # Milestone
        if p.has_milestone:
            user_parts.append("\n=== ACHIEVEMENT TO CELEBRATE ===")
            if p.milestone_details:
                user_parts.append(p.milestone_details)
            else:
                user_parts.append("New achievement!")

        # Student context from memory
        student_summary = context.get_student_summary()
        if student_summary:
            user_parts.append(f"\n=== STUDENT CONTEXT ===\n{student_summary}")

        # Output format
        user_parts.append(self._get_output_format_instruction())

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_output_format_instruction(self) -> str:
        """Get the JSON output format instruction."""
        return """
=== MAKE YOUR DECISION ===

Respond in the following JSON format:
```json
{
  "should_speak": true or false,
  "intention": "intention_type",
  "priority": 1-5,
  "approach": "How to approach (brief description)",
  "reasoning": "Why you made this decision (brief)"
}
```

Intention types:
- "greet": New login, welcome
- "greet_with_support": Welcome + support (if absent for long or stressed)
- "support": Emotional support needed
- "celebrate": Achievement celebration
- "guide": Activity guidance
- "remind": Reminder (review, continue)
- "intervene": Struggle intervention
- "idle": No need to speak right now

Decision rules:
- If session just started (0 minutes) and haven't spoken yet → greet or greet_with_support
- If critical alert exists → appropriate intention (support, intervene, celebrate)
- If student is busy (practice) and no problem → idle
- If long time since last conversation (>15 min) and student is idle → guide
- If milestone exists → celebrate
- If uncertain → idle (don't disturb)
"""

    def parse_response(self, response: str) -> CompanionDecisionResult:
        """Parse LLM response into CompanionDecisionResult.

        Args:
            response: Raw LLM response text.

        Returns:
            CompanionDecisionResult with decision details.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        data = self._extract_json_from_response(response)

        intention = data.get("intention", "idle")
        should_speak = data.get("should_speak", False)

        # Map intention to target capability
        target_capability = INTENTION_TO_CAPABILITY.get(intention)

        # Validate priority
        priority = data.get("priority", 1)
        if not isinstance(priority, int) or priority < 1:
            priority = 1
        elif priority > 5:
            priority = 5

        return CompanionDecisionResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            should_speak=should_speak,
            intention=intention,
            priority=priority,
            approach=data.get("approach"),
            target_capability=target_capability,
            reasoning=data.get("reasoning"),
        )
