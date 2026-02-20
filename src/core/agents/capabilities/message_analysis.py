# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Message analysis capability for intent and sentiment detection.

This capability analyzes student messages to determine:
- Intent: What the student is trying to do (ask question, clarify, acknowledge, etc.)
- Sentiment: Emotional state and intensity

Replaces the rule-based TutoringWorkflow._detect_intent() and the
direct-LLM SentimentAnalyzer with a proper agent capability pattern.

Usage:
    capability = MessageAnalysisCapability()
    prompt = capability.build_prompt(
        {"message": "I don't understand this at all!"},
        context
    )
    # Agent sends prompt to LLM
    result = capability.parse_response(llm_response)
    # result.intent = "clarification"
    # result.emotional_state = "frustrated"
    # result.intensity = "high"
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.core.emotional import EmotionalIntensity, EmotionalState


class MessageAnalysisParams(BaseModel):
    """Parameters for message analysis.

    Attributes:
        message: The student's message to analyze.
        conversation_history: Recent conversation context (optional).
        language: Language of the message.
    """

    message: str = Field(
        description="The student's message to analyze",
        min_length=1,
    )
    conversation_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Recent conversation messages for context",
    )
    language: str = Field(
        default="en",
        description="Language of the message",
    )


class MessageIntent(str):
    """Intent types for student messages."""

    QUESTION = "question"
    CLARIFICATION = "clarification"
    ACKNOWLEDGMENT = "acknowledgment"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    GREETING = "greeting"
    FAREWELL = "farewell"
    OFF_TOPIC = "off_topic"
    HELP_REQUEST = "help_request"
    ANSWER_ATTEMPT = "answer_attempt"


class MessageAnalysisResult(CapabilityResult):
    """Result of message analysis.

    Attributes:
        message: The analyzed message (echoed back).
        intent: Detected intent of the message.
        intent_confidence: Confidence in intent detection (0.0-1.0).
        emotional_state: Detected emotional state.
        intensity: Intensity of the emotional state.
        sentiment_confidence: Confidence in sentiment detection (0.0-1.0).
        triggers: Factors that triggered this emotional state.
        requires_support: Whether the student needs emotional support.
        suggested_response_tone: Suggested tone for responding.
        analysis_notes: Additional analysis notes from LLM.
    """

    message: str = Field(description="The analyzed message")
    intent: str = Field(description="Detected intent")
    intent_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Intent confidence",
    )
    emotional_state: str = Field(
        default="neutral",
        description="Detected emotional state",
    )
    intensity: str = Field(
        default="low",
        description="Emotional intensity: low, moderate, high",
    )
    sentiment_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Sentiment confidence",
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="Emotional triggers identified",
    )
    requires_support: bool = Field(
        default=False,
        description="Whether emotional support is needed",
    )
    suggested_response_tone: str = Field(
        default="neutral",
        description="Suggested tone for response",
    )
    analysis_notes: str | None = Field(
        default=None,
        description="Additional analysis notes",
    )

    def get_emotional_state(self) -> EmotionalState:
        """Convert string emotional_state to EmotionalState enum.

        Returns:
            EmotionalState enum value.
        """
        state_map = {
            "neutral": EmotionalState.NEUTRAL,
            "curious": EmotionalState.CURIOUS,
            "engaged": EmotionalState.ENGAGED,
            "confident": EmotionalState.CONFIDENT,
            "confused": EmotionalState.CONFUSED,
            "frustrated": EmotionalState.FRUSTRATED,
            "anxious": EmotionalState.ANXIOUS,
            "bored": EmotionalState.BORED,
        }
        return state_map.get(self.emotional_state.lower(), EmotionalState.NEUTRAL)

    def get_intensity(self) -> EmotionalIntensity:
        """Convert string intensity to EmotionalIntensity enum.

        Returns:
            EmotionalIntensity enum value.
        """
        intensity_map = {
            "low": EmotionalIntensity.LOW,
            "moderate": EmotionalIntensity.MODERATE,
            "high": EmotionalIntensity.HIGH,
        }
        return intensity_map.get(self.intensity.lower(), EmotionalIntensity.LOW)


class MessageAnalysisCapability(Capability):
    """Capability for analyzing student messages.

    Combines intent detection and sentiment analysis in a single LLM call.
    This replaces rule-based intent detection and the direct-LLM
    SentimentAnalyzer with a proper capability pattern.

    Example:
        capability = MessageAnalysisCapability()
        params = {"message": "I don't understand fractions"}
        prompt = capability.build_prompt(params, context)
        # Agent handles LLM call
        result = capability.parse_response(llm_response)
        # result.intent = "clarification"
        # result.emotional_state = "confused"
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "message_analysis"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Analyzes student messages for intent and emotional sentiment"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate analysis parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            MessageAnalysisParams(**params)
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
        """Build prompt for message analysis.

        Args:
            params: Analysis parameters including the message.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = MessageAnalysisParams(**params)

        # Store for parse_response
        self._last_params = p

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are an expert educational psychologist analyzing student messages. "
            "Your task is to understand both what the student wants (intent) and how they feel (sentiment). "
            "Be sensitive to subtle emotional cues, especially frustration, confusion, and anxiety. "
            "Students may not explicitly state their emotions, so look for implicit signals."
        )

        # Emotional states guide
        system_parts.append(
            "Emotional states to detect:\n"
            "- neutral: No strong emotion detected\n"
            "- curious: Genuine interest, wanting to learn more\n"
            "- engaged: Actively participating, motivated\n"
            "- confident: Self-assured, comfortable with material\n"
            "- confused: Struggling to understand, lost\n"
            "- frustrated: Annoyed, struggling repeatedly\n"
            "- anxious: Worried, stressed about performance\n"
            "- bored: Disengaged, uninterested"
        )

        # Intent guide
        system_parts.append(
            "Intent types to detect:\n"
            "- question: Asking for information or explanation\n"
            "- clarification: Requesting clearer explanation of something\n"
            "- acknowledgment: Confirming understanding (ok, got it, I see)\n"
            "- confusion: Expressing lack of understanding\n"
            "- frustration: Expressing difficulty or annoyance\n"
            "- greeting: Hello, hi, starting conversation\n"
            "- farewell: Goodbye, ending conversation\n"
            "- off_topic: Unrelated to learning\n"
            "- help_request: Explicitly asking for help\n"
            "- answer_attempt: Trying to answer a question"
        )

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(f"Analysis persona context:\n{persona_prompt}")

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(f"Student context:\n{student_summary}")

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(f"Analyze this student message:\n\n\"{p.message}\"")

        # Add conversation history if provided
        if p.conversation_history:
            history_text = "\n".join(
                f"- {msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in p.conversation_history[-5:]  # Last 5 messages
            )
            user_parts.append(f"\nRecent conversation context:\n{history_text}")

        user_parts.append(f"\nLanguage: {p.language}")

        # Output format
        user_parts.append(self._get_output_format_instruction())

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_output_format_instruction(self) -> str:
        """Get the JSON output format instruction.

        Returns:
            Output format instruction string.
        """
        return """
Respond with a valid JSON object in this exact format:
```json
{
  "intent": "clarification",
  "intent_confidence": 0.85,
  "emotional_state": "confused",
  "intensity": "moderate",
  "sentiment_confidence": 0.80,
  "triggers": ["difficulty understanding concept", "repeated attempts"],
  "requires_support": true,
  "suggested_response_tone": "encouraging",
  "analysis_notes": "Student appears to be struggling with the concept and may need additional scaffolding"
}
```

Rules:
- intent must be one of: question, clarification, acknowledgment, confusion, frustration, greeting, farewell, off_topic, help_request, answer_attempt
- emotional_state must be one of: neutral, curious, engaged, confident, confused, frustrated, anxious, bored
- intensity must be one of: low, moderate, high
- requires_support should be true for frustrated, anxious, or confused states with moderate/high intensity
- suggested_response_tone should match the emotional support needed (encouraging, supportive, calm, neutral, enthusiastic)
"""

    def parse_response(self, response: str) -> MessageAnalysisResult:
        """Parse LLM response into MessageAnalysisResult.

        Args:
            response: Raw LLM response text.

        Returns:
            MessageAnalysisResult with analysis.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        data = self._extract_json_from_response(response)

        # Get original message
        message = ""
        if hasattr(self, "_last_params") and self._last_params:
            message = self._last_params.message

        # Validate and normalize emotional_state
        emotional_state = data.get("emotional_state", "neutral").lower()
        valid_states = {"neutral", "curious", "engaged", "confident", "confused", "frustrated", "anxious", "bored"}
        if emotional_state not in valid_states:
            emotional_state = "neutral"

        # Validate and normalize intensity
        intensity = data.get("intensity", "low").lower()
        valid_intensities = {"low", "moderate", "high"}
        if intensity not in valid_intensities:
            intensity = "low"

        # Validate and normalize intent
        intent = data.get("intent", "question").lower()
        valid_intents = {
            "question", "clarification", "acknowledgment", "confusion",
            "frustration", "greeting", "farewell", "off_topic",
            "help_request", "answer_attempt"
        }
        if intent not in valid_intents:
            intent = "question"

        return MessageAnalysisResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            message=message,
            intent=intent,
            intent_confidence=float(data.get("intent_confidence", 0.8)),
            emotional_state=emotional_state,
            intensity=intensity,
            sentiment_confidence=float(data.get("sentiment_confidence", 0.8)),
            triggers=data.get("triggers", []),
            requires_support=data.get("requires_support", False),
            suggested_response_tone=data.get("suggested_response_tone", "neutral"),
            analysis_notes=data.get("analysis_notes"),
        )
