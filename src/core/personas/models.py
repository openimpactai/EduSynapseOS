# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persona data models for EduSynapseOS.

This module defines the Pydantic models for the persona system, which determines
HOW an AI agent communicates with students. Personas define identity, voice,
and response templates for different teaching styles.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Tone(str, Enum):
    """Voice tone options for persona communication."""

    FORMAL = "formal"
    INFORMAL = "informal"
    MOTIVATIONAL = "motivational"
    SUPPORTIVE = "supportive"
    CHALLENGING = "challenging"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    WARM = "warm"
    CALM = "calm"
    ENCOURAGING = "encouraging"


class Formality(str, Enum):
    """Formality level options."""

    VERY_FORMAL = "very_formal"
    FORMAL = "formal"
    NEUTRAL = "neutral"
    INFORMAL = "informal"
    VERY_INFORMAL = "very_informal"


class EmojiUsage(str, Enum):
    """Emoji usage level options."""

    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    FREQUENT = "frequent"


class PersonaIdentity(BaseModel):
    """Defines the persona's identity and character.

    Attributes:
        role: The persona's role description (e.g., "Motivational coach")
        character: Detailed character description and personality traits
        background: Optional background story for the persona
        expertise: Optional areas of expertise
    """

    role: str = Field(
        ...,
        description="The persona's role description",
        min_length=1,
    )
    character: str = Field(
        ...,
        description="Detailed character description and personality traits",
        min_length=1,
    )
    background: Optional[str] = Field(
        default=None,
        description="Optional background story for the persona",
    )
    expertise: Optional[list[str]] = Field(
        default=None,
        description="Optional areas of expertise",
    )


class PersonaVoice(BaseModel):
    """Defines how the persona communicates.

    Attributes:
        tone: The overall tone of communication
        formality: Level of formality in language
        language: Primary language code (e.g., "tr", "en")
        emoji_usage: How often emojis are used
        vocabulary_level: Complexity of vocabulary (simple, moderate, advanced)
        sentence_style: Preferred sentence structure (short, varied, complex)
    """

    tone: Tone = Field(
        default=Tone.SUPPORTIVE,
        description="The overall tone of communication",
    )
    formality: Formality = Field(
        default=Formality.INFORMAL,
        description="Level of formality in language",
    )
    language: str = Field(
        default="en",
        description="Primary language code",
    )
    emoji_usage: EmojiUsage = Field(
        default=EmojiUsage.MODERATE,
        description="How often emojis are used",
    )
    vocabulary_level: str = Field(
        default="moderate",
        description="Complexity of vocabulary",
    )
    sentence_style: str = Field(
        default="varied",
        description="Preferred sentence structure",
    )


class PersonaTemplates(BaseModel):
    """Response templates for common situations.

    These templates are used to maintain consistent persona voice across
    different types of responses. Templates support placeholder variables
    that are filled in at runtime.

    Attributes:
        on_correct: Response when student answers correctly
        on_incorrect: Response when student answers incorrectly
        on_partial: Response when student partially correct
        encouragement: General encouragement message
        greeting: Greeting message at session start
        farewell: Farewell message at session end
        hint_intro: How to introduce a hint
        explanation_intro: How to start an explanation
        challenge: How to present a challenge
        celebration: How to celebrate achievements
        struggle_support: How to support a struggling student
        timeout_reminder: Message when student takes too long
    """

    on_correct: str = Field(
        default="Great! That's correct!",
        description="Response when student answers correctly",
    )
    on_incorrect: str = Field(
        default="Not quite, but you're getting close!",
        description="Response when student answers incorrectly",
    )
    on_partial: str = Field(
        default="Partially correct, let's think a bit more.",
        description="Response when student partially correct",
    )
    encouragement: str = Field(
        default="Keep going, you're doing great!",
        description="General encouragement message",
    )
    greeting: str = Field(
        default="Hello! What shall we learn together today?",
        description="Greeting message at session start",
    )
    farewell: str = Field(
        default="Great work today! See you later.",
        description="Farewell message at session end",
    )
    hint_intro: str = Field(
        default="Let me give you a hint:",
        description="How to introduce a hint",
    )
    explanation_intro: str = Field(
        default="Let's look at this together:",
        description="How to start an explanation",
    )
    challenge: str = Field(
        default="Ready? Here's a challenge for you!",
        description="How to present a challenge",
    )
    celebration: str = Field(
        default="Congratulations! Great achievement!",
        description="How to celebrate achievements",
    )
    struggle_support: str = Field(
        default="Don't worry, this is a tough topic. We'll figure it out together.",
        description="How to support a struggling student",
    )
    timeout_reminder: str = Field(
        default="Take your time to think, no rush!",
        description="Message when student takes too long",
    )


class PersonaBehavior(BaseModel):
    """Behavioral settings for the persona.

    Attributes:
        socratic_tendency: How much to use Socratic questioning (0-1)
        hint_eagerness: How quickly to offer hints (0-1, lower = more patient)
        explanation_depth: Preferred depth of explanations (shallow, moderate, deep)
        praise_frequency: How often to give praise (low, moderate, high)
        correction_style: How to handle corrections (direct, gentle, exploratory)
        patience_level: How patient with repeated mistakes (low, moderate, high)
    """

    socratic_tendency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much to use Socratic questioning",
    )
    hint_eagerness: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How quickly to offer hints",
    )
    explanation_depth: str = Field(
        default="moderate",
        description="Preferred depth of explanations",
    )
    praise_frequency: str = Field(
        default="moderate",
        description="How often to give praise",
    )
    correction_style: str = Field(
        default="gentle",
        description="How to handle corrections",
    )
    patience_level: str = Field(
        default="high",
        description="How patient with repeated mistakes",
    )


class Persona(BaseModel):
    """Complete persona definition.

    A persona defines HOW an AI agent communicates with students. It includes
    identity, voice characteristics, response templates, and behavioral settings.

    Attributes:
        id: Unique identifier for the persona
        name: Display name of the persona
        description: Brief description of the persona
        identity: The persona's identity and character
        voice: How the persona communicates
        templates: Response templates for common situations
        behavior: Behavioral settings
        suitable_for: Optional list of user types this persona suits
        tags: Optional tags for categorization
        enabled: Whether this persona is currently active
    """

    id: str = Field(
        ...,
        description="Unique identifier for the persona",
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    name: str = Field(
        ...,
        description="Display name of the persona",
        min_length=1,
    )
    description: str = Field(
        ...,
        description="Brief description of the persona",
        min_length=1,
    )
    identity: PersonaIdentity = Field(
        ...,
        description="The persona's identity and character",
    )
    voice: PersonaVoice = Field(
        default_factory=PersonaVoice,
        description="How the persona communicates",
    )
    templates: PersonaTemplates = Field(
        default_factory=PersonaTemplates,
        description="Response templates for common situations",
    )
    behavior: PersonaBehavior = Field(
        default_factory=PersonaBehavior,
        description="Behavioral settings",
    )
    suitable_for: Optional[list[str]] = Field(
        default=None,
        description="User types this persona suits (e.g., 'young_learners', 'advanced')",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Tags for categorization",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this persona is currently active",
    )

    def get_system_prompt_segment(self) -> str:
        """Generate a system prompt segment describing this persona.

        Returns:
            A formatted string that can be included in LLM system prompts
            to establish the persona's character and communication style.
        """
        parts = [
            f"You are {self.identity.role}.",
            "",
            "Character:",
            self.identity.character,
            "",
            f"Communication Style:",
            f"- Tone: {self.voice.tone.value}",
            f"- Formality: {self.voice.formality.value}",
            f"- Language: {self.voice.language}",
            f"- Emoji usage: {self.voice.emoji_usage.value}",
        ]

        if self.identity.expertise:
            parts.append("")
            parts.append(f"Areas of expertise: {', '.join(self.identity.expertise)}")

        parts.append("")
        parts.append("Behavioral Guidelines:")
        parts.append(
            f"- Socratic questioning tendency: {self.behavior.socratic_tendency:.0%}"
        )
        parts.append(f"- Explanation depth: {self.behavior.explanation_depth}")
        parts.append(f"- Correction style: {self.behavior.correction_style}")
        parts.append(f"- Patience level: {self.behavior.patience_level}")

        return "\n".join(parts)

    def format_response(self, template_name: str, **kwargs: str) -> str:
        """Format a template response with given parameters.

        Args:
            template_name: Name of the template to use (e.g., 'on_correct')
            **kwargs: Values to substitute in the template

        Returns:
            Formatted response string

        Raises:
            AttributeError: If template_name doesn't exist
        """
        template = getattr(self.templates, template_name)
        if kwargs:
            return template.format(**kwargs)
        return template
