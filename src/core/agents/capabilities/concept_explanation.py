# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Concept explanation capability for educational content.

This capability generates concept explanations tailored to:
- Student's current understanding level
- Learning style preferences (VARK)
- Personal interests (for analogies)
- Educational theory recommendations

The explanations include:
- Core concept explanation
- Examples at appropriate level
- Analogies based on student interests
- Prerequisites if needed
- Related concepts for further learning
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.core.educational.theories.base import ContentFormat


class ConceptExplanationParams(BaseModel):
    """Parameters for concept explanation.

    Attributes:
        concept: The concept to explain.
        concept_context: Additional context about the concept.
        target_level: Target understanding level (beginner, intermediate, advanced).
        preferred_format: Preferred content format (VARK).
        include_examples: Whether to include examples.
        example_count: Number of examples to include.
        include_analogies: Whether to include analogies.
        include_prerequisites: Whether to list prerequisites.
        include_related: Whether to list related concepts.
        language: Language for the explanation.
        max_length: Maximum length hint (short, medium, long).
    """

    concept: str = Field(
        description="The concept to explain",
        min_length=1,
    )
    concept_context: str | None = Field(
        default=None,
        description="Additional context about the concept",
    )
    target_level: str = Field(
        default="intermediate",
        description="Target level: beginner, intermediate, advanced",
    )
    preferred_format: ContentFormat = Field(
        default=ContentFormat.MULTIMODAL,
        description="Preferred content format (VARK)",
    )
    include_examples: bool = Field(
        default=True,
        description="Whether to include examples",
    )
    example_count: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of examples to include",
    )
    include_analogies: bool = Field(
        default=True,
        description="Whether to include analogies",
    )
    include_prerequisites: bool = Field(
        default=True,
        description="Whether to list prerequisites",
    )
    include_related: bool = Field(
        default=True,
        description="Whether to list related concepts",
    )
    language: str = Field(
        default="en",
        description="Language for the explanation",
    )
    max_length: str = Field(
        default="medium",
        description="Length: short, medium, long",
    )


class ConceptExample(BaseModel):
    """An example illustrating the concept.

    Attributes:
        title: Brief title of the example.
        content: The example content.
        explanation: Why this example illustrates the concept.
    """

    title: str = Field(description="Brief title of the example")
    content: str = Field(description="The example content")
    explanation: str | None = Field(
        default=None,
        description="Why this illustrates the concept",
    )


class ConceptAnalogy(BaseModel):
    """An analogy to help understand the concept.

    Attributes:
        source_domain: The familiar domain used in analogy.
        target_domain: The concept being explained.
        mapping: How elements map between domains.
        limitations: Where the analogy breaks down.
    """

    source_domain: str = Field(description="The familiar domain")
    target_domain: str = Field(description="The concept domain")
    mapping: str = Field(description="How elements map")
    limitations: str | None = Field(
        default=None,
        description="Where the analogy breaks down",
    )


class ConceptExplanation(CapabilityResult):
    """Result of concept explanation.

    Attributes:
        concept: The concept that was explained.
        explanation: The main explanation text.
        summary: Brief summary of the concept.
        examples: Illustrative examples.
        analogies: Analogies to familiar concepts.
        prerequisites: Required prior knowledge.
        related_concepts: Related concepts for further study.
        key_points: Key takeaways.
        common_mistakes: Common misconceptions to avoid.
        practice_suggestions: Suggested practice activities.
        language: Language of the explanation.
        format_used: Content format that was used.
    """

    concept: str = Field(description="The concept explained")
    explanation: str = Field(description="Main explanation text")
    summary: str = Field(description="Brief summary")
    examples: list[ConceptExample] = Field(
        default_factory=list,
        description="Illustrative examples",
    )
    analogies: list[ConceptAnalogy] = Field(
        default_factory=list,
        description="Analogies to familiar concepts",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Required prior knowledge",
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Related concepts for further study",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Key takeaways",
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="Common misconceptions to avoid",
    )
    practice_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested practice activities",
    )
    language: str = Field(default="en", description="Language")
    format_used: ContentFormat = Field(
        default=ContentFormat.MULTIMODAL,
        description="Content format used",
    )


class ConceptExplanationCapability(Capability):
    """Capability for explaining educational concepts.

    Generates explanations tailored to student's level, learning style,
    and interests. Uses educational theory recommendations to determine
    appropriate depth and scaffolding.

    Example:
        capability = ConceptExplanationCapability()
        params = ConceptExplanationParams(
            concept="Fractions",
            target_level="beginner",
            preferred_format=ContentFormat.VISUAL,
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        # Agent sends prompt to LLM, gets response
        result = capability.parse_response(llm_response)
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "concept_explanation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Explains concepts adaptively based on student context and learning style"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate explanation parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            ConceptExplanationParams(**params)
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
        """Build prompt for concept explanation.

        Args:
            params: Explanation parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = ConceptExplanationParams(**params)

        # Override preferred_format from theory if available
        if context.theory and context.theory.content_format:
            p.preferred_format = context.theory.content_format

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are an expert educational tutor. "
            "Explain concepts clearly, using appropriate examples and analogies. "
            "Adapt your explanation to the student's level and learning style."
        )

        # Add format-specific guidance
        format_guidance = self._get_format_guidance(p.preferred_format)
        if format_guidance:
            system_parts.append(format_guidance)

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        # Add theory guidance if available
        theory_guidance = context.get_theory_guidance()
        if theory_guidance:
            system_parts.append(theory_guidance)

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(student_summary)

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append(f"Explain the concept: **{p.concept}**")

        if p.concept_context:
            user_parts.append(f"Context: {p.concept_context}")

        user_parts.append(f"\n**Requirements:**")
        user_parts.append(f"- Target level: {p.target_level}")
        user_parts.append(f"- Language: {p.language}")
        user_parts.append(f"- Length: {p.max_length}")
        user_parts.append(f"- Format preference: {p.preferred_format.value}")

        if p.include_examples:
            user_parts.append(f"- Include {p.example_count} concrete examples")

        if p.include_analogies:
            # Get student interests for better analogies
            interests = self._get_student_interests(context)
            if interests:
                user_parts.append(f"- Use analogies related to: {', '.join(interests)}")
            else:
                user_parts.append("- Include relatable analogies")

        if p.include_prerequisites:
            user_parts.append("- List any prerequisite knowledge needed")

        if p.include_related:
            user_parts.append("- Suggest related concepts for further learning")

        # Add RAG context
        rag_context = context.get_rag_context(max_results=3)
        if rag_context:
            user_parts.append(f"\n{rag_context}")

        # Output format
        user_parts.append(self._get_output_format_instruction(p))

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_format_guidance(self, format: ContentFormat) -> str:
        """Get guidance based on preferred content format.

        Args:
            format: The preferred content format.

        Returns:
            Format-specific guidance text.
        """
        guidance_map = {
            ContentFormat.VISUAL: (
                "Use visual descriptions. Describe diagrams, charts, or images "
                "that would help illustrate the concept. Use spatial language "
                "and visual metaphors."
            ),
            ContentFormat.AUDITORY: (
                "Use conversational language. Include rhythmic elements, "
                "mnemonic phrases, or verbal patterns that aid memory. "
                "Write as if speaking to the student."
            ),
            ContentFormat.READING: (
                "Provide detailed written explanations. Use clear structure "
                "with headings and bullet points. Include definitions and "
                "technical terminology with explanations."
            ),
            ContentFormat.KINESTHETIC: (
                "Include hands-on activities and physical analogies. "
                "Describe actions, movements, and experiments. "
                "Suggest practical exercises the student can do."
            ),
            ContentFormat.MULTIMODAL: (
                "Use a balanced mix of visual descriptions, clear explanations, "
                "and practical examples. Cater to multiple learning styles."
            ),
        }
        return guidance_map.get(format, "")

    def _get_student_interests(self, context: CapabilityContext) -> list[str]:
        """Extract student interests from context.

        Args:
            context: The capability context.

        Returns:
            List of student interests.
        """
        interests = []

        if context.memory and context.memory.associative:
            if context.memory.associative.interests:
                interests = [
                    i.content for i in context.memory.associative.interests[:3]
                ]

        return interests

    def _get_output_format_instruction(self, params: ConceptExplanationParams) -> str:
        """Get the JSON output format instruction.

        Args:
            params: Explanation parameters.

        Returns:
            Output format instruction string.
        """
        return """
Respond with a valid JSON object in this exact format:
```json
{
  "explanation": "The main detailed explanation of the concept",
  "summary": "A brief 1-2 sentence summary",
  "examples": [
    {
      "title": "Example Title",
      "content": "The example content",
      "explanation": "Why this illustrates the concept"
    }
  ],
  "analogies": [
    {
      "source_domain": "Familiar concept",
      "target_domain": "The concept being explained",
      "mapping": "How they relate",
      "limitations": "Where the analogy breaks down"
    }
  ],
  "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
  "related_concepts": ["Related concept 1", "Related concept 2"],
  "key_points": ["Key point 1", "Key point 2", "Key point 3"],
  "common_mistakes": ["Common mistake to avoid"],
  "practice_suggestions": ["Practice activity suggestion"]
}
```
"""

    def parse_response(self, response: str) -> ConceptExplanation:
        """Parse LLM response into ConceptExplanation.

        Args:
            response: Raw LLM response text.

        Returns:
            ConceptExplanation result.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        # Parse examples
        examples = []
        if "examples" in data and data["examples"]:
            for ex in data["examples"]:
                examples.append(
                    ConceptExample(
                        title=ex.get("title", "Example"),
                        content=ex.get("content", ""),
                        explanation=ex.get("explanation"),
                    )
                )

        # Parse analogies
        analogies = []
        if "analogies" in data and data["analogies"]:
            for an in data["analogies"]:
                analogies.append(
                    ConceptAnalogy(
                        source_domain=an.get("source_domain", ""),
                        target_domain=an.get("target_domain", ""),
                        mapping=an.get("mapping", ""),
                        limitations=an.get("limitations"),
                    )
                )

        return ConceptExplanation(
            success=True,
            capability_name=self.name,
            raw_response=response,
            concept=data.get("concept", ""),
            explanation=data.get("explanation", ""),
            summary=data.get("summary", ""),
            examples=examples,
            analogies=analogies,
            prerequisites=data.get("prerequisites", []),
            related_concepts=data.get("related_concepts", []),
            key_points=data.get("key_points", []),
            common_mistakes=data.get("common_mistakes", []),
            practice_suggestions=data.get("practice_suggestions", []),
            language=data.get("language", "tr"),
            format_used=ContentFormat.MULTIMODAL,
        )

    def _parse_plain_text_response(self, response: str) -> ConceptExplanation:
        """Parse a plain text response when JSON parsing fails.

        Args:
            response: Plain text response.

        Returns:
            ConceptExplanation with content extracted from text.
        """
        lines = response.strip().split("\n")

        # Use entire response as explanation
        explanation = response.strip()

        # Try to extract first sentence as summary
        first_sentence = ""
        if lines:
            first_line = lines[0].strip()
            if "." in first_line:
                first_sentence = first_line.split(".")[0] + "."
            else:
                first_sentence = first_line

        return ConceptExplanation(
            success=True,
            capability_name=self.name,
            raw_response=response,
            concept="",
            explanation=explanation,
            summary=first_sentence,
            examples=[],
            analogies=[],
            prerequisites=[],
            related_concepts=[],
            key_points=[],
            common_mistakes=[],
            practice_suggestions=[],
            language="tr",
            format_used=ContentFormat.MULTIMODAL,
            metadata={"parse_method": "plain_text_fallback"},
        )
