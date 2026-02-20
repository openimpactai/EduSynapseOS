# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Question generation capability for educational content.

This capability generates educational questions tailored to:
- Student's current mastery level and learning style
- Target difficulty and cognitive level (Bloom's taxonomy)
- Specific question types (multiple choice, open-ended, etc.)
- Persona communication style

The generated questions include:
- Question content
- Answer options (for applicable types)
- Correct answer
- Hints at multiple levels
- Explanation for the correct answer
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.models.common import BloomLevel
from src.models.practice import QuestionType


class QuestionGenerationParams(BaseModel):
    """Parameters for question generation.

    Attributes:
        topic: The topic for the question.
        topic_description: Optional detailed description of the topic.
        difficulty: Target difficulty level (0.0-1.0).
        bloom_level: Target Bloom's taxonomy level.
        question_type: Type of question to generate.
        language: Language for the question (default: en). Usually derived from curriculum.
        include_hints: Whether to generate hints.
        hint_count: Number of hints to generate (1-3).
        include_explanation: Whether to include answer explanation.
        context_text: Additional context to inform question generation.
        avoid_concepts: Concepts to avoid in the question.
    """

    topic: str = Field(
        description="The topic for the question",
        min_length=1,
    )
    topic_description: str | None = Field(
        default=None,
        description="Detailed description of the topic",
    )
    difficulty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Target difficulty level (0.0-1.0)",
    )
    bloom_level: BloomLevel = Field(
        default=BloomLevel.UNDERSTAND,
        description="Target Bloom's taxonomy level",
    )
    question_type: QuestionType = Field(
        default=QuestionType.MULTIPLE_CHOICE,
        description="Type of question to generate",
    )
    language: str = Field(
        default="en",
        description="Language for the question (usually derived from curriculum country code)",
    )
    include_hints: bool = Field(
        default=True,
        description="Whether to generate hints",
    )
    hint_count: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Number of hints to generate",
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include answer explanation",
    )
    context_text: str | None = Field(
        default=None,
        description="Additional context to inform question generation",
    )
    avoid_concepts: list[str] = Field(
        default_factory=list,
        description="Concepts to avoid in the question",
    )


class QuestionOption(BaseModel):
    """A single option for multiple choice questions.

    Attributes:
        key: Option key (a, b, c, d).
        text: Option text content.
        is_correct: Whether this is the correct answer.
    """

    key: str = Field(description="Option key (a, b, c, d)")
    text: str = Field(description="Option text content")
    is_correct: bool = Field(default=False, description="Whether correct")


class QuestionHint(BaseModel):
    """A hint for the question.

    Attributes:
        level: Hint level (1=subtle, 2=moderate, 3=direct).
        text: Hint content.
    """

    level: int = Field(ge=1, le=3, description="Hint level (1-3)")
    text: str = Field(description="Hint content")


class GeneratedQuestion(CapabilityResult):
    """Result of question generation.

    Attributes:
        content: The question text.
        question_type: Type of question generated.
        difficulty: Actual difficulty level.
        bloom_level: Bloom's taxonomy level.
        options: Answer options (for multiple choice).
        correct_answer: The correct answer.
        reasoning: Step-by-step reasoning explaining why the answer is correct.
        hints: List of hints at different levels.
        explanation: Explanation for the correct answer.
        topic: Topic of the question.
        language: Language of the question.
        misconceptions_addressed: Common misconceptions this question tests.
    """

    content: str = Field(description="The question text")
    question_type: QuestionType = Field(description="Type of question")
    difficulty: float = Field(description="Difficulty level")
    bloom_level: BloomLevel = Field(description="Bloom's taxonomy level")
    options: list[QuestionOption] | None = Field(
        default=None,
        description="Answer options for multiple choice",
    )
    correct_answer: str = Field(description="The correct answer")
    reasoning: str | None = Field(
        default=None,
        description="Step-by-step reasoning for why the answer is correct",
    )
    hints: list[QuestionHint] = Field(
        default_factory=list,
        description="Hints at different levels",
    )
    explanation: str | None = Field(
        default=None,
        description="Explanation for the correct answer",
    )
    topic: str = Field(description="Topic of the question")
    language: str = Field(default="en", description="Language")
    misconceptions_addressed: list[str] = Field(
        default_factory=list,
        description="Misconceptions this question tests",
    )


class QuestionGenerationCapability(Capability):
    """Capability for generating educational questions.

    Generates questions tailored to student context, educational theories,
    and specified parameters like difficulty and Bloom level.

    Example:
        capability = QuestionGenerationCapability()
        params = QuestionGenerationParams(
            topic="Fractions",
            difficulty=0.6,
            bloom_level=BloomLevel.APPLY,
            question_type=QuestionType.MULTIPLE_CHOICE,
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        # Agent sends prompt to LLM, gets response
        result = capability.parse_response(llm_response)
    """

    def __init__(self) -> None:
        """Initialize the capability with default target values."""
        self._target_difficulty: float = 0.5
        self._target_bloom_level: BloomLevel = BloomLevel.UNDERSTAND
        self._target_question_type: QuestionType = QuestionType.MULTIPLE_CHOICE

    @property
    def name(self) -> str:
        """Return capability name."""
        return "question_generation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Generates educational questions based on topic, difficulty, and student context"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate question generation parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            QuestionGenerationParams(**params)
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
        """Build prompt for question generation.

        Args:
            params: Question generation parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = QuestionGenerationParams(**params)

        # Store target values for use in parse_response
        self._target_difficulty = p.difficulty
        self._target_bloom_level = p.bloom_level
        self._target_question_type = p.question_type

        # Build system message
        system_parts = []

        # Base instruction with strict topic adherence
        system_parts.append(
            "You are an expert educational content creator. "
            "Generate high-quality questions that are pedagogically sound, "
            "age-appropriate, and aligned with learning objectives.\n\n"
            "CRITICAL RULES:\n"
            "1. Questions MUST be DIRECTLY and EXCLUSIVELY about the specified topic\n"
            "2. Questions MUST be appropriate for the specified grade level and age range\n"
            "3. Questions MUST align with the curriculum, subject, and unit context provided\n"
            "4. DO NOT generate questions about tangential, related, or unrelated concepts"
        )

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        # Add educational context (grade level, subject, curriculum) - CRITICAL for age-appropriate content
        educational_context = context.get_educational_context()
        if educational_context:
            system_parts.append(educational_context)

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

        # Get language from context (curriculum-derived) or fall back to params
        language = context.get_language() if context else p.language

        # Get subject context for topic emphasis
        subject_name = context.additional.get("subject_name") if context and context.additional else None
        unit_name = context.additional.get("unit_name") if context and context.additional else None

        # Main instruction with strong topic emphasis
        topic_context = f"Topic: {p.topic}"
        if unit_name:
            topic_context = f"Unit: {unit_name} → Topic: {p.topic}"
        if subject_name:
            topic_context = f"Subject: {subject_name} → {topic_context}"

        user_parts.append(f"Generate a {p.question_type.value} question.\n\n{topic_context}")
        user_parts.append(f"\nThe question MUST test knowledge/skills specifically related to '{p.topic}'. Do not generate questions about other topics.")

        if p.topic_description:
            user_parts.append(f"Topic details: {p.topic_description}")

        # Parameters
        user_parts.append(f"\nRequirements:")
        user_parts.append(f"- Difficulty: {p.difficulty:.0%} (0%=very easy, 100%=very hard)")
        user_parts.append(f"- Bloom level: {p.bloom_level.value}")
        user_parts.append(f"- Language: {language}")

        # Add age-appropriate guidance if grade level is available
        grade_level = context.additional.get("grade_level") if context and context.additional else None
        age_range = context.additional.get("age_range") if context and context.additional else None
        if grade_level:
            if age_range:
                user_parts.append(f"- IMPORTANT: Content must be appropriate for {grade_level} students ({age_range})")
            else:
                user_parts.append(f"- IMPORTANT: Content must be appropriate for {grade_level} students")

        # Add learning objective if provided (for focused question generation)
        learning_objective = context.additional.get("learning_objective") if context and context.additional else None
        if learning_objective:
            user_parts.append(f"- Learning Objective: {learning_objective}")

        # Add concept focus if provided (for specific concept targeting)
        concept_focus = context.additional.get("concept_focus") if context and context.additional else None
        concept_description = context.additional.get("concept_description") if context and context.additional else None
        if concept_focus:
            if concept_description:
                user_parts.append(f"- Concept Focus: {concept_focus} - {concept_description}")
            else:
                user_parts.append(f"- Concept Focus: {concept_focus}")

        if p.question_type == QuestionType.MULTIPLE_CHOICE:
            user_parts.append("- Include 4 options (a, b, c, d) with exactly one correct answer")
            user_parts.append("- Distractors should represent common misconceptions")

        if p.include_hints:
            user_parts.append(f"- Include {p.hint_count} hints (level 1=subtle, 2=moderate, 3=direct)")

        if p.include_explanation:
            user_parts.append("- Include a detailed explanation for the correct answer")

        if p.avoid_concepts:
            user_parts.append(f"- Avoid these concepts: {', '.join(p.avoid_concepts)}")

        # Add RAG context if available
        rag_context = context.get_rag_context(max_results=3)
        if rag_context:
            user_parts.append(f"\n{rag_context}")

        if p.context_text:
            user_parts.append(f"\nAdditional context: {p.context_text}")

        # Chain-of-Thought instruction for answer accuracy and topic relevance
        user_parts.append(f"""
CRITICAL - Verification Checklist:
Before finalizing, verify:
1. TOPIC RELEVANCE: Is this question directly about '{p.topic}'? (If not, regenerate)
2. GRADE APPROPRIATENESS: Is the complexity suitable for the specified grade level?
3. ANSWER ACCURACY: Think step by step - is the correct answer definitely correct?
4. Include your verification in the "reasoning" field

This applies to ALL subjects (math, science, history, languages, etc.).""")

        # Output format
        user_parts.append(self._get_output_format_instruction(p))

        user_message = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_output_format_instruction(self, params: QuestionGenerationParams) -> str:
        """Get the JSON output format instruction.

        Args:
            params: Question generation parameters.

        Returns:
            Output format instruction string.
        """
        format_parts = [
            "\nRespond with a valid JSON object in this exact format:",
            "```json",
            "{",
            '  "content": "The question text",',
        ]

        if params.question_type == QuestionType.MULTIPLE_CHOICE:
            format_parts.append('  "options": [')
            format_parts.append('    {"key": "a", "text": "Option A", "is_correct": false},')
            format_parts.append('    {"key": "b", "text": "Option B", "is_correct": true},')
            format_parts.append('    {"key": "c", "text": "Option C", "is_correct": false},')
            format_parts.append('    {"key": "d", "text": "Option D", "is_correct": false}')
            format_parts.append('  ],')

        format_parts.append('  "correct_answer": "The correct answer text",')
        format_parts.append('  "reasoning": "Step-by-step verification: 1) ... 2) ... Therefore, the correct answer is ...",')

        if params.include_hints:
            format_parts.append('  "hints": [')
            for i in range(1, params.hint_count + 1):
                comma = "," if i < params.hint_count else ""
                format_parts.append(f'    {{"level": {i}, "text": "Hint level {i}"}}{comma}')
            format_parts.append('  ],')

        if params.include_explanation:
            format_parts.append('  "explanation": "Why this answer is correct",')

        format_parts.append('  "misconceptions_addressed": ["misconception1", "misconception2"]')
        format_parts.append("}")
        format_parts.append("```")

        return "\n".join(format_parts)

    def parse_response(self, response: str) -> GeneratedQuestion:
        """Parse LLM response into GeneratedQuestion.

        Args:
            response: Raw LLM response text.

        Returns:
            GeneratedQuestion result.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            # If JSON extraction fails, try to construct from plain text
            return self._parse_plain_text_response(response)

        # Parse options if present
        options = None
        if "options" in data and data["options"]:
            options = [
                QuestionOption(
                    key=opt.get("key", chr(97 + i)),  # a, b, c, d
                    text=opt.get("text", ""),
                    is_correct=opt.get("is_correct", False),
                )
                for i, opt in enumerate(data["options"])
            ]

        # Parse hints if present
        hints = []
        if "hints" in data and data["hints"]:
            hints = [
                QuestionHint(
                    level=h.get("level", i + 1),
                    text=h.get("text", ""),
                )
                for i, h in enumerate(data["hints"])
            ]

        # Use the target question type (set during build_prompt)
        # Fall back to inferring from options only if target is MCQ but no options returned
        question_type = self._target_question_type
        if question_type == QuestionType.MULTIPLE_CHOICE and not options:
            # LLM didn't return options for MCQ, fall back to short answer
            question_type = QuestionType.SHORT_ANSWER

        # Use target difficulty from params (set in build_prompt)
        # This ensures ZPD-calculated difficulty is preserved through the pipeline
        difficulty = self._target_difficulty

        # Use target Bloom level from params (set in build_prompt)
        bloom_level = self._target_bloom_level

        return GeneratedQuestion(
            success=True,
            capability_name=self.name,
            raw_response=response,
            content=data.get("content", ""),
            question_type=question_type,
            difficulty=difficulty,
            bloom_level=bloom_level,
            options=options,
            correct_answer=data.get("correct_answer", ""),
            reasoning=data.get("reasoning"),
            hints=hints,
            explanation=data.get("explanation"),
            topic=data.get("topic", ""),
            language=data.get("language", "tr"),
            misconceptions_addressed=data.get("misconceptions_addressed", []),
        )

    def _parse_plain_text_response(self, response: str) -> GeneratedQuestion:
        """Parse a plain text response when JSON parsing fails.

        Args:
            response: Plain text response.

        Returns:
            GeneratedQuestion with content extracted from text.
        """
        lines = response.strip().split("\n")
        content = lines[0] if lines else response

        return GeneratedQuestion(
            success=True,
            capability_name=self.name,
            raw_response=response,
            content=content,
            question_type=QuestionType.SHORT_ANSWER,
            difficulty=self._target_difficulty,
            bloom_level=self._target_bloom_level,
            options=None,
            correct_answer="",
            reasoning=None,
            hints=[],
            explanation=None,
            topic="",
            language="tr",
            misconceptions_addressed=[],
            metadata={"parse_method": "plain_text_fallback"},
        )

    def _estimate_difficulty(self, content: str, data: dict) -> float:
        """Estimate difficulty from question content.

        Args:
            content: Question content.
            data: Full parsed data.

        Returns:
            Estimated difficulty (0.0-1.0).
        """
        # Simple heuristics
        score = 0.5

        # Longer questions tend to be harder
        if len(content) > 200:
            score += 0.1
        elif len(content) < 50:
            score -= 0.1

        # More options = potentially harder
        if data.get("options") and len(data["options"]) > 4:
            score += 0.1

        # Presence of misconceptions suggests complexity
        if data.get("misconceptions_addressed"):
            score += 0.05 * len(data["misconceptions_addressed"])

        return min(1.0, max(0.0, score))

    def _estimate_bloom_level(self, content: str) -> BloomLevel:
        """Estimate Bloom's taxonomy level from question content.

        Args:
            content: Question content.

        Returns:
            Estimated BloomLevel.
        """
        content_lower = content.lower()

        # Create/Evaluate keywords
        if any(kw in content_lower for kw in [
            "design", "create", "develop", "propose", "construct",
            "evaluate", "judge", "justify", "critique", "assess"
        ]):
            return BloomLevel.CREATE

        # Analyze keywords
        if any(kw in content_lower for kw in [
            "analyze", "compare", "contrast", "examine", "investigate",
            "categorize", "differentiate", "distinguish"
        ]):
            return BloomLevel.ANALYZE

        # Apply keywords
        if any(kw in content_lower for kw in [
            "apply", "solve", "calculate", "use", "demonstrate",
            "implement", "execute", "show"
        ]):
            return BloomLevel.APPLY

        # Understand keywords
        if any(kw in content_lower for kw in [
            "explain", "describe", "summarize", "interpret", "classify",
            "discuss", "identify", "report"
        ]):
            return BloomLevel.UNDERSTAND

        # Remember keywords
        if any(kw in content_lower for kw in [
            "what", "when", "where", "who", "list", "name", "define",
            "recall", "recognize", "state"
        ]):
            return BloomLevel.REMEMBER

        return BloomLevel.UNDERSTAND
