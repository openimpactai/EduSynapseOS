# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base capability definitions and protocols.

This module provides the foundational abstractions for agent capabilities:
- CapabilityContext: Unified context from memory, theory, RAG, and persona
- Capability: Abstract base class for all capabilities
- CapabilityResult: Base class for capability outputs
- CapabilityError: Exception for capability-related errors

Capabilities are responsible for:
1. Building prompts from context (build_prompt)
2. Parsing LLM responses into structured outputs (parse_response)

Capabilities do NOT call the LLM - the Agent orchestrates that.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.educational.orchestrator import CombinedRecommendation
from src.core.memory.rag.retriever import RetrievalResult
from src.core.personas.models import Persona
from src.models.memory import FullMemoryContext


class CapabilityError(Exception):
    """Exception raised for capability-related errors.

    Attributes:
        message: Error description.
        capability_name: Name of the capability that raised the error.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        capability_name: str | None = None,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.capability_name = capability_name
        self.original_error = original_error
        super().__init__(self.message)


class CapabilityContext(BaseModel):
    """Context passed to capabilities for prompt building.

    This context is assembled by the Agent from various sources:
    - Memory: Student's learning history and state (4-layer memory)
    - Theory: Educational theory recommendations (7 theories combined)
    - RAG: Retrieved relevant documents (curriculum, memories, interests)
    - Persona: How the agent should communicate (tone, style, language)

    All fields are optional to allow partial context when full context
    is not available or needed.

    Attributes:
        memory: Full memory context from 4-layer memory system.
        theory: Combined recommendations from educational theories.
        rag_results: Retrieved documents from RAG pipeline.
        persona: Active persona for communication style.
        additional: Extra context data for specific capabilities.
    """

    memory: FullMemoryContext | None = Field(
        default=None,
        description="Full memory context from 4-layer memory system",
    )
    theory: CombinedRecommendation | None = Field(
        default=None,
        description="Combined recommendations from educational theories",
    )
    rag_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved documents from RAG pipeline",
    )
    persona: Persona | None = Field(
        default=None,
        description="Active persona for communication style",
    )
    additional: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context data for specific capabilities",
    )

    model_config = {"arbitrary_types_allowed": True}

    def get_persona_prompt(self) -> str:
        """Get the persona system prompt segment.

        Returns:
            Persona prompt if persona is set, empty string otherwise.
        """
        if self.persona:
            return self.persona.get_system_prompt_segment()
        return ""

    def get_theory_guidance(self) -> str:
        """Format theory recommendations as guidance text.

        Returns:
            Formatted theory guidance for prompt inclusion.
        """
        if not self.theory:
            return ""

        parts = [
            "Educational Guidance:",
            f"- Target difficulty: {self.theory.difficulty:.0%}",
            f"- Cognitive level: {self.theory.bloom_level.value}",
            f"- Content format: {self.theory.content_format.value}",
            f"- Scaffolding level: {self.theory.scaffold_level.name}",
            f"- Hints enabled: {self.theory.hints_enabled}",
            f"- Guide vs tell ratio: {self.theory.guide_vs_tell_ratio:.0%}",
            f"- Questioning style: {self.theory.questioning_style.value}",
        ]
        return "\n".join(parts)

    def get_rag_context(self, max_results: int = 5) -> str:
        """Format RAG results as context text.

        Args:
            max_results: Maximum number of results to include.

        Returns:
            Formatted RAG context for prompt inclusion.
        """
        if not self.rag_results:
            return ""

        parts = ["Relevant Context:"]
        for result in self.rag_results[:max_results]:
            source_label = result.source.value.title()
            parts.append(f"[{source_label}] {result.content}")

        return "\n".join(parts)

    def get_student_summary(self) -> str:
        """Get a summary of student context from memory.

        Returns:
            Student summary for prompt inclusion.
        """
        if not self.memory:
            return ""

        parts = ["Student Context:"]

        # Semantic - mastery
        if self.memory.semantic:
            parts.append(
                f"- Overall mastery: {self.memory.semantic.overall_mastery:.0%}"
            )
            parts.append(f"- Topics mastered: {self.memory.semantic.topics_mastered}")
            parts.append(
                f"- Topics struggling: {self.memory.semantic.topics_struggling}"
            )

        # Procedural - learning patterns
        if self.memory.procedural:
            proc = self.memory.procedural
            if proc.preferred_content_format:
                parts.append(
                    f"- Preferred format: {proc.preferred_content_format}"
                )
            if proc.best_time_of_day:
                parts.append(f"- Best study time: {proc.best_time_of_day}")
            if proc.vark_profile:
                dominant = proc.vark_profile.dominant_style
                parts.append(f"- Learning style: {dominant.value}")

        # Associative - interests
        if self.memory.associative and self.memory.associative.interests:
            interests = [i.content for i in self.memory.associative.interests[:3]]
            if interests:
                parts.append(f"- Interests: {', '.join(interests)}")

        return "\n".join(parts) if len(parts) > 1 else ""

    def get_educational_context(self) -> str:
        """Get educational context for age-appropriate content generation.

        This method extracts curriculum hierarchy information from the additional
        context, including grade level, subject, and curriculum details. This is
        critical for generating content appropriate for the student's age and
        educational level.

        Returns:
            Formatted educational context for prompt inclusion.
        """
        if not self.additional:
            return ""

        parts = []

        # Grade level and age range - most critical for age-appropriate content
        grade_level = self.additional.get("grade_level")
        age_range = self.additional.get("age_range")
        if grade_level:
            if age_range:
                parts.append(f"- Grade Level: {grade_level} (ages {age_range})")
            else:
                parts.append(f"- Grade Level: {grade_level}")

        # Subject and unit context
        subject_name = self.additional.get("subject_name")
        unit_name = self.additional.get("unit_name")
        if subject_name:
            if unit_name:
                parts.append(f"- Subject: {subject_name} - {unit_name}")
            else:
                parts.append(f"- Subject: {subject_name}")

        # Curriculum context
        curriculum = self.additional.get("curriculum")
        if curriculum:
            parts.append(f"- Curriculum: {curriculum}")

        if not parts:
            return ""

        return "Educational Context:\n" + "\n".join(parts)

    def get_language(self) -> str:
        """Get the language for content generation.

        Returns the language code from additional context, defaulting to English.

        Returns:
            Language code (e.g., "en", "tr").
        """
        return self.additional.get("language", "en") if self.additional else "en"


class CapabilityResult(BaseModel):
    """Base class for capability execution results.

    All capability-specific result types should inherit from this.

    Attributes:
        success: Whether the capability executed successfully.
        capability_name: Name of the capability that produced this result.
        generated_at: Timestamp when the result was generated.
        raw_response: Original LLM response text (for debugging).
        metadata: Additional metadata about the execution.
    """

    success: bool = Field(
        default=True,
        description="Whether the capability executed successfully",
    )
    capability_name: str = Field(
        description="Name of the capability that produced this result",
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the result was generated",
    )
    raw_response: str | None = Field(
        default=None,
        description="Original LLM response text",
        repr=False,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution",
    )


class Capability(ABC):
    """Abstract base class for all agent capabilities.

    A capability defines a specific function an agent can perform.
    Each capability is responsible for:
    1. Building prompts from provided context (memory, theory, RAG, persona)
    2. Parsing LLM responses into structured outputs

    The capability does NOT call the LLM directly. The Agent:
    1. Collects context (memory, theory, RAG)
    2. Calls capability.build_prompt(params, context)
    3. Sends prompt to LLM
    4. Calls capability.parse_response(llm_response)

    Example:
        class MyCapability(Capability):
            @property
            def name(self) -> str:
                return "my_capability"

            def build_prompt(self, params, context):
                return [
                    {"role": "system", "content": "You are a helpful tutor."},
                    {"role": "user", "content": f"Help with: {params['topic']}"}
                ]

            def parse_response(self, response):
                return MyResult(content=response)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this capability.

        Returns:
            Capability name (e.g., 'question_generation').
        """
        ...

    @property
    def description(self) -> str:
        """Return a description of what this capability does.

        Returns:
            Human-readable description.
        """
        return ""

    @abstractmethod
    def build_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> list[dict[str, str]]:
        """Build the prompt messages for LLM.

        Creates a list of messages (system, user, assistant roles) that
        will be sent to the LLM. Uses context from memory, theory, RAG,
        and persona to create a personalized prompt.

        Args:
            params: Capability-specific input parameters.
            context: Unified context from memory, theory, RAG, persona.

        Returns:
            List of message dicts with 'role' and 'content' keys.
            Roles can be: 'system', 'user', 'assistant'.

        Example:
            return [
                {"role": "system", "content": persona_prompt + theory_guidance},
                {"role": "user", "content": "Generate a question about fractions"}
            ]
        """
        ...

    @abstractmethod
    def parse_response(self, response: str) -> CapabilityResult:
        """Parse the LLM response into a structured result.

        Extracts structured data from the raw LLM response text.
        Should handle malformed responses gracefully.

        Args:
            response: Raw text response from LLM.

        Returns:
            Capability-specific result object.

        Raises:
            CapabilityError: If response cannot be parsed.
        """
        ...

    def build_user_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> str:
        """Build only the user prompt for YAML-driven system prompt mode.

        This method is used when the agent has system_prompt config in YAML.
        In that case, the system prompt comes from YAML via SystemPromptBuilder,
        and the capability only needs to provide the user prompt.

        By default, this extracts the user message from build_prompt().
        Capabilities can override this for optimized user prompt generation.

        Args:
            params: Capability-specific input parameters.
            context: Unified context from memory, theory, RAG, persona.

        Returns:
            User prompt string.

        Example:
            def build_user_prompt(self, params, context):
                return f"Generate a {params['type']} question about {params['topic']}"
        """
        # Default implementation: extract user message from build_prompt()
        messages = self.build_prompt(params, context)
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]

        # Fallback: combine all non-system messages
        non_system = [m["content"] for m in messages if m["role"] != "system"]
        return "\n\n".join(non_system)

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate input parameters before prompt building.

        Override this method to add parameter validation.
        Raises CapabilityError if validation fails.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        pass

    def _build_system_message(self, context: CapabilityContext) -> str:
        """Build the system message from context.

        Helper method to construct a system message combining
        persona, theory guidance, and student context.

        Args:
            context: The capability context.

        Returns:
            Combined system message content.
        """
        parts = []

        # Persona first (defines who the agent is)
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            parts.append(persona_prompt)

        # Theory guidance (how to approach pedagogically)
        theory_guidance = context.get_theory_guidance()
        if theory_guidance:
            parts.append(theory_guidance)

        # Student summary (who the student is)
        student_summary = context.get_student_summary()
        if student_summary:
            parts.append(student_summary)

        return "\n\n".join(parts) if parts else ""

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON object from LLM response.

        Handles common patterns like markdown code blocks.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            CapabilityError: If JSON cannot be extracted.
        """
        import json
        import re

        text = response.strip()

        # Try to find JSON in markdown code block
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        match = re.search(json_block_pattern, text)
        if match:
            text = match.group(1).strip()

        # Try to find JSON object directly
        json_object_pattern = r"\{[\s\S]*\}"
        match = re.search(json_object_pattern, text)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise CapabilityError(
                message=f"Failed to parse JSON from response: {e}",
                capability_name=self.name,
                original_error=e,
            ) from e

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"
