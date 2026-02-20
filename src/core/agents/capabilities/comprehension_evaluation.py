# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Comprehension evaluation capability for verifying genuine student understanding.

This capability evaluates whether a student truly understands a concept by analyzing
their explanation against key concepts. Unlike pattern-based detection (e.g., "I understand"),
this uses AI to verify genuine comprehension.

Key Features:
    - Analyzes student explanations against key concepts
    - Detects parroting (student repeating AI's words verbatim)
    - Identifies misconceptions
    - Provides actionable teaching suggestions
    - Determines readiness for practice/assessment

This capability is always LLM-based because comprehension evaluation requires
semantic understanding of student responses.

Trigger Points:
    - Student says "I understand" or similar patterns (SELF_REPORTED)
    - Mode transition from EXPLANATION to GUIDED_PRACTICE (MODE_TRANSITION)
    - Session completion verification (SESSION_END)
    - Periodic checkpoint every N turns (CHECKPOINT)
    - Explicit understanding check requested (EXPLICIT)

Example:
    capability = ComprehensionEvaluationCapability()
    params = {
        "student_explanation": "A fraction is like cutting a pizza...",
        "key_concepts": ["part-whole relationship", "numerator", "denominator"],
        "topic_name": "Introduction to Fractions",
        "ai_explanation": "A fraction represents a part of a whole...",
    }
    prompt = capability.build_prompt(params, context)
    llm_response = await llm.generate(prompt)
    result = capability.parse_response(llm_response)
"""

from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.models.comprehension import (
    ComprehensionEvaluationParams,
    ComprehensionEvaluationResult,
    ComprehensionTrigger,
    ConceptMasteryLevel,
    DetectedMisconception,
    KeyConceptAssessment,
    MisconceptionSeverity,
    UnderstandingLevel,
)


class ComprehensionEvaluationCapability(Capability):
    """Capability for evaluating student comprehension through AI analysis.

    This capability evaluates whether a student truly understands a concept
    by analyzing their explanation against key concepts. It detects parroting,
    identifies misconceptions, and provides actionable teaching suggestions.

    The capability is always LLM-based because comprehension evaluation requires
    semantic understanding of student responses.

    Attributes:
        COMPREHENSION_CHECK_PROMPTS: Subject-specific question templates for
            eliciting student explanations.

    Example:
        capability = ComprehensionEvaluationCapability()
        params = ComprehensionEvaluationParams(
            student_explanation="A fraction is like cutting a pizza...",
            key_concepts=["part-whole relationship", "numerator", "denominator"],
            topic_name="Introduction to Fractions",
        )
        prompt = capability.build_prompt(params.model_dump(), context)
        # Send to LLM and parse response
        result = capability.parse_response(llm_response)
    """

    # Subject-specific comprehension check question templates
    COMPREHENSION_CHECK_PROMPTS: dict[str, list[str]] = {
        "general": [
            "Can you explain in your own words what {topic} means?",
            "Pretend you're teaching this to a friend - how would you explain {topic}?",
            "What's the most important thing you learned about {topic}?",
            "How would you summarize what we discussed about {topic}?",
        ],
        "mathematics": [
            "Can you walk me through how you would solve a problem like this?",
            "Why do you think we use this method in this situation?",
            "Can you explain the steps in your own words?",
            "What would happen if we changed the numbers?",
        ],
        "science": [
            "Can you describe the process in your own words?",
            "Why do you think this happens?",
            "How does this concept connect to what we learned before?",
            "Can you give me an example from everyday life?",
        ],
        "history": [
            "Why do you think this event happened?",
            "What were the main causes and effects?",
            "How might things have been different if something changed?",
            "Can you explain the significance of this in your own words?",
        ],
        "language": [
            "Can you use this in a sentence of your own?",
            "What's the difference between these two concepts?",
            "When would you use this?",
            "Can you explain the rule in your own words?",
        ],
    }

    @property
    def name(self) -> str:
        """Return capability name."""
        return "comprehension_evaluation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return (
            "Evaluates student comprehension through AI analysis of explanations, "
            "detecting parroting and misconceptions"
        )

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate evaluation parameters.

        Args:
            params: Parameters to validate.

        Raises:
            CapabilityError: If parameters are invalid.
        """
        try:
            ComprehensionEvaluationParams(**params)
        except Exception as e:
            raise CapabilityError(
                message=f"Invalid parameters: {e}",
                capability_name=self.name,
                original_error=e,
            ) from e

    def get_comprehension_check_question(
        self,
        topic_name: str,
        subject: str = "general",
    ) -> str:
        """Get a comprehension check question for the topic.

        Selects an appropriate question template based on the subject
        and fills in the topic name.

        Args:
            topic_name: Name of the topic being learned.
            subject: Subject area (mathematics, science, etc.).

        Returns:
            A comprehension check question string.
        """
        subject_lower = subject.lower()

        # Map common subject variations to standard keys
        subject_mapping = {
            "math": "mathematics",
            "maths": "mathematics",
            "biology": "science",
            "physics": "science",
            "chemistry": "science",
            "social_studies": "history",
            "geography": "history",
            "english": "language",
            "literature": "language",
        }
        subject_key = subject_mapping.get(subject_lower, subject_lower)

        # Get templates for subject, fallback to general
        templates = self.COMPREHENSION_CHECK_PROMPTS.get(
            subject_key,
            self.COMPREHENSION_CHECK_PROMPTS["general"],
        )

        # Select first template and format with topic
        template = templates[0]
        return template.format(topic=topic_name)

    def build_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> list[dict[str, str]]:
        """Build prompt for LLM-based comprehension evaluation.

        Creates a detailed prompt that instructs the LLM to:
        1. Verify key concepts in student's explanation
        2. Detect parroting (copying AI's words)
        3. Identify misconceptions
        4. Assess depth of understanding
        5. Provide actionable teaching suggestions

        Args:
            params: Evaluation parameters (ComprehensionEvaluationParams).
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = ComprehensionEvaluationParams(**params)

        # Store for parse_response
        self._last_params = p

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(self._get_system_prompt())

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(student_summary)

        # Add educational context
        educational_context = context.get_educational_context()
        if educational_context:
            system_parts.append(educational_context)

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_message = self._build_user_prompt(p)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for comprehension evaluation."""
        return """You are an expert educational evaluator assessing student comprehension.

Your task is to evaluate whether a student truly understands a concept based on
their explanation. You must:

1. VERIFY KEY CONCEPTS
   - Check if each key concept is demonstrated in the student's response
   - Look for evidence of understanding, not just keywords
   - Assess the DEPTH of understanding (surface, functional, deep)
   - Surface: Can repeat definition but not explain
   - Functional: Can explain and give basic examples
   - Deep: Can apply, analyze, and make connections

2. DETECT PARROTING
   - Compare student's response to the AI's original explanation
   - Flag if student is just repeating back the same words
   - Value original formulations and personal analogies
   - Parroting score: 0.0 = completely original, 1.0 = exact copy

3. IDENTIFY MISCONCEPTIONS
   - Look for subtle errors in understanding
   - Identify common misconceptions for this topic
   - Note severity (critical, significant, minor)
   - Provide correction suggestions

4. ASSESS BLOOM'S LEVEL
   - Remember: Can recall facts
   - Understand: Can explain concepts
   - Apply: Can use knowledge in new situations
   - Analyze: Can break down concepts and find patterns
   - Evaluate: Can make judgments
   - Create: Can produce new ideas

5. PROVIDE ACTIONABLE OUTPUT
   - Clear understanding score (0.0-1.0)
   - Specific teaching suggestions
   - Student-friendly feedback
   - Next step recommendation

IMPORTANT:
- Be encouraging but accurate
- Consider age-appropriate expectations (grade_level)
- Value effort and progress
- Distinguish between "doesn't know" and "can't articulate well"
- Look for the essence of understanding, not perfect wording"""

    def _build_user_prompt(self, params: ComprehensionEvaluationParams) -> str:
        """Build the user prompt for evaluation."""
        parts = []

        parts.append("Evaluate this student's comprehension:\n")

        # Topic information
        parts.append(f"**Topic:** {params.topic_name}")
        if params.topic_description:
            parts.append(f"**Topic Description:** {params.topic_description}")

        # Key concepts
        parts.append("\n**Key Concepts to Verify:**")
        for i, concept in enumerate(params.key_concepts, 1):
            parts.append(f"  {i}. {concept}")

        # Learning objectives if available
        if params.learning_objectives:
            parts.append("\n**Learning Objectives:**")
            for i, obj in enumerate(params.learning_objectives, 1):
                parts.append(f"  {i}. {obj}")

        # AI's explanation for parroting detection
        if params.ai_explanation:
            parts.append(f"\n**AI's Original Explanation (for parroting detection):**\n{params.ai_explanation}")

        # Student's explanation
        parts.append(f"\n**Student's Explanation (to evaluate):**\n{params.student_explanation}")

        # Context
        parts.append("\n**Context:**")
        if params.grade_level:
            parts.append(f"- Grade Level: {params.grade_level}")
        parts.append(f"- Language: {params.language}")
        parts.append(f"- Trigger: {params.trigger.value}")
        if params.current_mastery > 0:
            parts.append(f"- Current Mastery: {params.current_mastery:.0%}")
        if params.session_turn_count > 0:
            parts.append(f"- Session Turn: {params.session_turn_count}")

        # Recent conversation context
        if params.conversation_context:
            parts.append("\n**Recent Conversation:**")
            for turn in params.conversation_context[-3:]:  # Last 3 turns
                role = turn.get("role", "unknown")
                content = turn.get("content", "")[:200]  # Truncate
                parts.append(f"  [{role}]: {content}")

        # Output format
        parts.append(self._get_output_format_instruction())

        return "\n".join(parts)

    def _get_output_format_instruction(self) -> str:
        """Get the JSON output format instruction."""
        return """

Evaluate the student's comprehension and respond with a valid JSON object in this exact format:
```json
{
  "understanding_score": 0.75,
  "understanding_level": "partial",
  "verified": false,
  "parroting_detected": false,
  "concept_assessments": [
    {
      "concept": "concept name",
      "mastery_level": "understood",
      "score": 0.8,
      "mentioned": true,
      "understood": true,
      "explanation_quality": "good",
      "evidence": "quote from student showing understanding",
      "gaps": [],
      "feedback": "Good understanding of this concept"
    }
  ],
  "concepts_understood": ["list of understood concepts"],
  "concepts_missing": ["list of concepts not demonstrated"],
  "concepts_weak": ["list of concepts with weak understanding"],
  "misconceptions": [
    {
      "id": "MISC_001",
      "description": "description of the misconception",
      "severity": "significant",
      "related_concept": "concept name",
      "student_statement": "what student said",
      "correct_understanding": "what it should be",
      "correction_suggestion": "how to address"
    }
  ],
  "overall_feedback": "Encouraging feedback for the student",
  "detailed_feedback": {
    "strengths": "What the student did well",
    "areas_to_improve": "What needs work",
    "next_steps": "Suggested next steps"
  },
  "confidence": 0.9,
  "recommended_action": "continue_learning",
  "should_continue_learning": true,
  "ready_for_practice": false,
  "ready_for_assessment": false,
  "clarification_needed": ["concepts needing clarification"],
  "reinforcement_needed": ["concepts needing reinforcement"]
}
```

Understanding levels:
- "verified": >= 0.8 and original explanation (not parroting)
- "partial": 0.5 - 0.79
- "surface": 0.3 - 0.49 (likely parroting or shallow)
- "minimal": < 0.3

Mastery levels for concepts:
- "mastered": Full understanding demonstrated
- "understood": Good understanding with minor gaps
- "developing": Partial understanding, needs reinforcement
- "not_understood": Concept not grasped
- "not_assessed": Concept not mentioned

Recommended actions:
- "continue_learning": Need more explanation
- "clarify": Address specific gaps
- "reinforce": Strengthen weak concepts
- "practice": Ready for guided practice
- "assessment": Ready for formal assessment

IMPORTANT: Set verified=true ONLY if understanding_score >= 0.8 AND parroting_detected=false"""

    def parse_response(self, response: str) -> ComprehensionEvaluationResult:
        """Parse LLM response into ComprehensionEvaluationResult.

        Args:
            response: Raw LLM response text.

        Returns:
            ComprehensionEvaluationResult with evaluation details.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            # Fallback to basic parsing
            return self._parse_plain_text_response(response)

        # Parse concept assessments
        concept_assessments = []
        if "concept_assessments" in data and data["concept_assessments"]:
            for ca in data["concept_assessments"]:
                mastery_str = ca.get("mastery_level", "not_assessed")
                try:
                    mastery_level = ConceptMasteryLevel(mastery_str)
                except ValueError:
                    mastery_level = ConceptMasteryLevel.NOT_ASSESSED

                concept_assessments.append(
                    KeyConceptAssessment(
                        concept=ca.get("concept", ""),
                        mastery_level=mastery_level,
                        score=float(ca.get("score", 0.0)),
                        mentioned=ca.get("mentioned", False),
                        understood=ca.get("understood", False),
                        explanation_quality=ca.get("explanation_quality", ""),
                        evidence=ca.get("evidence"),
                        gaps=ca.get("gaps", []),
                        feedback=ca.get("feedback", ""),
                    )
                )

        # Parse misconceptions
        misconceptions = []
        if "misconceptions" in data and data["misconceptions"]:
            for m in data["misconceptions"]:
                severity_str = m.get("severity", "significant")
                try:
                    severity = MisconceptionSeverity(severity_str)
                except ValueError:
                    severity = MisconceptionSeverity.SIGNIFICANT

                misconceptions.append(
                    DetectedMisconception(
                        id=m.get("id", f"MISC_{len(misconceptions):03d}"),
                        description=m.get("description", ""),
                        severity=severity,
                        related_concept=m.get("related_concept", ""),
                        student_statement=m.get("student_statement"),
                        correct_understanding=m.get("correct_understanding", ""),
                        correction_suggestion=m.get("correction_suggestion", ""),
                    )
                )

        # Parse understanding level
        level_str = data.get("understanding_level", "partial")
        try:
            understanding_level = UnderstandingLevel(level_str)
        except ValueError:
            # Infer from score
            score = float(data.get("understanding_score", 0.5))
            if score >= 0.8:
                understanding_level = UnderstandingLevel.VERIFIED
            elif score >= 0.5:
                understanding_level = UnderstandingLevel.PARTIAL
            elif score >= 0.3:
                understanding_level = UnderstandingLevel.SURFACE
            else:
                understanding_level = UnderstandingLevel.MINIMAL

        # Get trigger from stored params
        trigger = ComprehensionTrigger.SELF_REPORTED
        if hasattr(self, "_last_params") and self._last_params:
            trigger = self._last_params.trigger

        # Build detailed feedback dict
        detailed_feedback = {}
        if "detailed_feedback" in data and data["detailed_feedback"]:
            df = data["detailed_feedback"]
            if isinstance(df, dict):
                detailed_feedback = {
                    "strengths": df.get("strengths", ""),
                    "areas_to_improve": df.get("areas_to_improve", ""),
                    "next_steps": df.get("next_steps", ""),
                }
            elif isinstance(df, str):
                detailed_feedback = {"summary": df}

        return ComprehensionEvaluationResult(
            success=True,
            understanding_level=understanding_level,
            understanding_score=min(1.0, max(0.0, float(data.get("understanding_score", 0.5)))),
            verified=data.get("verified", False),
            parroting_detected=data.get("parroting_detected", False),
            concept_assessments=concept_assessments,
            concepts_understood=data.get("concepts_understood", []),
            concepts_missing=data.get("concepts_missing", []),
            concepts_weak=data.get("concepts_weak", []),
            misconceptions=misconceptions,
            overall_feedback=data.get("overall_feedback", ""),
            detailed_feedback=detailed_feedback,
            confidence=float(data.get("confidence", 0.9)),
            recommended_action=data.get("recommended_action", "continue_learning"),
            should_continue_learning=data.get("should_continue_learning", True),
            ready_for_practice=data.get("ready_for_practice", False),
            ready_for_assessment=data.get("ready_for_assessment", False),
            clarification_needed=data.get("clarification_needed", []),
            reinforcement_needed=data.get("reinforcement_needed", []),
            trigger=trigger,
            raw_response=response,
        )

    def _parse_plain_text_response(
        self,
        response: str,
    ) -> ComprehensionEvaluationResult:
        """Parse a plain text response when JSON parsing fails.

        Args:
            response: Raw LLM response text.

        Returns:
            ComprehensionEvaluationResult with best-effort parsing.
        """
        response_lower = response.lower()

        # Try to infer understanding from keywords
        positive_keywords = [
            "understand", "correct", "good", "excellent", "well done",
            "comprehends", "grasps", "demonstrates", "shows understanding",
        ]
        negative_keywords = [
            "doesn't understand", "misconception", "incorrect", "wrong",
            "confused", "needs clarification", "missing", "lacks",
        ]

        positive_count = sum(1 for kw in positive_keywords if kw in response_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in response_lower)

        # Estimate score based on keyword balance
        if positive_count > negative_count * 2:
            score = 0.75
            level = UnderstandingLevel.PARTIAL
            verified = False
        elif negative_count > positive_count:
            score = 0.35
            level = UnderstandingLevel.SURFACE
            verified = False
        else:
            score = 0.5
            level = UnderstandingLevel.PARTIAL
            verified = False

        # Get trigger from stored params
        trigger = ComprehensionTrigger.SELF_REPORTED
        if hasattr(self, "_last_params") and self._last_params:
            trigger = self._last_params.trigger

        return ComprehensionEvaluationResult(
            success=True,
            understanding_level=level,
            understanding_score=score,
            verified=verified,
            parroting_detected=False,
            overall_feedback=response[:500],  # Use response as feedback
            confidence=0.5,  # Low confidence for fallback parsing
            recommended_action="continue_learning",
            should_continue_learning=True,
            ready_for_practice=False,
            ready_for_assessment=False,
            trigger=trigger,
            raw_response=response,
            metadata={"parse_method": "plain_text_fallback"},
        )

    def build_user_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> str:
        """Build only the user prompt for YAML-driven system prompt mode.

        This method is used when the agent has system_prompt config in YAML.
        In that case, the system prompt comes from YAML via SystemPromptBuilder,
        and the capability only needs to provide the user prompt.

        Args:
            params: Capability-specific input parameters.
            context: Unified context from memory, theory, RAG, persona.

        Returns:
            User prompt string.
        """
        self.validate_params(params)
        p = ComprehensionEvaluationParams(**params)
        self._last_params = p
        return self._build_user_prompt(p)
