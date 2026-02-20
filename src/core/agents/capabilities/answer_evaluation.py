# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Answer evaluation capability with hybrid rule-based and LLM approach.

This capability evaluates student answers using a hybrid approach:
- Rule-based evaluation for structured questions (MCQ, True/False, numerical)
- LLM-based evaluation for open-ended and semantic questions

The evaluation strategy is determined by EvaluationConfig, which can be:
- Inferred from question_type
- Explicitly provided in params

Rule-based strategies (no LLM needed):
    - CHOICE: Multiple choice matching (key comparison)
    - BOOLEAN: True/False evaluation
    - EXACT: Exact string match
    - NUMERICAL: Numerical comparison with tolerance
    - SET: Unordered set matching
    - SEQUENCE: Ordered sequence matching
    - PATTERN: Regex pattern matching

LLM-based strategies:
    - SEMANTIC: Semantic similarity evaluation
    - RUBRIC: Rubric-based scoring
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.models.practice import (
    EvaluationConfig,
    EvaluationStrategy,
    QuestionType,
)


class MisconceptionInfo(BaseModel):
    """Information about a detected misconception.

    Attributes:
        code: Unique identifier for the misconception.
        description: Description of the misconception.
        severity: How serious the misconception is (low, medium, high).
        suggestion: How to address this misconception.
    """

    code: str = Field(description="Misconception identifier")
    description: str = Field(description="Description of the misconception")
    severity: str = Field(
        default="medium",
        description="Severity: low, medium, high",
    )
    suggestion: str | None = Field(
        default=None,
        description="How to address this misconception",
    )


class QuestionOption(BaseModel):
    """Option for multiple choice questions."""

    key: str = Field(description="Option key (a, b, c, d)")
    text: str = Field(description="Option text")
    is_correct: bool = Field(default=False, description="Whether this is correct")


class AnswerEvaluationParams(BaseModel):
    """Parameters for answer evaluation.

    Attributes:
        question_content: The original question text.
        question_type: Type of question (for display/inference).
        student_answer: The student's answer.
        expected_answer: The correct/expected answer.
        options: Options for multiple choice questions.
        evaluation_config: Explicit evaluation configuration.
        topic: Topic of the question.
        language: Language for feedback.
    """

    question_content: str = Field(
        description="The original question text",
        min_length=1,
    )
    question_type: QuestionType = Field(
        default=QuestionType.SHORT_ANSWER,
        description="Type of question",
    )
    student_answer: str = Field(
        description="The student's answer",
    )
    expected_answer: str = Field(
        description="The correct/expected answer",
    )
    options: list[QuestionOption] | None = Field(
        default=None,
        description="Options for multiple choice questions",
    )
    evaluation_config: EvaluationConfig | None = Field(
        default=None,
        description="Explicit evaluation configuration",
    )
    topic: str = Field(
        default="",
        description="Topic of the question",
    )
    language: str = Field(
        default="en",
        description="Language for feedback",
    )

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation config, inferring from question_type if not set."""
        if self.evaluation_config:
            return self.evaluation_config
        return EvaluationConfig.from_question_type(self.question_type)


class AnswerEvaluationResult(CapabilityResult):
    """Result of answer evaluation.

    Attributes:
        is_correct: Whether the answer is fully correct.
        score: Score from 0.0 to 1.0.
        feedback: Feedback message for the student.
        detailed_feedback: Detailed breakdown if applicable.
        correct_answer: The correct answer.
        student_answer: The student's answer (echoed back).
        misconceptions: Detected misconceptions.
        partial_credit_breakdown: How partial credit was calculated.
        evaluation_strategy: Strategy used for evaluation.
        confidence: Confidence in the evaluation (0.0-1.0).
        improvement_suggestions: Suggestions for the student.
    """

    is_correct: bool = Field(description="Whether fully correct")
    score: float = Field(ge=0.0, le=1.0, description="Score 0.0-1.0")
    feedback: str = Field(description="Feedback message")
    detailed_feedback: dict[str, Any] | None = Field(
        default=None,
        description="Detailed breakdown",
    )
    correct_answer: str = Field(description="The correct answer")
    student_answer: str = Field(description="Student's answer")
    misconceptions: list[MisconceptionInfo] = Field(
        default_factory=list,
        description="Detected misconceptions",
    )
    partial_credit_breakdown: dict[str, float] | None = Field(
        default=None,
        description="How partial credit was calculated",
    )
    evaluation_strategy: EvaluationStrategy = Field(
        default=EvaluationStrategy.EXACT,
        description="Strategy used for evaluation",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the evaluation",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement",
    )


class AnswerEvaluationCapability(Capability):
    """Capability for evaluating student answers with hybrid approach.

    Uses rule-based evaluation for structured questions (MCQ, T/F, numerical)
    and LLM-based evaluation for open-ended questions.

    Example (rule-based - no LLM):
        capability = AnswerEvaluationCapability()
        params = {
            "question_content": "What is 2 + 2?",
            "question_type": "multiple_choice",
            "student_answer": "b",
            "expected_answer": "4",
            "options": [
                {"key": "a", "text": "3", "is_correct": False},
                {"key": "b", "text": "4", "is_correct": True},
                {"key": "c", "text": "5", "is_correct": False},
            ],
        }
        if not capability.needs_llm(params):
            result = capability.evaluate_direct(params)

    Example (LLM-based):
        params = {
            "question_content": "Explain photosynthesis",
            "question_type": "open_ended",
            "student_answer": "Plants convert sunlight to energy",
            "expected_answer": "...",
        }
        if capability.needs_llm(params):
            prompt = capability.build_prompt(params, context)
            llm_response = await llm.generate(prompt)
            result = capability.parse_response(llm_response)
    """

    @property
    def name(self) -> str:
        """Return capability name."""
        return "answer_evaluation"

    @property
    def description(self) -> str:
        """Return capability description."""
        return "Evaluates student answers with hybrid rule-based and LLM approach"

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate evaluation parameters."""
        try:
            AnswerEvaluationParams(**params)
        except Exception as e:
            raise CapabilityError(
                message=f"Invalid parameters: {e}",
                capability_name=self.name,
                original_error=e,
            ) from e

    def needs_llm(self, params: dict[str, Any]) -> bool:
        """Check if this evaluation requires LLM.

        Args:
            params: Evaluation parameters.

        Returns:
            True if LLM is needed, False for rule-based evaluation.
        """
        p = AnswerEvaluationParams(**params)
        config = p.get_evaluation_config()
        return config.needs_llm

    def evaluate_direct(
        self,
        params: dict[str, Any],
        context: CapabilityContext | None = None,
    ) -> AnswerEvaluationResult:
        """Evaluate answer directly without LLM (rule-based).

        This method handles all rule-based evaluation strategies:
        - CHOICE: Multiple choice
        - BOOLEAN: True/False
        - EXACT: Exact match
        - NUMERICAL: Numerical with tolerance
        - SET: Unordered set
        - SEQUENCE: Ordered sequence
        - PATTERN: Regex pattern

        Args:
            params: Evaluation parameters.
            context: Optional context (not used for rule-based).

        Returns:
            AnswerEvaluationResult with evaluation.

        Raises:
            CapabilityError: If strategy requires LLM.
        """
        self.validate_params(params)
        p = AnswerEvaluationParams(**params)
        config = p.get_evaluation_config()

        if config.needs_llm:
            raise CapabilityError(
                message=f"Strategy {config.strategy} requires LLM. Use build_prompt() instead.",
                capability_name=self.name,
            )

        # Route to appropriate evaluation method
        if config.strategy == EvaluationStrategy.CHOICE:
            return self._evaluate_choice(p, config)
        elif config.strategy == EvaluationStrategy.BOOLEAN:
            return self._evaluate_boolean(p, config)
        elif config.strategy == EvaluationStrategy.EXACT:
            return self._evaluate_exact(p, config)
        elif config.strategy == EvaluationStrategy.NUMERICAL:
            return self._evaluate_numerical(p, config)
        elif config.strategy == EvaluationStrategy.SET:
            return self._evaluate_set(p, config)
        elif config.strategy == EvaluationStrategy.SEQUENCE:
            return self._evaluate_sequence(p, config)
        elif config.strategy == EvaluationStrategy.PATTERN:
            return self._evaluate_pattern(p, config)
        else:
            # Fallback to exact match
            return self._evaluate_exact(p, config)

    def _evaluate_choice(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate multiple choice answer.

        Compares student's answer (option key like 'a', 'b', 'c') with
        the correct option from the options list.
        """
        student_answer = params.student_answer.strip().lower()

        # Find the correct option key from options
        correct_key = None
        correct_text = None

        if params.options:
            for option in params.options:
                opt = QuestionOption(**option) if isinstance(option, dict) else option
                if opt.is_correct:
                    correct_key = opt.key.lower()
                    correct_text = opt.text
                    break

        # If no correct option found, fallback to expected_answer
        if correct_key is None:
            correct_key = params.expected_answer.strip().lower()
            correct_text = params.expected_answer

        # Compare
        is_correct = student_answer == correct_key

        # Also check if student answered with full text (edge case)
        if not is_correct and params.options:
            for option in params.options:
                opt = QuestionOption(**option) if isinstance(option, dict) else option
                if student_answer == opt.text.strip().lower() and opt.is_correct:
                    is_correct = True
                    break

        score = 1.0 if is_correct else 0.0
        feedback = (
            "Correct!"
            if is_correct
            else f"The correct answer is: {correct_key.upper()}) {correct_text}"
        )

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            correct_answer=f"{correct_key}) {correct_text}" if correct_text else correct_key,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.CHOICE,
            confidence=1.0,
        )

    def _evaluate_boolean(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate true/false answer."""
        answer_lower = params.student_answer.strip().lower()
        correct_lower = params.expected_answer.strip().lower()

        true_values = {"true", "t", "yes", "y", "1", "doğru", "evet", "d"}
        false_values = {"false", "f", "no", "n", "0", "yanlış", "hayır", "h"}

        student_is_true = answer_lower in true_values
        student_is_false = answer_lower in false_values
        correct_is_true = correct_lower in true_values

        if student_is_true:
            is_correct = correct_is_true
        elif student_is_false:
            is_correct = not correct_is_true
        else:
            is_correct = answer_lower == correct_lower

        score = 1.0 if is_correct else 0.0
        correct_display = "True" if correct_is_true else "False"
        feedback = "Correct!" if is_correct else f"The answer is: {correct_display}"

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            correct_answer=correct_display,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.BOOLEAN,
            confidence=1.0,
        )

    def _evaluate_exact(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate exact match answer."""
        if config.case_sensitive:
            normalized_answer = params.student_answer.strip()
            normalized_correct = params.expected_answer.strip()
        else:
            normalized_answer = params.student_answer.strip().lower()
            normalized_correct = params.expected_answer.strip().lower()

        # Check exact match
        if normalized_answer == normalized_correct:
            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=True,
                score=1.0,
                feedback="Correct!",
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.EXACT,
                confidence=1.0,
            )

        # Check acceptable answers
        for acceptable in config.acceptable_answers:
            compare_acceptable = (
                acceptable.strip()
                if config.case_sensitive
                else acceptable.strip().lower()
            )
            if normalized_answer == compare_acceptable:
                return AnswerEvaluationResult(
                    success=True,
                    capability_name=self.name,
                    is_correct=True,
                    score=1.0,
                    feedback="Correct!",
                    correct_answer=params.expected_answer,
                    student_answer=params.student_answer,
                    evaluation_strategy=EvaluationStrategy.EXACT,
                    confidence=1.0,
                )

        # Check partial match for partial credit
        if config.partial_credit:
            partial_score = self._calculate_partial_score(
                normalized_answer, normalized_correct
            )
            if partial_score > 0.7:
                return AnswerEvaluationResult(
                    success=True,
                    capability_name=self.name,
                    is_correct=True,
                    score=partial_score,
                    feedback=f"Close! The exact answer is: {params.expected_answer}",
                    correct_answer=params.expected_answer,
                    student_answer=params.student_answer,
                    evaluation_strategy=EvaluationStrategy.EXACT,
                    confidence=0.9,
                )

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=False,
            score=0.0,
            feedback=f"The correct answer is: {params.expected_answer}",
            correct_answer=params.expected_answer,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.EXACT,
            confidence=1.0,
        )

    def _evaluate_numerical(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate numerical answer with tolerance."""
        try:
            student_value = self._parse_number(params.student_answer)
            correct_value = self._parse_number(params.expected_answer)
        except ValueError as e:
            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=False,
                score=0.0,
                feedback=f"Could not parse your answer as a number: {e}",
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.NUMERICAL,
                confidence=1.0,
            )

        tolerance = config.tolerance or 0.01
        difference = abs(student_value - correct_value)

        # Check within tolerance
        if difference <= tolerance:
            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=True,
                score=1.0,
                feedback="Correct!",
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.NUMERICAL,
                confidence=1.0,
            )

        # Check for partial credit based on how close
        if config.partial_credit:
            relative_error = difference / max(abs(correct_value), 1e-10)
            if relative_error < 0.1:  # Within 10%
                score = 0.8
                feedback = f"Very close! The exact answer is {params.expected_answer}."
            elif relative_error < 0.25:  # Within 25%
                score = 0.5
                feedback = f"Getting there. The correct answer is {params.expected_answer}."
            else:
                score = 0.0
                feedback = f"The correct answer is {params.expected_answer}."

            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=False,
                score=score,
                feedback=feedback,
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.NUMERICAL,
                confidence=0.9,
            )

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=False,
            score=0.0,
            feedback=f"The correct answer is {params.expected_answer}.",
            correct_answer=params.expected_answer,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.NUMERICAL,
            confidence=1.0,
        )

    def _evaluate_set(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate unordered set answer."""
        student_items = self._parse_items(params.student_answer, config.case_sensitive)
        correct_items = self._parse_items(params.expected_answer, config.case_sensitive)

        student_set = set(student_items)
        correct_set = set(correct_items)

        if student_set == correct_set:
            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=True,
                score=1.0,
                feedback="Correct! You got all items.",
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.SET,
                confidence=1.0,
            )

        intersection = student_set & correct_set
        if len(intersection) > 0:
            if config.partial_credit:
                score = len(intersection) / len(correct_set)
            else:
                score = 0.0
            missing = correct_set - student_set
            feedback = f"Partially correct. Missing: {', '.join(missing)}"
        else:
            score = 0.0
            feedback = f"The correct items are: {', '.join(correct_set)}"

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=False,
            score=score,
            feedback=feedback,
            correct_answer=params.expected_answer,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.SET,
            confidence=1.0,
        )

    def _evaluate_sequence(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate ordered sequence answer."""
        student_items = self._parse_items(params.student_answer, config.case_sensitive)
        correct_items = self._parse_items(params.expected_answer, config.case_sensitive)

        if student_items == correct_items:
            return AnswerEvaluationResult(
                success=True,
                capability_name=self.name,
                is_correct=True,
                score=1.0,
                feedback="Correct! Perfect order.",
                correct_answer=params.expected_answer,
                student_answer=params.student_answer,
                evaluation_strategy=EvaluationStrategy.SEQUENCE,
                confidence=1.0,
            )

        # Calculate partial score based on position matches
        if config.partial_credit and len(student_items) > 0 and len(correct_items) > 0:
            correct_positions = sum(
                1 for i, item in enumerate(student_items)
                if i < len(correct_items) and item == correct_items[i]
            )
            score = correct_positions / len(correct_items)
        else:
            score = 0.0

        feedback = f"The correct sequence is: {', '.join(correct_items)}"

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=False,
            score=score,
            feedback=feedback,
            correct_answer=params.expected_answer,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.SEQUENCE,
            confidence=1.0,
        )

    def _evaluate_pattern(
        self,
        params: AnswerEvaluationParams,
        config: EvaluationConfig,
    ) -> AnswerEvaluationResult:
        """Evaluate answer against regex pattern."""
        pattern = config.pattern
        if not pattern:
            # Use expected_answer as pattern
            pattern = params.expected_answer

        flags = 0 if config.case_sensitive else re.IGNORECASE

        try:
            if re.match(pattern, params.student_answer.strip(), flags):
                return AnswerEvaluationResult(
                    success=True,
                    capability_name=self.name,
                    is_correct=True,
                    score=1.0,
                    feedback="Correct!",
                    correct_answer=params.expected_answer,
                    student_answer=params.student_answer,
                    evaluation_strategy=EvaluationStrategy.PATTERN,
                    confidence=1.0,
                )
        except re.error:
            pass

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            is_correct=False,
            score=0.0,
            feedback=f"The correct answer is: {params.expected_answer}",
            correct_answer=params.expected_answer,
            student_answer=params.student_answer,
            evaluation_strategy=EvaluationStrategy.PATTERN,
            confidence=1.0,
        )

    def build_prompt(
        self,
        params: dict[str, Any],
        context: CapabilityContext,
    ) -> list[dict[str, str]]:
        """Build prompt for LLM-based answer evaluation.

        Only used for SEMANTIC and RUBRIC strategies.

        Args:
            params: Evaluation parameters.
            context: Context from memory, theory, RAG, persona.

        Returns:
            List of messages for LLM.
        """
        self.validate_params(params)
        p = AnswerEvaluationParams(**params)
        config = p.get_evaluation_config()

        # Store for parse_response
        self._last_params = p
        self._last_config = config

        # Build system message
        system_parts = []

        # Base instruction
        system_parts.append(
            "You are an expert educational evaluator. "
            "Evaluate student answers with semantic understanding, not just exact matching. "
            "Be encouraging while being accurate. Identify misconceptions constructively."
        )

        # Add persona if available
        persona_prompt = context.get_persona_prompt()
        if persona_prompt:
            system_parts.append(persona_prompt)

        # Add student context if available
        student_summary = context.get_student_summary()
        if student_summary:
            system_parts.append(student_summary)

        system_message = "\n\n".join(system_parts)

        # Build user message
        user_parts = []

        user_parts.append("Evaluate this student's answer:\n")
        user_parts.append(f"**Question:** {p.question_content}")
        user_parts.append(f"**Question Type:** {p.question_type.value}")
        if p.topic:
            user_parts.append(f"**Topic:** {p.topic}")
        user_parts.append(f"**Expected Answer:** {p.expected_answer}")
        user_parts.append(f"**Student's Answer:** {p.student_answer}")

        user_parts.append("\n**Evaluation Guidelines:**")
        user_parts.append(f"- Language for feedback: {p.language}")

        if config.partial_credit:
            user_parts.append("- Award partial credit for partially correct answers")
            user_parts.append("- Score should reflect the degree of correctness (0.0-1.0)")
        else:
            user_parts.append("- No partial credit: answer is either fully correct (1.0) or incorrect (0.0)")

        user_parts.append("- Identify any misconceptions in the student's thinking")
        user_parts.append("- Provide constructive feedback that helps learning")
        user_parts.append("- Suggest specific improvements if the answer is incorrect")

        if config.rubric:
            user_parts.append("\n**Evaluation Rubric:**")
            for criterion, details in config.rubric.items():
                if isinstance(details, dict):
                    weight = details.get("weight", 0) * 100
                    desc = details.get("description", criterion)
                    user_parts.append(f"- {criterion} ({weight:.0f}%): {desc}")
                else:
                    user_parts.append(f"- {criterion}: {details}")

        # Add RAG context if relevant
        rag_context = context.get_rag_context(max_results=2)
        if rag_context:
            user_parts.append(f"\n{rag_context}")

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
Respond with a valid JSON object in this exact format:
```json
{
  "is_correct": true,
  "score": 0.85,
  "feedback": "Your personalized feedback message here",
  "detailed_feedback": {
    "strengths": ["What the student did well"],
    "weaknesses": ["Areas for improvement"]
  },
  "misconceptions": [
    {
      "code": "MISC_001",
      "description": "Description of the misconception",
      "severity": "medium",
      "suggestion": "How to correct this misconception"
    }
  ],
  "partial_credit_breakdown": {
    "concept_understanding": 0.9,
    "accuracy": 0.8
  },
  "confidence": 0.95,
  "improvement_suggestions": ["Specific suggestion 1", "Specific suggestion 2"]
}
```
Note: Set is_correct to true only if score is 1.0. Include misconceptions only if detected.
"""

    def parse_response(self, response: str) -> AnswerEvaluationResult:
        """Parse LLM response into AnswerEvaluationResult.

        Args:
            response: Raw LLM response text.

        Returns:
            AnswerEvaluationResult.
        """
        try:
            data = self._extract_json_from_response(response)
        except CapabilityError:
            return self._parse_plain_text_response(response)

        # Parse misconceptions
        misconceptions = []
        if "misconceptions" in data and data["misconceptions"]:
            for m in data["misconceptions"]:
                misconceptions.append(
                    MisconceptionInfo(
                        code=m.get("code", "UNKNOWN"),
                        description=m.get("description", ""),
                        severity=m.get("severity", "medium"),
                        suggestion=m.get("suggestion"),
                    )
                )

        score = float(data.get("score", 0.0))
        is_correct = data.get("is_correct", score >= 1.0)

        # Get student and expected answers from stored params
        student_answer = ""
        correct_answer = ""
        if hasattr(self, "_last_params") and self._last_params:
            student_answer = self._last_params.student_answer
            correct_answer = self._last_params.expected_answer

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            is_correct=is_correct,
            score=min(1.0, max(0.0, score)),
            feedback=data.get("feedback", ""),
            detailed_feedback=data.get("detailed_feedback"),
            correct_answer=correct_answer,
            student_answer=student_answer,
            misconceptions=misconceptions,
            partial_credit_breakdown=data.get("partial_credit_breakdown"),
            evaluation_strategy=EvaluationStrategy.SEMANTIC,
            confidence=float(data.get("confidence", 0.9)),
            improvement_suggestions=data.get("improvement_suggestions", []),
        )

    def _parse_plain_text_response(self, response: str) -> AnswerEvaluationResult:
        """Parse a plain text response when JSON parsing fails."""
        response_lower = response.lower()

        is_correct = any(word in response_lower for word in [
            "correct", "doğru", "right", "exactly", "perfect", "tam"
        ]) and not any(word in response_lower for word in [
            "incorrect", "yanlış", "wrong", "not correct", "değil"
        ])

        score = 1.0 if is_correct else 0.0

        return AnswerEvaluationResult(
            success=True,
            capability_name=self.name,
            raw_response=response,
            is_correct=is_correct,
            score=score,
            feedback=response.strip(),
            detailed_feedback=None,
            correct_answer="",
            student_answer="",
            misconceptions=[],
            partial_credit_breakdown=None,
            evaluation_strategy=EvaluationStrategy.SEMANTIC,
            confidence=0.5,
            improvement_suggestions=[],
            metadata={"parse_method": "plain_text_fallback"},
        )

    # Helper methods

    def _calculate_partial_score(self, answer: str, correct: str) -> float:
        """Calculate partial match score."""
        if not answer or not correct:
            return 0.0

        # Check containment
        if answer in correct or correct in answer:
            return 0.8

        # Character overlap
        answer_chars = set(answer)
        correct_chars = set(correct)
        overlap = len(answer_chars & correct_chars)
        max_len = max(len(answer_chars), len(correct_chars))

        if max_len > 0:
            return overlap / max_len

        return 0.0

    def _parse_number(self, text: str) -> float:
        """Parse a number from text."""
        cleaned = text.strip()
        # Remove common formatting
        cleaned = cleaned.replace(",", "").replace(" ", "")
        # Handle fractions
        if "/" in cleaned:
            parts = cleaned.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        # Handle percentages
        if "%" in cleaned:
            cleaned = cleaned.replace("%", "")
            return float(cleaned) / 100

        return float(cleaned)

    def _parse_items(self, text: str, case_sensitive: bool) -> list[str]:
        """Parse comma-separated items."""
        items = [item.strip() for item in text.split(",")]
        if not case_sensitive:
            items = [item.lower() for item in items]
        return [item for item in items if item]
