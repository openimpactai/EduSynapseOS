# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for agent capabilities.

Tests cover:
- Capability context building
- Prompt generation
- Response parsing
- Registry operations
- Error handling
"""

import json
from datetime import datetime
from uuid import uuid4

import pytest

from src.core.agents.capabilities.base import (
    Capability,
    CapabilityContext,
    CapabilityError,
    CapabilityResult,
)
from src.core.agents.capabilities.question_generation import (
    GeneratedQuestion,
    QuestionGenerationCapability,
    QuestionGenerationParams,
    QuestionOption,
)
from src.core.agents.capabilities.answer_evaluation import (
    AnswerEvaluationCapability,
    AnswerEvaluationParams,
    AnswerEvaluationResult,
)
from src.core.agents.capabilities.concept_explanation import (
    ConceptExplanationCapability,
    ConceptExplanationParams,
    ConceptExplanation,
)
from src.core.agents.capabilities.feedback_generation import (
    FeedbackGenerationCapability,
    FeedbackGenerationParams,
    FeedbackTone,
    FeedbackType,
    GeneratedFeedback,
    PerformanceData,
)
from src.core.agents.capabilities.diagnostic_analysis import (
    DiagnosticAnalysisCapability,
    DiagnosticAnalysisParams,
    DiagnosticResult,
    LearningPatternData,
)
from src.core.agents.capabilities.registry import (
    CapabilityNotFoundError,
    CapabilityRegistry,
    get_default_registry,
    reset_default_registry,
)
from src.models.common import BloomLevel
from src.models.practice import EvaluationMethod, QuestionType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_context() -> CapabilityContext:
    """Create an empty capability context."""
    return CapabilityContext()


@pytest.fixture
def question_generation_capability() -> QuestionGenerationCapability:
    """Create a question generation capability instance."""
    return QuestionGenerationCapability()


@pytest.fixture
def answer_evaluation_capability() -> AnswerEvaluationCapability:
    """Create an answer evaluation capability instance."""
    return AnswerEvaluationCapability()


@pytest.fixture
def concept_explanation_capability() -> ConceptExplanationCapability:
    """Create a concept explanation capability instance."""
    return ConceptExplanationCapability()


@pytest.fixture
def feedback_generation_capability() -> FeedbackGenerationCapability:
    """Create a feedback generation capability instance."""
    return FeedbackGenerationCapability()


@pytest.fixture
def diagnostic_analysis_capability() -> DiagnosticAnalysisCapability:
    """Create a diagnostic analysis capability instance."""
    return DiagnosticAnalysisCapability()


# =============================================================================
# CapabilityContext Tests
# =============================================================================


class TestCapabilityContext:
    """Tests for CapabilityContext."""

    def test_empty_context_defaults(self):
        """Test that empty context has correct defaults."""
        ctx = CapabilityContext()

        assert ctx.memory is None
        assert ctx.theory is None
        assert ctx.rag_results == []
        assert ctx.persona is None
        assert ctx.additional == {}

    def test_get_persona_prompt_without_persona(self):
        """Test get_persona_prompt returns empty string without persona."""
        ctx = CapabilityContext()
        assert ctx.get_persona_prompt() == ""

    def test_get_theory_guidance_without_theory(self):
        """Test get_theory_guidance returns empty string without theory."""
        ctx = CapabilityContext()
        assert ctx.get_theory_guidance() == ""

    def test_get_rag_context_without_results(self):
        """Test get_rag_context returns empty string without results."""
        ctx = CapabilityContext()
        assert ctx.get_rag_context() == ""

    def test_get_student_summary_without_memory(self):
        """Test get_student_summary returns empty string without memory."""
        ctx = CapabilityContext()
        assert ctx.get_student_summary() == ""


# =============================================================================
# QuestionGenerationCapability Tests
# =============================================================================


class TestQuestionGenerationCapability:
    """Tests for QuestionGenerationCapability."""

    def test_capability_name(self, question_generation_capability):
        """Test capability has correct name."""
        assert question_generation_capability.name == "question_generation"

    def test_capability_description(self, question_generation_capability):
        """Test capability has description."""
        assert question_generation_capability.description != ""

    def test_validate_params_valid(self, question_generation_capability):
        """Test validation passes for valid params."""
        params = {
            "topic": "Fractions",
            "difficulty": 0.5,
            "bloom_level": "understand",
        }
        # Should not raise
        question_generation_capability.validate_params(params)

    def test_validate_params_invalid_topic(self, question_generation_capability):
        """Test validation fails for empty topic."""
        params = {
            "topic": "",
        }
        with pytest.raises(CapabilityError):
            question_generation_capability.validate_params(params)

    def test_validate_params_invalid_difficulty(self, question_generation_capability):
        """Test validation fails for invalid difficulty."""
        params = {
            "topic": "Fractions",
            "difficulty": 1.5,  # Out of range
        }
        with pytest.raises(CapabilityError):
            question_generation_capability.validate_params(params)

    def test_build_prompt_returns_messages(
        self, question_generation_capability, empty_context
    ):
        """Test build_prompt returns list of messages."""
        params = {
            "topic": "Fractions",
            "difficulty": 0.5,
            "question_type": "multiple_choice",
        }
        messages = question_generation_capability.build_prompt(params, empty_context)

        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_build_prompt_includes_topic(
        self, question_generation_capability, empty_context
    ):
        """Test build_prompt includes the topic in user message."""
        params = {
            "topic": "Fractions",
        }
        messages = question_generation_capability.build_prompt(params, empty_context)
        user_message = messages[-1]["content"]

        assert "Fractions" in user_message

    def test_parse_response_valid_json(self, question_generation_capability):
        """Test parse_response handles valid JSON."""
        response = json.dumps({
            "content": "What is 1/2 + 1/4?",
            "options": [
                {"key": "a", "text": "1/4", "is_correct": False},
                {"key": "b", "text": "2/4", "is_correct": False},
                {"key": "c", "text": "3/4", "is_correct": True},
                {"key": "d", "text": "1", "is_correct": False},
            ],
            "correct_answer": "3/4",
            "hints": [
                {"level": 1, "text": "Convert to same denominator"},
                {"level": 2, "text": "1/2 = 2/4"},
            ],
            "explanation": "Add numerators when denominators are same",
            "misconceptions_addressed": ["adding_denominators"],
        })

        result = question_generation_capability.parse_response(response)

        assert isinstance(result, GeneratedQuestion)
        assert result.success is True
        assert result.content == "What is 1/2 + 1/4?"
        assert len(result.options) == 4
        assert result.correct_answer == "3/4"
        assert len(result.hints) == 2
        assert result.explanation is not None

    def test_parse_response_json_in_markdown(self, question_generation_capability):
        """Test parse_response handles JSON in markdown code block."""
        response = """Here is the question:
```json
{
    "content": "What is 2+2?",
    "correct_answer": "4",
    "hints": [],
    "misconceptions_addressed": []
}
```
"""
        result = question_generation_capability.parse_response(response)

        assert result.success is True
        assert result.content == "What is 2+2?"

    def test_parse_response_plain_text_fallback(self, question_generation_capability):
        """Test parse_response falls back for plain text."""
        response = "What is the capital of France?"

        result = question_generation_capability.parse_response(response)

        assert result.success is True
        assert "France" in result.content
        assert result.metadata.get("parse_method") == "plain_text_fallback"


# =============================================================================
# AnswerEvaluationCapability Tests
# =============================================================================


class TestAnswerEvaluationCapability:
    """Tests for AnswerEvaluationCapability."""

    def test_capability_name(self, answer_evaluation_capability):
        """Test capability has correct name."""
        assert answer_evaluation_capability.name == "answer_evaluation"

    def test_build_prompt_includes_question_and_answers(
        self, answer_evaluation_capability, empty_context
    ):
        """Test build_prompt includes question and answers."""
        params = {
            "question_content": "What is 2 + 2?",
            "student_answer": "4",
            "expected_answer": "4",
        }
        messages = answer_evaluation_capability.build_prompt(params, empty_context)
        user_message = messages[-1]["content"]

        assert "What is 2 + 2?" in user_message
        assert "4" in user_message

    def test_parse_response_correct_answer(self, answer_evaluation_capability):
        """Test parse_response for correct answer."""
        response = json.dumps({
            "is_correct": True,
            "score": 1.0,
            "feedback": "Perfect! That's exactly right.",
            "misconceptions": [],
            "confidence": 0.99,
            "improvement_suggestions": [],
        })

        result = answer_evaluation_capability.parse_response(response)

        assert isinstance(result, AnswerEvaluationResult)
        assert result.is_correct is True
        assert result.score == 1.0
        assert result.confidence == 0.99

    def test_parse_response_incorrect_with_misconceptions(
        self, answer_evaluation_capability
    ):
        """Test parse_response for incorrect answer with misconceptions."""
        response = json.dumps({
            "is_correct": False,
            "score": 0.3,
            "feedback": "Not quite right.",
            "misconceptions": [
                {
                    "code": "CALC_001",
                    "description": "Adding denominators incorrectly",
                    "severity": "medium",
                    "suggestion": "Remember to find common denominator first",
                }
            ],
            "confidence": 0.85,
            "improvement_suggestions": ["Review fraction addition rules"],
        })

        result = answer_evaluation_capability.parse_response(response)

        assert result.is_correct is False
        assert result.score == 0.3
        assert len(result.misconceptions) == 1
        assert result.misconceptions[0].code == "CALC_001"


# =============================================================================
# ConceptExplanationCapability Tests
# =============================================================================


class TestConceptExplanationCapability:
    """Tests for ConceptExplanationCapability."""

    def test_capability_name(self, concept_explanation_capability):
        """Test capability has correct name."""
        assert concept_explanation_capability.name == "concept_explanation"

    def test_build_prompt_includes_concept(
        self, concept_explanation_capability, empty_context
    ):
        """Test build_prompt includes the concept."""
        params = {
            "concept": "Photosynthesis",
            "target_level": "beginner",
        }
        messages = concept_explanation_capability.build_prompt(params, empty_context)
        user_message = messages[-1]["content"]

        assert "Photosynthesis" in user_message
        assert "beginner" in user_message

    def test_parse_response_full_explanation(self, concept_explanation_capability):
        """Test parse_response for full explanation."""
        response = json.dumps({
            "explanation": "Photosynthesis is how plants make food.",
            "summary": "Plants use sunlight to make food.",
            "examples": [
                {
                    "title": "Leaf in sunlight",
                    "content": "A leaf absorbs sunlight",
                    "explanation": "This shows light absorption",
                }
            ],
            "analogies": [
                {
                    "source_domain": "Kitchen",
                    "target_domain": "Plant cell",
                    "mapping": "Like cooking with solar energy",
                    "limitations": "No actual heat involved",
                }
            ],
            "prerequisites": ["Basic biology", "Light energy"],
            "related_concepts": ["Respiration", "Chlorophyll"],
            "key_points": ["Uses sunlight", "Produces oxygen"],
            "common_mistakes": ["Confusing with respiration"],
            "practice_suggestions": ["Draw the process"],
        })

        result = concept_explanation_capability.parse_response(response)

        assert isinstance(result, ConceptExplanation)
        assert "food" in result.explanation
        assert len(result.examples) == 1
        assert len(result.analogies) == 1
        assert len(result.key_points) == 2


# =============================================================================
# FeedbackGenerationCapability Tests
# =============================================================================


class TestFeedbackGenerationCapability:
    """Tests for FeedbackGenerationCapability."""

    def test_capability_name(self, feedback_generation_capability):
        """Test capability has correct name."""
        assert feedback_generation_capability.name == "feedback_generation"

    def test_build_prompt_includes_performance(
        self, feedback_generation_capability, empty_context
    ):
        """Test build_prompt includes performance data."""
        params = {
            "feedback_type": "session_complete",
            "performance": {
                "questions_total": 10,
                "questions_correct": 8,
                "score": 0.8,
            },
        }
        messages = feedback_generation_capability.build_prompt(params, empty_context)
        user_message = messages[-1]["content"]

        assert "80%" in user_message or "8/10" in user_message

    def test_parse_response_celebratory(self, feedback_generation_capability):
        """Test parse_response for celebratory feedback."""
        response = json.dumps({
            "main_message": "Fantastic work! You're really improving!",
            "summary": "Great session with 90% accuracy",
            "encouragement": "Keep up this amazing progress!",
            "statistics": {"accuracy": "90%"},
            "strengths_highlighted": ["Quick responses", "Consistent accuracy"],
            "improvement_suggestions": [],
            "next_steps": [
                {
                    "action": "Try harder problems",
                    "reason": "You're ready for more challenge",
                    "priority": "medium",
                }
            ],
            "celebration_message": "You've mastered this topic!",
        })

        result = feedback_generation_capability.parse_response(response)

        assert isinstance(result, GeneratedFeedback)
        assert "Fantastic" in result.main_message
        assert result.celebration_message is not None


# =============================================================================
# DiagnosticAnalysisCapability Tests
# =============================================================================


class TestDiagnosticAnalysisCapability:
    """Tests for DiagnosticAnalysisCapability."""

    def test_capability_name(self, diagnostic_analysis_capability):
        """Test capability has correct name."""
        assert diagnostic_analysis_capability.name == "diagnostic_analysis"

    def test_disclaimer_constant_exists(self, diagnostic_analysis_capability):
        """Test disclaimer text is defined."""
        assert hasattr(diagnostic_analysis_capability, "DISCLAIMER_TEXT")
        assert "diagnosis" in diagnostic_analysis_capability.DISCLAIMER_TEXT.lower()

    def test_build_prompt_includes_pattern_data(
        self, diagnostic_analysis_capability, empty_context
    ):
        """Test build_prompt includes pattern data."""
        params = {
            "pattern_data": {
                "topic_performance": {"fractions": 0.4, "decimals": 0.8},
                "error_patterns": {"calculation": 10},
            },
        }
        messages = diagnostic_analysis_capability.build_prompt(params, empty_context)
        user_message = messages[-1]["content"]

        assert "fractions" in user_message
        assert "calculation" in user_message

    def test_parse_response_with_referral_flag(self, diagnostic_analysis_capability):
        """Test parse_response includes referral flags with disclaimer."""
        response = json.dumps({
            "summary": "Consistent calculation difficulties observed",
            "pattern_observations": [
                {
                    "pattern_type": "topic_based",
                    "observation": "Struggles with number operations",
                    "significance": "high",
                    "evidence": ["Low accuracy", "Long response times"],
                    "implication": "May need additional support",
                }
            ],
            "struggle_patterns": [],
            "strengths_identified": ["Good conceptual understanding"],
            "learning_style_insights": {"preferred_format": "visual"},
            "strategy_recommendations": [
                {
                    "strategy": "Use visual aids",
                    "rationale": "Strong visual preference",
                    "expected_benefit": "Better understanding",
                    "implementation": "Include diagrams",
                }
            ],
            "referral_flags": [
                {
                    "area": "Mathematical processing",
                    "observations": ["Consistent calculation errors"],
                    "disclaimer": "Not a diagnosis - consult specialists",
                    "suggestion": "Consider educational assessment",
                }
            ],
            "adaptation_suggestions": ["More visual content"],
            "confidence": 0.7,
            "data_quality": "sufficient",
        })

        result = diagnostic_analysis_capability.parse_response(response)

        assert isinstance(result, DiagnosticResult)
        assert len(result.referral_flags) == 1
        assert result.referral_flags[0].disclaimer != ""


# =============================================================================
# CapabilityRegistry Tests
# =============================================================================


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_default_registry()

    def test_empty_registry(self):
        """Test empty registry has no capabilities."""
        registry = CapabilityRegistry()
        assert len(registry) == 0
        assert registry.list_names() == []

    def test_register_capability(self, question_generation_capability):
        """Test registering a capability."""
        registry = CapabilityRegistry()
        registry.register(question_generation_capability)

        assert len(registry) == 1
        assert "question_generation" in registry
        assert registry.has("question_generation")

    def test_register_duplicate_raises(self, question_generation_capability):
        """Test registering duplicate capability raises error."""
        registry = CapabilityRegistry()
        registry.register(question_generation_capability)

        with pytest.raises(ValueError):
            registry.register(question_generation_capability)

    def test_replace_capability(self, question_generation_capability):
        """Test replacing a capability."""
        registry = CapabilityRegistry()
        registry.register(question_generation_capability)
        registry.replace(question_generation_capability)  # Should not raise

        assert len(registry) == 1

    def test_get_capability(self, question_generation_capability):
        """Test getting a capability by name."""
        registry = CapabilityRegistry()
        registry.register(question_generation_capability)

        cap = registry.get("question_generation")
        assert cap is question_generation_capability

    def test_get_nonexistent_raises(self):
        """Test getting nonexistent capability raises error."""
        registry = CapabilityRegistry()

        with pytest.raises(CapabilityNotFoundError) as exc_info:
            registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_optional_returns_none(self):
        """Test get_optional returns None for nonexistent."""
        registry = CapabilityRegistry()

        cap = registry.get_optional("nonexistent")
        assert cap is None

    def test_unregister_capability(self, question_generation_capability):
        """Test unregistering a capability."""
        registry = CapabilityRegistry()
        registry.register(question_generation_capability)

        result = registry.unregister("question_generation")
        assert result is True
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent returns False."""
        registry = CapabilityRegistry()

        result = registry.unregister("nonexistent")
        assert result is False

    def test_default_registry(self):
        """Test default registry has all built-in capabilities."""
        registry = CapabilityRegistry.default()

        assert len(registry) == 5
        assert "question_generation" in registry
        assert "answer_evaluation" in registry
        assert "concept_explanation" in registry
        assert "feedback_generation" in registry
        assert "diagnostic_analysis" in registry

    def test_subset_registry(self):
        """Test creating a subset registry."""
        registry = CapabilityRegistry.default()
        subset = registry.subset(["question_generation", "answer_evaluation"])

        assert len(subset) == 2
        assert "question_generation" in subset
        assert "answer_evaluation" in subset
        assert "concept_explanation" not in subset

    def test_merge_registries(self, question_generation_capability):
        """Test merging registries."""
        registry1 = CapabilityRegistry()
        registry2 = CapabilityRegistry()
        registry2.register(question_generation_capability)

        registry1.merge(registry2)

        assert "question_generation" in registry1

    def test_get_descriptions(self):
        """Test getting capability descriptions."""
        registry = CapabilityRegistry.default()
        descriptions = registry.get_descriptions()

        assert isinstance(descriptions, dict)
        assert "question_generation" in descriptions
        assert descriptions["question_generation"] != ""

    def test_iteration(self):
        """Test iterating over registry."""
        registry = CapabilityRegistry.default()

        names = list(registry)
        assert len(names) == 5

    def test_global_default_registry(self):
        """Test global default registry singleton."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_reset_default_registry(self):
        """Test resetting global default registry."""
        registry1 = get_default_registry()
        reset_default_registry()
        registry2 = get_default_registry()

        assert registry1 is not registry2


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestCapabilityIntegration:
    """Tests for capability integration scenarios."""

    def test_full_question_generation_flow(
        self, question_generation_capability, empty_context
    ):
        """Test complete question generation flow."""
        # Build params
        params = {
            "topic": "Basic Addition",
            "difficulty": 0.3,
            "bloom_level": "remember",
            "question_type": "multiple_choice",
            "include_hints": True,
            "hint_count": 2,
        }

        # Build prompt
        messages = question_generation_capability.build_prompt(params, empty_context)

        # Verify prompt structure
        assert len(messages) == 2
        assert "system" in messages[0]["role"]
        assert "user" in messages[1]["role"]
        assert "Basic Addition" in messages[1]["content"]

        # Simulate LLM response
        llm_response = json.dumps({
            "content": "What is 5 + 3?",
            "options": [
                {"key": "a", "text": "6", "is_correct": False},
                {"key": "b", "text": "7", "is_correct": False},
                {"key": "c", "text": "8", "is_correct": True},
                {"key": "d", "text": "9", "is_correct": False},
            ],
            "correct_answer": "8",
            "hints": [
                {"level": 1, "text": "Count on your fingers"},
                {"level": 2, "text": "Start from 5 and count 3 more"},
            ],
            "explanation": "5 + 3 = 8 because adding 3 to 5 gives 8",
            "misconceptions_addressed": [],
        })

        # Parse response
        result = question_generation_capability.parse_response(llm_response)

        # Verify result
        assert result.success
        assert result.content == "What is 5 + 3?"
        assert len(result.options) == 4
        assert any(opt.is_correct for opt in result.options)
        assert len(result.hints) == 2

    def test_full_evaluation_flow(
        self, answer_evaluation_capability, empty_context
    ):
        """Test complete answer evaluation flow."""
        # Build params
        params = {
            "question_content": "What is 5 + 3?",
            "student_answer": "8",
            "expected_answer": "8",
            "question_type": "short_answer",
            "partial_credit": True,
        }

        # Build prompt
        messages = answer_evaluation_capability.build_prompt(params, empty_context)

        # Verify prompt structure
        assert len(messages) == 2

        # Simulate LLM response
        llm_response = json.dumps({
            "is_correct": True,
            "score": 1.0,
            "feedback": "Excellent! That's exactly right!",
            "detailed_feedback": {
                "strengths": ["Correct calculation"],
                "weaknesses": [],
            },
            "misconceptions": [],
            "confidence": 0.99,
            "improvement_suggestions": [],
        })

        # Parse response
        result = answer_evaluation_capability.parse_response(llm_response)

        # Verify result
        assert result.success
        assert result.is_correct
        assert result.score == 1.0
        assert result.confidence > 0.9
