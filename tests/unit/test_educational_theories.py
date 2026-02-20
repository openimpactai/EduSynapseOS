# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for educational theories.

Tests the 7 educational theories and the TheoryOrchestrator:
- ZPD: Zone of Proximal Development
- Bloom: Bloom's Taxonomy
- VARK: Learning styles
- Scaffolding: Support levels
- Mastery: Learning progression
- Socratic: Questioning style
- SpacedRepetition: FSRS scheduling
"""

import uuid
from datetime import datetime, timezone

import pytest

from src.core.educational.orchestrator import CombinedRecommendation, TheoryOrchestrator
from src.core.educational.theories.base import (
    BaseTheory,
    BloomLevel,
    ContentFormat,
    QuestioningStyle,
    RecentPerformance,
    ScaffoldLevel,
    StudentContext,
    TheoryConfig,
    TheoryRecommendation,
    TopicState,
    VARKScores,
)
from src.core.educational.theories.bloom import BloomTheory
from src.core.educational.theories.mastery import MasteryTheory
from src.core.educational.theories.scaffolding import ScaffoldingTheory
from src.core.educational.theories.socratic import SocraticTheory
from src.core.educational.theories.spaced_repetition import (
    ReviewRating,
    SpacedRepetitionTheory,
)
from src.core.educational.theories.vark import VARKTheory
from src.core.educational.theories.zpd import ZPDTheory


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def student_id() -> uuid.UUID:
    """Create a test student ID."""
    return uuid.uuid4()


@pytest.fixture
def topic_id() -> uuid.UUID:
    """Create a test topic ID."""
    return uuid.uuid4()


@pytest.fixture
def basic_topic(topic_id: uuid.UUID) -> TopicState:
    """Create a basic topic state."""
    return TopicState(
        topic_id=topic_id,
        topic_name="Fractions",
        mastery_level=0.5,
        attempts_total=10,
        attempts_correct=6,
        current_streak=2,
    )


@pytest.fixture
def struggling_topic(topic_id: uuid.UUID) -> TopicState:
    """Create a topic state for struggling student."""
    return TopicState(
        topic_id=topic_id,
        topic_name="Algebra",
        mastery_level=0.2,
        attempts_total=15,
        attempts_correct=5,
        current_streak=0,
    )


@pytest.fixture
def mastered_topic(topic_id: uuid.UUID) -> TopicState:
    """Create a topic state for mastered content."""
    return TopicState(
        topic_id=topic_id,
        topic_name="Addition",
        mastery_level=0.9,
        attempts_total=50,
        attempts_correct=45,
        current_streak=10,
    )


@pytest.fixture
def basic_context(student_id: uuid.UUID, basic_topic: TopicState) -> StudentContext:
    """Create a basic student context."""
    return StudentContext(
        student_id=student_id,
        current_topic=basic_topic,
        overall_mastery=0.5,
        vark_scores=VARKScores(visual=0.4, auditory=0.2, reading=0.3, kinesthetic=0.1),
        recent_performance=RecentPerformance(
            last_n_correct=6,
            last_n_incorrect=4,
            consecutive_correct=2,
            consecutive_incorrect=0,
        ),
    )


@pytest.fixture
def struggling_context(
    student_id: uuid.UUID, struggling_topic: TopicState
) -> StudentContext:
    """Create context for struggling student."""
    return StudentContext(
        student_id=student_id,
        current_topic=struggling_topic,
        overall_mastery=0.3,
        recent_performance=RecentPerformance(
            last_n_correct=2,
            last_n_incorrect=8,
            consecutive_correct=0,
            consecutive_incorrect=4,
        ),
        hint_dependency=0.7,
    )


@pytest.fixture
def succeeding_context(
    student_id: uuid.UUID, mastered_topic: TopicState
) -> StudentContext:
    """Create context for succeeding student."""
    return StudentContext(
        student_id=student_id,
        current_topic=mastered_topic,
        overall_mastery=0.85,
        recent_performance=RecentPerformance(
            last_n_correct=9,
            last_n_incorrect=1,
            consecutive_correct=6,
            consecutive_incorrect=0,
        ),
    )


# ============================================================================
# Base Theory Tests
# ============================================================================


@pytest.mark.unit
class TestTheoryConfig:
    """Test cases for TheoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TheoryConfig()
        assert config.enabled is True
        assert config.weight == 1.0
        assert config.parameters == {}

    def test_custom_config(self):
        """Test custom configuration."""
        config = TheoryConfig(
            enabled=False,
            weight=0.5,
            parameters={"key": "value"},
        )
        assert config.enabled is False
        assert config.weight == 0.5
        assert config.parameters["key"] == "value"


@pytest.mark.unit
class TestStudentContext:
    """Test cases for StudentContext."""

    def test_context_creation(self, student_id: uuid.UUID, basic_topic: TopicState):
        """Test context can be created with required fields."""
        context = StudentContext(
            student_id=student_id,
            current_topic=basic_topic,
        )
        assert context.student_id == student_id
        assert context.current_topic == basic_topic
        assert context.overall_mastery == 0.0

    def test_vark_dominant_style(self):
        """Test VARK dominant style calculation."""
        visual_dominant = VARKScores(visual=0.6, auditory=0.2, reading=0.1, kinesthetic=0.1)
        assert visual_dominant.dominant_style == ContentFormat.VISUAL

        multimodal = VARKScores(visual=0.3, auditory=0.25, reading=0.25, kinesthetic=0.2)
        assert multimodal.dominant_style == ContentFormat.MULTIMODAL


# ============================================================================
# ZPD Theory Tests
# ============================================================================


@pytest.mark.unit
class TestZPDTheory:
    """Test cases for ZPD theory."""

    def test_name(self):
        """Test theory name."""
        theory = ZPDTheory()
        assert theory.name == "zpd"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic ZPD calculation."""
        theory = ZPDTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "zpd"
        assert result.difficulty is not None
        assert 0.0 <= result.difficulty <= 1.0
        assert result.scaffold_level is not None
        assert result.confidence > 0

    def test_difficulty_above_mastery(self, basic_context: StudentContext):
        """Test that difficulty is set above current mastery in ZPD."""
        theory = ZPDTheory()
        result = theory.calculate(basic_context)

        mastery = basic_context.current_topic.mastery_level
        assert result.difficulty >= mastery

    def test_frustration_reduces_difficulty(self, struggling_context: StudentContext):
        """Test that frustration reduces difficulty."""
        theory = ZPDTheory()
        result = theory.calculate(struggling_context)

        assert "frustration" in result.extra.get("zone", "") or result.difficulty < 0.5

    def test_success_increases_difficulty(self, succeeding_context: StudentContext):
        """Test that success streak increases difficulty."""
        theory = ZPDTheory()
        result = theory.calculate(succeeding_context)

        assert result.extra.get("zone") != "frustration"


# ============================================================================
# Bloom Theory Tests
# ============================================================================


@pytest.mark.unit
class TestBloomTheory:
    """Test cases for Bloom's Taxonomy theory."""

    def test_name(self):
        """Test theory name."""
        theory = BloomTheory()
        assert theory.name == "bloom"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic Bloom calculation."""
        theory = BloomTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "bloom"
        assert result.bloom_level is not None
        assert isinstance(result.bloom_level, BloomLevel)

    def test_low_mastery_gives_lower_levels(self, struggling_context: StudentContext):
        """Test that low mastery results in lower Bloom levels."""
        theory = BloomTheory()
        result = theory.calculate(struggling_context)

        assert result.bloom_level in [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND]

    def test_high_mastery_gives_higher_levels(self, succeeding_context: StudentContext):
        """Test that high mastery results in higher Bloom levels."""
        theory = BloomTheory()
        result = theory.calculate(succeeding_context)

        assert result.bloom_level in [
            BloomLevel.ANALYZE,
            BloomLevel.EVALUATE,
            BloomLevel.CREATE,
        ]

    def test_level_order(self):
        """Test that LEVEL_ORDER is correctly defined."""
        assert BloomTheory.LEVEL_ORDER[0] == BloomLevel.REMEMBER
        assert BloomTheory.LEVEL_ORDER[-1] == BloomLevel.CREATE

    def test_get_question_verbs(self):
        """Test question verb retrieval."""
        verbs = BloomTheory.get_question_verbs(BloomLevel.APPLY)
        assert "apply" in verbs
        assert "solve" in verbs


# ============================================================================
# VARK Theory Tests
# ============================================================================


@pytest.mark.unit
class TestVARKTheory:
    """Test cases for VARK learning styles theory."""

    def test_name(self):
        """Test theory name."""
        theory = VARKTheory()
        assert theory.name == "vark"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic VARK calculation."""
        theory = VARKTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "vark"
        assert result.content_format is not None
        assert isinstance(result.content_format, ContentFormat)

    def test_visual_dominant_student(self, student_id: uuid.UUID, basic_topic: TopicState):
        """Test visual learner gets visual format."""
        context = StudentContext(
            student_id=student_id,
            current_topic=basic_topic,
            vark_scores=VARKScores(
                visual=0.6, auditory=0.15, reading=0.15, kinesthetic=0.1
            ),
        )
        theory = VARKTheory()
        result = theory.calculate(context)

        assert result.content_format == ContentFormat.VISUAL

    def test_multimodal_student(self, student_id: uuid.UUID, basic_topic: TopicState):
        """Test balanced learner gets multimodal format."""
        context = StudentContext(
            student_id=student_id,
            current_topic=basic_topic,
            vark_scores=VARKScores(
                visual=0.28, auditory=0.25, reading=0.24, kinesthetic=0.23
            ),
        )
        theory = VARKTheory()
        result = theory.calculate(context)

        assert result.content_format == ContentFormat.MULTIMODAL

    def test_get_format_strategies(self):
        """Test strategy retrieval for formats."""
        strategies = VARKTheory.get_format_strategies(ContentFormat.VISUAL)
        assert len(strategies) > 0
        assert any("diagram" in s.lower() for s in strategies)


# ============================================================================
# Scaffolding Theory Tests
# ============================================================================


@pytest.mark.unit
class TestScaffoldingTheory:
    """Test cases for Scaffolding theory."""

    def test_name(self):
        """Test theory name."""
        theory = ScaffoldingTheory()
        assert theory.name == "scaffolding"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic scaffolding calculation."""
        theory = ScaffoldingTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "scaffolding"
        assert result.scaffold_level is not None
        assert isinstance(result.scaffold_level, ScaffoldLevel)
        assert result.hints_enabled is not None

    def test_struggling_gets_more_support(self, struggling_context: StudentContext):
        """Test that struggling students get more support."""
        theory = ScaffoldingTheory()
        result = theory.calculate(struggling_context)

        assert result.scaffold_level.value >= ScaffoldLevel.MODERATE.value
        assert result.hints_enabled is True

    def test_succeeding_gets_less_support(self, succeeding_context: StudentContext):
        """Test that succeeding students get less support."""
        theory = ScaffoldingTheory()
        result = theory.calculate(succeeding_context)

        assert result.scaffold_level.value <= ScaffoldLevel.MODERATE.value

    def test_get_support_strategies(self):
        """Test support strategy retrieval."""
        strategies = ScaffoldingTheory.get_support_strategies(ScaffoldLevel.HIGH)
        assert len(strategies) > 0


# ============================================================================
# Mastery Theory Tests
# ============================================================================


@pytest.mark.unit
class TestMasteryTheory:
    """Test cases for Mastery Learning theory."""

    def test_name(self):
        """Test theory name."""
        theory = MasteryTheory()
        assert theory.name == "mastery"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic mastery calculation."""
        theory = MasteryTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "mastery"
        assert result.advance_to_next is not None
        assert result.difficulty is not None

    def test_not_mastered_cannot_advance(self, basic_context: StudentContext):
        """Test that unmastered topic blocks advancement."""
        theory = MasteryTheory()
        result = theory.calculate(basic_context)

        assert result.advance_to_next is False

    def test_mastered_can_advance(self, succeeding_context: StudentContext):
        """Test that mastered topic allows advancement."""
        config = TheoryConfig(
            parameters={
                "mastery_threshold": 0.85,
                "min_attempts": 5,
                "min_accuracy": 0.75,
                "streak_requirement": 3,
            }
        )
        theory = MasteryTheory(config)
        result = theory.calculate(succeeding_context)

        assert result.advance_to_next is True

    def test_get_corrective_strategies(self):
        """Test corrective strategy retrieval."""
        strategies = MasteryTheory.get_corrective_strategies(0.3)
        assert len(strategies) > 0


# ============================================================================
# Socratic Theory Tests
# ============================================================================


@pytest.mark.unit
class TestSocraticTheory:
    """Test cases for Socratic Method theory."""

    def test_name(self):
        """Test theory name."""
        theory = SocraticTheory()
        assert theory.name == "socratic"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic Socratic calculation."""
        theory = SocraticTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "socratic"
        assert result.guide_vs_tell_ratio is not None
        assert 0.0 <= result.guide_vs_tell_ratio <= 1.0
        assert result.questioning_style is not None

    def test_struggling_gets_direct_style(self, struggling_context: StudentContext):
        """Test struggling students get direct instruction."""
        theory = SocraticTheory()
        result = theory.calculate(struggling_context)

        assert result.questioning_style == QuestioningStyle.DIRECT
        assert result.guide_vs_tell_ratio < 0.6

    def test_succeeding_gets_exploratory_style(self, succeeding_context: StudentContext):
        """Test succeeding students get exploratory questioning."""
        theory = SocraticTheory()
        result = theory.calculate(succeeding_context)

        assert result.questioning_style in [
            QuestioningStyle.EXPLORATORY,
            QuestioningStyle.CHALLENGING,
        ]

    def test_get_question_starters(self):
        """Test question starter retrieval."""
        starters = SocraticTheory.get_question_starters(QuestioningStyle.GUIDED)
        assert len(starters) > 0


# ============================================================================
# Spaced Repetition Theory Tests
# ============================================================================


@pytest.mark.unit
class TestSpacedRepetitionTheory:
    """Test cases for Spaced Repetition (FSRS) theory."""

    def test_name(self):
        """Test theory name."""
        theory = SpacedRepetitionTheory()
        assert theory.name == "spaced_repetition"

    def test_basic_calculation(self, basic_context: StudentContext):
        """Test basic spaced repetition calculation."""
        theory = SpacedRepetitionTheory()
        result = theory.calculate(basic_context)

        assert result.theory_name == "spaced_repetition"
        assert result.difficulty is not None

    def test_create_new_card(self):
        """Test new card creation."""
        theory = SpacedRepetitionTheory()
        card = theory.create_new_card()

        assert "stability" in card
        assert "difficulty" in card
        assert "state" in card
        assert card["state"] == "new"

    def test_get_retrievability(self):
        """Test retrievability calculation."""
        theory = SpacedRepetitionTheory()

        high_ret = theory.get_retrievability(stability=10.0, elapsed_days=1)
        low_ret = theory.get_retrievability(stability=10.0, elapsed_days=30)

        assert high_ret > low_ret
        assert 0.0 <= high_ret <= 1.0
        assert 0.0 <= low_ret <= 1.0

    def test_rating_from_performance(self):
        """Test rating determination from performance."""
        assert (
            SpacedRepetitionTheory.rating_from_performance(
                correct=False, response_time_seconds=None, used_hint=False, difficulty=0.5
            )
            == ReviewRating.AGAIN
        )

        assert (
            SpacedRepetitionTheory.rating_from_performance(
                correct=True, response_time_seconds=None, used_hint=True, difficulty=0.5
            )
            == ReviewRating.HARD
        )

        assert (
            SpacedRepetitionTheory.rating_from_performance(
                correct=True, response_time_seconds=30, used_hint=False, difficulty=0.5
            )
            == ReviewRating.GOOD
        )

    def test_get_interval_description(self):
        """Test interval description generation."""
        assert SpacedRepetitionTheory.get_interval_description(1) == "1 day"
        assert SpacedRepetitionTheory.get_interval_description(7) == "1 week"
        assert SpacedRepetitionTheory.get_interval_description(30) == "1 month"


# ============================================================================
# Theory Orchestrator Tests
# ============================================================================


@pytest.mark.unit
class TestTheoryOrchestrator:
    """Test cases for TheoryOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = TheoryOrchestrator()

        enabled = orchestrator.get_enabled_theories()
        assert len(enabled) == 7
        assert "zpd" in enabled
        assert "bloom" in enabled

    def test_apply_all_theories(self, basic_context: StudentContext):
        """Test applying all theories."""
        orchestrator = TheoryOrchestrator()
        result = orchestrator.apply(basic_context)

        assert isinstance(result, CombinedRecommendation)
        assert result.difficulty is not None
        assert result.bloom_level is not None
        assert result.content_format is not None
        assert result.scaffold_level is not None
        assert len(result.theory_contributions) == 7

    def test_combined_difficulty_is_weighted(self, basic_context: StudentContext):
        """Test that combined difficulty is a weighted average."""
        orchestrator = TheoryOrchestrator()
        result = orchestrator.apply(basic_context)

        assert 0.0 <= result.difficulty <= 1.0

    def test_get_theory(self):
        """Test getting a specific theory."""
        orchestrator = TheoryOrchestrator()

        zpd = orchestrator.get_theory("zpd")
        assert isinstance(zpd, ZPDTheory)

        invalid = orchestrator.get_theory("invalid")
        assert invalid is None

    def test_get_theory_summary(self):
        """Test theory summary."""
        orchestrator = TheoryOrchestrator()
        summary = orchestrator.get_theory_summary()

        assert len(summary) == 7
        assert "zpd" in summary
        assert summary["zpd"]["enabled"] is True
