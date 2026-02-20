# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base theory class and shared models for educational theories.

This module provides the abstract base class for all educational theories
and the shared data models they use for input (StudentContext) and
output (TheoryRecommendation).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.emotional import EmotionalContext


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels (ordered by complexity)."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ContentFormat(str, Enum):
    """Content delivery format preferences (VARK)."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"
    MULTIMODAL = "multimodal"


class ScaffoldLevel(int, Enum):
    """Scaffolding support levels (1=minimal, 5=maximum)."""

    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    MAXIMUM = 5


class QuestioningStyle(str, Enum):
    """Socratic questioning styles."""

    DIRECT = "direct"
    GUIDED = "guided"
    EXPLORATORY = "exploratory"
    CHALLENGING = "challenging"


class TopicState(BaseModel):
    """State of a topic for a student.

    Aggregated from semantic memory (topic mastery records).
    Uses topic_full_code from Central Curriculum (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

    Streak fields are used by ZPD theory for frustration/confidence adjustments:
    - current_streak: Positive = consecutive correct, negative = consecutive incorrect
    - best_streak: Highest consecutive correct answers achieved
    """

    topic_full_code: str
    topic_name: str
    mastery_level: float = Field(ge=0.0, le=1.0)
    attempts_total: int = 0
    attempts_correct: int = 0
    last_practiced_at: datetime | None = None
    current_streak: int = 0
    best_streak: int = 0
    time_spent_seconds: int = 0

    @property
    def accuracy(self) -> float | None:
        """Calculate accuracy if there are attempts."""
        if self.attempts_total == 0:
            return None
        return self.attempts_correct / self.attempts_total


class VARKScores(BaseModel):
    """VARK learning style profile scores."""

    visual: float = Field(default=0.25, ge=0.0, le=1.0)
    auditory: float = Field(default=0.25, ge=0.0, le=1.0)
    reading: float = Field(default=0.25, ge=0.0, le=1.0)
    kinesthetic: float = Field(default=0.25, ge=0.0, le=1.0)

    @property
    def dominant_style(self) -> ContentFormat:
        """Get the dominant learning style."""
        scores = {
            ContentFormat.VISUAL: self.visual,
            ContentFormat.AUDITORY: self.auditory,
            ContentFormat.READING: self.reading,
            ContentFormat.KINESTHETIC: self.kinesthetic,
        }
        dominant = max(scores, key=lambda k: scores[k])
        max_score = scores[dominant]
        if max_score < 0.35:
            return ContentFormat.MULTIMODAL
        return dominant


class RecentPerformance(BaseModel):
    """Recent performance metrics for theory calculations."""

    last_n_correct: int = 0
    last_n_incorrect: int = 0
    last_n_hint_used: int = 0
    consecutive_correct: int = 0
    consecutive_incorrect: int = 0
    avg_response_time_seconds: float | None = None
    avg_difficulty: float | None = None


class DiagnosticIndicatorScores(BaseModel):
    """Diagnostic indicator risk scores for theory adjustments.

    Contains risk scores from the diagnostic system that theories
    can use to adapt their recommendations. All scores are 0.0-1.0.

    Threshold levels:
    - < 0.3: LOW (no concern)
    - 0.3 - 0.5: MEDIUM (monitor)
    - 0.5 - 0.7: ELEVATED (attention needed)
    - >= 0.7: HIGH (professional evaluation recommended)

    When None in StudentContext, theories use default behavior
    (student has no diagnostic scan or is new).
    """

    dyslexia_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Reading/writing difficulty risk"
    )
    dyscalculia_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Math difficulty risk"
    )
    attention_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Attention-related risk"
    )
    auditory_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Auditory processing risk"
    )
    visual_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Visual processing risk"
    )
    last_scan_at: datetime | None = Field(
        default=None, description="When diagnostic scan was performed"
    )

    # Threshold constants
    ELEVATED_THRESHOLD: float = 0.5
    HIGH_THRESHOLD: float = 0.7

    @property
    def max_risk(self) -> float:
        """Get the maximum risk score across all indicators."""
        return max(
            self.dyslexia_risk,
            self.dyscalculia_risk,
            self.attention_risk,
            self.auditory_risk,
            self.visual_risk,
        )

    @property
    def has_elevated_risk(self) -> bool:
        """Check if any indicator is at ELEVATED level (>= 0.5)."""
        return self.max_risk >= self.ELEVATED_THRESHOLD

    @property
    def has_high_risk(self) -> bool:
        """Check if any indicator is at HIGH level (>= 0.7)."""
        return self.max_risk >= self.HIGH_THRESHOLD

    def is_risk_elevated(self, risk_value: float) -> bool:
        """Check if a specific risk value is at ELEVATED level."""
        return risk_value >= self.ELEVATED_THRESHOLD

    def is_risk_high(self, risk_value: float) -> bool:
        """Check if a specific risk value is at HIGH level."""
        return risk_value >= self.HIGH_THRESHOLD


class StudentContext(BaseModel):
    """Complete student context for theory calculations.

    Aggregated from all 4 memory layers plus diagnostic indicators
    to provide theories with the information they need to make
    personalized recommendations.

    The diagnostic field is None when:
    - Student is new and has no diagnostic scan
    - System just started with no prior scans
    - Diagnostic feature is not enabled

    Theories MUST handle diagnostic=None gracefully by using
    default behavior (no diagnostic-based adjustments).
    """

    student_id: UUID
    current_topic: TopicState | None = None
    related_topics: list[TopicState] = Field(default_factory=list)
    overall_mastery: float = Field(default=0.0, ge=0.0, le=1.0)
    topics_mastered_count: int = 0
    topics_struggling_count: int = 0
    vark_scores: VARKScores = Field(default_factory=VARKScores)
    recent_performance: RecentPerformance = Field(default_factory=RecentPerformance)
    best_time_of_day: str | None = None
    optimal_session_duration_minutes: int | None = None
    hint_dependency: float = Field(default=0.0, ge=0.0, le=1.0)
    interests: list[str] = Field(default_factory=list)
    effective_analogies: list[str] = Field(default_factory=list)
    fsrs_due_items_count: int = 0
    session_duration_minutes: int = 0
    diagnostic: DiagnosticIndicatorScores | None = Field(
        default=None,
        description="Diagnostic indicator scores (None if no scan performed)",
    )
    emotional: "EmotionalContext | None" = Field(
        default=None,
        description="Current emotional state (None if not available)",
    )
    context_retrieved_at: datetime = Field(default_factory=datetime.now)

    def get_emotional_difficulty_adjustment(self) -> float:
        """Get difficulty adjustment based on emotional state.

        Returns:
            Adjustment value (-0.2 to +0.1):
            - Negative: Lower difficulty (frustrated, anxious, confused)
            - Positive: Can increase difficulty (confident, excited)
            - Zero: No adjustment (neutral, curious, bored)
        """
        if not self.emotional:
            return 0.0
        return self.emotional.get_difficulty_adjustment()


class TheoryRecommendation(BaseModel):
    """Recommendation output from a single theory.

    Each theory produces its own recommendation which the
    TheoryOrchestrator combines into unified guidance.
    """

    theory_name: str = Field(description="Name of the theory")
    difficulty: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Recommended difficulty 0-1"
    )
    bloom_level: BloomLevel | None = Field(
        default=None, description="Recommended Bloom's taxonomy level"
    )
    content_format: ContentFormat | None = Field(
        default=None, description="Recommended content format"
    )
    scaffold_level: ScaffoldLevel | None = Field(
        default=None, description="Recommended scaffolding level"
    )
    hints_enabled: bool | None = Field(default=None, description="Whether to show hints")
    guide_vs_tell_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Socratic guide vs tell ratio"
    )
    questioning_style: QuestioningStyle | None = Field(
        default=None, description="Recommended questioning style"
    )
    next_review_at: datetime | None = Field(
        default=None, description="Next scheduled review time"
    )
    advance_to_next: bool | None = Field(
        default=None, description="Whether student should advance"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this recommendation"
    )
    rationale: str = Field(default="", description="Explanation for recommendation")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Theory-specific extra data"
    )


class TheoryConfig(BaseModel):
    """Configuration for a theory loaded from YAML."""

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class BaseTheory(ABC):
    """Abstract base class for all educational theories.

    Each theory implements the calculate method which takes a StudentContext
    and returns a TheoryRecommendation with its pedagogical guidance.
    """

    def __init__(self, config: TheoryConfig | None = None) -> None:
        """Initialize the theory with optional configuration.

        Args:
            config: Theory-specific configuration from YAML
        """
        self._config = config or TheoryConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the theory name."""
        ...

    @property
    def enabled(self) -> bool:
        """Check if the theory is enabled."""
        return self._config.enabled

    @property
    def weight(self) -> float:
        """Get the theory's weight for orchestration."""
        return self._config.weight

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter value.

        Args:
            key: Parameter key
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        return self._config.parameters.get(key, default)

    @abstractmethod
    def calculate(self, context: StudentContext) -> TheoryRecommendation:
        """Calculate recommendation based on student context.

        Args:
            context: Student's current learning context

        Returns:
            Theory-specific recommendation
        """
        ...

    def _create_recommendation(self, **kwargs: Any) -> TheoryRecommendation:
        """Helper to create a recommendation with theory name auto-filled.

        Args:
            **kwargs: TheoryRecommendation fields

        Returns:
            New TheoryRecommendation instance
        """
        return TheoryRecommendation(theory_name=self.name, **kwargs)


# Rebuild StudentContext to resolve EmotionalContext forward reference.
# This must be done after module load to avoid circular imports since
# EmotionalContext is a dataclass in src.core.emotional.context.
def _rebuild_student_context_model() -> None:
    """Rebuild StudentContext model to resolve forward references."""
    from src.core.emotional.context import EmotionalContext  # noqa: F401

    StudentContext.model_rebuild()


_rebuild_student_context_model()
