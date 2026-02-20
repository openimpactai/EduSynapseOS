# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base detector classes for learning difficulty detection.

This module provides the abstract base class and data structures
for all learning difficulty detectors in the diagnostic system.

IMPORTANT: Detectors identify INDICATORS only, not diagnoses.
Professional evaluation is recommended for concerning patterns.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from src.infrastructure.database.models.tenant.memory import (
    EpisodicMemory,
    SemanticMemory,
)
from src.infrastructure.database.models.tenant.practice import (
    EvaluationResult,
    PracticeSession,
    StudentAnswer,
)


class IndicatorType(str, Enum):
    """Types of learning difficulty indicators."""

    DYSLEXIA = "dyslexia"
    DYSCALCULIA = "dyscalculia"
    ATTENTION = "attention"
    AUDITORY = "auditory"
    VISUAL = "visual"


class ThresholdLevel(str, Enum):
    """Risk threshold levels."""

    LOW = "low"  # < 0.3
    MEDIUM = "medium"  # 0.3 - 0.5
    ELEVATED = "elevated"  # 0.5 - 0.7
    HIGH = "high"  # >= 0.7


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types.

    Handles common non-serializable types:
    - set -> list
    - tuple -> list
    - Decimal -> float
    - datetime -> isoformat string
    - bytes -> string (decoded as UTF-8)

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version of the object.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {
            str(k): _make_json_serializable(v)
            for k, v in obj.items()
        }
    # Fallback: convert to string
    return str(obj)


@dataclass
class Evidence:
    """Evidence supporting a detected indicator.

    Attributes:
        category: Category of evidence (e.g., "pattern", "performance", "behavior").
        description: Human-readable description.
        data: Raw data supporting the evidence.
        timestamp: When the evidence was observed.
        weight: How much this evidence contributes to the score (0.0-1.0).
    """

    category: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence to dictionary for JSON storage.

        Ensures all nested data is JSON-serializable by converting
        non-serializable types (set, tuple, Decimal, etc.) to
        their serializable equivalents.
        """
        return {
            "category": self.category,
            "description": self.description,
            "data": _make_json_serializable(self.data),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "weight": self.weight,
        }


@dataclass
class DetectorResult:
    """Result from a detector analysis.

    Attributes:
        indicator_type: Type of indicator detected.
        risk_score: Risk score (0.0-1.0).
        confidence: Confidence in the result (0.0-1.0).
        threshold_level: Categorized threshold level.
        evidence: List of supporting evidence.
        sample_size: Number of data points analyzed.
        analysis_summary: Brief summary of the analysis.
    """

    indicator_type: IndicatorType
    risk_score: float
    confidence: float
    threshold_level: ThresholdLevel
    evidence: list[Evidence] = field(default_factory=list)
    sample_size: int = 0
    analysis_summary: str = ""

    @classmethod
    def no_data(cls, indicator_type: IndicatorType) -> "DetectorResult":
        """Create a result indicating insufficient data."""
        return cls(
            indicator_type=indicator_type,
            risk_score=0.0,
            confidence=0.0,
            threshold_level=ThresholdLevel.LOW,
            evidence=[],
            sample_size=0,
            analysis_summary="Insufficient data for analysis",
        )

    @property
    def is_significant(self) -> bool:
        """Check if the result is significant (threshold >= elevated)."""
        return self.threshold_level in [ThresholdLevel.ELEVATED, ThresholdLevel.HIGH]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "indicator_type": self.indicator_type.value,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "threshold_level": self.threshold_level.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "sample_size": self.sample_size,
            "analysis_summary": self.analysis_summary,
        }


@dataclass
class StudentData:
    """Aggregated student data for detector analysis.

    Attributes:
        student_id: Student identifier.
        semantic_memories: Knowledge state and mastery data.
        episodic_memories: Learning event history.
        practice_sessions: Practice session records.
        student_answers: Answer records with timing.
        evaluation_results: Evaluation results with misconceptions.
    """

    student_id: str
    semantic_memories: list[SemanticMemory] = field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = field(default_factory=list)
    practice_sessions: list[PracticeSession] = field(default_factory=list)
    student_answers: list[StudentAnswer] = field(default_factory=list)
    evaluation_results: list[EvaluationResult] = field(default_factory=list)

    @property
    def has_sufficient_data(self) -> bool:
        """Check if there's enough data for meaningful analysis."""
        return (
            len(self.practice_sessions) >= 3
            and len(self.student_answers) >= 10
        )

    @property
    def total_answers(self) -> int:
        """Get total number of answers."""
        return len(self.student_answers)

    @property
    def total_sessions(self) -> int:
        """Get total number of sessions."""
        return len(self.practice_sessions)


class BaseDetector(ABC):
    """Abstract base class for learning difficulty detectors.

    Each detector analyzes student data to identify potential indicators
    of specific learning difficulties. Detectors use rule-based algorithms
    and statistical analysis, not AI/LLM.

    IMPORTANT: Results are indicators for monitoring only, not diagnoses.

    Configuration is loaded from YAML files and provides:
    - Detector-specific thresholds (alert, concern)
    - Indicator weights and thresholds
    - Minimum data point requirements
    """

    # Minimum data requirements (defaults, can be overridden by config)
    MIN_ANSWERS_REQUIRED = 10
    MIN_SESSIONS_REQUIRED = 3

    # Default thresholds (used when config is not available)
    LOW_THRESHOLD = 0.3
    MEDIUM_THRESHOLD = 0.5
    ELEVATED_THRESHOLD = 0.7

    def __init__(self) -> None:
        """Initialize the detector."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._config = None  # Lazy loaded detector config

    @property
    @abstractmethod
    def indicator_type(self) -> IndicatorType:
        """Return the type of indicator this detector identifies."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of the detector."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return description of what this detector identifies."""
        pass

    @property
    def detector_type(self) -> str:
        """Return detector type string for config lookup.

        Maps IndicatorType enum to YAML config key.

        Returns:
            Detector type string (e.g., "dyslexia", "attention").
        """
        return self.indicator_type.value

    @property
    def config(self):
        """Get detector-specific configuration from YAML.

        Lazy loads configuration on first access.

        Returns:
            DetectorConfig or None if not configured.
        """
        if self._config is None:
            from src.core.diagnostics.config import get_diagnostic_config

            diagnostic_config = get_diagnostic_config()
            self._config = diagnostic_config.get_detector_config(self.detector_type)
        return self._config

    @property
    def alert_threshold(self) -> float:
        """Get alert threshold from config.

        Returns:
            Alert threshold (risk score that triggers alert level).
        """
        if self.config:
            return self.config.alert_threshold
        return 0.6  # Default

    @property
    def concern_threshold(self) -> float:
        """Get concern threshold from config.

        Returns:
            Concern threshold (risk score that triggers concern level).
        """
        if self.config:
            return self.config.concern_threshold
        return 0.4  # Default

    @property
    def min_data_points(self) -> int:
        """Get minimum data points from config.

        Returns:
            Minimum number of data points required for analysis.
        """
        if self.config:
            return self.config.min_data_points
        return 30  # Default

    def get_indicator_weight(self, indicator_name: str) -> float:
        """Get weight for a specific indicator from config.

        Args:
            indicator_name: Name of the indicator (e.g., "spelling_errors").

        Returns:
            Weight value from config or default 0.0.
        """
        if self.config and indicator_name in self.config.indicators:
            return self.config.indicators[indicator_name].weight
        return 0.0

    def get_indicator_threshold(self, indicator_name: str) -> float:
        """Get threshold for a specific indicator from config.

        Args:
            indicator_name: Name of the indicator.

        Returns:
            Threshold value from config or default 0.0.
        """
        if self.config and indicator_name in self.config.indicators:
            return self.config.indicators[indicator_name].threshold
        return 0.0

    @abstractmethod
    async def analyze(self, student_data: StudentData) -> DetectorResult:
        """Analyze student data to detect indicators.

        Args:
            student_data: Aggregated student data for analysis.

        Returns:
            DetectorResult with risk score, confidence, and evidence.
        """
        pass

    def calculate_threshold_level(self, risk_score: float) -> ThresholdLevel:
        """Determine threshold level from risk score.

        Args:
            risk_score: Risk score between 0.0 and 1.0.

        Returns:
            Corresponding threshold level.
        """
        if risk_score >= self.ELEVATED_THRESHOLD:
            return ThresholdLevel.HIGH
        elif risk_score >= self.MEDIUM_THRESHOLD:
            return ThresholdLevel.ELEVATED
        elif risk_score >= self.LOW_THRESHOLD:
            return ThresholdLevel.MEDIUM
        return ThresholdLevel.LOW

    def calculate_confidence(
        self,
        sample_size: int,
        evidence_count: int,
        evidence_consistency: float = 1.0,
    ) -> float:
        """Calculate confidence based on available data.

        Confidence is influenced by:
        - Sample size (more data = higher confidence)
        - Number of evidence items
        - Consistency of evidence

        Args:
            sample_size: Number of data points analyzed.
            evidence_count: Number of evidence items found.
            evidence_consistency: How consistent the evidence is (0.0-1.0).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Sample size factor (logarithmic scale, caps at ~100 samples)
        sample_factor = min(1.0, (sample_size / 50) ** 0.5)

        # Evidence factor
        evidence_factor = min(1.0, evidence_count / 5)

        # Combined confidence
        confidence = (
            sample_factor * 0.4
            + evidence_factor * 0.3
            + evidence_consistency * 0.3
        )

        return round(min(1.0, max(0.0, confidence)), 2)

    def has_sufficient_data(self, student_data: StudentData) -> bool:
        """Check if student has enough data for meaningful analysis.

        Args:
            student_data: Student data to check.

        Returns:
            True if sufficient data exists.
        """
        return (
            len(student_data.student_answers) >= self.MIN_ANSWERS_REQUIRED
            and len(student_data.practice_sessions) >= self.MIN_SESSIONS_REQUIRED
        )

    def create_evidence(
        self,
        category: str,
        description: str,
        data: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        weight: float = 1.0,
    ) -> Evidence:
        """Create an evidence item.

        Args:
            category: Category of evidence.
            description: Human-readable description.
            data: Supporting data.
            timestamp: When evidence was observed.
            weight: Weight of this evidence.

        Returns:
            Evidence instance.
        """
        return Evidence(
            category=category,
            description=description,
            data=data or {},
            timestamp=timestamp,
            weight=weight,
        )

    def _calculate_accuracy(
        self,
        correct: int,
        total: int,
    ) -> float:
        """Calculate accuracy percentage.

        Args:
            correct: Number of correct items.
            total: Total items.

        Returns:
            Accuracy as a decimal (0.0-1.0).
        """
        if total == 0:
            return 0.0
        return correct / total

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values.

        Args:
            values: List of numeric values.

        Returns:
            Variance of the values.
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / len(values)

    def _calculate_coefficient_of_variation(self, values: list[float]) -> float:
        """Calculate coefficient of variation (CV).

        CV = standard deviation / mean
        Higher CV indicates more variability.

        Args:
            values: List of numeric values.

        Returns:
            Coefficient of variation.
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = self._calculate_variance(values)
        std_dev = variance ** 0.5
        return std_dev / mean
