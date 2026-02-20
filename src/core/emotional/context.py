# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emotional context data structures.

This module defines the EmotionalContext dataclass that represents
a student's current emotional state and recent emotional history.

EmotionalContext is the primary data structure shared between:
- EmotionalStateService (producer)
- Educational Theories (consumer)
- Proactive Monitors (consumer)
- Companion Agent (consumer)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.emotional.constants import (
    EmotionalIntensity,
    EmotionalResponseAction,
    EmotionalState,
)


@dataclass
class EmotionalSignalData:
    """A single emotional signal from any source.

    Attributes:
        signal_type: Type of signal (e.g., consecutive_errors, sentiment_positive).
        source: Where the signal came from (practice, game, chat, etc.).
        value: Numeric or structured value of the signal.
        detected_emotion: Emotion detected from this signal (if any).
        confidence: Confidence score for the detection (0-1).
        activity_id: ID of the activity that generated this signal.
        activity_type: Type of activity (practice_session, game_session, etc.).
        context: Additional context data.
        created_at: When the signal was recorded.
    """

    signal_type: str
    source: str
    value: Any
    detected_emotion: EmotionalState | None = None
    confidence: float = 0.5
    activity_id: str | None = None
    activity_type: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EmotionalTrend:
    """Trend analysis of emotional signals over time.

    Attributes:
        direction: Overall direction (improving, stable, declining).
        dominant_emotion: Most frequent emotion in the window.
        emotion_counts: Count of each emotion in the window.
        volatility: How much the emotional state is changing (0-1).
        window_minutes: Time window used for analysis.
    """

    direction: str  # "improving", "stable", "declining"
    dominant_emotion: EmotionalState
    emotion_counts: dict[str, int] = field(default_factory=dict)
    volatility: float = 0.0
    window_minutes: int = 15


@dataclass
class ParentMoodInput:
    """Parent-reported mood information.

    Attributes:
        mood: Reported mood value.
        note: Optional note from parent.
        reported_at: When the mood was reported.
        valid_until: When this mood info expires.
    """

    mood: str
    note: str | None = None
    reported_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None


@dataclass
class EmotionalContext:
    """Complete emotional context for a student.

    This is the primary data structure passed to consumers like
    Educational Theories, Proactive Monitors, and Companion Agent.

    Attributes:
        student_id: The student's ID.
        current_state: Current detected emotional state.
        intensity: Intensity of the current state.
        confidence: Confidence in the current state detection.
        triggers: List of triggers that contributed to the current state.
        trend: Recent trend analysis.
        recent_signals: Recent signals (last N minutes).
        recent_signals_count: Total number of signals in the window.
        parent_mood: Parent-reported mood if available.
        recommended_actions: Actions recommended for this state.
        updated_at: When this context was last updated.
    """

    student_id: UUID
    current_state: EmotionalState = EmotionalState.NEUTRAL
    intensity: EmotionalIntensity = EmotionalIntensity.LOW
    confidence: float = 0.5
    triggers: list[str] = field(default_factory=list)
    trend: EmotionalTrend | None = None
    recent_signals: list[EmotionalSignalData] = field(default_factory=list)
    recent_signals_count: int = 0
    parent_mood: ParentMoodInput | None = None
    recommended_actions: list[EmotionalResponseAction] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_negative(self) -> bool:
        """Check if current state is negative."""
        return self.current_state in (
            EmotionalState.FRUSTRATED,
            EmotionalState.CONFUSED,
            EmotionalState.ANXIOUS,
        )

    @property
    def is_positive(self) -> bool:
        """Check if current state is positive."""
        return self.current_state in (
            EmotionalState.CONFIDENT,
            EmotionalState.CURIOUS,
            EmotionalState.EXCITED,
        )

    @property
    def needs_support(self) -> bool:
        """Check if student needs immediate support."""
        return (
            self.is_negative
            and self.intensity in (EmotionalIntensity.MODERATE, EmotionalIntensity.HIGH)
        )

    def get_difficulty_adjustment(self) -> float:
        """Get recommended difficulty adjustment.

        Returns:
            Float between -0.3 and +0.3 for difficulty adjustment.
            Negative = easier, Positive = harder.
        """
        adjustments = {
            # Negative states - reduce difficulty
            (EmotionalState.FRUSTRATED, EmotionalIntensity.HIGH): -0.3,
            (EmotionalState.FRUSTRATED, EmotionalIntensity.MODERATE): -0.2,
            (EmotionalState.FRUSTRATED, EmotionalIntensity.LOW): -0.1,
            (EmotionalState.CONFUSED, EmotionalIntensity.HIGH): -0.2,
            (EmotionalState.CONFUSED, EmotionalIntensity.MODERATE): -0.15,
            (EmotionalState.CONFUSED, EmotionalIntensity.LOW): -0.1,
            (EmotionalState.ANXIOUS, EmotionalIntensity.HIGH): -0.25,
            (EmotionalState.ANXIOUS, EmotionalIntensity.MODERATE): -0.15,
            (EmotionalState.ANXIOUS, EmotionalIntensity.LOW): -0.05,
            # Positive states - increase difficulty
            (EmotionalState.BORED, EmotionalIntensity.HIGH): 0.3,
            (EmotionalState.BORED, EmotionalIntensity.MODERATE): 0.2,
            (EmotionalState.BORED, EmotionalIntensity.LOW): 0.1,
            (EmotionalState.CONFIDENT, EmotionalIntensity.HIGH): 0.15,
            (EmotionalState.CONFIDENT, EmotionalIntensity.MODERATE): 0.1,
            (EmotionalState.CONFIDENT, EmotionalIntensity.LOW): 0.05,
        }
        return adjustments.get((self.current_state, self.intensity), 0.0)

    def get_scaffold_adjustment(self) -> int:
        """Get recommended scaffold level adjustment.

        Returns:
            Integer between -2 and +2.
            Negative = less support, Positive = more support.
        """
        if self.current_state in (EmotionalState.FRUSTRATED, EmotionalState.CONFUSED):
            if self.intensity == EmotionalIntensity.HIGH:
                return 2
            if self.intensity == EmotionalIntensity.MODERATE:
                return 1
        if self.current_state == EmotionalState.ANXIOUS:
            return 1
        if self.current_state == EmotionalState.BORED:
            if self.intensity == EmotionalIntensity.HIGH:
                return -2
            if self.intensity == EmotionalIntensity.MODERATE:
                return -1
        if self.current_state == EmotionalState.CONFIDENT:
            if self.intensity == EmotionalIntensity.HIGH:
                return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "student_id": str(self.student_id),
            "current_state": self.current_state.value,
            "intensity": self.intensity.value,
            "confidence": self.confidence,
            "triggers": self.triggers,
            "trend": {
                "direction": self.trend.direction,
                "dominant_emotion": self.trend.dominant_emotion.value,
                "volatility": self.trend.volatility,
            } if self.trend else None,
            "recent_signals_count": self.recent_signals_count,
            "parent_mood": {
                "mood": self.parent_mood.mood,
                "note": self.parent_mood.note,
            } if self.parent_mood else None,
            "is_negative": self.is_negative,
            "is_positive": self.is_positive,
            "needs_support": self.needs_support,
            "difficulty_adjustment": self.get_difficulty_adjustment(),
            "scaffold_adjustment": self.get_scaffold_adjustment(),
            "recommended_actions": [a.value for a in self.recommended_actions],
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create_neutral(cls, student_id: UUID) -> "EmotionalContext":
        """Create a neutral emotional context for a new session."""
        return cls(
            student_id=student_id,
            current_state=EmotionalState.NEUTRAL,
            intensity=EmotionalIntensity.LOW,
            confidence=0.5,
            triggers=[],
        )
