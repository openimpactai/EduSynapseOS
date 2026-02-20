# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Signal processing utilities for the Emotional Intelligence System.

This module provides utilities for:
- Aggregating emotional signals over time windows
- Analyzing trends in emotional state
- Calculating current emotional state from signals
"""

from collections import Counter
from datetime import datetime, timedelta

from src.core.emotional.constants import (
    EMOTION_PRIORITY,
    EmotionalIntensity,
    EmotionalState,
    EmotionalThresholds,
)
from src.core.emotional.context import EmotionalSignalData, EmotionalTrend


def aggregate_signals(
    signals: list[EmotionalSignalData],
    window_minutes: int = EmotionalThresholds.SIGNAL_AGGREGATION_WINDOW,
) -> list[EmotionalSignalData]:
    """Filter signals to those within the time window.

    Args:
        signals: List of all signals.
        window_minutes: Time window in minutes.

    Returns:
        Signals within the time window, sorted by time.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    recent = [s for s in signals if s.created_at >= cutoff]
    return sorted(recent, key=lambda s: s.created_at)


def analyze_trend(
    signals: list[EmotionalSignalData],
    window_minutes: int = EmotionalThresholds.SIGNAL_AGGREGATION_WINDOW,
) -> EmotionalTrend:
    """Analyze trend in emotional signals.

    Determines:
    - Overall direction (improving, stable, declining)
    - Dominant emotion in the window
    - Emotional volatility

    Args:
        signals: List of signals to analyze.
        window_minutes: Time window used for analysis.

    Returns:
        EmotionalTrend with analysis results.
    """
    recent_signals = aggregate_signals(signals, window_minutes)

    if not recent_signals:
        return EmotionalTrend(
            direction="stable",
            dominant_emotion=EmotionalState.NEUTRAL,
            emotion_counts={EmotionalState.NEUTRAL.value: 1},
            volatility=0.0,
            window_minutes=window_minutes,
        )

    # Count emotions
    emotion_counts: Counter[str] = Counter()
    for signal in recent_signals:
        if signal.detected_emotion:
            emotion_counts[signal.detected_emotion.value] += 1
        else:
            emotion_counts[EmotionalState.NEUTRAL.value] += 1

    # Find dominant emotion
    if emotion_counts:
        dominant_str = emotion_counts.most_common(1)[0][0]
        dominant_emotion = EmotionalState(dominant_str)
    else:
        dominant_emotion = EmotionalState.NEUTRAL

    # Calculate direction by comparing first and second half
    if len(recent_signals) >= 4:
        mid = len(recent_signals) // 2
        first_half = recent_signals[:mid]
        second_half = recent_signals[mid:]

        first_positive = sum(
            1 for s in first_half
            if s.detected_emotion in (
                EmotionalState.CONFIDENT,
                EmotionalState.EXCITED,
                EmotionalState.CURIOUS,
            )
        )
        second_positive = sum(
            1 for s in second_half
            if s.detected_emotion in (
                EmotionalState.CONFIDENT,
                EmotionalState.EXCITED,
                EmotionalState.CURIOUS,
            )
        )

        first_negative = sum(
            1 for s in first_half
            if s.detected_emotion in (
                EmotionalState.FRUSTRATED,
                EmotionalState.ANXIOUS,
                EmotionalState.CONFUSED,
            )
        )
        second_negative = sum(
            1 for s in second_half
            if s.detected_emotion in (
                EmotionalState.FRUSTRATED,
                EmotionalState.ANXIOUS,
                EmotionalState.CONFUSED,
            )
        )

        positive_change = second_positive - first_positive
        negative_change = second_negative - first_negative

        if positive_change > 1 or negative_change < -1:
            direction = "improving"
        elif negative_change > 1 or positive_change < -1:
            direction = "declining"
        else:
            direction = "stable"
    else:
        direction = "stable"

    # Calculate volatility (how much emotion changes)
    if len(recent_signals) >= 2:
        changes = 0
        for i in range(1, len(recent_signals)):
            prev = recent_signals[i - 1].detected_emotion
            curr = recent_signals[i].detected_emotion
            if prev != curr:
                changes += 1
        volatility = min(1.0, changes / (len(recent_signals) - 1))
    else:
        volatility = 0.0

    return EmotionalTrend(
        direction=direction,
        dominant_emotion=dominant_emotion,
        emotion_counts=dict(emotion_counts),
        volatility=volatility,
        window_minutes=window_minutes,
    )


def calculate_current_state(
    signals: list[EmotionalSignalData],
    window_minutes: int = EmotionalThresholds.SIGNAL_AGGREGATION_WINDOW,
) -> tuple[EmotionalState, EmotionalIntensity, float, list[str]]:
    """Calculate current emotional state from recent signals.

    Uses weighted voting where:
    - More recent signals have higher weight
    - Higher confidence signals have higher weight
    - Negative emotions have priority (safety first)

    Args:
        signals: List of signals to analyze.
        window_minutes: Time window for analysis.

    Returns:
        Tuple of (state, intensity, confidence, triggers).
    """
    recent_signals = aggregate_signals(signals, window_minutes)

    if not recent_signals:
        return (EmotionalState.NEUTRAL, EmotionalIntensity.LOW, 0.5, [])

    # Calculate weighted votes for each emotion
    now = datetime.utcnow()
    window_seconds = window_minutes * 60
    emotion_scores: dict[EmotionalState, float] = {}
    triggers: list[str] = []

    for signal in recent_signals:
        if not signal.detected_emotion:
            continue

        # Time decay: more recent signals have higher weight
        age_seconds = (now - signal.created_at).total_seconds()
        time_weight = max(0.1, 1.0 - (age_seconds / window_seconds))

        # Confidence weight
        confidence_weight = signal.confidence

        # Calculate weighted score
        score = time_weight * confidence_weight

        emotion = signal.detected_emotion
        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

        # Collect triggers
        if signal.signal_type not in triggers:
            triggers.append(signal.signal_type)

    if not emotion_scores:
        return (EmotionalState.NEUTRAL, EmotionalIntensity.LOW, 0.5, triggers)

    # Apply emotion priority (negative emotions are prioritized)
    prioritized_scores = {}
    for emotion, score in emotion_scores.items():
        priority = EMOTION_PRIORITY.get(emotion, 1)
        prioritized_scores[emotion] = score * (1 + priority * 0.1)

    # Find winning emotion
    winning_emotion = max(prioritized_scores.keys(), key=lambda e: prioritized_scores[e])
    winning_score = emotion_scores[winning_emotion]

    # Calculate confidence
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        confidence = winning_score / total_score
    else:
        confidence = 0.5

    # Calculate intensity based on signal count and recency
    recent_count = sum(
        1 for s in recent_signals
        if s.detected_emotion == winning_emotion
        and (now - s.created_at).total_seconds() < 300  # Last 5 minutes
    )

    if recent_count >= 3 or winning_score > 2.0:
        intensity = EmotionalIntensity.HIGH
    elif recent_count >= 2 or winning_score > 1.0:
        intensity = EmotionalIntensity.MODERATE
    else:
        intensity = EmotionalIntensity.LOW

    return (winning_emotion, intensity, confidence, triggers[:5])
