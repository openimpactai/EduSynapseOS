# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Constants for the Emotional Intelligence System.

This module defines all enums and thresholds used throughout
the emotional detection and response system.

All emotional analysis is performed by LLM through the
emotional_analyzer agent with MessageAnalysisCapability.
"""

from enum import Enum


class EmotionalState(str, Enum):
    """Possible emotional states for a student.

    States are detected from various signals and used to adapt
    the learning experience.
    """

    CONFIDENT = "confident"  # Feeling capable, high self-efficacy
    CURIOUS = "curious"  # Engaged, exploring, asking questions
    FRUSTRATED = "frustrated"  # Struggling, giving up signs
    CONFUSED = "confused"  # Not understanding, needs clarification
    EXCITED = "excited"  # High engagement, discovery moments
    BORED = "bored"  # Disengaged, rushing through
    ANXIOUS = "anxious"  # Worried about performance
    NEUTRAL = "neutral"  # Baseline state
    ENGAGED = "engaged"  # Actively participating


class EmotionalIntensity(str, Enum):
    """Intensity level of the emotional state."""

    LOW = "low"  # Mild indication, may not require action
    MODERATE = "moderate"  # Clear signal, consider adaptation
    HIGH = "high"  # Strong signal, requires immediate response


class EmotionalSignalSource(str, Enum):
    """Source of emotional signals."""

    LEARNING = "learning"  # Learning interactions (LLM sentiment analysis)
    PARENT_INPUT = "parent_input"  # Parent-provided context
    TEACHER_INPUT = "teacher_input"  # Teacher observations
    SYSTEM = "system"  # System-generated signals
    SELF_REPORT = "self_report"  # Student's own emotional expression (via companion)


# =============================================================================
# Thresholds
# =============================================================================

class EmotionalThresholds:
    """Thresholds for emotional state management."""

    # Confidence thresholds for detection
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MODERATE = 0.6
    CONFIDENCE_LOW = 0.4

    # Time windows (in minutes)
    SIGNAL_AGGREGATION_WINDOW = 15  # Look at last 15 minutes
    ALERT_RATE_LIMIT_WINDOW = 30  # Don't re-alert within 30 minutes

    # State change thresholds
    MIN_SIGNALS_FOR_STATE_CHANGE = 2  # Need multiple signals to change state
    DECAY_RATE = 0.1  # How fast emotional state decays toward neutral


# Emotion priority for conflict resolution (higher = more urgent)
EMOTION_PRIORITY = {
    EmotionalState.FRUSTRATED: 5,
    EmotionalState.ANXIOUS: 4,
    EmotionalState.CONFUSED: 3,
    EmotionalState.BORED: 2,
    EmotionalState.NEUTRAL: 1,
    EmotionalState.CONFIDENT: 2,
    EmotionalState.CURIOUS: 2,
    EmotionalState.EXCITED: 3,
    EmotionalState.ENGAGED: 2,
}


# =============================================================================
# Response Actions
# =============================================================================

class EmotionalResponseAction(str, Enum):
    """Actions to take in response to emotional states."""

    REDUCE_DIFFICULTY = "reduce_difficulty"
    INCREASE_DIFFICULTY = "increase_difficulty"
    OFFER_SUPPORT = "offer_support"
    OFFER_BREAK = "offer_break"
    OFFER_HINTS = "offer_hints"
    CELEBRATE_SUCCESS = "celebrate_success"
    ENCOURAGE = "encourage"
    CHANGE_ACTIVITY = "change_activity"
    ALERT_TEACHER = "alert_teacher"
    ALERT_PARENT = "alert_parent"
    NO_ACTION = "no_action"


# State to recommended actions mapping
STATE_TO_ACTIONS = {
    EmotionalState.FRUSTRATED: [
        EmotionalResponseAction.REDUCE_DIFFICULTY,
        EmotionalResponseAction.OFFER_SUPPORT,
        EmotionalResponseAction.OFFER_HINTS,
        EmotionalResponseAction.OFFER_BREAK,
    ],
    EmotionalState.CONFUSED: [
        EmotionalResponseAction.OFFER_SUPPORT,
        EmotionalResponseAction.OFFER_HINTS,
    ],
    EmotionalState.BORED: [
        EmotionalResponseAction.INCREASE_DIFFICULTY,
        EmotionalResponseAction.CHANGE_ACTIVITY,
    ],
    EmotionalState.ANXIOUS: [
        EmotionalResponseAction.OFFER_SUPPORT,
        EmotionalResponseAction.REDUCE_DIFFICULTY,
        EmotionalResponseAction.ENCOURAGE,
    ],
    EmotionalState.CONFIDENT: [
        EmotionalResponseAction.CELEBRATE_SUCCESS,
    ],
    EmotionalState.EXCITED: [
        EmotionalResponseAction.CELEBRATE_SUCCESS,
    ],
    EmotionalState.CURIOUS: [
        EmotionalResponseAction.NO_ACTION,
    ],
    EmotionalState.ENGAGED: [
        EmotionalResponseAction.NO_ACTION,
    ],
    EmotionalState.NEUTRAL: [
        EmotionalResponseAction.NO_ACTION,
    ],
}
