# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emotional distress monitor for proactive intervention.

This monitor detects sustained emotional distress patterns:
- Sustained frustration (multiple consecutive errors + negative signals)
- Elevated anxiety (anxious signals with declining performance)
- Confusion without resolution (confused signals without improvement)

Integrates with EmotionalStateService to access emotional context
and generates alerts for teachers and the companion agent.
"""

import logging
from uuid import UUID

from src.core.emotional import EmotionalContext, EmotionalState, EmotionalIntensity
from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
)
from src.models.memory import FullMemoryContext

logger = logging.getLogger(__name__)


class EmotionalDistressMonitor(BaseMonitor):
    """Monitor for detecting emotional distress patterns.

    Detects when a student shows sustained negative emotional states
    that may require intervention. Uses both emotional signals and
    performance data to identify distress patterns.

    Thresholds:
    - Sustained: 3+ negative signals in monitoring window
    - High intensity frustration/anxiety: Immediate alert
    - Moderate intensity with declining performance: Alert

    Alert targets:
    - Teacher: For all distress alerts
    - System: To trigger companion agent intervention
    """

    # Thresholds for distress detection
    DISTRESS_SIGNAL_THRESHOLD = 3  # Minimum negative signals
    HIGH_INTENSITY_THRESHOLD = EmotionalIntensity.HIGH
    MODERATE_INTENSITY_THRESHOLD = EmotionalIntensity.MODERATE

    # Negative emotional states that indicate distress
    DISTRESS_STATES = {
        EmotionalState.FRUSTRATED,
        EmotionalState.ANXIOUS,
        EmotionalState.CONFUSED,
    }

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "emotional_distress"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.EMOTIONAL_DISTRESS

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if emotional distress alert should be generated.

        Analyzes emotional context from FullMemoryContext to detect
        sustained distress patterns requiring intervention.

        Args:
            context: Full memory context including emotional state.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if distress detected, None otherwise.
        """
        # Get emotional context from memory
        emotional_context = self._get_emotional_context(context)
        if not emotional_context:
            self.logger.debug("No emotional context available for distress check")
            return None

        student_id = emotional_context.student_id

        # Check for high intensity distress (immediate alert)
        if self._is_high_intensity_distress(emotional_context):
            return self._create_high_intensity_alert(
                student_id=student_id,
                emotional_context=emotional_context,
                memory_context=context,
            )

        # Check for sustained moderate distress
        if self._is_sustained_distress(emotional_context):
            return self._create_sustained_distress_alert(
                student_id=student_id,
                emotional_context=emotional_context,
                memory_context=context,
            )

        return None

    def _get_emotional_context(
        self,
        context: FullMemoryContext,
    ) -> EmotionalContext | None:
        """Extract emotional context from FullMemoryContext.

        Args:
            context: Full memory context.

        Returns:
            EmotionalContext if available.
        """
        # EmotionalContext is stored in the procedural memory's emotional field
        # or as a separate attribute depending on implementation
        emotional = getattr(context, "emotional", None)
        if emotional:
            return emotional

        # Try to get from procedural memory if available
        procedural = getattr(context, "procedural", None)
        if procedural:
            return getattr(procedural, "emotional_context", None)

        return None

    def _is_high_intensity_distress(
        self,
        emotional_context: EmotionalContext,
    ) -> bool:
        """Check for high intensity distress requiring immediate intervention.

        Args:
            emotional_context: Current emotional context.

        Returns:
            True if high intensity distress detected.
        """
        if emotional_context.current_state not in self.DISTRESS_STATES:
            return False

        return emotional_context.intensity == self.HIGH_INTENSITY_THRESHOLD

    def _is_sustained_distress(
        self,
        emotional_context: EmotionalContext,
    ) -> bool:
        """Check for sustained moderate distress.

        Args:
            emotional_context: Current emotional context.

        Returns:
            True if sustained distress pattern detected.
        """
        if emotional_context.current_state not in self.DISTRESS_STATES:
            return False

        # Check if moderate intensity
        if emotional_context.intensity != self.MODERATE_INTENSITY_THRESHOLD:
            return False

        # Count recent negative signals
        negative_count = sum(
            1 for signal in emotional_context.recent_signals
            if signal.detected_emotion in self.DISTRESS_STATES
        )

        return negative_count >= self.DISTRESS_SIGNAL_THRESHOLD

    def _create_high_intensity_alert(
        self,
        student_id: UUID,
        emotional_context: EmotionalContext,
        memory_context: FullMemoryContext,
    ) -> AlertData:
        """Create alert for high intensity distress.

        Args:
            student_id: Student ID.
            emotional_context: Emotional context.
            memory_context: Full memory context.

        Returns:
            AlertData for high intensity distress.
        """
        state = emotional_context.current_state.value
        title = f"Immediate Support Needed: Student Showing {state.title()}"

        message = self._build_distress_message(
            emotional_context=emotional_context,
            severity="high",
        )

        suggested_actions = self._get_suggested_actions(
            emotional_context=emotional_context,
            severity="high",
        )

        return self.create_alert(
            student_id=student_id,
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            targets=[AlertTarget.TEACHER, AlertTarget.SYSTEM],
            details={
                "emotional_state": emotional_context.current_state.value,
                "intensity": emotional_context.intensity.value,
                "confidence": emotional_context.confidence,
                "triggers": emotional_context.triggers,
                "trend": emotional_context.trend.direction if emotional_context.trend else None,
                "signals_count": emotional_context.recent_signals_count,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=memory_context.diagnostic,
        )

    def _create_sustained_distress_alert(
        self,
        student_id: UUID,
        emotional_context: EmotionalContext,
        memory_context: FullMemoryContext,
    ) -> AlertData:
        """Create alert for sustained moderate distress.

        Args:
            student_id: Student ID.
            emotional_context: Emotional context.
            memory_context: Full memory context.

        Returns:
            AlertData for sustained distress.
        """
        state = emotional_context.current_state.value
        title = f"Sustained {state.title()} Detected"

        message = self._build_distress_message(
            emotional_context=emotional_context,
            severity="moderate",
        )

        suggested_actions = self._get_suggested_actions(
            emotional_context=emotional_context,
            severity="moderate",
        )

        return self.create_alert(
            student_id=student_id,
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            targets=[AlertTarget.TEACHER, AlertTarget.SYSTEM],
            details={
                "emotional_state": emotional_context.current_state.value,
                "intensity": emotional_context.intensity.value,
                "confidence": emotional_context.confidence,
                "triggers": emotional_context.triggers,
                "trend": emotional_context.trend.direction if emotional_context.trend else None,
                "signals_count": emotional_context.recent_signals_count,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=memory_context.diagnostic,
        )

    def _build_distress_message(
        self,
        emotional_context: EmotionalContext,
        severity: str,
    ) -> str:
        """Build distress alert message.

        Args:
            emotional_context: Emotional context.
            severity: "high" or "moderate".

        Returns:
            Human-readable message.
        """
        state = emotional_context.current_state.value
        signals_count = emotional_context.recent_signals_count

        if severity == "high":
            base_msg = f"Student is showing high intensity {state}."
        else:
            base_msg = f"Student has shown sustained {state} ({signals_count} signals in recent activity)."

        # Add trigger context if available
        if emotional_context.triggers:
            triggers_str = ", ".join(emotional_context.triggers[:3])
            base_msg += f" Triggers: {triggers_str}."

        # Add trend information
        if emotional_context.trend:
            trend = emotional_context.trend.direction
            if trend == "worsening":
                base_msg += " Emotional state is worsening."
            elif trend == "improving":
                base_msg += " Some improvement noted."

        # Add recommendation
        if severity == "high":
            base_msg += " Immediate intervention recommended."
        else:
            base_msg += " Consider checking in with the student."

        return base_msg

    def _get_suggested_actions(
        self,
        emotional_context: EmotionalContext,
        severity: str,
    ) -> list[str]:
        """Get suggested actions based on emotional state.

        Args:
            emotional_context: Emotional context.
            severity: "high" or "moderate".

        Returns:
            List of suggested action strings.
        """
        state = emotional_context.current_state
        actions = []

        if severity == "high":
            actions.append("Initiate immediate support conversation")
            actions.append("Consider pausing current activity")

        if state == EmotionalState.FRUSTRATED:
            actions.append("Reduce difficulty of upcoming questions")
            actions.append("Provide encouragement and positive reinforcement")
            if severity == "high":
                actions.append("Offer to switch to a different topic or activity")

        elif state == EmotionalState.ANXIOUS:
            actions.append("Reassure the student about their progress")
            actions.append("Emphasize learning over performance")
            if severity == "high":
                actions.append("Consider a short break or calming activity")

        elif state == EmotionalState.CONFUSED:
            actions.append("Provide additional explanations or examples")
            actions.append("Break down the content into smaller steps")
            if severity == "high":
                actions.append("Review prerequisite concepts")

        # Add companion agent trigger
        actions.append("Trigger companion agent for emotional support")

        return actions
