# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Struggle monitor for detecting student difficulties.

This monitor detects when a student is struggling with content,
based on consecutive incorrect answers and error patterns.
It is diagnostic-aware, adjusting thresholds for students with
elevated learning difficulty indicators.

Diagnostic integration:
- HIGH risk students: Trigger alert after 2 consecutive errors (earlier intervention)
- Normal students: Trigger alert after 3 consecutive errors
- Alert messages include diagnostic-specific recommendations
"""

import logging

from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
    ELEVATED_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)
from src.models.memory import FullMemoryContext

logger = logging.getLogger(__name__)


class StruggleMonitor(BaseMonitor):
    """Monitor that detects student struggle patterns.

    Analyzes recent performance to identify students who are
    having difficulty and need intervention. Uses diagnostic
    context to adjust sensitivity for at-risk students.

    Configuration:
        default_error_threshold: Errors before alert (default: 3)
        high_risk_error_threshold: Errors for HIGH risk students (default: 2)
        low_accuracy_threshold: Accuracy below this triggers concern (default: 0.4)
        min_answers_for_accuracy: Min answers for accuracy calculation (default: 5)
    """

    # Configuration constants
    DEFAULT_ERROR_THRESHOLD = 3
    HIGH_RISK_ERROR_THRESHOLD = 2
    LOW_ACCURACY_THRESHOLD = 0.4
    MIN_ANSWERS_FOR_ACCURACY = 5

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "struggle_monitor"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.STRUGGLE_DETECTED

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if student is struggling and needs intervention.

        Checks for:
        1. Consecutive incorrect answers above threshold
        2. Low recent accuracy (if enough data)

        Diagnostic adjustments:
        - HIGH risk: Lower error threshold for earlier intervention
        - Elevated indicators: Include specific recommendations in alert

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if struggle detected, None otherwise.
        """
        # Get recent performance from procedural memory
        if not context.procedural:
            logger.debug("No procedural data for student %s", context.student_id)
            return None

        # Determine error threshold based on diagnostic risk
        error_threshold = self._get_error_threshold(context)

        # Check consecutive errors
        consecutive_errors = self._get_consecutive_errors(context)

        if consecutive_errors >= error_threshold:
            return self._create_struggle_alert(
                context=context,
                reason="consecutive_errors",
                consecutive_errors=consecutive_errors,
                error_threshold=error_threshold,
            )

        # Check low accuracy if enough data
        accuracy = self._calculate_recent_accuracy(context)
        if accuracy is not None and accuracy < self.LOW_ACCURACY_THRESHOLD:
            return self._create_struggle_alert(
                context=context,
                reason="low_accuracy",
                accuracy=accuracy,
            )

        return None

    def _get_error_threshold(self, context: FullMemoryContext) -> int:
        """Get error threshold based on diagnostic risk level.

        HIGH risk students get lower threshold for earlier intervention.

        Args:
            context: Full memory context.

        Returns:
            Error count threshold.
        """
        if self.has_high_risk(context):
            logger.debug(
                "Using lower error threshold for HIGH risk student %s",
                context.student_id,
            )
            return self.HIGH_RISK_ERROR_THRESHOLD

        return self.DEFAULT_ERROR_THRESHOLD

    def _get_consecutive_errors(self, context: FullMemoryContext) -> int:
        """Get consecutive error count from episodic memory.

        Args:
            context: Full memory context.

        Returns:
            Number of consecutive incorrect answers.
        """
        if not context.episodic:
            return 0

        consecutive = 0
        for episode in context.episodic:
            # Check if this is an answer event
            if episode.event_type in ["incorrect_answer", "correct_answer"]:
                if episode.event_type == "incorrect_answer":
                    consecutive += 1
                else:
                    # Correct answer breaks the streak
                    break

        return consecutive

    def _calculate_recent_accuracy(self, context: FullMemoryContext) -> float | None:
        """Calculate recent accuracy from episodic memory.

        Args:
            context: Full memory context.

        Returns:
            Accuracy ratio (0-1) or None if insufficient data.
        """
        if not context.episodic:
            return None

        correct = 0
        total = 0

        for episode in context.episodic:
            if episode.event_type == "correct_answer":
                correct += 1
                total += 1
            elif episode.event_type == "incorrect_answer":
                total += 1

        if total < self.MIN_ANSWERS_FOR_ACCURACY:
            return None

        return correct / total

    def _create_struggle_alert(
        self,
        context: FullMemoryContext,
        reason: str,
        consecutive_errors: int = 0,
        error_threshold: int = 0,
        accuracy: float | None = None,
    ) -> AlertData:
        """Create a struggle alert with diagnostic-aware messaging.

        Args:
            context: Full memory context.
            reason: Why alert was triggered.
            consecutive_errors: Number of consecutive errors.
            error_threshold: Threshold that was exceeded.
            accuracy: Recent accuracy if applicable.

        Returns:
            AlertData for the struggle event.
        """
        # Determine severity based on diagnostic and severity
        severity = self._determine_severity(context, consecutive_errors, accuracy)

        # Build title and message
        if reason == "consecutive_errors":
            title = f"Student Struggling: {consecutive_errors} Consecutive Errors"
            message = self._build_consecutive_error_message(
                context, consecutive_errors, error_threshold
            )
        else:
            title = f"Student Struggling: Low Accuracy ({accuracy:.0%})"
            message = self._build_low_accuracy_message(context, accuracy)

        # Get topic info if available
        topic_full_code, topic_name = self._get_current_topic_info(context)
        topic_codes = self._parse_topic_full_code(topic_full_code) if topic_full_code else None

        # Build suggested actions
        suggested_actions = self._build_suggested_actions(context, reason)

        # Determine targets
        targets = self._determine_targets(context, severity)

        return self.create_alert(
            student_id=context.student_id,
            severity=severity,
            title=title,
            message=message,
            targets=targets,
            topic_codes=topic_codes,
            details={
                "reason": reason,
                "consecutive_errors": consecutive_errors,
                "error_threshold": error_threshold,
                "accuracy": accuracy,
                "risk_level": self.get_risk_level_label(context),
                "elevated_indicators": [
                    {"name": name, "score": score}
                    for name, score in self.get_elevated_indicators(context)
                ],
                "topic_full_code": topic_full_code,
                "topic_name": topic_name,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _determine_severity(
        self,
        context: FullMemoryContext,
        consecutive_errors: int,
        accuracy: float | None,
    ) -> AlertSeverity:
        """Determine alert severity based on context.

        Args:
            context: Full memory context.
            consecutive_errors: Number of consecutive errors.
            accuracy: Recent accuracy.

        Returns:
            Appropriate severity level.
        """
        # Critical if HIGH diagnostic risk AND struggling significantly
        if self.has_high_risk(context):
            if consecutive_errors >= 4 or (accuracy and accuracy < 0.25):
                return AlertSeverity.CRITICAL
            return AlertSeverity.WARNING

        # Warning for significant struggles
        if consecutive_errors >= 5 or (accuracy and accuracy < 0.3):
            return AlertSeverity.WARNING

        return AlertSeverity.INFO

    def _determine_targets(
        self,
        context: FullMemoryContext,
        severity: AlertSeverity,
    ) -> list[AlertTarget]:
        """Determine who should receive the alert.

        Args:
            context: Full memory context.
            severity: Alert severity.

        Returns:
            List of targets.
        """
        targets = [AlertTarget.TEACHER]

        # Critical alerts also go to parents
        if severity == AlertSeverity.CRITICAL:
            targets.append(AlertTarget.PARENT)

        # HIGH risk students: always include parent for struggle alerts
        if self.has_high_risk(context):
            if AlertTarget.PARENT not in targets:
                targets.append(AlertTarget.PARENT)

        return targets

    def _build_consecutive_error_message(
        self,
        context: FullMemoryContext,
        consecutive_errors: int,
        error_threshold: int,
    ) -> str:
        """Build message for consecutive error alert.

        Args:
            context: Full memory context.
            consecutive_errors: Number of consecutive errors.
            error_threshold: Threshold that was exceeded.

        Returns:
            Human-readable message.
        """
        base_message = (
            f"The student has made {consecutive_errors} consecutive incorrect answers. "
            "This pattern suggests they may be struggling with the current material "
            "and could benefit from additional support or a different approach."
        )

        # Add diagnostic context if available
        if self.has_elevated_risk(context):
            elevated = self.get_elevated_indicators(context)
            indicator_names = ", ".join(name for name, _ in elevated)

            if self.has_high_risk(context):
                base_message += (
                    f"\n\nNote: This student has elevated diagnostic indicators "
                    f"({indicator_names}). Earlier intervention is recommended."
                )
            else:
                base_message += (
                    f"\n\nNote: Consider the student's learning profile "
                    f"(elevated: {indicator_names}) when providing support."
                )

        return base_message

    def _build_low_accuracy_message(
        self,
        context: FullMemoryContext,
        accuracy: float | None,
    ) -> str:
        """Build message for low accuracy alert.

        Args:
            context: Full memory context.
            accuracy: Recent accuracy.

        Returns:
            Human-readable message.
        """
        accuracy_pct = (accuracy or 0) * 100
        base_message = (
            f"The student's recent accuracy is {accuracy_pct:.0f}%, which is below "
            "the expected threshold. This may indicate difficulty with the current "
            "topic or a need for foundational review."
        )

        if self.has_elevated_risk(context):
            elevated = self.get_elevated_indicators(context)
            indicator_names = ", ".join(name for name, _ in elevated)
            base_message += (
                f"\n\nConsider accommodations for the student's learning profile "
                f"(indicators: {indicator_names})."
            )

        return base_message

    def _build_suggested_actions(
        self,
        context: FullMemoryContext,
        reason: str,
    ) -> list[str]:
        """Build suggested actions based on context.

        Args:
            context: Full memory context.
            reason: Alert trigger reason.

        Returns:
            List of suggested actions.
        """
        actions = [
            "Review the topic with the student",
            "Consider providing additional worked examples",
            "Break down the problem into smaller steps",
        ]

        # Add diagnostic-specific suggestions
        if context.diagnostic:
            diag = context.diagnostic

            if diag.dyslexia_risk >= ELEVATED_RISK_THRESHOLD:
                actions.append("Provide audio or visual alternatives to text-heavy content")

            if diag.dyscalculia_risk >= ELEVATED_RISK_THRESHOLD:
                actions.append("Use visual math aids and manipulatives")
                actions.append("Allow extra time for math calculations")

            if diag.attention_risk >= ELEVATED_RISK_THRESHOLD:
                actions.append("Break tasks into shorter segments")
                actions.append("Provide more frequent breaks")

            if diag.visual_risk >= ELEVATED_RISK_THRESHOLD:
                actions.append("Provide verbal explanations alongside visual content")

            if diag.auditory_risk >= ELEVATED_RISK_THRESHOLD:
                actions.append("Provide written instructions in addition to verbal")

        return actions

    def _get_current_topic_info(
        self, context: FullMemoryContext
    ) -> tuple[str | None, str | None]:
        """Get current topic full code and name from context.

        Args:
            context: Full memory context.

        Returns:
            Tuple of (topic_full_code, topic_name) or (None, None).
        """
        # Try to get from recent episodic memory
        if context.episodic:
            for episode in context.episodic:
                if episode.topic_full_code:
                    return episode.topic_full_code, episode.topic_name

        return None, None

    def _parse_topic_full_code(self, topic_full_code: str) -> dict[str, str]:
        """Parse topic full code into component parts.

        Args:
            topic_full_code: Full topic code (e.g., 'UK-NC-2014.MAT.Y4.NPV.001').

        Returns:
            Dictionary with framework_code, subject_code, grade_code, unit_code, code.
        """
        parts = topic_full_code.split(".")
        if len(parts) != 5:
            # Return partial data if format doesn't match
            return {"full_code": topic_full_code}

        return {
            "framework_code": parts[0],
            "subject_code": parts[1],
            "grade_code": parts[2],
            "unit_code": parts[3],
            "code": parts[4],
        }
