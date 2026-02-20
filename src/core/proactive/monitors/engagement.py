# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Engagement monitor for detecting drops in student engagement.

This monitor detects when student engagement is declining based on
session duration patterns, activity frequency, and interaction patterns.
It is diagnostic-aware, adjusting expectations for students with
attention-related indicators.

Diagnostic integration:
- Attention risk students: More tolerant of shorter sessions (expected)
- Adjusts severity based on whether drop is consistent with diagnostic profile
- Provides attention-specific recommendations
"""

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
    ELEVATED_RISK_THRESHOLD,
)
from src.models.memory import EpisodicEventType, FullMemoryContext

logger = logging.getLogger(__name__)


class EngagementMonitor(BaseMonitor):
    """Monitor that detects declining engagement patterns.

    Analyzes session patterns, activity frequency, and interaction
    quality to identify students whose engagement may be dropping.
    Uses diagnostic context to adjust expectations for students
    with attention-related indicators.

    Configuration:
        session_duration_drop_threshold: Percentage drop to trigger (default: 0.4)
        min_sessions_for_comparison: Min sessions needed (default: 3)
        activity_gap_days: Days without activity for concern (default: 3)
        attention_tolerance_factor: Extra tolerance for attention risk (default: 0.2)
    """

    # Configuration constants
    SESSION_DURATION_DROP_THRESHOLD = 0.4  # 40% drop
    MIN_SESSIONS_FOR_COMPARISON = 3
    ACTIVITY_GAP_DAYS = 3
    ATTENTION_TOLERANCE_FACTOR = 0.2  # Extra 20% tolerance for attention risk

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "engagement_monitor"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.ENGAGEMENT_DROP

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if student engagement is declining.

        Checks for:
        1. Significant drop in session duration
        2. Reduced activity frequency
        3. Declining interaction quality (fewer positive events)

        Diagnostic adjustments:
        - Attention risk: Higher tolerance for session duration drops
        - Adjusts severity based on whether pattern matches diagnostic profile

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if engagement drop detected, None otherwise.
        """
        if not context.episodic or len(context.episodic) < 5:
            logger.debug(
                "Insufficient episodic data for student %s", context.student_id
            )
            return None

        # Check session duration drop
        duration_drop = self._check_session_duration_drop(context)
        if duration_drop:
            return duration_drop

        # Check activity frequency decline
        frequency_drop = self._check_activity_frequency(context)
        if frequency_drop:
            return frequency_drop

        # Check interaction quality decline
        quality_drop = self._check_interaction_quality(context)
        if quality_drop:
            return quality_drop

        return None

    def _check_session_duration_drop(
        self, context: FullMemoryContext
    ) -> AlertData | None:
        """Check for significant drop in session duration.

        Args:
            context: Full memory context.

        Returns:
            AlertData if significant drop detected, None otherwise.
        """
        if not context.procedural:
            return None

        patterns = context.procedural

        # Need historical data to compare
        if not patterns.avg_session_minutes:
            return None

        # Get current session duration from recent episodes
        current_duration = self._estimate_current_session_duration(context)
        if current_duration is None:
            return None

        avg_duration = patterns.avg_session_minutes

        # Calculate drop percentage
        if avg_duration <= 0:
            return None

        drop_percentage = (avg_duration - current_duration) / avg_duration

        # Adjust threshold based on attention risk
        threshold = self._get_duration_drop_threshold(context)

        if drop_percentage >= threshold:
            # Check if this is expected for attention risk students
            is_expected = self._is_drop_expected_for_profile(context, "session_duration")

            return self._create_engagement_alert(
                context=context,
                reason="session_duration_drop",
                details={
                    "current_duration_minutes": current_duration,
                    "average_duration_minutes": avg_duration,
                    "drop_percentage": round(drop_percentage * 100, 1),
                    "threshold_used": round(threshold * 100, 1),
                    "is_expected_for_profile": is_expected,
                },
                is_expected_for_profile=is_expected,
            )

        return None

    def _check_activity_frequency(self, context: FullMemoryContext) -> AlertData | None:
        """Check for declining activity frequency.

        Args:
            context: Full memory context.

        Returns:
            AlertData if activity gap detected, None otherwise.
        """
        if not context.episodic:
            return None

        # Find the most recent activity
        now = datetime.now(timezone.utc)
        most_recent = None

        for episode in context.episodic:
            if episode.occurred_at:
                if most_recent is None or episode.occurred_at > most_recent:
                    most_recent = episode.occurred_at

        if most_recent is None:
            return None

        # Calculate days since last activity
        days_inactive = (now - most_recent).days

        if days_inactive >= self.ACTIVITY_GAP_DAYS:
            return self._create_engagement_alert(
                context=context,
                reason="activity_gap",
                details={
                    "days_inactive": days_inactive,
                    "last_activity": most_recent.isoformat(),
                    "threshold_days": self.ACTIVITY_GAP_DAYS,
                },
            )

        return None

    def _check_interaction_quality(self, context: FullMemoryContext) -> AlertData | None:
        """Check for declining interaction quality.

        Compares recent positive vs negative event ratio.

        Args:
            context: Full memory context.

        Returns:
            AlertData if quality decline detected, None otherwise.
        """
        if not context.episodic or len(context.episodic) < 10:
            return None

        # Split episodes into recent and older
        mid_point = len(context.episodic) // 2
        recent = context.episodic[:mid_point]
        older = context.episodic[mid_point:]

        recent_ratio = self._calculate_positive_ratio(recent)
        older_ratio = self._calculate_positive_ratio(older)

        if older_ratio is None or recent_ratio is None:
            return None

        # Check for significant drop
        if older_ratio > 0 and recent_ratio < older_ratio * 0.6:  # 40% drop
            return self._create_engagement_alert(
                context=context,
                reason="quality_decline",
                details={
                    "recent_positive_ratio": round(recent_ratio, 2),
                    "older_positive_ratio": round(older_ratio, 2),
                    "ratio_drop": round((older_ratio - recent_ratio) / older_ratio * 100, 1),
                },
            )

        return None

    def _estimate_current_session_duration(
        self, context: FullMemoryContext
    ) -> float | None:
        """Estimate current session duration from recent episodes.

        Args:
            context: Full memory context.

        Returns:
            Estimated duration in minutes, or None.
        """
        if not context.episodic:
            return None

        # Find session boundaries in recent episodes
        session_episodes = []
        current_session_id = None

        for episode in context.episodic:
            if not episode.details:
                continue

            session_id = episode.details.get("session_id")
            if session_id:
                if current_session_id is None:
                    current_session_id = session_id
                    session_episodes.append(episode)
                elif session_id == current_session_id:
                    session_episodes.append(episode)
                else:
                    break  # Different session

        if len(session_episodes) < 2:
            return None

        # Calculate duration from first to last episode
        times = [ep.occurred_at for ep in session_episodes if ep.occurred_at]
        if len(times) < 2:
            return None

        duration = max(times) - min(times)
        return duration.total_seconds() / 60

    def _get_duration_drop_threshold(self, context: FullMemoryContext) -> float:
        """Get duration drop threshold based on diagnostic profile.

        Students with attention risk get higher tolerance (expected shorter sessions).

        Args:
            context: Full memory context.

        Returns:
            Threshold percentage.
        """
        threshold = self.SESSION_DURATION_DROP_THRESHOLD

        # Increase tolerance for attention risk students
        if context.diagnostic and context.diagnostic.attention_risk >= ELEVATED_RISK_THRESHOLD:
            threshold += self.ATTENTION_TOLERANCE_FACTOR
            logger.debug(
                "Increased duration threshold for attention risk student %s",
                context.student_id,
            )

        return threshold

    def _is_drop_expected_for_profile(
        self, context: FullMemoryContext, drop_type: str
    ) -> bool:
        """Check if the drop is expected given the student's diagnostic profile.

        Args:
            context: Full memory context.
            drop_type: Type of drop (session_duration, etc.).

        Returns:
            True if drop is consistent with diagnostic profile.
        """
        if not context.diagnostic:
            return False

        diag = context.diagnostic

        if drop_type == "session_duration":
            # Session duration drops are expected for attention risk
            return diag.attention_risk >= ELEVATED_RISK_THRESHOLD

        return False

    def _calculate_positive_ratio(self, episodes: list) -> float | None:
        """Calculate ratio of positive to total events.

        Args:
            episodes: List of episodic memories.

        Returns:
            Positive ratio (0-1) or None.
        """
        positive_types = [
            "correct_answer",
            "breakthrough",
            "mastery",
            "engagement",
        ]
        negative_types = [
            "incorrect_answer",
            "struggle",
            "confusion",
            "frustration",
        ]

        positive = 0
        total = 0

        for episode in episodes:
            if episode.event_type in positive_types:
                positive += 1
                total += 1
            elif episode.event_type in negative_types:
                total += 1

        if total == 0:
            return None

        return positive / total

    def _create_engagement_alert(
        self,
        context: FullMemoryContext,
        reason: str,
        details: dict,
        is_expected_for_profile: bool = False,
    ) -> AlertData:
        """Create an engagement drop alert.

        Args:
            context: Full memory context.
            reason: Why alert was triggered.
            details: Additional details.
            is_expected_for_profile: Whether drop is expected for profile.

        Returns:
            AlertData for the engagement drop.
        """
        # Adjust severity based on context
        severity = self._determine_severity(context, reason, is_expected_for_profile)

        # Build title and message
        title = self._build_title(reason, details)
        message = self._build_message(context, reason, details, is_expected_for_profile)

        # Build suggested actions
        suggested_actions = self._build_suggested_actions(context, reason)

        # Determine targets
        targets = [AlertTarget.TEACHER]
        if severity == AlertSeverity.CRITICAL:
            targets.append(AlertTarget.PARENT)

        return self.create_alert(
            student_id=context.student_id,
            severity=severity,
            title=title,
            message=message,
            targets=targets,
            details={
                **details,
                "reason": reason,
                "risk_level": self.get_risk_level_label(context),
                "is_expected_for_profile": is_expected_for_profile,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _determine_severity(
        self,
        context: FullMemoryContext,
        reason: str,
        is_expected_for_profile: bool,
    ) -> AlertSeverity:
        """Determine alert severity.

        Args:
            context: Full memory context.
            reason: Alert trigger reason.
            is_expected_for_profile: Whether drop is expected.

        Returns:
            Appropriate severity level.
        """
        # If drop is expected for diagnostic profile, lower severity
        if is_expected_for_profile:
            return AlertSeverity.INFO

        # Activity gaps are concerning
        if reason == "activity_gap":
            return AlertSeverity.WARNING

        # Quality decline is warning-level
        if reason == "quality_decline":
            return AlertSeverity.WARNING

        return AlertSeverity.INFO

    def _build_title(self, reason: str, details: dict) -> str:
        """Build alert title.

        Args:
            reason: Alert trigger reason.
            details: Alert details.

        Returns:
            Human-readable title.
        """
        if reason == "session_duration_drop":
            drop_pct = details.get("drop_percentage", 0)
            return f"Engagement Drop: Session Duration Down {drop_pct:.0f}%"

        if reason == "activity_gap":
            days = details.get("days_inactive", 0)
            return f"Engagement Concern: {days} Days Without Activity"

        if reason == "quality_decline":
            return "Engagement Drop: Interaction Quality Declining"

        return "Engagement Drop Detected"

    def _build_message(
        self,
        context: FullMemoryContext,
        reason: str,
        details: dict,
        is_expected_for_profile: bool,
    ) -> str:
        """Build alert message.

        Args:
            context: Full memory context.
            reason: Alert trigger reason.
            details: Alert details.
            is_expected_for_profile: Whether drop is expected.

        Returns:
            Human-readable message.
        """
        if reason == "session_duration_drop":
            current = details.get("current_duration_minutes", 0)
            avg = details.get("average_duration_minutes", 0)
            message = (
                f"The student's current session duration ({current:.0f} min) is "
                f"significantly shorter than their average ({avg:.0f} min). "
                "This may indicate reduced engagement or focus issues."
            )

        elif reason == "activity_gap":
            days = details.get("days_inactive", 0)
            message = (
                f"The student has not been active for {days} days. "
                "Consider reaching out to check on their progress and motivation."
            )

        elif reason == "quality_decline":
            recent_ratio = details.get("recent_positive_ratio", 0)
            message = (
                f"The student's recent interaction quality has declined. "
                f"Only {recent_ratio:.0%} of recent interactions were positive, "
                "compared to earlier performance. This may indicate frustration or disengagement."
            )

        else:
            message = "A decline in student engagement has been detected."

        # Add diagnostic context note
        if is_expected_for_profile:
            message += (
                "\n\nNote: This pattern may be consistent with the student's "
                "attention-related learning profile. Consider attention accommodations."
            )
        elif self.has_elevated_risk(context):
            elevated = self.get_elevated_indicators(context)
            indicator_names = ", ".join(name for name, _ in elevated)
            message += (
                f"\n\nNote: Consider the student's learning profile "
                f"(elevated indicators: {indicator_names})."
            )

        return message

    def _build_suggested_actions(
        self, context: FullMemoryContext, reason: str
    ) -> list[str]:
        """Build suggested actions.

        Args:
            context: Full memory context.
            reason: Alert trigger reason.

        Returns:
            List of suggested actions.
        """
        actions = []

        if reason == "session_duration_drop":
            actions.extend([
                "Check in with the student about their learning experience",
                "Consider shorter, more focused activities",
                "Provide more frequent encouragement and feedback",
            ])

        elif reason == "activity_gap":
            actions.extend([
                "Send a friendly reminder or check-in message",
                "Review if there were any recent difficulties that may have discouraged the student",
                "Consider scheduling a brief one-on-one session",
            ])

        elif reason == "quality_decline":
            actions.extend([
                "Review recent content difficulty and adjust if needed",
                "Provide additional scaffolding and support",
                "Consider a different approach or content format",
            ])

        # Add attention-specific actions
        if context.diagnostic and context.diagnostic.attention_risk >= ELEVATED_RISK_THRESHOLD:
            actions.extend([
                "Break learning into shorter, focused segments",
                "Incorporate movement breaks between activities",
                "Use interactive and varied content formats",
            ])

        return actions
