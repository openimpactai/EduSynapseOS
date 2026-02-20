# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inactivity monitor for detecting prolonged student absence.

This monitor detects when students have been inactive for extended
periods and generates alerts to prompt re-engagement efforts.
Different inactivity thresholds are used based on severity and
the student's historical patterns.

Inactivity levels:
- Warning: 3-6 days (gentle reminder)
- Concerning: 7-13 days (active outreach needed)
- Critical: 14+ days (urgent intervention)
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
)
from src.models.memory import FullMemoryContext

logger = logging.getLogger(__name__)


class InactivityMonitor(BaseMonitor):
    """Monitor that detects prolonged student inactivity.

    Analyzes last activity timestamp and learning patterns to
    determine if a student has been inactive for concerning
    durations. Generates appropriate alerts for different
    inactivity levels.

    Configuration:
        warning_days: Days for warning level (default: 3)
        concerning_days: Days for concerning level (default: 7)
        critical_days: Days for critical level (default: 14)
        check_scheduled_reviews: Whether to alert on missed FSRS reviews
    """

    # Configuration constants
    WARNING_DAYS = 3
    CONCERNING_DAYS = 7
    CRITICAL_DAYS = 14

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "inactivity_monitor"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.INACTIVITY_WARNING

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if student has been inactive for concerning duration.

        Checks for:
        1. Days since last activity
        2. Missed scheduled reviews (FSRS)
        3. Pattern deviation from normal activity

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if concerning inactivity detected, None otherwise.
        """
        # Get last activity timestamp
        last_activity = self._get_last_activity(context)

        if last_activity is None:
            # No activity recorded - this might be a new student
            logger.debug(
                "No activity found for student %s", context.student_id
            )
            return None

        # Calculate days inactive
        now = datetime.now(timezone.utc)
        days_inactive = (now - last_activity).days

        # Check inactivity level
        if days_inactive >= self.CRITICAL_DAYS:
            return self._create_inactivity_alert(
                context=context,
                days_inactive=days_inactive,
                level="critical",
                last_activity=last_activity,
            )

        if days_inactive >= self.CONCERNING_DAYS:
            return self._create_inactivity_alert(
                context=context,
                days_inactive=days_inactive,
                level="concerning",
                last_activity=last_activity,
            )

        if days_inactive >= self.WARNING_DAYS:
            # Check if this is unusual for this student
            if self._is_unusual_gap(context, days_inactive):
                return self._create_inactivity_alert(
                    context=context,
                    days_inactive=days_inactive,
                    level="warning",
                    last_activity=last_activity,
                )

        # Check for missed FSRS reviews
        missed_reviews = self._check_missed_reviews(context)
        if missed_reviews:
            return missed_reviews

        return None

    def _get_last_activity(self, context: FullMemoryContext) -> datetime | None:
        """Get the timestamp of the last student activity.

        Args:
            context: Full memory context.

        Returns:
            Last activity datetime or None.
        """
        if not context.episodic:
            return None

        # Find the most recent episode with a timestamp
        for episode in context.episodic:
            if episode.occurred_at:
                return episode.occurred_at

        return None

    def _is_unusual_gap(self, context: FullMemoryContext, days: int) -> bool:
        """Check if the inactivity gap is unusual for this student.

        Compare to their normal activity patterns.

        Args:
            context: Full memory context.
            days: Current days inactive.

        Returns:
            True if gap is unusual for this student.
        """
        if not context.procedural:
            return True  # No pattern data, assume unusual

        # Check if student typically has regular sessions
        patterns = context.procedural

        # If student has historical data showing frequent activity,
        # even a 3-day gap might be unusual
        if patterns.avg_session_minutes and patterns.avg_session_minutes > 20:
            # Engaged student - even 3 days is notable
            return True

        return False

    def _check_missed_reviews(self, context: FullMemoryContext) -> AlertData | None:
        """Check if student has overdue FSRS reviews.

        Args:
            context: Full memory context.

        Returns:
            AlertData if significant overdue reviews, None otherwise.
        """
        # Check semantic memory for overdue items
        if not context.semantic:
            return None

        # Look for FSRS data in procedural or semantic memory
        # This is a simplified check - full implementation would query ReviewItem table
        overdue_count = 0

        # Check semantic memories for items that might be due
        for topic in getattr(context.semantic, 'topic_masteries', []):
            if hasattr(topic, 'last_reviewed_at') and hasattr(topic, 'next_review_at'):
                if topic.next_review_at and datetime.now(timezone.utc) > topic.next_review_at:
                    overdue_count += 1

        if overdue_count >= 5:  # Threshold for alert
            return self._create_inactivity_alert(
                context=context,
                days_inactive=0,  # Not about days, about missed reviews
                level="review_overdue",
                overdue_count=overdue_count,
            )

        return None

    def _create_inactivity_alert(
        self,
        context: FullMemoryContext,
        days_inactive: int,
        level: str,
        last_activity: datetime | None = None,
        overdue_count: int = 0,
    ) -> AlertData:
        """Create an inactivity alert.

        Args:
            context: Full memory context.
            days_inactive: Number of days inactive.
            level: Inactivity level (warning, concerning, critical, review_overdue).
            last_activity: Last activity timestamp.
            overdue_count: Number of overdue reviews.

        Returns:
            AlertData for the inactivity.
        """
        severity = self._get_severity(level)
        title = self._build_title(level, days_inactive, overdue_count)
        message = self._build_message(context, level, days_inactive, last_activity, overdue_count)
        targets = self._get_targets(level)
        suggested_actions = self._build_suggested_actions(context, level, days_inactive)

        return self.create_alert(
            student_id=context.student_id,
            severity=severity,
            title=title,
            message=message,
            targets=targets,
            details={
                "level": level,
                "days_inactive": days_inactive,
                "last_activity": last_activity.isoformat() if last_activity else None,
                "overdue_reviews": overdue_count,
                "risk_level": self.get_risk_level_label(context),
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _get_severity(self, level: str) -> AlertSeverity:
        """Get alert severity based on inactivity level.

        Args:
            level: Inactivity level.

        Returns:
            Alert severity.
        """
        severity_map = {
            "warning": AlertSeverity.INFO,
            "concerning": AlertSeverity.WARNING,
            "critical": AlertSeverity.CRITICAL,
            "review_overdue": AlertSeverity.INFO,
        }
        return severity_map.get(level, AlertSeverity.INFO)

    def _get_targets(self, level: str) -> list[AlertTarget]:
        """Get alert targets based on inactivity level.

        Args:
            level: Inactivity level.

        Returns:
            List of alert targets.
        """
        if level == "critical":
            return [AlertTarget.TEACHER, AlertTarget.PARENT]

        if level == "concerning":
            return [AlertTarget.TEACHER, AlertTarget.PARENT]

        return [AlertTarget.TEACHER]

    def _build_title(self, level: str, days: int, overdue_count: int) -> str:
        """Build alert title.

        Args:
            level: Inactivity level.
            days: Days inactive.
            overdue_count: Overdue review count.

        Returns:
            Alert title.
        """
        if level == "critical":
            return f"Critical: Student Inactive for {days} Days"

        if level == "concerning":
            return f"Inactivity Alert: {days} Days Without Activity"

        if level == "warning":
            return f"Reminder: Student Inactive for {days} Days"

        if level == "review_overdue":
            return f"Review Reminder: {overdue_count} Overdue Items"

        return f"Inactivity Notice: {days} Days"

    def _build_message(
        self,
        context: FullMemoryContext,
        level: str,
        days: int,
        last_activity: datetime | None,
        overdue_count: int,
    ) -> str:
        """Build alert message.

        Args:
            context: Full memory context.
            level: Inactivity level.
            days: Days inactive.
            last_activity: Last activity timestamp.
            overdue_count: Overdue review count.

        Returns:
            Alert message.
        """
        if level == "critical":
            message = (
                f"The student has not been active for {days} days, which is significantly "
                "longer than expected. This extended absence may indicate disengagement, "
                "external factors, or technical issues. Immediate outreach is recommended."
            )

        elif level == "concerning":
            message = (
                f"The student has been inactive for {days} days. While some breaks are normal, "
                "this duration warrants a check-in to ensure everything is okay and help "
                "them re-engage with their learning."
            )

        elif level == "warning":
            message = (
                f"The student has not logged in for {days} days. A gentle reminder "
                "might help them maintain their learning momentum."
            )

        elif level == "review_overdue":
            message = (
                f"The student has {overdue_count} scheduled review items that are overdue. "
                "Regular review is important for long-term retention. Consider encouraging "
                "them to complete their reviews."
            )

        else:
            message = f"The student has been inactive for {days} days."

        # Add last activity context
        if last_activity:
            message += f"\n\nLast activity: {last_activity.strftime('%Y-%m-%d %H:%M')} UTC"

        # Add diagnostic context if relevant
        if self.has_elevated_risk(context):
            elevated = self.get_elevated_indicators(context)
            indicator_names = ", ".join(name for name, _ in elevated)
            message += (
                f"\n\nNote: This student has elevated diagnostic indicators "
                f"({indicator_names}). Re-engagement should consider their learning needs."
            )

        return message

    def _build_suggested_actions(
        self, context: FullMemoryContext, level: str, days: int
    ) -> list[str]:
        """Build suggested actions.

        Args:
            context: Full memory context.
            level: Inactivity level.
            days: Days inactive.

        Returns:
            List of suggested actions.
        """
        actions = []

        if level == "critical":
            actions.extend([
                "Contact the student's parent/guardian to check in",
                "Verify there are no technical access issues",
                "Schedule a one-on-one meeting when they return",
                "Consider reviewing and adjusting their learning plan",
            ])

        elif level == "concerning":
            actions.extend([
                "Send a friendly check-in message",
                "Review recent learning experience for any frustrations",
                "Consider contacting parents if no response",
                "Prepare a welcome-back activity",
            ])

        elif level == "warning":
            actions.extend([
                "Send a gentle reminder about their learning journey",
                "Highlight any upcoming interesting topics",
                "Offer a quick achievement recap to motivate",
            ])

        elif level == "review_overdue":
            actions.extend([
                "Encourage completing overdue reviews",
                "Explain the importance of spaced repetition",
                "Consider scheduling a review session",
            ])

        # Add personalized suggestions based on diagnostic profile
        if context.diagnostic:
            diag = context.diagnostic

            if diag.attention_risk >= 0.5:
                actions.append(
                    "For re-engagement, start with short, focused activities"
                )

            if diag.dyslexia_risk >= 0.5 or diag.dyscalculia_risk >= 0.5:
                actions.append(
                    "Ensure accommodations are in place for their return"
                )

        return actions
