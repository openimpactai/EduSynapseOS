# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Milestone monitor for celebrating student achievements.

This monitor detects when students achieve significant milestones
and generates positive alerts to celebrate their progress.
This is a positive reinforcement monitor that helps maintain
student motivation.

Milestones detected:
- Topic mastery achieved (mastery level >= threshold)
- Streak achievements (consecutive correct answers)
- Overall progress milestones (total topics mastered)
- Recovery milestones (improvement after struggle)
"""

import logging
from datetime import datetime, timezone
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


class MilestoneMonitor(BaseMonitor):
    """Monitor that detects and celebrates student achievements.

    Analyzes student progress to identify milestone achievements
    that warrant recognition. This helps maintain motivation
    and engagement through positive reinforcement.

    Configuration:
        mastery_threshold: Mastery level for topic completion (default: 0.8)
        streak_milestone_small: Small streak achievement (default: 5)
        streak_milestone_medium: Medium streak achievement (default: 10)
        streak_milestone_large: Large streak achievement (default: 20)
        topics_mastered_milestones: List of topic count milestones
    """

    # Configuration constants
    MASTERY_THRESHOLD = 0.8
    STREAK_MILESTONE_SMALL = 5
    STREAK_MILESTONE_MEDIUM = 10
    STREAK_MILESTONE_LARGE = 20
    TOPICS_MASTERED_MILESTONES = [5, 10, 25, 50, 100]

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "milestone_monitor"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.MILESTONE_ACHIEVED

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if student has achieved a milestone.

        Checks for:
        1. Topic mastery achievement
        2. Streak milestones
        3. Total topics mastered milestones
        4. Recovery from struggle

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if milestone achieved, None otherwise.
        """
        # Check for topic mastery
        mastery_milestone = self._check_topic_mastery(context)
        if mastery_milestone:
            return mastery_milestone

        # Check for streak achievements
        streak_milestone = self._check_streak_milestone(context)
        if streak_milestone:
            return streak_milestone

        # Check for total topics milestones
        topics_milestone = self._check_topics_mastered_milestone(context)
        if topics_milestone:
            return topics_milestone

        # Check for recovery milestone
        recovery_milestone = self._check_recovery_milestone(context)
        if recovery_milestone:
            return recovery_milestone

        return None

    def _check_topic_mastery(self, context: FullMemoryContext) -> AlertData | None:
        """Check if student just mastered a topic.

        Args:
            context: Full memory context.

        Returns:
            AlertData if mastery achieved, None otherwise.
        """
        if not context.episodic:
            return None

        # Look for recent mastery events
        for episode in context.episodic[:5]:  # Check recent episodes
            if episode.event_type == "mastery":
                # Check if this is a new mastery (not already celebrated)
                # Get topic info from episode fields (preferred) or details (fallback)
                topic_code = episode.topic_code
                topic_name = episode.topic_name or (
                    episode.details.get("topic_name", "a topic") if episode.details else "a topic"
                )
                mastery_level = episode.details.get("mastery_level", 0.8) if episode.details else 0.8

                return self._create_milestone_alert(
                    context=context,
                    milestone_type="topic_mastery",
                    title=f"Topic Mastered: {topic_name}",
                    message=self._build_mastery_message(context, topic_name, mastery_level),
                    details={
                        "topic_code": topic_code,
                        "topic_name": topic_name,
                        "mastery_level": mastery_level,
                    },
                )

        return None

    def _check_streak_milestone(self, context: FullMemoryContext) -> AlertData | None:
        """Check if student achieved a streak milestone.

        Args:
            context: Full memory context.

        Returns:
            AlertData if streak milestone achieved, None otherwise.
        """
        if not context.episodic:
            return None

        # Count current streak
        streak = 0
        for episode in context.episodic:
            if episode.event_type == "correct_answer":
                streak += 1
            elif episode.event_type == "incorrect_answer":
                break

        # Check if streak hits a milestone
        milestone = None
        if streak >= self.STREAK_MILESTONE_LARGE and streak % self.STREAK_MILESTONE_LARGE == 0:
            milestone = streak
        elif streak >= self.STREAK_MILESTONE_MEDIUM and streak % self.STREAK_MILESTONE_MEDIUM == 0:
            milestone = streak
        elif streak == self.STREAK_MILESTONE_SMALL:
            milestone = streak

        if milestone:
            return self._create_milestone_alert(
                context=context,
                milestone_type="streak_achievement",
                title=f"Amazing Streak: {milestone} Correct in a Row!",
                message=self._build_streak_message(context, milestone),
                details={
                    "streak_count": milestone,
                    "streak_category": self._get_streak_category(milestone),
                },
            )

        return None

    def _check_topics_mastered_milestone(
        self, context: FullMemoryContext
    ) -> AlertData | None:
        """Check if student reached a topics mastered milestone.

        Args:
            context: Full memory context.

        Returns:
            AlertData if milestone reached, None otherwise.
        """
        if not context.semantic:
            return None

        topics_mastered = context.semantic.topics_mastered

        # Check if this is a milestone
        if topics_mastered in self.TOPICS_MASTERED_MILESTONES:
            return self._create_milestone_alert(
                context=context,
                milestone_type="topics_milestone",
                title=f"Milestone: {topics_mastered} Topics Mastered!",
                message=self._build_topics_milestone_message(context, topics_mastered),
                details={
                    "topics_mastered": topics_mastered,
                    "next_milestone": self._get_next_topics_milestone(topics_mastered),
                },
            )

        return None

    def _check_recovery_milestone(self, context: FullMemoryContext) -> AlertData | None:
        """Check if student recovered from a struggle.

        Detects pattern: multiple errors followed by multiple correct answers.

        Args:
            context: Full memory context.

        Returns:
            AlertData if recovery detected, None otherwise.
        """
        if not context.episodic or len(context.episodic) < 6:
            return None

        # Look for pattern: 3+ correct after 3+ incorrect
        recent = context.episodic[:10]

        # Count recent correct streak
        correct_streak = 0
        for episode in recent:
            if episode.event_type == "correct_answer":
                correct_streak += 1
            else:
                break

        # Not enough correct answers
        if correct_streak < 3:
            return None

        # Check if there were previous errors
        error_count = 0
        for episode in recent[correct_streak:correct_streak + 5]:
            if episode.event_type == "incorrect_answer":
                error_count += 1

        if error_count >= 3:
            return self._create_milestone_alert(
                context=context,
                milestone_type="recovery",
                title="Great Comeback: Recovered from Difficulty!",
                message=self._build_recovery_message(context, correct_streak, error_count),
                details={
                    "correct_streak": correct_streak,
                    "previous_errors": error_count,
                },
            )

        return None

    def _create_milestone_alert(
        self,
        context: FullMemoryContext,
        milestone_type: str,
        title: str,
        message: str,
        details: dict,
    ) -> AlertData:
        """Create a milestone achievement alert.

        Args:
            context: Full memory context.
            milestone_type: Type of milestone.
            title: Alert title.
            message: Alert message.
            details: Additional details.

        Returns:
            AlertData for the milestone.
        """
        # All milestone alerts are INFO severity (positive)
        severity = AlertSeverity.INFO

        # Student and teacher should celebrate together
        targets = [AlertTarget.STUDENT, AlertTarget.TEACHER]

        # Major milestones also notify parents
        if milestone_type in ["topics_milestone", "recovery"]:
            if details.get("topics_mastered", 0) >= 10 or milestone_type == "recovery":
                targets.append(AlertTarget.PARENT)

        return self.create_alert(
            student_id=context.student_id,
            severity=severity,
            title=title,
            message=message,
            targets=targets,
            details={
                **details,
                "milestone_type": milestone_type,
                "risk_level": self.get_risk_level_label(context),
            },
            suggested_actions=self._build_milestone_actions(milestone_type, details),
            diagnostic_context=context.diagnostic,
        )

    def _build_mastery_message(
        self, context: FullMemoryContext, topic_name: str, mastery_level: float
    ) -> str:
        """Build message for topic mastery.

        Args:
            context: Full memory context.
            topic_name: Name of mastered topic.
            mastery_level: Mastery level achieved.

        Returns:
            Celebratory message.
        """
        message = (
            f"Congratulations! The student has achieved mastery of '{topic_name}' "
            f"with a mastery level of {mastery_level:.0%}. "
            "This is a significant accomplishment that demonstrates solid understanding of the material."
        )

        # Extra encouragement for students with diagnostic indicators
        if self.has_elevated_risk(context):
            message += (
                "\n\nThis achievement is especially noteworthy given the student's "
                "learning profile. Their persistence and hard work are paying off!"
            )

        return message

    def _build_streak_message(self, context: FullMemoryContext, streak: int) -> str:
        """Build message for streak achievement.

        Args:
            context: Full memory context.
            streak: Number of consecutive correct answers.

        Returns:
            Celebratory message.
        """
        category = self._get_streak_category(streak)

        if category == "large":
            message = (
                f"Incredible achievement! The student has answered {streak} questions "
                "correctly in a row. This demonstrates exceptional focus and mastery!"
            )
        elif category == "medium":
            message = (
                f"Excellent work! The student has a streak of {streak} correct answers. "
                "They're really on a roll!"
            )
        else:
            message = (
                f"Great progress! The student has answered {streak} questions correctly "
                "in a row. Keep up the momentum!"
            )

        return message

    def _build_topics_milestone_message(
        self, context: FullMemoryContext, topics_mastered: int
    ) -> str:
        """Build message for topics mastered milestone.

        Args:
            context: Full memory context.
            topics_mastered: Number of topics mastered.

        Returns:
            Celebratory message.
        """
        message = (
            f"Major milestone achieved! The student has now mastered {topics_mastered} topics. "
            "This represents significant progress in their learning journey."
        )

        next_milestone = self._get_next_topics_milestone(topics_mastered)
        if next_milestone:
            message += f" Next milestone: {next_milestone} topics!"

        return message

    def _build_recovery_message(
        self, context: FullMemoryContext, correct_streak: int, error_count: int
    ) -> str:
        """Build message for recovery from struggle.

        Args:
            context: Full memory context.
            correct_streak: Current correct streak.
            error_count: Previous error count.

        Returns:
            Celebratory message.
        """
        message = (
            f"The student has shown great resilience! After facing some difficulty "
            f"({error_count} incorrect answers), they've bounced back with {correct_streak} "
            "correct answers in a row. This demonstrates perseverance and learning from mistakes."
        )

        if self.has_elevated_risk(context):
            message += (
                "\n\nThis recovery is particularly impressive and shows the student's "
                "determination to succeed despite learning challenges."
            )

        return message

    def _get_streak_category(self, streak: int) -> str:
        """Get streak category based on count.

        Args:
            streak: Streak count.

        Returns:
            Category string.
        """
        if streak >= self.STREAK_MILESTONE_LARGE:
            return "large"
        if streak >= self.STREAK_MILESTONE_MEDIUM:
            return "medium"
        return "small"

    def _get_next_topics_milestone(self, current: int) -> int | None:
        """Get the next topics mastered milestone.

        Args:
            current: Current topics mastered.

        Returns:
            Next milestone or None if at max.
        """
        for milestone in self.TOPICS_MASTERED_MILESTONES:
            if milestone > current:
                return milestone
        return None

    def _build_milestone_actions(
        self, milestone_type: str, details: dict
    ) -> list[str]:
        """Build suggested actions for milestone.

        Args:
            milestone_type: Type of milestone.
            details: Milestone details.

        Returns:
            List of suggested actions.
        """
        if milestone_type == "topic_mastery":
            return [
                "Congratulate the student on their achievement",
                "Consider moving to more advanced content",
                "Review related topics to reinforce learning",
            ]

        if milestone_type == "streak_achievement":
            return [
                "Acknowledge the student's focus and effort",
                "Consider introducing slightly more challenging material",
                "Encourage continued practice",
            ]

        if milestone_type == "topics_milestone":
            return [
                "Celebrate this significant achievement",
                "Review progress and set new goals",
                "Share progress with parents/guardians",
            ]

        if milestone_type == "recovery":
            return [
                "Praise the student's perseverance",
                "Discuss what helped them overcome the difficulty",
                "Build on this success to maintain confidence",
            ]

        return []
