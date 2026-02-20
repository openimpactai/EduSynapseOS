# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics service module.

This module provides the main analytics service for fetching
dashboards, reports, and class-level analytics.

The AnalyticsService queries aggregated data from:
- DailySummary: Daily aggregated metrics
- MasterySnapshot: Point-in-time mastery records
- EngagementMetric: Engagement patterns
- AnalyticsEvent: Raw event data (for recent activity)

Usage:
    from src.domains.analytics import AnalyticsService

    service = AnalyticsService(db=db_session, tenant_code="acme")

    # Get student dashboard
    dashboard = await service.get_dashboard(
        student_id=student_uuid,
        period_days=7,
    )

    # Get progress report
    report = await service.get_progress_report(
        student_id=student_uuid,
        period_days=30,
    )

    # Get class analytics
    class_data = await service.get_class_analytics(
        class_id=class_uuid,
        period_days=30,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.analytics import (
    AnalyticsEvent,
    DailySummary,
    EngagementMetric,
    MasterySnapshot,
)
from src.infrastructure.database.models.tenant.curriculum import Topic
from src.infrastructure.database.models.tenant.memory import SemanticMemory
from src.infrastructure.database.models.tenant.user import User

logger = logging.getLogger(__name__)


@dataclass
class DashboardStats:
    """Summary statistics for dashboard."""

    total_sessions: int = 0
    total_time_minutes: int = 0
    questions_answered: int = 0
    questions_correct: int = 0
    average_accuracy: float | None = None
    streak_days: int = 0


@dataclass
class TopicProgress:
    """Progress for a specific topic."""

    topic_code: str
    topic_name: str
    mastery_level: float
    questions_answered: int = 0
    accuracy: float | None = None
    last_practiced_at: datetime | None = None


@dataclass
class WeakArea:
    """Identified weak area needing attention."""

    topic_code: str
    topic_name: str
    mastery_level: float
    recommended_action: str


@dataclass
class RecentActivity:
    """Recent learning activity entry."""

    date: datetime
    activity_type: str
    topic_name: str | None = None
    duration_minutes: int = 0
    score: float | None = None


@dataclass
class StudentDashboard:
    """Complete student dashboard data."""

    student_id: str
    stats: DashboardStats
    topic_progress: list[TopicProgress] = field(default_factory=list)
    weak_areas: list[WeakArea] = field(default_factory=list)
    recent_activity: list[RecentActivity] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "student_id": self.student_id,
            "stats": {
                "total_sessions": self.stats.total_sessions,
                "total_time_minutes": self.stats.total_time_minutes,
                "questions_answered": self.stats.questions_answered,
                "questions_correct": self.stats.questions_correct,
                "average_accuracy": self.stats.average_accuracy,
                "streak_days": self.stats.streak_days,
            },
            "topic_progress": [
                {
                    "topic_code": tp.topic_code,
                    "topic_name": tp.topic_name,
                    "mastery_level": tp.mastery_level,
                    "questions_answered": tp.questions_answered,
                    "accuracy": tp.accuracy,
                    "last_practiced_at": tp.last_practiced_at.isoformat() if tp.last_practiced_at else None,
                }
                for tp in self.topic_progress
            ],
            "weak_areas": [
                {
                    "topic_code": wa.topic_code,
                    "topic_name": wa.topic_name,
                    "mastery_level": wa.mastery_level,
                    "recommended_action": wa.recommended_action,
                }
                for wa in self.weak_areas
            ],
            "recent_activity": [
                {
                    "date": ra.date.isoformat(),
                    "activity_type": ra.activity_type,
                    "topic_name": ra.topic_name,
                    "duration_minutes": ra.duration_minutes,
                    "score": ra.score,
                }
                for ra in self.recent_activity
            ],
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class ProgressReport:
    """Detailed progress report for a student."""

    student_id: str
    student_name: str | None
    period_start: datetime
    period_end: datetime
    stats: DashboardStats
    topic_progress: list[TopicProgress] = field(default_factory=list)
    growth_percentage: float | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "stats": {
                "total_sessions": self.stats.total_sessions,
                "total_time_minutes": self.stats.total_time_minutes,
                "questions_answered": self.stats.questions_answered,
                "questions_correct": self.stats.questions_correct,
                "average_accuracy": self.stats.average_accuracy,
                "streak_days": self.stats.streak_days,
            },
            "topic_progress": [
                {
                    "topic_code": tp.topic_code,
                    "topic_name": tp.topic_name,
                    "mastery_level": tp.mastery_level,
                    "questions_answered": tp.questions_answered,
                    "accuracy": tp.accuracy,
                    "last_practiced_at": tp.last_practiced_at.isoformat() if tp.last_practiced_at else None,
                }
                for tp in self.topic_progress
            ],
            "growth_percentage": self.growth_percentage,
            "recommendations": self.recommendations,
        }


@dataclass
class StudentSummary:
    """Summary for a student in class view."""

    student_id: str
    student_name: str
    total_sessions: int = 0
    average_accuracy: float | None = None
    total_time_minutes: int = 0
    last_active_at: datetime | None = None


@dataclass
class ClassAnalytics:
    """Class-level analytics for teachers."""

    class_id: str
    class_name: str
    total_students: int = 0
    active_students: int = 0
    average_mastery: float = 0.0
    topic_performance: list[dict[str, Any]] = field(default_factory=list)
    student_summaries: list[StudentSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "total_students": self.total_students,
            "active_students": self.active_students,
            "average_mastery": self.average_mastery,
            "topic_performance": self.topic_performance,
            "student_summaries": [
                {
                    "student_id": ss.student_id,
                    "student_name": ss.student_name,
                    "total_sessions": ss.total_sessions,
                    "average_accuracy": ss.average_accuracy,
                    "total_time_minutes": ss.total_time_minutes,
                    "last_active_at": ss.last_active_at.isoformat() if ss.last_active_at else None,
                }
                for ss in self.student_summaries
            ],
        }


class AnalyticsService:
    """Service for fetching analytics dashboards and reports.

    This service queries aggregated analytics data to provide
    dashboards for students, progress reports, and class-level
    analytics for teachers.

    Attributes:
        _db: Database session.
        tenant_code: Tenant code for multi-tenancy.
    """

    # Mastery threshold for identifying weak areas
    WEAK_AREA_THRESHOLD = 0.4

    def __init__(
        self,
        db: AsyncSession,
        tenant_code: str,
    ) -> None:
        """Initialize the analytics service.

        Args:
            db: Database session.
            tenant_code: Tenant code for multi-tenancy.
        """
        self._db = db
        self.tenant_code = tenant_code

    async def get_dashboard(
        self,
        student_id: str | UUID,
        period_days: int = 7,
    ) -> StudentDashboard:
        """Get student dashboard data.

        Fetches aggregated statistics, topic progress, weak areas,
        and recent activity for the student.

        Args:
            student_id: Student identifier.
            period_days: Number of days for statistics.

        Returns:
            StudentDashboard with all dashboard data.

        Example:
            >>> service = AnalyticsService(db, "acme")
            >>> dashboard = await service.get_dashboard(student_uuid)
            >>> print(f"Streak: {dashboard.stats.streak_days} days")
        """
        student_id_str = str(student_id)

        # Calculate date range
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=period_days)

        # Get dashboard stats from daily summaries
        stats = await self._get_stats(student_id_str, start_date, end_date)

        # Get topic progress from mastery data
        topic_progress = await self._get_topic_progress(student_id_str)

        # Identify weak areas
        weak_areas = self._identify_weak_areas(topic_progress)

        # Get recent activity
        recent_activity = await self._get_recent_activity(
            student_id_str,
            limit=10,
        )

        return StudentDashboard(
            student_id=student_id_str,
            stats=stats,
            topic_progress=topic_progress,
            weak_areas=weak_areas,
            recent_activity=recent_activity,
        )

    async def get_progress_report(
        self,
        student_id: str | UUID,
        period_days: int = 30,
    ) -> ProgressReport:
        """Get detailed progress report for a student.

        Provides comprehensive progress data including growth
        comparison with previous period and recommendations.

        Args:
            student_id: Student identifier.
            period_days: Number of days for report.

        Returns:
            ProgressReport with detailed progress data.

        Example:
            >>> report = await service.get_progress_report(
            ...     student_uuid,
            ...     period_days=30,
            ... )
            >>> if report.growth_percentage:
            ...     print(f"Growth: {report.growth_percentage}%")
        """
        student_id_str = str(student_id)

        # Get student name
        student_name = await self._get_student_name(student_id_str)

        # Calculate date range
        now = datetime.now(timezone.utc)
        period_end = now
        period_start = now - timedelta(days=period_days)

        # Get current period stats
        stats = await self._get_stats(
            student_id_str,
            period_start.date(),
            period_end.date(),
        )

        # Get topic progress
        topic_progress = await self._get_topic_progress(student_id_str)

        # Calculate growth by comparing with previous period
        growth_percentage = await self._calculate_growth(
            student_id_str,
            period_start.date(),
            period_end.date(),
        )

        # Generate recommendations based on weak areas and activity
        weak_areas = self._identify_weak_areas(topic_progress)
        recommendations = self._generate_recommendations(
            stats=stats,
            weak_areas=weak_areas,
            topic_progress=topic_progress,
        )

        return ProgressReport(
            student_id=student_id_str,
            student_name=student_name,
            period_start=period_start,
            period_end=period_end,
            stats=stats,
            topic_progress=topic_progress,
            growth_percentage=growth_percentage,
            recommendations=recommendations,
        )

    async def get_class_analytics(
        self,
        class_id: str | UUID,
        period_days: int = 30,
    ) -> ClassAnalytics:
        """Get class-level analytics for teachers.

        Provides aggregated analytics for all students in a class
        including individual summaries and topic performance.

        Args:
            class_id: Class identifier.
            period_days: Number of days for analytics.

        Returns:
            ClassAnalytics with class-level data.

        Example:
            >>> class_data = await service.get_class_analytics(
            ...     class_uuid,
            ...     period_days=30,
            ... )
            >>> print(f"Active: {class_data.active_students}/{class_data.total_students}")
        """
        class_id_str = str(class_id)

        # Calculate date range
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=period_days)

        # Get class info and students
        class_name, student_ids = await self._get_class_info(class_id_str)

        if not student_ids:
            return ClassAnalytics(
                class_id=class_id_str,
                class_name=class_name or "Unknown Class",
            )

        # Get student summaries
        student_summaries: list[StudentSummary] = []
        active_count = 0
        mastery_values: list[float] = []

        for student_id in student_ids:
            summary = await self._get_student_summary(
                student_id,
                start_date,
                end_date,
            )
            if summary:
                student_summaries.append(summary)
                if summary.total_sessions > 0:
                    active_count += 1

            # Get mastery for averaging
            mastery = await self._get_overall_mastery(student_id)
            if mastery is not None:
                mastery_values.append(mastery)

        # Calculate average mastery
        average_mastery = (
            sum(mastery_values) / len(mastery_values)
            if mastery_values
            else 0.0
        )

        # Get topic performance
        topic_performance = await self._get_class_topic_performance(
            student_ids,
            start_date,
            end_date,
        )

        return ClassAnalytics(
            class_id=class_id_str,
            class_name=class_name or "Class",
            total_students=len(student_ids),
            active_students=active_count,
            average_mastery=round(average_mastery, 2),
            topic_performance=topic_performance,
            student_summaries=student_summaries,
        )

    async def _get_stats(
        self,
        student_id: str,
        start_date: date,
        end_date: date,
    ) -> DashboardStats:
        """Get aggregated stats from daily summaries.

        Args:
            student_id: Student identifier.
            start_date: Start date.
            end_date: End date.

        Returns:
            DashboardStats with aggregated values.
        """
        stmt = select(DailySummary).where(
            and_(
                DailySummary.student_id == student_id,
                DailySummary.summary_date >= start_date,
                DailySummary.summary_date <= end_date,
            )
        ).order_by(desc(DailySummary.summary_date))

        result = await self._db.execute(stmt)
        summaries = result.scalars().all()

        if not summaries:
            return DashboardStats()

        total_sessions = sum(s.sessions_count for s in summaries)
        total_time_seconds = sum(s.total_time_seconds for s in summaries)
        questions_answered = sum(s.questions_attempted for s in summaries)
        questions_correct = sum(s.questions_correct for s in summaries)

        average_accuracy = None
        if questions_answered > 0:
            average_accuracy = round(
                (questions_correct / questions_answered) * 100, 1
            )

        # Get current streak from most recent summary
        streak_days = summaries[0].daily_streak if summaries else 0

        return DashboardStats(
            total_sessions=total_sessions,
            total_time_minutes=total_time_seconds // 60,
            questions_answered=questions_answered,
            questions_correct=questions_correct,
            average_accuracy=average_accuracy,
            streak_days=streak_days,
        )

    async def _get_topic_progress(
        self,
        student_id: str,
    ) -> list[TopicProgress]:
        """Get topic progress from mastery data.

        Args:
            student_id: Student identifier.

        Returns:
            List of TopicProgress for each topic.
        """
        # Get mastery from SemanticMemory
        # Join Topic using entity_full_code by concatenating Topic's composite key
        topic_full_code_expr = func.concat(
            Topic.framework_code, ".",
            Topic.subject_code, ".",
            Topic.grade_code, ".",
            Topic.unit_code, ".",
            Topic.code,
        )

        stmt = (
            select(SemanticMemory, Topic)
            .join(Topic, SemanticMemory.entity_full_code == topic_full_code_expr, isouter=True)
            .where(
                and_(
                    SemanticMemory.student_id == student_id,
                    SemanticMemory.entity_type == "topic",
                )
            )
        )

        result = await self._db.execute(stmt)
        rows = result.all()

        progress_list: list[TopicProgress] = []

        for memory, topic in rows:
            topic_name = topic.name if topic else f"Topic {memory.entity_full_code}"

            # Use entity_full_code as topic_code (it contains full composite key)
            topic_code = memory.entity_full_code

            # Skip entries without topic_code (orphan semantic memories)
            if not topic_code:
                continue

            progress_list.append(
                TopicProgress(
                    topic_code=topic_code,
                    topic_name=topic_name,
                    mastery_level=float(memory.mastery_level),
                    questions_answered=0,  # Would need event aggregation
                    accuracy=None,
                    last_practiced_at=memory.updated_at,
                )
            )

        return progress_list

    def _identify_weak_areas(
        self,
        topic_progress: list[TopicProgress],
    ) -> list[WeakArea]:
        """Identify weak areas from topic progress.

        Args:
            topic_progress: List of topic progress.

        Returns:
            List of weak areas needing attention.
        """
        weak_areas: list[WeakArea] = []

        for tp in topic_progress:
            if tp.mastery_level < self.WEAK_AREA_THRESHOLD:
                # Generate recommendation based on mastery level
                if tp.mastery_level < 0.2:
                    action = "Start with foundational concepts"
                elif tp.mastery_level < 0.3:
                    action = "Review basic principles"
                else:
                    action = "Practice more exercises"

                weak_areas.append(
                    WeakArea(
                        topic_code=tp.topic_code,
                        topic_name=tp.topic_name,
                        mastery_level=tp.mastery_level,
                        recommended_action=action,
                    )
                )

        # Sort by mastery level (lowest first)
        weak_areas.sort(key=lambda x: x.mastery_level)

        return weak_areas[:5]  # Return top 5 weak areas

    async def _get_recent_activity(
        self,
        student_id: str,
        limit: int = 10,
    ) -> list[RecentActivity]:
        """Get recent activity from analytics events.

        Args:
            student_id: Student identifier.
            limit: Maximum number of activities.

        Returns:
            List of recent activities.
        """
        stmt = (
            select(AnalyticsEvent)
            .where(AnalyticsEvent.student_id == student_id)
            .order_by(desc(AnalyticsEvent.occurred_at))
            .limit(limit)
        )

        result = await self._db.execute(stmt)
        events = result.scalars().all()

        activities: list[RecentActivity] = []

        for event in events:
            event_data = event.data or {}

            # Determine activity type from event
            activity_type = self._map_event_to_activity_type(event.event_type)

            # Get topic name if available
            topic_name = event_data.get("topic_name")

            # Get duration if available
            duration_minutes = event_data.get("duration_minutes", 0)

            # Get score if available
            score = event_data.get("score")

            activities.append(
                RecentActivity(
                    date=event.occurred_at,
                    activity_type=activity_type,
                    topic_name=topic_name,
                    duration_minutes=duration_minutes,
                    score=score,
                )
            )

        return activities

    def _map_event_to_activity_type(self, event_type: str) -> str:
        """Map event type to user-friendly activity type.

        Args:
            event_type: Event type string.

        Returns:
            User-friendly activity type.
        """
        mapping = {
            "practice.session.started": "practice_started",
            "practice.session.completed": "practice_completed",
            "student.answer.evaluated": "answered_question",
            "conversation.started": "conversation_started",
            "conversation.ended": "conversation_completed",
            "student.concept.mastered": "mastered_concept",
        }

        return mapping.get(event_type, event_type.split(".")[-1])

    async def _calculate_growth(
        self,
        student_id: str,
        period_start: date,
        period_end: date,
    ) -> float | None:
        """Calculate growth by comparing with previous period.

        Args:
            student_id: Student identifier.
            period_start: Current period start.
            period_end: Current period end.

        Returns:
            Growth percentage or None if not enough data.
        """
        period_length = (period_end - period_start).days

        # Previous period dates
        prev_end = period_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=period_length)

        # Get mastery snapshots for comparison
        current_snapshot = await self._get_mastery_snapshot_near_date(
            student_id,
            period_end,
        )
        previous_snapshot = await self._get_mastery_snapshot_near_date(
            student_id,
            prev_end,
        )

        if not current_snapshot or not previous_snapshot:
            return None

        current_mastery = float(current_snapshot.overall_mastery or 0)
        previous_mastery = float(previous_snapshot.overall_mastery or 0)

        if previous_mastery == 0:
            return None

        growth = ((current_mastery - previous_mastery) / previous_mastery) * 100
        return round(growth, 1)

    async def _get_mastery_snapshot_near_date(
        self,
        student_id: str,
        target_date: date,
    ) -> MasterySnapshot | None:
        """Get mastery snapshot closest to target date.

        Args:
            student_id: Student identifier.
            target_date: Target date.

        Returns:
            MasterySnapshot or None if not found.
        """
        stmt = (
            select(MasterySnapshot)
            .where(
                and_(
                    MasterySnapshot.student_id == student_id,
                    MasterySnapshot.snapshot_date <= target_date,
                )
            )
            .order_by(desc(MasterySnapshot.snapshot_date))
            .limit(1)
        )

        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    def _generate_recommendations(
        self,
        stats: DashboardStats,
        weak_areas: list[WeakArea],
        topic_progress: list[TopicProgress],
    ) -> list[str]:
        """Generate personalized recommendations.

        Args:
            stats: Dashboard statistics.
            weak_areas: Identified weak areas.
            topic_progress: Topic progress list.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        # Low activity recommendation
        if stats.total_sessions < 3:
            recommendations.append(
                "Increase practice frequency to at least 3 sessions per week"
            )

        # Low accuracy recommendation
        if stats.average_accuracy and stats.average_accuracy < 60:
            recommendations.append(
                "Focus on understanding concepts before attempting more questions"
            )

        # Weak area recommendations
        if weak_areas:
            topic_names = [wa.topic_name for wa in weak_areas[:2]]
            recommendations.append(
                f"Priority focus areas: {', '.join(topic_names)}"
            )

        # Streak recommendation
        if stats.streak_days == 0:
            recommendations.append(
                "Build a daily practice habit for consistent progress"
            )
        elif stats.streak_days >= 7:
            recommendations.append(
                f"Great streak! Keep up the momentum at {stats.streak_days} days"
            )

        # General progress recommendation
        if topic_progress:
            avg_mastery = sum(tp.mastery_level for tp in topic_progress) / len(topic_progress)
            if avg_mastery > 0.7:
                recommendations.append(
                    "Consider challenging yourself with harder topics"
                )

        return recommendations[:5]

    async def _get_student_name(self, student_id: str) -> str | None:
        """Get student name from user table.

        Args:
            student_id: Student identifier.

        Returns:
            Student name or None.
        """
        stmt = select(User.first_name, User.last_name).where(User.id == student_id)
        result = await self._db.execute(stmt)
        row = result.one_or_none()

        if row:
            first_name, last_name = row
            if first_name and last_name:
                return f"{first_name} {last_name}"
            return first_name or last_name

        return None

    async def _get_class_info(
        self,
        class_id: str,
    ) -> tuple[str | None, list[str]]:
        """Get class name and student IDs.

        Args:
            class_id: Class identifier.

        Returns:
            Tuple of (class_name, list of student_ids).
        """
        # Query class and its students
        # Note: This assumes a class_memberships or similar table exists
        # For now, return empty until class infrastructure is implemented
        # The API can still be used with direct student queries

        logger.debug("Class info lookup not yet implemented: %s", class_id)
        return None, []

    async def _get_student_summary(
        self,
        student_id: str,
        start_date: date,
        end_date: date,
    ) -> StudentSummary | None:
        """Get summary for a single student.

        Args:
            student_id: Student identifier.
            start_date: Start date.
            end_date: End date.

        Returns:
            StudentSummary or None.
        """
        # Get student name
        student_name = await self._get_student_name(student_id)

        # Get stats for period
        stats = await self._get_stats(student_id, start_date, end_date)

        # Get last activity
        stmt = (
            select(DailySummary.summary_date)
            .where(DailySummary.student_id == student_id)
            .order_by(desc(DailySummary.summary_date))
            .limit(1)
        )
        result = await self._db.execute(stmt)
        row = result.scalar_one_or_none()

        last_active_at = None
        if row:
            last_active_at = datetime.combine(row, datetime.min.time())

        return StudentSummary(
            student_id=student_id,
            student_name=student_name or "Unknown",
            total_sessions=stats.total_sessions,
            average_accuracy=stats.average_accuracy,
            total_time_minutes=stats.total_time_minutes,
            last_active_at=last_active_at,
        )

    async def _get_overall_mastery(self, student_id: str) -> float | None:
        """Get overall mastery for a student.

        Args:
            student_id: Student identifier.

        Returns:
            Overall mastery value or None.
        """
        stmt = (
            select(func.avg(SemanticMemory.mastery_level))
            .where(SemanticMemory.student_id == student_id)
        )
        result = await self._db.execute(stmt)
        avg = result.scalar_one_or_none()

        if avg is not None:
            return float(avg)
        return None

    async def _get_class_topic_performance(
        self,
        student_ids: list[str],
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Get topic performance across class.

        Args:
            student_ids: List of student identifiers.
            start_date: Start date.
            end_date: End date.

        Returns:
            List of topic performance dictionaries.
        """
        if not student_ids:
            return []

        # Get aggregated mastery per topic across all students
        stmt = (
            select(
                SemanticMemory.entity_full_code,
                func.avg(SemanticMemory.mastery_level).label("avg_mastery"),
                func.count(SemanticMemory.student_id).label("student_count"),
            )
            .where(
                and_(
                    SemanticMemory.student_id.in_(student_ids),
                    SemanticMemory.entity_type == "topic",
                )
            )
            .group_by(SemanticMemory.entity_full_code)
        )

        result = await self._db.execute(stmt)
        rows = result.all()

        performance: list[dict[str, Any]] = []

        for entity_full_code, avg_mastery, student_count in rows:
            # Parse entity_full_code to get topic composite keys
            parts = entity_full_code.split(".") if entity_full_code else []
            if len(parts) != 5:
                # Skip orphan semantic memories without valid full code
                continue

            # Get topic name using composite key
            topic_stmt = (
                select(Topic.name)
                .where(
                    Topic.framework_code == parts[0],
                    Topic.subject_code == parts[1],
                    Topic.grade_code == parts[2],
                    Topic.unit_code == parts[3],
                    Topic.code == parts[4],
                )
            )
            topic_result = await self._db.execute(topic_stmt)
            topic_name = topic_result.scalar_one_or_none()

            if not topic_name:
                # Skip orphan semantic memories without valid topic
                continue

            # Count struggling students (mastery < threshold)
            struggling_stmt = (
                select(func.count(SemanticMemory.student_id))
                .where(
                    and_(
                        SemanticMemory.entity_full_code == entity_full_code,
                        SemanticMemory.student_id.in_(student_ids),
                        SemanticMemory.mastery_level < Decimal("0.4"),
                    )
                )
            )
            struggling_result = await self._db.execute(struggling_stmt)
            struggling_count = struggling_result.scalar_one_or_none() or 0

            performance.append({
                "topic_code": entity_full_code,
                "topic_name": topic_name,
                "average_mastery": round(float(avg_mastery), 2),
                "students_engaged": student_count,
                "students_struggling": struggling_count,
            })

        # Sort by average mastery (lowest first for attention)
        performance.sort(key=lambda x: x["average_mastery"])

        return performance
