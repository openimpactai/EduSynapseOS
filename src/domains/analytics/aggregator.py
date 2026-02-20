# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics data aggregation module.

This module provides aggregation capabilities for analytics data:
- Daily aggregation: Creates/updates DailySummary records
- Weekly aggregation: Aggregates daily summaries into weekly view
- Mastery snapshots: Creates point-in-time mastery records

The AnalyticsAggregator can either:
1. Trigger background tasks for async aggregation
2. Perform direct aggregation for immediate results

Usage:
    from src.domains.analytics import AnalyticsAggregator

    aggregator = AnalyticsAggregator(tenant_code="acme")

    # Trigger async aggregation (via Dramatiq)
    await aggregator.aggregate_daily_async()
    await aggregator.create_mastery_snapshot_async()

    # Direct aggregation (blocking)
    result = await aggregator.aggregate_daily(db, date=yesterday)
    weekly = await aggregator.aggregate_weekly(db, week_start=start_date)
"""

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.analytics import (
    AnalyticsEvent,
    DailySummary,
    EngagementMetric,
    MasterySnapshot,
)
from src.infrastructure.database.models.tenant.memory import SemanticMemory

logger = logging.getLogger(__name__)


class AggregationResult:
    """Result of an aggregation operation.

    Attributes:
        success: Whether aggregation succeeded.
        date: Target date for aggregation.
        records_created: Number of records created.
        records_updated: Number of records updated.
        students_processed: Number of students processed.
        error: Error message if failed.
    """

    def __init__(
        self,
        success: bool,
        target_date: date | None = None,
        records_created: int = 0,
        records_updated: int = 0,
        students_processed: int = 0,
        error: str | None = None,
    ) -> None:
        """Initialize aggregation result.

        Args:
            success: Whether aggregation succeeded.
            target_date: Target date for aggregation.
            records_created: Number of records created.
            records_updated: Number of records updated.
            students_processed: Number of students processed.
            error: Error message if failed.
        """
        self.success = success
        self.date = target_date
        self.records_created = records_created
        self.records_updated = records_updated
        self.students_processed = students_processed
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "date": str(self.date) if self.date else None,
            "records_created": self.records_created,
            "records_updated": self.records_updated,
            "students_processed": self.students_processed,
            "error": self.error,
        }


class WeeklyAggregation:
    """Weekly aggregated analytics data.

    Attributes:
        student_id: Student identifier.
        week_start: Start date of the week.
        week_end: End date of the week.
        total_sessions: Total sessions for the week.
        total_time_seconds: Total learning time.
        questions_attempted: Total questions attempted.
        questions_correct: Total correct answers.
        messages_sent: Total messages sent.
        topics_practiced: Unique topics practiced.
        average_score: Average score for the week.
        daily_streak_end: Streak at end of week.
        days_active: Number of active days.
    """

    def __init__(
        self,
        student_id: str,
        week_start: date,
        week_end: date,
        total_sessions: int = 0,
        total_time_seconds: int = 0,
        questions_attempted: int = 0,
        questions_correct: int = 0,
        messages_sent: int = 0,
        topics_practiced: list[str] | None = None,
        average_score: float | None = None,
        daily_streak_end: int = 0,
        days_active: int = 0,
    ) -> None:
        """Initialize weekly aggregation.

        Args:
            student_id: Student identifier.
            week_start: Start date of the week.
            week_end: End date of the week.
            total_sessions: Total sessions.
            total_time_seconds: Total learning time.
            questions_attempted: Total questions.
            questions_correct: Correct answers.
            messages_sent: Messages sent.
            topics_practiced: Topics practiced.
            average_score: Average score.
            daily_streak_end: Streak at week end.
            days_active: Number of active days.
        """
        self.student_id = student_id
        self.week_start = week_start
        self.week_end = week_end
        self.total_sessions = total_sessions
        self.total_time_seconds = total_time_seconds
        self.questions_attempted = questions_attempted
        self.questions_correct = questions_correct
        self.messages_sent = messages_sent
        self.topics_practiced = topics_practiced or []
        self.average_score = average_score
        self.daily_streak_end = daily_streak_end
        self.days_active = days_active

    @property
    def accuracy(self) -> float | None:
        """Calculate accuracy percentage."""
        if self.questions_attempted == 0:
            return None
        return (self.questions_correct / self.questions_attempted) * 100

    @property
    def total_time_hours(self) -> float:
        """Get total time in hours."""
        return self.total_time_seconds / 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "student_id": self.student_id,
            "week_start": str(self.week_start),
            "week_end": str(self.week_end),
            "total_sessions": self.total_sessions,
            "total_time_seconds": self.total_time_seconds,
            "total_time_hours": self.total_time_hours,
            "questions_attempted": self.questions_attempted,
            "questions_correct": self.questions_correct,
            "accuracy": self.accuracy,
            "messages_sent": self.messages_sent,
            "topics_practiced": self.topics_practiced,
            "average_score": self.average_score,
            "daily_streak_end": self.daily_streak_end,
            "days_active": self.days_active,
        }


class AnalyticsAggregator:
    """Service for aggregating analytics data.

    This class provides methods for aggregating analytics events
    into summaries and snapshots. It supports both async (via
    background tasks) and sync (direct) aggregation.

    Attributes:
        tenant_code: Tenant code for multi-tenancy.
    """

    def __init__(self, tenant_code: str) -> None:
        """Initialize the aggregator.

        Args:
            tenant_code: Tenant code for multi-tenancy.
        """
        self.tenant_code = tenant_code

    async def aggregate_daily_async(
        self,
        target_date: date | None = None,
    ) -> None:
        """Trigger async daily aggregation via Dramatiq.

        Queues a background task to aggregate daily analytics.
        The task runs asynchronously and updates DailySummary records.

        Args:
            target_date: Date to aggregate (defaults to yesterday).

        Example:
            >>> aggregator = AnalyticsAggregator(tenant_code="acme")
            >>> await aggregator.aggregate_daily_async()
        """
        from src.infrastructure.background.tasks import aggregate_daily_analytics

        date_str = str(target_date) if target_date else None

        aggregate_daily_analytics.send(
            tenant_code=self.tenant_code,
            date_str=date_str,
        )

        logger.info(
            "Queued daily analytics aggregation: tenant=%s, date=%s",
            self.tenant_code,
            date_str or "yesterday",
        )

    async def create_mastery_snapshot_async(self) -> None:
        """Trigger async mastery snapshot creation via Dramatiq.

        Queues a background task to create mastery snapshots
        for all students from their SemanticMemory records.

        Example:
            >>> aggregator = AnalyticsAggregator(tenant_code="acme")
            >>> await aggregator.create_mastery_snapshot_async()
        """
        from src.infrastructure.background.tasks import create_mastery_snapshots

        create_mastery_snapshots.send(tenant_code=self.tenant_code)

        logger.info(
            "Queued mastery snapshot creation: tenant=%s",
            self.tenant_code,
        )

    async def aggregate_daily(
        self,
        db: AsyncSession,
        target_date: date | None = None,
    ) -> AggregationResult:
        """Aggregate daily analytics directly (blocking).

        Creates or updates DailySummary records for all students
        with events on the specified date. This method runs
        synchronously and blocks until completion.

        Args:
            db: Database session.
            target_date: Date to aggregate (defaults to yesterday).

        Returns:
            AggregationResult with operation details.

        Example:
            >>> async with db.begin():
            ...     result = await aggregator.aggregate_daily(db)
            ...     print(f"Processed {result.students_processed} students")
        """
        if target_date is None:
            target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

        start_of_day = datetime.combine(target_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_of_day = datetime.combine(target_date, datetime.max.time()).replace(
            tzinfo=timezone.utc
        )

        try:
            # Get unique students with events on this date
            stmt = (
                select(AnalyticsEvent.student_id)
                .where(
                    and_(
                        AnalyticsEvent.occurred_at >= start_of_day,
                        AnalyticsEvent.occurred_at <= end_of_day,
                        AnalyticsEvent.student_id.isnot(None),
                    )
                )
                .distinct()
            )

            result = await db.execute(stmt)
            student_ids = [row[0] for row in result.fetchall()]

            records_created = 0
            records_updated = 0

            for student_id in student_ids:
                created, updated = await self._aggregate_student_daily(
                    db=db,
                    student_id=student_id,
                    target_date=target_date,
                    start_of_day=start_of_day,
                    end_of_day=end_of_day,
                )
                records_created += created
                records_updated += updated

            logger.info(
                "Daily aggregation complete: date=%s, students=%d, created=%d, updated=%d",
                target_date,
                len(student_ids),
                records_created,
                records_updated,
            )

            return AggregationResult(
                success=True,
                target_date=target_date,
                records_created=records_created,
                records_updated=records_updated,
                students_processed=len(student_ids),
            )

        except Exception as e:
            logger.error(
                "Daily aggregation failed: date=%s, error=%s",
                target_date,
                str(e),
                exc_info=True,
            )
            return AggregationResult(
                success=False,
                target_date=target_date,
                error=str(e),
            )

    async def _aggregate_student_daily(
        self,
        db: AsyncSession,
        student_id: str,
        target_date: date,
        start_of_day: datetime,
        end_of_day: datetime,
    ) -> tuple[int, int]:
        """Aggregate daily data for a single student.

        Args:
            db: Database session.
            student_id: Student identifier.
            target_date: Target date.
            start_of_day: Start of day datetime.
            end_of_day: End of day datetime.

        Returns:
            Tuple of (created_count, updated_count).
        """
        # Get all events for this student on this date
        events_stmt = select(AnalyticsEvent).where(
            and_(
                AnalyticsEvent.student_id == student_id,
                AnalyticsEvent.occurred_at >= start_of_day,
                AnalyticsEvent.occurred_at <= end_of_day,
            )
        )
        events_result = await db.execute(events_stmt)
        events = events_result.scalars().all()

        # Calculate metrics from events
        sessions = set()
        questions_attempted = 0
        questions_correct = 0
        messages_sent = 0
        topics_practiced: set[str] = set()

        for event in events:
            event_data = event.data or {}

            if event.session_id:
                sessions.add(event.session_id)

            category = event_data.get("analytics_category", "")

            if category == "performance":
                questions_attempted += 1
                if event_data.get("is_correct"):
                    questions_correct += 1

            if category == "interaction":
                if "message" in event.event_type.lower():
                    messages_sent += 1

            if event.topic_full_code:
                topics_practiced.add(event.topic_full_code)

        # Check for existing summary
        existing_stmt = select(DailySummary).where(
            and_(
                DailySummary.student_id == student_id,
                DailySummary.summary_date == target_date,
            )
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            # Update existing record
            existing.sessions_count = len(sessions)
            existing.questions_attempted = questions_attempted
            existing.questions_correct = questions_correct
            existing.messages_sent = messages_sent
            existing.topics_practiced = list(topics_practiced)
            if questions_attempted > 0:
                existing.average_score = Decimal(
                    str(round(questions_correct / questions_attempted * 100, 2))
                )
            return 0, 1
        else:
            # Create new summary
            summary = DailySummary(
                id=str(uuid4()),
                student_id=student_id,
                summary_date=target_date,
                total_time_seconds=0,  # Would need session duration tracking
                sessions_count=len(sessions),
                questions_attempted=questions_attempted,
                questions_correct=questions_correct,
                messages_sent=messages_sent,
                topics_practiced=list(topics_practiced),
                average_score=(
                    Decimal(str(round(questions_correct / questions_attempted * 100, 2)))
                    if questions_attempted > 0
                    else None
                ),
            )
            db.add(summary)
            return 1, 0

    async def aggregate_weekly(
        self,
        db: AsyncSession,
        student_id: str | UUID,
        week_start: date | None = None,
    ) -> WeeklyAggregation | None:
        """Aggregate weekly analytics from daily summaries.

        Aggregates DailySummary records for a week into a
        WeeklyAggregation. This is a read-only aggregation
        that doesn't persist to the database.

        Args:
            db: Database session.
            student_id: Student identifier.
            week_start: Start of week (Monday). Defaults to current week.

        Returns:
            WeeklyAggregation or None if no data.

        Example:
            >>> weekly = await aggregator.aggregate_weekly(
            ...     db=db,
            ...     student_id=student_uuid,
            ... )
            >>> if weekly:
            ...     print(f"Active {weekly.days_active} days, accuracy: {weekly.accuracy}%")
        """
        if week_start is None:
            today = datetime.now(timezone.utc).date()
            # Get Monday of current week
            week_start = today - timedelta(days=today.weekday())

        week_end = week_start + timedelta(days=6)

        # Get daily summaries for the week
        stmt = select(DailySummary).where(
            and_(
                DailySummary.student_id == str(student_id),
                DailySummary.summary_date >= week_start,
                DailySummary.summary_date <= week_end,
            )
        ).order_by(DailySummary.summary_date)

        result = await db.execute(stmt)
        summaries = result.scalars().all()

        if not summaries:
            return None

        # Aggregate data from daily summaries
        total_sessions = 0
        total_time_seconds = 0
        questions_attempted = 0
        questions_correct = 0
        messages_sent = 0
        topics: set[str] = set()
        scores: list[float] = []
        last_streak = 0

        for summary in summaries:
            total_sessions += summary.sessions_count
            total_time_seconds += summary.total_time_seconds
            questions_attempted += summary.questions_attempted
            questions_correct += summary.questions_correct
            messages_sent += summary.messages_sent

            if summary.topics_practiced:
                for topic in summary.topics_practiced:
                    topics.add(str(topic))

            if summary.average_score is not None:
                scores.append(float(summary.average_score))

            last_streak = summary.daily_streak

        average_score = sum(scores) / len(scores) if scores else None

        return WeeklyAggregation(
            student_id=str(student_id),
            week_start=week_start,
            week_end=week_end,
            total_sessions=total_sessions,
            total_time_seconds=total_time_seconds,
            questions_attempted=questions_attempted,
            questions_correct=questions_correct,
            messages_sent=messages_sent,
            topics_practiced=list(topics),
            average_score=average_score,
            daily_streak_end=last_streak,
            days_active=len(summaries),
        )

    async def create_mastery_snapshot(
        self,
        db: AsyncSession,
        student_id: str | UUID,
        snapshot_date: date | None = None,
    ) -> MasterySnapshot | None:
        """Create mastery snapshot for a student.

        Creates a point-in-time snapshot of the student's mastery
        levels from their SemanticMemory records.

        Args:
            db: Database session.
            student_id: Student identifier.
            snapshot_date: Date for snapshot (defaults to today).

        Returns:
            Created MasterySnapshot or None if already exists/no data.

        Example:
            >>> snapshot = await aggregator.create_mastery_snapshot(
            ...     db=db,
            ...     student_id=student_uuid,
            ... )
            >>> if snapshot:
            ...     print(f"Overall mastery: {snapshot.overall_mastery}")
        """
        if snapshot_date is None:
            snapshot_date = datetime.now(timezone.utc).date()

        student_id_str = str(student_id)

        # Check if snapshot already exists
        existing_stmt = select(MasterySnapshot).where(
            and_(
                MasterySnapshot.student_id == student_id_str,
                MasterySnapshot.snapshot_date == snapshot_date,
            )
        )
        existing_result = await db.execute(existing_stmt)
        if existing_result.scalar_one_or_none():
            logger.debug(
                "Mastery snapshot already exists: student=%s, date=%s",
                student_id,
                snapshot_date,
            )
            return None

        # Get current mastery levels from SemanticMemory
        mastery_stmt = select(SemanticMemory).where(
            SemanticMemory.student_id == student_id_str
        )
        mastery_result = await db.execute(mastery_stmt)
        memories = mastery_result.scalars().all()

        if not memories:
            return None

        # Build mastery dictionaries
        subject_mastery: dict[str, float] = {}
        topic_mastery: dict[str, float] = {}

        for memory in memories:
            entity_full_code = memory.entity_full_code
            mastery_level = float(memory.mastery_level)

            if memory.entity_type == "subject":
                subject_mastery[entity_full_code] = mastery_level
            elif memory.entity_type == "topic":
                topic_mastery[entity_full_code] = mastery_level

        # Calculate overall mastery
        all_levels = list(subject_mastery.values()) + list(topic_mastery.values())
        overall_mastery = sum(all_levels) / len(all_levels) if all_levels else 0.0

        # Create snapshot
        snapshot = MasterySnapshot(
            id=str(uuid4()),
            student_id=student_id_str,
            snapshot_date=snapshot_date,
            subject_mastery=subject_mastery,
            topic_mastery=topic_mastery,
            overall_mastery=Decimal(str(round(overall_mastery, 2))),
            created_at=datetime.now(timezone.utc),
        )

        db.add(snapshot)

        logger.info(
            "Created mastery snapshot: student=%s, date=%s, overall=%.2f",
            student_id,
            snapshot_date,
            overall_mastery,
        )

        return snapshot

    async def update_engagement_metrics(
        self,
        db: AsyncSession,
        student_id: str | UUID,
        metric_date: date | None = None,
    ) -> EngagementMetric | None:
        """Update engagement metrics for a student.

        Calculates and stores engagement metrics based on
        recent activity patterns.

        Args:
            db: Database session.
            student_id: Student identifier.
            metric_date: Date for metrics (defaults to today).

        Returns:
            Updated EngagementMetric or None if no data.

        Example:
            >>> metric = await aggregator.update_engagement_metrics(
            ...     db=db,
            ...     student_id=student_uuid,
            ... )
            >>> if metric and metric.is_improving:
            ...     print("Student is improving!")
        """
        if metric_date is None:
            metric_date = datetime.now(timezone.utc).date()

        student_id_str = str(student_id)

        # Get recent daily summaries (last 7 days)
        week_ago = metric_date - timedelta(days=7)
        summaries_stmt = select(DailySummary).where(
            and_(
                DailySummary.student_id == student_id_str,
                DailySummary.summary_date >= week_ago,
                DailySummary.summary_date <= metric_date,
            )
        ).order_by(DailySummary.summary_date)

        result = await db.execute(summaries_stmt)
        summaries = list(result.scalars().all())

        if not summaries:
            return None

        # Calculate engagement metrics
        login_count = len(summaries)  # Days with activity

        # Average session duration
        total_sessions = sum(s.sessions_count for s in summaries)
        total_time = sum(s.total_time_seconds for s in summaries)
        avg_session_duration = total_time // total_sessions if total_sessions > 0 else 0

        # Questions per session
        total_questions = sum(s.questions_attempted for s in summaries)
        questions_per_session = (
            Decimal(str(round(total_questions / total_sessions, 2)))
            if total_sessions > 0
            else None
        )

        # Accuracy trend (comparing first half to second half)
        accuracy_trend: Decimal | None = None
        if len(summaries) >= 2:
            mid = len(summaries) // 2
            first_half = summaries[:mid]
            second_half = summaries[mid:]

            first_correct = sum(s.questions_correct for s in first_half)
            first_total = sum(s.questions_attempted for s in first_half)
            second_correct = sum(s.questions_correct for s in second_half)
            second_total = sum(s.questions_attempted for s in second_half)

            if first_total > 0 and second_total > 0:
                first_accuracy = first_correct / first_total
                second_accuracy = second_correct / second_total
                trend = second_accuracy - first_accuracy
                accuracy_trend = Decimal(str(round(trend, 2)))

        # Calculate streak
        streak_days = 0
        check_date = metric_date
        while True:
            has_summary = any(s.summary_date == check_date for s in summaries)
            if not has_summary and check_date < metric_date:
                break
            if has_summary:
                streak_days += 1
            check_date -= timedelta(days=1)
            if check_date < week_ago:
                break

        # Check for existing metric
        existing_stmt = select(EngagementMetric).where(
            and_(
                EngagementMetric.student_id == student_id_str,
                EngagementMetric.metric_date == metric_date,
            )
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            existing.login_count = login_count
            existing.session_duration_avg = avg_session_duration
            existing.questions_per_session = questions_per_session
            existing.accuracy_trend = accuracy_trend
            existing.streak_days = streak_days
            return existing
        else:
            metric = EngagementMetric(
                id=str(uuid4()),
                student_id=student_id_str,
                metric_date=metric_date,
                login_count=login_count,
                session_duration_avg=avg_session_duration,
                questions_per_session=questions_per_session,
                accuracy_trend=accuracy_trend,
                streak_days=streak_days,
            )
            db.add(metric)
            return metric
