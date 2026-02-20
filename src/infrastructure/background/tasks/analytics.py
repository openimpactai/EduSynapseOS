# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics background tasks for EduSynapseOS.

Tasks for processing analytics events and generating summaries.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.middleware import get_current_tenant
from src.infrastructure.background.tasks.base import run_async

# Setup broker before defining actors
setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.ANALYTICS,
    max_retries=2,
    time_limit=30000,  # 30 seconds
    priority=Priority.NORMAL,
)
def process_analytics_event(
    event_type: str,
    student_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Process an analytics event from the Event-to-Dramatiq bridge.

    This actor receives events forwarded by the EventToDramatiqBridge and
    stores them in the analytics_events table.

    Args:
        event_type: Analytics event category (engagement, interaction, performance, behavior).
        student_id: Student identifier.
        data: Event data containing:
            - original_event_type: The original EventTypes.* constant
            - session_id: Session identifier
            - session_type: "practice" or "conversation"
            - tenant_code: Tenant identifier
            - Plus original payload fields

    Returns:
        Processing result with event_id and processed status.
    """

    async def _process() -> dict[str, Any]:
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        # Skip if student_id is unknown
        if student_id == "unknown" or not student_id:
            return {
                "event_type": event_type,
                "student_id": student_id,
                "processed": False,
                "reason": "Unknown student_id",
            }

        tenant_code = data.get("tenant_code") or get_current_tenant()
        if not tenant_code:
            return {
                "event_type": event_type,
                "student_id": student_id,
                "processed": False,
                "reason": "No tenant_code provided",
            }

        original_event = data.get("original_event_type", event_type)
        session_id = data.get("session_id")

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                from src.infrastructure.database.models.tenant.analytics import (
                    AnalyticsEvent,
                )

                # Parse UUIDs safely
                student_uuid = None
                session_uuid = None
                try:
                    if len(student_id) == 36:
                        student_uuid = student_id
                except (ValueError, TypeError):
                    pass

                try:
                    if session_id and len(str(session_id)) == 36:
                        session_uuid = str(session_id)
                except (ValueError, TypeError):
                    pass

                # Create analytics event record using existing model structure
                event_record = AnalyticsEvent(
                    id=str(uuid4()),
                    student_id=student_uuid,
                    event_type=original_event,
                    occurred_at=datetime.now(timezone.utc),
                    session_id=session_uuid,
                    data={
                        "analytics_category": event_type,
                        "session_type": data.get("session_type"),
                        "event_id": data.get("event_id"),
                        "timestamp": data.get("timestamp"),
                        **{k: v for k, v in data.items()
                           if k not in ("tenant_code", "original_event_type", "event_id", "timestamp", "session_type")},
                    },
                )

                session.add(event_record)
                await session.commit()

                logger.debug(
                    "Analytics event stored: %s (type: %s, student: %s)",
                    event_record.id,
                    original_event,
                    student_id,
                )

                return {
                    "event_id": str(event_record.id),
                    "event_type": original_event,
                    "analytics_category": event_type,
                    "student_id": student_id,
                    "processed": True,
                }

        except Exception as e:
            logger.error("Failed to store analytics event: %s", str(e), exc_info=True)
            return {
                "event_type": event_type,
                "student_id": student_id,
                "processed": False,
                "error": str(e),
            }

    return run_async(_process())


@dramatiq.actor(
    queue_name=Queues.ANALYTICS,
    max_retries=1,
    time_limit=300000,  # 5 minutes
    priority=Priority.LOW,
)
def aggregate_daily_analytics(
    tenant_code: str,
    date_str: str | None = None,
) -> dict[str, Any]:
    """Aggregate daily analytics summaries.

    Creates or updates DailySummary records for all students with activity
    on the given date. Calculates real metrics from AnalyticsEvent records.

    Args:
        tenant_code: Tenant code.
        date_str: Date to aggregate (YYYY-MM-DD). Defaults to yesterday.

    Returns:
        Aggregation result with counts.
    """

    async def _aggregate() -> dict[str, Any]:
        from sqlalchemy import and_, select
        from sqlalchemy.dialects.postgresql import insert

        from src.infrastructure.database.models.tenant.analytics import (
            AnalyticsEvent,
            DailySummary,
        )
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        # Default to yesterday
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

        start_of_day = datetime.combine(target_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_of_day = datetime.combine(target_date, datetime.max.time()).replace(
            tzinfo=timezone.utc
        )

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Get all unique students with events on this date
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

                result = await session.execute(stmt)
                student_ids = [row[0] for row in result.fetchall()]

                summaries_created = 0
                summaries_updated = 0

                for student_id in student_ids:
                    # Count events and extract metrics for this student
                    events_stmt = select(AnalyticsEvent).where(
                        and_(
                            AnalyticsEvent.student_id == student_id,
                            AnalyticsEvent.occurred_at >= start_of_day,
                            AnalyticsEvent.occurred_at <= end_of_day,
                        )
                    )
                    events_result = await session.execute(events_stmt)
                    events = events_result.scalars().all()

                    # Calculate metrics from events
                    sessions_count = len(
                        set(e.session_id for e in events if e.session_id)
                    )

                    # Count questions from performance events
                    questions_attempted = 0
                    questions_correct = 0
                    messages_sent = 0
                    topics_practiced = set()

                    for event in events:
                        event_data = event.data or {}
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

                    # Check if summary already exists
                    existing_stmt = select(DailySummary).where(
                        and_(
                            DailySummary.student_id == student_id,
                            DailySummary.summary_date == target_date,
                        )
                    )
                    existing_result = await session.execute(existing_stmt)
                    existing = existing_result.scalar_one_or_none()

                    if existing:
                        # Update existing
                        existing.sessions_count = sessions_count
                        existing.questions_attempted = questions_attempted
                        existing.questions_correct = questions_correct
                        existing.messages_sent = messages_sent
                        existing.topics_practiced = list(topics_practiced)
                        if questions_attempted > 0:
                            existing.average_score = (
                                questions_correct / questions_attempted * 100
                            )
                        summaries_updated += 1
                    else:
                        # Create new summary
                        summary = DailySummary(
                            id=str(uuid4()),
                            student_id=student_id,
                            summary_date=target_date,
                            total_time_seconds=0,  # Would need session duration tracking
                            sessions_count=sessions_count,
                            questions_attempted=questions_attempted,
                            questions_correct=questions_correct,
                            messages_sent=messages_sent,
                            topics_practiced=list(topics_practiced),
                            average_score=(
                                questions_correct / questions_attempted * 100
                                if questions_attempted > 0
                                else None
                            ),
                        )
                        session.add(summary)
                        summaries_created += 1

                await session.commit()

                logger.info(
                    "Daily analytics aggregated for %s: %d created, %d updated",
                    target_date,
                    summaries_created,
                    summaries_updated,
                )

                return {
                    "tenant_code": tenant_code,
                    "date": str(target_date),
                    "students_processed": len(student_ids),
                    "summaries_created": summaries_created,
                    "summaries_updated": summaries_updated,
                }

        except Exception as e:
            logger.error("Failed to aggregate daily analytics: %s", str(e), exc_info=True)
            return {
                "tenant_code": tenant_code,
                "date": str(target_date) if date_str else "yesterday",
                "error": str(e),
            }

    return run_async(_aggregate())


def get_analytics_actors() -> list:
    """Get all analytics actors.

    Returns:
        List of analytics actor functions.
    """
    return [
        process_analytics_event,
        aggregate_daily_analytics,
    ]
