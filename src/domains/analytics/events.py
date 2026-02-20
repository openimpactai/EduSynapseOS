# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytics event tracking module.

This module provides event tracking capabilities for the analytics system.
Events can be tracked through:
1. EventBus (async, flows to Dramatiq via bridge for background processing)
2. Direct database storage (sync, for immediate persistence)

The EventTracker is the primary interface for tracking events. It wraps
the EventBus and provides a clean API for domain services.

Usage:
    from src.domains.analytics import EventTracker

    # Create tracker
    tracker = EventTracker(tenant_code="acme")

    # Track via EventBus (recommended - async, non-blocking)
    await tracker.track_event(
        event_type="student.answer.evaluated",
        student_id=student_uuid,
        data={"is_correct": True, "score": 0.95},
        session_id=session_uuid,
    )

    # Track directly to database (sync, blocking)
    event = await tracker.track_event_sync(
        db=db_session,
        event_type="custom.event.type",
        student_id=student_uuid,
        data={"custom": "data"},
    )
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.analytics import AnalyticsEvent
from src.infrastructure.events import EventBus, EventCategory, get_event_bus

logger = logging.getLogger(__name__)


class EventTracker:
    """Service for tracking analytics events.

    This class provides a clean interface for tracking events in the
    analytics system. It supports both async (EventBus) and sync
    (direct DB) tracking modes.

    The EventBus mode is preferred as it's non-blocking and events
    flow through Dramatiq for durable processing.

    Attributes:
        tenant_code: The tenant code for multi-tenancy.
        _event_bus: EventBus instance for async event publishing.
    """

    def __init__(
        self,
        tenant_code: str,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the event tracker.

        Args:
            tenant_code: Tenant code for multi-tenancy.
            event_bus: Optional EventBus instance (defaults to singleton).
        """
        self.tenant_code = tenant_code
        self._event_bus = event_bus or get_event_bus()

    async def track_event(
        self,
        event_type: str,
        student_id: UUID | str,
        data: dict[str, Any] | None = None,
        session_id: UUID | str | None = None,
        topic_codes: dict[str, str] | None = None,
        user_id: UUID | str | None = None,
    ) -> str:
        """Track an analytics event via EventBus.

        The event is published to the EventBus and flows through the
        EventToDramatiqBridge to the process_analytics_event Dramatiq
        actor for async database storage.

        This is the preferred method as it's non-blocking.

        Args:
            event_type: Type of event (e.g., "student.answer.evaluated").
            student_id: Student identifier.
            data: Additional event data.
            session_id: Related session identifier.
            topic_codes: Topic composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, code.
            user_id: Acting user identifier (if different from student).

        Returns:
            Event ID string.

        Example:
            >>> tracker = EventTracker(tenant_code="acme")
            >>> event_id = await tracker.track_event(
            ...     event_type="student.answer.evaluated",
            ...     student_id=student_uuid,
            ...     data={"is_correct": True},
            ... )
        """
        event_data = {
            "student_id": str(student_id),
            "tenant_code": self.tenant_code,
            **(data or {}),
        }

        if session_id:
            event_data["session_id"] = str(session_id)

        if topic_codes:
            event_data["topic_codes"] = topic_codes

        if user_id:
            event_data["user_id"] = str(user_id)

        event = await self._event_bus.publish(
            event_type=event_type,
            payload=event_data,
            tenant_code=self.tenant_code,
        )

        logger.debug(
            "Event tracked via EventBus: type=%s, student=%s, event_id=%s",
            event_type,
            student_id,
            event.event_id,
        )

        return event.event_id

    async def track_event_sync(
        self,
        db: AsyncSession,
        event_type: str,
        student_id: UUID | str,
        data: dict[str, Any] | None = None,
        session_id: UUID | str | None = None,
        topic_codes: dict[str, str] | None = None,
        user_id: UUID | str | None = None,
        device_type: str | None = None,
        client_version: str | None = None,
    ) -> AnalyticsEvent:
        """Track an analytics event directly to database.

        This method directly creates an AnalyticsEvent record in the
        database. Use this for events that need immediate persistence
        or when the EventBus/Dramatiq infrastructure is not available.

        Note: The caller is responsible for committing the transaction.

        Args:
            db: Database session.
            event_type: Type of event.
            student_id: Student identifier.
            data: Additional event data.
            session_id: Related session identifier.
            topic_codes: Topic composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, code.
            user_id: Acting user identifier.
            device_type: Device type (mobile, web, etc.).
            client_version: Client application version.

        Returns:
            Created AnalyticsEvent model instance.

        Example:
            >>> tracker = EventTracker(tenant_code="acme")
            >>> async with db.begin():
            ...     event = await tracker.track_event_sync(
            ...         db=db,
            ...         event_type="custom.event",
            ...         student_id=student_uuid,
            ...         data={"custom": "value"},
            ...     )
        """
        event_id = str(uuid4())

        event = AnalyticsEvent(
            id=event_id,
            user_id=str(user_id) if user_id else None,
            student_id=str(student_id),
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            session_id=str(session_id) if session_id else None,
            topic_framework_code=topic_codes.get("framework_code") if topic_codes else None,
            topic_subject_code=topic_codes.get("subject_code") if topic_codes else None,
            topic_grade_code=topic_codes.get("grade_code") if topic_codes else None,
            topic_unit_code=topic_codes.get("unit_code") if topic_codes else None,
            topic_code=topic_codes.get("code") if topic_codes else None,
            data=data or {},
            device_type=device_type,
            client_version=client_version,
        )

        db.add(event)

        logger.debug(
            "Event tracked to DB: type=%s, student=%s, event_id=%s",
            event_type,
            student_id,
            event_id,
        )

        return event

    async def track_engagement_event(
        self,
        event_subtype: str,
        student_id: UUID | str,
        session_id: UUID | str,
        session_type: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Track an engagement event (session lifecycle).

        Convenience method for tracking engagement events like
        session starts, completions, and ends.

        Args:
            event_subtype: Subtype (started, completed, ended).
            student_id: Student identifier.
            session_id: Session identifier.
            session_type: Type of session (practice, conversation).
            data: Additional event data.

        Returns:
            Event ID string.

        Example:
            >>> await tracker.track_engagement_event(
            ...     event_subtype="started",
            ...     student_id=student_uuid,
            ...     session_id=session_uuid,
            ...     session_type="practice",
            ... )
        """
        event_type = f"{session_type}.session.{event_subtype}"
        event_data = {
            "session_type": session_type,
            **(data or {}),
        }

        return await self.track_event(
            event_type=event_type,
            student_id=student_id,
            session_id=session_id,
            data=event_data,
        )

    async def track_performance_event(
        self,
        student_id: UUID | str,
        session_id: UUID | str,
        is_correct: bool,
        score: float,
        question_type: str | None = None,
        topic_codes: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Track a performance event (answer evaluation).

        Convenience method for tracking student answer evaluations.

        Args:
            student_id: Student identifier.
            session_id: Session identifier.
            is_correct: Whether the answer was correct.
            score: Score value (0.0 to 1.0).
            question_type: Type of question.
            topic_codes: Topic composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, code.
            data: Additional event data.

        Returns:
            Event ID string.

        Example:
            >>> await tracker.track_performance_event(
            ...     student_id=student_uuid,
            ...     session_id=session_uuid,
            ...     is_correct=True,
            ...     score=0.95,
            ...     topic_codes={"framework_code": "UK-NC-2014", ...},
            ... )
        """
        event_data = {
            "is_correct": is_correct,
            "score": score,
            "analytics_category": EventCategory.PERFORMANCE,
            **(data or {}),
        }

        if question_type:
            event_data["question_type"] = question_type

        return await self.track_event(
            event_type="student.answer.evaluated",
            student_id=student_id,
            session_id=session_id,
            topic_codes=topic_codes,
            data=event_data,
        )

    async def track_interaction_event(
        self,
        interaction_type: str,
        student_id: UUID | str,
        session_id: UUID | str | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Track an interaction event (user actions).

        Convenience method for tracking user interactions like
        messages, hint requests, etc.

        Args:
            interaction_type: Type of interaction (message, hint_request, etc.).
            student_id: Student identifier.
            session_id: Session identifier.
            data: Additional event data.

        Returns:
            Event ID string.

        Example:
            >>> await tracker.track_interaction_event(
            ...     interaction_type="hint_request",
            ...     student_id=student_uuid,
            ...     session_id=session_uuid,
            ... )
        """
        event_data = {
            "interaction_type": interaction_type,
            "analytics_category": EventCategory.INTERACTION,
            **(data or {}),
        }

        return await self.track_event(
            event_type=f"student.{interaction_type}",
            student_id=student_id,
            session_id=session_id,
            data=event_data,
        )

    async def track_behavior_event(
        self,
        behavior_type: str,
        student_id: UUID | str,
        severity: str | None = None,
        topic_codes: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Track a behavior event (learning patterns).

        Convenience method for tracking learning behavior patterns
        like misconceptions, struggles, mastery achievements.

        Args:
            behavior_type: Type of behavior (struggling, mastered, misconception).
            student_id: Student identifier.
            severity: Severity level (low, medium, high).
            topic_codes: Topic composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, code.
            data: Additional event data.

        Returns:
            Event ID string.

        Example:
            >>> await tracker.track_behavior_event(
            ...     behavior_type="struggling.detected",
            ...     student_id=student_uuid,
            ...     severity="high",
            ...     topic_codes={"framework_code": "UK-NC-2014", ...},
            ... )
        """
        event_data = {
            "behavior_type": behavior_type,
            "analytics_category": EventCategory.BEHAVIOR,
            **(data or {}),
        }

        if severity:
            event_data["severity"] = severity

        return await self.track_event(
            event_type=f"student.{behavior_type}",
            student_id=student_id,
            topic_codes=topic_codes,
            data=event_data,
        )


async def track_event(
    tenant_code: str,
    event_type: str,
    student_id: UUID | str,
    data: dict[str, Any] | None = None,
    session_id: UUID | str | None = None,
    topic_codes: dict[str, str] | None = None,
) -> str:
    """Track an analytics event (convenience function).

    This is a convenience function that creates an EventTracker
    and tracks a single event. For multiple events, prefer creating
    an EventTracker instance.

    Args:
        tenant_code: Tenant code for multi-tenancy.
        event_type: Type of event.
        student_id: Student identifier.
        data: Additional event data.
        session_id: Related session identifier.
        topic_codes: Topic composite key dict with keys:
            framework_code, subject_code, grade_code, unit_code, code.

    Returns:
        Event ID string.

    Example:
        >>> from src.domains.analytics import track_event
        >>> event_id = await track_event(
        ...     tenant_code="acme",
        ...     event_type="student.answer.evaluated",
        ...     student_id=student_uuid,
        ...     data={"is_correct": True},
        ... )
    """
    tracker = EventTracker(tenant_code=tenant_code)
    return await tracker.track_event(
        event_type=event_type,
        student_id=student_id,
        data=data,
        session_id=session_id,
        topic_codes=topic_codes,
    )
