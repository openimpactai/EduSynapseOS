# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Event-to-Dramatiq Bridge for EduSynapseOS.

This module provides a bridge that forwards EventBus events to Dramatiq actors
for durable, asynchronous processing.

Architecture:
    API/Service → EventBus → Bridge → Redis Queue → Dramatiq Worker → DB

The bridge:
- Subscribes to EventBus events at app startup
- Categorizes events into analytics types (engagement, interaction, performance, behavior)
- Forwards events to Dramatiq actors for durable async processing
- Ensures events survive app restarts (Redis-backed queue)
- Enables worker scaling for high-throughput processing

Event Categories:
    ENGAGEMENT: Session lifecycle events (start, complete, end)
    INTERACTION: User interaction events (messages, questions)
    PERFORMANCE: Learning outcome events (answers, evaluations)
    BEHAVIOR: Learning pattern events (misconceptions, milestones)

Example:
    from src.infrastructure.events.bridge import start_event_bridge, stop_event_bridge

    # In app lifespan
    async with lifespan(app):
        bridge = await start_event_bridge()
        yield
        await stop_event_bridge()
"""

import logging
from typing import Any

from src.infrastructure.events.bus import EventData, get_event_bus
from src.infrastructure.events.types import (
    EventCategory,
    EventRegistry,
    EventTypes,
)

logger = logging.getLogger(__name__)


class EventToDramatiqBridge:
    """Bridge that forwards EventBus events to Dramatiq actors.

    This runs during the app lifecycle and:
    - Subscribes to all analytics-relevant events at startup
    - Normalizes event payloads for consistent processing
    - Sends events to Dramatiq actors for async processing

    Event Subscriptions by Category:

    ENGAGEMENT (Session Lifecycle):
        - practice.session.started
        - practice.session.completed
        - conversation.started
        - conversation.ended

    INTERACTION (User Interactions):
        - conversation.message.received
        - conversation.response.generated
        - practice.question.generated

    PERFORMANCE (Learning Outcomes):
        - student.answer.submitted
        - student.answer.evaluated

    BEHAVIOR (Learning Patterns):
        - practice.misconception.detected
        - student.milestone.reached
        - student.concept.mastered
        - student.struggling.detected
        - student.engagement.changed
        - student.emotion.detected

    Attributes:
        _event_bus: EventBus instance for subscriptions.
        _running: Whether the bridge is active.
        _subscriptions: List of (event_type, handler) tuples for cleanup.
    """

    def __init__(self) -> None:
        """Initialize the bridge."""
        self._event_bus = get_event_bus()
        self._running = False
        self._subscriptions: list[tuple[str, Any]] = []
        self._events_forwarded = 0
        self._errors = 0

    @property
    def is_running(self) -> bool:
        """Check if bridge is running."""
        return self._running

    async def start(self) -> None:
        """Start the bridge and subscribe to events.

        Should be called during app startup after Dramatiq broker is initialized.
        """
        if self._running:
            logger.warning("Event bridge already running")
            return

        logger.info("Starting Event-to-Dramatiq bridge...")

        # ============================================================
        # ENGAGEMENT EVENTS - Session lifecycle
        # ============================================================

        # Practice session events
        self._subscribe(
            EventTypes.Practice.SESSION_STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.Practice.SESSION_COMPLETED,
            self._on_engagement_event,
        )

        # Conversation session events
        self._subscribe(
            EventTypes.Conversation.STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.Conversation.ENDED,
            self._on_engagement_event,
        )

        # Learning Tutor session events
        self._subscribe(
            EventTypes.LearningTutor.SESSION_STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.LearningTutor.SESSION_COMPLETED,
            self._on_engagement_event,
        )

        # Companion session events
        self._subscribe(
            EventTypes.Companion.SESSION_STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.Companion.SESSION_ENDED,
            self._on_engagement_event,
        )

        # Gaming session events
        self._subscribe(
            EventTypes.Gaming.SESSION_STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.Gaming.SESSION_COMPLETED,
            self._on_engagement_event,
        )

        # Practice Helper session events
        self._subscribe(
            EventTypes.PracticeHelper.SESSION_STARTED,
            self._on_engagement_event,
        )
        self._subscribe(
            EventTypes.PracticeHelper.SESSION_COMPLETED,
            self._on_engagement_event,
        )

        # ============================================================
        # INTERACTION EVENTS - User interactions
        # ============================================================

        self._subscribe(
            EventTypes.Conversation.MESSAGE_RECEIVED,
            self._on_interaction_event,
        )
        self._subscribe(
            EventTypes.Conversation.RESPONSE_GENERATED,
            self._on_interaction_event,
        )
        self._subscribe(
            EventTypes.Practice.QUESTION_GENERATED,
            self._on_interaction_event,
        )
        self._subscribe(
            EventTypes.Gaming.MOVE_MADE,
            self._on_interaction_event,
        )

        # ============================================================
        # PERFORMANCE EVENTS - Learning outcomes
        # ============================================================

        self._subscribe(
            EventTypes.Student.ANSWER_SUBMITTED,
            self._on_performance_event,
        )
        self._subscribe(
            EventTypes.Student.ANSWER_EVALUATED,
            self._on_performance_event,
        )
        self._subscribe(
            EventTypes.LearningTutor.UNDERSTANDING_UPDATED,
            self._on_performance_event,
        )

        # ============================================================
        # BEHAVIOR EVENTS - Learning patterns
        # ============================================================

        self._subscribe(
            EventTypes.Practice.MISCONCEPTION_DETECTED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Student.MILESTONE_REACHED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Student.CONCEPT_MASTERED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Student.STRUGGLING_DETECTED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Student.ENGAGEMENT_CHANGED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Student.EMOTION_DETECTED,
            self._on_behavior_event,
        )

        # Learning Tutor behavior events
        self._subscribe(
            EventTypes.LearningTutor.MODE_CHANGED,
            self._on_behavior_event,
        )

        # Companion behavior events
        self._subscribe(
            EventTypes.Companion.EMOTION_DETECTED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Companion.ACTIVITY_SUGGESTED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.Companion.HANDOFF_INITIATED,
            self._on_behavior_event,
        )

        # Gaming behavior events
        self._subscribe(
            EventTypes.Gaming.MISTAKE_DETECTED,
            self._on_behavior_event,
        )

        # Practice Helper behavior events
        self._subscribe(
            EventTypes.PracticeHelper.MODE_ESCALATED,
            self._on_behavior_event,
        )
        self._subscribe(
            EventTypes.PracticeHelper.UNDERSTOOD,
            self._on_behavior_event,
        )

        # ============================================================
        # MEMORY EVENTS - Memory updates
        # ============================================================

        self._subscribe(
            EventTypes.Memory.EPISODIC_STORED,
            self._on_memory_event,
        )
        self._subscribe(
            EventTypes.Memory.MASTERY_UPDATED,
            self._on_memory_event,
        )

        # ============================================================
        # DIAGNOSTIC TRIGGER EVENTS - Conditional diagnostic scans
        # ============================================================

        from src.infrastructure.events.diagnostic_triggers import (
            on_answer_evaluated,
            on_gaming_mistake_detected,
            on_learning_tutor_understanding_updated,
            on_misconception_detected,
            on_practice_helper_mode_escalated,
            on_session_completed,
            on_struggling_detected,
        )

        # Low accuracy trigger (tracks session accuracy, triggers on < 40%)
        self._subscribe(
            EventTypes.Student.ANSWER_EVALUATED,
            on_answer_evaluated,
        )

        # Misconception trigger (maps to relevant detector indicators)
        self._subscribe(
            EventTypes.Practice.MISCONCEPTION_DETECTED,
            on_misconception_detected,
        )

        # Struggling trigger (quick threshold check)
        self._subscribe(
            EventTypes.Student.STRUGGLING_DETECTED,
            on_struggling_detected,
        )

        # Session end trigger (threshold check on low accuracy < 60%)
        self._subscribe(
            EventTypes.Practice.SESSION_COMPLETED,
            on_session_completed,
        )

        # Learning tutor low understanding trigger
        self._subscribe(
            EventTypes.LearningTutor.UNDERSTANDING_UPDATED,
            on_learning_tutor_understanding_updated,
        )

        # Gaming consecutive mistakes trigger (attention indicator)
        self._subscribe(
            EventTypes.Gaming.MISTAKE_DETECTED,
            on_gaming_mistake_detected,
        )

        # Practice helper mode escalation trigger
        self._subscribe(
            EventTypes.PracticeHelper.MODE_ESCALATED,
            on_practice_helper_mode_escalated,
        )

        self._running = True

        logger.info(
            "Event-to-Dramatiq bridge started with %d subscriptions",
            len(self._subscriptions),
        )

    def _subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type string.
            handler: Async handler function.
        """
        self._event_bus.subscribe(event_type, handler)
        self._subscriptions.append((event_type, handler))
        logger.debug("Subscribed to: %s", event_type)

    async def stop(self) -> None:
        """Stop the bridge and unsubscribe from events."""
        if not self._running:
            return

        logger.info("Stopping Event-to-Dramatiq bridge...")

        # Unsubscribe from all events
        for event_type, handler in self._subscriptions:
            self._event_bus.unsubscribe(event_type, handler)

        self._subscriptions.clear()
        self._running = False

        logger.info(
            "Event-to-Dramatiq bridge stopped (forwarded: %d, errors: %d)",
            self._events_forwarded,
            self._errors,
        )

    async def _on_engagement_event(self, event: EventData) -> None:
        """Handle engagement events (session lifecycle)."""
        await self._forward_to_dramatiq(event, EventCategory.ENGAGEMENT)

    async def _on_interaction_event(self, event: EventData) -> None:
        """Handle interaction events (messages, questions)."""
        await self._forward_to_dramatiq(event, EventCategory.INTERACTION)

    async def _on_performance_event(self, event: EventData) -> None:
        """Handle performance events (answers, evaluations)."""
        await self._forward_to_dramatiq(event, EventCategory.PERFORMANCE)

    async def _on_behavior_event(self, event: EventData) -> None:
        """Handle behavior events (misconceptions, milestones)."""
        await self._forward_to_dramatiq(event, EventCategory.BEHAVIOR)

    async def _on_memory_event(self, event: EventData) -> None:
        """Handle memory events (episodic, mastery updates)."""
        # Memory events don't go to analytics but could trigger other tasks
        logger.debug("Memory event received: %s", event.event_type)

    async def _forward_to_dramatiq(
        self,
        event: EventData,
        analytics_type: str,
    ) -> None:
        """Forward an event to Dramatiq for async processing.

        Args:
            event: Event data from EventBus.
            analytics_type: Analytics category.
        """
        try:
            # Import here to avoid circular imports and ensure broker is initialized
            from src.infrastructure.background.tasks import process_analytics_event

            # Extract payload
            payload = event.payload

            # Normalize student_id from various field names
            student_id = (
                payload.get("student_id")
                or payload.get("user_id")
                or payload.get("sender_id")
                or "unknown"
            )

            # Normalize session_id from various field names
            session_id = (
                payload.get("session_id")
                or payload.get("conversation_id")
            )

            # Determine session type from event type
            session_type = EventRegistry.get_session_type(event.event_type)

            # Build normalized data payload for Dramatiq actor
            # Only include JSON-serializable fields
            safe_fields = {}
            for k, v in payload.items():
                if k in ("student_id", "user_id", "sender_id"):
                    continue
                # Only include basic JSON-serializable types
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    safe_fields[k] = v

            data = {
                "original_event_type": event.event_type,
                "event_id": event.event_id,
                "session_id": session_id,
                "session_type": session_type,
                "tenant_code": event.tenant_code or payload.get("tenant_code"),
                "timestamp": event.timestamp.isoformat(),
                **safe_fields,
            }

            # Send to Dramatiq (non-blocking, queues in Redis)
            process_analytics_event.send(
                event_type=analytics_type,
                student_id=str(student_id),
                data=data,
            )

            self._events_forwarded += 1

            logger.debug(
                "Event forwarded to Dramatiq: %s (type: %s, student: %s)",
                event.event_type,
                analytics_type,
                student_id,
            )

        except Exception as e:
            self._errors += 1
            # Log but don't fail - analytics should not break main flow
            logger.error(
                "Failed to forward event to Dramatiq: %s (%s)",
                event.event_type,
                str(e),
                exc_info=True,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "is_running": self._running,
            "subscriptions": len(self._subscriptions),
            "events_forwarded": self._events_forwarded,
            "errors": self._errors,
        }


# Singleton instance
_bridge_instance: EventToDramatiqBridge | None = None


def get_event_bridge() -> EventToDramatiqBridge:
    """Get the singleton bridge instance.

    Returns:
        EventToDramatiqBridge instance.
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = EventToDramatiqBridge()
    return _bridge_instance


async def start_event_bridge() -> EventToDramatiqBridge:
    """Start the event bridge.

    Should be called AFTER setup_dramatiq() to ensure the broker is initialized.

    Returns:
        Started EventToDramatiqBridge instance.
    """
    bridge = get_event_bridge()
    await bridge.start()
    return bridge


async def stop_event_bridge() -> None:
    """Stop the event bridge."""
    global _bridge_instance
    if _bridge_instance is not None:
        await _bridge_instance.stop()
        _bridge_instance = None
