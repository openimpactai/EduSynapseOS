# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""In-memory event bus for EduSynapseOS.

This module provides an async event bus for decoupled communication
between system components. Events are published and subscribed to
by event type strings.

The EventBus supports:
- Exact event type matching (e.g., "student.answer.evaluated")
- Wildcard pattern matching (e.g., "student.*", "*.session.completed")
- Async handlers
- Multiple handlers per event type

Example:
    from src.infrastructure.events import get_event_bus, EventTypes

    event_bus = get_event_bus()

    # Subscribe to specific event
    async def on_answer_evaluated(data):
        print(f"Answer evaluated: {data}")

    event_bus.subscribe(EventTypes.Student.ANSWER_EVALUATED, on_answer_evaluated)

    # Subscribe to pattern
    event_bus.subscribe("student.*", on_any_student_event)

    # Publish event
    await event_bus.publish(
        EventTypes.Student.ANSWER_EVALUATED,
        {"student_id": "123", "is_correct": True}
    )
"""

import asyncio
import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[[Any], Awaitable[None]]


@dataclass
class EventData:
    """Container for event data with metadata.

    Attributes:
        event_type: The event type string.
        payload: The event payload data.
        event_id: Unique event identifier.
        timestamp: When the event was published.
        tenant_code: Optional tenant code for multi-tenancy.
    """

    event_type: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "tenant_code": self.tenant_code,
        }


class EventBus:
    """In-memory async event bus with pattern matching support.

    This event bus allows components to communicate without direct
    dependencies. Publishers emit events, and subscribers receive
    them based on exact match or wildcard patterns.

    Thread-safety: This implementation is designed for single-threaded
    async use. For multi-process scenarios, use Redis pub/sub or
    similar distributed messaging.

    Attributes:
        _handlers: Dictionary mapping event types to handler lists.
        _pattern_handlers: Dictionary mapping patterns to handler lists.

    Example:
        bus = EventBus()

        # Exact subscription
        bus.subscribe("user.created", handler)

        # Pattern subscription
        bus.subscribe("user.*", pattern_handler)

        # Publish
        await bus.publish("user.created", {"id": "123"})
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[str, list[EventHandler]] = {}
        self._pattern_handlers: dict[str, list[EventHandler]] = {}
        self._event_count = 0
        logger.debug("EventBus initialized")

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe a handler to an event type or pattern.

        Args:
            event_type: Event type string or pattern with wildcards.
            handler: Async function to call when event is published.

        Example:
            # Exact match
            bus.subscribe("student.answer.evaluated", my_handler)

            # Pattern match (any student event)
            bus.subscribe("student.*", my_handler)

            # Pattern match (any session completed)
            bus.subscribe("*.session.completed", my_handler)
        """
        # Check if this is a pattern (contains wildcards)
        if "*" in event_type or "?" in event_type:
            if event_type not in self._pattern_handlers:
                self._pattern_handlers[event_type] = []
            self._pattern_handlers[event_type].append(handler)
            logger.debug("Subscribed pattern handler to: %s", event_type)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            logger.debug("Subscribed handler to: %s", event_type)

    def unsubscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> bool:
        """Unsubscribe a handler from an event type or pattern.

        Args:
            event_type: Event type string or pattern.
            handler: The handler function to remove.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        # Check patterns first
        if "*" in event_type or "?" in event_type:
            if event_type in self._pattern_handlers:
                try:
                    self._pattern_handlers[event_type].remove(handler)
                    if not self._pattern_handlers[event_type]:
                        del self._pattern_handlers[event_type]
                    return True
                except ValueError:
                    return False
        else:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                    if not self._handlers[event_type]:
                        del self._handlers[event_type]
                    return True
                except ValueError:
                    return False
        return False

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        tenant_code: str | None = None,
    ) -> EventData:
        """Publish an event to all matching subscribers.

        Handlers are called concurrently using asyncio.gather.
        Errors in individual handlers are logged but don't stop
        other handlers from executing.

        Args:
            event_type: The event type string.
            payload: Event data dictionary.
            tenant_code: Optional tenant code for multi-tenancy.

        Returns:
            EventData object with event metadata.

        Example:
            event = await bus.publish(
                "student.answer.evaluated",
                {"student_id": "123", "is_correct": True},
                tenant_code="acme",
            )
        """
        event = EventData(
            event_type=event_type,
            payload=payload,
            tenant_code=tenant_code,
        )

        self._event_count += 1

        # Collect all matching handlers
        handlers_to_call: list[EventHandler] = []

        # Exact match handlers
        if event_type in self._handlers:
            handlers_to_call.extend(self._handlers[event_type])

        # Pattern match handlers
        for pattern, pattern_handlers in self._pattern_handlers.items():
            if fnmatch.fnmatch(event_type, pattern):
                handlers_to_call.extend(pattern_handlers)

        if not handlers_to_call:
            logger.debug(
                "No handlers for event: %s (tenant: %s)",
                event_type,
                tenant_code,
            )
            return event

        # Call all handlers concurrently
        logger.debug(
            "Publishing event %s to %d handlers (tenant: %s)",
            event_type,
            len(handlers_to_call),
            tenant_code,
        )

        async def safe_call(handler: EventHandler) -> None:
            """Call handler with error handling."""
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    "Handler error for event %s: %s",
                    event_type,
                    str(e),
                    exc_info=True,
                )

        await asyncio.gather(
            *[safe_call(handler) for handler in handlers_to_call],
            return_exceptions=True,
        )

        return event

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
        self._pattern_handlers.clear()
        logger.debug("EventBus cleared all subscriptions")

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics.

        Returns:
            Dictionary with subscription and event counts.
        """
        exact_count = sum(len(h) for h in self._handlers.values())
        pattern_count = sum(len(h) for h in self._pattern_handlers.values())

        return {
            "exact_subscriptions": len(self._handlers),
            "pattern_subscriptions": len(self._pattern_handlers),
            "total_handlers": exact_count + pattern_count,
            "events_published": self._event_count,
            "event_types": list(self._handlers.keys()),
            "patterns": list(self._pattern_handlers.keys()),
        }


# Singleton instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the singleton event bus instance.

    Returns:
        EventBus instance.
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the event bus singleton.

    Useful for testing to ensure clean state between tests.
    """
    global _event_bus
    if _event_bus is not None:
        _event_bus.clear()
    _event_bus = None
