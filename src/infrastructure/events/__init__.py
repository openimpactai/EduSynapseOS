# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Event infrastructure module for EduSynapseOS.

This module provides an in-memory event bus for decoupled communication
between system components, with automatic forwarding to Dramatiq for
durable background processing.

Components:
- EventBus: In-memory pub/sub with pattern matching
- EventTypes: Centralized event type constants
- EventToDramatiqBridge: Forwards events to Dramatiq workers

Architecture:
    API/Service → EventBus.publish() → Bridge → Dramatiq → Worker → DB

Quick Start:
    # Setup (in app lifespan)
    from src.infrastructure.events import (
        get_event_bus,
        EventTypes,
        start_event_bridge,
        stop_event_bridge,
    )

    # Subscribe to events
    event_bus = get_event_bus()
    event_bus.subscribe(EventTypes.Student.ANSWER_EVALUATED, my_handler)

    # Publish events
    await event_bus.publish(
        EventTypes.Student.ANSWER_EVALUATED,
        {"student_id": "123", "is_correct": True},
        tenant_code="acme",
    )

    # Bridge forwards to Dramatiq automatically
    await start_event_bridge()
"""

from src.infrastructure.events.bus import (
    EventBus,
    EventData,
    EventHandler,
    get_event_bus,
    reset_event_bus,
)
from src.infrastructure.events.bridge import (
    EventToDramatiqBridge,
    get_event_bridge,
    start_event_bridge,
    stop_event_bridge,
)
from src.infrastructure.events.types import (
    EventCategory,
    EventPatterns,
    EventRegistry,
    EventTypes,
)
from src.infrastructure.events.diagnostic_triggers import (
    on_answer_evaluated,
    on_misconception_detected,
    on_session_completed,
    on_struggling_detected,
)

__all__ = [
    # Event Bus
    "EventBus",
    "EventData",
    "EventHandler",
    "get_event_bus",
    "reset_event_bus",
    # Event Types
    "EventTypes",
    "EventPatterns",
    "EventCategory",
    "EventRegistry",
    # Bridge
    "EventToDramatiqBridge",
    "get_event_bridge",
    "start_event_bridge",
    "stop_event_bridge",
    # Diagnostic Triggers
    "on_answer_evaluated",
    "on_misconception_detected",
    "on_struggling_detected",
    "on_session_completed",
]
