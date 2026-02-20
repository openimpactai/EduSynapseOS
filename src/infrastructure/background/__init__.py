# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Background task infrastructure module for EduSynapseOS.

Provides production-ready background task processing with Dramatiq:
- Redis broker for message persistence and durability
- Custom middleware for tenant context propagation
- Actor-based task definitions for analytics, memory, reviews
- APScheduler integration for periodic tasks

Quick Start:
    # Setup broker (call once at startup)
    from src.infrastructure.background import setup_dramatiq
    setup_dramatiq()

    # Send tasks
    from src.infrastructure.background import (
        process_analytics_event,
        record_learning_event,
        sync_due_reviews,
    )

    process_analytics_event.send(
        event_type="interaction",
        student_id="123",
        data={"action": "clicked"},
    )

Running Workers:
    dramatiq src.infrastructure.background.tasks --processes 2 --threads 4

Scheduler:
    from src.infrastructure.background import start_scheduler, stop_scheduler

    # Start scheduler with default jobs
    await start_scheduler()

    # Stop at shutdown
    await stop_scheduler()
"""

# Re-export from broker module
from src.infrastructure.background.broker import (
    BrokerManager,
    Priority,
    Queues,
    TenantMiddleware,
    get_broker,
    get_broker_manager,
    get_current_tenant,
    set_current_tenant,
    setup_dramatiq,
    shutdown_dramatiq,
)

# Re-export from scheduler module
from src.infrastructure.background.scheduler import (
    DramatiqScheduler,
    ScheduledTask,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)

# Re-export task actors (import after broker setup)
# Note: Tasks are imported lazily to avoid circular imports
# Use: from src.infrastructure.background.tasks import process_analytics_event

__all__ = [
    # Broker
    "BrokerManager",
    "Priority",
    "Queues",
    "TenantMiddleware",
    "get_broker",
    "get_broker_manager",
    "get_current_tenant",
    "set_current_tenant",
    "setup_dramatiq",
    "shutdown_dramatiq",
    # Scheduler
    "DramatiqScheduler",
    "ScheduledTask",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
]
