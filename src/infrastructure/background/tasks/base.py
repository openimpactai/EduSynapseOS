# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base utilities for Dramatiq tasks.

Provides common utilities used across all task modules.

Thread-Local Event Loop Management:
    Dramatiq workers use multiple threads (--threads N) to process tasks
    concurrently. SQLAlchemy async engines and asyncpg connections are
    bound to specific event loops and cannot be used across different loops.

    This module provides thread-local event loop management that:
    1. Creates a persistent event loop per worker thread
    2. Reuses the same loop for all tasks in that thread
    3. Ensures database connections remain bound to the correct loop

    This solves the "attached to a different loop" error that occurs when
    engines cached in one event loop are used in another.
"""

import asyncio
import logging
import threading
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Thread-local storage for event loops
_thread_local = threading.local()


def _get_thread_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the current thread.

    Each Dramatiq worker thread maintains its own event loop that persists
    for the lifetime of the thread. This ensures SQLAlchemy async engines
    and asyncpg connections remain bound to the correct event loop across
    multiple task executions.

    When a new loop is created (first task in thread or after loop closure),
    any cached database connections are cleared to prevent stale references.

    Returns:
        Event loop for current thread.
    """
    loop = getattr(_thread_local, "event_loop", None)

    if loop is None or loop.is_closed():
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.event_loop = loop

        # Clear any cached DB connections from previous loop
        # Import here to avoid circular imports
        from src.infrastructure.database.tenant_manager import (
            _clear_thread_db_connections,
        )

        _clear_thread_db_connections()

        logger.debug(
            "Created new event loop for thread %s",
            threading.current_thread().name,
        )

    return loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine in sync Dramatiq worker context.

    Uses thread-local persistent event loops to ensure SQLAlchemy
    async engines remain properly bound across task executions within
    the same thread.

    This is the standard way to bridge sync Dramatiq actors with
    async database operations. Each worker thread maintains its own
    event loop and database connection pool.

    Args:
        coro: Coroutine to run.

    Returns:
        Result of coroutine.

    Example:
        @dramatiq.actor
        def my_task(tenant_code: str):
            async def _process():
                db = get_worker_db_manager()
                async with db.get_session(tenant_code) as session:
                    # Database operations
                    pass
            return run_async(_process())
    """
    loop = _get_thread_event_loop()
    return loop.run_until_complete(coro)
