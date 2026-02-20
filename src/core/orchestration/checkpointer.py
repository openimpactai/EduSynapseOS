# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PostgreSQL checkpointer for LangGraph workflows.

This module provides checkpointing functionality that persists workflow
state to PostgreSQL, enabling:
- Workflow interruption and resumption
- Human-in-the-loop interactions
- State recovery after failures
- Session persistence across API calls

The checkpointer is initialized once at application startup and shared
across all workflow services via dependency injection.

Uses psycopg_pool.AsyncConnectionPool for persistent connections that
survive the full application lifespan (not context manager based).

Usage:
    # In app.py lifespan:
    await init_checkpointer(connection_string)

    # In dependencies.py:
    def get_checkpointer() -> AsyncPostgresSaver | None:
        return get_checkpointer_instance()

    # In service:
    def __init__(self, checkpointer: AsyncPostgresSaver | None = None):
        self._checkpointer = checkpointer

References:
    - https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres
    - langgraph-checkpoint-postgres 3.0.2
"""

import logging
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

# Module-level singleton instances
_connection_pool: Optional[AsyncConnectionPool] = None
_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_initialized: bool = False


async def init_checkpointer(connection_string: str) -> AsyncPostgresSaver:
    """Initialize the global checkpointer singleton.

    This should be called once at application startup (in lifespan).
    Creates an AsyncConnectionPool and AsyncPostgresSaver, then sets up
    the required database tables.

    The connection pool is managed separately from the checkpointer to ensure
    proper lifecycle management (pool must outlive all checkpointer operations).

    Args:
        connection_string: PostgreSQL connection string (asyncpg format will be
            converted to psycopg format automatically).

    Returns:
        Initialized AsyncPostgresSaver instance.
    """
    global _connection_pool, _checkpointer, _checkpointer_initialized

    if _checkpointer_initialized:
        logger.warning("Checkpointer already initialized, returning existing instance")
        return _checkpointer

    logger.info("Initializing PostgreSQL checkpointer with connection pool")

    # Convert asyncpg URL format to psycopg format if needed
    # asyncpg: postgresql+asyncpg://user:pass@host:port/db
    # psycopg: postgresql://user:pass@host:port/db
    psycopg_conn_string = connection_string.replace("+asyncpg", "")

    # Create connection pool with proper settings
    # autocommit=True is required for setup() to commit DDL statements
    # row_factory=dict_row is required for langgraph's dictionary-style row access
    _connection_pool = AsyncConnectionPool(
        conninfo=psycopg_conn_string,
        min_size=2,
        max_size=10,
        open=False,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },
    )
    await _connection_pool.open()

    # Create checkpointer with the connection pool
    _checkpointer = AsyncPostgresSaver(conn=_connection_pool)

    # Setup creates required tables (checkpoint_migrations, checkpoint_blobs, etc.)
    await _checkpointer.setup()

    _checkpointer_initialized = True
    logger.info("PostgreSQL checkpointer initialized successfully")

    return _checkpointer


def get_checkpointer_instance() -> Optional[AsyncPostgresSaver]:
    """Get the global checkpointer instance.

    Returns the singleton checkpointer if initialized, None otherwise.
    This is a sync function suitable for use as a FastAPI dependency.

    Returns:
        AsyncPostgresSaver instance or None if not initialized.
    """
    if not _checkpointer_initialized:
        logger.warning("Checkpointer not initialized - call init_checkpointer first")
        return None

    return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer and release resources.

    Should be called during application shutdown.
    Closes the connection pool which releases all database connections.
    """
    global _connection_pool, _checkpointer, _checkpointer_initialized

    if _connection_pool is not None:
        logger.info("Closing PostgreSQL checkpointer connection pool")
        await _connection_pool.close()
        _connection_pool = None

    _checkpointer = None
    _checkpointer_initialized = False


async def reset_checkpointer() -> None:
    """Reset the global checkpointer instance.

    Useful for testing or when database connection changes.
    Closes any existing connection pool before resetting.
    """
    await close_checkpointer()
    logger.debug("Checkpointer reset")


def create_thread_config(
    thread_id: str,
    checkpoint_ns: str = "",
) -> dict:
    """Create a LangGraph configuration dict for checkpointing.

    Args:
        thread_id: Unique identifier for this workflow thread.
            Typically: f"{session_type}_{session_id}"
        checkpoint_ns: Namespace for the checkpoint (optional).

    Returns:
        Config dict for use with workflow.ainvoke().

    Example:
        >>> config = create_thread_config("practice_abc123")
        >>> result = await workflow.ainvoke(state, config=config)
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def create_session_thread_id(
    session_type: str,
    session_id: str,
) -> str:
    """Create a thread ID for a practice/tutoring session.

    Args:
        session_type: Type of session (practice, tutoring).
        session_id: Session identifier.

    Returns:
        Thread ID string for checkpointing.

    Example:
        >>> thread_id = create_session_thread_id("practice", "abc123")
        >>> # Returns: "practice_abc123"
    """
    return f"{session_type}_{session_id}"
