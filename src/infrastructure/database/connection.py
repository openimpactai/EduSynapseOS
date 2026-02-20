# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Central database connection management using SQLAlchemy async.

This module provides async database connections for the central database.
The central database stores platform-level data like tenants, system admins,
and licenses.

Uses SQLAlchemy 2.0 async API with asyncpg driver.

Example:
    from src.infrastructure.database.connection import (
        init_central_database,
        get_central_session,
    )

    # Initialize at application startup
    await init_central_database(settings)

    # Use in request handlers
    async with get_central_session() as session:
        result = await session.execute(select(Tenant))
        tenants = result.scalars().all()
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

if TYPE_CHECKING:
    from src.core.config.settings import Settings

# Module-level state for the central database connection
_central_engine: Optional[AsyncEngine] = None
_central_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


class DatabaseError(Exception):
    """Base exception for database operations.

    Attributes:
        message: Human-readable error description.
        original_error: The underlying SQLAlchemy or database error.
    """

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        """Initialize the database error.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.original_error:
            return f"{self.message}: {self.original_error}"
        return self.message


async def init_central_database(settings: "Settings") -> None:
    """Initialize the central database connection pool.

    This should be called once at application startup to create
    the connection pool for the central database.

    Args:
        settings: Application settings containing database configuration.

    Raises:
        DatabaseError: If connection pool creation fails.
    """
    global _central_engine, _central_sessionmaker

    try:
        _central_engine = create_async_engine(
            settings.central_db.url,
            pool_size=settings.central_db.pool_size,
            max_overflow=settings.central_db.max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=settings.debug,
        )

        _central_sessionmaker = async_sessionmaker(
            bind=_central_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    except SQLAlchemyError as e:
        raise DatabaseError("Failed to initialize central database connection", e) from e


async def close_central_database() -> None:
    """Close the central database connection pool.

    This should be called at application shutdown to properly
    close all connections in the pool.
    """
    global _central_engine, _central_sessionmaker

    if _central_engine is not None:
        await _central_engine.dispose()
        _central_engine = None
        _central_sessionmaker = None


def get_central_engine() -> AsyncEngine:
    """Get the central database async engine.

    Returns:
        The SQLAlchemy async engine for the central database.

    Raises:
        DatabaseError: If the database has not been initialized.
    """
    if _central_engine is None:
        raise DatabaseError(
            "Central database not initialized. Call init_central_database() first."
        )
    return _central_engine


def get_central_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Get the central database sessionmaker.

    Returns:
        The SQLAlchemy async sessionmaker for the central database.

    Raises:
        DatabaseError: If the database has not been initialized.
    """
    if _central_sessionmaker is None:
        raise DatabaseError(
            "Central database not initialized. Call init_central_database() first."
        )
    return _central_sessionmaker


@asynccontextmanager
async def get_central_session() -> AsyncIterator[AsyncSession]:
    """Get an async session for the central database.

    This is the primary way to interact with the central database.
    The session is automatically committed on success and rolled back
    on exception.

    Yields:
        AsyncSession for database operations.

    Raises:
        DatabaseError: If the database has not been initialized or
            if a database operation fails.

    Example:
        async with get_central_session() as session:
            result = await session.execute(select(Tenant))
            tenants = result.scalars().all()
    """
    sessionmaker = get_central_sessionmaker()

    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            raise DatabaseError("Database operation failed", e) from e
        except Exception:
            await session.rollback()
            raise


async def check_central_database_connection() -> bool:
    """Check if the central database is reachable.

    Performs a simple query to verify database connectivity.

    Returns:
        True if the database is reachable, False otherwise.
    """
    if _central_engine is None:
        return False

    try:
        async with _central_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
