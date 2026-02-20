# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for PostgreSQL central database connection.

These tests require a running PostgreSQL instance.
Run with: pytest tests/integration/test_postgres_connection.py -v

Prerequisites:
    - PostgreSQL running at localhost:5432
    - Database 'edusynapse_central' exists
    - User 'edusynapse' with password from CENTRAL_DB_PASSWORD
"""

import pytest

from src.core.config.settings import Settings, clear_settings_cache
from src.infrastructure.database.connection import (
    DatabaseError,
    check_central_database_connection,
    close_central_database,
    get_central_engine,
    get_central_session,
    get_central_sessionmaker,
    init_central_database,
)


@pytest.fixture
def settings() -> Settings:
    """Provide fresh settings for each test."""
    clear_settings_cache()
    return Settings()


@pytest.fixture
async def initialized_database(settings: Settings) -> None:
    """Initialize and cleanup the central database connection."""
    await init_central_database(settings)
    yield
    await close_central_database()


@pytest.mark.integration
class TestCentralDatabaseInitialization:
    """Tests for central database initialization."""

    async def test_init_creates_engine(self, settings: Settings) -> None:
        """Test that initialization creates the database engine."""
        await init_central_database(settings)

        try:
            engine = get_central_engine()
            assert engine is not None
        finally:
            await close_central_database()

    async def test_init_creates_sessionmaker(self, settings: Settings) -> None:
        """Test that initialization creates the sessionmaker."""
        await init_central_database(settings)

        try:
            sessionmaker = get_central_sessionmaker()
            assert sessionmaker is not None
        finally:
            await close_central_database()

    async def test_double_init_succeeds(self, settings: Settings) -> None:
        """Test that re-initialization is safe."""
        await init_central_database(settings)
        await init_central_database(settings)

        try:
            engine = get_central_engine()
            assert engine is not None
        finally:
            await close_central_database()

    async def test_close_clears_state(self, settings: Settings) -> None:
        """Test that close clears the module state."""
        await init_central_database(settings)
        await close_central_database()

        with pytest.raises(DatabaseError) as exc_info:
            get_central_engine()

        assert "not initialized" in str(exc_info.value)


@pytest.mark.integration
class TestCentralDatabaseConnection:
    """Tests for central database connection operations."""

    async def test_check_connection_returns_true_when_connected(
        self, initialized_database: None
    ) -> None:
        """Test that check_connection returns True when database is reachable."""
        result = await check_central_database_connection()
        assert result is True

    async def test_session_can_execute_query(
        self, initialized_database: None
    ) -> None:
        """Test that sessions can execute queries."""
        from sqlalchemy import text

        async with get_central_session() as session:
            result = await session.execute(text("SELECT 1 as value"))
            row = result.fetchone()
            assert row is not None
            assert row.value == 1

    async def test_session_commits_on_success(
        self, initialized_database: None
    ) -> None:
        """Test that sessions commit automatically on success."""
        from sqlalchemy import text

        # This test just verifies the context manager works
        async with get_central_session() as session:
            await session.execute(text("SELECT 1"))
        # If we get here without exception, commit worked

    async def test_session_rollbacks_on_exception(
        self, initialized_database: None
    ) -> None:
        """Test that sessions rollback on exception."""
        from sqlalchemy import text

        with pytest.raises(ValueError):
            async with get_central_session() as session:
                await session.execute(text("SELECT 1"))
                raise ValueError("Test exception")

        # Session should have been rolled back, but we can't easily verify
        # in this simple test. The important thing is no exception from rollback.


@pytest.mark.integration
class TestCentralDatabaseErrors:
    """Tests for database error handling."""

    async def test_get_engine_without_init_raises_error(self) -> None:
        """Test that getting engine without initialization raises error."""
        # Ensure not initialized
        await close_central_database()

        with pytest.raises(DatabaseError) as exc_info:
            get_central_engine()

        assert "not initialized" in str(exc_info.value)

    async def test_get_sessionmaker_without_init_raises_error(self) -> None:
        """Test that getting sessionmaker without initialization raises error."""
        await close_central_database()

        with pytest.raises(DatabaseError) as exc_info:
            get_central_sessionmaker()

        assert "not initialized" in str(exc_info.value)

    async def test_get_session_without_init_raises_error(self) -> None:
        """Test that getting session without initialization raises error."""
        await close_central_database()

        with pytest.raises(DatabaseError):
            async with get_central_session():
                pass

    async def test_check_connection_returns_false_when_not_initialized(
        self,
    ) -> None:
        """Test that check_connection returns False when not initialized."""
        await close_central_database()

        result = await check_central_database_connection()
        assert result is False
