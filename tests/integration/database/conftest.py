# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pytest fixtures for database integration tests.

Provides database sessions and engines for testing.
"""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.infrastructure.database.models.base import Base


@pytest.fixture(scope="session")
def central_db_url() -> str:
    """Get central database URL for tests."""
    return os.environ.get(
        "TEST_CENTRAL_DB_URL",
        os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://edusynapse:edusynapse@localhost:34001/edusynapse_central_test"),
    )


@pytest.fixture(scope="session")
def tenant_db_url() -> str:
    """Get tenant database URL for tests."""
    return os.environ.get(
        "TEST_TENANT_DB_URL",
        os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://edusynapse:edusynapse@localhost:34001/edusynapse_tenant_test"),
    )


@pytest_asyncio.fixture(scope="function")
async def central_db_engine(central_db_url: str):
    """Create async engine for central database tests."""
    engine = create_async_engine(central_db_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def tenant_db_engine(tenant_db_url: str):
    """Create async engine for tenant database tests."""
    engine = create_async_engine(tenant_db_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def central_db_session(central_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for central database tests."""
    async_session = async_sessionmaker(
        central_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def tenant_db_session(tenant_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for tenant database tests."""
    async_session = async_sessionmaker(
        tenant_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()
