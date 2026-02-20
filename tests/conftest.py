# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pytest configuration and shared fixtures.

This module provides fixtures used across all test types:
- Unit tests
- Integration tests
- End-to-end tests
"""

import asyncio
from collections.abc import Generator
from typing import Any

import pytest


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session.

    This fixture provides a single event loop for all async tests
    in the session, improving performance.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_environment() -> dict[str, str]:
    """Provide test environment variables.

    Returns:
        Dictionary of environment variables for testing.
    """
    return {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "CENTRAL_DATABASE_URL": "postgresql+asyncpg://edusynapse:edusynapse_central_password@localhost:34001/edusynapse_central",
        "TENANT_DB_USER": "edusynapse",
        "TENANT_DB_PASSWORD": "edusynapse_tenant_password",
        "TENANT_DB_PORT_RANGE_START": "44000",
        "REDIS_URL": "redis://localhost:34002/0",
        "QDRANT_URL": "http://localhost:34003",
        "JWT_SECRET_KEY": "test-secret-key-for-testing-only",
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
        "REFRESH_TOKEN_EXPIRE_DAYS": "7",
    }


# =============================================================================
# Marker Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires services)"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test (requires full stack)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def sample_student_id() -> str:
    """Provide a sample student ID for testing."""
    return "550e8400-e29b-41d4-a716-446655440001"


@pytest.fixture
def sample_tenant_id() -> str:
    """Provide a sample tenant ID for testing."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def sample_tenant_code() -> str:
    """Provide a sample tenant code for testing."""
    return "test_tenant"


@pytest.fixture
def sample_user_data() -> dict[str, Any]:
    """Provide sample user data for testing."""
    return {
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "user_type": "student",
        "status": "active",
    }


@pytest.fixture
def sample_practice_session_data() -> dict[str, Any]:
    """Provide sample practice session data for testing."""
    return {
        "topic_id": "550e8400-e29b-41d4-a716-446655440002",
        "session_type": "quick",
        "persona_id": "coach",
    }
