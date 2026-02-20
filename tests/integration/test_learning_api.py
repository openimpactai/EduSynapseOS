# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for learning API endpoints."""

from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import router as v1_router


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(v1_router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestLearningEndpoints:
    """Tests for learning endpoints."""

    def test_routes_registered(self, app):
        """Test that learning routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/learning/start" in routes
        assert "/api/v1/learning/" in routes
        assert "/api/v1/learning/{conversation_id}" in routes
        assert "/api/v1/learning/{conversation_id}/history" in routes
        assert "/api/v1/learning/{conversation_id}/message" in routes
        assert "/api/v1/learning/{conversation_id}/archive" in routes

    def test_websocket_route_registered(self, app):
        """Test that WebSocket route is registered."""
        websocket_routes = [
            route.path
            for route in app.routes
            if hasattr(route, "path") and "ws" in route.path
        ]

        assert "/api/v1/learning/{conversation_id}/ws" in websocket_routes


class TestAssessmentEndpoints:
    """Tests for assessment endpoints."""

    def test_routes_registered(self, app):
        """Test that assessment routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/assessment/start" in routes
        assert "/api/v1/assessment/{session_id}" in routes
        assert "/api/v1/assessment/{session_id}/question" in routes
        assert "/api/v1/assessment/{session_id}/answer" in routes
        assert "/api/v1/assessment/{session_id}/result" in routes


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    def test_routes_registered(self, app):
        """Test that analytics routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/analytics/dashboard" in routes
        assert "/api/v1/analytics/progress/{student_id}" in routes
        assert "/api/v1/analytics/class/{class_id}" in routes


class TestMemoryEndpoints:
    """Tests for memory endpoints."""

    def test_routes_registered(self, app):
        """Test that memory routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/memory/context" in routes
        assert "/api/v1/memory/mastery" in routes
        assert "/api/v1/memory/weak-areas" in routes
        assert "/api/v1/memory/review-schedule" in routes
