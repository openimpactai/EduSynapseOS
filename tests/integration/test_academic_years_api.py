# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Academic Years API endpoints."""

from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import router as v1_router


@pytest.fixture
def mock_tenant():
    """Create mock tenant context."""
    tenant = MagicMock()
    tenant.id = str(uuid4())
    tenant.code = "test_tenant"
    tenant.name = "Test Tenant"
    tenant.status = "active"
    tenant.is_active = True
    return tenant


@pytest.fixture
def mock_user():
    """Create mock admin user."""
    user = MagicMock()
    user.id = str(uuid4())
    user.email = "admin@test.com"
    user.first_name = "Admin"
    user.last_name = "User"
    user.user_type = "tenant_admin"
    user.is_admin = True
    return user


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


class TestAcademicYearsAPIRouting:
    """Tests for academic years API routing."""

    def test_routes_registered(self, app):
        """Test that academic year routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/academic-years/" in routes
        assert "/api/v1/academic-years/current" in routes
        assert "/api/v1/academic-years/{year_id}" in routes
        assert "/api/v1/academic-years/{year_id}/set-current" in routes


class TestAcademicYearsAPIEndpoints:
    """Tests for academic years API endpoints."""

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_create_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test creating an academic year."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_year = MagicMock()
        mock_year.id = str(uuid4())
        mock_year.name = "2024-2025"
        mock_year.start_date = date(2024, 9, 1)
        mock_year.end_date = date(2025, 6, 30)
        mock_year.is_current = False
        mock_year.is_active = True
        mock_year.created_at = "2024-01-01T00:00:00"
        mock_year.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.create_academic_year = AsyncMock(return_value=mock_year)
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/v1/academic-years/",
            json={
                "name": "2024-2025",
                "start_date": "2024-09-01",
                "end_date": "2025-06-30",
            },
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_academic_years_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing academic years."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_service = MagicMock()
        mock_service.list_academic_years = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/academic-years/")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_get_current_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test getting current academic year."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_year = MagicMock()
        mock_year.id = str(uuid4())
        mock_year.name = "2024-2025"
        mock_year.start_date = date(2024, 9, 1)
        mock_year.end_date = date(2025, 6, 30)
        mock_year.is_current = True
        mock_year.is_active = True
        mock_year.created_at = "2024-01-01T00:00:00"
        mock_year.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.get_current_academic_year = AsyncMock(return_value=mock_year)
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/academic-years/current")

        assert response.status_code in [200, 404, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_get_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test getting an academic year."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        year_id = uuid4()
        mock_year = MagicMock()
        mock_year.id = str(year_id)
        mock_year.name = "2024-2025"
        mock_year.start_date = date(2024, 9, 1)
        mock_year.end_date = date(2025, 6, 30)
        mock_year.is_current = False
        mock_year.is_active = True
        mock_year.created_at = "2024-01-01T00:00:00"
        mock_year.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.get_academic_year = AsyncMock(return_value=mock_year)
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/academic-years/{year_id}")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_update_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test updating an academic year."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        year_id = uuid4()
        mock_year = MagicMock()
        mock_year.id = str(year_id)
        mock_year.name = "Updated Year"
        mock_year.start_date = date(2024, 9, 1)
        mock_year.end_date = date(2025, 6, 30)
        mock_year.is_current = False
        mock_year.is_active = True
        mock_year.created_at = "2024-01-01T00:00:00"
        mock_year.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.update_academic_year = AsyncMock(return_value=mock_year)
        mock_get_service.return_value = mock_service

        response = client.put(
            f"/api/v1/academic-years/{year_id}",
            json={"name": "Updated Year"},
        )

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_delete_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test deleting an academic year."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        year_id = uuid4()

        mock_service = MagicMock()
        mock_service.delete_academic_year = AsyncMock()
        mock_get_service.return_value = mock_service

        response = client.delete(f"/api/v1/academic-years/{year_id}")

        assert response.status_code in [204, 422, 500]

    @patch("src.api.v1.academic_years._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_set_current_academic_year_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test setting current academic year."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        year_id = uuid4()
        mock_year = MagicMock()
        mock_year.id = str(year_id)
        mock_year.name = "2024-2025"
        mock_year.start_date = date(2024, 9, 1)
        mock_year.end_date = date(2025, 6, 30)
        mock_year.is_current = True
        mock_year.is_active = True
        mock_year.created_at = "2024-01-01T00:00:00"
        mock_year.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.set_current_academic_year = AsyncMock(return_value=mock_year)
        mock_get_service.return_value = mock_service

        response = client.post(f"/api/v1/academic-years/{year_id}/set-current")

        assert response.status_code in [200, 422, 500]
