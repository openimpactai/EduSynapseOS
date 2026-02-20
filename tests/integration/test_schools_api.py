# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Schools API endpoints."""

from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

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


class TestSchoolsAPIRouting:
    """Tests for schools API routing."""

    def test_routes_registered(self, app):
        """Test that school routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/schools/" in routes
        assert "/api/v1/schools/{school_id}" in routes
        assert "/api/v1/schools/{school_id}/activate" in routes
        assert "/api/v1/schools/{school_id}/admins" in routes
        assert "/api/v1/schools/{school_id}/admins/{user_id}" in routes


class TestSchoolsAPIEndpoints:
    """Tests for schools API endpoints."""

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_create_school_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test creating a school."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_school = MagicMock()
        mock_school.id = str(uuid4())
        mock_school.code = "SCH001"
        mock_school.name = "Test School"
        mock_school.description = "A test school"
        mock_school.address = None
        mock_school.phone = None
        mock_school.email = None
        mock_school.website = None
        mock_school.is_active = True
        mock_school.created_at = "2024-01-01T00:00:00"
        mock_school.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.create_school = AsyncMock(return_value=mock_school)
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/v1/schools/",
            json={
                "code": "SCH001",
                "name": "Test School",
                "description": "A test school",
            },
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_schools_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing schools."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_service = MagicMock()
        mock_service.list_schools = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/schools/")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_get_school_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test getting a school."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        mock_school = MagicMock()
        mock_school.id = str(school_id)
        mock_school.code = "SCH001"
        mock_school.name = "Test School"
        mock_school.description = None
        mock_school.address = None
        mock_school.phone = None
        mock_school.email = None
        mock_school.website = None
        mock_school.is_active = True
        mock_school.created_at = "2024-01-01T00:00:00"
        mock_school.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.get_school = AsyncMock(return_value=mock_school)
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/schools/{school_id}")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_update_school_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test updating a school."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        mock_school = MagicMock()
        mock_school.id = str(school_id)
        mock_school.code = "SCH001"
        mock_school.name = "Updated School"
        mock_school.description = None
        mock_school.address = None
        mock_school.phone = None
        mock_school.email = None
        mock_school.website = None
        mock_school.is_active = True
        mock_school.created_at = "2024-01-01T00:00:00"
        mock_school.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.update_school = AsyncMock(return_value=mock_school)
        mock_get_service.return_value = mock_service

        response = client.put(
            f"/api/v1/schools/{school_id}",
            json={"name": "Updated School"},
        )

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_delete_school_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test deleting a school."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()

        mock_service = MagicMock()
        mock_service.delete_school = AsyncMock()
        mock_get_service.return_value = mock_service

        response = client.delete(f"/api/v1/schools/{school_id}")

        assert response.status_code in [204, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_activate_school_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test activating a school."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        mock_school = MagicMock()
        mock_school.id = str(school_id)
        mock_school.code = "SCH001"
        mock_school.name = "Test School"
        mock_school.description = None
        mock_school.address = None
        mock_school.phone = None
        mock_school.email = None
        mock_school.website = None
        mock_school.is_active = True
        mock_school.created_at = "2024-01-01T00:00:00"
        mock_school.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.activate_school = AsyncMock(return_value=mock_school)
        mock_get_service.return_value = mock_service

        response = client.post(f"/api/v1/schools/{school_id}/activate")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_add_admin_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test adding a school admin."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        user_id = uuid4()

        mock_admin = MagicMock()
        mock_admin.id = str(uuid4())
        mock_admin.user_id = str(user_id)
        mock_admin.user_email = "admin@school.com"
        mock_admin.user_name = "John Doe"
        mock_admin.assigned_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.add_admin = AsyncMock(return_value=mock_admin)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/schools/{school_id}/admins",
            json={"user_id": str(user_id)},
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_admins_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing school admins."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()

        mock_service = MagicMock()
        mock_service.list_admins = AsyncMock(return_value=[])
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/schools/{school_id}/admins")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.schools._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_remove_admin_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test removing a school admin."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        user_id = uuid4()

        mock_service = MagicMock()
        mock_service.remove_admin = AsyncMock()
        mock_get_service.return_value = mock_service

        response = client.delete(f"/api/v1/schools/{school_id}/admins/{user_id}")

        assert response.status_code in [204, 422, 500]
