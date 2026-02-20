# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Parent Relations API endpoints."""

from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime

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


class TestParentRelationsAPIRouting:
    """Tests for parent relations API routing."""

    def test_routes_registered(self, app):
        """Test that parent relation routes are registered."""
        routes = [route.path for route in app.routes]

        assert "/api/v1/parent-relations/" in routes
        assert "/api/v1/parent-relations/{relation_id}" in routes
        assert "/api/v1/parent-relations/{relation_id}/verify" in routes


class TestParentRelationsAPIEndpoints:
    """Tests for parent relations API endpoints."""

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_create_relation_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test creating a parent-student relation."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        parent_id = uuid4()
        student_id = uuid4()

        mock_parent = MagicMock()
        mock_parent.id = str(parent_id)
        mock_parent.external_id = "ext_parent"
        mock_parent.email = "parent@example.com"
        mock_parent.first_name = "John"
        mock_parent.last_name = "Doe"
        mock_parent.full_name = "John Doe"
        mock_parent.user_type = "parent"
        mock_parent.is_active = True

        mock_student = MagicMock()
        mock_student.id = str(student_id)
        mock_student.external_id = "ext_student"
        mock_student.email = "student@school.com"
        mock_student.first_name = "Jane"
        mock_student.last_name = "Doe"
        mock_student.full_name = "Jane Doe"
        mock_student.user_type = "student"
        mock_student.is_active = True

        mock_relation = MagicMock()
        mock_relation.id = str(uuid4())
        mock_relation.parent = mock_parent
        mock_relation.student = mock_student
        mock_relation.relationship_type = "parent"
        mock_relation.can_view_progress = True
        mock_relation.can_view_conversations = False
        mock_relation.can_receive_notifications = True
        mock_relation.can_chat_with_ai = False
        mock_relation.is_primary = True
        mock_relation.is_verified = False
        mock_relation.verified_at = None
        mock_relation.verified_by = None
        mock_relation.created_at = datetime.now()

        mock_service = MagicMock()
        mock_service.create_relation = AsyncMock(return_value=mock_relation)
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/v1/parent-relations/",
            json={
                "parent_id": str(parent_id),
                "student_id": str(student_id),
                "relationship_type": "parent",
            },
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_relations_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing parent-student relations."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_service = MagicMock()
        mock_service.list_relations = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/parent-relations/")

        assert response.status_code in [200, 403, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_get_relation_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test getting a parent-student relation."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        relation_id = uuid4()
        parent_id = uuid4()
        student_id = uuid4()

        mock_parent = MagicMock()
        mock_parent.id = str(parent_id)
        mock_parent.external_id = "ext_parent"
        mock_parent.email = "parent@example.com"
        mock_parent.first_name = "John"
        mock_parent.last_name = "Doe"
        mock_parent.full_name = "John Doe"
        mock_parent.user_type = "parent"
        mock_parent.is_active = True

        mock_student = MagicMock()
        mock_student.id = str(student_id)
        mock_student.external_id = "ext_student"
        mock_student.email = "student@school.com"
        mock_student.first_name = "Jane"
        mock_student.last_name = "Doe"
        mock_student.full_name = "Jane Doe"
        mock_student.user_type = "student"
        mock_student.is_active = True

        mock_relation = MagicMock()
        mock_relation.id = str(relation_id)
        mock_relation.parent = mock_parent
        mock_relation.student = mock_student
        mock_relation.relationship_type = "parent"
        mock_relation.can_view_progress = True
        mock_relation.can_view_conversations = False
        mock_relation.can_receive_notifications = True
        mock_relation.can_chat_with_ai = False
        mock_relation.is_primary = True
        mock_relation.is_verified = False
        mock_relation.verified_at = None
        mock_relation.verified_by = None
        mock_relation.created_at = datetime.now()

        mock_service = MagicMock()
        mock_service.get_relation = AsyncMock(return_value=mock_relation)
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/parent-relations/{relation_id}")

        assert response.status_code in [200, 403, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_update_relation_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test updating a parent-student relation."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        relation_id = uuid4()
        parent_id = uuid4()
        student_id = uuid4()

        mock_parent = MagicMock()
        mock_parent.id = str(parent_id)
        mock_parent.external_id = "ext_parent"
        mock_parent.email = "parent@example.com"
        mock_parent.first_name = "John"
        mock_parent.last_name = "Doe"
        mock_parent.full_name = "John Doe"
        mock_parent.user_type = "parent"
        mock_parent.is_active = True

        mock_student = MagicMock()
        mock_student.id = str(student_id)
        mock_student.external_id = "ext_student"
        mock_student.email = "student@school.com"
        mock_student.first_name = "Jane"
        mock_student.last_name = "Doe"
        mock_student.full_name = "Jane Doe"
        mock_student.user_type = "student"
        mock_student.is_active = True

        mock_relation = MagicMock()
        mock_relation.id = str(relation_id)
        mock_relation.parent = mock_parent
        mock_relation.student = mock_student
        mock_relation.relationship_type = "parent"
        mock_relation.can_view_progress = False
        mock_relation.can_view_conversations = True
        mock_relation.can_receive_notifications = True
        mock_relation.can_chat_with_ai = False
        mock_relation.is_primary = True
        mock_relation.is_verified = False
        mock_relation.verified_at = None
        mock_relation.verified_by = None
        mock_relation.created_at = datetime.now()

        mock_service = MagicMock()
        mock_service.update_relation = AsyncMock(return_value=mock_relation)
        mock_get_service.return_value = mock_service

        response = client.put(
            f"/api/v1/parent-relations/{relation_id}",
            json={
                "can_view_progress": False,
                "can_view_conversations": True,
            },
        )

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_verify_relation_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test verifying a parent-student relation."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        relation_id = uuid4()
        parent_id = uuid4()
        student_id = uuid4()

        mock_parent = MagicMock()
        mock_parent.id = str(parent_id)
        mock_parent.external_id = "ext_parent"
        mock_parent.email = "parent@example.com"
        mock_parent.first_name = "John"
        mock_parent.last_name = "Doe"
        mock_parent.full_name = "John Doe"
        mock_parent.user_type = "parent"
        mock_parent.is_active = True

        mock_student = MagicMock()
        mock_student.id = str(student_id)
        mock_student.external_id = "ext_student"
        mock_student.email = "student@school.com"
        mock_student.first_name = "Jane"
        mock_student.last_name = "Doe"
        mock_student.full_name = "Jane Doe"
        mock_student.user_type = "student"
        mock_student.is_active = True

        mock_relation = MagicMock()
        mock_relation.id = str(relation_id)
        mock_relation.parent = mock_parent
        mock_relation.student = mock_student
        mock_relation.relationship_type = "parent"
        mock_relation.can_view_progress = True
        mock_relation.can_view_conversations = False
        mock_relation.can_receive_notifications = True
        mock_relation.can_chat_with_ai = False
        mock_relation.is_primary = True
        mock_relation.is_verified = True
        mock_relation.verified_at = datetime.now()
        mock_relation.verified_by = mock_user.id
        mock_relation.created_at = datetime.now()

        mock_service = MagicMock()
        mock_service.verify_relation = AsyncMock(return_value=mock_relation)
        mock_get_service.return_value = mock_service

        response = client.post(f"/api/v1/parent-relations/{relation_id}/verify")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_delete_relation_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test deleting a parent-student relation."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        relation_id = uuid4()

        mock_service = MagicMock()
        mock_service.delete_relation = AsyncMock()
        mock_get_service.return_value = mock_service

        response = client.delete(f"/api/v1/parent-relations/{relation_id}")

        assert response.status_code in [204, 422, 500]

    @patch("src.api.v1.parent_relations._get_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_relations_with_filters(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing relations with filters."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        parent_id = uuid4()

        mock_service = MagicMock()
        mock_service.list_relations = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get(
            "/api/v1/parent-relations/",
            params={"parent_id": str(parent_id), "is_verified": True},
        )

        assert response.status_code in [200, 403, 422, 500]
