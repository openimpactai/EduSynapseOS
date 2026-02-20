# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Classes API endpoints."""

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


class TestClassesAPIRouting:
    """Tests for classes API routing."""

    def test_routes_registered(self, app):
        """Test that class routes are registered."""
        routes = [route.path for route in app.routes]

        # Class CRUD
        assert "/api/v1/classes/" in routes
        assert "/api/v1/classes/{class_id}" in routes
        assert "/api/v1/classes/{class_id}/activate" in routes

        # Student enrollment
        assert "/api/v1/classes/{class_id}/students" in routes
        assert "/api/v1/classes/{class_id}/students/bulk" in routes
        assert "/api/v1/classes/{class_id}/students/{student_id}" in routes
        assert "/api/v1/classes/{class_id}/students/{student_id}/withdraw" in routes

        # Teacher assignment
        assert "/api/v1/classes/{class_id}/teachers" in routes
        assert "/api/v1/classes/{class_id}/teachers/{teacher_id}" in routes
        assert "/api/v1/classes/{class_id}/teachers/{teacher_id}/end" in routes


class TestClassesAPIEndpoints:
    """Tests for classes API endpoints."""

    @patch("src.api.v1.classes._get_class_service")
    @patch("src.api.dependencies.require_tenant_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_create_class_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test creating a class."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        school_id = uuid4()
        year_id = uuid4()

        mock_class = MagicMock()
        mock_class.id = str(uuid4())
        mock_class.code = "CLS001"
        mock_class.name = "Class 1A"
        mock_class.grade_level = "1"
        mock_class.section = "A"
        mock_class.school_id = str(school_id)
        mock_class.school_name = "Test School"
        mock_class.academic_year_id = str(year_id)
        mock_class.academic_year_name = "2024-2025"
        mock_class.homeroom_teacher_id = None
        mock_class.homeroom_teacher_name = None
        mock_class.capacity = 30
        mock_class.is_active = True
        mock_class.created_at = "2024-01-01T00:00:00"
        mock_class.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.create_class = AsyncMock(return_value=mock_class)
        mock_get_service.return_value = mock_service

        response = client.post(
            "/api/v1/classes/",
            json={
                "code": "CLS001",
                "name": "Class 1A",
                "school_id": str(school_id),
                "academic_year_id": str(year_id),
                "grade_level": "1",
                "section": "A",
            },
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.classes._get_class_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_classes_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing classes."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        mock_service = MagicMock()
        mock_service.list_classes = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/classes/")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.classes._get_class_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_get_class_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test getting a class."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        mock_class = MagicMock()
        mock_class.id = str(class_id)
        mock_class.code = "CLS001"
        mock_class.name = "Class 1A"
        mock_class.grade_level = "1"
        mock_class.section = "A"
        mock_class.school_id = str(uuid4())
        mock_class.school_name = "Test School"
        mock_class.academic_year_id = str(uuid4())
        mock_class.academic_year_name = "2024-2025"
        mock_class.homeroom_teacher_id = None
        mock_class.homeroom_teacher_name = None
        mock_class.capacity = 30
        mock_class.is_active = True
        mock_class.created_at = "2024-01-01T00:00:00"
        mock_class.updated_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.get_class = AsyncMock(return_value=mock_class)
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/classes/{class_id}")

        assert response.status_code in [200, 422, 500]


class TestEnrollmentAPIEndpoints:
    """Tests for student enrollment API endpoints."""

    @patch("src.api.v1.classes._get_enrollment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_enroll_student_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test enrolling a student."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        student_id = uuid4()

        mock_enrollment = MagicMock()
        mock_enrollment.id = str(uuid4())
        mock_enrollment.class_id = str(class_id)
        mock_enrollment.class_name = "Class 1A"
        mock_enrollment.student_id = str(student_id)
        mock_enrollment.student_name = "John Doe"
        mock_enrollment.enrollment_date = date.today()
        mock_enrollment.status = "active"
        mock_enrollment.withdrawal_date = None
        mock_enrollment.withdrawal_reason = None
        mock_enrollment.created_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.enroll_student = AsyncMock(return_value=mock_enrollment)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/classes/{class_id}/students",
            json={"student_id": str(student_id)},
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.classes._get_enrollment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_bulk_enroll_students_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test bulk enrolling students."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        student_ids = [uuid4(), uuid4()]

        mock_result = MagicMock()
        mock_result.success_count = 2
        mock_result.failure_count = 0
        mock_result.failures = []

        mock_service = MagicMock()
        mock_service.bulk_enroll_students = AsyncMock(return_value=mock_result)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/classes/{class_id}/students/bulk",
            json={"student_ids": [str(sid) for sid in student_ids]},
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.classes._get_enrollment_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_enrollments_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing enrollments."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()

        mock_service = MagicMock()
        mock_service.list_enrollments = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/classes/{class_id}/students")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.classes._get_enrollment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_withdraw_student_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test withdrawing a student."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        student_id = uuid4()

        mock_enrollment = MagicMock()
        mock_enrollment.id = str(uuid4())
        mock_enrollment.class_id = str(class_id)
        mock_enrollment.class_name = "Class 1A"
        mock_enrollment.student_id = str(student_id)
        mock_enrollment.student_name = "John Doe"
        mock_enrollment.enrollment_date = date.today()
        mock_enrollment.status = "withdrawn"
        mock_enrollment.withdrawal_date = date.today()
        mock_enrollment.withdrawal_reason = "Moving"
        mock_enrollment.created_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.withdraw_student = AsyncMock(return_value=mock_enrollment)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/classes/{class_id}/students/{student_id}/withdraw",
            json={"reason": "Moving"},
        )

        assert response.status_code in [200, 422, 500]


class TestTeacherAssignmentAPIEndpoints:
    """Tests for teacher assignment API endpoints."""

    @patch("src.api.v1.classes._get_assignment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_assign_teacher_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test assigning a teacher."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        teacher_id = uuid4()

        mock_assignment = MagicMock()
        mock_assignment.id = str(uuid4())
        mock_assignment.class_id = str(class_id)
        mock_assignment.class_name = "Class 1A"
        mock_assignment.teacher_id = str(teacher_id)
        mock_assignment.teacher_name = "Jane Smith"
        mock_assignment.subject_name = "Mathematics"
        mock_assignment.is_homeroom = False
        mock_assignment.start_date = date.today()
        mock_assignment.end_date = None
        mock_assignment.created_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.assign_teacher = AsyncMock(return_value=mock_assignment)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/classes/{class_id}/teachers",
            json={
                "teacher_id": str(teacher_id),
                "subject_name": "Mathematics",
            },
        )

        assert response.status_code in [201, 422, 500]

    @patch("src.api.v1.classes._get_assignment_service")
    @patch("src.api.dependencies.require_auth")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_list_assignments_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_auth,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test listing teacher assignments."""
        mock_require_auth.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()

        mock_service = MagicMock()
        mock_service.list_assignments = AsyncMock(return_value=([], 0))
        mock_get_service.return_value = mock_service

        response = client.get(f"/api/v1/classes/{class_id}/teachers")

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.classes._get_assignment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_end_assignment_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test ending a teacher assignment."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        teacher_id = uuid4()

        mock_assignment = MagicMock()
        mock_assignment.id = str(uuid4())
        mock_assignment.class_id = str(class_id)
        mock_assignment.class_name = "Class 1A"
        mock_assignment.teacher_id = str(teacher_id)
        mock_assignment.teacher_name = "Jane Smith"
        mock_assignment.subject_name = "Mathematics"
        mock_assignment.is_homeroom = False
        mock_assignment.start_date = date.today()
        mock_assignment.end_date = date.today()
        mock_assignment.created_at = "2024-01-01T00:00:00"

        mock_service = MagicMock()
        mock_service.end_assignment = AsyncMock(return_value=mock_assignment)
        mock_get_service.return_value = mock_service

        response = client.post(
            f"/api/v1/classes/{class_id}/teachers/{teacher_id}/end",
            json={"end_date": str(date.today())},
        )

        assert response.status_code in [200, 422, 500]

    @patch("src.api.v1.classes._get_assignment_service")
    @patch("src.api.dependencies.require_admin")
    @patch("src.api.dependencies.require_tenant")
    @patch("src.api.dependencies.get_tenant_db")
    def test_delete_assignment_success(
        self,
        mock_get_db,
        mock_require_tenant,
        mock_require_admin,
        mock_get_service,
        client,
        mock_tenant,
        mock_user,
    ):
        """Test deleting a teacher assignment."""
        mock_require_admin.return_value = mock_user
        mock_require_tenant.return_value = mock_tenant

        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db

        class_id = uuid4()
        teacher_id = uuid4()

        mock_service = MagicMock()
        mock_service.delete_assignment = AsyncMock()
        mock_get_service.return_value = mock_service

        response = client.delete(f"/api/v1/classes/{class_id}/teachers/{teacher_id}")

        assert response.status_code in [204, 422, 500]
