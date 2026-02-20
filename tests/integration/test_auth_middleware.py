# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for authentication and tenant middleware.

Tests the middleware components in isolation from database.
"""

from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.responses import JSONResponse

from src.api.middleware.auth import AuthMiddleware, CurrentUser, get_current_user
from src.api.middleware.tenant import TenantMiddleware, TenantContext, get_tenant_from_request
from src.domains.auth.jwt import JWTManager


@pytest.fixture
def jwt_settings() -> MagicMock:
    """Create mock JWT settings."""
    settings = MagicMock()
    settings.secret_key = SecretStr("test-secret-key-for-jwt-testing")
    settings.algorithm = "HS256"
    settings.access_token_expire_minutes = 30
    settings.refresh_token_expire_days = 7
    return settings


@pytest.fixture
def jwt_manager(jwt_settings: MagicMock) -> JWTManager:
    """Create JWT manager with test settings."""
    return JWTManager(jwt_settings)


@pytest.fixture
def mock_get_central_db() -> AsyncMock:
    """Create mock central database getter."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

    async def context_manager():
        yield mock_db

    return context_manager


class TestAuthMiddleware:
    """Tests for AuthMiddleware."""

    def test_public_path_bypasses_auth(self) -> None:
        """Test that public paths don't require authentication."""
        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200

    @patch("src.api.middleware.auth.get_settings")
    def test_valid_token_sets_user(
        self,
        mock_settings: MagicMock,
        jwt_settings: MagicMock,
        jwt_manager: JWTManager,
    ) -> None:
        """Test that valid token sets request.state.user."""
        mock_settings.return_value.jwt = jwt_settings

        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/api/v1/test")
        async def test_endpoint(request: Request) -> dict:
            user = get_current_user(request)
            return {"user_id": user.id if user else None}

        user_id = str(uuid4())
        token = jwt_manager.create_access_token(
            user_id=user_id,
            user_type="student",
            roles=["student"],
        )

        client = TestClient(app)
        response = client.get(
            "/api/v1/test",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["user_id"] == user_id

    @patch("src.api.middleware.auth.get_settings")
    def test_no_token_sets_user_none(
        self,
        mock_settings: MagicMock,
        jwt_settings: MagicMock,
    ) -> None:
        """Test that missing token sets request.state.user to None."""
        mock_settings.return_value.jwt = jwt_settings

        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/api/v1/test")
        async def test_endpoint(request: Request) -> dict:
            user = get_current_user(request)
            return {"user_id": user.id if user else None}

        client = TestClient(app)
        response = client.get("/api/v1/test")

        assert response.status_code == 200
        assert response.json()["user_id"] is None

    @patch("src.api.middleware.auth.get_settings")
    def test_invalid_token_sets_user_none(
        self,
        mock_settings: MagicMock,
        jwt_settings: MagicMock,
    ) -> None:
        """Test that invalid token sets request.state.user to None."""
        mock_settings.return_value.jwt = jwt_settings

        app = FastAPI()
        app.add_middleware(AuthMiddleware)

        @app.get("/api/v1/test")
        async def test_endpoint(request: Request) -> dict:
            user = get_current_user(request)
            return {"user_id": user.id if user else None}

        client = TestClient(app)
        response = client.get(
            "/api/v1/test",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 200
        assert response.json()["user_id"] is None


class TestCurrentUser:
    """Tests for CurrentUser class."""

    def test_has_role(self, jwt_settings: MagicMock) -> None:
        """Test has_role method."""
        jwt_manager = JWTManager(jwt_settings)
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            roles=["student", "reader"],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.has_role("student") is True
        assert user.has_role("reader") is True
        assert user.has_role("admin") is False

    def test_has_permission(self, jwt_settings: MagicMock) -> None:
        """Test has_permission method."""
        jwt_manager = JWTManager(jwt_settings)
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            permissions=["practice.create", "practice.view"],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.has_permission("practice.create") is True
        assert user.has_permission("practice.view") is True
        assert user.has_permission("users.delete") is False

    def test_has_any_role(self, jwt_settings: MagicMock) -> None:
        """Test has_any_role method."""
        jwt_manager = JWTManager(jwt_settings)
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            roles=["student"],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.has_any_role("student", "teacher") is True
        assert user.has_any_role("admin", "teacher") is False

    def test_has_any_permission(self, jwt_settings: MagicMock) -> None:
        """Test has_any_permission method."""
        jwt_manager = JWTManager(jwt_settings)
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            permissions=["practice.create"],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.has_any_permission("practice.create", "users.view") is True
        assert user.has_any_permission("users.delete", "admin.manage") is False

    def test_has_all_permissions(self, jwt_settings: MagicMock) -> None:
        """Test has_all_permissions method."""
        jwt_manager = JWTManager(jwt_settings)
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            permissions=["practice.create", "practice.view"],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.has_all_permissions("practice.create", "practice.view") is True
        assert user.has_all_permissions("practice.create", "users.delete") is False

    def test_can_access_school(self, jwt_settings: MagicMock) -> None:
        """Test can_access_school method."""
        jwt_manager = JWTManager(jwt_settings)
        school_id = str(uuid4())
        other_school = str(uuid4())

        # Regular user can only access their schools
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            user_type="teacher",
            school_ids=[school_id],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.can_access_school(school_id) is True
        assert user.can_access_school(other_school) is False

    def test_tenant_admin_can_access_any_school(self, jwt_settings: MagicMock) -> None:
        """Test that tenant admin can access any school."""
        jwt_manager = JWTManager(jwt_settings)
        any_school = str(uuid4())

        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            user_type="tenant_admin",
            school_ids=[],
        )
        payload = jwt_manager.decode_token(token)
        user = CurrentUser(payload)

        assert user.can_access_school(any_school) is True

    def test_is_admin_property(self, jwt_settings: MagicMock) -> None:
        """Test is_admin property."""
        jwt_manager = JWTManager(jwt_settings)

        # Tenant admin
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            user_type="tenant_admin",
        )
        user = CurrentUser(jwt_manager.decode_token(token))
        assert user.is_admin is True

        # School admin
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            user_type="school_admin",
        )
        user = CurrentUser(jwt_manager.decode_token(token))
        assert user.is_admin is True

        # Student is not admin
        token = jwt_manager.create_access_token(
            user_id=str(uuid4()),
            user_type="student",
        )
        user = CurrentUser(jwt_manager.decode_token(token))
        assert user.is_admin is False


class TestTenantMiddleware:
    """Tests for TenantMiddleware."""

    def test_public_path_bypasses_tenant(
        self,
        mock_get_central_db: AsyncMock,
    ) -> None:
        """Test that public paths don't require tenant."""
        app = FastAPI()
        app.add_middleware(
            TenantMiddleware,
            get_central_db=mock_get_central_db,
            base_domain="edusynapse.com",
        )

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200

    def test_tenant_header_extraction(
        self,
        mock_get_central_db: AsyncMock,
    ) -> None:
        """Test that X-Tenant-Code header is extracted."""
        app = FastAPI()
        app.add_middleware(
            TenantMiddleware,
            get_central_db=mock_get_central_db,
            base_domain="edusynapse.com",
        )

        @app.get("/api/v1/test")
        async def test_endpoint(request: Request) -> dict:
            tenant = get_tenant_from_request(request)
            return {"tenant_code": tenant.code if tenant else None}

        client = TestClient(app)
        response = client.get(
            "/api/v1/test",
            headers={"X-Tenant-Code": "test_tenant"},
        )

        assert response.status_code == 200
        # Tenant is None because mock returns None
        assert response.json()["tenant_code"] is None


class TestTenantContext:
    """Tests for TenantContext class."""

    def test_tenant_context_properties(self) -> None:
        """Test TenantContext properties."""
        tenant = TenantContext(
            tenant_id=str(uuid4()),
            code="test_tenant",
            name="Test Tenant",
            status="active",
            tier="standard",
        )

        assert tenant.is_active is True

        inactive_tenant = TenantContext(
            tenant_id=str(uuid4()),
            code="inactive_tenant",
            name="Inactive Tenant",
            status="suspended",
            tier="standard",
        )

        assert inactive_tenant.is_active is False
