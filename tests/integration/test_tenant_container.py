# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for tenant container management.

These tests require Docker to be running and accessible.
Run with: pytest tests/integration/test_tenant_container.py -v

Prerequisites:
    - Docker daemon running
    - Docker network 'edusynapse-network' exists
    - postgres:16-alpine image available
"""

import uuid

import pytest

from src.core.config.settings import Settings, clear_settings_cache
from src.infrastructure.database.tenant_manager import (
    TenantConnectionInfo,
    TenantDatabaseManager,
    TenantNotFoundError,
    TenantProvisioningError,
)
from src.infrastructure.docker.tenant_container import (
    ContainerStatus,
    TenantContainerInfo,
    TenantContainerManager,
)


@pytest.fixture
def settings() -> Settings:
    """Provide fresh settings for each test."""
    clear_settings_cache()
    return Settings()


@pytest.fixture
def container_manager() -> TenantContainerManager:
    """Provide a TenantContainerManager instance."""
    return TenantContainerManager()


@pytest.fixture
def tenant_manager(settings: Settings) -> TenantDatabaseManager:
    """Provide a TenantDatabaseManager instance."""
    return TenantDatabaseManager(settings)


def generate_test_tenant_code() -> str:
    """Generate a unique tenant code for testing."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
@pytest.mark.slow
class TestTenantContainerManager:
    """Tests for TenantContainerManager."""

    async def test_create_tenant_container(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test creating a new tenant container."""
        tenant_code = generate_test_tenant_code()

        try:
            info = container_manager.create_tenant_container(tenant_code)

            assert isinstance(info, TenantContainerInfo)
            assert info.tenant_code == tenant_code
            assert info.status == ContainerStatus.RUNNING
            assert info.database_name == f"edusynapse_tenant_{tenant_code}"
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_create_duplicate_container_raises_error(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test that creating a duplicate container raises error."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            with pytest.raises(ValueError) as exc_info:
                container_manager.create_tenant_container(tenant_code)

            assert "already exists" in str(exc_info.value)
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_get_tenant_container_status(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test getting container status."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            status = container_manager.get_tenant_container_status(tenant_code)
            assert status == ContainerStatus.RUNNING
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_get_nonexistent_container_status(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test getting status of non-existent container."""
        status = container_manager.get_tenant_container_status("nonexistent_12345")
        assert status == ContainerStatus.NOT_FOUND

    async def test_stop_and_start_container(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test stopping and starting a container."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            # Stop
            container_manager.stop_tenant_container(tenant_code)
            status = container_manager.get_tenant_container_status(tenant_code)
            assert status == ContainerStatus.EXITED

            # Start
            container_manager.start_tenant_container(tenant_code)
            status = container_manager.get_tenant_container_status(tenant_code)
            assert status == ContainerStatus.RUNNING
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_get_tenant_connection_url(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test getting connection URL for a tenant."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            # Internal URL (uses container name)
            internal_url = container_manager.get_tenant_connection_url(
                tenant_code, internal=True
            )
            assert "asyncpg://" in internal_url
            assert f"edusynapse-tenant-{tenant_code}-db" in internal_url

            # External URL (uses localhost)
            external_url = container_manager.get_tenant_connection_url(
                tenant_code, internal=False
            )
            assert "asyncpg://" in external_url
            assert "localhost" in external_url
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_wait_for_healthy(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test waiting for container to become healthy."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            healthy = container_manager.wait_for_healthy(tenant_code, timeout=120)
            assert healthy is True
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_list_tenant_containers(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test listing all tenant containers."""
        tenant_code = generate_test_tenant_code()

        try:
            container_manager.create_tenant_container(tenant_code)

            containers = container_manager.list_tenant_containers()

            # Should find at least our container
            our_container = next(
                (c for c in containers if c.tenant_code == tenant_code), None
            )
            assert our_container is not None
            assert our_container.status == ContainerStatus.RUNNING
        finally:
            container_manager.remove_tenant_container(tenant_code, remove_volume=True)

    async def test_invalid_tenant_code_raises_error(
        self, container_manager: TenantContainerManager
    ) -> None:
        """Test that invalid tenant codes raise error."""
        with pytest.raises(ValueError) as exc_info:
            container_manager.create_tenant_container("invalid-code-with-dashes")

        assert "Invalid tenant_code" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.slow
class TestTenantDatabaseManager:
    """Tests for TenantDatabaseManager."""

    async def test_provision_tenant(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test provisioning a new tenant database."""
        tenant_code = generate_test_tenant_code()

        try:
            info = await tenant_manager.provision_tenant(tenant_code)

            assert isinstance(info, TenantConnectionInfo)
            assert info.tenant_code == tenant_code
            assert info.is_healthy is True
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_get_session(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test getting a database session for a tenant."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            from sqlalchemy import text

            async with tenant_manager.get_session(tenant_code) as session:
                result = await session.execute(text("SELECT 1 as value"))
                row = result.fetchone()
                assert row is not None
                assert row.value == 1
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_check_connection(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test checking connection to tenant database."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            connected = await tenant_manager.check_connection(tenant_code)
            assert connected is True
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_get_connection_info(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test getting connection info for a tenant."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            info = tenant_manager.get_connection_info(tenant_code)

            assert info.tenant_code == tenant_code
            assert info.database_name == f"edusynapse_tenant_{tenant_code}"
            assert info.is_healthy is True
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_get_session_for_nonexistent_tenant_raises_error(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test that getting session for non-existent tenant raises error."""
        with pytest.raises(TenantNotFoundError):
            async with tenant_manager.get_session("nonexistent_12345"):
                pass

    async def test_provision_duplicate_tenant_raises_error(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test that provisioning a duplicate tenant raises error."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            with pytest.raises(TenantProvisioningError) as exc_info:
                await tenant_manager.provision_tenant(tenant_code)

            assert "already exists" in str(exc_info.value)
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_deprovision_tenant(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test deprovisioning a tenant."""
        tenant_code = generate_test_tenant_code()

        await tenant_manager.provision_tenant(tenant_code)
        await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

        # Should not be accessible anymore
        with pytest.raises(TenantNotFoundError):
            tenant_manager.get_connection_info(tenant_code)

    async def test_list_tenants(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test listing all tenants."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            tenants = tenant_manager.list_tenants()

            # Should find at least our tenant
            our_tenant = next(
                (t for t in tenants if t.tenant_code == tenant_code), None
            )
            assert our_tenant is not None
        finally:
            await tenant_manager.deprovision_tenant(tenant_code, remove_data=True)

    async def test_close_all(
        self, tenant_manager: TenantDatabaseManager
    ) -> None:
        """Test closing all connections."""
        tenant_code = generate_test_tenant_code()

        try:
            await tenant_manager.provision_tenant(tenant_code)

            # Get a session to create a connection
            async with tenant_manager.get_session(tenant_code) as session:
                from sqlalchemy import text

                await session.execute(text("SELECT 1"))

            # Close all connections
            await tenant_manager.close_all()

            # Engine cache should be cleared
            assert tenant_code not in tenant_manager._engines
        finally:
            # Cleanup container directly since manager state is cleared
            container_manager = TenantContainerManager()
            try:
                container_manager.remove_tenant_container(
                    tenant_code, remove_volume=True
                )
            except ValueError:
                pass  # Already removed
