# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Docker SDK tenant container lifecycle management.

This module provides functionality to create, start, stop, and remove
PostgreSQL containers for each tenant. Each tenant gets its own isolated
database container for complete data isolation.

Container naming convention: edusynapse-tenant-{tenant_code}-db
Network: edusynapse-network (shared with other services)
Port allocation: Starting from TENANT_DB_PORT_RANGE_START (default 5500)
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import docker
from docker.errors import APIError, NotFound
from docker.models.containers import Container


class ContainerStatus(str, Enum):
    """Status of a tenant container."""

    RUNNING = "running"
    STOPPED = "stopped"
    CREATED = "created"
    RESTARTING = "restarting"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"
    NOT_FOUND = "not_found"


@dataclass
class TenantContainerInfo:
    """Information about a tenant's PostgreSQL container."""

    tenant_code: str
    container_name: str
    container_id: str
    status: ContainerStatus
    host_port: int
    internal_host: str
    database_name: str
    username: str


class TenantContainerManager:
    """Manages tenant PostgreSQL containers using Docker SDK.

    Each tenant gets its own PostgreSQL container for complete data isolation.
    Containers are created on the shared edusynapse-network so they can
    communicate with the API and worker services.

    Example:
        manager = TenantContainerManager()

        # Create and start a new tenant container
        info = manager.create_tenant_container("acme")
        manager.wait_for_healthy("acme", timeout=60)

        # Get connection URL for the tenant
        url = manager.get_tenant_connection_url("acme")

        # Stop and remove when tenant is deleted
        manager.stop_tenant_container("acme")
        manager.remove_tenant_container("acme")
    """

    CONTAINER_PREFIX = "edusynapse-tenant-"
    CONTAINER_SUFFIX = "-db"
    NETWORK_NAME = "edusynapse-network"
    POSTGRES_IMAGE = "postgres:16-alpine"
    DEFAULT_PORT_RANGE_START = 44000

    def __init__(
        self,
        docker_client: Optional[docker.DockerClient] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        port_range_start: Optional[int] = None,
        init_script_host_path: Optional[str] = None,
    ) -> None:
        """Initialize the tenant container manager.

        Args:
            docker_client: Docker client instance. If None, creates from environment.
            db_user: PostgreSQL user for tenant databases. Defaults to TENANT_DB_USER env.
            db_password: PostgreSQL password. Defaults to TENANT_DB_PASSWORD env.
            port_range_start: Starting port for tenant containers. Defaults to TENANT_DB_PORT_RANGE_START env.
            init_script_host_path: Host path to the tenant init SQL script. Defaults to TENANT_INIT_SCRIPT_HOST_PATH env.
        """
        self._client = docker_client or docker.from_env()
        self._db_user = db_user or os.getenv("TENANT_DB_USER", "edusynapse")
        self._db_password = db_password or os.getenv(
            "TENANT_DB_PASSWORD", "edusynapse_tenant_password"
        )
        self._port_range_start = port_range_start or int(
            os.getenv("TENANT_DB_PORT_RANGE_START", str(self.DEFAULT_PORT_RANGE_START))
        )
        self._init_script_host_path = init_script_host_path or os.getenv(
            "TENANT_INIT_SCRIPT_HOST_PATH"
        )
        self._allocated_ports: dict[str, int] = {}
        self._load_existing_containers()

    def _load_existing_containers(self) -> None:
        """Load existing tenant containers and their port allocations."""
        try:
            containers = self._client.containers.list(
                all=True, filters={"name": self.CONTAINER_PREFIX}
            )
            for container in containers:
                tenant_code = self._extract_tenant_code(container.name)
                if tenant_code:
                    port = self._get_container_host_port(container)
                    if port:
                        self._allocated_ports[tenant_code] = port
        except APIError:
            pass

    def _extract_tenant_code(self, container_name: str) -> Optional[str]:
        """Extract tenant code from container name."""
        if container_name.startswith(
            self.CONTAINER_PREFIX
        ) and container_name.endswith(self.CONTAINER_SUFFIX):
            return container_name[
                len(self.CONTAINER_PREFIX) : -len(self.CONTAINER_SUFFIX)
            ]
        return None

    def _get_container_name(self, tenant_code: str) -> str:
        """Get container name for a tenant."""
        return f"{self.CONTAINER_PREFIX}{tenant_code}{self.CONTAINER_SUFFIX}"

    def _get_database_name(self, tenant_code: str) -> str:
        """Get database name for a tenant."""
        return f"edusynapse_tenant_{tenant_code}"

    def _allocate_port(self, tenant_code: str) -> int:
        """Allocate a host port for a tenant container."""
        if tenant_code in self._allocated_ports:
            return self._allocated_ports[tenant_code]

        used_ports = set(self._allocated_ports.values())
        port = self._port_range_start
        while port in used_ports:
            port += 1

        self._allocated_ports[tenant_code] = port
        return port

    def _get_container_host_port(self, container: Container) -> Optional[int]:
        """Get the host port mapping for a container."""
        try:
            ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            port_mapping = ports.get("5432/tcp")
            if port_mapping and len(port_mapping) > 0:
                return int(port_mapping[0].get("HostPort", 0))
        except (KeyError, IndexError, ValueError, TypeError):
            pass
        return None

    def _get_container(self, tenant_code: str) -> Optional[Container]:
        """Get container instance for a tenant."""
        container_name = self._get_container_name(tenant_code)
        try:
            return self._client.containers.get(container_name)
        except NotFound:
            return None

    def create_tenant_container(self, tenant_code: str) -> TenantContainerInfo:
        """Create a new PostgreSQL container for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant (alphanumeric and underscores only).

        Returns:
            TenantContainerInfo with container details.

        Raises:
            ValueError: If tenant_code is invalid or container already exists.
            docker.errors.APIError: If Docker operation fails.
        """
        if not tenant_code or not tenant_code.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid tenant_code: {tenant_code}. Must be alphanumeric with optional underscores."
            )

        container_name = self._get_container_name(tenant_code)
        database_name = self._get_database_name(tenant_code)

        existing = self._get_container(tenant_code)
        if existing:
            raise ValueError(
                f"Container already exists for tenant: {tenant_code}"
            )

        host_port = self._allocate_port(tenant_code)

        volumes = {
            f"edusynapse-tenant-{tenant_code}-data": {
                "bind": "/var/lib/postgresql/data",
                "mode": "rw",
            }
        }

        if self._init_script_host_path:
            volumes[self._init_script_host_path] = {
                "bind": "/docker-entrypoint-initdb.d/01-init.sql",
                "mode": "ro",
            }

        container = self._client.containers.create(
            image=self.POSTGRES_IMAGE,
            name=container_name,
            detach=True,
            environment={
                "POSTGRES_USER": self._db_user,
                "POSTGRES_PASSWORD": self._db_password,
                "POSTGRES_DB": database_name,
            },
            ports={"5432/tcp": host_port},
            volumes=volumes,
            network=self.NETWORK_NAME,
            restart_policy={"Name": "unless-stopped"},
            healthcheck={
                "test": [
                    "CMD-SHELL",
                    f"pg_isready -U {self._db_user} -d {database_name}",
                ],
                "interval": 10_000_000_000,
                "timeout": 5_000_000_000,
                "retries": 5,
                "start_period": 30_000_000_000,
            },
            labels={
                "edusynapse.tenant": tenant_code,
                "edusynapse.service": "tenant-db",
            },
        )

        container.start()

        return TenantContainerInfo(
            tenant_code=tenant_code,
            container_name=container_name,
            container_id=container.id,
            status=ContainerStatus.RUNNING,
            host_port=host_port,
            internal_host=container_name,
            database_name=database_name,
            username=self._db_user,
        )

    def start_tenant_container(self, tenant_code: str) -> None:
        """Start a stopped tenant container.

        Args:
            tenant_code: Unique identifier for the tenant.

        Raises:
            ValueError: If container does not exist.
            docker.errors.APIError: If Docker operation fails.
        """
        container = self._get_container(tenant_code)
        if not container:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        container.start()

    def stop_tenant_container(self, tenant_code: str, timeout: int = 10) -> None:
        """Stop a running tenant container.

        Args:
            tenant_code: Unique identifier for the tenant.
            timeout: Seconds to wait before killing the container.

        Raises:
            ValueError: If container does not exist.
            docker.errors.APIError: If Docker operation fails.
        """
        container = self._get_container(tenant_code)
        if not container:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        container.stop(timeout=timeout)

    def remove_tenant_container(
        self, tenant_code: str, remove_volume: bool = False
    ) -> None:
        """Remove a tenant container.

        Args:
            tenant_code: Unique identifier for the tenant.
            remove_volume: If True, also removes the data volume.

        Raises:
            ValueError: If container does not exist.
            docker.errors.APIError: If Docker operation fails.
        """
        container = self._get_container(tenant_code)
        if not container:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        try:
            container.stop(timeout=5)
        except APIError:
            pass

        container.remove(force=True)

        if tenant_code in self._allocated_ports:
            del self._allocated_ports[tenant_code]

        if remove_volume:
            volume_name = f"edusynapse-tenant-{tenant_code}-data"
            try:
                volume = self._client.volumes.get(volume_name)
                volume.remove()
            except NotFound:
                pass

    def get_tenant_container_status(self, tenant_code: str) -> ContainerStatus:
        """Get the status of a tenant container.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            ContainerStatus enum value.
        """
        container = self._get_container(tenant_code)
        if not container:
            return ContainerStatus.NOT_FOUND

        container.reload()
        status = container.status.lower()

        status_map = {
            "running": ContainerStatus.RUNNING,
            "exited": ContainerStatus.EXITED,
            "created": ContainerStatus.CREATED,
            "restarting": ContainerStatus.RESTARTING,
            "paused": ContainerStatus.PAUSED,
            "dead": ContainerStatus.DEAD,
        }

        return status_map.get(status, ContainerStatus.STOPPED)

    def get_tenant_container_info(
        self, tenant_code: str
    ) -> Optional[TenantContainerInfo]:
        """Get full information about a tenant container.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            TenantContainerInfo if container exists, None otherwise.
        """
        container = self._get_container(tenant_code)
        if not container:
            return None

        container.reload()
        host_port = self._get_container_host_port(container)
        if not host_port:
            host_port = self._allocated_ports.get(tenant_code, 0)

        status = self.get_tenant_container_status(tenant_code)

        return TenantContainerInfo(
            tenant_code=tenant_code,
            container_name=container.name,
            container_id=container.id,
            status=status,
            host_port=host_port,
            internal_host=container.name,
            database_name=self._get_database_name(tenant_code),
            username=self._db_user,
        )

    def get_tenant_connection_url(
        self, tenant_code: str, internal: bool = True
    ) -> str:
        """Get the async PostgreSQL connection URL for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.
            internal: If True, returns URL for internal Docker network.
                     If False, returns URL for external access via host port.

        Returns:
            PostgreSQL async connection URL string (asyncpg driver).

        Raises:
            ValueError: If container does not exist.
        """
        info = self.get_tenant_container_info(tenant_code)
        if not info:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        if internal:
            host = info.internal_host
            port = 5432
        else:
            host = "localhost"
            port = info.host_port

        return (
            f"postgresql+asyncpg://{info.username}:{self._db_password}"
            f"@{host}:{port}/{info.database_name}"
        )

    def get_tenant_connection_url_sync(
        self, tenant_code: str, internal: bool = True
    ) -> str:
        """Get the sync PostgreSQL connection URL for a tenant.

        This returns a psycopg2-based URL for synchronous database operations.
        Used by LangGraph workflow nodes that run in thread pool executors
        where async greenlet context is not available.

        Args:
            tenant_code: Unique identifier for the tenant.
            internal: If True, returns URL for internal Docker network.
                     If False, returns URL for external access via host port.

        Returns:
            PostgreSQL sync connection URL string (psycopg2 driver).

        Raises:
            ValueError: If container does not exist.
        """
        info = self.get_tenant_container_info(tenant_code)
        if not info:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        if internal:
            host = info.internal_host
            port = 5432
        else:
            host = "localhost"
            port = info.host_port

        return (
            f"postgresql+psycopg2://{info.username}:{self._db_password}"
            f"@{host}:{port}/{info.database_name}"
        )

    def wait_for_healthy(
        self, tenant_code: str, timeout: int = 60, interval: int = 2
    ) -> bool:
        """Wait for a tenant container to become healthy.

        Args:
            tenant_code: Unique identifier for the tenant.
            timeout: Maximum seconds to wait.
            interval: Seconds between health checks.

        Returns:
            True if container became healthy, False if timeout.

        Raises:
            ValueError: If container does not exist.
        """
        container = self._get_container(tenant_code)
        if not container:
            raise ValueError(f"Container not found for tenant: {tenant_code}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            container.reload()
            health = container.attrs.get("State", {}).get("Health", {})
            status = health.get("Status", "")

            if status == "healthy":
                return True

            if status == "unhealthy":
                return False

            time.sleep(interval)

        return False

    def list_tenant_containers(self) -> list[TenantContainerInfo]:
        """List all tenant containers.

        Returns:
            List of TenantContainerInfo for all tenant containers.
        """
        containers = self._client.containers.list(
            all=True, filters={"name": self.CONTAINER_PREFIX}
        )

        result = []
        for container in containers:
            tenant_code = self._extract_tenant_code(container.name)
            if tenant_code:
                info = self.get_tenant_container_info(tenant_code)
                if info:
                    result.append(info)

        return result
