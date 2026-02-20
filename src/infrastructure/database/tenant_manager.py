# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant database connection management.

This module provides database connections for tenant databases.
Each tenant has its own isolated PostgreSQL database running in a
Docker container managed by TenantContainerManager.

Connection pools are lazily created and cached per tenant.

Two types of connections are supported:
1. Async (asyncpg): For FastAPI request handlers and async code
2. Sync (psycopg2): For LangGraph workflow nodes that run in thread executors

Example:
    from src.infrastructure.database import TenantDatabaseManager

    # Create manager
    manager = TenantDatabaseManager(settings)

    # Get async session for a tenant (API handlers)
    async with manager.get_session("acme") as session:
        result = await session.execute(select(User))
        users = result.scalars().all()

    # Get sync session for a tenant (LangGraph workflow nodes)
    with manager.get_sync_session("acme") as session:
        result = session.execute(select(User))
        users = result.scalars().all()

    # Provision a new tenant
    info = await manager.provision_tenant("new_tenant")

    # Cleanup on shutdown
    await manager.close_all()
"""

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.infrastructure.docker import (
    ContainerStatus,
    TenantContainerInfo,
    TenantContainerManager,
)

if TYPE_CHECKING:
    from src.core.config.settings import Settings


class TenantNotFoundError(Exception):
    """Raised when a tenant database is not found or not accessible.

    Attributes:
        tenant_code: The tenant code that was not found.
    """

    def __init__(self, tenant_code: str) -> None:
        """Initialize the error.

        Args:
            tenant_code: The tenant code that was not found.
        """
        super().__init__(f"Tenant not found or not accessible: {tenant_code}")
        self.tenant_code = tenant_code


class TenantProvisioningError(Exception):
    """Raised when tenant database provisioning fails.

    Attributes:
        tenant_code: The tenant code that failed to provision.
        reason: The reason for the failure.
    """

    def __init__(self, tenant_code: str, reason: str) -> None:
        """Initialize the error.

        Args:
            tenant_code: The tenant code that failed to provision.
            reason: The reason for the failure.
        """
        super().__init__(f"Failed to provision tenant {tenant_code}: {reason}")
        self.tenant_code = tenant_code
        self.reason = reason


@dataclass
class TenantConnectionInfo:
    """Information about a tenant database connection.

    Attributes:
        tenant_code: Unique identifier for the tenant.
        database_name: Name of the PostgreSQL database.
        host: Database host (container name for internal, localhost for external).
        port: Database port.
        username: Database username.
        is_healthy: Whether the database is reachable.
    """

    tenant_code: str
    database_name: str
    host: str
    port: int
    username: str
    is_healthy: bool


class TenantDatabaseManager:
    """Manages database connections for multiple tenants.

    This class provides lazy connection pooling for tenant databases.
    Each tenant gets its own async engine and session maker, created
    on first access and cached for subsequent requests.

    The underlying PostgreSQL containers are managed by TenantContainerManager.

    Attributes:
        settings: Application settings containing database configuration.

    Example:
        manager = TenantDatabaseManager(settings)

        # Get session for tenant
        async with manager.get_session("acme") as session:
            await session.execute(...)

        # Provision new tenant
        info = await manager.provision_tenant("new_tenant")
    """

    def __init__(
        self,
        settings: "Settings",
        container_manager: TenantContainerManager | None = None,
    ) -> None:
        """Initialize the tenant database manager.

        Args:
            settings: Application settings containing database configuration.
            container_manager: Optional TenantContainerManager instance.
                If not provided, creates a new one.
        """
        self._settings = settings
        self._container_manager = container_manager or TenantContainerManager(
            db_user=settings.tenant_db.user,
            db_password=settings.tenant_db.password.get_secret_value(),
            port_range_start=settings.tenant_db.port_range_start,
        )
        # Async engines and sessionmakers (asyncpg driver)
        self._engines: dict[str, AsyncEngine] = {}
        self._sessionmakers: dict[str, async_sessionmaker[AsyncSession]] = {}
        # Sync engines and sessionmakers (psycopg2 driver)
        # Used by LangGraph workflow nodes that run in thread pool executors
        self._sync_engines: dict[str, Engine] = {}
        self._sync_sessionmakers: dict[str, sessionmaker[Session]] = {}

    def _get_connection_url(self, tenant_code: str, internal: bool = True) -> str:
        """Get the async database connection URL for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.
            internal: If True, use Docker internal networking.
                If False, use localhost with host port.

        Returns:
            PostgreSQL async connection URL (asyncpg driver).

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        try:
            return self._container_manager.get_tenant_connection_url(
                tenant_code, internal=internal
            )
        except ValueError as e:
            raise TenantNotFoundError(tenant_code) from e

    def _get_sync_connection_url(self, tenant_code: str, internal: bool = True) -> str:
        """Get the sync database connection URL for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.
            internal: If True, use Docker internal networking.
                If False, use localhost with host port.

        Returns:
            PostgreSQL sync connection URL (psycopg2 driver).

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        try:
            return self._container_manager.get_tenant_connection_url_sync(
                tenant_code, internal=internal
            )
        except ValueError as e:
            raise TenantNotFoundError(tenant_code) from e

    def _get_or_create_engine(self, tenant_code: str) -> AsyncEngine:
        """Get or create an async engine for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            AsyncEngine for the tenant database.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        if tenant_code not in self._engines:
            url = self._get_connection_url(tenant_code)

            self._engines[tenant_code] = create_async_engine(
                url,
                pool_size=self._settings.tenant_db.pool_size,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=1800,
                echo=self._settings.debug,
            )

        return self._engines[tenant_code]

    def _get_or_create_sessionmaker(
        self, tenant_code: str
    ) -> async_sessionmaker[AsyncSession]:
        """Get or create a sessionmaker for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            async_sessionmaker for the tenant database.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        if tenant_code not in self._sessionmakers:
            engine = self._get_or_create_engine(tenant_code)

            self._sessionmakers[tenant_code] = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

        return self._sessionmakers[tenant_code]

    def _get_or_create_sync_engine(self, tenant_code: str) -> Engine:
        """Get or create a sync engine for a tenant.

        This provides a synchronous database engine using psycopg2 driver.
        Used by LangGraph workflow nodes that run in thread pool executors
        where async greenlet context is not available.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            Engine for synchronous database operations.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        if tenant_code not in self._sync_engines:
            url = self._get_sync_connection_url(tenant_code)

            self._sync_engines[tenant_code] = create_engine(
                url,
                pool_size=self._settings.tenant_db.pool_size,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=1800,
                echo=self._settings.debug,
            )

        return self._sync_engines[tenant_code]

    def _get_or_create_sync_sessionmaker(
        self, tenant_code: str
    ) -> sessionmaker[Session]:
        """Get or create a sync sessionmaker for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            sessionmaker for synchronous database operations.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        if tenant_code not in self._sync_sessionmakers:
            engine = self._get_or_create_sync_engine(tenant_code)

            self._sync_sessionmakers[tenant_code] = sessionmaker(
                bind=engine,
                class_=Session,
                expire_on_commit=False,
                autoflush=False,
            )

        return self._sync_sessionmakers[tenant_code]

    @asynccontextmanager
    async def get_session(self, tenant_code: str) -> AsyncIterator[AsyncSession]:
        """Get an async session for a tenant database.

        This is the primary way to interact with tenant databases in async code.
        The session is automatically committed on success and rolled back
        on exception.

        Args:
            tenant_code: Unique identifier for the tenant.

        Yields:
            AsyncSession for database operations.

        Raises:
            TenantNotFoundError: If the tenant database does not exist.
            SQLAlchemyError: If a database operation fails.

        Example:
            async with manager.get_session("acme") as session:
                result = await session.execute(select(User))
                users = result.scalars().all()
        """
        sessionmaker = self._get_or_create_sessionmaker(tenant_code)

        async with sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except SQLAlchemyError:
                await session.rollback()
                raise
            except Exception:
                await session.rollback()
                raise

    @contextmanager
    def get_sync_session(self, tenant_code: str) -> Iterator[Session]:
        """Get a sync session for a tenant database.

        This provides synchronous database access for code that runs
        in thread pool executors (e.g., LangGraph workflow nodes)
        where async greenlet context is not available.

        The session is automatically committed on success and rolled back
        on exception.

        Args:
            tenant_code: Unique identifier for the tenant.

        Yields:
            Session for synchronous database operations.

        Raises:
            TenantNotFoundError: If the tenant database does not exist.
            SQLAlchemyError: If a database operation fails.

        Example:
            # In a LangGraph workflow node (runs in thread pool):
            with manager.get_sync_session("acme") as session:
                result = session.execute(select(User))
                users = result.scalars().all()
        """
        session_factory = self._get_or_create_sync_sessionmaker(tenant_code)
        session = session_factory()

        try:
            yield session
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_engine(self, tenant_code: str) -> AsyncEngine:
        """Get the async engine for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            AsyncEngine for the tenant database.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        return self._get_or_create_engine(tenant_code)

    async def check_connection(self, tenant_code: str) -> bool:
        """Check if a tenant database is reachable.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            True if the database is reachable, False otherwise.
        """
        try:
            engine = self._get_or_create_engine(tenant_code)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except (SQLAlchemyError, TenantNotFoundError):
            return False

    def get_connection_info(self, tenant_code: str) -> TenantConnectionInfo:
        """Get connection information for a tenant.

        Args:
            tenant_code: Unique identifier for the tenant.

        Returns:
            TenantConnectionInfo with connection details.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        container_info = self._container_manager.get_tenant_container_info(tenant_code)
        if container_info is None:
            raise TenantNotFoundError(tenant_code)

        return TenantConnectionInfo(
            tenant_code=tenant_code,
            database_name=container_info.database_name,
            host=container_info.internal_host,
            port=5432,
            username=container_info.username,
            is_healthy=container_info.status == ContainerStatus.RUNNING,
        )

    async def provision_tenant(
        self, tenant_code: str, wait_healthy: bool = True, timeout: int = 60
    ) -> TenantConnectionInfo:
        """Provision a new tenant database.

        Creates a new PostgreSQL container for the tenant using
        TenantContainerManager.

        Args:
            tenant_code: Unique identifier for the tenant.
            wait_healthy: If True, wait for the container to become healthy.
            timeout: Maximum seconds to wait for healthy status.

        Returns:
            TenantConnectionInfo with the new database details.

        Raises:
            TenantProvisioningError: If provisioning fails.
        """
        try:
            container_info = self._container_manager.create_tenant_container(tenant_code)

            if wait_healthy:
                healthy = self._container_manager.wait_for_healthy(
                    tenant_code, timeout=timeout
                )
                if not healthy:
                    raise TenantProvisioningError(
                        tenant_code, "Container did not become healthy in time"
                    )

            return TenantConnectionInfo(
                tenant_code=tenant_code,
                database_name=container_info.database_name,
                host=container_info.internal_host,
                port=5432,
                username=container_info.username,
                is_healthy=True,
            )
        except ValueError as e:
            raise TenantProvisioningError(tenant_code, str(e)) from e

    async def deprovision_tenant(
        self, tenant_code: str, remove_data: bool = False
    ) -> None:
        """Deprovision a tenant database.

        Stops and removes the PostgreSQL container for the tenant.
        Optionally removes the data volume.

        Args:
            tenant_code: Unique identifier for the tenant.
            remove_data: If True, also remove the data volume.

        Raises:
            TenantNotFoundError: If the tenant container does not exist.
        """
        # Close and remove cached connections
        await self._close_tenant_connections(tenant_code)

        try:
            self._container_manager.remove_tenant_container(
                tenant_code, remove_volume=remove_data
            )
        except ValueError as e:
            raise TenantNotFoundError(tenant_code) from e

    async def _close_tenant_connections(self, tenant_code: str) -> None:
        """Close connections for a specific tenant.

        Closes both async and sync engines/sessions for the tenant.

        Args:
            tenant_code: Unique identifier for the tenant.
        """
        # Close async connections
        if tenant_code in self._engines:
            await self._engines[tenant_code].dispose()
            del self._engines[tenant_code]

        if tenant_code in self._sessionmakers:
            del self._sessionmakers[tenant_code]

        # Close sync connections
        if tenant_code in self._sync_engines:
            self._sync_engines[tenant_code].dispose()
            del self._sync_engines[tenant_code]

        if tenant_code in self._sync_sessionmakers:
            del self._sync_sessionmakers[tenant_code]

    async def close_all(self) -> None:
        """Close all tenant database connections.

        This should be called at application shutdown to properly
        close all connection pools (both async and sync).
        """
        # Close async engines
        for tenant_code in list(self._engines.keys()):
            await self._close_tenant_connections(tenant_code)

        # Close any remaining sync engines (if async wasn't created for them)
        for tenant_code in list(self._sync_engines.keys()):
            self._sync_engines[tenant_code].dispose()
            del self._sync_engines[tenant_code]

        if self._sync_sessionmakers:
            self._sync_sessionmakers.clear()

    def list_tenants(self) -> list[TenantContainerInfo]:
        """List all tenant containers.

        Returns:
            List of TenantContainerInfo for all tenant containers.
        """
        return self._container_manager.list_tenant_containers()


# =============================================================================
# WORKER THREAD-LOCAL MANAGER
# =============================================================================

import threading

# Thread-local storage for worker DB managers
# Each Dramatiq worker thread gets its own manager instance
_thread_local_manager = threading.local()


def get_worker_db_manager() -> TenantDatabaseManager:
    """Get TenantDatabaseManager instance for current worker thread.

    Each Dramatiq worker thread gets its own TenantDatabaseManager instance
    with its own engine and sessionmaker caches. This ensures async engines
    remain bound to the correct thread's event loop.

    Why thread-local?
    - Dramatiq uses multiple threads (--threads N) per process
    - SQLAlchemy async engines are bound to the event loop they're created in
    - Each thread has its own persistent event loop (see base.py)
    - Sharing engines across threads would cause "attached to different loop" errors

    Returns:
        Thread-local TenantDatabaseManager instance.

    Example:
        @dramatiq.actor
        def my_task(tenant_code: str):
            db_manager = get_worker_db_manager()
            async with db_manager.get_session(tenant_code) as session:
                # Use session - bound to this thread's event loop
                pass
    """
    manager = getattr(_thread_local_manager, "db_manager", None)

    if manager is None:
        from src.core.config import get_settings

        manager = TenantDatabaseManager(get_settings())
        _thread_local_manager.db_manager = manager

    return manager


def _clear_thread_db_connections() -> None:
    """Clear database connections for current thread.

    Called by run_async() when a new event loop is created for a thread.
    This ensures engines don't reference a closed or different event loop.

    This function is safe to call even if no manager exists for the thread.
    """
    manager = getattr(_thread_local_manager, "db_manager", None)
    if manager is not None:
        # Clear cached async engines and sessionmakers
        # They will be recreated on next access, bound to the new event loop
        manager._engines.clear()
        manager._sessionmakers.clear()
        # Clear cached sync engines and sessionmakers
        manager._sync_engines.clear()
        manager._sync_sessionmakers.clear()


def reset_worker_db_manager() -> None:
    """Reset worker DB manager for current thread.

    Clears the thread-local manager instance and all its connections.
    Primarily used for testing to ensure clean state between tests.
    """
    manager = getattr(_thread_local_manager, "db_manager", None)
    if manager is not None:
        manager._engines.clear()
        manager._sessionmakers.clear()
        manager._sync_engines.clear()
        manager._sync_sessionmakers.clear()
        _thread_local_manager.db_manager = None
