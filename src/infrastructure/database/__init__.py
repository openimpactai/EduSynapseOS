# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Database infrastructure for PostgreSQL connections.

This package provides SQLAlchemy async database connections for:
- Central database: Platform-level data (tenants, system admins, licenses)
- Tenant databases: Per-tenant isolated data (users, sessions, memory)

The central database is a static PostgreSQL instance.
Tenant databases are dynamically created Docker containers managed by TenantContainerManager.

Example:
    from src.infrastructure.database import (
        get_central_engine,
        get_central_session,
        TenantDatabaseManager,
    )

    # Get central database session
    async with get_central_session() as session:
        result = await session.execute(select(Tenant))

    # Get tenant database session
    tenant_manager = TenantDatabaseManager(settings)
    async with tenant_manager.get_session("acme") as session:
        result = await session.execute(select(User))
"""

from src.infrastructure.database.connection import (
    DatabaseError,
    check_central_database_connection,
    close_central_database,
    get_central_engine,
    get_central_session,
    get_central_sessionmaker,
    init_central_database,
)
from src.infrastructure.database.tenant_manager import (
    TenantConnectionInfo,
    TenantDatabaseManager,
    TenantNotFoundError,
    TenantProvisioningError,
    _clear_thread_db_connections,
    get_worker_db_manager,
    reset_worker_db_manager,
)

__all__ = [
    # Central database
    "DatabaseError",
    "check_central_database_connection",
    "close_central_database",
    "get_central_engine",
    "get_central_session",
    "get_central_sessionmaker",
    "init_central_database",
    # Tenant database
    "TenantConnectionInfo",
    "TenantDatabaseManager",
    "TenantNotFoundError",
    "TenantProvisioningError",
    # Worker thread-local manager
    "get_worker_db_manager",
    "reset_worker_db_manager",
    "_clear_thread_db_connections",
]
