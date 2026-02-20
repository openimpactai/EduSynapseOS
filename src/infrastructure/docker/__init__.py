# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Docker infrastructure for tenant container management.

This package provides Docker SDK integration for managing tenant PostgreSQL containers.
Each tenant gets its own isolated PostgreSQL container for complete data isolation.

Container naming convention:
    edusynapse-tenant-{tenant_code}-db

Example:
    edusynapse-tenant-acme-db
    edusynapse-tenant-school123-db
"""

from src.infrastructure.docker.tenant_container import (
    ContainerStatus,
    TenantContainerInfo,
    TenantContainerManager,
)

__all__ = [
    "ContainerStatus",
    "TenantContainerInfo",
    "TenantContainerManager",
]
