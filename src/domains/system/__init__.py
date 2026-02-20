# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System domain for platform-level administration.

This module provides services for:
- System admin authentication (Central DB)
- Tenant lifecycle management (Docker SDK integration)
"""

from src.domains.system.auth_service import (
    SystemAuthService,
    SystemAuthError,
    SystemInvalidCredentialsError,
    SystemAccountLockedError,
    SystemAccountInactiveError,
    SystemSessionNotFoundError,
)
from src.domains.system.tenant_service import (
    TenantService,
    TenantProvisioningError,
    TenantNotFoundError,
    TenantAlreadyExistsError,
)

__all__ = [
    "SystemAuthService",
    "SystemAuthError",
    "SystemInvalidCredentialsError",
    "SystemAccountLockedError",
    "SystemAccountInactiveError",
    "SystemSessionNotFoundError",
    "TenantService",
    "TenantProvisioningError",
    "TenantNotFoundError",
    "TenantAlreadyExistsError",
]
