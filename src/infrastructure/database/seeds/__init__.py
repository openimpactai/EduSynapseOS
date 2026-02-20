# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Database seed package.

This package contains seed data for initializing databases:
- Central seeds: Licenses, system users
- Tenant seeds: Roles, permissions, curricula, sample data
"""

from src.infrastructure.database.seeds.central import seed_central_database
from src.infrastructure.database.seeds.tenant import seed_tenant_database

__all__ = ["seed_central_database", "seed_tenant_database"]
