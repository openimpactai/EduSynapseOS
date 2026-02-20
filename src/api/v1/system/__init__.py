# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System administration API endpoints.

This module provides API routes for platform-level administration:
- /system/auth - System admin authentication
- /system/tenants - Tenant lifecycle management
- /system/stats - System statistics
- /system/configs - Configuration management
"""

from fastapi import APIRouter

from src.api.v1.system.auth import router as auth_router
from src.api.v1.system.tenants import router as tenants_router
from src.api.v1.system.stats import router as stats_router
from src.api.v1.system.configs import router as configs_router

router = APIRouter(prefix="/system", tags=["System Administration"])

router.include_router(auth_router, prefix="/auth", tags=["System Auth"])
router.include_router(tenants_router, prefix="/tenants", tags=["Tenants"])
router.include_router(stats_router, tags=["System Stats"])
router.include_router(configs_router, tags=["Configuration"])

__all__ = ["router"]
