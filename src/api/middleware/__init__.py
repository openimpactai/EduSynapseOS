# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API middleware components.

This package provides middleware for request processing:
- TenantMiddleware: Resolves tenant from subdomain or header.
- AuthMiddleware: JWT authentication and authorization.
- APIKeyAuthMiddleware: API key authentication for LMS integration.
- RateLimitMiddleware: Rate limiting per client.

Exports:
    TenantMiddleware: Tenant resolution middleware.
    AuthMiddleware: JWT authentication middleware.
    APIKeyAuthMiddleware: API key authentication middleware.
    RateLimitMiddleware: Rate limiting middleware.
"""

from src.api.middleware.api_key_auth import APIKeyAuthMiddleware
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.tenant import TenantMiddleware

__all__ = [
    "TenantMiddleware",
    "AuthMiddleware",
    "APIKeyAuthMiddleware",
    "RateLimitMiddleware",
]
