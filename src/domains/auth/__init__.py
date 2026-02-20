# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Authentication domain services.

This module provides authentication and authorization services:
- JWT token creation and validation
- Session management for tenant users
- API Key service for tenant LMS authentication
- Password hashing (for system admin only)

Tenant users are authenticated via LMS integration - the LMS authenticates
users and asserts their identity to EduSynapseOS using API credentials.

Exports:
    PasswordHasher: Secure password hashing using bcrypt (system admin only).
    JWTManager: JWT token creation and validation.
    AuthService: Session management service.
    APIKeyService: API key management for tenant LMS systems.
"""

from src.domains.auth.api_key_service import APIKeyService
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.auth.service import AuthService

__all__ = [
    "PasswordHasher",
    "JWTManager",
    "AuthService",
    "APIKeyService",
]
