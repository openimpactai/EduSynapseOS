# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""API Key service for tenant LMS authentication.

This module provides the APIKeyService for managing tenant API credentials:
- Generate API keys and secrets
- Validate API credentials
- Record usage statistics
- Audit logging for security monitoring

Example:
    >>> api_key_service = APIKeyService(central_db_session)
    >>> credential = await api_key_service.create_credential(
    ...     tenant_id=tenant.id,
    ...     name="Production API Key",
    ...     created_by_id=admin.id,
    ... )
    >>> result = await api_key_service.validate_credential(api_key, api_secret)
"""

import ipaddress
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NamedTuple
from uuid import UUID

import bcrypt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.central.api_credential import (
    APIKeyAuditLog,
    TenantAPICredential,
)
from src.infrastructure.database.models.central.tenant import Tenant

logger = logging.getLogger(__name__)


class APIKeyError(Exception):
    """Base exception for API key operations."""

    pass


class InvalidAPIKeyError(APIKeyError):
    """Raised when API key is invalid or not found."""

    pass


class InvalidAPISecretError(APIKeyError):
    """Raised when API secret does not match."""

    pass


class APIKeyRevokedError(APIKeyError):
    """Raised when API key has been revoked."""

    pass


class APIKeyInactiveError(APIKeyError):
    """Raised when API key is not active."""

    pass


class IPNotAllowedError(APIKeyError):
    """Raised when request IP is not in whitelist."""

    pass


class RateLimitExceededError(APIKeyError):
    """Raised when rate limit is exceeded."""

    pass


@dataclass
class GeneratedCredential:
    """Result of creating a new API credential.

    Attributes:
        credential: The created TenantAPICredential record.
        api_key: The full API key (only returned once at creation).
        api_secret: The full API secret (only returned once at creation).
    """

    credential: TenantAPICredential
    api_key: str
    api_secret: str


class ValidationResult(NamedTuple):
    """Result of API credential validation.

    Attributes:
        credential: The validated TenantAPICredential.
        tenant: The associated Tenant.
    """

    credential: TenantAPICredential
    tenant: Tenant


class APIKeyService:
    """Service for managing tenant API credentials.

    Handles creation, validation, and management of API keys that
    allow tenant LMS systems to authenticate with EduSynapseOS.

    Attributes:
        _db: Central database session.

    Example:
        >>> service = APIKeyService(central_db)
        >>> cred = await service.create_credential(tenant_id, "My Key")
        >>> print(f"API Key: {cred.api_key}")
        >>> print(f"API Secret: {cred.api_secret}")
    """

    API_KEY_PREFIX = "tk_"
    API_SECRET_PREFIX = "ts_"
    KEY_LENGTH = 32
    SECRET_LENGTH = 48

    def __init__(
        self,
        db: AsyncSession,
        bcrypt_rounds: int = 12,
    ) -> None:
        """Initialize the API Key service.

        Args:
            db: Central database async session.
            bcrypt_rounds: Number of bcrypt rounds for secret hashing.
        """
        self._db = db
        self._bcrypt_rounds = bcrypt_rounds

    async def create_credential(
        self,
        tenant_id: str | UUID,
        name: str,
        created_by_id: str | UUID | None = None,
        description: str | None = None,
        allowed_ips: list[str] | None = None,
        allowed_origins: list[str] | None = None,
        rate_limit_per_minute: int = 1000,
    ) -> GeneratedCredential:
        """Create a new API credential for a tenant.

        Generates a unique API key and secret. The secret is hashed before
        storage and can only be retrieved at creation time.

        Args:
            tenant_id: Tenant identifier.
            name: Descriptive name for the credential.
            created_by_id: System user who created this credential.
            description: Optional detailed description.
            allowed_ips: IP whitelist (None = allow all).
            allowed_origins: CORS origins whitelist.
            rate_limit_per_minute: Rate limit for this credential.

        Returns:
            GeneratedCredential with the credential record and plain secrets.

        Raises:
            ValueError: If tenant_id is invalid.
        """
        tenant_id_str = str(tenant_id)

        # Verify tenant exists
        stmt = select(Tenant).where(Tenant.id == tenant_id_str)
        result = await self._db.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        # Generate API key and secret
        api_key = self._generate_api_key()
        api_secret = self._generate_api_secret()

        # Create prefixes for display
        api_key_prefix = f"{api_key[:8]}..."
        api_secret_prefix = f"{api_secret[:8]}..."

        # Hash the secret
        api_secret_hash = self._hash_secret(api_secret)

        # Create the credential
        credential = TenantAPICredential(
            tenant_id=tenant_id_str,
            api_key=api_key,
            api_key_prefix=api_key_prefix,
            api_secret_hash=api_secret_hash,
            api_secret_prefix=api_secret_prefix,
            name=name,
            description=description,
            allowed_ips=allowed_ips,
            allowed_origins=allowed_origins,
            rate_limit_per_minute=rate_limit_per_minute,
            is_active=True,
            created_by=str(created_by_id) if created_by_id else None,
        )

        self._db.add(credential)
        await self._db.flush()

        logger.info(
            "API credential created for tenant %s: %s (key: %s)",
            tenant_id,
            name,
            api_key_prefix,
        )

        return GeneratedCredential(
            credential=credential,
            api_key=api_key,
            api_secret=api_secret,
        )

    async def validate_credential(
        self,
        api_key: str,
        api_secret: str,
        client_ip: str | None = None,
    ) -> ValidationResult:
        """Validate API credentials.

        Verifies the API key exists, secret matches, credential is active,
        and optionally checks IP whitelist.

        Args:
            api_key: The API key to validate.
            api_secret: The API secret to verify.
            client_ip: Optional client IP for whitelist check.

        Returns:
            ValidationResult with the credential and tenant.

        Raises:
            InvalidAPIKeyError: If API key is not found.
            InvalidAPISecretError: If secret doesn't match.
            APIKeyRevokedError: If credential has been revoked.
            APIKeyInactiveError: If credential is not active.
            IPNotAllowedError: If client IP is not in whitelist.
        """
        # Find the credential
        stmt = (
            select(TenantAPICredential)
            .options(selectinload(TenantAPICredential.tenant))
            .where(TenantAPICredential.api_key == api_key)
        )
        result = await self._db.execute(stmt)
        credential = result.scalar_one_or_none()

        if not credential:
            logger.warning("Invalid API key attempted: %s...", api_key[:8] if api_key else "empty")
            raise InvalidAPIKeyError("Invalid API key")

        # Check if revoked
        if credential.is_revoked:
            logger.warning(
                "Revoked API key attempted: %s (tenant: %s)",
                credential.api_key_prefix,
                credential.tenant_id,
            )
            raise APIKeyRevokedError("API key has been revoked")

        # Check if active
        if not credential.is_active:
            logger.warning(
                "Inactive API key attempted: %s (tenant: %s)",
                credential.api_key_prefix,
                credential.tenant_id,
            )
            raise APIKeyInactiveError("API key is not active")

        # Verify secret
        if not self._verify_secret(api_secret, credential.api_secret_hash):
            logger.warning(
                "Invalid API secret for key: %s (tenant: %s)",
                credential.api_key_prefix,
                credential.tenant_id,
            )
            raise InvalidAPISecretError("Invalid API secret")

        # Check IP whitelist
        if client_ip and credential.allowed_ips:
            if not self._is_ip_allowed(client_ip, credential.allowed_ips):
                logger.warning(
                    "IP not in whitelist: %s for key %s (tenant: %s)",
                    client_ip,
                    credential.api_key_prefix,
                    credential.tenant_id,
                )
                raise IPNotAllowedError(f"IP address {client_ip} is not allowed")

        # Check tenant is active
        if not credential.tenant.is_active:
            logger.warning(
                "Tenant not active for API key: %s (tenant: %s)",
                credential.api_key_prefix,
                credential.tenant_id,
            )
            raise APIKeyInactiveError("Tenant is not active")

        return ValidationResult(credential=credential, tenant=credential.tenant)

    async def record_usage(
        self,
        credential: TenantAPICredential,
    ) -> None:
        """Record successful usage of an API credential.

        Updates last_used_at and increments usage_count.

        Args:
            credential: The credential that was used.
        """
        credential.record_usage()
        await self._db.flush()

    async def log_audit_event(
        self,
        credential: TenantAPICredential,
        action: str,
        success: bool,
        endpoint: str | None = None,
        method: str | None = None,
        user_id_asserted: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        response_time_ms: int | None = None,
    ) -> APIKeyAuditLog:
        """Log an API key authentication/usage event.

        Args:
            credential: The API credential used.
            action: Type of action (authenticate, exchange_token, api_call, validate).
            success: Whether the operation succeeded.
            endpoint: API endpoint called.
            method: HTTP method used.
            user_id_asserted: User ID asserted by the LMS.
            ip_address: Client IP address.
            user_agent: Client user agent string.
            error_code: Error code if failed.
            error_message: Error message if failed.
            response_time_ms: Response time in milliseconds.

        Returns:
            The created audit log entry.
        """
        log = APIKeyAuditLog(
            credential_id=credential.id,
            tenant_id=credential.tenant_id,
            action=action,
            endpoint=endpoint,
            method=method,
            user_id_asserted=user_id_asserted,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_code=error_code,
            error_message=error_message,
            response_time_ms=response_time_ms,
        )

        self._db.add(log)
        await self._db.flush()

        return log

    async def revoke_credential(
        self,
        credential_id: str | UUID,
        revoked_by_id: str | UUID | None = None,
        reason: str | None = None,
    ) -> TenantAPICredential | None:
        """Revoke an API credential.

        Args:
            credential_id: Credential to revoke.
            revoked_by_id: System user performing the revocation.
            reason: Reason for revocation.

        Returns:
            The revoked credential, or None if not found.
        """
        stmt = select(TenantAPICredential).where(
            TenantAPICredential.id == str(credential_id)
        )
        result = await self._db.execute(stmt)
        credential = result.scalar_one_or_none()

        if not credential:
            return None

        credential.revoke(
            revoked_by_id=str(revoked_by_id) if revoked_by_id else None,
            reason=reason,
        )

        await self._db.flush()

        logger.info(
            "API credential revoked: %s (tenant: %s, reason: %s)",
            credential.api_key_prefix,
            credential.tenant_id,
            reason or "not specified",
        )

        return credential

    async def get_tenant_credentials(
        self,
        tenant_id: str | UUID,
        include_revoked: bool = False,
    ) -> list[TenantAPICredential]:
        """Get all API credentials for a tenant.

        Args:
            tenant_id: Tenant identifier.
            include_revoked: Whether to include revoked credentials.

        Returns:
            List of API credentials.
        """
        stmt = select(TenantAPICredential).where(
            TenantAPICredential.tenant_id == str(tenant_id)
        )

        if not include_revoked:
            stmt = stmt.where(TenantAPICredential.revoked_at.is_(None))

        stmt = stmt.order_by(TenantAPICredential.created_at.desc())

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def get_credential_by_id(
        self,
        credential_id: str | UUID,
    ) -> TenantAPICredential | None:
        """Get an API credential by ID.

        Args:
            credential_id: Credential identifier.

        Returns:
            The credential, or None if not found.
        """
        stmt = (
            select(TenantAPICredential)
            .options(selectinload(TenantAPICredential.tenant))
            .where(TenantAPICredential.id == str(credential_id))
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_credential_by_api_key(
        self,
        api_key: str,
    ) -> TenantAPICredential | None:
        """Get an API credential by API key.

        Args:
            api_key: The API key.

        Returns:
            The credential, or None if not found.
        """
        stmt = (
            select(TenantAPICredential)
            .options(selectinload(TenantAPICredential.tenant))
            .where(TenantAPICredential.api_key == api_key)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_credential(
        self,
        credential_id: str | UUID,
        name: str | None = None,
        description: str | None = None,
        allowed_ips: list[str] | None = None,
        allowed_origins: list[str] | None = None,
        rate_limit_per_minute: int | None = None,
        is_active: bool | None = None,
    ) -> TenantAPICredential | None:
        """Update an API credential's settings.

        Args:
            credential_id: Credential to update.
            name: New name (if changing).
            description: New description (if changing).
            allowed_ips: New IP whitelist (if changing).
            allowed_origins: New CORS origins (if changing).
            rate_limit_per_minute: New rate limit (if changing).
            is_active: New active status (if changing).

        Returns:
            The updated credential, or None if not found.
        """
        stmt = select(TenantAPICredential).where(
            TenantAPICredential.id == str(credential_id)
        )
        result = await self._db.execute(stmt)
        credential = result.scalar_one_or_none()

        if not credential:
            return None

        if name is not None:
            credential.name = name
        if description is not None:
            credential.description = description
        if allowed_ips is not None:
            credential.allowed_ips = allowed_ips
        if allowed_origins is not None:
            credential.allowed_origins = allowed_origins
        if rate_limit_per_minute is not None:
            credential.rate_limit_per_minute = rate_limit_per_minute
        if is_active is not None:
            credential.is_active = is_active

        await self._db.flush()

        return credential

    async def list_credentials(
        self,
        tenant_id: str | UUID,
        include_revoked: bool = False,
    ) -> list[TenantAPICredential]:
        """List all API credentials for a tenant.

        Args:
            tenant_id: Tenant ID to list credentials for.
            include_revoked: Include revoked credentials in results.

        Returns:
            List of TenantAPICredential objects.
        """
        stmt = select(TenantAPICredential).where(
            TenantAPICredential.tenant_id == str(tenant_id)
        )

        if not include_revoked:
            stmt = stmt.where(TenantAPICredential.revoked_at.is_(None))

        stmt = stmt.order_by(TenantAPICredential.created_at.desc())

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def regenerate_secret(
        self,
        credential_id: str | UUID,
    ) -> GeneratedCredential:
        """Regenerate the API secret for a credential.

        The old secret will no longer work after this operation.

        Args:
            credential_id: Credential to regenerate secret for.

        Returns:
            GeneratedCredential with updated credential and new secret.

        Raises:
            ValueError: If credential not found or is revoked.
        """
        stmt = select(TenantAPICredential).where(
            TenantAPICredential.id == str(credential_id)
        )
        result = await self._db.execute(stmt)
        credential = result.scalar_one_or_none()

        if not credential:
            raise ValueError(f"Credential not found: {credential_id}")

        if credential.is_revoked:
            raise ValueError("Cannot regenerate secret for revoked credential")

        # Generate new secret
        new_secret = self._generate_api_secret()
        credential.api_secret_hash = self._hash_secret(new_secret)
        credential.api_secret_prefix = f"{new_secret[:8]}..."

        await self._db.flush()

        logger.info(
            "API secret regenerated for credential: %s (tenant: %s)",
            credential.api_key_prefix,
            credential.tenant_id,
        )

        return GeneratedCredential(
            credential=credential,
            api_key=credential.api_key,
            api_secret=new_secret,
        )

    def _generate_api_key(self) -> str:
        """Generate a unique API key.

        Format: tk_<32 hex characters>

        Returns:
            A unique API key string.
        """
        random_bytes = secrets.token_hex(self.KEY_LENGTH // 2)
        return f"{self.API_KEY_PREFIX}{random_bytes}"

    def _generate_api_secret(self) -> str:
        """Generate a secure API secret.

        Format: ts_<48 hex characters>

        Returns:
            A unique API secret string.
        """
        random_bytes = secrets.token_hex(self.SECRET_LENGTH // 2)
        return f"{self.API_SECRET_PREFIX}{random_bytes}"

    def _hash_secret(self, secret: str) -> str:
        """Hash an API secret using bcrypt.

        Args:
            secret: Plain text secret to hash.

        Returns:
            Bcrypt hash string.
        """
        salt = bcrypt.gensalt(rounds=self._bcrypt_rounds)
        hashed = bcrypt.hashpw(secret.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def _verify_secret(self, secret: str, secret_hash: str) -> bool:
        """Verify an API secret against its hash.

        Args:
            secret: Plain text secret to verify.
            secret_hash: Bcrypt hash to verify against.

        Returns:
            True if secret matches, False otherwise.
        """
        if not secret or not secret_hash:
            return False

        try:
            return bcrypt.checkpw(
                secret.encode("utf-8"),
                secret_hash.encode("utf-8"),
            )
        except Exception as e:
            logger.warning("Secret verification failed: %s", str(e))
            return False

    def _is_ip_allowed(self, client_ip: str, allowed_ips: list[str]) -> bool:
        """Check if a client IP is in the allowed list.

        Supports both individual IPs and CIDR notation.

        Args:
            client_ip: Client IP address.
            allowed_ips: List of allowed IPs/CIDRs.

        Returns:
            True if IP is allowed, False otherwise.
        """
        try:
            client = ipaddress.ip_address(client_ip)

            for allowed in allowed_ips:
                try:
                    if "/" in allowed:
                        network = ipaddress.ip_network(allowed, strict=False)
                        if client in network:
                            return True
                    else:
                        if client == ipaddress.ip_address(allowed):
                            return True
                except ValueError:
                    continue

            return False
        except ValueError:
            return False
