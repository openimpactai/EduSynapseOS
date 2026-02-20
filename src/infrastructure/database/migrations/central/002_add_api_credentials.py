# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add tenant API credentials tables.

Revision ID: 002_add_api_credentials
Revises: 001_central_initial
Create Date: 2024-12-30

This migration adds tables for API key-based authentication:
- tenant_api_credentials: Stores API keys for tenant LMS systems
- api_key_audit_logs: Audit trail for API key authentication attempts
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002_add_api_credentials"
down_revision: Union[str, None] = "001_central_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create API credentials tables."""
    # ==========================================================================
    # 1. tenant_api_credentials table
    # ==========================================================================
    op.create_table(
        "tenant_api_credentials",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # API Key (public identifier)
        sa.Column("api_key", sa.String(50), unique=True, nullable=False),
        sa.Column("api_key_prefix", sa.String(15), nullable=False),
        # API Secret (hashed, never stored plain)
        sa.Column("api_secret_hash", sa.String(255), nullable=False),
        sa.Column("api_secret_prefix", sa.String(15), nullable=False),
        # Metadata
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        # Security - IP Whitelist
        sa.Column("allowed_ips", postgresql.ARRAY(postgresql.INET), nullable=True),
        # Security - CORS Origins
        sa.Column("allowed_origins", postgresql.ARRAY(sa.Text), nullable=True),
        # Rate Limiting
        sa.Column(
            "rate_limit_per_minute",
            sa.Integer,
            nullable=False,
            server_default="1000",
        ),
        # Status
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        # Usage tracking
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("usage_count", sa.BigInteger, nullable=False, server_default="0"),
        # Audit - Creation
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("system_users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Revocation
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "revoked_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("system_users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("revoke_reason", sa.Text, nullable=True),
    )

    # Indexes for tenant_api_credentials
    op.create_index(
        "ix_tenant_api_credentials_api_key",
        "tenant_api_credentials",
        ["api_key"],
    )
    op.create_index(
        "ix_tenant_api_credentials_tenant_id",
        "tenant_api_credentials",
        ["tenant_id"],
    )
    # Partial index for active credentials per tenant
    op.create_index(
        "idx_tenant_api_credentials_active",
        "tenant_api_credentials",
        ["tenant_id", "is_active"],
        postgresql_where=sa.text("is_active = true"),
    )

    # ==========================================================================
    # 2. api_key_audit_logs table
    # ==========================================================================
    op.create_table(
        "api_key_audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "credential_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenant_api_credentials.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Action details
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("endpoint", sa.String(255), nullable=True),
        sa.Column("method", sa.String(10), nullable=True),
        # User assertion
        sa.Column("user_id_asserted", sa.String(255), nullable=True),
        # Client info
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        # Result
        sa.Column("success", sa.Boolean, nullable=False),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # Performance
        sa.Column("response_time_ms", sa.Integer, nullable=True),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Check constraint for valid actions
        sa.CheckConstraint(
            "action IN ('authenticate', 'exchange_token', 'api_call', 'validate')",
            name="valid_action",
        ),
    )

    # Indexes for api_key_audit_logs
    op.create_index(
        "ix_api_key_audit_logs_credential_id",
        "api_key_audit_logs",
        ["credential_id"],
    )
    op.create_index(
        "idx_api_key_audit_tenant_date",
        "api_key_audit_logs",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "idx_api_key_audit_credential_date",
        "api_key_audit_logs",
        ["credential_id", "created_at"],
    )


def downgrade() -> None:
    """Drop API credentials tables."""
    op.drop_table("api_key_audit_logs")
    op.drop_table("tenant_api_credentials")
