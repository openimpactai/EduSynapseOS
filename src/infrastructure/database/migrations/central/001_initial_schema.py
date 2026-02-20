# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Initial central database schema.

Revision ID: 001_central_initial
Revises: None
Create Date: 2024-12-24

This migration creates all central database tables based on the
SQLAlchemy models in src/infrastructure/database/models/central/.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_central_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = ("central",)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create central database tables."""
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ==========================================================================
    # 1. licenses table
    # ==========================================================================
    op.create_table(
        "licenses",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("license_key", sa.String(100), unique=True, nullable=False),
        sa.Column("max_students", sa.Integer, nullable=True),
        sa.Column("max_teachers", sa.Integer, nullable=True),
        sa.Column("max_schools", sa.Integer, nullable=True),
        sa.Column("features", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'expired', 'revoked')",
            name="valid_license_status",
        ),
    )
    op.create_index("ix_licenses_license_key", "licenses", ["license_key"])

    # ==========================================================================
    # 2. tenants table
    # ==========================================================================
    op.create_table(
        "tenants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("code", sa.String(50), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="provisioning"),
        sa.Column("hosting_type", sa.String(20), nullable=False, server_default="managed"),
        sa.Column("db_host", sa.String(255), nullable=False),
        sa.Column("db_port", sa.Integer, nullable=False, server_default="5432"),
        sa.Column("db_name", sa.String(100), nullable=False),
        sa.Column("db_username", sa.String(100), nullable=False),
        sa.Column("db_password_encrypted", sa.LargeBinary, nullable=False),
        sa.Column("db_ssl_mode", sa.String(20), nullable=False, server_default="require"),
        sa.Column("db_pool_size", sa.Integer, nullable=False, server_default="10"),
        sa.Column("data_region", sa.String(50), nullable=False, server_default="eu-west-1"),
        sa.Column("compliance_flags", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("tier", sa.String(20), nullable=False, server_default="standard"),
        sa.Column(
            "license_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("licenses.id"),
            nullable=True,
        ),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("admin_email", sa.String(255), nullable=False),
        sa.Column("admin_name", sa.String(255), nullable=True),
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
        sa.Column("provisioned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('provisioning', 'active', 'suspended', 'archived', 'deleted')",
            name="valid_status",
        ),
        sa.CheckConstraint(
            "hosting_type IN ('managed', 'self_hosted')",
            name="valid_hosting",
        ),
        sa.CheckConstraint(
            "tier IN ('free', 'standard', 'premium', 'enterprise')",
            name="valid_tier",
        ),
    )
    op.create_index("ix_tenants_code", "tenants", ["code"])
    op.create_index("ix_tenants_status", "tenants", ["status"])
    op.create_index("ix_tenants_license_id", "tenants", ["license_id"])

    # ==========================================================================
    # 3. system_users table
    # ==========================================================================
    op.create_table(
        "system_users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="admin"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("mfa_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("mfa_secret_encrypted", sa.LargeBinary, nullable=True),
        sa.Column("failed_login_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_ip", postgresql.INET, nullable=True),
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
        sa.CheckConstraint(
            "role IN ('admin', 'super_admin')",
            name="valid_role",
        ),
    )
    op.create_index("ix_system_users_email", "system_users", ["email"])

    # ==========================================================================
    # 4. system_sessions table
    # ==========================================================================
    op.create_table(
        "system_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("system_users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("access_token_hash", sa.String(64), nullable=False),
        sa.Column("refresh_token_hash", sa.String(64), nullable=True),
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("access_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_system_sessions_user_id", "system_sessions", ["user_id"])
    op.create_index("ix_system_sessions_access_token_hash", "system_sessions", ["access_token_hash"])

    # ==========================================================================
    # 5. system_audit_logs table
    # ==========================================================================
    op.create_table(
        "system_audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("system_users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_email", sa.String(255), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("entity_name", sa.String(255), nullable=True),
        sa.Column("details", postgresql.JSONB, nullable=True),
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_system_audit_logs_user_id", "system_audit_logs", ["user_id"])
    op.create_index("ix_system_audit_logs_action", "system_audit_logs", ["action"])
    op.create_index("ix_system_audit_logs_entity_type", "system_audit_logs", ["entity_type"])
    op.create_index("ix_system_audit_logs_created_at", "system_audit_logs", ["created_at"])

    # ==========================================================================
    # 6. tenant_feature_flags table
    # ==========================================================================
    op.create_table(
        "tenant_feature_flags",
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
        sa.Column("feature_key", sa.String(100), nullable=False),
        sa.Column("is_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
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
        sa.UniqueConstraint("tenant_id", "feature_key", name="unique_tenant_feature"),
    )
    op.create_index("ix_tenant_feature_flags_tenant_id", "tenant_feature_flags", ["tenant_id"])
    op.create_index("ix_tenant_feature_flags_feature_key", "tenant_feature_flags", ["feature_key"])


def downgrade() -> None:
    """Drop central database tables."""
    op.drop_table("tenant_feature_flags")
    op.drop_table("system_audit_logs")
    op.drop_table("system_sessions")
    op.drop_table("system_users")
    op.drop_table("tenants")
    op.drop_table("licenses")
