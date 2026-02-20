# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Remove password-based authentication.

This migration removes password-based authentication fields and tables
as part of the transition to LMS-based authentication via API keys.

Users are now authenticated via LMS integration where:
1. LMS authenticates users internally
2. LMS calls EduSynapseOS API with API key/secret and user assertion
3. EduSynapseOS trusts the LMS assertion after validating API credentials

Removed fields from users table:
- password_hash
- failed_login_attempts
- locked_until
- password_changed_at
- must_change_password

Dropped tables:
- password_reset_tokens

Revision ID: 006_remove_password_auth
Revises: 005_add_student_notes
Create Date: 2024-12-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006_remove_password_auth"
down_revision: str = "005_add_student_notes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove password authentication fields and tables.

    This migration is idempotent - it checks if tables/columns exist before dropping.
    This is necessary because fresh tenant databases created via SQLAlchemy models
    may not have these password-related tables/columns.
    """
    from sqlalchemy import inspect
    from alembic import context

    # Get connection and inspector
    conn = context.get_context().connection
    inspector = inspect(conn)

    # Get existing tables and columns
    existing_tables = inspector.get_table_names()
    users_columns = []
    if "users" in existing_tables:
        users_columns = [col["name"] for col in inspector.get_columns("users")]

    # Drop password_reset_tokens table if exists
    if "password_reset_tokens" in existing_tables:
        op.drop_index(
            "ix_password_reset_tokens_user_id",
            table_name="password_reset_tokens",
            if_exists=True,
        )
        op.drop_index(
            "ix_password_reset_tokens_token_hash",
            table_name="password_reset_tokens",
            if_exists=True,
        )
        op.drop_table("password_reset_tokens")

    # Remove password-related columns from users table if they exist
    if "password_hash" in users_columns:
        op.drop_column("users", "password_hash")
    if "failed_login_attempts" in users_columns:
        op.drop_column("users", "failed_login_attempts")
    if "locked_until" in users_columns:
        op.drop_column("users", "locked_until")
    if "password_changed_at" in users_columns:
        op.drop_column("users", "password_changed_at")
    if "must_change_password" in users_columns:
        op.drop_column("users", "must_change_password")


def downgrade() -> None:
    """Restore password authentication fields and tables."""

    # Add back password-related columns to users table
    op.add_column(
        "users",
        sa.Column(
            "password_hash",
            sa.String(255),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "failed_login_attempts",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "locked_until",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "password_changed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "must_change_password",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
    )

    # Recreate password_reset_tokens table
    op.create_table(
        "password_reset_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requested_ip", postgresql.INET, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "ix_password_reset_tokens_user_id",
        "password_reset_tokens",
        ["user_id"],
    )
    op.create_index(
        "ix_password_reset_tokens_token_hash",
        "password_reset_tokens",
        ["token_hash"],
    )
