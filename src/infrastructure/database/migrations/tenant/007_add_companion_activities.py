# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add companion_activities table.

This migration creates the companion_activities table for storing
available activities that the companion can suggest to students.

Activities are categorized by type (learning, fun, creative, break)
and filtered by grade level and difficulty for personalized suggestions.

Revision ID: 007_add_companion_activities
Revises: 006_remove_password_auth
Create Date: 2025-01-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_add_companion_activities"
down_revision: str = "006_remove_password_auth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create companion_activities table."""
    op.create_table(
        "companion_activities",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "code",
            sa.String(50),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "description",
            sa.Text,
            nullable=True,
        ),
        sa.Column(
            "icon",
            sa.String(10),
            nullable=True,
        ),
        sa.Column(
            "category",
            sa.String(30),
            nullable=False,
        ),
        sa.Column(
            "route",
            sa.String(255),
            nullable=True,
        ),
        sa.Column(
            "min_grade",
            sa.Integer,
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "max_grade",
            sa.Integer,
            nullable=False,
            server_default="12",
        ),
        sa.Column(
            "difficulty",
            sa.String(20),
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "is_enabled",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "display_order",
            sa.Integer,
            nullable=False,
            server_default="0",
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code", name="uq_companion_activities_code"),
        sa.CheckConstraint(
            "category IN ('learning', 'fun', 'creative', 'break')",
            name="companion_activities_valid_category",
        ),
        sa.CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')",
            name="companion_activities_valid_difficulty",
        ),
        sa.CheckConstraint(
            "min_grade >= 1 AND min_grade <= 12",
            name="companion_activities_valid_min_grade",
        ),
        sa.CheckConstraint(
            "max_grade >= 1 AND max_grade <= 12",
            name="companion_activities_valid_max_grade",
        ),
        sa.CheckConstraint(
            "min_grade <= max_grade",
            name="companion_activities_grade_range",
        ),
    )

    # Index for category filtering
    op.create_index(
        "idx_companion_activities_category",
        "companion_activities",
        ["category"],
    )

    # Index for enabled activities with grade range filtering
    op.create_index(
        "idx_companion_activities_enabled_grade",
        "companion_activities",
        ["is_enabled", "min_grade", "max_grade"],
        postgresql_where=sa.text("is_enabled = true"),
    )

    # Index for display ordering
    op.create_index(
        "idx_companion_activities_order",
        "companion_activities",
        ["category", "display_order"],
    )


def downgrade() -> None:
    """Drop companion_activities table."""
    op.drop_index(
        "idx_companion_activities_order",
        table_name="companion_activities",
    )
    op.drop_index(
        "idx_companion_activities_enabled_grade",
        table_name="companion_activities",
    )
    op.drop_index(
        "idx_companion_activities_category",
        table_name="companion_activities",
    )
    op.drop_table("companion_activities")
