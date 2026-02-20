# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add skill tables for skill taxonomy, categories, and subject-skill mappings.

This migration creates tables for the Skill system:
- skill_taxonomies: Skill classification systems (e.g., SUBJ-6-ENH, LIFE-SKILLS)
- skill_categories: Individual skills within taxonomies (e.g., Literacy, Numeracy)
- subject_skill_mappings: Links subjects to skill categories with impact weights

The Skill system provides:
- Multiple taxonomies for different skill frameworks
- Skill categories with visual themes for gamification
- Subject-to-skill mappings for tracking skill development through curriculum

Synced from Central Curriculum service.

Revision ID: 013_add_skill_tables
Revises: 012_curriculum_composite_keys
Create Date: 2026-01-16
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "013_add_skill_tables"
down_revision: str = "012_curriculum_composite_keys"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create skill_taxonomies, skill_categories, and subject_skill_mappings tables."""

    # =========================================================================
    # Create skill_taxonomies table
    # =========================================================================
    op.create_table(
        "skill_taxonomies",
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "taxonomy_type",
            sa.String(30),
            nullable=False,
            server_default="hybrid"
        ),
        sa.Column("author", sa.String(100), nullable=True),
        sa.Column("year_published", sa.Integer(), nullable=True),
        sa.Column("age_range", sa.String(20), nullable=True),
        sa.Column("skill_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "visualization_type",
            sa.String(20),
            nullable=False,
            server_default="hexagon"
        ),
        sa.Column("balance_required", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_official", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("extra_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("code"),
        sa.UniqueConstraint("slug", name="uq_skill_taxonomy_slug"),
        sa.CheckConstraint(
            "taxonomy_type IN ('hybrid', 'competency_based', 'subject_based', 'behavioral')",
            name="chk_skill_taxonomy_type"
        ),
        sa.CheckConstraint(
            "visualization_type IN ('hexagon', 'circle', 'radar', 'bar', 'tree')",
            name="chk_skill_visualization_type"
        ),
    )
    op.create_index("idx_skill_taxonomies_active", "skill_taxonomies", ["is_active"])
    op.create_index("idx_skill_taxonomies_default", "skill_taxonomies", ["is_default"])

    # =========================================================================
    # Create skill_categories table
    # =========================================================================
    op.create_table(
        "skill_categories",
        sa.Column("taxonomy_code", sa.String(30), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("short_code", sa.String(10), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("color", sa.String(7), nullable=True),
        sa.Column("icon", sa.String(50), nullable=True),
        sa.Column("age_appropriate_from", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("age_appropriate_to", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("visual_themes", postgresql.JSONB(), nullable=True),
        sa.Column("extra_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("taxonomy_code", "code"),
        sa.ForeignKeyConstraint(
            ["taxonomy_code"],
            ["skill_taxonomies.code"],
            ondelete="CASCADE"
        ),
    )
    op.create_index("idx_skill_categories_taxonomy", "skill_categories", ["taxonomy_code"])
    op.create_index("idx_skill_categories_active", "skill_categories", ["is_active"])

    # =========================================================================
    # Create subject_skill_mappings table
    # =========================================================================
    op.create_table(
        "subject_skill_mappings",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("subject_code", sa.String(50), nullable=False),
        sa.Column("taxonomy_code", sa.String(30), nullable=False),
        sa.Column("skill_code", sa.String(50), nullable=False),
        sa.Column(
            "impact_weight",
            sa.Numeric(3, 2),
            nullable=False,
            server_default="1.00"
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint(
            "framework_code", "subject_code", "taxonomy_code", "skill_code"
        ),
        sa.ForeignKeyConstraint(
            ["framework_code", "subject_code"],
            ["subjects.framework_code", "subjects.code"],
            ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["taxonomy_code", "skill_code"],
            ["skill_categories.taxonomy_code", "skill_categories.code"],
            ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "impact_weight >= 0 AND impact_weight <= 1",
            name="chk_skill_mapping_weight"
        ),
    )
    op.create_index(
        "idx_skill_mapping_subject",
        "subject_skill_mappings",
        ["framework_code", "subject_code"]
    )
    op.create_index(
        "idx_skill_mapping_skill",
        "subject_skill_mappings",
        ["taxonomy_code", "skill_code"]
    )
    op.create_index("idx_skill_mapping_active", "subject_skill_mappings", ["is_active"])


def downgrade() -> None:
    """Drop skill tables in reverse order."""
    op.drop_table("subject_skill_mappings")
    op.drop_table("skill_categories")
    op.drop_table("skill_taxonomies")
