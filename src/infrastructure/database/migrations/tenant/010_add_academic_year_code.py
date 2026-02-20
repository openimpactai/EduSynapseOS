# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add code column to academic_years table.

This migration adds a unique `code` column to the academic_years table
for LMS integration. The code will be used as a human-readable identifier
instead of UUIDs in API communications.

Migration strategy:
1. Add code column as nullable
2. Populate existing rows using name as code value
3. Make column NOT NULL
4. Add unique constraint

Revision ID: 010_add_academic_year_code
Revises: 009_add_learning_sessions
Create Date: 2026-01-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010_add_academic_year_code"
down_revision: str = "009_add_learning_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add code column to academic_years table."""
    # Step 1: Add code column as nullable
    op.add_column(
        "academic_years",
        sa.Column("code", sa.String(50), nullable=True),
    )

    # Step 2: Populate existing rows - use name as code
    # (name is typically "2024-2025" which is a good code format)
    op.execute(
        "UPDATE academic_years SET code = name WHERE code IS NULL"
    )

    # Step 3: Make column NOT NULL
    op.alter_column(
        "academic_years",
        "code",
        nullable=False,
        existing_type=sa.String(50),
    )

    # Step 4: Add unique constraint
    op.create_unique_constraint(
        "uq_academic_years_code",
        "academic_years",
        ["code"],
    )


def downgrade() -> None:
    """Remove code column from academic_years table."""
    # Drop unique constraint first
    op.drop_constraint(
        "uq_academic_years_code",
        "academic_years",
        type_="unique",
    )

    # Drop column
    op.drop_column("academic_years", "code")
