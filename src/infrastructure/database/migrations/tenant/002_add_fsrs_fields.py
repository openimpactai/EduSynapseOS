# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add FSRS spaced repetition fields to semantic_memories.

Revision ID: 002_add_fsrs_fields
Revises: 001_tenant_initial
Create Date: 2024-12-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_add_fsrs_fields"
down_revision: str = "001_tenant_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add FSRS state, step, and last_review fields to semantic_memories."""
    # Add fsrs_state column
    op.add_column(
        "semantic_memories",
        sa.Column("fsrs_state", sa.String(20), nullable=True),
    )

    # Add fsrs_step column
    op.add_column(
        "semantic_memories",
        sa.Column("fsrs_step", sa.Integer, nullable=True),
    )

    # Add fsrs_last_review column
    op.add_column(
        "semantic_memories",
        sa.Column("fsrs_last_review", sa.DateTime(timezone=True), nullable=True),
    )

    # Update fsrs_stability precision from Numeric(10,4) to Numeric(10,6)
    op.alter_column(
        "semantic_memories",
        "fsrs_stability",
        type_=sa.Numeric(10, 6),
        existing_type=sa.Numeric(10, 4),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Remove FSRS fields."""
    op.drop_column("semantic_memories", "fsrs_last_review")
    op.drop_column("semantic_memories", "fsrs_step")
    op.drop_column("semantic_memories", "fsrs_state")

    # Revert fsrs_stability precision
    op.alter_column(
        "semantic_memories",
        "fsrs_stability",
        type_=sa.Numeric(10, 4),
        existing_type=sa.Numeric(10, 6),
        existing_nullable=True,
    )
