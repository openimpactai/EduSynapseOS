# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add learning_session_id to practice_sessions for cross-session linking.

This migration adds the ability to track which Practice sessions originated
from Learning Tutor sessions by adding a foreign key reference.

This enables:
- Tracking student learning paths (learning → practice flow)
- Analytics on how learning sessions lead to practice outcomes
- Correlating understanding verification with practice performance

Revision ID: 014_add_practice_learning_session_link
Revises: 013_add_skill_tables
Create Date: 2026-01-17
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "014_add_practice_learning_session_link"
down_revision: str = "013_add_skill_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add learning_session_id column to practice_sessions table."""

    # Add the learning_session_id column with foreign key to learning_sessions
    op.add_column(
        "practice_sessions",
        sa.Column(
            "learning_session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("learning_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # Create index for efficient lookups
    op.create_index(
        "idx_practice_sessions_learning_session",
        "practice_sessions",
        ["learning_session_id"],
    )


def downgrade() -> None:
    """Remove learning_session_id column from practice_sessions table."""
    op.drop_index("idx_practice_sessions_learning_session", table_name="practice_sessions")
    op.drop_column("practice_sessions", "learning_session_id")
