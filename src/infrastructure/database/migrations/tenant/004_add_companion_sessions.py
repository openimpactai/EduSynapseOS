# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add companion sessions table for AI companion tracking.

This migration adds the companion_sessions table to track
persistent companion conversation state and relationship history.

Revision ID: 004_add_companion_sessions
Revises: 003_add_emotional_intelligence_core
Create Date: 2024-12-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004_add_companion_sessions"
down_revision: str = "003_add_emotional_intelligence_core"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add companion_sessions table."""

    op.create_table(
        "companion_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        # Session state
        sa.Column(
            "status",
            sa.String(20),
            server_default="active",
            nullable=False,
        ),
        sa.Column("session_type", sa.String(50), nullable=False),
        # Emotional tracking for this session
        sa.Column("emotional_state_start", sa.String(30), nullable=True),
        sa.Column("emotional_state_end", sa.String(30), nullable=True),
        sa.Column(
            "emotional_transitions",
            postgresql.JSONB,
            server_default="[]",
            nullable=False,
        ),
        # Relationship metrics
        sa.Column(
            "rapport_score",
            sa.Numeric(3, 2),
            server_default="0.50",
            nullable=False,
        ),
        sa.Column(
            "interaction_count",
            sa.Integer,
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "positive_interactions",
            sa.Integer,
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "support_provided_count",
            sa.Integer,
            server_default="0",
            nullable=False,
        ),
        # Context
        sa.Column(
            "notes_considered",
            postgresql.JSONB,
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "topics_discussed",
            postgresql.JSONB,
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "activities_suggested",
            postgresql.JSONB,
            server_default="[]",
            nullable=False,
        ),
        # Timing
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_interaction_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Metadata
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Constraints
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "rapport_score >= 0 AND rapport_score <= 1",
            name="companion_sessions_valid_rapport",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'paused', 'ended')",
            name="companion_sessions_valid_status",
        ),
    )

    # Indexes
    op.create_index(
        "ix_companion_sessions_student_id",
        "companion_sessions",
        ["student_id"],
    )
    op.create_index(
        "ix_companion_sessions_student_status",
        "companion_sessions",
        ["student_id", "status"],
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "ix_companion_sessions_conversation",
        "companion_sessions",
        ["conversation_id"],
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )
    op.create_index(
        "ix_companion_sessions_student_started",
        "companion_sessions",
        ["student_id", sa.text("started_at DESC")],
    )


def downgrade() -> None:
    """Remove companion_sessions table."""

    op.drop_index("ix_companion_sessions_student_started", table_name="companion_sessions")
    op.drop_index("ix_companion_sessions_conversation", table_name="companion_sessions")
    op.drop_index("ix_companion_sessions_student_status", table_name="companion_sessions")
    op.drop_index("ix_companion_sessions_student_id", table_name="companion_sessions")
    op.drop_table("companion_sessions")
