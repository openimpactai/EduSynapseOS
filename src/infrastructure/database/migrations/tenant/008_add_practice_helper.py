# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add practice helper tables.

This migration creates tables for practice helper tutoring sessions:
- practice_helper_sessions: Tutoring sessions when student needs help
- practice_helper_messages: Conversation messages in tutoring sessions

The practice helper is triggered when a student answers incorrectly
during practice and clicks "Get Help" to understand the concept.

Features:
- Tutoring modes: hint, guided, step_by_step
- Mode escalation tracking
- Understanding progress tracking
- Session completion with reason

Revision ID: 008_add_practice_helper
Revises: 007_add_companion_activities
Create Date: 2025-01-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008_add_practice_helper"
down_revision: str = "007_add_companion_activities"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create practice_helper_sessions and practice_helper_messages tables."""

    # Create practice_helper_sessions table
    op.create_table(
        "practice_helper_sessions",
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
            "practice_session_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "practice_question_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            sa.String(50),
            nullable=False,
        ),
        sa.Column(
            "initial_mode",
            sa.String(20),
            nullable=False,
            server_default="hint",
        ),
        sa.Column(
            "final_mode",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "mode_escalations",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "turn_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "current_step",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "total_steps",
            sa.Integer,
            nullable=True,
        ),
        sa.Column(
            "understanding_progress",
            sa.Numeric(3, 2),
            nullable=False,
            server_default="0.0",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "ended_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "time_spent_seconds",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "completion_reason",
            sa.String(30),
            nullable=True,
        ),
        sa.Column(
            "understood",
            sa.Boolean,
            nullable=True,
        ),
        sa.Column(
            "wants_retry",
            sa.Boolean,
            nullable=True,
        ),
        sa.Column(
            "subject",
            sa.String(50),
            nullable=False,
        ),
        sa.Column(
            "topic_name",
            sa.String(200),
            nullable=False,
        ),
        sa.Column(
            "question_type",
            sa.String(30),
            nullable=False,
        ),
        sa.Column(
            "checkpoint_data",
            postgresql.JSONB,
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
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        # Foreign keys
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["users.id"],
            name="fk_practice_helper_sessions_student",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["practice_session_id"],
            ["practice_sessions.id"],
            name="fk_practice_helper_sessions_practice_session",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["practice_question_id"],
            ["practice_questions.id"],
            name="fk_practice_helper_sessions_practice_question",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'abandoned', 'expired')",
            name="valid_helper_session_status",
        ),
        sa.CheckConstraint(
            "initial_mode IN ('hint', 'guided', 'step_by_step')",
            name="valid_initial_mode",
        ),
        sa.CheckConstraint(
            "final_mode IS NULL OR final_mode IN ('hint', 'guided', 'step_by_step')",
            name="valid_final_mode",
        ),
        sa.CheckConstraint(
            "completion_reason IS NULL OR completion_reason IN ('understood', 'max_turns', 'timeout', 'user_ended')",
            name="valid_completion_reason",
        ),
    )

    # Indexes for practice_helper_sessions
    op.create_index(
        "idx_practice_helper_sessions_student",
        "practice_helper_sessions",
        ["student_id"],
    )
    op.create_index(
        "idx_practice_helper_sessions_practice_session",
        "practice_helper_sessions",
        ["practice_session_id"],
    )
    op.create_index(
        "idx_practice_helper_sessions_practice_question",
        "practice_helper_sessions",
        ["practice_question_id"],
    )
    op.create_index(
        "idx_practice_helper_sessions_status",
        "practice_helper_sessions",
        ["status"],
    )
    op.create_index(
        "idx_practice_helper_sessions_student_active",
        "practice_helper_sessions",
        ["student_id", "status"],
        postgresql_where=sa.text("status = 'active'"),
    )

    # Create practice_helper_messages table
    op.create_table(
        "practice_helper_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "sequence",
            sa.Integer,
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.String(10),
            nullable=False,
        ),
        sa.Column(
            "content",
            sa.Text,
            nullable=False,
        ),
        sa.Column(
            "action",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "mode_at_time",
            sa.String(20),
            nullable=False,
        ),
        sa.Column(
            "step_at_time",
            sa.Integer,
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        # Foreign keys
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["practice_helper_sessions.id"],
            name="fk_practice_helper_messages_session",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "role IN ('student', 'tutor')",
            name="valid_message_role",
        ),
        sa.CheckConstraint(
            "action IS NULL OR action IN ('respond', 'next_step', 'show_me', 'i_understand', 'end')",
            name="valid_message_action",
        ),
    )

    # Indexes for practice_helper_messages
    op.create_index(
        "idx_practice_helper_messages_session",
        "practice_helper_messages",
        ["session_id"],
    )
    op.create_index(
        "idx_practice_helper_messages_session_sequence",
        "practice_helper_messages",
        ["session_id", "sequence"],
    )


def downgrade() -> None:
    """Drop practice_helper_messages and practice_helper_sessions tables."""
    # Drop messages table first (has FK to sessions)
    op.drop_index(
        "idx_practice_helper_messages_session_sequence",
        table_name="practice_helper_messages",
    )
    op.drop_index(
        "idx_practice_helper_messages_session",
        table_name="practice_helper_messages",
    )
    op.drop_table("practice_helper_messages")

    # Drop sessions table
    op.drop_index(
        "idx_practice_helper_sessions_student_active",
        table_name="practice_helper_sessions",
    )
    op.drop_index(
        "idx_practice_helper_sessions_status",
        table_name="practice_helper_sessions",
    )
    op.drop_index(
        "idx_practice_helper_sessions_practice_question",
        table_name="practice_helper_sessions",
    )
    op.drop_index(
        "idx_practice_helper_sessions_practice_session",
        table_name="practice_helper_sessions",
    )
    op.drop_index(
        "idx_practice_helper_sessions_student",
        table_name="practice_helper_sessions",
    )
    op.drop_table("practice_helper_sessions")
