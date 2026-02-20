# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add learning sessions tables.

This migration creates tables for learning tutor sessions:
- learning_sessions: Proactive teaching sessions for learning new topics
- learning_session_messages: Conversation messages in learning sessions

The learning tutor is triggered when a student wants to learn a new concept,
either via companion handoff, practice "I need to learn", direct access,
LMS deep link, spaced repetition review, or weakness suggestion.

Features:
- Learning modes: discovery, explanation, worked_example, guided_practice, assessment
- Mode transition tracking
- Understanding progress tracking
- Practice and assessment tracking within sessions
- Mastery impact tracking (initial vs final)
- Multiple entry points support

Revision ID: 009_add_learning_sessions
Revises: 008_add_practice_helper
Create Date: 2026-01-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "009_add_learning_sessions"
down_revision: str = "008_add_practice_helper"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create learning_sessions and learning_session_messages tables."""

    # Create learning_sessions table
    op.create_table(
        "learning_sessions",
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
            "topic_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "topic_name",
            sa.String(255),
            nullable=False,
        ),
        sa.Column(
            "subject",
            sa.String(100),
            nullable=True,
        ),
        sa.Column(
            "subject_code",
            sa.String(50),
            nullable=True,
        ),
        sa.Column(
            "agent_id",
            sa.String(50),
            nullable=False,
            server_default="learning_tutor_general",
        ),
        sa.Column(
            "entry_point",
            sa.String(50),
            nullable=False,
            server_default="direct",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "initial_mode",
            sa.String(30),
            nullable=False,
            server_default="explanation",
        ),
        sa.Column(
            "current_mode",
            sa.String(30),
            nullable=False,
            server_default="explanation",
        ),
        sa.Column(
            "mode_transition_count",
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
            "understanding_progress",
            sa.Numeric(3, 2),
            nullable=False,
            server_default="0.00",
        ),
        sa.Column(
            "practice_questions_attempted",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "practice_questions_correct",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "assessment_passed",
            sa.Boolean,
            nullable=True,
        ),
        sa.Column(
            "completion_reason",
            sa.String(50),
            nullable=True,
        ),
        sa.Column(
            "initial_mastery",
            sa.Numeric(3, 2),
            nullable=True,
        ),
        sa.Column(
            "final_mastery",
            sa.Numeric(3, 2),
            nullable=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
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
            name="fk_learning_sessions_student",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["topic_id"],
            ["topics.id"],
            name="fk_learning_sessions_topic",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "status IN ('pending', 'active', 'paused', 'completed', 'error')",
            name="learning_sessions_status_check",
        ),
        sa.CheckConstraint(
            "initial_mode IN ('discovery', 'explanation', 'worked_example', 'guided_practice', 'assessment')",
            name="learning_sessions_initial_mode_check",
        ),
        sa.CheckConstraint(
            "current_mode IN ('discovery', 'explanation', 'worked_example', 'guided_practice', 'assessment')",
            name="learning_sessions_current_mode_check",
        ),
        sa.CheckConstraint(
            "entry_point IN ('companion_handoff', 'practice_help', 'direct', 'lms', 'review', 'weakness')",
            name="learning_sessions_entry_point_check",
        ),
        sa.CheckConstraint(
            "completion_reason IS NULL OR completion_reason IN ('user_ended', 'mastery_achieved', 'max_turns', 'error', 'timeout')",
            name="learning_sessions_completion_reason_check",
        ),
    )

    # Indexes for learning_sessions
    op.create_index(
        "idx_learning_sessions_student",
        "learning_sessions",
        ["student_id"],
    )
    op.create_index(
        "idx_learning_sessions_topic",
        "learning_sessions",
        ["topic_id"],
    )
    op.create_index(
        "idx_learning_sessions_status",
        "learning_sessions",
        ["status"],
    )
    op.create_index(
        "idx_learning_sessions_created",
        "learning_sessions",
        ["created_at"],
        postgresql_using="btree",
    )
    # Partial index for active sessions (prevents multiple active sessions per student)
    op.create_index(
        "idx_learning_sessions_student_active",
        "learning_sessions",
        ["student_id", "status"],
        postgresql_where=sa.text("status IN ('pending', 'active', 'paused')"),
    )
    # Index for entry point filtering
    op.create_index(
        "idx_learning_sessions_entry_point",
        "learning_sessions",
        ["entry_point"],
    )

    # Create learning_session_messages table
    op.create_table(
        "learning_session_messages",
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
            sa.String(30),
            nullable=True,
        ),
        sa.Column(
            "learning_mode",
            sa.String(30),
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
            ["learning_sessions.id"],
            name="fk_learning_session_messages_session",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "role IN ('student', 'tutor')",
            name="learning_session_messages_role_check",
        ),
    )

    # Indexes for learning_session_messages
    op.create_index(
        "idx_learning_session_messages_session",
        "learning_session_messages",
        ["session_id"],
    )
    op.create_index(
        "idx_learning_session_messages_session_sequence",
        "learning_session_messages",
        ["session_id", "sequence"],
    )
    op.create_index(
        "idx_learning_session_messages_created",
        "learning_session_messages",
        ["created_at"],
    )


def downgrade() -> None:
    """Drop learning_session_messages and learning_sessions tables."""
    # Drop messages table first (has FK to sessions)
    op.drop_index(
        "idx_learning_session_messages_created",
        table_name="learning_session_messages",
    )
    op.drop_index(
        "idx_learning_session_messages_session_sequence",
        table_name="learning_session_messages",
    )
    op.drop_index(
        "idx_learning_session_messages_session",
        table_name="learning_session_messages",
    )
    op.drop_table("learning_session_messages")

    # Drop sessions table
    op.drop_index(
        "idx_learning_sessions_entry_point",
        table_name="learning_sessions",
    )
    op.drop_index(
        "idx_learning_sessions_student_active",
        table_name="learning_sessions",
    )
    op.drop_index(
        "idx_learning_sessions_created",
        table_name="learning_sessions",
    )
    op.drop_index(
        "idx_learning_sessions_status",
        table_name="learning_sessions",
    )
    op.drop_index(
        "idx_learning_sessions_topic",
        table_name="learning_sessions",
    )
    op.drop_index(
        "idx_learning_sessions_student",
        table_name="learning_sessions",
    )
    op.drop_table("learning_sessions")
