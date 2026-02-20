# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add student notes table for multi-source observations.

This migration adds the student_notes table which stores notes about students
from various sources: parents, teachers, AI agents, counselors, system, etc.

This replaces the originally planned parent_notes table with a more flexible
design that accommodates any source type.

Revision ID: 005_add_student_notes
Revises: 004_add_companion_sessions
Create Date: 2024-12-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_add_student_notes"
down_revision: str = "004_add_companion_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add student_notes table."""

    op.create_table(
        "student_notes",
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
        # Author information (flexible - supports users and non-users)
        sa.Column("source_type", sa.String(30), nullable=False),
        # 'parent', 'teacher', 'counselor', 'ai_agent', 'companion', 'system'
        sa.Column(
            "author_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        # Nullable for AI/system sources
        sa.Column("author_name", sa.String(100), nullable=True),
        # Display name for any source
        # Note content
        sa.Column("note_type", sa.String(50), nullable=False),
        # 'daily_mood', 'concern', 'context', 'achievement', 'preference',
        # 'observation', 'recommendation', 'milestone'
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        # Emotional context (optional)
        sa.Column("reported_emotion", sa.String(30), nullable=True),
        sa.Column("emotion_intensity", sa.String(20), nullable=True),
        # 'low', 'moderate', 'high'
        # Validity period
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        # Linking to related entities
        sa.Column(
            "related_topic_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column(
            "related_session_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        # Processing status
        sa.Column(
            "ai_processed",
            sa.Boolean,
            server_default="false",
            nullable=False,
        ),
        sa.Column("ai_processed_at", sa.DateTime(timezone=True), nullable=True),
        # Visibility control
        sa.Column(
            "visibility",
            sa.String(30),
            server_default="internal",
            nullable=False,
        ),
        # 'internal' (staff only), 'parent_visible', 'student_visible', 'all'
        # Priority/importance
        sa.Column(
            "priority",
            sa.String(20),
            server_default="normal",
            nullable=False,
        ),
        # 'low', 'normal', 'high', 'urgent'
        # Flexible extra data
        sa.Column(
            "extra_data",
            postgresql.JSONB,
            server_default="{}",
            nullable=False,
        ),
        # Timestamps
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
            ["author_id"],
            ["users.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_topic_id"],
            ["topics.id"],
            ondelete="SET NULL",
        ),
    )

    # Indexes
    op.create_index(
        "ix_student_notes_student_id",
        "student_notes",
        ["student_id"],
    )
    op.create_index(
        "ix_student_notes_author_id",
        "student_notes",
        ["author_id"],
        postgresql_where=sa.text("author_id IS NOT NULL"),
    )
    op.create_index(
        "ix_student_notes_source_type",
        "student_notes",
        ["source_type"],
    )
    op.create_index(
        "ix_student_notes_note_type",
        "student_notes",
        ["note_type"],
    )
    op.create_index(
        "ix_student_notes_student_created",
        "student_notes",
        ["student_id", sa.text("created_at DESC")],
    )
    # Note: We use a simpler predicate because now() is not immutable
    # Active notes are typically those with valid_until IS NULL
    op.create_index(
        "ix_student_notes_active",
        "student_notes",
        ["student_id", "valid_from", "valid_until"],
        postgresql_where=sa.text("valid_until IS NULL"),
    )
    op.create_index(
        "ix_student_notes_priority",
        "student_notes",
        ["student_id", "priority"],
        postgresql_where=sa.text("priority IN ('high', 'urgent')"),
    )


def downgrade() -> None:
    """Remove student_notes table."""

    op.drop_index("ix_student_notes_priority", table_name="student_notes")
    op.drop_index("ix_student_notes_active", table_name="student_notes")
    op.drop_index("ix_student_notes_student_created", table_name="student_notes")
    op.drop_index("ix_student_notes_note_type", table_name="student_notes")
    op.drop_index("ix_student_notes_source_type", table_name="student_notes")
    op.drop_index("ix_student_notes_author_id", table_name="student_notes")
    op.drop_index("ix_student_notes_student_id", table_name="student_notes")
    op.drop_table("student_notes")
