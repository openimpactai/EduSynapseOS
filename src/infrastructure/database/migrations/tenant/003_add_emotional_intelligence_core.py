# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add emotional intelligence core infrastructure.

This migration adds:
1. emotional_signals table - Real-time emotional signals from all activities
2. New fields to users, episodic_memories, practice_sessions,
   conversation_messages, and alerts tables for emotional tracking

Revision ID: 003_add_emotional_intelligence_core
Revises: 002_add_fsrs_fields
Create Date: 2024-12-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_add_emotional_intelligence_core"
down_revision: str = "002_add_fsrs_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add emotional intelligence tables and fields."""

    # ==========================================================================
    # 1. Create emotional_signals table
    # ==========================================================================
    op.create_table(
        "emotional_signals",
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
        # Signal identification
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("signal_type", sa.String(100), nullable=False),
        # Signal data
        sa.Column("raw_value", postgresql.JSONB, nullable=False),
        sa.Column("detected_emotion", sa.String(30), nullable=True),
        sa.Column("emotion_intensity", sa.Numeric(3, 2), nullable=True),
        sa.Column("emotion_confidence", sa.Numeric(3, 2), nullable=True),
        # Context
        sa.Column("activity_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("activity_type", sa.String(50), nullable=True),
        sa.Column("trigger_context", postgresql.JSONB, nullable=True),
        # Processing
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_method", sa.String(50), nullable=True),
        # Metadata
        sa.Column(
            "created_at",
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
        sa.CheckConstraint(
            "emotion_intensity IS NULL OR (emotion_intensity >= 0 AND emotion_intensity <= 1)",
            name="emotional_signals_valid_intensity",
        ),
        sa.CheckConstraint(
            "emotion_confidence IS NULL OR (emotion_confidence >= 0 AND emotion_confidence <= 1)",
            name="emotional_signals_valid_confidence",
        ),
    )

    # Indexes for emotional_signals
    op.create_index(
        "ix_emotional_signals_student_id",
        "emotional_signals",
        ["student_id"],
    )
    op.create_index(
        "ix_emotional_signals_student_created",
        "emotional_signals",
        ["student_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_emotional_signals_source",
        "emotional_signals",
        ["source"],
    )
    op.create_index(
        "ix_emotional_signals_detected_emotion",
        "emotional_signals",
        ["detected_emotion"],
        postgresql_where=sa.text("detected_emotion IS NOT NULL"),
    )
    op.create_index(
        "ix_emotional_signals_activity",
        "emotional_signals",
        ["activity_id", "activity_type"],
        postgresql_where=sa.text("activity_id IS NOT NULL"),
    )

    # ==========================================================================
    # 2. Add fields to users table
    # ==========================================================================
    op.add_column(
        "users",
        sa.Column(
            "emotional_profile",
            postgresql.JSONB,
            server_default="{}",
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "last_emotional_checkin_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # ==========================================================================
    # 3. Add fields to episodic_memories table
    # emotional_state already exists, adding intensity, trigger, and signal reference
    # ==========================================================================
    op.add_column(
        "episodic_memories",
        sa.Column("emotional_intensity", sa.Numeric(3, 2), nullable=True),
    )
    op.add_column(
        "episodic_memories",
        sa.Column("emotional_trigger", sa.String(100), nullable=True),
    )
    op.add_column(
        "episodic_memories",
        sa.Column(
            "emotional_signal_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_episodic_memories_emotional_signal",
        "episodic_memories",
        "emotional_signals",
        ["emotional_signal_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_episodic_memories_emotional",
        "episodic_memories",
        ["student_id", "emotional_state"],
        postgresql_where=sa.text("emotional_state IS NOT NULL"),
    )

    # ==========================================================================
    # 4. Add fields to practice_sessions table
    # ==========================================================================
    op.add_column(
        "practice_sessions",
        sa.Column("emotional_state_start", sa.String(30), nullable=True),
    )
    op.add_column(
        "practice_sessions",
        sa.Column("emotional_state_end", sa.String(30), nullable=True),
    )
    op.add_column(
        "practice_sessions",
        sa.Column("emotional_flow_quality", sa.Numeric(3, 2), nullable=True),
    )
    op.add_column(
        "practice_sessions",
        sa.Column(
            "emotional_context",
            postgresql.JSONB,
            server_default="{}",
            nullable=False,
        ),
    )

    # ==========================================================================
    # 5. Add fields to conversation_messages table
    # ==========================================================================
    op.add_column(
        "conversation_messages",
        sa.Column("detected_emotion", sa.String(30), nullable=True),
    )
    op.add_column(
        "conversation_messages",
        sa.Column("emotion_confidence", sa.Numeric(3, 2), nullable=True),
    )
    op.add_column(
        "conversation_messages",
        sa.Column(
            "emotion_keywords",
            postgresql.JSONB,
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "conversation_messages",
        sa.Column(
            "is_support_message",
            sa.Boolean,
            server_default="false",
            nullable=False,
        ),
    )

    # ==========================================================================
    # 6. Add fields to alerts table
    # ==========================================================================
    op.add_column(
        "alerts",
        sa.Column(
            "emotional_trigger",
            sa.String(100),
            nullable=True,
        ),
    )
    op.add_column(
        "alerts",
        sa.Column(
            "emotional_signal_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_alerts_emotional_signal",
        "alerts",
        "emotional_signals",
        ["emotional_signal_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_alerts_emotional",
        "alerts",
        ["student_id", "emotional_trigger"],
        postgresql_where=sa.text("emotional_trigger IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove emotional intelligence tables and fields."""

    # 6. Remove fields from alerts
    op.drop_index("ix_alerts_emotional", table_name="alerts")
    op.drop_constraint("fk_alerts_emotional_signal", "alerts", type_="foreignkey")
    op.drop_column("alerts", "emotional_signal_id")
    op.drop_column("alerts", "emotional_trigger")

    # 5. Remove fields from conversation_messages
    op.drop_column("conversation_messages", "is_support_message")
    op.drop_column("conversation_messages", "emotion_keywords")
    op.drop_column("conversation_messages", "emotion_confidence")
    op.drop_column("conversation_messages", "detected_emotion")

    # 4. Remove fields from practice_sessions
    op.drop_column("practice_sessions", "emotional_context")
    op.drop_column("practice_sessions", "emotional_flow_quality")
    op.drop_column("practice_sessions", "emotional_state_end")
    op.drop_column("practice_sessions", "emotional_state_start")

    # 3. Remove fields from episodic_memories
    op.drop_index("ix_episodic_memories_emotional", table_name="episodic_memories")
    op.drop_constraint(
        "fk_episodic_memories_emotional_signal", "episodic_memories", type_="foreignkey"
    )
    op.drop_column("episodic_memories", "emotional_signal_id")
    op.drop_column("episodic_memories", "emotional_trigger")
    op.drop_column("episodic_memories", "emotional_intensity")

    # 2. Remove fields from users
    op.drop_column("users", "last_emotional_checkin_at")
    op.drop_column("users", "emotional_profile")

    # 1. Drop emotional_signals table
    op.drop_index("ix_emotional_signals_activity", table_name="emotional_signals")
    op.drop_index("ix_emotional_signals_detected_emotion", table_name="emotional_signals")
    op.drop_index("ix_emotional_signals_source", table_name="emotional_signals")
    op.drop_index("ix_emotional_signals_student_created", table_name="emotional_signals")
    op.drop_index("ix_emotional_signals_student_id", table_name="emotional_signals")
    op.drop_table("emotional_signals")
