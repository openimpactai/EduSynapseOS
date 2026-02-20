# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add gaming tables for educational games.

This migration creates tables for the Game Coach system:
- game_sessions: Game session records (chess, connect4, etc.)
- game_moves: Moves played in game sessions
- game_analyses: AI analysis of positions and games

The Game Coach system provides educational gaming experiences
where students can play strategy games (chess, connect4) with
AI opponents while receiving coaching feedback.

Features:
- Multiple game types (chess, connect4)
- Game modes (tutorial, practice, challenge, puzzle, analysis)
- Difficulty levels (beginner to expert)
- Move quality tracking (excellent, good, inaccuracy, mistake, blunder)
- Position analysis with AI coaching
- Learning points extraction

Revision ID: 011_add_gaming_tables
Revises: 010_add_academic_year_code
Create Date: 2026-01-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "011_add_gaming_tables"
down_revision: str = "010_add_academic_year_code"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create game_sessions, game_moves, and game_analyses tables."""

    # =========================================================================
    # Create game_sessions table
    # =========================================================================
    op.create_table(
        "game_sessions",
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
            "game_type",
            sa.String(30),
            nullable=False,
        ),
        sa.Column(
            "game_mode",
            sa.String(30),
            nullable=False,
            server_default="practice",
        ),
        sa.Column(
            "difficulty",
            sa.String(20),
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "player_color",
            sa.String(20),
            nullable=False,
            server_default="white",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "result",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "winner",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "total_moves",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "time_spent_seconds",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "hints_used",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "mistakes_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "excellent_moves_count",
            sa.Integer,
            nullable=False,
            server_default="0",
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
            "initial_position",
            postgresql.JSONB,
            nullable=True,
        ),
        sa.Column(
            "final_position",
            postgresql.JSONB,
            nullable=True,
        ),
        sa.Column(
            "game_state",
            postgresql.JSONB,
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "checkpoint_data",
            postgresql.JSONB,
            nullable=True,
        ),
        sa.Column(
            "coach_summary",
            sa.Text,
            nullable=True,
        ),
        sa.Column(
            "learning_points",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
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
            name="fk_game_sessions_student",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "game_type IN ('chess', 'connect4', 'gomoku', 'othello', 'checkers')",
            name="game_sessions_game_type_check",
        ),
        sa.CheckConstraint(
            "game_mode IN ('tutorial', 'practice', 'challenge', 'puzzle', 'analysis')",
            name="game_sessions_game_mode_check",
        ),
        sa.CheckConstraint(
            "difficulty IN ('beginner', 'easy', 'medium', 'hard', 'expert')",
            name="game_sessions_difficulty_check",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'paused', 'completed', 'abandoned')",
            name="game_sessions_status_check",
        ),
        sa.CheckConstraint(
            "result IS NULL OR result IN ('win', 'loss', 'draw', 'timeout', 'resignation')",
            name="game_sessions_result_check",
        ),
    )

    # Indexes for game_sessions
    op.create_index(
        "idx_game_sessions_student",
        "game_sessions",
        ["student_id"],
    )
    op.create_index(
        "idx_game_sessions_game_type",
        "game_sessions",
        ["game_type"],
    )
    op.create_index(
        "idx_game_sessions_status",
        "game_sessions",
        ["status"],
    )
    op.create_index(
        "idx_game_sessions_created",
        "game_sessions",
        ["created_at"],
        postgresql_using="btree",
    )
    # Composite index for filtering by student and game type
    op.create_index(
        "idx_game_sessions_student_type",
        "game_sessions",
        ["student_id", "game_type"],
    )
    # Partial index for active sessions
    op.create_index(
        "idx_game_sessions_student_active",
        "game_sessions",
        ["student_id", "game_type"],
        postgresql_where=sa.text("status IN ('active', 'paused')"),
    )

    # =========================================================================
    # Create game_moves table
    # =========================================================================
    op.create_table(
        "game_moves",
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
            "move_number",
            sa.Integer,
            nullable=False,
        ),
        sa.Column(
            "player",
            sa.String(20),
            nullable=False,
        ),
        sa.Column(
            "notation",
            sa.String(20),
            nullable=False,
        ),
        sa.Column(
            "position_before",
            postgresql.JSONB,
            nullable=False,
        ),
        sa.Column(
            "position_after",
            postgresql.JSONB,
            nullable=False,
        ),
        sa.Column(
            "time_spent_seconds",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "evaluation_before",
            sa.Numeric(7, 2),
            nullable=True,
        ),
        sa.Column(
            "evaluation_after",
            sa.Numeric(7, 2),
            nullable=True,
        ),
        sa.Column(
            "quality",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "is_best_move",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "best_move",
            sa.String(20),
            nullable=True,
        ),
        sa.Column(
            "coach_comment",
            sa.Text,
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
            ["game_sessions.id"],
            name="fk_game_moves_session",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "player IN ('player', 'ai')",
            name="game_moves_player_check",
        ),
        sa.CheckConstraint(
            "quality IS NULL OR quality IN ('excellent', 'good', 'inaccuracy', 'mistake', 'blunder')",
            name="game_moves_quality_check",
        ),
    )

    # Indexes for game_moves
    op.create_index(
        "idx_game_moves_session",
        "game_moves",
        ["session_id"],
    )
    op.create_index(
        "idx_game_moves_session_number",
        "game_moves",
        ["session_id", "move_number"],
    )
    op.create_index(
        "idx_game_moves_quality",
        "game_moves",
        ["quality"],
        postgresql_where=sa.text("quality IS NOT NULL"),
    )

    # =========================================================================
    # Create game_analyses table
    # =========================================================================
    op.create_table(
        "game_analyses",
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
            "move_number",
            sa.Integer,
            nullable=True,
        ),
        sa.Column(
            "analysis_type",
            sa.String(20),
            nullable=False,
            server_default="position",
        ),
        sa.Column(
            "position",
            postgresql.JSONB,
            nullable=False,
        ),
        sa.Column(
            "evaluation",
            sa.Numeric(7, 2),
            nullable=True,
        ),
        sa.Column(
            "evaluation_text",
            sa.String(200),
            nullable=True,
        ),
        sa.Column(
            "best_moves",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "threats",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "opportunities",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "strategic_themes",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "coach_explanation",
            sa.Text,
            nullable=True,
        ),
        sa.Column(
            "learning_points",
            postgresql.JSONB,
            nullable=False,
            server_default="[]",
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
            ["game_sessions.id"],
            name="fk_game_analyses_session",
            ondelete="CASCADE",
        ),
        # Check constraints
        sa.CheckConstraint(
            "analysis_type IN ('position', 'move', 'phase', 'game', 'opening', 'endgame')",
            name="game_analyses_type_check",
        ),
    )

    # Indexes for game_analyses
    op.create_index(
        "idx_game_analyses_session",
        "game_analyses",
        ["session_id"],
    )
    op.create_index(
        "idx_game_analyses_session_move",
        "game_analyses",
        ["session_id", "move_number"],
    )
    op.create_index(
        "idx_game_analyses_type",
        "game_analyses",
        ["analysis_type"],
    )


def downgrade() -> None:
    """Drop game_analyses, game_moves, and game_sessions tables."""

    # Drop game_analyses table (has FK to game_sessions)
    op.drop_index("idx_game_analyses_type", table_name="game_analyses")
    op.drop_index("idx_game_analyses_session_move", table_name="game_analyses")
    op.drop_index("idx_game_analyses_session", table_name="game_analyses")
    op.drop_table("game_analyses")

    # Drop game_moves table (has FK to game_sessions)
    op.drop_index("idx_game_moves_quality", table_name="game_moves")
    op.drop_index("idx_game_moves_session_number", table_name="game_moves")
    op.drop_index("idx_game_moves_session", table_name="game_moves")
    op.drop_table("game_moves")

    # Drop game_sessions table
    op.drop_index("idx_game_sessions_student_active", table_name="game_sessions")
    op.drop_index("idx_game_sessions_student_type", table_name="game_sessions")
    op.drop_index("idx_game_sessions_created", table_name="game_sessions")
    op.drop_index("idx_game_sessions_status", table_name="game_sessions")
    op.drop_index("idx_game_sessions_game_type", table_name="game_sessions")
    op.drop_index("idx_game_sessions_student", table_name="game_sessions")
    op.drop_table("game_sessions")
