# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add new_session_started to learning_sessions completion_reason constraint.

This migration adds 'new_session_started' as a valid completion_reason for
learning sessions. This value is used when a student starts a new learning
session while having an active session - the old session is auto-completed
with this reason to preserve data and enable seamless session transitions.

Revision ID: 015_add_new_session_started_completion_reason
Revises: 014_add_practice_learning_session_link
Create Date: 2026-01-17
"""

from typing import Sequence, Union

from alembic import op

revision: str = "015_add_new_session_started_completion_reason"
down_revision: str = "014_add_practice_learning_session_link"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new_session_started to completion_reason check constraint."""
    # Drop the old constraint
    op.drop_constraint(
        "learning_sessions_completion_reason_check",
        "learning_sessions",
        type_="check",
    )

    # Create new constraint with new_session_started included
    op.create_check_constraint(
        "learning_sessions_completion_reason_check",
        "learning_sessions",
        "completion_reason IS NULL OR completion_reason IN "
        "('user_ended', 'mastery_achieved', 'max_turns', 'error', 'timeout', 'new_session_started')",
    )


def downgrade() -> None:
    """Remove new_session_started from completion_reason check constraint."""
    # Drop the new constraint
    op.drop_constraint(
        "learning_sessions_completion_reason_check",
        "learning_sessions",
        type_="check",
    )

    # Restore original constraint
    op.create_check_constraint(
        "learning_sessions_completion_reason_check",
        "learning_sessions",
        "completion_reason IS NULL OR completion_reason IN "
        "('user_ended', 'mastery_achieved', 'max_turns', 'error', 'timeout')",
    )
