# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Initial tenant database schema.

Revision ID: 001_tenant_initial
Revises: None
Create Date: 2024-12-24
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_tenant_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = ("tenant",)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create tenant database tables."""
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # =========================================================================
    # USER MANAGEMENT TABLES
    # =========================================================================

    # Create roles table
    op.create_table(
        "roles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(50), unique=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_system", sa.Boolean, nullable=False, server_default="false"),
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
    )

    # Create permissions table
    op.create_table(
        "permissions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("code", sa.String(100), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("category", sa.String(50), nullable=False),
    )
    op.create_index("ix_permissions_category", "permissions", ["category"])

    # Create role_permissions junction table
    op.create_table(
        "role_permissions",
        sa.Column(
            "role_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "permission_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("permissions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create users table
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("user_type", sa.String(20), nullable=False),
        sa.Column("avatar_url", sa.Text, nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("birth_date", sa.Date, nullable=True),
        sa.Column("gender", sa.String(10), nullable=True),
        sa.Column("grade_level", sa.Integer, nullable=True),
        sa.Column("school_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("language", sa.String(10), nullable=False, server_default="'tr'"),
        sa.Column("timezone", sa.String(50), nullable=False, server_default="'Europe/Istanbul'"),
        sa.Column("metadata_", postgresql.JSONB, nullable=False, server_default="{}"),
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_user_type", "users", ["user_type"])
    op.create_index("ix_users_school_id", "users", ["school_id"])

    # Create user_roles junction table
    op.create_table(
        "user_roles",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "role_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "assigned_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # =========================================================================
    # SESSION MANAGEMENT TABLES
    # =========================================================================

    # Create user_sessions table
    op.create_table(
        "user_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("device_type", sa.String(20), nullable=True),
        sa.Column("device_name", sa.String(100), nullable=True),
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_token_hash", "user_sessions", ["token_hash"])
    op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"])

    # Create refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("user_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revoked", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"])

    # Create password_reset_tokens table
    op.create_table(
        "password_reset_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_used", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])
    op.create_index("ix_password_reset_tokens_token_hash", "password_reset_tokens", ["token_hash"])

    # Create email_verifications table
    op.create_table(
        "email_verifications",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("token_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_email_verifications_user_id", "email_verifications", ["user_id"])
    op.create_index("ix_email_verifications_token_hash", "email_verifications", ["token_hash"])

    # =========================================================================
    # SCHOOL STRUCTURE TABLES
    # =========================================================================

    # Create schools table
    op.create_table(
        "schools",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), unique=True, nullable=True),
        sa.Column("school_type", sa.String(30), nullable=False),
        sa.Column("address", sa.Text, nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("district", sa.String(100), nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column(
            "principal_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
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
    )

    # Add foreign key to users.school_id
    op.create_foreign_key(
        "fk_users_school_id",
        "users",
        "schools",
        ["school_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create academic_years table
    op.create_table(
        "academic_years",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("start_date", sa.Date, nullable=False),
        sa.Column("end_date", sa.Date, nullable=False),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create classes table
    op.create_table(
        "classes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "school_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("schools.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "academic_year_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("academic_years.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("grade_level", sa.Integer, nullable=False),
        sa.Column("section", sa.String(10), nullable=True),
        sa.Column("capacity", sa.Integer, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
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
    )
    op.create_index("ix_classes_school_id", "classes", ["school_id"])
    op.create_index("ix_classes_academic_year_id", "classes", ["academic_year_id"])

    # Create class_students junction table
    op.create_table(
        "class_students",
        sa.Column(
            "class_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("classes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("student_number", sa.String(20), nullable=True),
        sa.Column(
            "enrolled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )

    # Create class_teachers junction table
    op.create_table(
        "class_teachers",
        sa.Column(
            "class_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("classes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "teacher_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "subject_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "assigned_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create parent_student_relations table
    op.create_table(
        "parent_student_relations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relationship_type", sa.String(20), nullable=False, server_default="'parent'"),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("can_view_grades", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("can_view_progress", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("can_contact_teachers", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("parent_id", "student_id", name="unique_parent_student"),
    )
    op.create_index("ix_parent_student_relations_parent_id", "parent_student_relations", ["parent_id"])
    op.create_index("ix_parent_student_relations_student_id", "parent_student_relations", ["student_id"])

    # =========================================================================
    # CURRICULUM TABLES
    # =========================================================================

    # Create curricula table
    op.create_table(
        "curricula",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), unique=True, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("country", sa.String(2), nullable=False, server_default="'TR'"),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
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
    )

    # Create grade_levels table
    op.create_table(
        "grade_levels",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "curriculum_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("curricula.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("grade_number", sa.Integer, nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("age_range_start", sa.Integer, nullable=True),
        sa.Column("age_range_end", sa.Integer, nullable=True),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.UniqueConstraint("curriculum_id", "grade_number", name="unique_curriculum_grade"),
    )
    op.create_index("ix_grade_levels_curriculum_id", "grade_levels", ["curriculum_id"])

    # Create subjects table
    op.create_table(
        "subjects",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "grade_level_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("grade_levels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("weekly_hours", sa.Integer, nullable=True),
        sa.Column("color", sa.String(7), nullable=True),
        sa.Column("icon", sa.String(50), nullable=True),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_elective", sa.Boolean, nullable=False, server_default="false"),
        sa.UniqueConstraint("grade_level_id", "code", name="unique_grade_subject"),
    )
    op.create_index("ix_subjects_grade_level_id", "subjects", ["grade_level_id"])

    # Add foreign key for class_teachers.subject_id
    op.create_foreign_key(
        "fk_class_teachers_subject_id",
        "class_teachers",
        "subjects",
        ["subject_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create units table
    op.create_table(
        "units",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "subject_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("subjects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("estimated_hours", sa.Integer, nullable=True),
        sa.UniqueConstraint("subject_id", "code", name="unique_subject_unit"),
    )
    op.create_index("ix_units_subject_id", "units", ["subject_id"])

    # Create topics table
    op.create_table(
        "topics",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "unit_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("units.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("difficulty_level", sa.Integer, nullable=False, server_default="1"),
        sa.Column("estimated_minutes", sa.Integer, nullable=True),
        sa.Column("content", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint("unit_id", "code", name="unique_unit_topic"),
    )
    op.create_index("ix_topics_unit_id", "topics", ["unit_id"])

    # Create learning_objectives table
    op.create_table(
        "learning_objectives",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("bloom_level", sa.String(20), nullable=False),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_core", sa.Boolean, nullable=False, server_default="true"),
        sa.UniqueConstraint("topic_id", "code", name="unique_topic_objective"),
    )
    op.create_index("ix_learning_objectives_topic_id", "learning_objectives", ["topic_id"])

    # Create knowledge_components table
    op.create_table(
        "knowledge_components",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "objective_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("learning_objectives.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("kc_type", sa.String(20), nullable=False, server_default="'concept'"),
        sa.Column("difficulty", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column("discrimination", sa.Numeric(3, 2), nullable=False, server_default="1.0"),
        sa.Column("guessing", sa.Numeric(3, 2), nullable=False, server_default="0.25"),
    )
    op.create_index("ix_knowledge_components_objective_id", "knowledge_components", ["objective_id"])

    # Create prerequisites table
    op.create_table(
        "prerequisites",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "prerequisite_topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("strength", sa.Numeric(3, 2), nullable=False, server_default="1.0"),
        sa.Column("is_hard", sa.Boolean, nullable=False, server_default="true"),
        sa.UniqueConstraint("topic_id", "prerequisite_topic_id", name="unique_prerequisite"),
    )
    op.create_index("ix_prerequisites_topic_id", "prerequisites", ["topic_id"])
    op.create_index("ix_prerequisites_prerequisite_topic_id", "prerequisites", ["prerequisite_topic_id"])

    # =========================================================================
    # PRACTICE & ASSESSMENT TABLES
    # =========================================================================

    # Create practice_sessions table
    op.create_table(
        "practice_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("session_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="'active'"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_questions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("correct_answers", sa.Integer, nullable=False, server_default="0"),
        sa.Column("score", sa.Numeric(5, 2), nullable=True),
        sa.Column("time_spent_seconds", sa.Integer, nullable=False, server_default="0"),
        sa.Column("settings", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("metadata_", postgresql.JSONB, nullable=False, server_default="{}"),
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
    )
    op.create_index("ix_practice_sessions_student_id", "practice_sessions", ["student_id"])
    op.create_index("ix_practice_sessions_topic_id", "practice_sessions", ["topic_id"])
    op.create_index("ix_practice_sessions_status", "practice_sessions", ["status"])
    op.create_index("ix_practice_sessions_started_at", "practice_sessions", ["started_at"])

    # Create practice_questions table
    op.create_table(
        "practice_questions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("practice_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column(
            "objective_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("learning_objectives.id"),
            nullable=True,
        ),
        sa.Column("question_order", sa.Integer, nullable=False),
        sa.Column("question_type", sa.String(30), nullable=False),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("question_data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("correct_answer", postgresql.JSONB, nullable=False),
        sa.Column("difficulty", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column("points", sa.Integer, nullable=False, server_default="1"),
        sa.Column("time_limit_seconds", sa.Integer, nullable=True),
        sa.Column("hints", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_practice_questions_session_id", "practice_questions", ["session_id"])

    # Create student_answers table
    op.create_table(
        "student_answers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "question_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("practice_questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("answer", postgresql.JSONB, nullable=False),
        sa.Column("is_correct", sa.Boolean, nullable=True),
        sa.Column("partial_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("time_spent_seconds", sa.Integer, nullable=True),
        sa.Column("hints_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("attempt_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("feedback", sa.Text, nullable=True),
        sa.Column(
            "answered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_student_answers_question_id", "student_answers", ["question_id"])
    op.create_index("ix_student_answers_student_id", "student_answers", ["student_id"])

    # Create evaluation_results table
    op.create_table(
        "evaluation_results",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "answer_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("student_answers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evaluator_type", sa.String(20), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=False),
        sa.Column("rubric_scores", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("feedback", sa.Text, nullable=True),
        sa.Column("strengths", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("weaknesses", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("suggestions", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("raw_response", postgresql.JSONB, nullable=True),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_evaluation_results_answer_id", "evaluation_results", ["answer_id"])

    # =========================================================================
    # CONVERSATION TABLES
    # =========================================================================

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column("persona", sa.String(50), nullable=False, server_default="'tutor'"),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="'active'"),
        sa.Column("message_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("context", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("metadata_", postgresql.JSONB, nullable=False, server_default="{}"),
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
    )
    op.create_index("ix_conversations_student_id", "conversations", ["student_id"])
    op.create_index("ix_conversations_topic_id", "conversations", ["topic_id"])
    op.create_index("ix_conversations_status", "conversations", ["status"])

    # Create conversation_messages table
    op.create_table(
        "conversation_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_type", sa.String(20), nullable=False, server_default="'text'"),
        sa.Column("message_metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("tokens_used", sa.Integer, nullable=True),
        sa.Column("model_used", sa.String(50), nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_conversation_messages_conversation_id", "conversation_messages", ["conversation_id"])
    op.create_index("ix_conversation_messages_created_at", "conversation_messages", ["created_at"])

    # Create conversation_summaries table
    op.create_table(
        "conversation_summaries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary_type", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("key_points", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("topics_covered", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("learning_outcomes", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("message_range_start", sa.Integer, nullable=False),
        sa.Column("message_range_end", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_conversation_summaries_conversation_id", "conversation_summaries", ["conversation_id"])

    # =========================================================================
    # MEMORY TABLES
    # =========================================================================

    # Create episodic_memories table
    op.create_table(
        "episodic_memories",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("context", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("emotional_valence", sa.Numeric(3, 2), nullable=True),
        sa.Column("importance_score", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("embedding", postgresql.JSONB, nullable=True),
        sa.Column("access_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decay_factor", sa.Numeric(5, 4), nullable=False, server_default="1.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_episodic_memories_student_id", "episodic_memories", ["student_id"])
    op.create_index("ix_episodic_memories_event_type", "episodic_memories", ["event_type"])
    op.create_index("ix_episodic_memories_topic_id", "episodic_memories", ["topic_id"])

    # Create semantic_memories table
    op.create_table(
        "semantic_memories",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("concept", sa.String(255), nullable=False),
        sa.Column("definition", sa.Text, nullable=False),
        sa.Column("examples", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("relationships", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column("source_type", sa.String(30), nullable=True),
        sa.Column("embedding", postgresql.JSONB, nullable=True),
        sa.Column("reinforcement_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_reinforced_at", sa.DateTime(timezone=True), nullable=True),
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
    )
    op.create_index("ix_semantic_memories_student_id", "semantic_memories", ["student_id"])
    op.create_index("ix_semantic_memories_concept", "semantic_memories", ["concept"])
    op.create_index("ix_semantic_memories_topic_id", "semantic_memories", ["topic_id"])

    # Create procedural_memories table
    op.create_table(
        "procedural_memories",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("skill_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("steps", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("common_errors", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("tips", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("proficiency", sa.Numeric(3, 2), nullable=False, server_default="0.0"),
        sa.Column("practice_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("success_rate", sa.Numeric(3, 2), nullable=True),
        sa.Column("last_practiced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("automaticity_level", sa.Numeric(3, 2), nullable=False, server_default="0.0"),
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
    )
    op.create_index("ix_procedural_memories_student_id", "procedural_memories", ["student_id"])
    op.create_index("ix_procedural_memories_skill_name", "procedural_memories", ["skill_name"])
    op.create_index("ix_procedural_memories_topic_id", "procedural_memories", ["topic_id"])

    # Create associative_memories table
    op.create_table(
        "associative_memories",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_concept", sa.String(255), nullable=False),
        sa.Column("target_concept", sa.String(255), nullable=False),
        sa.Column("association_type", sa.String(30), nullable=False),
        sa.Column("strength", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column("context", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("bidirectional", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("activation_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_activated_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint(
            "student_id", "source_concept", "target_concept", "association_type",
            name="unique_association"
        ),
    )
    op.create_index("ix_associative_memories_student_id", "associative_memories", ["student_id"])
    op.create_index("ix_associative_memories_source_concept", "associative_memories", ["source_concept"])
    op.create_index("ix_associative_memories_target_concept", "associative_memories", ["target_concept"])

    # =========================================================================
    # SPACED REPETITION TABLES
    # =========================================================================

    # Create review_items table
    op.create_table(
        "review_items",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("item_type", sa.String(30), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("stability", sa.Numeric(10, 6), nullable=False, server_default="1.0"),
        sa.Column("difficulty", sa.Numeric(5, 4), nullable=False, server_default="0.3"),
        sa.Column("elapsed_days", sa.Numeric(10, 4), nullable=False, server_default="0"),
        sa.Column("scheduled_days", sa.Numeric(10, 4), nullable=False, server_default="1"),
        sa.Column("reps", sa.Integer, nullable=False, server_default="0"),
        sa.Column("lapses", sa.Integer, nullable=False, server_default="0"),
        sa.Column("state", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_review", sa.DateTime(timezone=True), nullable=True),
        sa.Column("due", sa.DateTime(timezone=True), nullable=False),
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
        sa.UniqueConstraint("student_id", "item_type", "item_id", name="unique_review_item"),
    )
    op.create_index("ix_review_items_student_id", "review_items", ["student_id"])
    op.create_index("ix_review_items_due", "review_items", ["due"])
    op.create_index("ix_review_items_topic_id", "review_items", ["topic_id"])

    # Create review_logs table
    op.create_table(
        "review_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "item_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("review_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("state", sa.Integer, nullable=False),
        sa.Column("scheduled_days", sa.Numeric(10, 4), nullable=False),
        sa.Column("elapsed_days", sa.Numeric(10, 4), nullable=False),
        sa.Column("stability_before", sa.Numeric(10, 6), nullable=False),
        sa.Column("stability_after", sa.Numeric(10, 6), nullable=False),
        sa.Column("difficulty_before", sa.Numeric(5, 4), nullable=False),
        sa.Column("difficulty_after", sa.Numeric(5, 4), nullable=False),
        sa.Column("review_duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "reviewed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_review_logs_item_id", "review_logs", ["item_id"])
    op.create_index("ix_review_logs_reviewed_at", "review_logs", ["reviewed_at"])

    # =========================================================================
    # DIAGNOSTIC TABLES
    # =========================================================================

    # Create diagnostic_scans table
    op.create_table(
        "diagnostic_scans",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scan_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="'pending'"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_range_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_range_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("findings_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("raw_data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_diagnostic_scans_student_id", "diagnostic_scans", ["student_id"])
    op.create_index("ix_diagnostic_scans_scan_type", "diagnostic_scans", ["scan_type"])
    op.create_index("ix_diagnostic_scans_status", "diagnostic_scans", ["status"])

    # Create diagnostic_indicators table
    op.create_table(
        "diagnostic_indicators",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "scan_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("diagnostic_scans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("indicator_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="'info'"),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("evidence", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_diagnostic_indicators_scan_id", "diagnostic_indicators", ["scan_id"])
    op.create_index("ix_diagnostic_indicators_indicator_type", "diagnostic_indicators", ["indicator_type"])

    # Create diagnostic_recommendations table
    op.create_table(
        "diagnostic_recommendations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "scan_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("diagnostic_scans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "indicator_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("diagnostic_indicators.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("recommendation_type", sa.String(30), nullable=False),
        sa.Column("priority", sa.Integer, nullable=False, server_default="1"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("action_items", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("resources", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("status", sa.String(20), nullable=False, server_default="'pending'"),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_diagnostic_recommendations_scan_id", "diagnostic_recommendations", ["scan_id"])
    op.create_index("ix_diagnostic_recommendations_status", "diagnostic_recommendations", ["status"])

    # =========================================================================
    # NOTIFICATION TABLES
    # =========================================================================

    # Create alerts table
    op.create_table(
        "alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="'info'"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("details", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("suggested_actions", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("status", sa.String(20), nullable=False, server_default="'active'"),
        sa.Column(
            "acknowledged_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_alerts_student_id", "alerts", ["student_id"])
    op.create_index("ix_alerts_alert_type", "alerts", ["alert_type"])

    # Create notifications table
    op.create_table(
        "notifications",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("notification_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("channels", postgresql.JSONB, nullable=False, server_default='["in_app"]'),
        sa.Column("delivery_status", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("action_url", sa.Text, nullable=True),
        sa.Column("action_label", sa.String(100), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])
    op.create_index("ix_notifications_notification_type", "notifications", ["notification_type"])

    # Create notification_preferences table
    op.create_table(
        "notification_preferences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("notification_type", sa.String(50), nullable=False),
        sa.Column("in_app", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("push", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("email", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("sms", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("frequency", sa.String(20), nullable=False, server_default="'immediate'"),
        sa.Column("quiet_start", sa.Time, nullable=True),
        sa.Column("quiet_end", sa.Time, nullable=True),
        sa.Column("is_enabled", sa.Boolean, nullable=False, server_default="true"),
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
        sa.UniqueConstraint("user_id", "notification_type", name="unique_user_notification_type"),
    )
    op.create_index("ix_notification_preferences_user_id", "notification_preferences", ["user_id"])

    # Create notification_templates table
    op.create_table(
        "notification_templates",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("notification_type", sa.String(50), nullable=False),
        sa.Column("language_code", sa.String(10), nullable=False),
        sa.Column("title_template", sa.Text, nullable=False),
        sa.Column("message_template", sa.Text, nullable=False),
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
        sa.UniqueConstraint("notification_type", "language_code", name="unique_template"),
    )

    # =========================================================================
    # ANALYTICS TABLES
    # =========================================================================

    # Create analytics_events table
    op.create_table(
        "analytics_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("student_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column(
            "topic_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("topics.id"),
            nullable=True,
        ),
        sa.Column("data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("device_type", sa.String(20), nullable=True),
        sa.Column("client_version", sa.String(20), nullable=True),
    )
    op.create_index("ix_analytics_events_user_id", "analytics_events", ["user_id"])
    op.create_index("ix_analytics_events_event_type", "analytics_events", ["event_type"])
    op.create_index("ix_analytics_events_occurred_at", "analytics_events", ["occurred_at"])

    # Create daily_summaries table
    op.create_table(
        "daily_summaries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary_date", sa.Date, nullable=False),
        sa.Column("total_time_seconds", sa.Integer, nullable=False, server_default="0"),
        sa.Column("sessions_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("questions_attempted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("questions_correct", sa.Integer, nullable=False, server_default="0"),
        sa.Column("messages_sent", sa.Integer, nullable=False, server_default="0"),
        sa.Column("topics_practiced", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("average_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("engagement_score", sa.Numeric(3, 2), nullable=True),
        sa.Column("daily_streak", sa.Integer, nullable=False, server_default="0"),
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
        sa.UniqueConstraint("student_id", "summary_date", name="unique_student_date"),
    )
    op.create_index("ix_daily_summaries_student_id", "daily_summaries", ["student_id"])
    op.create_index("ix_daily_summaries_summary_date", "daily_summaries", ["summary_date"])

    # Create mastery_snapshots table
    op.create_table(
        "mastery_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("subject_mastery", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("topic_mastery", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("overall_mastery", sa.Numeric(3, 2), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("student_id", "snapshot_date", name="unique_student_snapshot_date"),
    )
    op.create_index("ix_mastery_snapshots_student_id", "mastery_snapshots", ["student_id"])
    op.create_index("ix_mastery_snapshots_snapshot_date", "mastery_snapshots", ["snapshot_date"])

    # Create engagement_metrics table
    op.create_table(
        "engagement_metrics",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "student_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric_date", sa.Date, nullable=False),
        sa.Column("login_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("session_duration_avg", sa.Integer, nullable=False, server_default="0"),
        sa.Column("questions_per_session", sa.Numeric(5, 2), nullable=True),
        sa.Column("accuracy_trend", sa.Numeric(3, 2), nullable=True),
        sa.Column("streak_days", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("student_id", "metric_date", name="unique_student_metric_date"),
    )
    op.create_index("ix_engagement_metrics_student_id", "engagement_metrics", ["student_id"])
    op.create_index("ix_engagement_metrics_metric_date", "engagement_metrics", ["metric_date"])

    # =========================================================================
    # SETTINGS TABLES
    # =========================================================================

    # Create tenant_settings table
    op.create_table(
        "tenant_settings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("setting_key", sa.String(100), unique=True, nullable=False),
        sa.Column("setting_value", postgresql.JSONB, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("value_type", sa.String(20), nullable=False, server_default="'string'"),
        sa.Column("allow_school_override", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("allow_user_override", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create user_preferences table
    op.create_table(
        "user_preferences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("theme", sa.String(20), nullable=False, server_default="'system'"),
        sa.Column("language", sa.String(10), nullable=False, server_default="'tr'"),
        sa.Column("font_size", sa.String(20), nullable=False, server_default="'medium'"),
        sa.Column("high_contrast", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("reduce_motion", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("sound_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("default_session_type", sa.String(30), nullable=True),
        sa.Column("default_persona", sa.String(50), nullable=True),
        sa.Column("preferences", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create feature_flags table
    op.create_table(
        "feature_flags",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("feature_key", sa.String(100), unique=True, nullable=False),
        sa.Column("is_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("rollout_percentage", sa.Integer, nullable=False, server_default="100"),
        sa.Column("user_ids", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # =========================================================================
    # LOCALIZATION TABLES
    # =========================================================================

    # Create languages table
    op.create_table(
        "languages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("code", sa.String(10), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("native_name", sa.String(100), nullable=True),
        sa.Column("is_rtl", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("translation_progress", sa.Integer, nullable=False, server_default="0"),
    )

    # Create translations table
    op.create_table(
        "translations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("field_name", sa.String(50), nullable=False),
        sa.Column("language_code", sa.String(10), nullable=False),
        sa.Column("translated_text", sa.Text, nullable=False),
        sa.Column("is_reviewed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "reviewed_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
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
        sa.UniqueConstraint(
            "entity_type", "entity_id", "field_name", "language_code",
            name="unique_translation"
        ),
    )
    op.create_index("ix_translations_entity_type", "translations", ["entity_type"])
    op.create_index("ix_translations_entity_id", "translations", ["entity_id"])
    op.create_index("ix_translations_language_code", "translations", ["language_code"])

    # =========================================================================
    # AUDIT LOG TABLE
    # =========================================================================

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_email", sa.String(255), nullable=True),
        sa.Column("user_type", sa.String(20), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("old_values", postgresql.JSONB, nullable=True),
        sa.Column("new_values", postgresql.JSONB, nullable=True),
        sa.Column("ip_address", postgresql.INET, nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_entity_type", "audit_logs", ["entity_type"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    """Drop all tenant database tables."""
    # Drop in reverse order to handle foreign keys
    op.drop_table("audit_logs")
    op.drop_table("translations")
    op.drop_table("languages")
    op.drop_table("feature_flags")
    op.drop_table("user_preferences")
    op.drop_table("tenant_settings")
    op.drop_table("engagement_metrics")
    op.drop_table("mastery_snapshots")
    op.drop_table("daily_summaries")
    op.drop_table("analytics_events")
    op.drop_table("notification_templates")
    op.drop_table("notification_preferences")
    op.drop_table("notifications")
    op.drop_table("alerts")
    op.drop_table("diagnostic_recommendations")
    op.drop_table("diagnostic_indicators")
    op.drop_table("diagnostic_scans")
    op.drop_table("review_logs")
    op.drop_table("review_items")
    op.drop_table("associative_memories")
    op.drop_table("procedural_memories")
    op.drop_table("semantic_memories")
    op.drop_table("episodic_memories")
    op.drop_table("conversation_summaries")
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
    op.drop_table("evaluation_results")
    op.drop_table("student_answers")
    op.drop_table("practice_questions")
    op.drop_table("practice_sessions")
    op.drop_table("prerequisites")
    op.drop_table("knowledge_components")
    op.drop_table("learning_objectives")
    op.drop_table("topics")
    op.drop_table("units")
    op.drop_constraint("fk_class_teachers_subject_id", "class_teachers", type_="foreignkey")
    op.drop_table("subjects")
    op.drop_table("grade_levels")
    op.drop_table("curricula")
    op.drop_table("parent_student_relations")
    op.drop_table("class_teachers")
    op.drop_table("class_students")
    op.drop_table("classes")
    op.drop_table("academic_years")
    op.drop_constraint("fk_users_school_id", "users", type_="foreignkey")
    op.drop_table("schools")
    op.drop_table("email_verifications")
    op.drop_table("password_reset_tokens")
    op.drop_table("refresh_tokens")
    op.drop_table("user_sessions")
    op.drop_table("user_roles")
    op.drop_table("users")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
