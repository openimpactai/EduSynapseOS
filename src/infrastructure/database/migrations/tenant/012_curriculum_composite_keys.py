# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Refactor curriculum to use composite keys from Central Curriculum.

This migration restructures the curriculum schema to match Central Curriculum:
- Rename Curriculum to CurriculumFramework with code as primary key
- Add CurriculumStage entity
- Change GradeLevel to composite key (framework_code + stage_code + code)
- Change Subject to composite key (framework_code + code)
- Change Unit to composite key (4 parts)
- Change Topic to composite key (5 parts)
- Change LearningObjective to composite key (6 parts)
- Remove KnowledgeComponent (not in Central Curriculum)
- Update all foreign key references in related tables

IMPORTANT: This is a breaking migration. Existing data will be lost.
This should only be run on fresh databases or after data backup.

Revision ID: 012_curriculum_composite_keys
Revises: 011_add_gaming_tables
Create Date: 2026-01-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "012_curriculum_composite_keys"
down_revision: str = "011_add_gaming_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Restructure curriculum to use composite keys."""
    # ==========================================================================
    # STEP 1: Drop foreign key constraints from referencing tables
    # ==========================================================================

    # Drop FK from practice_sessions
    op.drop_constraint("fk_practice_sessions_topic_id", "practice_sessions", type_="foreignkey")

    # Drop FK from practice_questions (objective_id, knowledge_component_id)
    op.drop_constraint("fk_practice_questions_objective_id", "practice_questions", type_="foreignkey")
    try:
        op.drop_constraint("fk_practice_questions_kc_id", "practice_questions", type_="foreignkey")
    except Exception:
        pass  # May not exist

    # Drop FK from learning_sessions
    op.drop_constraint("fk_learning_sessions_topic_id", "learning_sessions", type_="foreignkey")

    # Drop FK from episodic_memories
    try:
        op.drop_constraint("fk_episodic_memories_topic_id", "episodic_memories", type_="foreignkey")
    except Exception:
        pass

    # Drop FK from semantic_memories
    try:
        op.drop_constraint("fk_semantic_memories_entity_id", "semantic_memories", type_="foreignkey")
    except Exception:
        pass

    # Drop FK from procedural_memories
    try:
        op.drop_constraint("fk_procedural_memories_subject_id", "procedural_memories", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_procedural_memories_topic_id", "procedural_memories", type_="foreignkey")
    except Exception:
        pass

    # Drop FK from classes (grade_level_id)
    op.drop_constraint("fk_classes_grade_level_id", "classes", type_="foreignkey")

    # Drop FK from class_teachers (subject_id)
    op.drop_constraint("fk_class_teachers_subject_id", "class_teachers", type_="foreignkey")

    # ==========================================================================
    # STEP 2: Drop old curriculum tables
    # ==========================================================================

    # Drop in reverse order of dependencies
    try:
        op.drop_table("knowledge_components")
    except Exception:
        pass

    op.drop_table("prerequisites")
    op.drop_table("learning_objectives")
    op.drop_table("topics")
    op.drop_table("units")
    op.drop_table("subjects")
    op.drop_table("grade_levels")
    op.drop_table("curricula")

    # ==========================================================================
    # STEP 3: Create new curriculum tables with composite keys
    # ==========================================================================

    # CurriculumFramework (was Curriculum)
    op.create_table(
        "curriculum_frameworks",
        sa.Column("code", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("framework_type", sa.String(50), nullable=False, server_default="national"),
        sa.Column("country_code", sa.String(3), nullable=True),
        sa.Column("organization", sa.String(200), nullable=True),
        sa.Column("version", sa.String(50), nullable=True),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_published", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # CurriculumStage (new)
    op.create_table(
        "curriculum_stages",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("age_start", sa.Integer, nullable=True),
        sa.Column("age_end", sa.Integer, nullable=True),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code"],
            ["curriculum_frameworks.code"],
            name="fk_stages_framework",
            ondelete="CASCADE",
        ),
    )

    # GradeLevel
    op.create_table(
        "grade_levels",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("stage_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("typical_age", sa.Integer, nullable=True),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "stage_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code", "stage_code"],
            ["curriculum_stages.framework_code", "curriculum_stages.code"],
            name="fk_grades_stage",
            ondelete="CASCADE",
        ),
    )

    # Subject
    op.create_table(
        "subjects",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("icon", sa.String(50), nullable=True),
        sa.Column("color", sa.String(20), nullable=True),
        sa.Column("is_core", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code"],
            ["curriculum_frameworks.code"],
            name="fk_subjects_framework",
            ondelete="CASCADE",
        ),
    )

    # Unit
    op.create_table(
        "units",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("subject_code", sa.String(50), nullable=False),
        sa.Column("grade_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("estimated_hours", sa.Integer, nullable=True),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "subject_code", "grade_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code", "subject_code"],
            ["subjects.framework_code", "subjects.code"],
            name="fk_units_subject",
            ondelete="CASCADE",
        ),
    )

    # Topic
    op.create_table(
        "topics",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("subject_code", sa.String(50), nullable=False),
        sa.Column("grade_code", sa.String(50), nullable=False),
        sa.Column("unit_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("base_difficulty", sa.Numeric(3, 2), nullable=False, server_default="0.5"),
        sa.Column("estimated_minutes", sa.Integer, nullable=True),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "subject_code", "grade_code", "unit_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code", "subject_code", "grade_code", "unit_code"],
            ["units.framework_code", "units.subject_code", "units.grade_code", "units.code"],
            name="fk_topics_unit",
            ondelete="CASCADE",
        ),
    )

    # LearningObjective
    op.create_table(
        "learning_objectives",
        sa.Column("framework_code", sa.String(50), nullable=False),
        sa.Column("subject_code", sa.String(50), nullable=False),
        sa.Column("grade_code", sa.String(50), nullable=False),
        sa.Column("unit_code", sa.String(50), nullable=False),
        sa.Column("topic_code", sa.String(50), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("objective", sa.Text, nullable=False),
        sa.Column("bloom_level", sa.String(50), nullable=True),
        sa.Column("mastery_threshold", sa.Numeric(3, 2), nullable=False, server_default="0.8"),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("framework_code", "subject_code", "grade_code", "unit_code", "topic_code", "code"),
        sa.ForeignKeyConstraint(
            ["framework_code", "subject_code", "grade_code", "unit_code", "topic_code"],
            ["topics.framework_code", "topics.subject_code", "topics.grade_code", "topics.unit_code", "topics.code"],
            name="fk_objectives_topic",
            ondelete="CASCADE",
        ),
    )

    # Prerequisite
    op.create_table(
        "prerequisites",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_framework_code", sa.String(50), nullable=False),
        sa.Column("source_subject_code", sa.String(50), nullable=False),
        sa.Column("source_grade_code", sa.String(50), nullable=False),
        sa.Column("source_unit_code", sa.String(50), nullable=False),
        sa.Column("source_topic_code", sa.String(50), nullable=False),
        sa.Column("target_framework_code", sa.String(50), nullable=False),
        sa.Column("target_subject_code", sa.String(50), nullable=False),
        sa.Column("target_grade_code", sa.String(50), nullable=False),
        sa.Column("target_unit_code", sa.String(50), nullable=False),
        sa.Column("target_topic_code", sa.String(50), nullable=False),
        sa.Column("strength", sa.Numeric(3, 2), nullable=False, server_default="1.0"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["source_framework_code", "source_subject_code", "source_grade_code", "source_unit_code", "source_topic_code"],
            ["topics.framework_code", "topics.subject_code", "topics.grade_code", "topics.unit_code", "topics.code"],
            name="fk_prereq_source_topic",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_framework_code", "target_subject_code", "target_grade_code", "target_unit_code", "target_topic_code"],
            ["topics.framework_code", "topics.subject_code", "topics.grade_code", "topics.unit_code", "topics.code"],
            name="fk_prereq_target_topic",
            ondelete="CASCADE",
        ),
    )

    # ==========================================================================
    # STEP 4: Update referencing tables
    # ==========================================================================

    # Update practice_sessions: Replace topic_id with composite keys
    op.drop_column("practice_sessions", "topic_id")
    op.add_column("practice_sessions", sa.Column("topic_framework_code", sa.String(50), nullable=True))
    op.add_column("practice_sessions", sa.Column("topic_subject_code", sa.String(50), nullable=True))
    op.add_column("practice_sessions", sa.Column("topic_grade_code", sa.String(50), nullable=True))
    op.add_column("practice_sessions", sa.Column("topic_unit_code", sa.String(50), nullable=True))
    op.add_column("practice_sessions", sa.Column("topic_code", sa.String(50), nullable=True))

    # Update practice_questions: Replace objective_id, knowledge_component_id
    op.drop_column("practice_questions", "objective_id")
    try:
        op.drop_column("practice_questions", "knowledge_component_id")
    except Exception:
        pass
    op.add_column("practice_questions", sa.Column("objective_framework_code", sa.String(50), nullable=True))
    op.add_column("practice_questions", sa.Column("objective_subject_code", sa.String(50), nullable=True))
    op.add_column("practice_questions", sa.Column("objective_grade_code", sa.String(50), nullable=True))
    op.add_column("practice_questions", sa.Column("objective_unit_code", sa.String(50), nullable=True))
    op.add_column("practice_questions", sa.Column("objective_topic_code", sa.String(50), nullable=True))
    op.add_column("practice_questions", sa.Column("objective_code", sa.String(50), nullable=True))

    # Update learning_sessions: Replace topic_id
    op.drop_column("learning_sessions", "topic_id")
    op.add_column("learning_sessions", sa.Column("topic_framework_code", sa.String(50), nullable=True))
    op.add_column("learning_sessions", sa.Column("topic_subject_code", sa.String(50), nullable=True))
    op.add_column("learning_sessions", sa.Column("topic_grade_code", sa.String(50), nullable=True))
    op.add_column("learning_sessions", sa.Column("topic_unit_code", sa.String(50), nullable=True))
    op.add_column("learning_sessions", sa.Column("topic_code", sa.String(50), nullable=True))

    # Update episodic_memories: Replace topic_id
    try:
        op.drop_column("episodic_memories", "topic_id")
    except Exception:
        pass
    op.add_column("episodic_memories", sa.Column("topic_framework_code", sa.String(50), nullable=True))
    op.add_column("episodic_memories", sa.Column("topic_subject_code", sa.String(50), nullable=True))
    op.add_column("episodic_memories", sa.Column("topic_grade_code", sa.String(50), nullable=True))
    op.add_column("episodic_memories", sa.Column("topic_unit_code", sa.String(50), nullable=True))
    op.add_column("episodic_memories", sa.Column("topic_code", sa.String(50), nullable=True))

    # Update semantic_memories: Replace entity_id with entity_full_code
    try:
        op.drop_column("semantic_memories", "entity_id")
    except Exception:
        pass
    op.add_column("semantic_memories", sa.Column("entity_full_code", sa.String(300), nullable=False, server_default=""))
    op.alter_column("semantic_memories", "entity_full_code", server_default=None)

    # Update procedural_memories: Replace subject_id, topic_id
    try:
        op.drop_column("procedural_memories", "subject_id")
    except Exception:
        pass
    try:
        op.drop_column("procedural_memories", "topic_id")
    except Exception:
        pass
    op.add_column("procedural_memories", sa.Column("subject_framework_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("subject_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("topic_framework_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("topic_subject_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("topic_grade_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("topic_unit_code", sa.String(50), nullable=True))
    op.add_column("procedural_memories", sa.Column("topic_code", sa.String(50), nullable=True))

    # Update classes: Replace grade_level_id
    op.drop_column("classes", "grade_level_id")
    op.add_column("classes", sa.Column("framework_code", sa.String(50), nullable=True))
    op.add_column("classes", sa.Column("stage_code", sa.String(50), nullable=True))
    op.add_column("classes", sa.Column("grade_code", sa.String(50), nullable=True))

    # Update class_teachers: Replace subject_id
    op.drop_column("class_teachers", "subject_id")
    op.add_column("class_teachers", sa.Column("subject_framework_code", sa.String(50), nullable=True))
    op.add_column("class_teachers", sa.Column("subject_code", sa.String(50), nullable=True))

    # ==========================================================================
    # STEP 5: Create indexes
    # ==========================================================================

    op.create_index("ix_topics_subject", "topics", ["framework_code", "subject_code"])
    op.create_index("ix_topics_grade", "topics", ["framework_code", "grade_code"])
    op.create_index("ix_objectives_topic", "learning_objectives", ["framework_code", "subject_code", "grade_code", "unit_code", "topic_code"])
    op.create_index("ix_semantic_memories_entity_full_code", "semantic_memories", ["entity_full_code"])


def downgrade() -> None:
    """Revert curriculum schema changes.

    WARNING: This will NOT restore data. Only schema structure is reverted.
    """
    # Drop new curriculum tables
    op.drop_table("prerequisites")
    op.drop_table("learning_objectives")
    op.drop_table("topics")
    op.drop_table("units")
    op.drop_table("subjects")
    op.drop_table("grade_levels")
    op.drop_table("curriculum_stages")
    op.drop_table("curriculum_frameworks")

    # Restore original curriculum structure is not implemented
    # as this is a breaking migration
    raise NotImplementedError(
        "Downgrade not fully implemented. This migration requires data backup. "
        "Please restore from backup if needed."
    )
