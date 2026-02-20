# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get student mastery tool for teachers.

Returns detailed mastery levels for a specific student,
broken down by subject and topic.

Uses code-based composite keys from Central Curriculum structure.
SemanticMemory stores entity_full_code instead of UUID references.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.infrastructure.database.models.tenant.curriculum import Subject, Topic, Unit
from src.infrastructure.database.models.tenant.memory import SemanticMemory
from src.infrastructure.database.models.tenant.user import User
from src.tools.teacher.helpers import verify_teacher_has_student_access

logger = logging.getLogger(__name__)


class GetStudentMasteryTool(BaseTool):
    """Tool to get detailed student mastery levels."""

    @property
    def name(self) -> str:
        return "get_student_mastery"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_student_mastery",
                "description": "Get detailed mastery levels for a specific student, broken down by subject and topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string",
                            "description": "The UUID of the student",
                        },
                        "subject_full_code": {
                            "type": "string",
                            "description": "Optional: Filter to a specific subject by full code (e.g., 'UK-NC-2014.MAT')",
                        },
                        "min_attempts": {
                            "type": "integer",
                            "description": "Minimum attempts to include a topic (default: 3)",
                        },
                    },
                    "required": ["student_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_student_mastery tool.

        Args:
            params: Tool parameters (student_id, subject_full_code, min_attempts).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with detailed mastery breakdown.
        """
        if not context.is_teacher:
            return ToolResult(
                success=False,
                error="This tool is only available for teachers.",
            )

        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available.",
            )

        student_id_str = params.get("student_id")
        if not student_id_str:
            return ToolResult(
                success=False,
                error="student_id is required.",
            )

        try:
            student_id = UUID(student_id_str)
        except ValueError:
            return ToolResult(
                success=False,
                error="Invalid student_id format.",
            )

        subject_full_code = params.get("subject_full_code")
        min_attempts = params.get("min_attempts", 3)
        teacher_id = context.user_id

        try:
            # Verify teacher has access to this student
            has_access = await verify_teacher_has_student_access(
                context.session, teacher_id, student_id
            )
            if not has_access:
                return ToolResult(
                    success=False,
                    error="You don't have access to this student.",
                )

            # Get student info
            student_query = (
                select(User.first_name, User.last_name)
                .where(User.id == str(student_id))
            )
            student_result = await context.session.execute(student_query)
            student_row = student_result.first()

            if not student_row:
                return ToolResult(
                    success=False,
                    error="Student not found.",
                )

            student_name = f"{student_row.first_name} {student_row.last_name}"

            # Parse subject_full_code if provided (format: "framework_code.subject_code")
            subject_framework_code = None
            subject_code = None
            if subject_full_code:
                parts = subject_full_code.split(".")
                if len(parts) == 2:
                    subject_framework_code, subject_code = parts
                else:
                    return ToolResult(
                        success=False,
                        error=f"Invalid subject_full_code format: {subject_full_code}. Expected format: 'framework_code.subject_code'",
                    )

            # Build the join using SQL concat to match entity_full_code with topic composite key
            topic_full_code_expr = func.concat(
                Topic.framework_code, ".",
                Topic.subject_code, ".",
                Topic.grade_code, ".",
                Topic.unit_code, ".",
                Topic.code,
            )

            # Build query for mastery by topic
            query = (
                select(
                    Subject.code.label("subject_code"),
                    Subject.framework_code.label("subject_framework_code"),
                    Subject.name.label("subject_name"),
                    Subject.icon.label("subject_icon"),
                    Topic.code.label("topic_code"),
                    Topic.name.label("topic_name"),
                    SemanticMemory.entity_full_code.label("topic_full_code"),
                    SemanticMemory.mastery_level,
                    SemanticMemory.attempts_total,
                    SemanticMemory.attempts_correct,
                )
                .join(
                    Topic,
                    SemanticMemory.entity_full_code == topic_full_code_expr,
                )
                .join(
                    Unit,
                    (Topic.framework_code == Unit.framework_code)
                    & (Topic.subject_code == Unit.subject_code)
                    & (Topic.grade_code == Unit.grade_code)
                    & (Topic.unit_code == Unit.code),
                )
                .join(
                    Subject,
                    (Topic.framework_code == Subject.framework_code)
                    & (Topic.subject_code == Subject.code),
                )
                .where(SemanticMemory.student_id == str(student_id))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= min_attempts)
            )

            if subject_framework_code and subject_code:
                query = query.where(
                    Subject.framework_code == subject_framework_code,
                    Subject.code == subject_code,
                )

            query = query.order_by(Subject.name, Topic.name)

            result = await context.session.execute(query)
            rows = result.all()

            # Group by subject (keyed by subject full_code)
            subjects_dict: dict[str, dict] = {}
            for row in rows:
                subject_full = f"{row.subject_framework_code}.{row.subject_code}"
                if subject_full not in subjects_dict:
                    subjects_dict[subject_full] = {
                        "subject_full_code": subject_full,
                        "subject_code": row.subject_code,
                        "subject_name": row.subject_name,
                        "subject_icon": row.subject_icon,
                        "topics": [],
                        "total_mastery": 0,
                        "topic_count": 0,
                    }

                mastery = float(row.mastery_level or 0)
                accuracy = (
                    (row.attempts_correct / row.attempts_total * 100)
                    if row.attempts_total > 0
                    else 0
                )

                subjects_dict[subject_full]["topics"].append({
                    "topic_full_code": row.topic_full_code,
                    "topic_code": row.topic_code,
                    "topic_name": row.topic_name,
                    "mastery": round(mastery * 100, 1),
                    "attempts": row.attempts_total,
                    "accuracy": round(accuracy, 1),
                    "status": self._get_mastery_status(mastery),
                })
                subjects_dict[subject_full]["total_mastery"] += mastery
                subjects_dict[subject_full]["topic_count"] += 1

            # Calculate averages and convert to list
            subjects = []
            for subject in subjects_dict.values():
                avg_mastery = (
                    subject["total_mastery"] / subject["topic_count"]
                    if subject["topic_count"] > 0
                    else 0
                )
                subjects.append({
                    "subject_full_code": subject["subject_full_code"],
                    "subject_code": subject["subject_code"],
                    "subject_name": subject["subject_name"],
                    "subject_icon": subject["subject_icon"],
                    "avg_mastery": round(avg_mastery * 100, 1),
                    "topic_count": subject["topic_count"],
                    "topics": subject["topics"],
                })

            # Build message
            if not subjects:
                message = f"{student_name} has no mastery data yet (minimum {min_attempts} attempts required)."
            else:
                total_topics = sum(s["topic_count"] for s in subjects)
                overall_mastery = sum(s["avg_mastery"] * s["topic_count"] for s in subjects) / total_topics
                message = f"{student_name} Mastery Overview:\n"
                message += f"- Overall: {round(overall_mastery)}% across {total_topics} topics\n"
                for s in subjects:
                    message += f"- {s['subject_name']}: {s['avg_mastery']}% ({s['topic_count']} topics)"

            logger.info(
                "get_student_mastery: teacher=%s, student=%s, subjects=%d",
                teacher_id,
                student_id,
                len(subjects),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "student_id": str(student_id),
                    "student_name": student_name,
                    "subjects": subjects,
                    "min_attempts": min_attempts,
                },
                passthrough_data={
                    "student_name": student_name,
                    "subjects": subjects,
                },
            )

        except Exception as e:
            logger.exception("get_student_mastery failed")
            return ToolResult(
                success=False,
                error=f"Failed to get mastery: {str(e)}",
            )

    def _get_mastery_status(self, mastery: float) -> str:
        """Get mastery status label."""
        if mastery >= 0.8:
            return "mastered"
        elif mastery >= 0.6:
            return "proficient"
        elif mastery >= 0.4:
            return "developing"
        else:
            return "struggling"
