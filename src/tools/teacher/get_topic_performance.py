# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get topic performance tool for teachers.

Returns performance metrics for specific topics across
a class or group of students.

Uses code-based composite keys from Central Curriculum structure.
SemanticMemory stores entity_full_code instead of UUID references.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import Float, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.tools.teacher.helpers import (
    get_teacher_class_ids,
    verify_teacher_has_class_access,
)

logger = logging.getLogger(__name__)


class GetTopicPerformanceTool(BaseTool):
    """Tool to get topic-level performance metrics."""

    @property
    def name(self) -> str:
        return "get_topic_performance"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_topic_performance",
                "description": "Get performance metrics for topics showing how students are doing across different curriculum topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_id": {
                            "type": "string",
                            "description": "Optional: Filter to a specific class",
                        },
                        "subject_full_code": {
                            "type": "string",
                            "description": "Optional: Filter to a specific subject by full code (e.g., 'UK-NC-2014.MAT')",
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["mastery_asc", "mastery_desc", "attempts_desc", "name"],
                            "description": "How to sort results (default: mastery_asc to show weakest first)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of topics to return (default: 20)",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_topic_performance tool.

        Args:
            params: Tool parameters (class_id, subject_full_code, sort_by, limit).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with topic performance metrics.
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

        class_id_str = params.get("class_id")
        subject_full_code = params.get("subject_full_code")
        sort_by = params.get("sort_by", "mastery_asc")
        limit = params.get("limit", 20)
        teacher_id = context.user_id

        try:
            # Determine which classes to include
            if class_id_str:
                try:
                    class_id = UUID(class_id_str)
                except ValueError:
                    return ToolResult(
                        success=False,
                        error="Invalid class_id format.",
                    )

                has_access = await verify_teacher_has_class_access(
                    context.session, teacher_id, class_id
                )
                if not has_access:
                    return ToolResult(
                        success=False,
                        error="You don't have access to this class.",
                    )
                class_ids = [str(class_id)]
            else:
                class_ids = await get_teacher_class_ids(context.session, teacher_id)

            if not class_ids:
                return ToolResult(
                    success=True,
                    data={
                        "message": "You don't have any classes assigned.",
                        "topics": [],
                        "count": 0,
                    },
                )

            from src.infrastructure.database.models.tenant.curriculum import (
                Subject,
                Topic,
                Unit,
            )
            from src.infrastructure.database.models.tenant.memory import SemanticMemory
            from src.infrastructure.database.models.tenant.school import ClassStudent

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

            # Get student IDs in teacher's classes
            student_ids_query = (
                select(ClassStudent.student_id)
                .where(ClassStudent.class_id.in_(class_ids))
                .where(ClassStudent.status == "active")
                .distinct()
            )
            student_ids_result = await context.session.execute(student_ids_query)
            student_ids = [str(row[0]) for row in student_ids_result.all()]

            if not student_ids:
                return ToolResult(
                    success=True,
                    data={
                        "message": "No active students found in your classes.",
                        "topics": [],
                        "count": 0,
                    },
                )

            # Build the join using SQL concat to match entity_full_code with topic composite key
            topic_full_code_expr = func.concat(
                Topic.framework_code, ".",
                Topic.subject_code, ".",
                Topic.grade_code, ".",
                Topic.unit_code, ".",
                Topic.code,
            )

            # Build query for topic performance (cast to Float to avoid PostgreSQL avg() type ambiguity)
            query = (
                select(
                    Topic.code.label("topic_code"),
                    Topic.name.label("topic_name"),
                    topic_full_code_expr.label("topic_full_code"),
                    Unit.name.label("unit_name"),
                    Subject.code.label("subject_code"),
                    Subject.framework_code.label("subject_framework_code"),
                    Subject.name.label("subject_name"),
                    Subject.icon.label("subject_icon"),
                    func.count(func.distinct(SemanticMemory.student_id)).label("students_practiced"),
                    func.avg(SemanticMemory.mastery_level.cast(Float)).label("avg_mastery"),
                    func.sum(SemanticMemory.attempts_total).label("total_attempts"),
                    func.sum(SemanticMemory.attempts_correct).label("total_correct"),
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
                .where(SemanticMemory.student_id.in_(student_ids))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= 1)
                .group_by(
                    Topic.code,
                    Topic.name,
                    Topic.framework_code,
                    Topic.subject_code,
                    Topic.grade_code,
                    Topic.unit_code,
                    Unit.name,
                    Subject.code,
                    Subject.framework_code,
                    Subject.name,
                    Subject.icon,
                )
            )

            if subject_framework_code and subject_code:
                query = query.where(
                    Subject.framework_code == subject_framework_code,
                    Subject.code == subject_code,
                )

            # Apply sorting
            if sort_by == "mastery_asc":
                query = query.order_by(func.avg(SemanticMemory.mastery_level.cast(Float)).asc())
            elif sort_by == "mastery_desc":
                query = query.order_by(func.avg(SemanticMemory.mastery_level.cast(Float)).desc())
            elif sort_by == "attempts_desc":
                query = query.order_by(func.sum(SemanticMemory.attempts_total).desc())
            else:  # name
                query = query.order_by(Subject.name, Topic.name)

            query = query.limit(limit)

            result = await context.session.execute(query)
            rows = result.all()

            topics = []
            for row in rows:
                total_attempts = row.total_attempts or 0
                total_correct = row.total_correct or 0
                accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
                subj_full = f"{row.subject_framework_code}.{row.subject_code}"

                topics.append({
                    "topic_code": row.topic_code,
                    "topic_full_code": row.topic_full_code,
                    "topic_name": row.topic_name,
                    "unit_name": row.unit_name,
                    "subject_code": row.subject_code,
                    "subject_full_code": subj_full,
                    "subject_name": row.subject_name,
                    "subject_icon": row.subject_icon,
                    "students_practiced": row.students_practiced,
                    "avg_mastery": round(float(row.avg_mastery or 0) * 100, 1),
                    "total_attempts": total_attempts,
                    "accuracy": round(accuracy, 1),
                    "status": self._get_topic_status(float(row.avg_mastery or 0)),
                })

            # Build message
            if not topics:
                message = "No topic performance data found. Students haven't practiced any topics yet."
            else:
                # Count by status
                struggling = sum(1 for t in topics if t["status"] == "struggling")
                developing = sum(1 for t in topics if t["status"] == "developing")
                proficient = sum(1 for t in topics if t["status"] == "proficient")
                mastered = sum(1 for t in topics if t["status"] == "mastered")

                message = f"Topic Performance Summary ({len(topics)} topics):\n"
                if struggling > 0:
                    message += f"- Struggling: {struggling} topics\n"
                if developing > 0:
                    message += f"- Developing: {developing} topics\n"
                if proficient > 0:
                    message += f"- Proficient: {proficient} topics\n"
                if mastered > 0:
                    message += f"- Mastered: {mastered} topics\n"

                # Show weakest topics
                weakest = [t for t in topics if t["status"] == "struggling"][:3]
                if weakest:
                    message += "\nTopics needing attention:\n"
                    for t in weakest:
                        message += f"- {t['topic_name']}: {t['avg_mastery']}% mastery\n"

            logger.info(
                "get_topic_performance: teacher=%s, topics=%d",
                teacher_id,
                len(topics),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "topics": topics,
                    "count": len(topics),
                    "total_students": len(student_ids),
                    "filters": {
                        "class_id": class_id_str,
                        "subject_full_code": subject_full_code,
                        "sort_by": sort_by,
                    },
                },
                passthrough_data={
                    "topics": topics[:5],
                    "count": len(topics),
                },
            )

        except Exception as e:
            logger.exception("get_topic_performance failed")
            return ToolResult(
                success=False,
                error=f"Failed to get topic performance: {str(e)}",
            )

    def _get_topic_status(self, mastery: float) -> str:
        """Get status label for average mastery level."""
        if mastery >= 0.8:
            return "mastered"
        elif mastery >= 0.6:
            return "proficient"
        elif mastery >= 0.4:
            return "developing"
        else:
            return "struggling"
