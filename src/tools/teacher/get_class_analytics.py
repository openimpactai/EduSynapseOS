# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get class analytics tool for teachers.

Returns aggregate analytics for a specific class including
overall performance, engagement metrics, and trends.

Uses code-based composite keys from Central Curriculum structure.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Float, case, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.tools.teacher.helpers import verify_teacher_has_class_access

logger = logging.getLogger(__name__)


class GetClassAnalyticsTool(BaseTool):
    """Tool to get class-level analytics."""

    @property
    def name(self) -> str:
        return "get_class_analytics"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_class_analytics",
                "description": "Get aggregate analytics for a class including performance metrics, engagement levels, and trends.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_id": {
                            "type": "string",
                            "description": "The UUID of the class",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to include in analytics (default: 30)",
                        },
                        "subject_full_code": {
                            "type": "string",
                            "description": "Optional: Filter analytics to a specific subject by full code (e.g., 'UK-NC-2014.MAT')",
                        },
                    },
                    "required": ["class_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_class_analytics tool.

        Args:
            params: Tool parameters (class_id, days, subject_full_code).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with class analytics data.
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
        if not class_id_str:
            return ToolResult(
                success=False,
                error="class_id is required.",
            )

        try:
            class_id = UUID(class_id_str)
        except ValueError:
            return ToolResult(
                success=False,
                error="Invalid class_id format.",
            )

        days = params.get("days", 30)
        subject_full_code = params.get("subject_full_code")
        teacher_id = context.user_id

        try:
            # Verify teacher has access to this class
            has_access = await verify_teacher_has_class_access(
                context.session, teacher_id, class_id
            )
            if not has_access:
                return ToolResult(
                    success=False,
                    error="You don't have access to this class.",
                )

            from src.infrastructure.database.models.tenant.curriculum import Subject
            from src.infrastructure.database.models.tenant.memory import SemanticMemory
            from src.infrastructure.database.models.tenant.practice import PracticeSession
            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
            )

            # Get class info
            class_query = select(Class.name, Class.code).where(Class.id == str(class_id))
            class_result = await context.session.execute(class_query)
            class_row = class_result.first()
            class_name = class_row.name if class_row else "Unknown"

            # Get student IDs in this class
            student_ids_query = (
                select(ClassStudent.student_id)
                .where(ClassStudent.class_id == str(class_id))
                .where(ClassStudent.status == "active")
            )
            student_ids_result = await context.session.execute(student_ids_query)
            student_ids = [str(row[0]) for row in student_ids_result.all()]

            if not student_ids:
                return ToolResult(
                    success=True,
                    data={
                        "message": f"Class {class_name} has no active students.",
                        "class_id": str(class_id),
                        "class_name": class_name,
                        "student_count": 0,
                    },
                )

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

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Practice session analytics
            # Note: PracticeSession.accuracy is a Python @property, calculate from columns
            practice_query = (
                select(
                    func.count(PracticeSession.id).label("total_sessions"),
                    func.count(func.distinct(PracticeSession.student_id)).label("active_students"),
                    func.sum(PracticeSession.questions_total).label("total_questions"),
                    func.sum(PracticeSession.questions_correct).label("total_correct"),
                    func.avg(
                        case(
                            (
                                PracticeSession.questions_answered > 0,
                                (PracticeSession.questions_correct * 100.0) / PracticeSession.questions_answered,
                            ),
                            else_=None,
                        )
                    ).label("avg_accuracy"),
                )
                .where(PracticeSession.student_id.in_(student_ids))
                .where(PracticeSession.started_at >= start_date)
            )

            if subject_framework_code and subject_code:
                practice_query = practice_query.where(
                    PracticeSession.topic_framework_code == subject_framework_code,
                    PracticeSession.topic_subject_code == subject_code,
                )

            practice_result = await context.session.execute(practice_query)
            practice_row = practice_result.first()

            total_sessions = practice_row.total_sessions or 0
            active_students = practice_row.active_students or 0
            total_questions = practice_row.total_questions or 0
            total_correct = practice_row.total_correct or 0
            avg_accuracy = float(practice_row.avg_accuracy or 0)

            # Mastery analytics (cast to Float to avoid PostgreSQL avg() type ambiguity)
            mastery_query = (
                select(
                    func.count(SemanticMemory.id).label("total_topic_records"),
                    func.avg(SemanticMemory.mastery_level.cast(Float)).label("avg_mastery"),
                    func.count(SemanticMemory.id).filter(
                        SemanticMemory.mastery_level >= 0.8
                    ).label("mastered_count"),
                    func.count(SemanticMemory.id).filter(
                        SemanticMemory.mastery_level < 0.4
                    ).label("struggling_count"),
                )
                .where(SemanticMemory.student_id.in_(student_ids))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= 3)
            )

            mastery_result = await context.session.execute(mastery_query)
            mastery_row = mastery_result.first()

            avg_mastery = float(mastery_row.avg_mastery or 0)
            mastered_count = mastery_row.mastered_count or 0
            struggling_count = mastery_row.struggling_count or 0

            # Engagement rate
            engagement_rate = (
                (active_students / len(student_ids) * 100)
                if student_ids
                else 0
            )

            # Subject breakdown if not filtered
            subject_breakdown = []
            if not subject_full_code:
                # Join PracticeSession with Subject via composite keys
                subject_query = (
                    select(
                        Subject.code,
                        Subject.framework_code,
                        Subject.name,
                        Subject.icon,
                        func.count(PracticeSession.id).label("session_count"),
                        func.avg(
                            case(
                                (
                                    PracticeSession.questions_answered > 0,
                                    (PracticeSession.questions_correct * 100.0) / PracticeSession.questions_answered,
                                ),
                                else_=None,
                            )
                        ).label("avg_accuracy"),
                    )
                    .join(
                        Subject,
                        (PracticeSession.topic_framework_code == Subject.framework_code)
                        & (PracticeSession.topic_subject_code == Subject.code),
                    )
                    .where(PracticeSession.student_id.in_(student_ids))
                    .where(PracticeSession.started_at >= start_date)
                    .group_by(Subject.code, Subject.framework_code, Subject.name, Subject.icon)
                    .order_by(func.count(PracticeSession.id).desc())
                )

                subject_result = await context.session.execute(subject_query)
                for row in subject_result.all():
                    subj_full = f"{row.framework_code}.{row.code}"
                    subject_breakdown.append({
                        "subject_code": row.code,
                        "subject_full_code": subj_full,
                        "subject_name": row.name,
                        "subject_icon": row.icon,
                        "session_count": row.session_count,
                        "avg_accuracy": round(float(row.avg_accuracy or 0), 1),
                    })

            # Build message
            message_parts = [f"{class_name} - {days} Day Analytics:"]
            message_parts.append(f"- Students: {len(student_ids)} ({active_students} active)")
            message_parts.append(f"- Engagement: {round(engagement_rate)}%")
            message_parts.append(f"- Practice Sessions: {total_sessions}")
            message_parts.append(f"- Questions Answered: {total_questions}")
            message_parts.append(f"- Class Accuracy: {round(avg_accuracy)}%")
            message_parts.append(f"- Class Mastery: {round(avg_mastery * 100)}%")

            if struggling_count > 0:
                message_parts.append(f"- Topics Needing Attention: {struggling_count}")

            message = "\n".join(message_parts)

            logger.info(
                "get_class_analytics: teacher=%s, class=%s, sessions=%d",
                teacher_id,
                class_id,
                total_sessions,
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "class_id": str(class_id),
                    "class_name": class_name,
                    "period_days": days,
                    "student_count": len(student_ids),
                    "active_students": active_students,
                    "engagement_rate": round(engagement_rate, 1),
                    "practice": {
                        "total_sessions": total_sessions,
                        "total_questions": total_questions,
                        "total_correct": total_correct,
                        "avg_accuracy": round(avg_accuracy, 1),
                    },
                    "mastery": {
                        "avg_mastery": round(avg_mastery * 100, 1),
                        "mastered_topics": mastered_count,
                        "struggling_topics": struggling_count,
                    },
                    "subject_breakdown": subject_breakdown,
                },
                passthrough_data={
                    "class_name": class_name,
                    "engagement_rate": round(engagement_rate, 1),
                    "avg_accuracy": round(avg_accuracy, 1),
                    "avg_mastery": round(avg_mastery * 100, 1),
                },
            )

        except Exception as e:
            logger.exception("get_class_analytics failed")
            return ToolResult(
                success=False,
                error=f"Failed to get analytics: {str(e)}",
            )
