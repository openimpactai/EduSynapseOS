# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get student progress tool for teachers.

Returns a progress summary for a specific student, including
practice sessions, mastery levels, and recent activity.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Float, case, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.tools.teacher.helpers import verify_teacher_has_student_access

logger = logging.getLogger(__name__)


class GetStudentProgressTool(BaseTool):
    """Tool to get student progress summary."""

    @property
    def name(self) -> str:
        return "get_student_progress"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_student_progress",
                "description": "Get a progress summary for a specific student, including practice sessions, mastery levels, and recent activity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string",
                            "description": "The UUID of the student",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to include in the summary (default: 30)",
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
        """Execute the get_student_progress tool.

        Args:
            params: Tool parameters (student_id, days).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with student progress summary.
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

        days = params.get("days", 30)
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

            from src.infrastructure.database.models.tenant.memory import SemanticMemory
            from src.infrastructure.database.models.tenant.practice import PracticeSession
            from src.infrastructure.database.models.tenant.user import User

            # Get student info
            student_query = (
                select(User.first_name, User.last_name, User.email)
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

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get practice session stats
            # Note: PracticeSession.accuracy is a Python @property, calculate from columns
            practice_query = (
                select(
                    func.count(PracticeSession.id).label("session_count"),
                    func.sum(PracticeSession.questions_total).label("total_questions"),
                    func.sum(PracticeSession.questions_correct).label("correct_answers"),
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
                .where(PracticeSession.student_id == str(student_id))
                .where(PracticeSession.started_at >= start_date)
            )

            practice_result = await context.session.execute(practice_query)
            practice_row = practice_result.first()

            session_count = practice_row.session_count or 0
            total_questions = practice_row.total_questions or 0
            correct_answers = practice_row.correct_answers or 0
            avg_accuracy = float(practice_row.avg_accuracy or 0)

            # Get mastery stats (cast to Float to avoid PostgreSQL avg() type ambiguity)
            mastery_query = (
                select(
                    func.count(SemanticMemory.id).label("topics_practiced"),
                    func.avg(SemanticMemory.mastery_level.cast(Float)).label("avg_mastery"),
                    func.count(SemanticMemory.id).filter(
                        SemanticMemory.mastery_level >= 0.8
                    ).label("topics_mastered"),
                    func.count(SemanticMemory.id).filter(
                        SemanticMemory.mastery_level < 0.4
                    ).label("topics_struggling"),
                )
                .where(SemanticMemory.student_id == str(student_id))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= 3)
            )

            mastery_result = await context.session.execute(mastery_query)
            mastery_row = mastery_result.first()

            topics_practiced = mastery_row.topics_practiced or 0
            avg_mastery = float(mastery_row.avg_mastery or 0)
            topics_mastered = mastery_row.topics_mastered or 0
            topics_struggling = mastery_row.topics_struggling or 0

            # Get strongest and weakest topics
            from src.infrastructure.database.models.tenant.curriculum import Topic
            from sqlalchemy import func

            # Build Topic full code expression for join
            topic_full_code_expr = func.concat(
                Topic.framework_code, ".",
                Topic.subject_code, ".",
                Topic.grade_code, ".",
                Topic.unit_code, ".",
                Topic.code,
            )

            strongest_query = (
                select(Topic.name, SemanticMemory.mastery_level)
                .join(Topic, SemanticMemory.entity_full_code == topic_full_code_expr)
                .where(SemanticMemory.student_id == str(student_id))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= 3)
                .order_by(SemanticMemory.mastery_level.desc())
                .limit(3)
            )

            strongest_result = await context.session.execute(strongest_query)
            strongest = [
                {"topic": row.name, "mastery": round(float(row.mastery_level) * 100)}
                for row in strongest_result.all()
            ]

            weakest_query = (
                select(Topic.name, SemanticMemory.mastery_level)
                .join(Topic, SemanticMemory.entity_full_code == topic_full_code_expr)
                .where(SemanticMemory.student_id == str(student_id))
                .where(SemanticMemory.entity_type == "topic")
                .where(SemanticMemory.attempts_total >= 3)
                .order_by(SemanticMemory.mastery_level.asc())
                .limit(3)
            )

            weakest_result = await context.session.execute(weakest_query)
            weakest = [
                {"topic": row.name, "mastery": round(float(row.mastery_level) * 100)}
                for row in weakest_result.all()
            ]

            # Build message
            message_parts = [f"{student_name} - {days} Day Summary:"]
            message_parts.append(f"- Practice Sessions: {session_count}")
            message_parts.append(f"- Questions Answered: {total_questions} ({correct_answers} correct)")
            message_parts.append(f"- Average Accuracy: {round(avg_accuracy)}%")
            message_parts.append(f"- Overall Mastery: {round(avg_mastery * 100)}%")

            if strongest:
                message_parts.append(f"- Strongest: {strongest[0]['topic']} ({strongest[0]['mastery']}%)")
            if weakest and weakest[0]["mastery"] < 50:
                message_parts.append(f"- Needs Work: {weakest[0]['topic']} ({weakest[0]['mastery']}%)")

            message = "\n".join(message_parts)

            logger.info(
                "get_student_progress: teacher=%s, student=%s, sessions=%d",
                teacher_id,
                student_id,
                session_count,
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "student_id": str(student_id),
                    "student_name": student_name,
                    "period_days": days,
                    "practice": {
                        "session_count": session_count,
                        "total_questions": total_questions,
                        "correct_answers": correct_answers,
                        "avg_accuracy": round(avg_accuracy, 1),
                    },
                    "mastery": {
                        "topics_practiced": topics_practiced,
                        "avg_mastery": round(avg_mastery * 100, 1),
                        "topics_mastered": topics_mastered,
                        "topics_struggling": topics_struggling,
                    },
                    "strongest_topics": strongest,
                    "weakest_topics": weakest,
                },
                passthrough_data={
                    "student_name": student_name,
                    "practice": {
                        "session_count": session_count,
                        "avg_accuracy": round(avg_accuracy, 1),
                    },
                    "mastery": {
                        "avg_mastery": round(avg_mastery * 100, 1),
                    },
                },
            )

        except Exception as e:
            logger.exception("get_student_progress failed")
            return ToolResult(
                success=False,
                error=f"Failed to get progress: {str(e)}",
            )
