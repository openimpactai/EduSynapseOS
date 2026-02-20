# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get struggling students tool for teachers.

Returns a list of students who are struggling based on
low mastery levels, declining performance, or lack of engagement.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Float, case, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.tools.teacher.helpers import (
    get_teacher_class_ids,
    verify_teacher_has_class_access,
)

logger = logging.getLogger(__name__)


class GetStrugglingStudentsTool(BaseTool):
    """Tool to identify struggling students."""

    @property
    def name(self) -> str:
        return "get_struggling_students"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_struggling_students",
                "description": "Get a list of students who are struggling based on low mastery, declining performance, or lack of engagement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_id": {
                            "type": "string",
                            "description": "Optional: Filter to a specific class. If not provided, checks all teacher's classes.",
                        },
                        "mastery_threshold": {
                            "type": "number",
                            "description": "Mastery level threshold (0-100). Students below this are considered struggling (default: 40)",
                        },
                        "include_inactive": {
                            "type": "boolean",
                            "description": "Include students with no recent activity (default: true)",
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
        """Execute the get_struggling_students tool.

        Args:
            params: Tool parameters (class_id, mastery_threshold, include_inactive).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with list of struggling students.
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
        mastery_threshold = params.get("mastery_threshold", 40) / 100  # Convert to 0-1
        include_inactive = params.get("include_inactive", True)
        teacher_id = context.user_id

        try:
            # If class_id provided, verify access
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
                # Get all teacher's classes
                class_ids = await get_teacher_class_ids(context.session, teacher_id)

            if not class_ids:
                return ToolResult(
                    success=True,
                    data={
                        "message": "You don't have any classes assigned.",
                        "students": [],
                        "count": 0,
                    },
                )

            from src.infrastructure.database.models.tenant.memory import SemanticMemory
            from src.infrastructure.database.models.tenant.practice import PracticeSession
            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
            )
            from src.infrastructure.database.models.tenant.user import User

            # Get students in teacher's classes
            students_query = (
                select(
                    User.id,
                    User.first_name,
                    User.last_name,
                    Class.id.label("class_id"),
                    Class.name.label("class_name"),
                )
                .join(ClassStudent, ClassStudent.student_id == User.id)
                .join(Class, Class.id == ClassStudent.class_id)
                .where(ClassStudent.class_id.in_(class_ids))
                .where(ClassStudent.status == "active")
                .where(User.status == "active")
            )

            students_result = await context.session.execute(students_query)
            all_students = students_result.all()

            if not all_students:
                return ToolResult(
                    success=True,
                    data={
                        "message": "No active students found in your classes.",
                        "students": [],
                        "count": 0,
                    },
                )

            # Calculate date range for activity check
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)

            struggling_students = []

            for student in all_students:
                student_id = str(student.id)
                reasons = []

                # Check mastery level (cast to Float to avoid PostgreSQL avg() type ambiguity)
                mastery_query = (
                    select(func.avg(SemanticMemory.mastery_level.cast(Float)))
                    .where(SemanticMemory.student_id == student_id)
                    .where(SemanticMemory.entity_type == "topic")
                    .where(SemanticMemory.attempts_total >= 3)
                )
                mastery_result = await context.session.execute(mastery_query)
                avg_mastery = mastery_result.scalar() or 0

                if avg_mastery < mastery_threshold:
                    reasons.append(f"Low mastery ({round(avg_mastery * 100)}%)")

                # Check recent accuracy (calculate from questions_correct/questions_answered)
                # Note: PracticeSession.accuracy is a Python @property, not a DB column
                accuracy_query = (
                    select(
                        func.avg(
                            case(
                                (
                                    PracticeSession.questions_answered > 0,
                                    (PracticeSession.questions_correct * 100.0) / PracticeSession.questions_answered,
                                ),
                                else_=None,
                            )
                        )
                    )
                    .where(PracticeSession.student_id == student_id)
                    .where(PracticeSession.started_at >= start_date)
                    .where(PracticeSession.questions_answered > 0)
                )
                accuracy_result = await context.session.execute(accuracy_query)
                recent_accuracy = accuracy_result.scalar()

                if recent_accuracy is not None and recent_accuracy < 50:
                    reasons.append(f"Low recent accuracy ({round(recent_accuracy)}%)")

                # Check activity level
                activity_query = (
                    select(func.count(PracticeSession.id))
                    .where(PracticeSession.student_id == student_id)
                    .where(PracticeSession.started_at >= start_date)
                )
                activity_result = await context.session.execute(activity_query)
                session_count = activity_result.scalar() or 0

                if include_inactive and session_count == 0:
                    reasons.append("No recent activity")

                # Add to struggling list if any reasons found
                if reasons:
                    struggling_students.append({
                        "student_id": student_id,
                        "full_name": f"{student.first_name} {student.last_name}",
                        "first_name": student.first_name,
                        "last_name": student.last_name,
                        "class_id": str(student.class_id),
                        "class_name": student.class_name,
                        "avg_mastery": round(float(avg_mastery) * 100, 1),
                        "recent_accuracy": round(float(recent_accuracy), 1) if recent_accuracy else None,
                        "recent_sessions": session_count,
                        "reasons": reasons,
                        "priority": self._calculate_priority(avg_mastery, recent_accuracy, session_count),
                    })

            # Sort by priority (highest first)
            struggling_students.sort(key=lambda x: x["priority"], reverse=True)

            # Build message
            if not struggling_students:
                message = "No struggling students identified. All students are performing well!"
            elif len(struggling_students) == 1:
                s = struggling_students[0]
                message = f"1 student needs attention: {s['full_name']} ({', '.join(s['reasons'])})"
            else:
                message = f"{len(struggling_students)} students need attention:\n"
                for s in struggling_students[:5]:
                    message += f"- {s['full_name']}: {', '.join(s['reasons'])}\n"
                if len(struggling_students) > 5:
                    message += f"... and {len(struggling_students) - 5} more"

            # Build UI element for student selection
            ui_element = None
            if struggling_students:
                options = [
                    UIElementOption(
                        id=s["student_id"],
                        label=s["full_name"],
                        description=f"{s['class_name']} - {', '.join(s['reasons'])}",
                    )
                    for s in struggling_students[:10]
                ]
                ui_element = UIElement(
                    type=UIElementType.SINGLE_SELECT,
                    id="struggling_student_selection",
                    title="Select a Student to View Details",
                    options=options,
                    allow_text_input=True,
                )

            logger.info(
                "get_struggling_students: teacher=%s, struggling=%d",
                teacher_id,
                len(struggling_students),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "students": struggling_students,
                    "count": len(struggling_students),
                    "mastery_threshold": round(mastery_threshold * 100),
                },
                ui_element=ui_element,
                passthrough_data={
                    "students": struggling_students[:5],
                    "count": len(struggling_students),
                },
            )

        except Exception as e:
            logger.exception("get_struggling_students failed")
            return ToolResult(
                success=False,
                error=f"Failed to get struggling students: {str(e)}",
            )

    def _calculate_priority(
        self,
        avg_mastery: float,
        recent_accuracy: float | None,
        session_count: int,
    ) -> int:
        """Calculate priority score for a struggling student.

        Higher score = higher priority for intervention.
        """
        priority = 0

        # Low mastery is high priority
        if avg_mastery < 0.2:
            priority += 30
        elif avg_mastery < 0.3:
            priority += 20
        elif avg_mastery < 0.4:
            priority += 10

        # Low recent accuracy adds to priority
        if recent_accuracy is not None:
            if recent_accuracy < 30:
                priority += 25
            elif recent_accuracy < 40:
                priority += 15
            elif recent_accuracy < 50:
                priority += 5

        # No activity is concerning
        if session_count == 0:
            priority += 20

        return priority
