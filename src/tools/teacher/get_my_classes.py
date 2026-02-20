# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get my classes tool for teachers.

Returns the list of classes that the teacher is assigned to,
along with student counts and subject information.
"""

import logging
from typing import Any

from sqlalchemy import and_, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType

logger = logging.getLogger(__name__)


class GetMyClassesTool(BaseTool):
    """Tool to get teacher's assigned classes."""

    @property
    def name(self) -> str:
        return "get_my_classes"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_my_classes",
                "description": "Get the list of classes that the teacher is assigned to, including student counts and subject information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_student_count": {
                            "type": "boolean",
                            "description": "Include the number of students in each class (default: true)",
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
        """Execute the get_my_classes tool.

        Args:
            params: Tool parameters (include_student_count).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with list of classes.
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

        include_student_count = params.get("include_student_count", True)
        teacher_id = context.user_id

        try:
            from src.infrastructure.database.models.tenant.curriculum import Subject
            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
                ClassTeacher,
            )

            # Build query for teacher's classes
            if include_student_count:
                query = (
                    select(
                        Class.id,
                        Class.name,
                        Class.code,
                        ClassTeacher.is_homeroom,
                        Subject.code.label("subject_code"),
                        Subject.name.label("subject_name"),
                        Subject.icon.label("subject_icon"),
                        func.count(ClassStudent.id).filter(
                            ClassStudent.status == "active"
                        ).label("student_count"),
                    )
                    .join(ClassTeacher, ClassTeacher.class_id == Class.id)
                    .outerjoin(
                        Subject,
                        and_(
                            ClassTeacher.subject_framework_code == Subject.framework_code,
                            ClassTeacher.subject_code == Subject.code,
                        ),
                    )
                    .outerjoin(ClassStudent, ClassStudent.class_id == Class.id)
                    .where(ClassTeacher.teacher_id == str(teacher_id))
                    .where(ClassTeacher.ended_at.is_(None))
                    .where(Class.is_active == True)
                    .group_by(
                        Class.id,
                        Class.name,
                        Class.code,
                        ClassTeacher.is_homeroom,
                        Subject.code,
                        Subject.name,
                        Subject.icon,
                    )
                    .order_by(Class.name)
                )
            else:
                query = (
                    select(
                        Class.id,
                        Class.name,
                        Class.code,
                        ClassTeacher.is_homeroom,
                        Subject.code.label("subject_code"),
                        Subject.name.label("subject_name"),
                        Subject.icon.label("subject_icon"),
                    )
                    .join(ClassTeacher, ClassTeacher.class_id == Class.id)
                    .outerjoin(
                        Subject,
                        and_(
                            ClassTeacher.subject_framework_code == Subject.framework_code,
                            ClassTeacher.subject_code == Subject.code,
                        ),
                    )
                    .where(ClassTeacher.teacher_id == str(teacher_id))
                    .where(ClassTeacher.ended_at.is_(None))
                    .where(Class.is_active == True)
                    .order_by(Class.name)
                )

            result = await context.session.execute(query)
            rows = result.all()

            classes = []
            for row in rows:
                class_info = {
                    "id": str(row.id),
                    "name": row.name,
                    "code": row.code,
                    "is_homeroom": row.is_homeroom,
                    "subject_code": row.subject_code if row.subject_code else None,
                    "subject_name": row.subject_name,
                    "subject_icon": row.subject_icon,
                }
                if include_student_count:
                    class_info["student_count"] = row.student_count or 0
                classes.append(class_info)

            # Build message
            if not classes:
                message = "You don't have any classes assigned."
            elif len(classes) == 1:
                c = classes[0]
                subject_info = f" ({c['subject_name']})" if c["subject_name"] else ""
                student_info = f" with {c['student_count']} students" if include_student_count else ""
                message = f"You have 1 class: {c['name']}{subject_info}{student_info}."
            else:
                total_students = sum(c.get("student_count", 0) for c in classes) if include_student_count else 0
                student_info = f" with {total_students} total students" if include_student_count else ""
                message = f"You have {len(classes)} classes{student_info}."

            # Build UI element for class selection
            ui_element = None
            if classes:
                options = [
                    UIElementOption(
                        id=c["id"],
                        label=c["name"],
                        description=f"{c['subject_name'] or 'General'} - {c.get('student_count', '?')} students",
                        icon=c["subject_icon"] or "📚",
                    )
                    for c in classes
                ]
                ui_element = UIElement(
                    type=UIElementType.SINGLE_SELECT,
                    id="class_selection",
                    title="Select a Class",
                    options=options,
                    allow_text_input=True,
                )

            logger.info(
                "get_my_classes: teacher=%s, classes=%d",
                teacher_id,
                len(classes),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "classes": classes,
                    "count": len(classes),
                },
                ui_element=ui_element,
                passthrough_data={"classes": classes},
            )

        except Exception as e:
            logger.exception("get_my_classes failed")
            return ToolResult(
                success=False,
                error=f"Failed to get classes: {str(e)}",
            )
