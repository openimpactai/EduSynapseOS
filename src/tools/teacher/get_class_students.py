# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get class students tool for teachers.

Returns the list of students in a specific class that the teacher is assigned to.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.tools.teacher.helpers import verify_teacher_has_class_access

logger = logging.getLogger(__name__)


class GetClassStudentsTool(BaseTool):
    """Tool to get students in a class."""

    @property
    def name(self) -> str:
        return "get_class_students"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_class_students",
                "description": "Get the list of students in a specific class. Requires the class ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_id": {
                            "type": "string",
                            "description": "The UUID of the class to get students for",
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
        """Execute the get_class_students tool.

        Args:
            params: Tool parameters (class_id).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with list of students.
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

            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
            )
            from src.infrastructure.database.models.tenant.user import User

            # Get class info
            class_query = select(Class.name, Class.code).where(Class.id == str(class_id))
            class_result = await context.session.execute(class_query)
            class_row = class_result.first()
            class_name = class_row.name if class_row else "Unknown"

            # Get students in class
            query = (
                select(
                    User.id,
                    User.first_name,
                    User.last_name,
                    User.email,
                    ClassStudent.student_number,
                    ClassStudent.enrolled_at,
                )
                .join(ClassStudent, ClassStudent.student_id == User.id)
                .where(ClassStudent.class_id == str(class_id))
                .where(ClassStudent.status == "active")
                .where(User.status == "active")
                .order_by(User.last_name, User.first_name)
            )

            result = await context.session.execute(query)
            rows = result.all()

            students = []
            for row in rows:
                students.append({
                    "id": str(row.id),
                    "first_name": row.first_name,
                    "last_name": row.last_name,
                    "full_name": f"{row.first_name} {row.last_name}",
                    "email": row.email,
                    "student_number": row.student_number,
                    "enrolled_at": row.enrolled_at.isoformat() if row.enrolled_at else None,
                })

            # Build message
            if not students:
                message = f"Class {class_name} has no active students."
            elif len(students) == 1:
                message = f"Class {class_name} has 1 student: {students[0]['full_name']}."
            else:
                message = f"Class {class_name} has {len(students)} students."

            # Build UI element for student selection
            ui_element = None
            if students:
                options = [
                    UIElementOption(
                        id=s["id"],
                        label=s["full_name"],
                        description=s["email"] if s["email"] else None,
                    )
                    for s in students
                ]
                ui_element = UIElement(
                    type=UIElementType.SINGLE_SELECT,
                    id="student_selection",
                    title=f"Select a Student from {class_name}",
                    options=options,
                    allow_text_input=True,
                    searchable=len(students) > 10,
                )

            logger.info(
                "get_class_students: teacher=%s, class=%s, students=%d",
                teacher_id,
                class_id,
                len(students),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "class_id": str(class_id),
                    "class_name": class_name,
                    "students": students,
                    "count": len(students),
                },
                ui_element=ui_element,
                passthrough_data={"students": students, "class_name": class_name},
            )

        except Exception as e:
            logger.exception("get_class_students failed")
            return ToolResult(
                success=False,
                error=f"Failed to get students: {str(e)}",
            )
