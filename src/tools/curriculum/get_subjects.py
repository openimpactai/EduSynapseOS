# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get subjects tool.

This tool retrieves available subjects for a student based on their grade level.
Used by agents to show subject options before practice or learning activities.

Returns a UIElement for frontend to render a proper selection control,
along with passthrough data containing full subject details.
"""

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.infrastructure.database.models.tenant.curriculum import (
    GradeLevel,
    Subject,
    Unit,
)


class GetSubjectsTool(BaseTool):
    """Tool to get available subjects for the student's grade level.

    Queries subjects that have units defined for the student's grade level.
    This ensures only relevant subjects are shown to the student.

    Used in clarification flow when student wants to:
    - Start practice session
    - Learn a topic
    - Review material

    The tool should be called BEFORE navigate to determine which
    subjects are available for the student.
    """

    @property
    def name(self) -> str:
        return "get_subjects"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_subjects",
                "description": (
                    "Get available subjects for the student's grade level. "
                    "CRITICAL: You MUST call this tool BEFORE suggesting practice or learning activities. "
                    "This tells you which subjects the student can study.\n\n"
                    "Use cases:\n"
                    "- Student says 'I want to practice' → Call get_subjects first\n"
                    "- Student says 'I want to learn something' → Call get_subjects first\n"
                    "- You want to suggest activities → Call get_subjects to know options\n\n"
                    "Returns a list of subjects with IDs and names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_progress": {
                            "type": "boolean",
                            "description": "Include student's progress for each subject. Optional, defaults to false. (not yet implemented)",
                        },
                    },
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_subjects tool.

        Queries subjects that have units for the student's grade level and framework.
        The query path is: subjects → units (with grade_level_id) → grade_levels

        Args:
            params: Tool parameters from LLM.
                - include_progress: Whether to include progress (not yet implemented)
            context: Execution context with grade_level and framework_code.

        Returns:
            ToolResult with list of subjects.
        """
        grade_level = context.grade_level
        framework_code = context.framework_code

        try:
            # Query subjects that have units for this grade level and framework
            # Join path: Subject → Unit (via framework_code + subject_code) → GradeLevel (via framework_code + grade_code)
            # Uses composite key relationships from Central Curriculum structure
            #
            # IMPORTANT: Filter by framework_code to prevent cross-curriculum pollution
            # Without this filter, students from different countries with same grade sequence
            # would see subjects from all frameworks (e.g., UK Year 2 and Malawi Standard 2)
            stmt = (
                select(Subject)
                .join(
                    Unit,
                    (Subject.framework_code == Unit.framework_code)
                    & (Subject.code == Unit.subject_code),
                )
                .join(
                    GradeLevel,
                    (Unit.framework_code == GradeLevel.framework_code)
                    & (Unit.grade_code == GradeLevel.code),
                )
                .where(GradeLevel.sequence == grade_level)
            )

            # Apply framework filter if available (critical for multi-curriculum support)
            if framework_code:
                stmt = stmt.where(Subject.framework_code == framework_code)

            stmt = stmt.distinct().order_by(Subject.sequence)

            result = await context.session.execute(stmt)
            subjects = result.scalars().all()

            # Format subjects for LLM (full_code is primary identifier)
            subjects_data = [
                {
                    "full_code": subject.full_code,  # e.g., "UK-NC-2014.MAT"
                    "framework_code": subject.framework_code,
                    "code": subject.code,
                    "name": subject.name,
                    "icon": subject.icon,
                    "color": subject.color,
                }
                for subject in subjects
            ]

            if not subjects_data:
                framework_info = f" (framework: {framework_code})" if framework_code else ""
                return ToolResult(
                    success=True,
                    data={
                        "subjects": [],
                        "count": 0,
                        "grade_level": grade_level,
                        "framework_code": framework_code,
                        "message": f"No subjects found for grade {grade_level}{framework_info}. The curriculum may not be set up for this grade level.",
                    },
                )

            # Build human-readable message with full codes for LLM to use in follow-up calls
            subject_list = [f"{s['name']} (full_code: {s['full_code']})" for s in subjects_data]
            message = f"Found {len(subjects_data)} subjects for grade {grade_level}: {', '.join(subject_list)}"

            # Build UI element for frontend selection (full_code as identifier)
            ui_options = [
                UIElementOption(
                    id=s["full_code"],
                    label=s["name"],
                    icon=s.get("icon"),
                    metadata={
                        "color": s.get("color"),
                        "framework_code": s.get("framework_code"),
                        "code": s.get("code"),
                    },
                )
                for s in subjects_data
            ]

            # Use unique ID with timestamp to prevent duplicate key issues
            unique_id = f"subject_selection_{int(time.time() * 1000)}"
            ui_element = UIElement(
                type=UIElementType.SINGLE_SELECT,
                id=unique_id,
                title="Choose a Subject",
                options=ui_options,
                allow_text_input=True,
                placeholder="Select a subject or type to search...",
            )

            # Passthrough data for frontend with navigation context
            # Intent is not yet clear at subject selection stage
            # (could be practice, learning, or review)
            passthrough_data = {
                "subjects": subjects_data,
                "grade_level": grade_level,
                "framework_code": framework_code,
                "intent": "subject_selection",
                "navigation": {
                    "type": None,  # Will be determined after subject selection
                    "ready": False,
                    "route": None,
                    "params": {},
                    "awaiting": "subject_selection",
                },
            }

            return ToolResult(
                success=True,
                data={
                    "subjects": subjects_data,
                    "count": len(subjects_data),
                    "grade_level": grade_level,
                    "framework_code": framework_code,
                    "message": message,
                },
                ui_element=ui_element,
                passthrough_data=passthrough_data,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get subjects: {str(e)}",
            )
