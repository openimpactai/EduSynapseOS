# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get topics tool.

This tool retrieves available topics for a specific subject at the student's grade level.
Used by agents to show topic options after a subject is selected.

Returns a UIElement for frontend to render a searchable selection control,
along with passthrough data containing full topic details.
"""

import time
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.domains.curriculum import CurriculumLookup
from src.infrastructure.database.models.tenant.curriculum import (
    GradeLevel,
    Subject,
    Topic,
    Unit,
)


class GetTopicsTool(BaseTool):
    """Tool to get available topics for a subject at the student's grade level.

    Queries topics through the curriculum hierarchy:
    Subject → Unit (filtered by grade) → Topic

    Used in clarification flow when:
    - Student has selected a subject
    - Agent needs to ask about specific topic or random practice

    The tool should be called AFTER get_subjects when the student
    has indicated which subject they're interested in.
    """

    @property
    def name(self) -> str:
        return "get_topics"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_topics",
                "description": (
                    "Get topics for a specific subject at the student's grade level. "
                    "CRITICAL: Call this AFTER the student has chosen a subject.\n\n"
                    "Use cases:\n"
                    "- Student chose 'Mathematics' → Call get_topics to show available topics\n"
                    "- Student wants specific topic practice → Call get_topics first\n"
                    "- You want to suggest topics → Call get_topics with the subject_code\n\n"
                    "After getting topics, ask the student:\n"
                    "- Do they want a specific topic?\n"
                    "- Or random/mixed practice from all topics?\n\n"
                    "Returns a list of topics with codes, names, and unit info."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_full_code": {
                            "type": "string",
                            "description": "Subject full code from get_subjects result (e.g., 'UK-NC-2014.MAT'). Preferred identifier.",
                        },
                        "subject_name": {
                            "type": "string",
                            "description": "Subject name (e.g., 'Mathematics'). Use if subject_full_code is not available.",
                        },
                        "include_mastery": {
                            "type": "boolean",
                            "description": "Include student's mastery level for each topic. Optional, defaults to false. (not yet implemented)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of topics to return. Optional, defaults to 10, max 20.",
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
        """Execute the get_topics tool.

        Queries topics through the curriculum hierarchy for the student's grade level.

        Args:
            params: Tool parameters from LLM.
                - subject_full_code: Subject full code to get topics for (e.g., "UK-NC-2014.MAT")
                - subject_name: Subject name as fallback
                - include_mastery: Whether to include mastery (not yet implemented)
                - limit: Maximum topics to return
            context: Execution context with grade_level.

        Returns:
            ToolResult with list of topics.
        """
        subject_full_code = params.get("subject_full_code")
        subject_name = params.get("subject_name")
        limit = min(params.get("limit", 10), 20)
        grade_level = context.grade_level

        if not subject_full_code and not subject_name:
            return ToolResult(
                success=False,
                error="Missing required parameter: subject_full_code or subject_name. Call get_subjects first to get available subjects.",
            )

        try:
            subject = None
            lookup = CurriculumLookup(context.session)

            # Try subject_full_code first (preferred)
            if subject_full_code:
                # Parse full_code to get framework_code and code
                parts = subject_full_code.split(".")
                if len(parts) == 2:
                    framework_code, code = parts
                    subject = await lookup.get_subject(framework_code, code)

            # Fallback to subject_name if subject not found by code
            if not subject and subject_name:
                # Case-insensitive search by name, filtered by framework_code if available
                subject_stmt = select(Subject).where(
                    func.lower(Subject.name) == func.lower(subject_name)
                )
                # Apply framework filter to prevent cross-curriculum pollution
                if context.framework_code:
                    subject_stmt = subject_stmt.where(
                        Subject.framework_code == context.framework_code
                    )
                subject_result = await context.session.execute(subject_stmt)
                subject = subject_result.scalar_one_or_none()

            if not subject:
                identifier = subject_full_code or subject_name
                return ToolResult(
                    success=False,
                    error=f"Subject not found: {identifier}. Call get_subjects to see available subjects.",
                )

            # Query topics through the hierarchy using composite keys
            # Topic → Unit (via composite key) → GradeLevel (via composite key, filtered by sequence)
            stmt = (
                select(Topic)
                .join(
                    Unit,
                    (Topic.framework_code == Unit.framework_code)
                    & (Topic.subject_code == Unit.subject_code)
                    & (Topic.grade_code == Unit.grade_code)
                    & (Topic.unit_code == Unit.code),
                )
                .join(
                    GradeLevel,
                    (Unit.framework_code == GradeLevel.framework_code)
                    & (Unit.grade_code == GradeLevel.code),
                )
                .where(
                    Topic.framework_code == subject.framework_code,
                    Topic.subject_code == subject.code,
                    GradeLevel.sequence == grade_level,
                )
                .options(selectinload(Topic.unit))
                .order_by(Unit.sequence, Topic.sequence)
                .limit(limit)
            )

            result = await context.session.execute(stmt)
            topics = result.scalars().all()

            # Format topics for LLM (full_code is primary identifier)
            topics_data = [
                {
                    "full_code": topic.full_code,  # e.g., "UK-NC-2014.MAT.Y4.NPV.001"
                    "code": topic.code,
                    "name": topic.name,
                    "unit_name": topic.unit.name if topic.unit else None,
                    "unit_code": topic.unit_code,
                    "difficulty": float(topic.base_difficulty),
                }
                for topic in topics
            ]

            if not topics_data:
                return ToolResult(
                    success=True,
                    data={
                        "topics": [],
                        "count": 0,
                        "subject_full_code": subject.full_code,
                        "subject_name": subject.name,
                        "grade_level": grade_level,
                        "message": f"No topics found for {subject.name} at grade {grade_level}. The curriculum may not cover this subject at this level.",
                    },
                )

            # Build human-readable message with full codes for LLM to use in follow-up calls
            # IMPORTANT: Use full_code when calling handoff_to_practice for direct navigation
            topic_list = [f"{t['name']} (topic_full_code: {t['full_code']})" for t in topics_data[:5]]
            more_text = f" and {len(topics_data) - 5} more" if len(topics_data) > 5 else ""
            message = f"Found {len(topics_data)} topics for {subject.name}: {', '.join(topic_list)}{more_text}. IMPORTANT: When calling handoff_to_practice, use the topic_full_code parameter with the full code (e.g., '{topics_data[0]['full_code']}')."

            # Build UI element for frontend selection (full_code as identifier)
            # Use SEARCHABLE_SELECT for topics since there can be many
            ui_options = [
                UIElementOption(
                    id=t["full_code"],
                    label=t["name"],
                    description=t.get("unit_name"),
                    metadata={
                        "difficulty": t.get("difficulty"),
                        "unit_name": t.get("unit_name"),
                        "code": t.get("code"),
                        "unit_code": t.get("unit_code"),
                    },
                )
                for t in topics_data
            ]

            # Add "Random/Mixed Practice" option at the beginning
            random_option = UIElementOption(
                id="random",
                label="Random Practice",
                description="Practice questions from all topics",
                icon="shuffle",
                metadata={"is_random": True},
            )
            ui_options.insert(0, random_option)

            # Use unique ID with timestamp to prevent duplicate key issues
            unique_id = f"topic_selection_{int(time.time() * 1000)}"
            ui_element = UIElement(
                type=UIElementType.SEARCHABLE_SELECT,
                id=unique_id,
                title=f"Choose a Topic in {subject.name}",
                options=ui_options,
                searchable=True,
                allow_text_input=True,
                placeholder="Search topics or select 'Random Practice'...",
            )

            # Passthrough data for frontend with navigation context
            # At topic selection stage, intent is practice and subject is already selected
            passthrough_data = {
                "topics": topics_data,
                "subject_full_code": subject.full_code,
                "subject_name": subject.name,
                "grade_level": grade_level,
                "intent": "practice",
                "navigation": {
                    "type": "practice",
                    "ready": False,  # Topic selection still needed
                    "route": "/practice",
                    "params": {
                        "subject_full_code": subject.full_code,
                        "subject_name": subject.name,
                    },
                    "awaiting": "topic_selection",
                },
            }

            return ToolResult(
                success=True,
                data={
                    "topics": topics_data,
                    "count": len(topics_data),
                    "subject_full_code": subject.full_code,
                    "subject_name": subject.name,
                    "grade_level": grade_level,
                    "message": message,
                },
                ui_element=ui_element,
                passthrough_data=passthrough_data,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get topics: {str(e)}",
            )
