# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get activities tool.

This tool retrieves available learning activities for a student,
filtered by category, grade level, and difficulty.
"""

from typing import Any

from sqlalchemy import and_, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.infrastructure.database.models.tenant import CompanionActivity


# Difficulty ordering for filtering
DIFFICULTY_ORDER = {"easy": 1, "medium": 2, "hard": 3}


class GetActivitiesTool(BaseTool):
    """Tool to get available learning activities.

    Retrieves activities from the companion_activities table,
    filtered by the student's grade level, emotional state,
    and requested category/difficulty.
    """

    @property
    def name(self) -> str:
        return "get_activities"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_activities",
                "description": (
                    "Get available learning activities for the student. "
                    "Use this when:\n"
                    "- Student asks what to do\n"
                    "- Suggesting activities after emotional support\n"
                    "- Student seems bored or needs direction\n"
                    "Returns activities filtered by grade level and emotional state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["learning", "fun", "creative", "break", "all"],
                            "description": (
                                "Activity category. Use 'fun' or 'break' for "
                                "stressed/frustrated students."
                            ),
                        },
                        "max_difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": (
                                "Maximum difficulty level. Use 'easy' for "
                                "frustrated or tired students."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of activities to return (default: 3)",
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
        """Execute the get_activities tool.

        Args:
            params: Tool parameters from LLM.
                - category: Activity category or "all"
                - max_difficulty: Maximum difficulty level
                - limit: Number of activities to return
            context: Execution context.

        Returns:
            ToolResult with list of activities.
        """
        # Parse parameters with defaults
        category = params.get("category", "all")
        max_difficulty = params.get("max_difficulty", "hard")
        limit = params.get("limit", 3)

        # Ensure limit is reasonable
        limit = min(max(1, limit), 10)

        # Get max difficulty order
        max_diff_order = DIFFICULTY_ORDER.get(max_difficulty, 3)

        # Build query
        conditions = [
            CompanionActivity.is_enabled == True,  # noqa: E712
            CompanionActivity.min_grade <= context.grade_level,
            CompanionActivity.max_grade >= context.grade_level,
        ]

        # Filter by category if not "all"
        if category != "all":
            conditions.append(CompanionActivity.category == category)

        # Filter by difficulty
        if max_difficulty != "hard":
            # Only include activities at or below max difficulty
            difficulty_conditions = []
            for diff, order in DIFFICULTY_ORDER.items():
                if order <= max_diff_order:
                    difficulty_conditions.append(
                        CompanionActivity.difficulty == diff
                    )
            if difficulty_conditions:
                from sqlalchemy import or_

                conditions.append(or_(*difficulty_conditions))

        stmt = (
            select(CompanionActivity)
            .where(and_(*conditions))
            .order_by(
                CompanionActivity.category,
                CompanionActivity.display_order,
            )
            .limit(limit)
        )

        result = await context.session.execute(stmt)
        activities = result.scalars().all()

        # Format for LLM
        activities_data = [
            {
                "code": activity.code,
                "name": activity.name,
                "description": activity.description,
                "icon": activity.icon,
                "category": activity.category,
                "route": activity.route,
                "difficulty": activity.difficulty,
            }
            for activity in activities
        ]

        # Build human-readable message for LLM
        if not activities_data:
            message = f"No activities found for {category} category."
            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "activities": [],
                    "count": 0,
                    "filters_applied": {
                        "category": category,
                        "max_difficulty": max_difficulty,
                        "grade_level": context.grade_level,
                    },
                },
            )

        activity_list = ", ".join(
            f"{a['name']} ({a['category']}, {a['difficulty']})"
            for a in activities_data
        )
        message = f"Found {len(activities_data)} activities: {activity_list}"

        # Build UI element for frontend selection
        ui_options = [
            UIElementOption(
                id=a["code"],
                label=a["name"],
                description=a.get("description"),
                icon=a.get("icon"),
                metadata={
                    "category": a["category"],
                    "difficulty": a.get("difficulty"),
                    "route": a.get("route"),
                },
            )
            for a in activities_data
        ]

        ui_element = UIElement(
            type=UIElementType.SINGLE_SELECT,
            id="activity_selection",
            title="Choose an Activity",
            options=ui_options,
            allow_text_input=False,
        )

        # Determine navigation type based on category
        nav_type = "activity"
        if category == "break":
            nav_type = "break"
        elif category == "creative":
            nav_type = "creative"

        # Passthrough data for frontend with navigation context
        passthrough_data = {
            "activities": activities_data,
            "grade_level": context.grade_level,
            "intent": nav_type,
            "navigation": {
                "type": nav_type,
                "ready": False,  # Activity selection still needed
                "route": f"/{nav_type}",
                "params": {
                    "category": category,
                },
                "awaiting": "activity_selection",
            },
        }

        return ToolResult(
            success=True,
            data={
                "message": message,
                "activities": activities_data,
                "count": len(activities_data),
                "filters_applied": {
                    "category": category,
                    "max_difficulty": max_difficulty,
                    "grade_level": context.grade_level,
                },
            },
            ui_element=ui_element,
            passthrough_data=passthrough_data,
        )
