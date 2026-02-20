# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get review schedule tool.

This tool retrieves the spaced repetition review schedule
using the FSRS algorithm for optimal review timing.

Uses code-based composite keys from Central Curriculum structure.
ReviewItem stores topic_full_code and topic_*_code fields.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.infrastructure.database.models.tenant.curriculum import Topic
from src.infrastructure.database.models.tenant.review import ReviewItem


class GetReviewScheduleTool(BaseTool):
    """Tool to get spaced repetition review schedule.

    Retrieves items due for review based on FSRS algorithm
    scheduling to help students maintain optimal retention.
    """

    @property
    def name(self) -> str:
        return "get_review_schedule"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_review_schedule",
                "description": (
                    "Get spaced repetition review schedule. "
                    "Use this when:\n"
                    "- Student asks what to study\n"
                    "- Suggesting review activities\n"
                    "- Checking if student has pending reviews\n"
                    "Uses FSRS algorithm for optimal review timing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_overdue": {
                            "type": "boolean",
                            "description": (
                                "Include overdue items (default: true)"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Number of items to return (default: 5)"
                            ),
                        },
                        "hours_ahead": {
                            "type": "integer",
                            "description": (
                                "Include items due within this many hours "
                                "(default: 4)"
                            ),
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
        """Execute the get_review_schedule tool.

        Args:
            params: Tool parameters from LLM.
                - include_overdue: Whether to include overdue items
                - limit: Maximum items to return
                - hours_ahead: Hours ahead to look for due items
            context: Execution context.

        Returns:
            ToolResult with review schedule.
        """
        include_overdue = params.get("include_overdue", True)
        limit = params.get("limit", 5)
        hours_ahead = params.get("hours_ahead", 4)

        # Ensure reasonable limits
        limit = min(max(1, limit), 20)
        hours_ahead = min(max(1, hours_ahead), 24)

        now = datetime.now(timezone.utc)
        due_before = now + timedelta(hours=hours_ahead)

        # Build query conditions
        conditions = [
            ReviewItem.student_id == str(context.student_id),
        ]

        if include_overdue:
            # Include all items due before the cutoff time
            conditions.append(ReviewItem.due <= due_before)
        else:
            # Only items due between now and cutoff (not overdue)
            conditions.append(ReviewItem.due >= now)
            conditions.append(ReviewItem.due <= due_before)

        # Build the join using SQL concat to match topic_full_code with Topic composite key
        topic_full_code_expr = func.concat(
            Topic.framework_code, ".",
            Topic.subject_code, ".",
            Topic.grade_code, ".",
            Topic.unit_code, ".",
            Topic.code,
        )

        # Build ReviewItem's topic_full_code expression for join
        review_topic_full_code_expr = func.concat(
            ReviewItem.topic_framework_code, ".",
            ReviewItem.topic_subject_code, ".",
            ReviewItem.topic_grade_code, ".",
            ReviewItem.topic_unit_code, ".",
            ReviewItem.topic_code,
        )

        # Join with Topic to get topic name
        stmt = (
            select(
                ReviewItem,
                Topic.name.label("topic_name"),
            )
            .outerjoin(
                Topic,
                review_topic_full_code_expr == topic_full_code_expr,
            )
            .where(and_(*conditions))
            .order_by(ReviewItem.due.asc())
            .limit(limit)
        )

        result = await context.session.execute(stmt)
        rows = result.all()

        # Categorize items
        overdue_items = []
        due_today_items = []
        upcoming_items = []

        for row in rows:
            item = row[0]
            topic_name = row.topic_name

            # Use item's topic_full_code property
            item_data = {
                "id": str(item.id),
                "item_type": item.item_type,
                "item_full_code": item.item_full_code,
                "topic_full_code": item.topic_full_code,
                "topic_name": topic_name,
                "due": item.due.isoformat() if item.due else None,
                "reps": item.reps,
                "state": self._get_state_label(item.state),
                "is_overdue": item.due < now if item.due else False,
            }

            if item.due and item.due < now:
                overdue_items.append(item_data)
            elif item.due and item.due.date() == now.date():
                due_today_items.append(item_data)
            else:
                upcoming_items.append(item_data)

        # Summary statistics
        total_overdue = len(overdue_items)
        total_due_today = len(due_today_items)
        total_upcoming = len(upcoming_items)

        # Generate message for LLM
        if total_overdue > 0:
            message = f"Student has {total_overdue} overdue reviews that need attention."
        elif total_due_today > 0:
            message = f"Student has {total_due_today} reviews due today."
        elif total_upcoming > 0:
            message = f"Student has {total_upcoming} reviews coming up soon."
        else:
            message = "No pending reviews - student is up to date!"

        has_pending = len(rows) > 0

        # Passthrough data for frontend with navigation context
        # If there are pending reviews, navigation is ready
        passthrough_data = {
            "reviews": {
                "overdue": overdue_items,
                "due_today": due_today_items,
                "upcoming": upcoming_items,
            },
            "summary": {
                "total_overdue": total_overdue,
                "total_due_today": total_due_today,
                "total_upcoming": total_upcoming,
                "total": len(rows),
            },
            "intent": "review",
            "navigation": {
                "type": "review",
                "ready": has_pending,  # Ready if there are reviews
                "route": "/review",
                "params": {
                    "has_overdue": total_overdue > 0,
                    "has_due_today": total_due_today > 0,
                },
                "awaiting": None if has_pending else "no_reviews",
            },
        }

        return ToolResult(
            success=True,
            data={
                "overdue": overdue_items,
                "due_today": due_today_items,
                "upcoming": upcoming_items,
                "summary": {
                    "total_overdue": total_overdue,
                    "total_due_today": total_due_today,
                    "total_upcoming": total_upcoming,
                    "total": len(rows),
                },
                "message": message,
                "has_pending_reviews": has_pending,
            },
            passthrough_data=passthrough_data,
        )

    def _get_state_label(self, state: int) -> str:
        """Get human-readable label for FSRS state."""
        state_labels = {
            0: "new",
            1: "learning",
            2: "review",
            3: "relearning",
        }
        return state_labels.get(state, "unknown")
