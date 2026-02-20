# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get student context tool.

This tool extracts specific student context from the pre-loaded
memory context for personalized conversations.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


# Context types that can be requested
CONTEXT_TYPES = frozenset({
    "weak_topics",
    "strong_topics",
    "interests",
    "recent_activities",
    "learning_patterns",
})


class GetStudentContextTool(BaseTool):
    """Tool to get additional student context for personalization.

    Extracts specific information from the pre-loaded memory context
    to help agents personalize their responses.
    """

    @property
    def name(self) -> str:
        return "get_student_context"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_student_context",
                "description": (
                    "Get additional student context for personalization. "
                    "Use this when:\n"
                    "- Need to know weak topics for suggestions\n"
                    "- Want to reference student interests for analogies\n"
                    "- Need to check recent activities\n"
                    "Context is already loaded, this gets specific details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string",
                            "enum": list(CONTEXT_TYPES),
                            "description": (
                                "Type of context to retrieve:\n"
                                "- weak_topics: Topics student is struggling with\n"
                                "- strong_topics: Topics student has mastered\n"
                                "- interests: Student's interests and hobbies\n"
                                "- recent_activities: Recent learning activities\n"
                                "- learning_patterns: Optimal learning patterns"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of items to return (default: 5)",
                        },
                    },
                    "required": ["context_type"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_student_context tool.

        Args:
            params: Tool parameters from LLM.
                - context_type: Type of context to retrieve
                - limit: Maximum items to return
            context: Execution context with pre-loaded memory.

        Returns:
            ToolResult with requested context data.
        """
        context_type = params.get("context_type")
        limit = params.get("limit", 5)

        if not context_type:
            return ToolResult(
                success=False,
                error="Missing required parameter: context_type",
            )

        if context_type not in CONTEXT_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid context_type: {context_type}. Valid: {', '.join(CONTEXT_TYPES)}",
            )

        # Ensure limit is reasonable
        limit = min(max(1, limit), 10)

        # Check if memory context is available
        if not context.memory_context:
            return ToolResult(
                success=True,
                data={
                    "context_type": context_type,
                    "items": [],
                    "message": "No memory context available yet.",
                },
            )

        # Extract requested context
        if context_type == "weak_topics":
            return self._get_weak_topics(context, limit)
        elif context_type == "strong_topics":
            return self._get_strong_topics(context, limit)
        elif context_type == "interests":
            return self._get_interests(context, limit)
        elif context_type == "recent_activities":
            return self._get_recent_activities(context, limit)
        elif context_type == "learning_patterns":
            return self._get_learning_patterns(context)

        return ToolResult(
            success=False,
            error=f"Unhandled context_type: {context_type}",
        )

    def _get_weak_topics(self, context: ToolContext, limit: int) -> ToolResult:
        """Extract weak/struggling topics from memory context."""
        mc = context.memory_context
        semantic = mc.semantic if mc else None

        if not semantic:
            return ToolResult(
                success=True,
                data={
                    "message": "No topic data available yet.",
                    "context_type": "weak_topics",
                    "items": [],
                    "count": 0,
                },
            )

        # Get struggling topic count
        struggling_count = getattr(semantic, "topics_struggling", 0)

        return ToolResult(
            success=True,
            data={
                "context_type": "weak_topics",
                "struggling_count": struggling_count,
                "total_topics": getattr(semantic, "total_topics", 0),
                "overall_mastery": getattr(semantic, "overall_mastery", 0.0),
                "message": (
                    f"Student has {struggling_count} topics they're struggling with."
                    if struggling_count > 0
                    else "No struggling topics identified yet."
                ),
            },
        )

    def _get_strong_topics(self, context: ToolContext, limit: int) -> ToolResult:
        """Extract mastered topics from memory context."""
        mc = context.memory_context
        semantic = mc.semantic if mc else None

        if not semantic:
            return ToolResult(
                success=True,
                data={
                    "message": "No topic data available yet.",
                    "context_type": "strong_topics",
                    "items": [],
                    "count": 0,
                },
            )

        mastered_count = getattr(semantic, "topics_mastered", 0)

        return ToolResult(
            success=True,
            data={
                "context_type": "strong_topics",
                "mastered_count": mastered_count,
                "total_topics": getattr(semantic, "total_topics", 0),
                "overall_mastery": getattr(semantic, "overall_mastery", 0.0),
                "message": (
                    f"Student has mastered {mastered_count} topics!"
                    if mastered_count > 0
                    else "No mastered topics yet - keep learning!"
                ),
            },
        )

    def _get_interests(self, context: ToolContext, limit: int) -> ToolResult:
        """Extract student interests from memory context."""
        mc = context.memory_context
        associative = mc.associative if mc else None

        if not associative:
            return ToolResult(
                success=True,
                data={
                    "message": "No interests data available yet.",
                    "context_type": "interests",
                    "items": [],
                    "count": 0,
                },
            )

        interests = getattr(associative, "interests", [])
        interest_items = [
            {
                "name": getattr(i, "name", str(i)),
                "category": getattr(i, "category", "general"),
                "strength": getattr(i, "strength", 0.5),
            }
            for i in interests[:limit]
        ]

        # Build human-readable message
        if interest_items:
            interest_names = ", ".join(i["name"] for i in interest_items)
            message = f"Student interests: {interest_names}"
        else:
            message = "No interests identified yet."

        return ToolResult(
            success=True,
            data={
                "message": message,
                "context_type": "interests",
                "items": interest_items,
                "count": len(interest_items),
                "total": len(interests),
            },
        )

    def _get_recent_activities(self, context: ToolContext, limit: int) -> ToolResult:
        """Extract recent learning activities from memory context."""
        mc = context.memory_context
        episodic = mc.episodic if mc else []

        if not episodic:
            return ToolResult(
                success=True,
                data={
                    "message": "No recent activities found.",
                    "context_type": "recent_activities",
                    "items": [],
                    "count": 0,
                },
            )

        activities = [
            {
                "type": getattr(e, "event_type", "unknown"),
                "topic": getattr(e, "topic", None),
                "timestamp": str(getattr(e, "timestamp", "")),
            }
            for e in episodic[:limit]
        ]

        # Build human-readable message
        if activities:
            activity_types = ", ".join(a["type"] for a in activities)
            message = f"Recent activities ({len(activities)}): {activity_types}"
        else:
            message = "No recent activities found."

        return ToolResult(
            success=True,
            data={
                "message": message,
                "context_type": "recent_activities",
                "items": activities,
                "count": len(activities),
            },
        )

    def _get_learning_patterns(self, context: ToolContext) -> ToolResult:
        """Extract learning patterns from memory context."""
        mc = context.memory_context
        procedural = mc.procedural if mc else None

        if not procedural:
            return ToolResult(
                success=True,
                data={
                    "context_type": "learning_patterns",
                    "patterns": {},
                    "message": "No learning patterns identified yet.",
                },
            )

        patterns = {
            "best_time_of_day": getattr(procedural, "best_time_of_day", None),
            "optimal_session_duration": getattr(
                procedural, "optimal_session_duration", None
            ),
            "preferred_content_format": getattr(
                procedural, "preferred_content_format", None
            ),
            "hint_preference": getattr(procedural, "hint_preference", None),
        }

        # Filter out None values
        patterns = {k: v for k, v in patterns.items() if v is not None}

        # Build human-readable message
        if patterns:
            pattern_summary = ", ".join(f"{k}: {v}" for k, v in patterns.items())
            message = f"Learning patterns: {pattern_summary}"
        else:
            message = "No learning patterns identified yet."

        return ToolResult(
            success=True,
            data={
                "message": message,
                "context_type": "learning_patterns",
                "patterns": patterns,
                "has_patterns": len(patterns) > 0,
            },
        )
