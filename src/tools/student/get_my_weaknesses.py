# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get my weaknesses tool.

This tool identifies topics where the student is struggling,
helping them focus on areas that need improvement.

Uses code-based composite keys from Central Curriculum structure.
SemanticMemory stores entity_full_code (e.g., "UK-NC-2014.MAT.Y4.NPV.001")
instead of UUID references.
"""

import time
from typing import Any

from sqlalchemy import and_, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.infrastructure.database.models.tenant.curriculum import Subject, Topic, Unit
from src.infrastructure.database.models.tenant.memory import SemanticMemory


# Threshold for considering a topic as "weak"
WEAK_MASTERY_THRESHOLD = 0.4


class GetMyWeaknessesTool(BaseTool):
    """Tool to identify topics where the student is struggling.

    Queries semantic_memories for topics with low mastery levels,
    providing actionable recommendations for improvement.

    Used when:
    - Student asks "What should I work on?"
    - Student asks "Where do I struggle?"
    - Agent wants to suggest practice areas
    """

    @property
    def name(self) -> str:
        return "get_my_weaknesses"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_my_weaknesses",
                "description": (
                    "Identify topics where the student is struggling. "
                    "Use this when:\n"
                    "- Student asks 'What should I work on?'\n"
                    "- Student asks 'Where do I struggle?'\n"
                    "- Student asks 'What do I need to practice?'\n"
                    "- You want to suggest improvement areas\n\n"
                    "Returns weak topics with recommendations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_full_code": {
                            "type": "string",
                            "description": (
                                "Filter by specific subject full code (e.g., 'UK-NC-2014.MAT'). "
                                "If not provided, shows weak topics across all subjects."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of weak topics to return. "
                                "Default: 5"
                            ),
                        },
                        "threshold": {
                            "type": "number",
                            "description": (
                                "Mastery threshold below which a topic is considered weak. "
                                "Default: 0.4 (40%)"
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
        """Execute the get_my_weaknesses tool.

        Queries semantic_memories for low-mastery topics.

        Args:
            params: Tool parameters from LLM.
                - subject_full_code: Optional subject full code filter (e.g., "UK-NC-2014.MAT")
                - limit: Maximum topics to return
                - threshold: Mastery threshold for "weak"
            context: Execution context with student_id.

        Returns:
            ToolResult with weak topics and recommendations.
        """
        subject_full_code = params.get("subject_full_code")
        limit = params.get("limit", 5)
        threshold = params.get("threshold", WEAK_MASTERY_THRESHOLD)

        # Ensure limits are reasonable
        limit = max(1, min(limit, 10))
        threshold = max(0.1, min(threshold, 0.6))

        try:
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

            # Build query for weak topics
            conditions = [
                SemanticMemory.student_id == str(context.student_id),
                SemanticMemory.entity_type == "topic",
                SemanticMemory.mastery_level < threshold,
                SemanticMemory.attempts_total >= 1,  # At least one attempt
            ]

            # Build the join using SQL concat to match entity_full_code with topic composite key
            topic_full_code_expr = func.concat(
                Topic.framework_code, ".",
                Topic.subject_code, ".",
                Topic.grade_code, ".",
                Topic.unit_code, ".",
                Topic.code,
            )

            # Query with composite key joins
            stmt = (
                select(
                    SemanticMemory,
                    Topic.name.label("topic_name"),
                    Topic.code.label("topic_code"),
                    Topic.framework_code.label("topic_framework_code"),
                    Topic.subject_code.label("topic_subject_code"),
                    Topic.grade_code.label("topic_grade_code"),
                    Topic.unit_code.label("topic_unit_code"),
                    Unit.name.label("unit_name"),
                    Subject.name.label("subject_name"),
                    Subject.code.label("subject_code"),
                    Subject.framework_code.label("subject_framework_code"),
                    Subject.icon.label("subject_icon"),
                )
                .join(
                    Topic,
                    SemanticMemory.entity_full_code == topic_full_code_expr,
                )
                .join(
                    Unit,
                    (Topic.framework_code == Unit.framework_code)
                    & (Topic.subject_code == Unit.subject_code)
                    & (Topic.grade_code == Unit.grade_code)
                    & (Topic.unit_code == Unit.code),
                )
                .join(
                    Subject,
                    (Topic.framework_code == Subject.framework_code)
                    & (Topic.subject_code == Subject.code),
                )
                .where(and_(*conditions))
                .order_by(
                    SemanticMemory.mastery_level.asc(),  # Weakest first
                    SemanticMemory.attempts_total.desc(),  # More attempts = more data
                )
                .limit(limit)
            )

            # Apply subject filter if provided
            if subject_framework_code and subject_code:
                stmt = stmt.where(
                    Subject.framework_code == subject_framework_code,
                    Subject.code == subject_code,
                )

            result = await context.session.execute(stmt)
            rows = result.all()

            if not rows:
                # No weak topics found - check if student has any data
                check_stmt = (
                    select(SemanticMemory)
                    .where(
                        SemanticMemory.student_id == str(context.student_id),
                        SemanticMemory.entity_type == "topic",
                    )
                    .limit(1)
                )
                check_result = await context.session.execute(check_stmt)
                has_any_data = check_result.scalar_one_or_none() is not None

                if has_any_data:
                    message = (
                        "Great news! You don't have any topics below "
                        f"{int(threshold * 100)}% mastery. Keep up the excellent work!"
                    )
                else:
                    message = (
                        "No practice data yet. Complete some practice sessions "
                        "to track your progress!"
                    )

                return ToolResult(
                    success=True,
                    data={
                        "message": message,
                        "weak_topics": [],
                        "count": 0,
                        "has_weak_topics": False,
                        "threshold": threshold,
                    },
                )

            # Format weak topics
            weak_topics = []
            for row in rows:
                memory = row[0]
                mastery = float(memory.mastery_level)
                attempts = memory.attempts_total
                accuracy = (
                    float(memory.attempts_correct / attempts * 100)
                    if attempts > 0 else 0.0
                )
                topic_full_code = memory.entity_full_code
                subj_full_code = f"{row.subject_framework_code}.{row.subject_code}"

                # Generate recommendation based on mastery level
                if mastery < 0.2:
                    suggested_action = "Start with basics - try some easy questions"
                elif mastery < 0.3:
                    suggested_action = "Review fundamentals and practice more"
                else:
                    suggested_action = "Keep practicing to strengthen understanding"

                weak_topics.append({
                    "topic_full_code": topic_full_code,
                    "topic_code": row.topic_code,
                    "topic_name": row.topic_name,
                    "unit_name": row.unit_name,
                    "subject_full_code": subj_full_code,
                    "subject_code": row.subject_code,
                    "subject_name": row.subject_name,
                    "subject_icon": row.subject_icon,
                    "mastery": round(mastery, 2),
                    "mastery_percent": int(mastery * 100),
                    "attempts": attempts,
                    "accuracy": round(accuracy, 1),
                    "current_streak": memory.current_streak,
                    "last_practiced": (
                        memory.last_practiced_at.isoformat()
                        if memory.last_practiced_at else None
                    ),
                    "suggested_action": suggested_action,
                })

            # Build UI element for topic selection (full_code as identifier)
            ui_options = [
                UIElementOption(
                    id=t["topic_full_code"],
                    label=t["topic_name"],
                    description=f"{t['subject_name']} - {t['mastery_percent']}% mastery",
                    icon=t["subject_icon"],
                    metadata={
                        "subject_full_code": t["subject_full_code"],
                        "subject_name": t["subject_name"],
                        "mastery": t["mastery"],
                    },
                )
                for t in weak_topics
            ]

            # Use unique ID with timestamp to prevent duplicate key issues
            unique_id = f"weak_topic_selection_{int(time.time() * 1000)}"
            ui_element = UIElement(
                type=UIElementType.SINGLE_SELECT,
                id=unique_id,
                title="Topics to Improve",
                options=ui_options,
                allow_text_input=False,
                placeholder="Choose a topic to practice...",
            )

            # Build human-readable message
            topic_count = len(weak_topics)
            weakest = weak_topics[0]

            if topic_count == 1:
                message = (
                    f"I found 1 topic that needs more practice: "
                    f"{weakest['topic_name']} ({weakest['mastery_percent']}% mastery). "
                    f"{weakest['suggested_action']}."
                )
            else:
                message = (
                    f"I found {topic_count} topics that need more practice. "
                    f"Your weakest is {weakest['topic_name']} ({weakest['mastery_percent']}% mastery). "
                    "Would you like to work on one of these?"
                )

            # Passthrough data for frontend navigation
            passthrough_data = {
                "weak_topics": weak_topics,
                "intent": "practice_weak",
                "navigation": {
                    "type": "practice",
                    "ready": False,
                    "route": "/practice",
                    "params": {},
                    "awaiting": "weak_topic_selection",
                },
            }

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "weak_topics": weak_topics,
                    "count": topic_count,
                    "has_weak_topics": True,
                    "threshold": threshold,
                    "recommendation": (
                        f"Focus on {weakest['topic_name']} first - "
                        "it needs the most attention."
                    ),
                },
                ui_element=ui_element,
                passthrough_data=passthrough_data,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get weakness data: {str(e)}",
            )
