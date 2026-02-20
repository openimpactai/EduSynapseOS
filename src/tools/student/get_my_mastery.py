# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get my mastery tool.

This tool retrieves the student's mastery levels across topics,
showing their progress and strengths in different subjects.

Uses code-based composite keys from Central Curriculum structure.
SemanticMemory stores entity_full_code (e.g., "UK-NC-2014.MAT.Y4.NPV.001")
instead of UUID references.
"""

from typing import Any

from sqlalchemy import and_, func, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.infrastructure.database.models.tenant.curriculum import Subject, Topic, Unit
from src.infrastructure.database.models.tenant.memory import SemanticMemory


class GetMyMasteryTool(BaseTool):
    """Tool to get student's mastery levels across topics.

    Queries semantic_memories table to retrieve mastery data,
    joined with topics and subjects for context using composite keys.

    Used when:
    - Student asks "How am I doing?"
    - Student asks "What am I good at?"
    - Agent needs to show progress overview
    """

    @property
    def name(self) -> str:
        return "get_my_mastery"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_my_mastery",
                "description": (
                    "Get the student's mastery levels across topics. "
                    "Use this when:\n"
                    "- Student asks 'How am I doing?'\n"
                    "- Student asks 'What subjects am I good at?'\n"
                    "- Student asks about their progress\n"
                    "- You want to celebrate their achievements\n\n"
                    "Returns mastery levels by subject and highlights strengths."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_full_code": {
                            "type": "string",
                            "description": (
                                "Filter by specific subject full code (e.g., 'UK-NC-2014.MAT'). "
                                "If not provided, shows all subjects."
                            ),
                        },
                        "min_attempts": {
                            "type": "integer",
                            "description": (
                                "Minimum attempts required to include a topic. "
                                "Default: 3 (to filter out topics with too little data)"
                            ),
                        },
                        "show_strengths": {
                            "type": "boolean",
                            "description": (
                                "Include list of strongest topics. Default: true"
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
        """Execute the get_my_mastery tool.

        Queries semantic_memories joined with topics to get mastery data.

        Args:
            params: Tool parameters from LLM.
                - subject_full_code: Optional subject full code filter (e.g., "UK-NC-2014.MAT")
                - min_attempts: Minimum attempts to include
                - show_strengths: Whether to show top topics
            context: Execution context with student_id.

        Returns:
            ToolResult with mastery data by subject and topic.
        """
        subject_full_code = params.get("subject_full_code")
        min_attempts = params.get("min_attempts", 3)
        show_strengths = params.get("show_strengths", True)

        # Ensure min_attempts is reasonable
        min_attempts = max(1, min(min_attempts, 10))

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

            # Build base query for topic-level mastery
            # Join SemanticMemory with Topic using entity_full_code matching concatenated topic codes
            # Then join Topic with Unit and Subject using composite keys
            conditions = [
                SemanticMemory.student_id == str(context.student_id),
                SemanticMemory.entity_type == "topic",
                SemanticMemory.attempts_total >= min_attempts,
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
                .order_by(SemanticMemory.mastery_level.desc())
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
                return ToolResult(
                    success=True,
                    data={
                        "message": (
                            "No mastery data available yet. "
                            "Complete some practice sessions to see your progress!"
                        ),
                        "overall_mastery": 0.0,
                        "subjects": [],
                        "strengths": [],
                        "has_data": False,
                    },
                )

            # Aggregate by subject (using subject full_code as key)
            subjects_map: dict[str, dict] = {}
            all_topics = []

            for row in rows:
                memory = row[0]
                subj_full_code = f"{row.subject_framework_code}.{row.subject_code}"
                topic_full_code = memory.entity_full_code

                topic_data = {
                    "topic_full_code": topic_full_code,
                    "topic_code": row.topic_code,
                    "topic_name": row.topic_name,
                    "unit_name": row.unit_name,
                    "subject_full_code": subj_full_code,
                    "subject_name": row.subject_name,
                    "mastery": float(memory.mastery_level),
                    "attempts": memory.attempts_total,
                    "accuracy": float(memory.attempts_correct / memory.attempts_total * 100)
                        if memory.attempts_total > 0 else 0.0,
                    "streak": memory.current_streak,
                    "last_practiced": (
                        memory.last_practiced_at.isoformat()
                        if memory.last_practiced_at else None
                    ),
                }
                all_topics.append(topic_data)

                # Aggregate into subjects (keyed by subject full_code)
                if subj_full_code not in subjects_map:
                    subjects_map[subj_full_code] = {
                        "subject_full_code": subj_full_code,
                        "subject_code": row.subject_code,
                        "subject_name": row.subject_name,
                        "icon": row.subject_icon,
                        "topics_practiced": 0,
                        "topics_mastered": 0,  # mastery >= 0.8
                        "topics_proficient": 0,  # mastery >= 0.6
                        "topics_developing": 0,  # mastery >= 0.4
                        "topics_struggling": 0,  # mastery < 0.4
                        "total_mastery": 0.0,
                        "total_attempts": 0,
                        "total_correct": 0,
                    }

                subjects_map[subj_full_code]["topics_practiced"] += 1
                subjects_map[subj_full_code]["total_mastery"] += float(memory.mastery_level)
                subjects_map[subj_full_code]["total_attempts"] += memory.attempts_total
                subjects_map[subj_full_code]["total_correct"] += memory.attempts_correct

                # Categorize by mastery level
                mastery = float(memory.mastery_level)
                if mastery >= 0.8:
                    subjects_map[subj_full_code]["topics_mastered"] += 1
                elif mastery >= 0.6:
                    subjects_map[subj_full_code]["topics_proficient"] += 1
                elif mastery >= 0.4:
                    subjects_map[subj_full_code]["topics_developing"] += 1
                else:
                    subjects_map[subj_full_code]["topics_struggling"] += 1

            # Calculate averages and format subjects
            subjects_data = []
            total_mastery_sum = 0.0
            total_topics = 0

            for subj in subjects_map.values():
                avg_mastery = (
                    subj["total_mastery"] / subj["topics_practiced"]
                    if subj["topics_practiced"] > 0 else 0.0
                )
                avg_accuracy = (
                    subj["total_correct"] / subj["total_attempts"] * 100
                    if subj["total_attempts"] > 0 else 0.0
                )

                subjects_data.append({
                    "subject_full_code": subj["subject_full_code"],
                    "subject_code": subj["subject_code"],
                    "subject_name": subj["subject_name"],
                    "icon": subj["icon"],
                    "average_mastery": round(avg_mastery, 2),
                    "average_accuracy": round(avg_accuracy, 1),
                    "topics_practiced": subj["topics_practiced"],
                    "topics_mastered": subj["topics_mastered"],
                    "topics_proficient": subj["topics_proficient"],
                    "topics_developing": subj["topics_developing"],
                    "topics_struggling": subj["topics_struggling"],
                })

                total_mastery_sum += subj["total_mastery"]
                total_topics += subj["topics_practiced"]

            # Sort subjects by average mastery (strongest first)
            subjects_data.sort(key=lambda x: x["average_mastery"], reverse=True)

            # Calculate overall mastery
            overall_mastery = (
                total_mastery_sum / total_topics if total_topics > 0 else 0.0
            )

            # Get top strengths (highest mastery topics)
            strengths = []
            if show_strengths:
                strengths = [
                    {
                        "topic_name": t["topic_name"],
                        "topic_full_code": t["topic_full_code"],
                        "subject_name": t["subject_name"],
                        "mastery": t["mastery"],
                        "accuracy": t["accuracy"],
                    }
                    for t in all_topics[:5]  # Top 5
                    if t["mastery"] >= 0.6  # Only show proficient+
                ]

            # Build human-readable message
            mastery_percent = int(overall_mastery * 100)
            if mastery_percent >= 80:
                message = f"Amazing progress! Your overall mastery is {mastery_percent}%. You're doing fantastic!"
            elif mastery_percent >= 60:
                message = f"Good job! Your overall mastery is {mastery_percent}%. Keep up the great work!"
            elif mastery_percent >= 40:
                message = f"You're making progress! Your overall mastery is {mastery_percent}%. Let's keep practicing!"
            else:
                message = f"You're just getting started. Your overall mastery is {mastery_percent}%. Every practice session helps!"

            # Add subject summary
            if subjects_data:
                best_subject = subjects_data[0]
                message += f" Your strongest subject is {best_subject['subject_name']}."

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "overall_mastery": round(overall_mastery, 2),
                    "overall_mastery_percent": mastery_percent,
                    "total_topics_practiced": total_topics,
                    "subjects": subjects_data,
                    "strengths": strengths,
                    "has_data": True,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get mastery data: {str(e)}",
            )
