# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get learning objectives tool for Learning Tutor.

This tool retrieves learning objectives for a specific topic to provide
curriculum context for teaching. Uses composite keys from Central Curriculum.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.tools.base import BaseTool, ToolContext, ToolResult
from src.domains.curriculum import CurriculumLookup
from src.infrastructure.database.models.tenant.curriculum import (
    LearningObjective,
    Topic,
)

logger = logging.getLogger(__name__)


class GetLearningObjectivesTool(BaseTool):
    """Get learning objectives for a topic.

    This tool queries the curriculum database to retrieve:
    - Learning objectives for the specified topic
    - Topic description and metadata

    The returned data provides curriculum context for the Learning Tutor
    to understand what needs to be taught.

    Usage:
        Called by Learning Tutor workflow in _load_curriculum_content node
        to populate learning_objectives and topic_description in state.

    Returns:
        ToolResult with:
        - learning_objectives: List of objectives
        - topic_description: Generated description from objectives
        - message: Human-readable summary for LLM
    """

    @property
    def name(self) -> str:
        """Return tool name."""
        return "get_learning_objectives"

    @property
    def definition(self) -> dict[str, Any]:
        """Return OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "get_learning_objectives",
                "description": (
                    "Get learning objectives for a topic. "
                    "Use this to understand what needs to be taught. "
                    "Returns structured curriculum data to guide teaching."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_full_code": {
                            "type": "string",
                            "description": "Full code of the topic (e.g., 'UK-NC-2014.MAT.Y4.NPV.001')",
                        },
                    },
                    "required": ["topic_full_code"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool to get learning objectives.

        Args:
            params: Tool parameters containing topic_full_code.
            context: Execution context with database session.

        Returns:
            ToolResult with learning objectives.
        """
        topic_full_code = params.get("topic_full_code")

        if not topic_full_code:
            return ToolResult(
                success=False,
                error="topic_full_code is required",
            )

        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available",
            )

        try:
            # Parse topic_full_code to get composite key parts
            parts = topic_full_code.split(".")
            if len(parts) != 5:
                return ToolResult(
                    success=False,
                    error=f"Invalid topic_full_code format: {topic_full_code}. Expected format: framework.subject.grade.unit.code",
                )

            framework_code, subject_code, grade_code, unit_code, topic_code = parts

            # Query topic with learning objectives using composite key
            stmt = (
                select(Topic)
                .options(selectinload(Topic.learning_objectives))
                .where(
                    Topic.framework_code == framework_code,
                    Topic.subject_code == subject_code,
                    Topic.grade_code == grade_code,
                    Topic.unit_code == unit_code,
                    Topic.code == topic_code,
                )
            )

            result = await context.session.execute(stmt)
            topic = result.scalar_one_or_none()

            if not topic:
                return ToolResult(
                    success=False,
                    error=f"Topic not found: {topic_full_code}",
                )

            # Build structured response
            objectives_data = []

            for obj in sorted(topic.learning_objectives, key=lambda x: x.sequence):
                obj_dict = {
                    "full_code": obj.full_code,
                    "code": obj.code,
                    "objective": obj.objective,
                    "bloom_level": obj.bloom_level,
                    "mastery_threshold": float(obj.mastery_threshold),
                    "sequence": obj.sequence,
                }
                objectives_data.append(obj_dict)

            # Generate topic description from objectives
            topic_description = self._generate_topic_description(
                topic_name=topic.name,
                topic_desc=topic.description,
                objectives=objectives_data,
            )

            # Build human-readable message for LLM
            message = self._build_message(
                topic_name=topic.name,
                objectives_count=len(objectives_data),
            )

            logger.info(
                "Retrieved %d learning objectives for topic %s",
                len(objectives_data),
                topic.name,
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "learning_objectives": objectives_data,
                    "topic_description": topic_description,
                    "topic_name": topic.name,
                    "topic_full_code": topic.full_code,
                    "base_difficulty": float(topic.base_difficulty),
                    "estimated_minutes": topic.estimated_minutes,
                },
                passthrough_data={
                    "learning_objectives": objectives_data,
                    "topic_description": topic_description,
                },
            )

        except Exception as e:
            logger.exception("Failed to get learning objectives: %s", str(e))
            return ToolResult(
                success=False,
                error=f"Failed to retrieve learning objectives: {str(e)}",
            )

    def _generate_topic_description(
        self,
        topic_name: str,
        topic_desc: str | None,
        objectives: list[dict[str, Any]],
    ) -> str:
        """Generate a teaching-focused topic description.

        Args:
            topic_name: Name of the topic.
            topic_desc: Optional topic description from database.
            objectives: List of learning objectives.

        Returns:
            Formatted topic description for teaching context.
        """
        parts = [f"# Topic: {topic_name}"]

        if topic_desc:
            parts.append(f"\n{topic_desc}")

        parts.append("\n## Learning Objectives")
        for i, obj in enumerate(objectives, 1):
            bloom = obj.get("bloom_level", "understand")
            parts.append(f"{i}. [{bloom.upper()}] {obj['objective']}")

        return "\n".join(parts)

    def _build_message(
        self,
        topic_name: str,
        objectives_count: int,
    ) -> str:
        """Build human-readable message for LLM context.

        Args:
            topic_name: Name of the topic.
            objectives_count: Number of learning objectives.

        Returns:
            Formatted message string.
        """
        message = f"[get_learning_objectives]\nLoaded curriculum for '{topic_name}':\n"
        message += f"- {objectives_count} learning objectives\n"
        message += "\nUse this curriculum context to guide your teaching."

        return message
