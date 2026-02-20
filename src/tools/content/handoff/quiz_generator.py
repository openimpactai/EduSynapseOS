# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to Quiz Generator Tool.

Delegates quiz content generation to the Quiz Generator agent.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class HandoffToQuizGeneratorTool(BaseTool):
    """Handoff tool for delegating to Quiz Generator agent.

    Supports:
    - Multiple Choice (H5P.MultiChoice)
    - True/False (H5P.TrueFalse)
    - Fill in the Blanks (H5P.Blanks)
    - Drag the Words (H5P.DragText)
    - Mark the Words (H5P.MarkTheWords)
    - Essay (H5P.Essay)
    - Question Set (H5P.QuestionSet)
    """

    CONTENT_TYPE_TO_H5P = {
        "multiple_choice": "H5P.MultiChoice 1.16",
        "true_false": "H5P.TrueFalse 1.8",
        "fill_blanks": "H5P.Blanks 1.14",
        "drag_words": "H5P.DragText 1.10",
        "mark_words": "H5P.MarkTheWords 1.11",
        "essay": "H5P.Essay 1.5",
        "question_set": "H5P.QuestionSet 1.20",
    }

    @property
    def name(self) -> str:
        return "handoff_to_quiz_generator"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_quiz_generator",
                "description": (
                    "Delegate quiz content generation to the Quiz Generator agent. "
                    "Supports multiple choice, true/false, fill-blanks, drag words, "
                    "mark words, essay, and question set content types."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Type of quiz content to generate",
                            "enum": [
                                "multiple_choice",
                                "true_false",
                                "fill_blanks",
                                "drag_words",
                                "mark_words",
                                "essay",
                                "question_set",
                            ],
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic to generate quiz content about",
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12)",
                        },
                        "difficulty": {
                            "type": "string",
                            "description": "Difficulty level",
                            "enum": ["easy", "medium", "hard"],
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of questions to generate",
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code for content generation",
                        },
                        "learning_objective": {
                            "type": "string",
                            "description": "Specific learning objective to target",
                        },
                        "bloom_level": {
                            "type": "string",
                            "description": "Target Bloom's taxonomy level",
                            "enum": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
                        },
                        "additional_instructions": {
                            "type": "string",
                            "description": "Extra instructions for the generator",
                        },
                    },
                    "required": ["content_type", "topic"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute handoff to Quiz Generator.

        This tool creates a handoff context and delegates to the Quiz Generator agent.
        The actual generation happens in the workflow layer.
        """
        content_type = params.get("content_type", "multiple_choice")
        topic = params.get("topic", "")
        grade_level = params.get("grade_level", context.grade_level or 5)
        difficulty = params.get("difficulty", "medium")
        count = params.get("count", 5)
        language = params.get("language", context.language or "en")
        learning_objective = params.get("learning_objective")
        bloom_level = params.get("bloom_level")
        additional_instructions = params.get("additional_instructions")

        if not topic:
            return ToolResult(
                success=False,
                data={"message": "Topic is required"},
                error="Missing required parameter: topic",
            )

        # Get H5P library
        h5p_library = self.CONTENT_TYPE_TO_H5P.get(content_type, "H5P.MultiChoice 1.16")

        # Build handoff context
        handoff_context = {
            "source_agent": "content_creation_orchestrator",
            "target_agent": "quiz_generator",
            "task_type": "generate",
            "content_type": content_type,
            "h5p_library": h5p_library,
            "topic": topic,
            "grade_level": grade_level,
            "difficulty": difficulty,
            "count": count,
            "language": language,
            "learning_objective": learning_objective,
            "bloom_level": bloom_level,
            "additional_instructions": additional_instructions,
            "tenant_code": context.tenant_code,
            "user_id": str(context.user_id),
        }

        # Build generation prompt based on content type
        generation_prompt = self._build_generation_prompt(handoff_context)

        logger.info(
            "Handoff to quiz_generator: type=%s, topic=%s, count=%d",
            content_type,
            topic,
            count,
        )

        # Return handoff action for workflow to process
        return ToolResult(
            success=True,
            data={
                "handoff": True,
                "target_agent": "quiz_generator",
                "handoff_context": handoff_context,
                "generation_prompt": generation_prompt,
                "message": (
                    f"Delegating {content_type.replace('_', ' ')} generation to Quiz Generator. "
                    f"Creating {count} questions about '{topic}' for grade {grade_level}."
                ),
            },
            state_update={
                "pending_handoff": "quiz_generator",
                "handoff_context": handoff_context,
            },
            stop_chaining=True,  # Stop tool chaining to allow handoff
        )

    def _build_generation_prompt(self, ctx: dict[str, Any]) -> str:
        """Build the generation prompt for the quiz generator agent."""
        content_type = ctx["content_type"]
        topic = ctx["topic"]
        grade_level = ctx["grade_level"]
        difficulty = ctx["difficulty"]
        count = ctx["count"]
        language = ctx["language"]
        learning_objective = ctx.get("learning_objective", "")
        bloom_level = ctx.get("bloom_level", "")
        additional = ctx.get("additional_instructions", "")

        prompt = f"""Generate {count} {content_type.replace('_', ' ')} questions about "{topic}".

**Requirements:**
- Grade Level: {grade_level}
- Difficulty: {difficulty}
- Language: {"Turkish" if language == "tr" else "English"}
"""

        if learning_objective:
            prompt += f"- Learning Objective: {learning_objective}\n"

        if bloom_level:
            prompt += f"- Bloom's Level: {bloom_level}\n"

        if additional:
            prompt += f"\n**Additional Instructions:**\n{additional}\n"

        prompt += f"""
**Output Format:**
Return JSON in the standard AI input format for {content_type}.
"""

        return prompt
