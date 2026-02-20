# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to Vocabulary Generator Tool.

Delegates vocabulary content generation to the Vocabulary Generator agent.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class HandoffToVocabularyGeneratorTool(BaseTool):
    """Handoff tool for delegating to Vocabulary Generator agent.

    Supports:
    - Flashcards (H5P.Flashcards)
    - Dialog Cards (H5P.Dialogcards)
    - Crossword (H5P.Crossword)
    - Word Search (H5P.FindTheWords)
    """

    CONTENT_TYPE_TO_H5P = {
        "flashcards": "H5P.Flashcards 1.7",
        "dialog_cards": "H5P.Dialogcards 1.9",
        "crossword": "H5P.Crossword 0.5",
        "word_search": "H5P.FindTheWords 1.5",
    }

    @property
    def name(self) -> str:
        return "handoff_to_vocabulary_generator"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_vocabulary_generator",
                "description": (
                    "Delegate vocabulary content generation to the Vocabulary Generator agent. "
                    "Supports flashcards, dialog cards, crossword, and word search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Type of vocabulary content to generate",
                            "enum": [
                                "flashcards",
                                "dialog_cards",
                                "crossword",
                                "word_search",
                            ],
                        },
                        "topic": {
                            "type": "string",
                            "description": "The vocabulary topic or domain",
                        },
                        "grade_level": {
                            "type": "integer",
                            "description": "Target grade level (1-12)",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of vocabulary items to generate",
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code for content generation",
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Whether to include image prompts for media generation",
                        },
                        "include_examples": {
                            "type": "boolean",
                            "description": "Whether to include usage examples",
                        },
                        "target_language": {
                            "type": "string",
                            "description": "For language learning, the target language being learned",
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
        """Execute handoff to Vocabulary Generator."""
        content_type = params.get("content_type", "flashcards")
        topic = params.get("topic", "")
        grade_level = params.get("grade_level", context.grade_level or 5)
        count = params.get("count", 10)
        language = params.get("language", context.language or "en")
        include_images = params.get("include_images", False)
        include_examples = params.get("include_examples", True)
        target_language = params.get("target_language")
        additional_instructions = params.get("additional_instructions")

        if not topic:
            return ToolResult(
                success=False,
                data={"message": "Topic is required"},
                error="Missing required parameter: topic",
            )

        # Get H5P library
        h5p_library = self.CONTENT_TYPE_TO_H5P.get(content_type, "H5P.Flashcards 1.7")

        # Build handoff context
        handoff_context = {
            "source_agent": "content_creation_orchestrator",
            "target_agent": "vocabulary_generator",
            "task_type": "generate",
            "content_type": content_type,
            "h5p_library": h5p_library,
            "topic": topic,
            "grade_level": grade_level,
            "count": count,
            "language": language,
            "include_images": include_images,
            "include_examples": include_examples,
            "target_language": target_language,
            "additional_instructions": additional_instructions,
            "tenant_code": context.tenant_code,
            "user_id": str(context.user_id),
        }

        # Build generation prompt
        generation_prompt = self._build_generation_prompt(handoff_context)

        logger.info(
            "Handoff to vocabulary_generator: type=%s, topic=%s, count=%d",
            content_type,
            topic,
            count,
        )

        return ToolResult(
            success=True,
            data={
                "handoff": True,
                "target_agent": "vocabulary_generator",
                "handoff_context": handoff_context,
                "generation_prompt": generation_prompt,
                "message": (
                    f"Delegating {content_type.replace('_', ' ')} generation to Vocabulary Generator. "
                    f"Creating {count} items about '{topic}' for grade {grade_level}."
                ),
            },
            state_update={
                "pending_handoff": "vocabulary_generator",
                "handoff_context": handoff_context,
            },
            stop_chaining=True,
        )

    def _build_generation_prompt(self, ctx: dict[str, Any]) -> str:
        """Build the generation prompt for the vocabulary generator agent."""
        content_type = ctx["content_type"]
        topic = ctx["topic"]
        grade_level = ctx["grade_level"]
        count = ctx["count"]
        language = ctx["language"]
        include_images = ctx.get("include_images", False)
        include_examples = ctx.get("include_examples", True)
        target_language = ctx.get("target_language")
        additional = ctx.get("additional_instructions", "")

        prompt = f"""Generate {count} {content_type.replace('_', ' ')} items about "{topic}".

**Requirements:**
- Grade Level: {grade_level}
- Language: {"Turkish" if language == "tr" else "English"}
- Include Examples: {"Yes" if include_examples else "No"}
- Include Image Prompts: {"Yes" if include_images else "No"}
"""

        if target_language:
            prompt += f"- Target Language (for language learning): {target_language}\n"

        if additional:
            prompt += f"\n**Additional Instructions:**\n{additional}\n"

        prompt += f"""
**Output Format:**
Return JSON in the standard AI input format for {content_type}.
"""

        return prompt
