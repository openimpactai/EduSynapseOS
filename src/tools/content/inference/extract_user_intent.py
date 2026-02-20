# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Extract User Intent Tool.

Analyzes user messages to extract content creation intent,
including content type preferences and media requests.
"""

import json
import logging
from typing import Any

from src.core.config import get_settings
from src.core.intelligence.llm import LLMClient
from src.core.tools import BaseTool, ToolContext, ToolResult
from src.services.h5p.schema_loader import H5PSchemaLoader

logger = logging.getLogger(__name__)


class ExtractUserIntentTool(BaseTool):
    """Extract content creation intent from user message.

    Uses LLM-based semantic analysis to understand what the user
    wants to create, including:
    - Content type (multiple-choice, flashcards, etc.)
    - Media/image requests
    - Media description if requested

    This tool is used in user-driven content creation mode to
    intelligently interpret user messages without requiring
    explicit structured input.

    Example messages this tool can interpret:
    - "Create a quiz about photosynthesis" -> multiple-choice/question-set
    - "I need flashcards for vocabulary" -> flashcards
    - "Make a timeline of World War 2 with images" -> timeline, wants_media=True
    - "Can you add a diagram showing the cell structure?" -> wants_media=True
    """

    def __init__(self) -> None:
        """Initialize the tool with schema loader."""
        self._schema_loader = H5PSchemaLoader()

    @property
    def name(self) -> str:
        return "extract_user_intent"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "extract_user_intent",
                "description": (
                    "Analyze user message to extract content creation intent. "
                    "Identifies requested content type and media/image requests."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type": "string",
                            "description": "The user's message to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "Language of the message (en, tr, etc.)",
                            "default": "en",
                        },
                    },
                    "required": ["user_message"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute intent extraction from user message."""
        user_message = params.get("user_message", "")
        language = params.get("language", context.language or "en")

        if not user_message:
            return ToolResult(
                success=False,
                data={"message": "User message is required"},
                error="Missing required parameter: user_message",
            )

        try:
            # Get available content types for context
            content_types_info = self._get_content_types_summary()

            # Extract intent using LLM
            inference_result = await self._extract_intent_with_llm(
                user_message=user_message,
                content_types_info=content_types_info,
                language=language,
            )

            logger.info(
                "Extracted intent: content_type=%s, wants_media=%s, confidence=%.2f",
                inference_result.get("content_type"),
                inference_result.get("wants_media"),
                inference_result.get("confidence", 0),
            )

            # Normalize content_type: LLM may return underscores or long forms
            raw_ct = inference_result.get("content_type")
            if raw_ct:
                # Strip H5P library prefix if present
                if raw_ct.startswith("H5P."):
                    raw_ct = raw_ct.split(" ")[0].replace("H5P.", "")
                raw_ct = raw_ct.replace("_", "-").lower()
                ct_aliases = {
                    "mark-the-words": "mark-words",
                    "markthewords": "mark-words",
                    "drag-the-words": "drag-words",
                    "dragtext": "drag-words",
                    "fill-the-blanks": "fill-blanks",
                    "fill-in-the-blanks": "fill-blanks",
                    "blanks": "fill-blanks",
                    "multiple-choice-question": "multiple-choice",
                    "multichoice": "multiple-choice",
                    "truefalse": "true-false",
                    "singlechoiceset": "single-choice-set",
                }
                raw_ct = ct_aliases.get(raw_ct, raw_ct)

            return ToolResult(
                success=True,
                data={
                    "content_type": raw_ct,
                    "wants_media": inference_result.get("wants_media", False),
                    "media_description": inference_result.get("media_description"),
                    "confidence": inference_result.get("confidence", 0.0),
                    "reasoning": inference_result.get("reasoning", ""),
                },
            )

        except Exception as e:
            logger.exception("Error extracting user intent")
            return ToolResult(
                success=False,
                data={"message": f"Failed to extract intent: {e}"},
                error=str(e),
            )

    def _get_content_types_summary(self) -> str:
        """Build a summary of available content types for LLM context."""
        content_types = self._schema_loader.get_all_content_types()

        summary_lines = []
        for ct_id, ct_info in content_types.items():
            name = ct_info.get("name", ct_id)
            description = ct_info.get("description", "")
            use_cases = ct_info.get("use_cases", [])
            requires_media = ct_info.get("requires_media", False)

            use_cases_str = ", ".join(use_cases) if use_cases else ""
            media_note = " (requires images)" if requires_media else ""

            summary_lines.append(
                f"- {ct_id}: {name}{media_note} - {description}. Use cases: {use_cases_str}"
            )

        return "\n".join(summary_lines)

    async def _extract_intent_with_llm(
        self,
        user_message: str,
        content_types_info: str,
        language: str,
    ) -> dict[str, Any]:
        """Extract user intent using LLM analysis.

        Args:
            user_message: The user's message to analyze.
            content_types_info: Summary of available content types.
            language: Language of the message.

        Returns:
            Dictionary with extracted intent.
        """
        settings = get_settings()
        llm_client = LLMClient(llm_settings=settings.llm)

        system_prompt = f"""You are an educational content creation assistant. Analyze the user's message to understand what they want to create.

Available H5P content types:
{content_types_info}

Your task:
1. Identify if the user is requesting a specific content type (or if one can be inferred)
2. Detect if the user wants images/media in their content
3. If media is requested, extract any description of what kind of media they want

Rules for content type detection:
- "quiz", "test", "sorular" -> multiple-choice or question-set
- "flashcard", "kartlar", "vocabulary", "kelime" -> flashcards
- "true/false", "doğru/yanlış" -> true-false
- "fill in", "boşluk doldur" -> fill-blanks
- "timeline", "zaman çizelgesi" -> timeline
- "memory game", "eşleştirme oyunu" -> memory-game
- "crossword", "bulmaca" -> crossword
- If user mentions "image", "picture", "görsel", "resim", "diagram" -> wants_media=True

Respond in JSON format:
{{
  "content_type": "content-type-id or null if not determinable",
  "wants_media": true/false,
  "media_description": "description of requested media or null",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your interpretation"
}}

If the user's message doesn't clearly indicate a content type, set content_type to null.
If unsure about media, set wants_media to false.
Be conservative - only set content_type if you're reasonably confident."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User message ({language}): {user_message}"},
        ]

        try:
            response = await llm_client.complete_with_messages(messages=messages)
            response_content = response.content

            # Try to parse JSON from response
            try:
                # Handle potential markdown code blocks
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    response_content = response_content[json_start:json_end].strip()
                elif "```" in response_content:
                    json_start = response_content.find("```") + 3
                    json_end = response_content.find("```", json_start)
                    response_content = response_content[json_start:json_end].strip()

                parsed = json.loads(response_content)

                # Validate content_type if provided
                content_type = parsed.get("content_type")
                if content_type:
                    valid_types = list(self._schema_loader.get_all_content_types().keys())
                    if content_type not in valid_types:
                        logger.warning(
                            "LLM returned invalid content type: %s", content_type
                        )
                        parsed["content_type"] = None
                        parsed["confidence"] = max(0, parsed.get("confidence", 0) - 0.3)

                return {
                    "content_type": parsed.get("content_type"),
                    "wants_media": parsed.get("wants_media", False),
                    "media_description": parsed.get("media_description"),
                    "confidence": min(1.0, max(0.0, parsed.get("confidence", 0.5))),
                    "reasoning": parsed.get("reasoning", ""),
                }

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON: %s", response_content)
                return {
                    "content_type": None,
                    "wants_media": False,
                    "media_description": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                }

        except Exception as e:
            logger.warning("LLM intent extraction failed: %s", e)
            return {
                "content_type": None,
                "wants_media": False,
                "media_description": None,
                "confidence": 0.0,
                "reasoning": f"LLM call failed: {e}",
            }
