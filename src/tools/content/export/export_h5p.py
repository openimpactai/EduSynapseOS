# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Export H5P Content Tool.

Converts AI-generated content to H5P format and creates content via Creatiq API.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.services.h5p import ConverterRegistry, H5PClient, H5PConversionError

logger = logging.getLogger(__name__)


class ExportH5PTool(BaseTool):
    """Export AI-generated content to H5P format.

    Converts content from AI input format to H5P params format,
    then creates the content via Creatiq API.

    Example usage by agent:
        - "Create the flashcards in H5P"
        - "Export this quiz"
        - "Generate the H5P content"
    """

    def __init__(self):
        """Initialize tool with converter registry."""
        self._registry = ConverterRegistry()
        self._registry.load_default_converters()

    @property
    def name(self) -> str:
        return "export_h5p"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "export_h5p",
                "description": (
                    "Export AI-generated content to H5P format. "
                    "Converts content and creates it via Creatiq API. "
                    "Use this after generating content to make it available to students."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "H5P content type (e.g., 'multiple_choice', 'flashcards')",
                        },
                        "ai_content": {
                            "type": "object",
                            "description": "AI-generated content in the AI input format",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the H5P content",
                        },
                        "folder_id": {
                            "type": "string",
                            "description": "Optional folder ID in Creatiq to save content",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for the content",
                        },
                        "language": {
                            "type": "string",
                            "description": "Content language code (defaults to context language)",
                        },
                    },
                    "required": ["content_type", "ai_content", "title"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the H5P export."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        title = params.get("title", "Untitled")
        folder_id = params.get("folder_id")
        tags = params.get("tags", [])
        language = params.get("language", context.language or "en")

        if not content_type:
            return ToolResult(
                success=False,
                data={"message": "Content type is required"},
                error="Missing required parameter: content_type",
            )

        if not ai_content:
            return ToolResult(
                success=False,
                data={"message": "AI content is required"},
                error="Missing required parameter: ai_content",
            )

        # Get converter for content type
        converter = self._registry.get(content_type)
        if not converter:
            available = list(self._registry._converters.keys())
            return ToolResult(
                success=False,
                data={
                    "message": f"No converter found for content type: {content_type}",
                    "available_types": available,
                },
                error=f"Unsupported content type: {content_type}",
            )

        try:
            # Convert AI content to H5P params
            h5p_params = converter.convert(ai_content, language)
        except H5PConversionError as e:
            logger.error("H5P conversion error: %s", e)
            return ToolResult(
                success=False,
                data={"message": f"Conversion error: {e}"},
                error=str(e),
            )
        except Exception as e:
            logger.exception("Unexpected conversion error")
            return ToolResult(
                success=False,
                data={"message": f"Unexpected conversion error: {e}"},
                error=str(e),
            )

        # Create H5P content via API
        try:
            client = H5PClient(
                tenant_code=context.tenant_code,
            )

            result = await client.create_content(
                library=converter.library,
                params=h5p_params,
                title=title,
                folder_id=folder_id,
                tags=tags,
            )

            content_id = result.get("id")
            preview_url = result.get("previewUrl")
            embed_url = result.get("embedUrl")

            logger.info(
                "Created H5P content: id=%s, type=%s, title=%s",
                content_id,
                content_type,
                title,
            )

            return ToolResult(
                success=True,
                data={
                    "content_id": content_id,
                    "title": title,
                    "library": converter.library,
                    "content_type": content_type,
                    "preview_url": preview_url,
                    "embed_url": embed_url,
                    "message": (
                        f"Successfully created {content_type.replace('_', ' ')} content: '{title}'. "
                        f"Content ID: {content_id}"
                    ),
                },
                state_update={
                    "last_exported_content": {
                        "content_id": content_id,
                        "content_type": content_type,
                        "title": title,
                        "preview_url": preview_url,
                    },
                },
            )

        except Exception as e:
            logger.exception("H5P API error during content creation")
            return ToolResult(
                success=False,
                data={"message": f"Failed to create H5P content: {e}"},
                error=str(e),
            )
