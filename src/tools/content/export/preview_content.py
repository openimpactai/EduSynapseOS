# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preview Content Tool.

Generates a preview URL for H5P content without saving to library.
"""

import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult, UIElement, UIElementType
from src.services.h5p import ConverterRegistry, H5PClient, H5PConversionError

logger = logging.getLogger(__name__)


class PreviewContentTool(BaseTool):
    """Generate preview URL for H5P content.

    Creates a temporary preview of the content without saving
    it to the user's library. Useful for review before final export.

    Example usage by agent:
        - "Show me a preview"
        - "Let me see what it looks like"
        - "Preview before saving"
    """

    def __init__(self):
        """Initialize tool with converter registry."""
        self._registry = ConverterRegistry()
        self._registry.load_default_converters()

    @property
    def name(self) -> str:
        return "preview_content"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "preview_content",
                "description": (
                    "Generate a preview URL for H5P content without saving it permanently. "
                    "Use this to show users what their content will look like before export."
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
                            "description": "Title for the preview",
                        },
                        "language": {
                            "type": "string",
                            "description": "Content language code (defaults to context language)",
                        },
                    },
                    "required": ["content_type", "ai_content"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the preview generation."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        title = params.get("title", "Preview")
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
            return ToolResult(
                success=False,
                data={"message": f"No converter found for content type: {content_type}"},
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

        # Generate preview via API
        try:
            client = H5PClient(
                tenant_code=context.tenant_code,
            )

            result = await client.create_preview(
                library=converter.library,
                params=h5p_params,
                title=title,
            )

            preview_url = result.get("previewUrl")
            preview_id = result.get("previewId")
            expires_at = result.get("expiresAt")

            logger.info(
                "Created H5P preview: id=%s, type=%s, expires=%s",
                preview_id,
                content_type,
                expires_at,
            )

            # Create UI element to display preview
            ui_element = UIElement(
                type=UIElementType.IFRAME,
                id=f"h5p_preview_{preview_id}",
                title=title,
                metadata={
                    "url": preview_url,
                    "width": "100%",
                    "height": "500px",
                    "preview_id": preview_id,
                    "expires_at": expires_at,
                },
            )

            return ToolResult(
                success=True,
                data={
                    "preview_id": preview_id,
                    "preview_url": preview_url,
                    "expires_at": expires_at,
                    "library": converter.library,
                    "content_type": content_type,
                    "message": (
                        f"Preview generated for {content_type.replace('_', ' ')}. "
                        f"The preview is available at the URL below."
                    ),
                },
                ui_element=ui_element,
                state_update={
                    "current_preview": {
                        "preview_id": preview_id,
                        "preview_url": preview_url,
                        "content_type": content_type,
                        "ai_content": ai_content,
                        "h5p_params": h5p_params,
                    },
                },
            )

        except Exception as e:
            logger.exception("H5P API error during preview generation")
            return ToolResult(
                success=False,
                data={"message": f"Failed to generate preview: {e}"},
                error=str(e),
            )
