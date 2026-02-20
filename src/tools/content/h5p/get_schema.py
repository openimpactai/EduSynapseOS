# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get H5P Schema Tool.

Returns detailed H5P schema for a specific content type including
semantics, required fields, and validation rules.

Schemas are loaded from config/h5p-schemas/ directory via H5PSchemaLoader,
providing a single source of truth for all content type schemas.
"""

import json
import logging
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.services.h5p.schema_loader import H5PSchemaLoader

logger = logging.getLogger(__name__)


class GetH5PSchemaTool(BaseTool):
    """Get detailed H5P schema for a content type.

    Returns the complete H5P semantic schema including:
    - Required and optional fields
    - Field types and validation rules
    - Default values
    - Example content structure

    Schemas are loaded from config/h5p-schemas/ directory.
    Used by generator agents to format content correctly.
    """

    def __init__(self) -> None:
        """Initialize the tool with schema loader."""
        self._schema_loader = H5PSchemaLoader()

    @property
    def name(self) -> str:
        return "get_h5p_schema"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_h5p_schema",
                "description": (
                    "Get detailed H5P schema for a specific content type. "
                    "Returns the semantics, required fields, validation rules, "
                    "and example content structure. Used by generators to format content correctly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Content type ID (e.g., 'multiple-choice', 'flashcards')",
                        },
                    },
                    "required": ["content_type"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool to get H5P schema."""
        content_type = params.get("content_type", "").lower()

        if not content_type:
            return ToolResult(
                success=False,
                data={"message": "content_type is required"},
                error="Missing required parameter: content_type",
            )

        # Get schema from loader (loads from config/h5p-schemas/)
        schema = self._schema_loader.get_tool_schema(content_type)

        if not schema:
            available = self._schema_loader.list_content_types()
            return ToolResult(
                success=False,
                data={
                    "message": f"Unknown content type: {content_type}",
                    "available_types": available,
                },
                error=f"Schema not found for content type: {content_type}",
            )

        # Build informative message
        ai_format = schema.get("ai_input_format", {})
        message = f"Schema for '{content_type}' ({schema.get('library')}):\n\n"
        message += f"AI Input Format:\n{json.dumps(ai_format, indent=2)}"

        return ToolResult(
            success=True,
            data={
                "content_type": content_type,
                "library": schema.get("library"),
                "name": schema.get("name"),
                "category": schema.get("category"),
                "description": schema.get("description"),
                "ai_input_format": ai_format,
                "h5p_params_format": schema.get("h5p_params_format"),
                "conversion_notes": schema.get("conversion_notes"),
                "requires_media": schema.get("requires_media", False),
                "ai_support": schema.get("ai_support", "full"),
                "message": message,
            },
            passthrough_data={
                "schema": schema,
                "content_type": content_type,
            },
        )
