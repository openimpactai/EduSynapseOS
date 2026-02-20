# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get Library Tool.

Lists user's H5P content library from Creatiq.
"""

import logging
import time
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult, UIElement, UIElementOption, UIElementType
from src.services.h5p import H5PClient

logger = logging.getLogger(__name__)


class GetLibraryTool(BaseTool):
    """Get user's H5P content library.

    Lists content from the user's Creatiq library with filtering options.

    Example usage by agent:
        - "Show my content"
        - "What H5P content do I have?"
        - "List my quizzes"
    """

    @property
    def name(self) -> str:
        return "get_library"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_library",
                "description": (
                    "Get the user's H5P content library from Creatiq. "
                    "Use this to show what content the user has already created."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library": {
                            "type": "string",
                            "description": "Filter by H5P library type (e.g., 'H5P.MultiChoice')",
                        },
                        "folder_id": {
                            "type": "string",
                            "description": "Filter by folder ID",
                        },
                        "search": {
                            "type": "string",
                            "description": "Search term for content titles",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of items to return (default 20)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Offset for pagination (default 0)",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the library fetch."""
        library = params.get("library")
        folder_id = params.get("folder_id")
        search = params.get("search")
        limit = params.get("limit", 20)
        offset = params.get("offset", 0)

        try:
            client = H5PClient(tenant_code=context.tenant_code)

            result = await client.list_content(
                library=library,
                folder_id=folder_id,
                search=search,
                limit=limit,
                offset=offset,
            )

            contents = result.get("contents", [])
            total = result.get("total", len(contents))

            if not contents:
                message = "No H5P content found in your library."
                if library:
                    message = f"No {library} content found."
                if search:
                    message = f"No content matching '{search}' found."

                return ToolResult(
                    success=True,
                    data={
                        "contents": [],
                        "total": 0,
                        "message": message,
                    },
                )

            # Format contents for display
            content_list = []
            for content in contents:
                content_list.append({
                    "id": content.get("id"),
                    "title": content.get("title"),
                    "library": content.get("library"),
                    "created_at": content.get("createdAt"),
                    "updated_at": content.get("updatedAt"),
                    "preview_url": content.get("previewUrl"),
                    "embed_url": content.get("embedUrl"),
                })

            # Build message
            message_lines = [f"Found {total} content item(s):"]
            for c in content_list[:5]:  # Show first 5 in message
                lib_short = c["library"].split(".")[-1] if c["library"] else "Unknown"
                message_lines.append(f"- {c['title']} ({lib_short}) - ID: {c['id']}")
            if total > 5:
                message_lines.append(f"... and {total - 5} more")

            # Build UI element for selection
            ui_options = []
            for c in content_list:
                lib_short = c["library"].split(".")[-1] if c["library"] else "Unknown"
                ui_options.append(
                    UIElementOption(
                        id=c["id"],
                        label=c["title"],
                        description=lib_short,
                        metadata={
                            "library": c["library"],
                            "preview_url": c["preview_url"],
                            "embed_url": c["embed_url"],
                        },
                    )
                )

            ui_element = UIElement(
                type=UIElementType.SINGLE_SELECT,
                id=f"h5p_library_selection_{int(time.time() * 1000)}",
                title="Select Content",
                options=ui_options,
                searchable=True,
                allow_text_input=False,
                placeholder="Choose content to view or edit...",
            )

            return ToolResult(
                success=True,
                data={
                    "contents": content_list,
                    "total": total,
                    "has_more": total > offset + limit,
                    "message": "\n".join(message_lines),
                },
                ui_element=ui_element,
            )

        except Exception as e:
            logger.exception("Error fetching library")
            return ToolResult(
                success=False,
                data={"message": f"Failed to fetch library: {e}"},
                error=str(e),
            )


class GetContentTool(BaseTool):
    """Get specific H5P content details.

    Retrieves full details of a specific H5P content item.

    Example usage by agent:
        - "Show details of this quiz"
        - "Get the content info"
        - "What's in this content?"
    """

    @property
    def name(self) -> str:
        return "get_content"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_content",
                "description": (
                    "Get detailed information about a specific H5P content item. "
                    "Use this when the user wants to view or edit existing content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_id": {
                            "type": "string",
                            "description": "The ID of the H5P content to retrieve",
                        },
                    },
                    "required": ["content_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the content fetch."""
        content_id = params.get("content_id", "")

        if not content_id:
            return ToolResult(
                success=False,
                data={"message": "Content ID is required"},
                error="Missing required parameter: content_id",
            )

        try:
            client = H5PClient(tenant_code=context.tenant_code)
            content = await client.get_content(content_id)

            if not content:
                return ToolResult(
                    success=False,
                    data={"message": f"Content not found: {content_id}"},
                    error="Content not found",
                )

            # Create UI element for preview iframe
            preview_url = content.get("previewUrl")
            if preview_url:
                ui_element = UIElement(
                    type=UIElementType.IFRAME,
                    id=f"h5p_content_{content_id}",
                    title=content.get("title", "H5P Content"),
                    metadata={
                        "url": preview_url,
                        "width": "100%",
                        "height": "500px",
                    },
                )
            else:
                ui_element = None

            lib_short = content.get("library", "").split(".")[-1]
            message = (
                f"Content: {content.get('title')}\n"
                f"Type: {lib_short}\n"
                f"Created: {content.get('createdAt', 'Unknown')}\n"
                f"Updated: {content.get('updatedAt', 'Unknown')}"
            )

            return ToolResult(
                success=True,
                data={
                    "id": content.get("id"),
                    "title": content.get("title"),
                    "library": content.get("library"),
                    "params": content.get("params", {}),
                    "created_at": content.get("createdAt"),
                    "updated_at": content.get("updatedAt"),
                    "preview_url": preview_url,
                    "embed_url": content.get("embedUrl"),
                    "message": message,
                },
                ui_element=ui_element,
                state_update={
                    "current_content": {
                        "content_id": content.get("id"),
                        "title": content.get("title"),
                        "library": content.get("library"),
                        "params": content.get("params", {}),
                    },
                },
            )

        except Exception as e:
            logger.exception("Error fetching content")
            return ToolResult(
                success=False,
                data={"message": f"Failed to fetch content: {e}"},
                error=str(e),
            )


class UpdateContentTool(BaseTool):
    """Update existing H5P content.

    Updates the params of an existing H5P content item.

    Example usage by agent:
        - "Update this quiz"
        - "Change the content"
        - "Modify the questions"
    """

    @property
    def name(self) -> str:
        return "update_content"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "update_content",
                "description": (
                    "Update existing H5P content with new parameters. "
                    "Use this when the user wants to modify their existing content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_id": {
                            "type": "string",
                            "description": "The ID of the H5P content to update",
                        },
                        "title": {
                            "type": "string",
                            "description": "New title for the content",
                        },
                        "params": {
                            "type": "object",
                            "description": "New H5P params for the content",
                        },
                    },
                    "required": ["content_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the content update."""
        content_id = params.get("content_id", "")
        title = params.get("title")
        h5p_params = params.get("params")

        if not content_id:
            return ToolResult(
                success=False,
                data={"message": "Content ID is required"},
                error="Missing required parameter: content_id",
            )

        if not title and not h5p_params:
            return ToolResult(
                success=False,
                data={"message": "At least title or params must be provided"},
                error="No update parameters provided",
            )

        try:
            client = H5PClient(tenant_code=context.tenant_code)

            result = await client.update_content(
                content_id=content_id,
                title=title,
                params=h5p_params,
            )

            logger.info(
                "Updated H5P content: id=%s, title=%s",
                content_id,
                title or "(unchanged)",
            )

            return ToolResult(
                success=True,
                data={
                    "content_id": content_id,
                    "title": result.get("title"),
                    "preview_url": result.get("previewUrl"),
                    "message": f"Successfully updated content '{result.get('title')}'.",
                },
                state_update={
                    "last_updated_content": {
                        "content_id": content_id,
                        "title": result.get("title"),
                    },
                },
            )

        except Exception as e:
            logger.exception("Error updating content")
            return ToolResult(
                success=False,
                data={"message": f"Failed to update content: {e}"},
                error=str(e),
            )
