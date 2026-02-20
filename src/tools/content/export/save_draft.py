# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Save Draft Tool.

Saves AI-generated content as a draft for later editing or export.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.services.h5p import ContentDraft, ContentStorageService

logger = logging.getLogger(__name__)


class SaveDraftTool(BaseTool):
    """Save content as a draft for later editing.

    Stores AI-generated content in the database for later retrieval,
    editing, or export. Drafts persist across sessions.

    Example usage by agent:
        - "Save this as a draft"
        - "Store this for later"
        - "Keep this draft"
    """

    @property
    def name(self) -> str:
        return "save_draft"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "save_draft",
                "description": (
                    "Save AI-generated content as a draft for later editing or export. "
                    "Use this when the user wants to save their work without creating H5P content yet."
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
                            "description": "Title for the draft",
                        },
                        "draft_id": {
                            "type": "string",
                            "description": "Optional draft ID to update existing draft",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for organizing drafts",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes about the draft",
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
        """Execute the draft save."""
        content_type = params.get("content_type", "")
        ai_content = params.get("ai_content", {})
        title = params.get("title", "Untitled Draft")
        draft_id = params.get("draft_id")
        tags = params.get("tags", [])
        notes = params.get("notes", "")

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

        try:
            storage = ContentStorageService(tenant_code=context.tenant_code)

            # Check if updating existing draft
            if draft_id:
                existing = await storage.get_draft(draft_id)
                if not existing:
                    return ToolResult(
                        success=False,
                        data={"message": f"Draft not found: {draft_id}"},
                        error="Draft not found",
                    )

                # Update existing draft
                draft = ContentDraft(
                    id=draft_id,
                    tenant_code=context.tenant_code,
                    user_id=str(context.user_id),
                    content_type=content_type,
                    title=title,
                    ai_content=ai_content,
                    tags=tags,
                    notes=notes,
                    created_at=existing.created_at,
                    updated_at=datetime.now(timezone.utc),
                    version=existing.version + 1,
                )
                is_new = False
            else:
                # Create new draft
                draft_id = str(uuid.uuid4())
                draft = ContentDraft(
                    id=draft_id,
                    tenant_code=context.tenant_code,
                    user_id=str(context.user_id),
                    content_type=content_type,
                    title=title,
                    ai_content=ai_content,
                    tags=tags,
                    notes=notes,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    version=1,
                )
                is_new = True

            # Save draft
            await storage.save_draft(draft)

            action = "Created" if is_new else "Updated"
            logger.info(
                "%s draft: id=%s, type=%s, title=%s",
                action,
                draft_id,
                content_type,
                title,
            )

            return ToolResult(
                success=True,
                data={
                    "draft_id": draft_id,
                    "title": title,
                    "content_type": content_type,
                    "version": draft.version,
                    "is_new": is_new,
                    "message": (
                        f"{action} draft '{title}' (ID: {draft_id}). "
                        f"You can continue editing or export it later."
                    ),
                },
                state_update={
                    "current_draft": {
                        "draft_id": draft_id,
                        "title": title,
                        "content_type": content_type,
                        "version": draft.version,
                    },
                },
            )

        except Exception as e:
            logger.exception("Error saving draft")
            return ToolResult(
                success=False,
                data={"message": f"Failed to save draft: {e}"},
                error=str(e),
            )


class LoadDraftTool(BaseTool):
    """Load a saved draft for editing.

    Retrieves a previously saved draft by ID.

    Example usage by agent:
        - "Load my draft"
        - "Open the saved content"
        - "Get my previous work"
    """

    @property
    def name(self) -> str:
        return "load_draft"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "load_draft",
                "description": (
                    "Load a saved draft for editing or export. "
                    "Use this when the user wants to continue working on a previous draft."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "The ID of the draft to load",
                        },
                    },
                    "required": ["draft_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the draft load."""
        draft_id = params.get("draft_id", "")

        if not draft_id:
            return ToolResult(
                success=False,
                data={"message": "Draft ID is required"},
                error="Missing required parameter: draft_id",
            )

        try:
            storage = ContentStorageService(tenant_code=context.tenant_code)
            draft = await storage.get_draft(draft_id)

            if not draft:
                return ToolResult(
                    success=False,
                    data={"message": f"Draft not found: {draft_id}"},
                    error="Draft not found",
                )

            # Verify ownership
            if draft.user_id != str(context.user_id):
                return ToolResult(
                    success=False,
                    data={"message": "You don't have permission to access this draft"},
                    error="Permission denied",
                )

            logger.info(
                "Loaded draft: id=%s, type=%s, title=%s",
                draft_id,
                draft.content_type,
                draft.title,
            )

            return ToolResult(
                success=True,
                data={
                    "draft_id": draft.id,
                    "title": draft.title,
                    "content_type": draft.content_type,
                    "ai_content": draft.ai_content,
                    "tags": draft.tags,
                    "notes": draft.notes,
                    "version": draft.version,
                    "created_at": draft.created_at.isoformat(),
                    "updated_at": draft.updated_at.isoformat(),
                    "message": (
                        f"Loaded draft '{draft.title}' ({draft.content_type}). "
                        f"Version {draft.version}, last updated {draft.updated_at.strftime('%Y-%m-%d %H:%M')}."
                    ),
                },
                state_update={
                    "current_draft": {
                        "draft_id": draft.id,
                        "title": draft.title,
                        "content_type": draft.content_type,
                        "ai_content": draft.ai_content,
                        "version": draft.version,
                    },
                },
            )

        except Exception as e:
            logger.exception("Error loading draft")
            return ToolResult(
                success=False,
                data={"message": f"Failed to load draft: {e}"},
                error=str(e),
            )


class ListDraftsTool(BaseTool):
    """List user's saved drafts.

    Returns a list of drafts with filtering and sorting options.

    Example usage by agent:
        - "Show my drafts"
        - "What drafts do I have?"
        - "List saved content"
    """

    @property
    def name(self) -> str:
        return "list_drafts"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_drafts",
                "description": (
                    "List user's saved drafts with optional filtering by content type or tags. "
                    "Use this to show what drafts the user has saved."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "description": "Filter by content type",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of drafts to return (default 20)",
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
        """Execute the draft listing."""
        content_type = params.get("content_type")
        tag = params.get("tag")
        limit = params.get("limit", 20)

        try:
            storage = ContentStorageService(tenant_code=context.tenant_code)
            drafts = await storage.list_drafts(
                user_id=str(context.user_id),
                content_type=content_type,
                tag=tag,
                limit=limit,
            )

            if not drafts:
                message = "No drafts found."
                if content_type:
                    message = f"No {content_type.replace('_', ' ')} drafts found."
                if tag:
                    message = f"No drafts found with tag '{tag}'."

                return ToolResult(
                    success=True,
                    data={
                        "drafts": [],
                        "count": 0,
                        "message": message,
                    },
                )

            # Format drafts for display
            draft_list = []
            for draft in drafts:
                draft_list.append({
                    "id": draft.id,
                    "title": draft.title,
                    "content_type": draft.content_type,
                    "tags": draft.tags,
                    "version": draft.version,
                    "updated_at": draft.updated_at.isoformat(),
                })

            message_lines = [f"Found {len(drafts)} draft(s):"]
            for d in draft_list[:5]:  # Show first 5 in message
                message_lines.append(
                    f"- {d['title']} ({d['content_type']}) - ID: {d['id'][:8]}..."
                )
            if len(drafts) > 5:
                message_lines.append(f"... and {len(drafts) - 5} more")

            return ToolResult(
                success=True,
                data={
                    "drafts": draft_list,
                    "count": len(drafts),
                    "message": "\n".join(message_lines),
                },
            )

        except Exception as e:
            logger.exception("Error listing drafts")
            return ToolResult(
                success=False,
                data={"message": f"Failed to list drafts: {e}"},
                error=str(e),
            )


class DeleteDraftTool(BaseTool):
    """Delete a saved draft.

    Permanently removes a draft from storage.

    Example usage by agent:
        - "Delete this draft"
        - "Remove the draft"
        - "Discard my saved content"
    """

    @property
    def name(self) -> str:
        return "delete_draft"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "delete_draft",
                "description": (
                    "Delete a saved draft permanently. "
                    "Use this when the user wants to remove a draft they no longer need."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "The ID of the draft to delete",
                        },
                    },
                    "required": ["draft_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the draft deletion."""
        draft_id = params.get("draft_id", "")

        if not draft_id:
            return ToolResult(
                success=False,
                data={"message": "Draft ID is required"},
                error="Missing required parameter: draft_id",
            )

        try:
            storage = ContentStorageService(tenant_code=context.tenant_code)

            # Check draft exists and user owns it
            draft = await storage.get_draft(draft_id)
            if not draft:
                return ToolResult(
                    success=False,
                    data={"message": f"Draft not found: {draft_id}"},
                    error="Draft not found",
                )

            if draft.user_id != str(context.user_id):
                return ToolResult(
                    success=False,
                    data={"message": "You don't have permission to delete this draft"},
                    error="Permission denied",
                )

            # Delete the draft
            await storage.delete_draft(draft_id)

            logger.info(
                "Deleted draft: id=%s, type=%s, title=%s",
                draft_id,
                draft.content_type,
                draft.title,
            )

            return ToolResult(
                success=True,
                data={
                    "draft_id": draft_id,
                    "title": draft.title,
                    "message": f"Draft '{draft.title}' has been deleted.",
                },
                state_update={
                    "deleted_draft_id": draft_id,
                },
            )

        except Exception as e:
            logger.exception("Error deleting draft")
            return ToolResult(
                success=False,
                data={"message": f"Failed to delete draft: {e}"},
                error=str(e),
            )
