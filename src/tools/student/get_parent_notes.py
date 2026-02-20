# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get parent notes tool.

This tool retrieves active parent notes to provide context
about the student's day or any concerns the parent has shared.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, or_, select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.infrastructure.database.models.tenant import StudentNote


# Valid note types for filtering
VALID_NOTE_TYPES = frozenset({
    "all",
    "context",
    "concern",
    "celebration",
    "daily_mood",
})


class GetParentNotesTool(BaseTool):
    """Tool to get active parent notes for context.

    Retrieves notes from parents that provide context about
    the student's day, concerns, or celebrations.
    """

    @property
    def name(self) -> str:
        return "get_parent_notes"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_parent_notes",
                "description": (
                    "Get active parent notes for context. "
                    "Use this when:\n"
                    "- Greeting a student (check for daily context)\n"
                    "- Student seems upset (parent may have provided context)\n"
                    "- Need additional background\n"
                    "Parent notes provide important context about student's day."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note_type": {
                            "type": "string",
                            "enum": list(VALID_NOTE_TYPES),
                            "description": (
                                "Type of notes to retrieve:\n"
                                "- all: All note types\n"
                                "- context: Background context\n"
                                "- concern: Parent concerns\n"
                                "- celebration: Achievements to celebrate\n"
                                "- daily_mood: Parent-reported mood"
                            ),
                        },
                        "include_expired": {
                            "type": "boolean",
                            "description": (
                                "Include notes past their valid_until date "
                                "(default: false)"
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
        """Execute the get_parent_notes tool.

        Args:
            params: Tool parameters from LLM.
                - note_type: Type of notes to retrieve
                - include_expired: Whether to include expired notes
            context: Execution context.

        Returns:
            ToolResult with parent notes.
        """
        note_type = params.get("note_type", "all")
        include_expired = params.get("include_expired", False)

        if note_type not in VALID_NOTE_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid note_type: {note_type}. Valid: {', '.join(VALID_NOTE_TYPES)}",
            )

        now = datetime.now(timezone.utc)

        # Build query conditions
        conditions = [
            StudentNote.student_id == str(context.student_id),
            StudentNote.source_type == "parent",
        ]

        # Filter by note type if not "all"
        if note_type != "all":
            conditions.append(StudentNote.note_type == note_type)

        # Filter by validity if not including expired
        if not include_expired:
            # Note is valid if:
            # - valid_from <= now AND (valid_until is NULL OR valid_until > now)
            conditions.append(StudentNote.valid_from <= now)
            conditions.append(
                or_(
                    StudentNote.valid_until.is_(None),
                    StudentNote.valid_until > now,
                )
            )

        stmt = (
            select(StudentNote)
            .where(and_(*conditions))
            .order_by(StudentNote.created_at.desc())
            .limit(5)  # Max 5 notes
        )

        result = await context.session.execute(stmt)
        notes = result.scalars().all()

        # Format notes for LLM
        notes_data = [
            {
                "type": note.note_type,
                "title": note.title,
                "content": note.content[:200] if note.content else None,  # Truncate
                "reported_emotion": note.reported_emotion,
                "emotion_intensity": note.emotion_intensity,
                "priority": note.priority,
                "created_at": note.created_at.isoformat() if note.created_at else None,
                "is_high_priority": note.priority in ("high", "urgent"),
            }
            for note in notes
        ]

        # Check for any concerns or high priority notes
        has_concerns = any(n["type"] == "concern" for n in notes_data)
        has_high_priority = any(n.get("is_high_priority") for n in notes_data)

        # Build human-readable message
        if not notes_data:
            message = "No parent notes found."
        else:
            note_summaries = []
            for n in notes_data:
                summary = f"{n['type']}"
                if n.get("title"):
                    summary += f": {n['title']}"
                if n.get("is_high_priority"):
                    summary += " (HIGH PRIORITY)"
                note_summaries.append(summary)
            message = f"Found {len(notes_data)} parent notes: {'; '.join(note_summaries)}"
            if has_concerns:
                message += " [Has concerns - please be sensitive]"

        return ToolResult(
            success=True,
            data={
                "message": message,
                "notes": notes_data,
                "count": len(notes_data),
                "has_concerns": has_concerns,
                "has_high_priority": has_high_priority,
                "filters_applied": {
                    "note_type": note_type,
                    "include_expired": include_expired,
                },
            },
        )
