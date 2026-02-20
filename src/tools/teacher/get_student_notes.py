# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get student notes tool for teachers.

Returns notes about a specific student from various sources
including parents, teachers, AI agents, and counselors.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.tools.teacher.helpers import verify_teacher_has_student_access

logger = logging.getLogger(__name__)


class GetStudentNotesTool(BaseTool):
    """Tool to get notes about a student."""

    @property
    def name(self) -> str:
        return "get_student_notes"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_student_notes",
                "description": "Get notes about a specific student from parents, teachers, AI agents, and other sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_id": {
                            "type": "string",
                            "description": "The UUID of the student",
                        },
                        "source_type": {
                            "type": "string",
                            "enum": ["parent", "teacher", "counselor", "ai_agent", "companion", "system"],
                            "description": "Optional: Filter by note source",
                        },
                        "note_type": {
                            "type": "string",
                            "enum": ["daily_mood", "concern", "context", "achievement", "preference", "observation", "recommendation", "milestone", "restriction"],
                            "description": "Optional: Filter by note type",
                        },
                        "include_expired": {
                            "type": "boolean",
                            "description": "Include expired notes (default: false)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of notes to return (default: 20)",
                        },
                    },
                    "required": ["student_id"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_student_notes tool.

        Args:
            params: Tool parameters (student_id, source_type, note_type, etc.).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with student notes.
        """
        if not context.is_teacher:
            return ToolResult(
                success=False,
                error="This tool is only available for teachers.",
            )

        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available.",
            )

        student_id_str = params.get("student_id")
        if not student_id_str:
            return ToolResult(
                success=False,
                error="student_id is required.",
            )

        try:
            student_id = UUID(student_id_str)
        except ValueError:
            return ToolResult(
                success=False,
                error="Invalid student_id format.",
            )

        source_type = params.get("source_type")
        note_type = params.get("note_type")
        include_expired = params.get("include_expired", False)
        limit = params.get("limit", 20)
        teacher_id = context.user_id

        try:
            # Verify teacher has access to this student
            has_access = await verify_teacher_has_student_access(
                context.session, teacher_id, student_id
            )
            if not has_access:
                return ToolResult(
                    success=False,
                    error="You don't have access to this student.",
                )

            from src.infrastructure.database.models.tenant.student_note import StudentNote
            from src.infrastructure.database.models.tenant.user import User

            # Get student info
            student_query = (
                select(User.first_name, User.last_name)
                .where(User.id == str(student_id))
            )
            student_result = await context.session.execute(student_query)
            student_row = student_result.first()

            if not student_row:
                return ToolResult(
                    success=False,
                    error="Student not found.",
                )

            student_name = f"{student_row.first_name} {student_row.last_name}"

            # Build query for notes
            now = datetime.now()
            query = (
                select(StudentNote)
                .where(StudentNote.student_id == str(student_id))
                .where(StudentNote.valid_from <= now)
            )

            # Filter by visibility - teachers can see internal and parent_visible
            query = query.where(
                StudentNote.visibility.in_(["internal", "parent_visible", "all"])
            )

            if source_type:
                query = query.where(StudentNote.source_type == source_type)

            if note_type:
                query = query.where(StudentNote.note_type == note_type)

            if not include_expired:
                query = query.where(
                    (StudentNote.valid_until.is_(None)) |
                    (StudentNote.valid_until > now)
                )

            query = query.order_by(
                StudentNote.priority.desc(),
                StudentNote.created_at.desc(),
            ).limit(limit)

            result = await context.session.execute(query)
            note_rows = result.scalars().all()

            notes = []
            priority_notes = []
            for note in note_rows:
                note_data = {
                    "id": str(note.id),
                    "source_type": note.source_type,
                    "author_name": note.author_name,
                    "note_type": note.note_type,
                    "title": note.title,
                    "content": note.content,
                    "reported_emotion": note.reported_emotion,
                    "emotion_intensity": note.emotion_intensity,
                    "priority": note.priority,
                    "created_at": note.created_at.isoformat() if note.created_at else None,
                    "valid_until": note.valid_until.isoformat() if note.valid_until else None,
                    "is_active": note.is_active,
                    "is_high_priority": note.is_high_priority,
                }
                notes.append(note_data)

                if note.is_high_priority:
                    priority_notes.append(note_data)

            # Group by source type
            by_source = {}
            for note in notes:
                src = note["source_type"]
                if src not in by_source:
                    by_source[src] = []
                by_source[src].append(note)

            # Build message
            if not notes:
                message = f"No notes found for {student_name}."
            else:
                message = f"{student_name} - {len(notes)} notes:\n"
                if priority_notes:
                    message += f"- {len(priority_notes)} high-priority notes\n"
                for src, src_notes in by_source.items():
                    message += f"- {src}: {len(src_notes)} notes\n"

                # Show most recent high-priority note if any
                if priority_notes:
                    latest = priority_notes[0]
                    message += f"\nLatest priority: {latest['title'] or latest['content'][:50]}..."

            logger.info(
                "get_student_notes: teacher=%s, student=%s, notes=%d",
                teacher_id,
                student_id,
                len(notes),
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "student_id": str(student_id),
                    "student_name": student_name,
                    "notes": notes,
                    "count": len(notes),
                    "priority_count": len(priority_notes),
                    "by_source": {k: len(v) for k, v in by_source.items()},
                },
                passthrough_data={
                    "student_name": student_name,
                    "notes": notes[:5],
                    "priority_count": len(priority_notes),
                },
            )

        except Exception as e:
            logger.exception("get_student_notes failed")
            return ToolResult(
                success=False,
                error=f"Failed to get notes: {str(e)}",
            )
