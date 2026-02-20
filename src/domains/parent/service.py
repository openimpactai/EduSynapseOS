# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parent Service - manages parent interactions with their children's learning data.

This service provides:
- List children linked to a parent
- View child summary (progress, emotional state, alerts)
- Create/read notes for companion context
- View and acknowledge alerts

All operations respect the permissions defined in ParentStudentRelation.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.emotional import EmotionalStateService
from src.infrastructure.database.models.tenant.notification import Alert
from src.infrastructure.database.models.tenant.school import ParentStudentRelation
from src.infrastructure.database.models.tenant.student_note import StudentNote
from src.infrastructure.database.models.tenant.user import User

logger = logging.getLogger(__name__)


class ParentServiceError(Exception):
    """Exception raised for parent service operations."""

    def __init__(
        self,
        message: str,
        code: str = "parent_error",
        original_error: Exception | None = None,
    ):
        self.message = message
        self.code = code
        self.original_error = original_error
        super().__init__(self.message)


class PermissionDeniedError(ParentServiceError):
    """Raised when parent doesn't have required permission."""

    def __init__(self, permission: str):
        super().__init__(
            message=f"Permission denied: {permission}",
            code="permission_denied",
        )
        self.permission = permission


class ChildNotFoundError(ParentServiceError):
    """Raised when child is not found or not linked to parent."""

    def __init__(self, child_id: UUID):
        super().__init__(
            message=f"Child not found: {child_id}",
            code="child_not_found",
        )
        self.child_id = child_id


class ParentService:
    """Service for parent-related operations.

    Handles:
    - Listing children linked to a parent
    - Viewing child summaries (progress, emotional state, alerts)
    - Managing notes (create, read)
    - Viewing alerts

    All operations respect ParentStudentRelation permissions.

    Attributes:
        _db: Async database session.
        _emotional_service: Service for emotional state retrieval.

    Example:
        service = ParentService(db, emotional_service)
        children = await service.get_children(parent_id)
        summary = await service.get_child_summary(parent_id, child_id)
    """

    def __init__(
        self,
        db: AsyncSession,
        emotional_service: EmotionalStateService | None = None,
    ):
        """Initialize the parent service.

        Args:
            db: Async database session.
            emotional_service: Optional emotional state service.
        """
        self._db = db
        self._emotional_service = emotional_service or EmotionalStateService(db=db)

    # =========================================================================
    # Children Management
    # =========================================================================

    async def get_children(self, parent_id: UUID) -> list[dict[str, Any]]:
        """Get all children linked to a parent.

        Args:
            parent_id: Parent user ID.

        Returns:
            List of child information with permissions.
        """
        result = await self._db.execute(
            select(ParentStudentRelation)
            .options(selectinload(ParentStudentRelation.student))
            .where(ParentStudentRelation.parent_id == str(parent_id))
        )
        relations = result.scalars().all()

        children = []
        for rel in relations:
            if not rel.student:
                continue

            children.append({
                "id": rel.student.id,
                "name": rel.student.full_name,
                "first_name": rel.student.first_name,
                "grade": rel.student.extra_data.get("grade"),
                "avatar_url": rel.student.extra_data.get("avatar_url"),
                "permissions": {
                    "can_view_progress": rel.can_view_progress,
                    "can_view_conversations": rel.can_view_conversations,
                    "can_receive_notifications": rel.can_receive_notifications,
                    "can_chat_with_ai": rel.can_chat_with_ai,
                },
                "relationship_type": rel.relationship_type,
                "is_primary": rel.is_primary,
            })

        logger.info(
            "Retrieved %d children for parent %s",
            len(children),
            parent_id,
        )

        return children

    async def get_child_relation(
        self,
        parent_id: UUID,
        child_id: UUID,
    ) -> ParentStudentRelation:
        """Get parent-child relation with permission check.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.

        Returns:
            ParentStudentRelation instance.

        Raises:
            ChildNotFoundError: If child is not linked to parent.
        """
        result = await self._db.execute(
            select(ParentStudentRelation)
            .options(selectinload(ParentStudentRelation.student))
            .where(
                ParentStudentRelation.parent_id == str(parent_id),
                ParentStudentRelation.student_id == str(child_id),
            )
        )
        relation = result.scalar_one_or_none()

        if not relation:
            raise ChildNotFoundError(child_id)

        return relation

    async def check_permission(
        self,
        parent_id: UUID,
        child_id: UUID,
        permission: str,
    ) -> ParentStudentRelation:
        """Check if parent has a specific permission for child.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.
            permission: Permission name (view_progress, view_conversations, etc.)

        Returns:
            ParentStudentRelation if permission granted.

        Raises:
            ChildNotFoundError: If child is not linked to parent.
            PermissionDeniedError: If permission is not granted.
        """
        relation = await self.get_child_relation(parent_id, child_id)

        permission_map = {
            "view_progress": relation.can_view_progress,
            "view_conversations": relation.can_view_conversations,
            "receive_notifications": relation.can_receive_notifications,
            "chat_with_ai": relation.can_chat_with_ai,
        }

        if permission in permission_map and not permission_map[permission]:
            raise PermissionDeniedError(permission)

        return relation

    # =========================================================================
    # Child Summary
    # =========================================================================

    async def get_child_summary(
        self,
        parent_id: UUID,
        child_id: UUID,
    ) -> dict[str, Any]:
        """Get comprehensive summary for a child.

        Includes progress, emotional state, recent activity, and alerts.
        Requires 'view_progress' permission.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.

        Returns:
            Child summary dictionary.

        Raises:
            ChildNotFoundError: If child is not linked.
            PermissionDeniedError: If view_progress not allowed.
        """
        relation = await self.check_permission(parent_id, child_id, "view_progress")
        child = relation.student

        # Get emotional state
        emotional_data = await self._get_emotional_summary(child_id)

        # Get progress summary
        progress_data = await self._get_progress_summary(child_id)

        # Get recent activity
        activity_data = await self._get_activity_summary(child_id)

        # Get active alerts (if can receive notifications)
        alerts_data = []
        if relation.can_receive_notifications:
            alerts_data = await self._get_active_alerts(child_id, limit=5)

        # Get companion status
        companion_data = await self._get_companion_status(child_id)

        return {
            "child": {
                "id": child.id,
                "name": child.full_name,
                "first_name": child.first_name,
                "grade": child.extra_data.get("grade"),
            },
            "emotional_state": emotional_data,
            "progress": progress_data,
            "recent_activity": activity_data,
            "active_alerts": alerts_data,
            "companion_status": companion_data,
        }

    async def _get_emotional_summary(self, child_id: UUID) -> dict[str, Any]:
        """Get emotional state summary for child."""
        try:
            context = await self._emotional_service.get_current_state(
                student_id=child_id,
            )
            if context:
                return {
                    "current": context.current_state.value,
                    "intensity": context.intensity.value,
                    "trend": context.trend.direction if context.trend else "stable",
                    "confidence": context.confidence,
                }
        except Exception as e:
            logger.warning("Failed to get emotional state for %s: %s", child_id, e)

        return {
            "current": "unknown",
            "intensity": "unknown",
            "trend": "unknown",
        }

    async def _get_progress_summary(self, child_id: UUID) -> dict[str, Any]:
        """Get learning progress summary for child.

        Queries semantic memory for mastery and daily summaries for activity.
        """
        from datetime import timedelta
        from src.infrastructure.database.models.tenant.analytics import DailySummary
        from src.infrastructure.database.models.tenant.memory import SemanticMemory

        # Get overall mastery from semantic memory
        mastery_result = await self._db.execute(
            select(func.avg(SemanticMemory.mastery_level))
            .where(SemanticMemory.student_id == str(child_id))
            .where(SemanticMemory.entity_type == "topic")
        )
        avg_mastery = mastery_result.scalar() or 0.0
        overall_mastery_percent = round(float(avg_mastery) * 100)

        # Get weekly activity from daily summaries
        week_ago = datetime.now() - timedelta(days=7)
        weekly_result = await self._db.execute(
            select(
                func.sum(DailySummary.total_time_seconds).label("total_time"),
                func.count(DailySummary.id).label("days_active"),
            )
            .where(DailySummary.student_id == str(child_id))
            .where(DailySummary.summary_date >= week_ago.date())
        )
        weekly_row = weekly_result.first()

        weekly_activity_minutes = 0
        current_streak_days = 0
        if weekly_row:
            total_seconds = weekly_row.total_time or 0
            weekly_activity_minutes = total_seconds // 60
            current_streak_days = weekly_row.days_active or 0

        # Count topics mastered this week (mastery >= 0.8)
        mastered_result = await self._db.execute(
            select(func.count(SemanticMemory.id))
            .where(SemanticMemory.student_id == str(child_id))
            .where(SemanticMemory.entity_type == "topic")
            .where(SemanticMemory.mastery_level >= 0.8)
            .where(SemanticMemory.updated_at >= week_ago)
        )
        topics_mastered_this_week = mastered_result.scalar() or 0

        return {
            "weekly_activity_minutes": weekly_activity_minutes,
            "topics_mastered_this_week": topics_mastered_this_week,
            "current_streak_days": current_streak_days,
            "overall_mastery_percent": overall_mastery_percent,
        }

    async def _get_activity_summary(self, child_id: UUID) -> dict[str, Any]:
        """Get recent activity summary for child.

        Queries user for last login and practice sessions for activity data.
        """
        from datetime import timedelta
        from src.infrastructure.database.models.tenant.practice import PracticeSession

        # Get last login from user
        result = await self._db.execute(
            select(User).where(User.id == str(child_id))
        )
        user = result.scalar_one_or_none()

        last_login = None
        if user and user.last_login_at:
            last_login = user.last_login_at.isoformat()

        # Get last practice session
        last_practice_result = await self._db.execute(
            select(PracticeSession.started_at)
            .where(PracticeSession.student_id == str(child_id))
            .order_by(PracticeSession.started_at.desc())
            .limit(1)
        )
        last_practice_row = last_practice_result.first()
        last_practice = last_practice_row.started_at.isoformat() if last_practice_row else None

        # Count sessions this week
        week_ago = datetime.now() - timedelta(days=7)
        sessions_result = await self._db.execute(
            select(func.count(PracticeSession.id))
            .where(PracticeSession.student_id == str(child_id))
            .where(PracticeSession.started_at >= week_ago)
        )
        sessions_this_week = sessions_result.scalar() or 0

        return {
            "last_login": last_login,
            "last_practice": last_practice,
            "sessions_this_week": sessions_this_week,
        }

    async def _get_active_alerts(
        self,
        child_id: UUID,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get active alerts for child."""
        result = await self._db.execute(
            select(Alert)
            .where(
                Alert.student_id == str(child_id),
                Alert.status == "active",
            )
            .order_by(Alert.created_at.desc())
            .limit(limit)
        )
        alerts = result.scalars().all()

        return [
            {
                "id": alert.id,
                "type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
            }
            for alert in alerts
        ]

    async def _get_companion_status(self, child_id: UUID) -> dict[str, Any]:
        """Get companion interaction status for child."""
        from src.infrastructure.database.models.tenant.companion import CompanionSession

        result = await self._db.execute(
            select(CompanionSession)
            .where(CompanionSession.student_id == str(child_id))
            .order_by(CompanionSession.created_at.desc())
            .limit(1)
        )
        session = result.scalar_one_or_none()

        if session:
            return {
                "last_checkin": (
                    session.created_at.isoformat() if session.created_at else None
                ),
                "mood_at_checkin": session.emotional_state_start,
            }

        return {
            "last_checkin": None,
            "mood_at_checkin": None,
        }

    # =========================================================================
    # Notes Management
    # =========================================================================

    async def get_notes(
        self,
        parent_id: UUID,
        child_id: UUID,
        status: str = "active",
        note_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get notes for a child created by this parent.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.
            status: Filter by status (active, expired, all).
            note_type: Filter by note type (optional).
            limit: Maximum number of notes to return.

        Returns:
            List of note dictionaries.

        Raises:
            ChildNotFoundError: If child is not linked to parent.
        """
        # Verify parent-child relationship (no specific permission needed for notes)
        await self.get_child_relation(parent_id, child_id)

        query = (
            select(StudentNote)
            .where(
                StudentNote.student_id == str(child_id),
                StudentNote.source_type == "parent",
                StudentNote.author_id == str(parent_id),
            )
            .order_by(StudentNote.created_at.desc())
            .limit(limit)
        )

        # Apply status filter
        if status == "active":
            from src.utils.datetime import utc_now
            now = utc_now()
            query = query.where(
                StudentNote.valid_from <= now,
                (StudentNote.valid_until.is_(None)) | (StudentNote.valid_until > now),
            )
        elif status == "expired":
            from src.utils.datetime import utc_now
            now = utc_now()
            query = query.where(
                StudentNote.valid_until.isnot(None),
                StudentNote.valid_until <= now,
            )

        # Apply note type filter
        if note_type and note_type != "all":
            query = query.where(StudentNote.note_type == note_type)

        result = await self._db.execute(query)
        notes = result.scalars().all()

        return [
            {
                "id": note.id,
                "note_type": note.note_type,
                "title": note.title,
                "content": note.content,
                "reported_emotion": note.reported_emotion,
                "emotion_intensity": note.emotion_intensity,
                "valid_from": note.valid_from.isoformat() if note.valid_from else None,
                "valid_until": (
                    note.valid_until.isoformat() if note.valid_until else None
                ),
                "priority": note.priority,
                "ai_processed": note.ai_processed,
                "created_at": note.created_at.isoformat() if note.created_at else None,
                "is_active": note.is_active,
            }
            for note in notes
        ]

    async def create_note(
        self,
        parent_id: UUID,
        child_id: UUID,
        note_type: str,
        content: str,
        title: str | None = None,
        reported_emotion: str | None = None,
        emotion_intensity: str | None = None,
        valid_until: datetime | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Create a note for a child.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.
            note_type: Type of note (daily_mood, concern, context, etc.).
            content: Note content.
            title: Optional note title.
            reported_emotion: Reported emotional state.
            emotion_intensity: Intensity (low, moderate, high).
            valid_until: When note expires.
            priority: Priority (low, normal, high, urgent).

        Returns:
            Created note dictionary.

        Raises:
            ChildNotFoundError: If child is not linked to parent.
        """
        # Verify parent-child relationship
        relation = await self.get_child_relation(parent_id, child_id)

        # Get parent info for author_name
        parent_result = await self._db.execute(
            select(User).where(User.id == str(parent_id))
        )
        parent = parent_result.scalar_one_or_none()
        author_name = parent.full_name if parent else "Parent"

        # Create note
        note = StudentNote(
            student_id=str(child_id),
            source_type="parent",
            author_id=str(parent_id),
            author_name=author_name,
            note_type=note_type,
            title=title,
            content=content,
            reported_emotion=reported_emotion,
            emotion_intensity=emotion_intensity,
            valid_until=valid_until,
            priority=priority,
            visibility="internal",  # Parent notes are internal (AI uses them)
        )

        self._db.add(note)
        await self._db.commit()
        await self._db.refresh(note)

        logger.info(
            "Created note %s for child %s by parent %s",
            note.id,
            child_id,
            parent_id,
        )

        return {
            "id": note.id,
            "note_type": note.note_type,
            "title": note.title,
            "content": note.content,
            "reported_emotion": note.reported_emotion,
            "emotion_intensity": note.emotion_intensity,
            "valid_from": note.valid_from.isoformat() if note.valid_from else None,
            "valid_until": note.valid_until.isoformat() if note.valid_until else None,
            "priority": note.priority,
            "created_at": note.created_at.isoformat() if note.created_at else None,
        }

    async def update_note(
        self,
        parent_id: UUID,
        child_id: UUID,
        note_id: UUID,
        **updates,
    ) -> dict[str, Any]:
        """Update a note.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.
            note_id: Note ID to update.
            **updates: Fields to update.

        Returns:
            Updated note dictionary.

        Raises:
            ChildNotFoundError: If child is not linked.
            ParentServiceError: If note not found or not owned by parent.
        """
        # Verify parent-child relationship
        await self.get_child_relation(parent_id, child_id)

        # Get note
        result = await self._db.execute(
            select(StudentNote).where(
                StudentNote.id == str(note_id),
                StudentNote.student_id == str(child_id),
                StudentNote.author_id == str(parent_id),
            )
        )
        note = result.scalar_one_or_none()

        if not note:
            raise ParentServiceError(
                message="Note not found",
                code="note_not_found",
            )

        # Update allowed fields
        allowed_fields = {
            "title", "content", "reported_emotion", "emotion_intensity",
            "valid_until", "priority",
        }
        for field, value in updates.items():
            if field in allowed_fields and value is not None:
                setattr(note, field, value)

        # Reset AI processed flag since content may have changed
        if "content" in updates:
            note.ai_processed = False
            note.ai_processed_at = None

        await self._db.commit()
        await self._db.refresh(note)

        return {
            "id": note.id,
            "note_type": note.note_type,
            "title": note.title,
            "content": note.content,
            "priority": note.priority,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
        }

    async def delete_note(
        self,
        parent_id: UUID,
        child_id: UUID,
        note_id: UUID,
    ) -> bool:
        """Delete (expire) a note.

        Args:
            parent_id: Parent user ID.
            child_id: Child user ID.
            note_id: Note ID to delete.

        Returns:
            True if deleted.

        Raises:
            ChildNotFoundError: If child is not linked.
            ParentServiceError: If note not found.
        """
        # Verify parent-child relationship
        await self.get_child_relation(parent_id, child_id)

        # Get note
        result = await self._db.execute(
            select(StudentNote).where(
                StudentNote.id == str(note_id),
                StudentNote.student_id == str(child_id),
                StudentNote.author_id == str(parent_id),
            )
        )
        note = result.scalar_one_or_none()

        if not note:
            raise ParentServiceError(
                message="Note not found",
                code="note_not_found",
            )

        # Expire the note instead of hard delete
        note.expire()
        await self._db.commit()

        logger.info("Expired note %s for child %s", note_id, child_id)

        return True
