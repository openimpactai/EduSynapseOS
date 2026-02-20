# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parent API endpoints.

This module provides endpoints for parents to:
- View their linked children
- Get child summaries (progress, emotional state, alerts)
- Create and manage notes for companion context
- View alerts (Phase 2)
- Chat with AI about their child (Phase 3)

Example:
    GET /api/v1/parent/children
    GET /api/v1/parent/children/{child_id}/summary
    POST /api/v1/parent/children/{child_id}/notes
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_tenant_db, require_auth, require_tenant
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.emotional import EmotionalStateService
from src.domains.parent import ParentService
from src.domains.parent.service import (
    ChildNotFoundError,
    ParentServiceError,
    PermissionDeniedError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class ChildPermissions(BaseModel):
    """Child permissions for parent."""

    can_view_progress: bool
    can_view_conversations: bool
    can_receive_notifications: bool
    can_chat_with_ai: bool


class ChildInfo(BaseModel):
    """Basic child information."""

    id: str
    name: str
    first_name: str | None = None
    grade: str | None = None
    avatar_url: str | None = None
    permissions: ChildPermissions
    relationship_type: str
    is_primary: bool


class ChildrenResponse(BaseModel):
    """Response for children list."""

    children: list[ChildInfo]


class EmotionalState(BaseModel):
    """Emotional state summary."""

    current: str
    intensity: str
    trend: str
    confidence: float | None = None


class ProgressSummary(BaseModel):
    """Progress summary."""

    weekly_activity_minutes: int
    topics_mastered_this_week: int
    current_streak_days: int
    overall_mastery_percent: int


class ActivitySummary(BaseModel):
    """Activity summary."""

    last_login: str | None = None
    last_practice: str | None = None
    sessions_this_week: int


class AlertInfo(BaseModel):
    """Alert information."""

    id: str
    type: str
    severity: str
    title: str
    message: str | None = None
    created_at: str | None = None
    acknowledged_at: str | None = None


class CompanionStatus(BaseModel):
    """Companion status."""

    last_checkin: str | None = None
    mood_at_checkin: str | None = None


class ChildSummary(BaseModel):
    """Child summary response."""

    child: ChildInfo
    emotional_state: EmotionalState
    progress: ProgressSummary
    recent_activity: ActivitySummary
    active_alerts: list[AlertInfo]
    companion_status: CompanionStatus


class NoteInfo(BaseModel):
    """Note information."""

    id: str
    note_type: str
    title: str | None = None
    content: str
    reported_emotion: str | None = None
    emotion_intensity: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    priority: str
    ai_processed: bool
    created_at: str | None = None
    is_active: bool


class NotesResponse(BaseModel):
    """Response for notes list."""

    notes: list[NoteInfo]


class CreateNoteRequest(BaseModel):
    """Request to create a note."""

    note_type: str = Field(
        description="Note type: daily_mood, concern, context, preference, restriction",
    )
    content: str = Field(
        min_length=1,
        max_length=2000,
        description="Note content",
    )
    title: str | None = Field(
        default=None,
        max_length=255,
        description="Optional note title",
    )
    reported_emotion: str | None = Field(
        default=None,
        description="Reported emotional state",
    )
    emotion_intensity: str | None = Field(
        default=None,
        description="Intensity: low, moderate, high",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="When note expires",
    )
    priority: str = Field(
        default="normal",
        description="Priority: low, normal, high, urgent",
    )


class CreateNoteResponse(BaseModel):
    """Response for note creation."""

    success: bool
    note: NoteInfo


class UpdateNoteRequest(BaseModel):
    """Request to update a note."""

    title: str | None = None
    content: str | None = None
    reported_emotion: str | None = None
    emotion_intensity: str | None = None
    valid_until: datetime | None = None
    priority: str | None = None


class UpdateNoteResponse(BaseModel):
    """Response for note update."""

    success: bool
    note: dict


class DeleteNoteResponse(BaseModel):
    """Response for note deletion."""

    success: bool
    message: str


# =============================================================================
# Helper Functions
# =============================================================================


def _get_parent_service(db: AsyncSession) -> ParentService:
    """Get parent service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured ParentService instance.
    """
    emotional_service = EmotionalStateService(db=db)
    return ParentService(db=db, emotional_service=emotional_service)


def _require_parent_user(user: CurrentUser) -> None:
    """Verify user is a parent.

    Args:
        user: Current authenticated user.

    Raises:
        HTTPException: If user is not a parent.
    """
    if user.user_type != "parent":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is for parents only",
        )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/children", response_model=ChildrenResponse)
async def get_children(
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Get all children linked to the parent.

    Returns list of children with their basic info and permissions.

    Args:
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of children with permissions.
    """
    _require_parent_user(user)

    service = _get_parent_service(db)

    try:
        children = await service.get_children(user.id)

        return ChildrenResponse(
            children=[
                ChildInfo(
                    id=c["id"],
                    name=c["name"],
                    first_name=c["first_name"],
                    grade=c["grade"],
                    avatar_url=c["avatar_url"],
                    permissions=ChildPermissions(**c["permissions"]),
                    relationship_type=c["relationship_type"],
                    is_primary=c["is_primary"],
                )
                for c in children
            ]
        )
    except Exception as e:
        logger.exception("Failed to get children: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get children list",
        )


@router.get("/children/{child_id}/summary")
async def get_child_summary(
    child_id: UUID,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Get comprehensive summary for a child.

    Includes progress, emotional state, recent activity, and alerts.
    Requires 'view_progress' permission.

    Args:
        child_id: Child user ID.
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Child summary with all available data.
    """
    _require_parent_user(user)

    service = _get_parent_service(db)

    try:
        summary = await service.get_child_summary(user.id, child_id)
        return summary
    except ChildNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found or you don't have access",
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission to access this: {e.permission}",
        )
    except Exception as e:
        logger.exception("Failed to get child summary: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get summary",
        )


@router.get("/children/{child_id}/notes", response_model=NotesResponse)
async def get_notes(
    child_id: UUID,
    status_filter: str = Query(
        default="active",
        alias="status",
        description="Filter: active, expired, all",
    ),
    note_type: str | None = Query(
        default=None,
        description="Filter by note type",
    ),
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Get notes created by parent for a child.

    Args:
        child_id: Child user ID.
        status_filter: Filter by status (active, expired, all).
        note_type: Filter by note type.
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        List of notes.
    """
    _require_parent_user(user)

    service = _get_parent_service(db)

    try:
        notes = await service.get_notes(
            parent_id=user.id,
            child_id=child_id,
            status=status_filter,
            note_type=note_type,
        )

        return NotesResponse(
            notes=[
                NoteInfo(
                    id=n["id"],
                    note_type=n["note_type"],
                    title=n["title"],
                    content=n["content"],
                    reported_emotion=n["reported_emotion"],
                    emotion_intensity=n["emotion_intensity"],
                    valid_from=n["valid_from"],
                    valid_until=n["valid_until"],
                    priority=n["priority"],
                    ai_processed=n["ai_processed"],
                    created_at=n["created_at"],
                    is_active=n["is_active"],
                )
                for n in notes
            ]
        )
    except ChildNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found",
        )
    except Exception as e:
        logger.exception("Failed to get notes: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get notes",
        )


@router.post("/children/{child_id}/notes", response_model=CreateNoteResponse)
async def create_note(
    child_id: UUID,
    request: CreateNoteRequest,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Create a note for a child.

    Notes are used by the companion to understand context
    (e.g., "has exam today", "feeling stressed").

    Args:
        child_id: Child user ID.
        request: Note details.
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Created note.
    """
    _require_parent_user(user)

    # Validate note type
    valid_types = {"daily_mood", "concern", "context", "preference", "restriction"}
    if request.note_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid note type. Valid types: {', '.join(valid_types)}",
        )

    # Validate priority
    valid_priorities = {"low", "normal", "high", "urgent"}
    if request.priority not in valid_priorities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority. Valid values: {', '.join(valid_priorities)}",
        )

    service = _get_parent_service(db)

    try:
        note = await service.create_note(
            parent_id=user.id,
            child_id=child_id,
            note_type=request.note_type,
            content=request.content,
            title=request.title,
            reported_emotion=request.reported_emotion,
            emotion_intensity=request.emotion_intensity,
            valid_until=request.valid_until,
            priority=request.priority,
        )

        return CreateNoteResponse(
            success=True,
            note=NoteInfo(
                id=note["id"],
                note_type=note["note_type"],
                title=note["title"],
                content=note["content"],
                reported_emotion=note["reported_emotion"],
                emotion_intensity=note["emotion_intensity"],
                valid_from=note["valid_from"],
                valid_until=note["valid_until"],
                priority=note["priority"],
                ai_processed=False,
                created_at=note["created_at"],
                is_active=True,
            ),
        )
    except ChildNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found",
        )
    except Exception as e:
        logger.exception("Failed to create note: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create note",
        )


@router.patch("/children/{child_id}/notes/{note_id}", response_model=UpdateNoteResponse)
async def update_note(
    child_id: UUID,
    note_id: UUID,
    request: UpdateNoteRequest,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Update a note.

    Args:
        child_id: Child user ID.
        note_id: Note ID.
        request: Fields to update.
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Updated note.
    """
    _require_parent_user(user)

    service = _get_parent_service(db)

    try:
        updates = request.model_dump(exclude_unset=True)
        note = await service.update_note(
            parent_id=user.id,
            child_id=child_id,
            note_id=note_id,
            **updates,
        )

        return UpdateNoteResponse(success=True, note=note)
    except ChildNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found",
        )
    except ParentServiceError as e:
        if e.code == "note_not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Failed to update note: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update note",
        )


@router.delete("/children/{child_id}/notes/{note_id}", response_model=DeleteNoteResponse)
async def delete_note(
    child_id: UUID,
    note_id: UUID,
    user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
):
    """Delete (expire) a note.

    Args:
        child_id: Child user ID.
        note_id: Note ID.
        user: Authenticated parent user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        Deletion confirmation.
    """
    _require_parent_user(user)

    service = _get_parent_service(db)

    try:
        await service.delete_note(
            parent_id=user.id,
            child_id=child_id,
            note_id=note_id,
        )

        return DeleteNoteResponse(
            success=True,
            message="Note deleted",
        )
    except ChildNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Child not found",
        )
    except ParentServiceError as e:
        if e.code == "note_not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except Exception as e:
        logger.exception("Failed to delete note: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete note",
        )
