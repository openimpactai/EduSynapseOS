# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Storage Service for H5P drafts and exports.

This module provides services for managing H5P content drafts,
including save, load, list, and delete operations.

Drafts are stored in Redis with tenant isolation for multi-tenant support.
Keys are prefixed with tenant code to ensure data isolation.
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Redis key prefix for content drafts
DRAFT_KEY_PREFIX = "content_draft"
# Default expiration for drafts (7 days)
DRAFT_EXPIRE_SECONDS = 7 * 24 * 60 * 60


class ContentDraft:
    """Represents a content draft.

    Attributes:
        id: Unique draft ID.
        tenant_code: Tenant identifier.
        user_id: User who created the draft.
        content_type: H5P content type (e.g., "multiple-choice").
        title: Draft title.
        ai_content: AI-generated content (pre-conversion).
        h5p_params: H5P params (post-conversion, if available).
        metadata: Additional metadata.
        tags: List of tags for organization.
        status: Draft status (draft, ready, exported).
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    def __init__(
        self,
        id: UUID | None = None,
        tenant_code: str = "",
        user_id: UUID | None = None,
        content_type: str = "",
        title: str = "",
        ai_content: dict[str, Any] | None = None,
        h5p_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        status: str = "draft",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        """Initialize content draft.

        Args:
            id: Unique draft ID.
            tenant_code: Tenant identifier.
            user_id: User who created the draft.
            content_type: H5P content type.
            title: Draft title.
            ai_content: AI-generated content.
            h5p_params: H5P params.
            metadata: Additional metadata.
            tags: List of tags.
            status: Draft status.
            created_at: Creation timestamp.
            updated_at: Last update timestamp.
        """
        self.id = id or uuid4()
        self.tenant_code = tenant_code
        self.user_id = user_id
        self.content_type = content_type
        self.title = title
        self.ai_content = ai_content or {}
        self.h5p_params = h5p_params
        self.metadata = metadata or {}
        self.tags = tags or []
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with draft data.
        """
        return {
            "id": str(self.id),
            "tenant_code": self.tenant_code,
            "user_id": str(self.user_id) if self.user_id else None,
            "content_type": self.content_type,
            "title": self.title,
            "ai_content": self.ai_content,
            "h5p_params": self.h5p_params,
            "metadata": self.metadata,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentDraft":
        """Create from dictionary.

        Args:
            data: Dictionary with draft data.

        Returns:
            ContentDraft instance.
        """
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            tenant_code=data.get("tenant_code", ""),
            user_id=UUID(data["user_id"]) if data.get("user_id") else None,
            content_type=data.get("content_type", ""),
            title=data.get("title", ""),
            ai_content=data.get("ai_content"),
            h5p_params=data.get("h5p_params"),
            metadata=data.get("metadata"),
            tags=data.get("tags"),
            status=data.get("status", "draft"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


class ContentStorageService:
    """Service for managing H5P content drafts.

    Provides CRUD operations for content drafts stored in Redis.
    Supports filtering by tenant, user, content type, and tags.
    Falls back to in-memory storage if Redis is not available.

    Usage:
        service = ContentStorageService()

        # Save draft
        draft = ContentDraft(
            tenant_code="playground",
            user_id=user_id,
            content_type="multiple-choice",
            title="Quiz about Plants",
            ai_content={...},
        )
        draft_id = await service.save_draft(draft)

        # Load draft
        draft = await service.get_draft(draft_id, tenant_code)

        # List drafts
        drafts = await service.list_drafts(
            tenant_code="playground",
            user_id=user_id,
            content_type="multiple-choice",
        )
    """

    def __init__(self, session: "AsyncSession | None" = None):
        """Initialize the storage service.

        Args:
            session: Database session (not used, kept for API compatibility).
        """
        self._session = session
        # In-memory storage as fallback (for development/testing when Redis is unavailable)
        self._memory_storage: dict[str, ContentDraft] = {}
        self._redis_client = None

    def _get_redis_client(self):
        """Get Redis client if available.

        Returns:
            RedisClient instance or None if not available.
        """
        if self._redis_client is not None:
            return self._redis_client

        try:
            from src.infrastructure.cache import get_redis
            self._redis_client = get_redis()
            return self._redis_client
        except Exception:
            # Redis not initialized, use fallback storage
            return None

    def _draft_key(self, tenant_code: str, draft_id: str) -> str:
        """Build Redis key for a draft.

        Args:
            tenant_code: Tenant code.
            draft_id: Draft ID.

        Returns:
            Redis key string.
        """
        return f"{DRAFT_KEY_PREFIX}:{draft_id}"

    def _index_key(self, tenant_code: str) -> str:
        """Build Redis key for tenant's draft index.

        Args:
            tenant_code: Tenant code.

        Returns:
            Redis key string for the index.
        """
        return f"{DRAFT_KEY_PREFIX}_index"

    def set_session(self, session: "AsyncSession") -> None:
        """Set database session.

        Args:
            session: Database session.
        """
        self._session = session

    async def save_draft(self, draft: ContentDraft) -> str:
        """Save content draft.

        Persists to Redis if available, otherwise uses in-memory storage.

        Args:
            draft: ContentDraft to save.

        Returns:
            Draft ID as string.
        """
        draft.updated_at = datetime.utcnow()
        draft_id = str(draft.id)
        draft_data = draft.to_dict()

        redis = self._get_redis_client()
        if redis:
            try:
                # Save draft data
                await redis.set_with_tenant(
                    draft.tenant_code,
                    self._draft_key(draft.tenant_code, draft_id),
                    draft_data,
                    expire_seconds=DRAFT_EXPIRE_SECONDS,
                )

                # Update index
                index_key = self._index_key(draft.tenant_code)
                index_data = await redis.get_with_tenant(draft.tenant_code, index_key) or []
                if draft_id not in index_data:
                    index_data.append(draft_id)
                    await redis.set_with_tenant(
                        draft.tenant_code,
                        index_key,
                        index_data,
                        expire_seconds=DRAFT_EXPIRE_SECONDS,
                    )

                logger.info(
                    "Saved content draft to Redis: id=%s, type=%s, title=%s",
                    draft.id,
                    draft.content_type,
                    draft.title,
                )
            except Exception as e:
                logger.warning("Redis save failed, using memory storage: %s", e)
                self._memory_storage[draft_id] = draft
        else:
            # Fallback to in-memory storage
            self._memory_storage[draft_id] = draft
            logger.info(
                "Saved content draft to memory: id=%s, type=%s, title=%s",
                draft.id,
                draft.content_type,
                draft.title,
            )

        return draft_id

    async def get_draft(
        self,
        draft_id: str,
        tenant_code: str,
    ) -> ContentDraft | None:
        """Get content draft by ID.

        Args:
            draft_id: Draft ID.
            tenant_code: Tenant code for access control.

        Returns:
            ContentDraft if found, None otherwise.
        """
        redis = self._get_redis_client()
        if redis:
            try:
                data = await redis.get_with_tenant(
                    tenant_code,
                    self._draft_key(tenant_code, draft_id),
                )
                if data:
                    return ContentDraft.from_dict(data)
            except Exception as e:
                logger.warning("Redis get failed, checking memory storage: %s", e)

        # Fallback to in-memory storage
        draft = self._memory_storage.get(draft_id)
        if draft and draft.tenant_code == tenant_code:
            return draft

        return None

    async def update_draft(
        self,
        draft_id: str,
        tenant_code: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update content draft.

        Args:
            draft_id: Draft ID.
            tenant_code: Tenant code for access control.
            updates: Dictionary of fields to update.

        Returns:
            True if update was successful.
        """
        draft = await self.get_draft(draft_id, tenant_code)

        if not draft:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(draft, key):
                setattr(draft, key, value)

        draft.updated_at = datetime.utcnow()

        # Save updated draft
        await self.save_draft(draft)

        logger.info("Updated content draft: id=%s", draft_id)

        return True

    async def delete_draft(
        self,
        draft_id: str,
        tenant_code: str,
    ) -> bool:
        """Delete content draft.

        Args:
            draft_id: Draft ID.
            tenant_code: Tenant code for access control.

        Returns:
            True if deletion was successful.
        """
        draft = await self.get_draft(draft_id, tenant_code)

        if not draft:
            return False

        redis = self._get_redis_client()
        if redis:
            try:
                # Delete draft
                await redis.delete_with_tenant(
                    tenant_code,
                    self._draft_key(tenant_code, draft_id),
                )

                # Update index
                index_key = self._index_key(tenant_code)
                index_data = await redis.get_with_tenant(tenant_code, index_key) or []
                if draft_id in index_data:
                    index_data.remove(draft_id)
                    await redis.set_with_tenant(
                        tenant_code,
                        index_key,
                        index_data,
                        expire_seconds=DRAFT_EXPIRE_SECONDS,
                    )

                logger.info("Deleted content draft from Redis: id=%s", draft_id)
            except Exception as e:
                logger.warning("Redis delete failed: %s", e)

        # Also remove from memory storage
        if draft_id in self._memory_storage:
            del self._memory_storage[draft_id]

        return True

    async def list_drafts(
        self,
        tenant_code: str,
        user_id: UUID | None = None,
        content_type: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ContentDraft]:
        """List content drafts with filtering.

        Args:
            tenant_code: Tenant code (required).
            user_id: Filter by user ID.
            content_type: Filter by content type.
            status: Filter by status.
            tags: Filter by tags (any match).
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of matching ContentDraft objects.
        """
        results = []

        redis = self._get_redis_client()
        if redis:
            try:
                # Get draft IDs from index
                index_key = self._index_key(tenant_code)
                index_data = await redis.get_with_tenant(tenant_code, index_key) or []

                for draft_id in index_data:
                    data = await redis.get_with_tenant(
                        tenant_code,
                        self._draft_key(tenant_code, draft_id),
                    )
                    if data:
                        draft = ContentDraft.from_dict(data)
                        results.append(draft)
            except Exception as e:
                logger.warning("Redis list failed, using memory storage: %s", e)
                results = []

        # Add any drafts from memory storage
        for draft in self._memory_storage.values():
            if draft.tenant_code == tenant_code and draft not in results:
                results.append(draft)

        # Apply filters
        filtered = []
        for draft in results:
            if user_id and draft.user_id != user_id:
                continue
            if content_type and draft.content_type != content_type:
                continue
            if status and draft.status != status:
                continue
            if tags and not any(t in draft.tags for t in tags):
                continue
            filtered.append(draft)

        # Sort by updated_at descending
        filtered.sort(key=lambda d: d.updated_at or datetime.min, reverse=True)

        # Apply pagination
        return filtered[offset:offset + limit]

    async def count_drafts(
        self,
        tenant_code: str,
        user_id: UUID | None = None,
        content_type: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count content drafts with filtering.

        Args:
            tenant_code: Tenant code (required).
            user_id: Filter by user ID.
            content_type: Filter by content type.
            status: Filter by status.

        Returns:
            Count of matching drafts.
        """
        drafts = await self.list_drafts(
            tenant_code=tenant_code,
            user_id=user_id,
            content_type=content_type,
            status=status,
            limit=1000,  # High limit for counting
        )
        return len(drafts)

    async def update_draft_status(
        self,
        draft_id: str,
        tenant_code: str,
        new_status: str,
        h5p_content_id: str | None = None,
    ) -> bool:
        """Update draft status after export.

        Args:
            draft_id: Draft ID.
            tenant_code: Tenant code.
            new_status: New status value.
            h5p_content_id: H5P content ID if exported.

        Returns:
            True if update was successful.
        """
        updates: dict[str, Any] = {"status": new_status}

        if h5p_content_id:
            # Store in metadata
            draft = await self.get_draft(draft_id, tenant_code)
            if draft:
                metadata = draft.metadata or {}
                metadata["h5p_content_id"] = h5p_content_id
                updates["metadata"] = metadata

        return await self.update_draft(draft_id, tenant_code, updates)
