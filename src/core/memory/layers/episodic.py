# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Episodic memory layer for session-based learning events.

This module implements the episodic memory layer which stores and retrieves
specific learning events and experiences. Uses both PostgreSQL for structured
data and Qdrant for semantic similarity search.

Event types include struggles, breakthroughs, confusion moments, mastery
demonstrations, and emotional states during learning sessions.

Session Injection Pattern:
    All database methods accept an optional `session` parameter. When provided,
    operations use the given session (sharing transaction context).
    When not provided, a new session is created for the operation.
    Note: Qdrant operations are independent of PostgreSQL session.

Example:
    layer = EpisodicMemoryLayer(
        tenant_db_manager=tenant_db,
        embedding_service=embedding_service,
        qdrant_client=qdrant,
    )

    # Store a new memory (creates own session)
    memory = await layer.store(
        tenant_code="acme",
        student_id=student_uuid,
        event_type="breakthrough",
        summary="Student understood fractions using pizza analogy",
        importance=0.9,
    )

    # With session injection (shares transaction)
    async with tenant_db.get_session("acme") as session:
        memories = await layer.get_recent(
            tenant_code="acme",
            student_id=student_uuid,
            session=session,
        )
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, joinedload

from src.core.intelligence.embeddings import EmbeddingService
from src.domains.curriculum import CurriculumLookup
from src.infrastructure.database.models.tenant.memory import EpisodicMemory
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.vectors import QdrantVectorClient, SearchResult
from src.models.memory import (
    EmotionalState,
    EpisodicEventType,
    EpisodicMemoryCreate,
    EpisodicMemoryResponse,
    EpisodicMemorySearchParams,
)

logger = logging.getLogger(__name__)

# Qdrant collection name for episodic memories
EPISODIC_COLLECTION = "episodic_memories"


class EpisodicMemoryError(Exception):
    """Exception raised for episodic memory operations.

    Attributes:
        message: Error description.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class EpisodicMemoryLayer:
    """Service layer for episodic memory operations.

    This layer manages learning event memories with both structured storage
    (PostgreSQL) and semantic search capabilities (Qdrant vectors).

    Session Injection:
        All methods accept an optional `session` parameter for transaction
        sharing. This enables callers to pass an existing database session
        to avoid creating new connections and to share transaction context.
        Note: Qdrant operations are independent of PostgreSQL sessions.

    Attributes:
        tenant_db_manager: Manager for tenant database connections.
        embedding_service: Service for generating text embeddings.
        qdrant_client: Client for vector similarity search.

    Example:
        layer = EpisodicMemoryLayer(
            tenant_db_manager=tenant_db,
            embedding_service=embedding_service,
            qdrant_client=qdrant,
        )

        # Store memory with automatic embedding
        memory = await layer.store(
            tenant_code="acme",
            student_id=uuid.UUID("..."),
            event_type="breakthrough",
            summary="Understood algebra concepts",
        )
    """

    def __init__(
        self,
        tenant_db_manager: TenantDatabaseManager,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantVectorClient,
    ) -> None:
        """Initialize the episodic memory layer.

        Args:
            tenant_db_manager: Manager for tenant database connections.
            embedding_service: Service for generating text embeddings.
            qdrant_client: Client for Qdrant vector database.
        """
        self._tenant_db = tenant_db_manager
        self._embedding = embedding_service
        self._qdrant = qdrant_client

    async def ensure_collection_exists(self, tenant_code: str) -> None:
        """Ensure the Qdrant collection exists for a tenant.

        Creates the collection if it doesn't exist.

        Args:
            tenant_code: Unique tenant identifier.
        """
        collection_name = self._qdrant._tenant_collection_name(
            tenant_code, EPISODIC_COLLECTION
        )

        exists = await self._qdrant.collection_exists(collection_name)
        if not exists:
            await self._qdrant.create_tenant_collection(
                tenant_code=tenant_code,
                collection=EPISODIC_COLLECTION,
                vector_size=self._embedding.dimension,
                distance="Cosine",
            )
            logger.info(
                "Created episodic memories collection for tenant %s", tenant_code
            )

    async def store(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        event_type: EpisodicEventType | str,
        summary: str,
        details: dict[str, Any] | None = None,
        emotional_state: EmotionalState | str | None = None,
        importance: float = 0.5,
        session_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> EpisodicMemoryResponse:
        """Store a new episodic memory.

        Creates both a database record and a vector embedding in Qdrant
        for semantic search.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            event_type: Type of learning event.
            summary: Human-readable event summary.
            details: Additional event details as dict.
            emotional_state: Detected emotional state.
            importance: Importance score between 0 and 1.
            session_id: Related practice session ID.
            conversation_id: Related conversation ID.
            topic_full_code: Related topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            session: Optional database session for transaction sharing.

        Returns:
            EpisodicMemoryResponse with the created memory.

        Raises:
            EpisodicMemoryError: If storage fails.
        """
        # Normalize enum values
        event_type_str = (
            event_type.value if isinstance(event_type, EpisodicEventType) else event_type
        )
        emotional_str = (
            emotional_state.value
            if isinstance(emotional_state, EmotionalState)
            else emotional_state
        )

        # Generate embedding for the summary
        try:
            embedding = await self._embedding.embed_text(summary)
        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            raise EpisodicMemoryError("Failed to generate embedding", e) from e

        # Generate unique IDs
        memory_id = str(uuid.uuid4())
        embedding_id = memory_id  # Qdrant requires valid UUID format
        now = datetime.now(timezone.utc)

        try:
            # Ensure collection exists
            await self.ensure_collection_exists(tenant_code)

            # Store in Qdrant first
            await self._qdrant.upsert_with_tenant(
                tenant_code=tenant_code,
                collection=EPISODIC_COLLECTION,
                points=[
                    {
                        "id": embedding_id,
                        "vector": embedding,
                        "payload": {
                            "memory_id": memory_id,
                            "student_id": str(student_id),
                            "event_type": event_type_str,
                            "summary": summary,
                            "importance": importance,
                            "occurred_at": now.isoformat(),
                        },
                    }
                ],
            )

            # Parse topic_full_code into composite key parts
            topic_framework_code = None
            topic_subject_code = None
            topic_grade_code = None
            topic_unit_code = None
            topic_code = None
            if topic_full_code:
                parts = topic_full_code.split(".")
                if len(parts) == 5:
                    topic_framework_code = parts[0]
                    topic_subject_code = parts[1]
                    topic_grade_code = parts[2]
                    topic_unit_code = parts[3]
                    topic_code = parts[4]

            # Store in PostgreSQL
            async def _execute(db: AsyncSession) -> None:
                memory = EpisodicMemory(
                    id=memory_id,
                    student_id=str(student_id),
                    event_type=event_type_str,
                    summary=summary,
                    details=details or {},
                    emotional_state=emotional_str,
                    importance=Decimal(str(importance)),
                    session_id=str(session_id) if session_id else None,
                    conversation_id=str(conversation_id) if conversation_id else None,
                    topic_framework_code=topic_framework_code,
                    topic_subject_code=topic_subject_code,
                    topic_grade_code=topic_grade_code,
                    topic_unit_code=topic_unit_code,
                    topic_code=topic_code,
                    occurred_at=now,
                    embedding_id=embedding_id,
                    access_count=0,
                )
                db.add(memory)

            if session:
                await _execute(session)
            else:
                async with self._tenant_db.get_session(tenant_code) as db:
                    await _execute(db)

            logger.info(
                "Stored episodic memory %s for student %s: %s",
                memory_id,
                student_id,
                event_type_str,
            )

            return EpisodicMemoryResponse(
                id=uuid.UUID(memory_id),
                student_id=student_id,
                event_type=EpisodicEventType(event_type_str),
                summary=summary,
                details=details or {},
                emotional_state=(
                    EmotionalState(emotional_str) if emotional_str else None
                ),
                importance=importance,
                session_id=session_id,
                conversation_id=conversation_id,
                topic_full_code=topic_full_code,
                topic_name=None,
                occurred_at=now,
                access_count=0,
                last_accessed_at=None,
            )

        except Exception as e:
            logger.error(
                "Failed to store episodic memory for student %s: %s",
                student_id,
                str(e),
            )
            raise EpisodicMemoryError("Failed to store episodic memory", e) from e

    async def get_by_id(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        record_access: bool = True,
        session: AsyncSession | None = None,
    ) -> EpisodicMemoryResponse | None:
        """Get an episodic memory by ID.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            record_access: Whether to increment access count.
            session: Optional database session for transaction sharing.

        Returns:
            EpisodicMemoryResponse if found, None otherwise.
        """

        async def _execute(db: AsyncSession) -> EpisodicMemoryResponse | None:
            result = await db.execute(
                select(EpisodicMemory).where(EpisodicMemory.id == str(memory_id))
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            if record_access:
                memory.record_access()

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_recent(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        limit: int = 10,
        event_types: list[EpisodicEventType] | None = None,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get recent episodic memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            limit: Maximum number of memories to return.
            event_types: Optional filter by event types.
            session: Optional database session for transaction sharing.

        Returns:
            List of recent episodic memories ordered by occurrence time.
        """

        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            query = select(EpisodicMemory).where(
                EpisodicMemory.student_id == str(student_id)
            )

            if event_types:
                type_values = [et.value for et in event_types]
                query = query.where(EpisodicMemory.event_type.in_(type_values))

            query = query.order_by(desc(EpisodicMemory.occurred_at)).limit(limit)

            result = await db.execute(query)
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def get_recent_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        limit: int = 10,
        event_types: list[EpisodicEventType] | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get recent episodic memories for a student (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            limit: Maximum number of memories to return.
            event_types: Optional filter by event types.

        Returns:
            List of recent episodic memories ordered by occurrence time.
        """

        def _execute(db: Session) -> list[EpisodicMemoryResponse]:
            query = select(EpisodicMemory).where(
                EpisodicMemory.student_id == str(student_id)
            )

            if event_types:
                type_values = [et.value for et in event_types]
                query = query.where(EpisodicMemory.event_type.in_(type_values))

            query = query.order_by(desc(EpisodicMemory.occurred_at)).limit(limit)

            result = db.execute(query)
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        with self._tenant_db.get_sync_session(tenant_code) as db:
            return _execute(db)

    async def search(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        event_types: list[EpisodicEventType] | None = None,
        session: AsyncSession | None = None,
    ) -> list[tuple[EpisodicMemoryResponse, float]]:
        """Search episodic memories using semantic similarity.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            query: Search query text.
            limit: Maximum number of results.
            min_score: Minimum similarity score (0-1).
            event_types: Optional filter by event types.
            session: Optional database session for transaction sharing.

        Returns:
            List of (memory, score) tuples ordered by similarity.
        """
        # Check if collection exists - return empty if no data stored yet
        collection_name = self._qdrant._tenant_collection_name(
            tenant_code, EPISODIC_COLLECTION
        )
        if not await self._qdrant.collection_exists(collection_name):
            logger.debug(
                "Episodic collection does not exist for tenant %s, returning empty",
                tenant_code,
            )
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding.embed_text(query)
        except Exception as e:
            logger.error("Failed to generate query embedding: %s", str(e))
            raise EpisodicMemoryError("Failed to generate query embedding", e) from e

        # Search in Qdrant
        filter_conditions = {"student_id": str(student_id)}
        if event_types:
            # Note: Qdrant doesn't support OR filters directly in simple conditions
            # For multiple event types, we'll filter in post-processing
            if len(event_types) == 1:
                filter_conditions["event_type"] = event_types[0].value

        try:
            results: list[SearchResult] = await self._qdrant.search_with_tenant(
                tenant_code=tenant_code,
                collection=EPISODIC_COLLECTION,
                query_vector=query_embedding,
                limit=limit * 2 if event_types and len(event_types) > 1 else limit,
                score_threshold=min_score,
                filter_conditions=filter_conditions,
            )
        except Exception as e:
            logger.error("Failed to search episodic memories: %s", str(e))
            raise EpisodicMemoryError("Failed to search episodic memories", e) from e

        # Post-filter by event types if multiple
        if event_types and len(event_types) > 1:
            type_values = {et.value for et in event_types}
            results = [
                r for r in results if r.payload.get("event_type") in type_values
            ][:limit]

        # Fetch full memories from database
        memory_ids = [r.payload.get("memory_id") for r in results if r.payload]
        scores_by_id = {r.payload.get("memory_id"): r.score for r in results if r.payload}

        if not memory_ids:
            return []

        # Use joinedload to eagerly load topic relationship
        async def _execute(db: AsyncSession) -> dict[str, EpisodicMemory]:
            result = await db.execute(
                select(EpisodicMemory)
                .options(joinedload(EpisodicMemory.topic))
                .where(EpisodicMemory.id.in_(memory_ids))
            )
            return {str(m.id): m for m in result.scalars().all()}

        if session:
            memories = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                memories = await _execute(db)

        # Build response with scores, maintaining order
        response: list[tuple[EpisodicMemoryResponse, float]] = []
        for memory_id in memory_ids:
            if memory_id in memories:
                memory = memories[memory_id]
                score = scores_by_id.get(memory_id, 0.0)
                response.append((self._to_response(memory), score))

        return response

    async def search_with_params(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        params: EpisodicMemorySearchParams,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Search episodic memories with comprehensive parameters.

        Combines semantic search (if query provided) with structured filters.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            params: Search parameters including filters and query.
            session: Optional database session for transaction sharing.

        Returns:
            List of matching episodic memories.
        """
        # If semantic query is provided, use vector search
        if params.query:
            results = await self.search(
                tenant_code=tenant_code,
                student_id=student_id,
                query=params.query,
                limit=params.limit,
                event_types=params.event_types,
                session=session,
            )
            return [memory for memory, _ in results]

        # Otherwise, use structured query
        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            query = select(EpisodicMemory).where(
                EpisodicMemory.student_id == str(student_id)
            )

            conditions = []

            if params.event_types:
                type_values = [et.value for et in params.event_types]
                conditions.append(EpisodicMemory.event_type.in_(type_values))

            # Topic filtering using composite keys from Central Curriculum
            if params.topic_full_code:
                parts = params.topic_full_code.split(".")
                if len(parts) == 5:
                    conditions.append(EpisodicMemory.topic_framework_code == parts[0])
                    conditions.append(EpisodicMemory.topic_subject_code == parts[1])
                    conditions.append(EpisodicMemory.topic_grade_code == parts[2])
                    conditions.append(EpisodicMemory.topic_unit_code == parts[3])
                    conditions.append(EpisodicMemory.topic_code == parts[4])

            if params.emotional_state:
                conditions.append(
                    EpisodicMemory.emotional_state == params.emotional_state.value
                )

            if params.min_importance is not None:
                conditions.append(
                    EpisodicMemory.importance >= Decimal(str(params.min_importance))
                )

            if params.date_from:
                conditions.append(EpisodicMemory.occurred_at >= params.date_from)

            if params.date_to:
                conditions.append(EpisodicMemory.occurred_at <= params.date_to)

            if conditions:
                query = query.where(and_(*conditions))

            query = query.order_by(desc(EpisodicMemory.occurred_at)).limit(params.limit)

            result = await db.execute(query)
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_important_memories(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_importance: float = 0.7,
        limit: int = 20,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get high-importance memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_importance: Minimum importance threshold.
            limit: Maximum number of memories.
            session: Optional database session for transaction sharing.

        Returns:
            List of important memories ordered by importance.
        """

        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            result = await db.execute(
                select(EpisodicMemory)
                .where(
                    and_(
                        EpisodicMemory.student_id == str(student_id),
                        EpisodicMemory.importance >= Decimal(str(min_importance)),
                    )
                )
                .order_by(desc(EpisodicMemory.importance))
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_by_session(
        self,
        tenant_code: str,
        session_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get all memories from a specific learning session.

        Args:
            tenant_code: Unique tenant identifier.
            session_id: Practice session's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            List of memories from the session.
        """

        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            result = await db.execute(
                select(EpisodicMemory)
                .where(EpisodicMemory.session_id == str(session_id))
                .order_by(EpisodicMemory.occurred_at)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def update_importance(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        importance: float,
        session: AsyncSession | None = None,
    ) -> EpisodicMemoryResponse | None:
        """Update the importance score of a memory.

        Also updates the importance in Qdrant payload.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            importance: New importance score (0-1).
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response, or None if not found.
        """

        async def _execute(db: AsyncSession) -> tuple[EpisodicMemoryResponse | None, str | None]:
            result = await db.execute(
                select(EpisodicMemory).where(EpisodicMemory.id == str(memory_id))
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None, None

            memory.importance = Decimal(str(importance))
            embedding_id = memory.embedding_id
            summary = memory.summary
            student_id = memory.student_id
            event_type = memory.event_type
            occurred_at = memory.occurred_at

            return self._to_response(memory), embedding_id

        if session:
            response, embedding_id = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                response, embedding_id = await _execute(db)

        if response is None:
            return None

        # Update Qdrant payload (outside transaction)
        if embedding_id:
            try:
                # Re-embed the summary to update the point
                embedding = await self._embedding.embed_text(response.summary)
                await self._qdrant.upsert_with_tenant(
                    tenant_code=tenant_code,
                    collection=EPISODIC_COLLECTION,
                    points=[
                        {
                            "id": embedding_id,
                            "vector": embedding,
                            "payload": {
                                "memory_id": str(response.id),
                                "student_id": str(response.student_id),
                                "event_type": response.event_type.value,
                                "summary": response.summary,
                                "importance": importance,
                                "occurred_at": response.occurred_at.isoformat(),
                            },
                        }
                    ],
                )
            except Exception as e:
                logger.warning(
                    "Failed to update Qdrant for memory %s: %s",
                    memory_id,
                    str(e),
                )

        return response

    async def delete(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> bool:
        """Delete an episodic memory.

        Removes from both PostgreSQL and Qdrant.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            True if deleted, False if not found.
        """

        async def _execute(db: AsyncSession) -> str | None:
            result = await db.execute(
                select(EpisodicMemory).where(EpisodicMemory.id == str(memory_id))
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            embedding_id = memory.embedding_id
            await db.delete(memory)
            return embedding_id

        if session:
            embedding_id = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                embedding_id = await _execute(db)

        if embedding_id is None:
            return False

        # Delete from Qdrant (outside transaction)
        try:
            await self._qdrant.delete_points(
                self._qdrant._tenant_collection_name(
                    tenant_code, EPISODIC_COLLECTION
                ),
                [embedding_id],
            )
        except Exception as e:
            logger.warning(
                "Failed to delete Qdrant point %s: %s",
                embedding_id,
                str(e),
            )

        logger.info("Deleted episodic memory %s", memory_id)
        return True

    async def count_by_student(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Count total episodic memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            Total count of memories.
        """

        async def _execute(db: AsyncSession) -> int:
            result = await db.execute(
                select(func.count(EpisodicMemory.id)).where(
                    EpisodicMemory.student_id == str(student_id)
                )
            )
            return result.scalar() or 0

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_event_type_stats(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> dict[str, int]:
        """Get count of memories by event type for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            Dictionary mapping event types to counts.
        """

        async def _execute(db: AsyncSession) -> dict[str, int]:
            result = await db.execute(
                select(
                    EpisodicMemory.event_type,
                    func.count(EpisodicMemory.id).label("count"),
                )
                .where(EpisodicMemory.student_id == str(student_id))
                .group_by(EpisodicMemory.event_type)
            )
            rows = result.all()

            return {row.event_type: row.count for row in rows}

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_reportable_events(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get events suitable for progress reports.

        Returns high-importance events with meaningful summaries
        that can be included in student progress reports.

        Reportable events:
        - Breakthroughs (new concept understood)
        - Mastery demonstrations
        - Struggles overcome
        - Session completions with learning gains
        - Comprehension verifications

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            date_from: Start date filter.
            date_to: End date filter.
            limit: Maximum number of events.
            session: Optional database session for transaction sharing.

        Returns:
            List of reportable events ordered by importance and date.
        """
        reportable_types = [
            EpisodicEventType.BREAKTHROUGH,
            EpisodicEventType.MASTERY,
            EpisodicEventType.STRUGGLE_OVERCOME,
            EpisodicEventType.CONCEPT_LEARNED,
        ]

        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            conditions = [
                EpisodicMemory.student_id == str(student_id),
                EpisodicMemory.event_type.in_([et.value for et in reportable_types]),
                EpisodicMemory.importance >= Decimal("0.5"),  # Min importance for reports
            ]

            if date_from:
                conditions.append(EpisodicMemory.occurred_at >= date_from)
            if date_to:
                conditions.append(EpisodicMemory.occurred_at <= date_to)

            result = await db.execute(
                select(EpisodicMemory)
                .where(and_(*conditions))
                .order_by(
                    desc(EpisodicMemory.importance),
                    desc(EpisodicMemory.occurred_at),
                )
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_events_for_topic(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic_full_code: str,
        limit: int = 20,
        session: AsyncSession | None = None,
    ) -> list[EpisodicMemoryResponse]:
        """Get all learning events for a specific topic.

        Returns events related to a topic, useful for understanding
        a student's learning journey on that topic.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic_full_code: Full topic code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            limit: Maximum number of events.
            session: Optional database session for transaction sharing.

        Returns:
            List of events for the topic ordered by occurrence time.
        """
        # Parse topic code into parts
        parts = topic_full_code.split(".")
        if len(parts) != 5:
            logger.warning("Invalid topic_full_code format: %s", topic_full_code)
            return []

        async def _execute(db: AsyncSession) -> list[EpisodicMemoryResponse]:
            result = await db.execute(
                select(EpisodicMemory)
                .where(
                    and_(
                        EpisodicMemory.student_id == str(student_id),
                        EpisodicMemory.topic_framework_code == parts[0],
                        EpisodicMemory.topic_subject_code == parts[1],
                        EpisodicMemory.topic_grade_code == parts[2],
                        EpisodicMemory.topic_unit_code == parts[3],
                        EpisodicMemory.topic_code == parts[4],
                    )
                )
                .order_by(desc(EpisodicMemory.occurred_at))
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def store_comprehension_evaluation(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic_full_code: str,
        evaluation_result: dict,
        session_id: uuid.UUID | None = None,
        session: AsyncSession | None = None,
    ) -> EpisodicMemoryResponse:
        """Store a comprehension evaluation as an episodic memory.

        Creates a rich episodic record from a ComprehensionEvaluationResult
        that can be used for reports and analysis.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic_full_code: Topic code for the evaluation.
            evaluation_result: ComprehensionEvaluationResult as dict.
            session_id: Related learning session ID.
            session: Optional database session for transaction sharing.

        Returns:
            EpisodicMemoryResponse for the stored evaluation.
        """
        score = evaluation_result.get("understanding_score", 0.0)
        verified = evaluation_result.get("verified", False)
        misconceptions = evaluation_result.get("misconceptions", [])
        concepts_understood = evaluation_result.get("concepts_understood", [])

        # Determine event type and importance based on result
        if verified and score >= 0.8:
            event_type = EpisodicEventType.BREAKTHROUGH
            importance = 0.85
            summary = f"Demonstrated strong understanding of the topic. Concepts mastered: {', '.join(concepts_understood[:3])}"
        elif misconceptions:
            event_type = EpisodicEventType.STRUGGLE
            importance = 0.7
            misconception_desc = misconceptions[0].get("description", "misconception")
            summary = f"Showed misconception during comprehension check: {misconception_desc}"
        elif score >= 0.5:
            event_type = EpisodicEventType.CONCEPT_LEARNED
            importance = 0.6
            summary = f"Partial understanding demonstrated. Score: {score:.0%}"
        else:
            event_type = EpisodicEventType.STRUGGLE
            importance = 0.5
            summary = f"Struggled with explanation during comprehension check. Score: {score:.0%}"

        # Store with full details
        return await self.store(
            tenant_code=tenant_code,
            student_id=student_id,
            event_type=event_type,
            summary=summary,
            details={
                "comprehension_score": score,
                "verified": verified,
                "parroting_detected": evaluation_result.get("parroting_detected", False),
                "concepts_understood": concepts_understood,
                "concepts_missing": evaluation_result.get("concepts_missing", []),
                "misconceptions": misconceptions,
                "trigger": evaluation_result.get("trigger", "unknown"),
                "recommended_action": evaluation_result.get("recommended_action", ""),
                "workflow_type": "learning_tutor",
                "reportable": True if verified or misconceptions else False,
            },
            importance=importance,
            session_id=session_id,
            topic_full_code=topic_full_code,
            session=session,
        )

    def _to_response(self, memory: EpisodicMemory) -> EpisodicMemoryResponse:
        """Convert database model to response DTO.

        Args:
            memory: Database model instance.

        Returns:
            EpisodicMemoryResponse DTO.
        """
        return EpisodicMemoryResponse(
            id=uuid.UUID(memory.id),
            student_id=uuid.UUID(memory.student_id),
            event_type=EpisodicEventType(memory.event_type),
            summary=memory.summary,
            details=memory.details,
            emotional_state=(
                EmotionalState(memory.emotional_state)
                if memory.emotional_state
                else None
            ),
            importance=float(memory.importance),
            session_id=uuid.UUID(memory.session_id) if memory.session_id else None,
            conversation_id=(
                uuid.UUID(memory.conversation_id) if memory.conversation_id else None
            ),
            topic_full_code=memory.topic_full_code,
            topic_name=memory.topic.name if memory.topic else None,
            occurred_at=memory.occurred_at,
            access_count=memory.access_count,
            last_accessed_at=memory.last_accessed_at,
        )
