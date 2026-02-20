# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Associative memory layer for interests and concept connections.

This module implements the associative memory layer which stores student
interests, effective analogies, and concept connections. Uses both
PostgreSQL for structured data and Qdrant for semantic similarity search.

Enables personalization by finding relevant analogies based on student
interests and tracking which connections are most effective.

All public methods support optional session injection to share transaction
context with the caller. If a session is provided, it will be used directly;
otherwise, a new session will be created.

Example:
    layer = AssociativeMemoryLayer(
        tenant_db_manager=tenant_db,
        embedding_service=embedding_service,
        qdrant_client=qdrant,
    )

    # Store a new interest
    memory = await layer.store(
        tenant_code="acme",
        student_id=student_uuid,
        association_type="interest",
        content="Minecraft and building games",
        tags=["gaming", "creativity"],
    )

    # Find relevant associations for a topic
    results = await layer.search(
        tenant_code="acme",
        student_id=student_uuid,
        query="fractions and division",
        limit=5,
    )
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.core.intelligence.embeddings import EmbeddingService
from src.infrastructure.database.models.tenant.memory import AssociativeMemory
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.vectors import QdrantVectorClient, SearchResult
from src.models.memory import (
    AssociativeMemoryCreate,
    AssociativeMemoryResponse,
    AssociationType,
    InterestItem,
    StudentInterests,
)

logger = logging.getLogger(__name__)

# Qdrant collection name for associative memories
ASSOCIATIVE_COLLECTION = "associative_memories"


class AssociativeMemoryError(Exception):
    """Exception raised for associative memory operations.

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


class AssociativeMemoryLayer:
    """Service layer for associative memory operations.

    This layer manages interests, analogies, and concept connections with
    both structured storage (PostgreSQL) and semantic search (Qdrant).

    All public methods support optional session injection to share transaction
    context with the caller.

    Attributes:
        tenant_db_manager: Manager for tenant database connections.
        embedding_service: Service for generating text embeddings.
        qdrant_client: Client for vector similarity search.

    Example:
        layer = AssociativeMemoryLayer(
            tenant_db_manager=tenant_db,
            embedding_service=embedding_service,
            qdrant_client=qdrant,
        )

        # Store interest with injected session
        memory = await layer.store(
            tenant_code="acme",
            student_id=uuid.UUID("..."),
            association_type="interest",
            content="Loves soccer and sports statistics",
            session=db_session,  # Optional: reuse existing session
        )
    """

    def __init__(
        self,
        tenant_db_manager: TenantDatabaseManager,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantVectorClient,
    ) -> None:
        """Initialize the associative memory layer.

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
            tenant_code, ASSOCIATIVE_COLLECTION
        )

        exists = await self._qdrant.collection_exists(collection_name)
        if not exists:
            await self._qdrant.create_tenant_collection(
                tenant_code=tenant_code,
                collection=ASSOCIATIVE_COLLECTION,
                vector_size=self._embedding.dimension,
                distance="Cosine",
            )
            logger.info(
                "Created associative memories collection for tenant %s", tenant_code
            )

    async def store(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        association_type: AssociationType | str,
        content: str,
        strength: float = 0.5,
        tags: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> AssociativeMemoryResponse:
        """Store a new associative memory.

        Creates both a database record and a vector embedding in Qdrant.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            association_type: Type of association (interest, analogy, etc.).
            content: Association content/description.
            strength: Initial association strength (0-1).
            tags: Optional tags for categorization.
            session: Optional database session for transaction sharing.

        Returns:
            AssociativeMemoryResponse with the created memory.

        Raises:
            AssociativeMemoryError: If storage fails.
        """
        # Normalize enum value
        assoc_type_str = (
            association_type.value
            if isinstance(association_type, AssociationType)
            else association_type
        )

        # Generate embedding
        try:
            embedding = await self._embedding.embed_text(content)
        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            raise AssociativeMemoryError("Failed to generate embedding", e) from e

        # Generate unique IDs
        memory_id = str(uuid.uuid4())
        embedding_id = memory_id  # Qdrant requires valid UUID format
        now = datetime.now(timezone.utc)

        async def _execute(db: AsyncSession) -> AssociativeMemoryResponse:
            # Ensure collection exists (Qdrant operation - independent of PostgreSQL)
            await self.ensure_collection_exists(tenant_code)

            # Store in Qdrant first
            await self._qdrant.upsert_with_tenant(
                tenant_code=tenant_code,
                collection=ASSOCIATIVE_COLLECTION,
                points=[
                    {
                        "id": embedding_id,
                        "vector": embedding,
                        "payload": {
                            "memory_id": memory_id,
                            "student_id": str(student_id),
                            "association_type": assoc_type_str,
                            "content": content,
                            "strength": strength,
                            "tags": tags or [],
                            "created_at": now.isoformat(),
                        },
                    }
                ],
            )

            # Store in PostgreSQL
            memory = AssociativeMemory(
                id=memory_id,
                student_id=str(student_id),
                association_type=assoc_type_str,
                content=content,
                strength=Decimal(str(strength)),
                times_used=0,
                times_effective=0,
                tags=tags or [],
                embedding_id=embedding_id,
            )
            db.add(memory)

            logger.info(
                "Stored associative memory %s for student %s: %s",
                memory_id,
                student_id,
                assoc_type_str,
            )

            return AssociativeMemoryResponse(
                id=uuid.UUID(memory_id),
                student_id=student_id,
                association_type=AssociationType(assoc_type_str),
                content=content,
                strength=strength,
                times_used=0,
                times_effective=0,
                tags=tags or [],
                last_used_at=None,
                created_at=now,
                updated_at=None,
            )

        try:
            if session:
                return await _execute(session)
            async with self._tenant_db.get_session(tenant_code) as db:
                return await _execute(db)
        except AssociativeMemoryError:
            raise
        except Exception as e:
            logger.error(
                "Failed to store associative memory for student %s: %s",
                student_id,
                str(e),
            )
            raise AssociativeMemoryError(
                "Failed to store associative memory", e
            ) from e

    async def get_by_id(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> AssociativeMemoryResponse | None:
        """Get an associative memory by ID.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            AssociativeMemoryResponse if found, None otherwise.
        """
        async def _execute(db: AsyncSession) -> AssociativeMemoryResponse | None:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id == str(memory_id)
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_by_type(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        association_type: AssociationType | str,
        limit: int = 20,
        session: AsyncSession | None = None,
    ) -> list[AssociativeMemoryResponse]:
        """Get associations of a specific type for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            association_type: Type of association to filter.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of associations ordered by strength.
        """
        assoc_type_str = (
            association_type.value
            if isinstance(association_type, AssociationType)
            else association_type
        )

        async def _execute(db: AsyncSession) -> list[AssociativeMemoryResponse]:
            result = await db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type == assoc_type_str,
                    )
                )
                .order_by(desc(AssociativeMemory.strength))
                .limit(limit)
            )
            memories = result.scalars().all()

            return [self._to_response(m) for m in memories]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_interests(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> list[AssociativeMemoryResponse]:
        """Get all interests for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            List of interest memories ordered by strength.
        """
        return await self.get_by_type(
            tenant_code=tenant_code,
            student_id=student_id,
            association_type=AssociationType.INTEREST,
            limit=50,
            session=session,
        )

    async def get_effective_analogies(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        min_effectiveness: float = 0.6,
        limit: int = 20,
        session: AsyncSession | None = None,
    ) -> list[AssociativeMemoryResponse]:
        """Get analogies that have been effective for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            min_effectiveness: Minimum effectiveness rate.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of effective analogies.
        """
        async def _execute(db: AsyncSession) -> list[AssociativeMemoryResponse]:
            result = await db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type
                        == AssociationType.ANALOGY.value,
                        AssociativeMemory.times_used > 0,
                    )
                )
                .order_by(desc(AssociativeMemory.times_effective))
                .limit(limit * 2)  # Over-fetch to filter by effectiveness
            )
            memories = result.scalars().all()

            # Filter by effectiveness rate
            effective = [
                m
                for m in memories
                if m.times_used > 0
                and (m.times_effective / m.times_used) >= min_effectiveness
            ]

            return [self._to_response(m) for m in effective[:limit]]

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def search(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        association_types: list[AssociationType] | None = None,
        session: AsyncSession | None = None,
    ) -> list[tuple[AssociativeMemoryResponse, float]]:
        """Search associative memories using semantic similarity.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            query: Search query text.
            limit: Maximum number of results.
            min_score: Minimum similarity score (0-1).
            association_types: Optional filter by association types.
            session: Optional database session for transaction sharing.

        Returns:
            List of (memory, score) tuples ordered by similarity.
        """
        # Check if collection exists - return empty if no data stored yet
        collection_name = self._qdrant._tenant_collection_name(
            tenant_code, ASSOCIATIVE_COLLECTION
        )
        if not await self._qdrant.collection_exists(collection_name):
            logger.debug(
                "Associative collection does not exist for tenant %s, returning empty",
                tenant_code,
            )
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding.embed_text(query)
        except Exception as e:
            logger.error("Failed to generate query embedding: %s", str(e))
            raise AssociativeMemoryError(
                "Failed to generate query embedding", e
            ) from e

        # Search in Qdrant
        filter_conditions = {"student_id": str(student_id)}
        if association_types and len(association_types) == 1:
            filter_conditions["association_type"] = association_types[0].value

        try:
            results: list[SearchResult] = await self._qdrant.search_with_tenant(
                tenant_code=tenant_code,
                collection=ASSOCIATIVE_COLLECTION,
                query_vector=query_embedding,
                limit=(
                    limit * 2
                    if association_types and len(association_types) > 1
                    else limit
                ),
                score_threshold=min_score,
                filter_conditions=filter_conditions,
            )
        except Exception as e:
            logger.error("Failed to search associative memories: %s", str(e))
            raise AssociativeMemoryError(
                "Failed to search associative memories", e
            ) from e

        # Post-filter by association types if multiple
        if association_types and len(association_types) > 1:
            type_values = {at.value for at in association_types}
            results = [
                r
                for r in results
                if r.payload.get("association_type") in type_values
            ][:limit]

        # Fetch full memories from database
        memory_ids = [r.payload.get("memory_id") for r in results if r.payload]
        scores_by_id = {
            r.payload.get("memory_id"): r.score for r in results if r.payload
        }

        if not memory_ids:
            return []

        async def _execute(db: AsyncSession) -> list[tuple[AssociativeMemoryResponse, float]]:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id.in_(memory_ids)
                )
            )
            memories = {str(m.id): m for m in result.scalars().all()}

            # Build response with scores, maintaining order
            response: list[tuple[AssociativeMemoryResponse, float]] = []
            for memory_id in memory_ids:
                if memory_id in memories:
                    memory = memories[memory_id]
                    score = scores_by_id.get(memory_id, 0.0)
                    response.append((self._to_response(memory), score))

            return response

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def find_relevant_for_topic(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic_description: str,
        limit: int = 5,
        session: AsyncSession | None = None,
    ) -> list[tuple[AssociativeMemoryResponse, float]]:
        """Find associations relevant to a learning topic.

        Searches for interests and analogies that could be used to
        personalize explanations for the given topic.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic_description: Description of the topic being learned.
            limit: Maximum number of results.
            session: Optional database session for transaction sharing.

        Returns:
            List of (memory, relevance_score) tuples.
        """
        return await self.search(
            tenant_code=tenant_code,
            student_id=student_id,
            query=topic_description,
            limit=limit,
            min_score=0.4,  # Lower threshold for personalization
            association_types=[
                AssociationType.INTEREST,
                AssociationType.ANALOGY,
            ],
            session=session,
        )

    async def record_usage(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        was_effective: bool,
        session: AsyncSession | None = None,
    ) -> AssociativeMemoryResponse | None:
        """Record usage of an association and update effectiveness.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            was_effective: Whether the usage was effective.
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response, or None if not found.
        """
        async def _execute(db: AsyncSession) -> AssociativeMemoryResponse | None:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id == str(memory_id)
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            memory.record_usage(was_effective)

            logger.debug(
                "Recorded usage for association %s: effective=%s, "
                "new_strength=%.2f",
                memory_id,
                was_effective,
                float(memory.strength),
            )

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def update_strength(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        strength: float,
        session: AsyncSession | None = None,
    ) -> AssociativeMemoryResponse | None:
        """Manually update association strength.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            strength: New strength value (0-1).
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response, or None if not found.
        """
        async def _execute(db: AsyncSession) -> AssociativeMemoryResponse | None:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id == str(memory_id)
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            memory.strength = Decimal(str(max(0.0, min(1.0, strength))))

            # Update Qdrant payload
            if memory.embedding_id:
                try:
                    embedding = await self._embedding.embed_text(memory.content)
                    await self._qdrant.upsert_with_tenant(
                        tenant_code=tenant_code,
                        collection=ASSOCIATIVE_COLLECTION,
                        points=[
                            {
                                "id": memory.embedding_id,
                                "vector": embedding,
                                "payload": {
                                    "memory_id": str(memory.id),
                                    "student_id": memory.student_id,
                                    "association_type": memory.association_type,
                                    "content": memory.content,
                                    "strength": float(memory.strength),
                                    "tags": memory.tags,
                                    "created_at": memory.created_at.isoformat(),
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

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def add_tags(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        tags: list[str],
        session: AsyncSession | None = None,
    ) -> AssociativeMemoryResponse | None:
        """Add tags to an association.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            tags: Tags to add.
            session: Optional database session for transaction sharing.

        Returns:
            Updated memory response, or None if not found.
        """
        async def _execute(db: AsyncSession) -> AssociativeMemoryResponse | None:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id == str(memory_id)
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return None

            # Add unique tags
            existing_tags = set(memory.tags)
            new_tags = list(existing_tags | set(tags))
            memory.tags = new_tags

            return self._to_response(memory)

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    async def get_student_interests(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> StudentInterests:
        """Get aggregated student interests.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            StudentInterests with categorized interests and analogies.
        """
        async def _execute(db: AsyncSession) -> StudentInterests:
            # Get all interests
            interests_result = await db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type
                        == AssociationType.INTEREST.value,
                    )
                )
                .order_by(desc(AssociativeMemory.strength))
            )
            interests = interests_result.scalars().all()

            # Get effective analogies
            analogies_result = await db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type
                        == AssociationType.ANALOGY.value,
                        AssociativeMemory.times_used > 0,
                    )
                )
                .order_by(desc(AssociativeMemory.times_effective))
                .limit(10)
            )
            analogies = analogies_result.scalars().all()

            # Build interest items with category from tags
            interest_items = []
            for interest in interests:
                category = interest.tags[0] if interest.tags else None
                interest_items.append(
                    InterestItem(
                        content=interest.content,
                        category=category,
                        strength=float(interest.strength),
                    )
                )

            # Filter for effective analogies
            effective_analogies = [
                a.content
                for a in analogies
                if a.times_used > 0 and (a.times_effective / a.times_used) >= 0.5
            ]

            return StudentInterests(
                student_id=student_id,
                interests=interest_items,
                effective_analogies=effective_analogies,
            )

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def get_student_interests_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
    ) -> StudentInterests:
        """Get aggregated student interests (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.

        Returns:
            StudentInterests with categorized interests and analogies.
        """
        def _execute(db: Session) -> StudentInterests:
            # Get all interests
            interests_result = db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type
                        == AssociationType.INTEREST.value,
                    )
                )
                .order_by(desc(AssociativeMemory.strength))
            )
            interests = interests_result.scalars().all()

            # Get effective analogies
            analogies_result = db.execute(
                select(AssociativeMemory)
                .where(
                    and_(
                        AssociativeMemory.student_id == str(student_id),
                        AssociativeMemory.association_type
                        == AssociationType.ANALOGY.value,
                        AssociativeMemory.times_used > 0,
                    )
                )
                .order_by(desc(AssociativeMemory.times_effective))
                .limit(10)
            )
            analogies = analogies_result.scalars().all()

            # Build interest items with category from tags
            interest_items = []
            for interest in interests:
                category = interest.tags[0] if interest.tags else None
                interest_items.append(
                    InterestItem(
                        content=interest.content,
                        category=category,
                        strength=float(interest.strength),
                    )
                )

            # Filter for effective analogies
            effective_analogies = [
                a.content
                for a in analogies
                if a.times_used > 0 and (a.times_effective / a.times_used) >= 0.5
            ]

            return StudentInterests(
                student_id=student_id,
                interests=interest_items,
                effective_analogies=effective_analogies,
            )

        with self._tenant_db.get_sync_session(tenant_code) as db:
            return _execute(db)

    async def delete(
        self,
        tenant_code: str,
        memory_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> bool:
        """Delete an associative memory.

        Removes from both PostgreSQL and Qdrant.

        Args:
            tenant_code: Unique tenant identifier.
            memory_id: Memory's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            True if deleted, False if not found.
        """
        async def _execute(db: AsyncSession) -> tuple[bool, str | None]:
            result = await db.execute(
                select(AssociativeMemory).where(
                    AssociativeMemory.id == str(memory_id)
                )
            )
            memory = result.scalar_one_or_none()

            if memory is None:
                return False, None

            embedding_id = memory.embedding_id

            await db.delete(memory)

            return True, embedding_id

        if session:
            deleted, embedding_id = await _execute(session)
        else:
            async with self._tenant_db.get_session(tenant_code) as db:
                deleted, embedding_id = await _execute(db)

        if not deleted:
            return False

        # Delete from Qdrant (outside transaction)
        if embedding_id:
            try:
                await self._qdrant.delete_points(
                    self._qdrant._tenant_collection_name(
                        tenant_code, ASSOCIATIVE_COLLECTION
                    ),
                    [embedding_id],
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete Qdrant point %s: %s",
                    embedding_id,
                    str(e),
                )

        logger.info("Deleted associative memory %s", memory_id)
        return True

    async def count_by_student(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Count total associative memories for a student.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            Total count of memories.
        """
        async def _execute(db: AsyncSession) -> int:
            result = await db.execute(
                select(func.count(AssociativeMemory.id)).where(
                    AssociativeMemory.student_id == str(student_id)
                )
            )
            return result.scalar() or 0

        if session:
            return await _execute(session)
        async with self._tenant_db.get_session(tenant_code) as db:
            return await _execute(db)

    def _to_response(self, memory: AssociativeMemory) -> AssociativeMemoryResponse:
        """Convert database model to response DTO.

        Args:
            memory: Database model instance.

        Returns:
            AssociativeMemoryResponse DTO.
        """
        return AssociativeMemoryResponse(
            id=uuid.UUID(memory.id),
            student_id=uuid.UUID(memory.student_id),
            association_type=AssociationType(memory.association_type),
            content=memory.content,
            strength=float(memory.strength),
            times_used=memory.times_used,
            times_effective=memory.times_effective,
            tags=memory.tags,
            last_used_at=memory.last_used_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )
