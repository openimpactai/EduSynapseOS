# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory manager orchestrating all memory layers.

This module provides the MemoryManager class which coordinates the four
memory layers and provides unified access to the complete memory context
for AI tutors.

The manager handles:
- Initialization of all memory layers
- Full context retrieval for AI tutor personalization
- Cross-layer memory search
- Lifecycle management (ensure collections, cleanup)

All public methods support optional session injection to share transaction
context with the caller. If a session is provided, it will be used directly;
otherwise, a new session will be created.

Example:
    from src.core.memory import MemoryManager

    # Initialize manager
    manager = MemoryManager(
        tenant_db_manager=tenant_db,
        embedding_service=embedding_service,
        qdrant_client=qdrant,
    )

    # Get full context for AI tutor with injected session
    context = await manager.get_full_context(
        tenant_code="acme",
        student_id=student_uuid,
        session=db_session,  # Optional: reuse existing session
    )

    # Search across all layers
    results = await manager.search_all(
        tenant_code="acme",
        student_id=student_uuid,
        query="fractions",
    )
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.intelligence.embeddings import EmbeddingService
from src.core.memory.layers.associative import AssociativeMemoryLayer
from src.core.memory.layers.episodic import EpisodicMemoryLayer
from src.core.memory.layers.procedural import ProceduralMemoryLayer
from src.core.memory.layers.semantic import SemanticMemoryLayer
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.vectors import QdrantVectorClient
from src.models.memory import (
    DiagnosticContext,
    EpisodicEventType,
    EpisodicMemoryResponse,
    FullMemoryContext,
    LearningPatterns,
    MasteryOverview,
    MemorySearchRequest,
    MemorySearchResponse,
    SemanticMemoryResponse,
    StrategyType,
    StudentInterests,
)

logger = logging.getLogger(__name__)


class MemoryManagerError(Exception):
    """Exception raised for memory manager operations.

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


class MemoryManager:
    """Orchestrates all memory layers for unified access.

    This manager provides a single entry point for all memory operations,
    coordinating the four memory layers (episodic, semantic, procedural,
    associative) and providing high-level operations like full context
    retrieval.

    Attributes:
        episodic: Episodic memory layer instance.
        semantic: Semantic memory layer instance.
        procedural: Procedural memory layer instance.
        associative: Associative memory layer instance.

    Example:
        manager = MemoryManager(
            tenant_db_manager=tenant_db,
            embedding_service=embedding_service,
            qdrant_client=qdrant,
        )

        # Access individual layers
        recent = await manager.episodic.get_recent(
            tenant_code="acme",
            student_id=student_uuid,
        )

        # Get unified context
        context = await manager.get_full_context(
            tenant_code="acme",
            student_id=student_uuid,
        )
    """

    def __init__(
        self,
        tenant_db_manager: TenantDatabaseManager,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantVectorClient,
    ) -> None:
        """Initialize the memory manager.

        Creates instances of all four memory layers.

        Args:
            tenant_db_manager: Manager for tenant database connections.
            embedding_service: Service for generating text embeddings.
            qdrant_client: Client for Qdrant vector database.
        """
        self._tenant_db = tenant_db_manager
        self._embedding = embedding_service
        self._qdrant = qdrant_client

        # Initialize layers
        self.episodic = EpisodicMemoryLayer(
            tenant_db_manager=tenant_db_manager,
            embedding_service=embedding_service,
            qdrant_client=qdrant_client,
        )

        self.semantic = SemanticMemoryLayer(
            tenant_db_manager=tenant_db_manager,
        )

        self.procedural = ProceduralMemoryLayer(
            tenant_db_manager=tenant_db_manager,
        )

        self.associative = AssociativeMemoryLayer(
            tenant_db_manager=tenant_db_manager,
            embedding_service=embedding_service,
            qdrant_client=qdrant_client,
        )

        logger.info("MemoryManager initialized with all four layers")

    async def ensure_collections(self, tenant_code: str) -> None:
        """Ensure all Qdrant collections exist for a tenant.

        Should be called when a new tenant is provisioned.

        Args:
            tenant_code: Unique tenant identifier.
        """
        await self.episodic.ensure_collection_exists(tenant_code)
        await self.associative.ensure_collection_exists(tenant_code)
        logger.info("Ensured memory collections exist for tenant %s", tenant_code)

    async def get_full_context(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic: str | None = None,
        subject: str | None = None,
        recent_episodes_limit: int = 10,
        important_episodes_limit: int = 5,
        min_pattern_samples: int = 5,
        include_diagnostic: bool = True,
        session: AsyncSession | None = None,
    ) -> FullMemoryContext:
        """Get complete memory context for AI tutor personalization.

        Retrieves data from all four memory layers plus diagnostic indicators
        to provide a comprehensive understanding of the student's learning
        history, knowledge state, preferences, interests, and any learning
        difficulty indicators.

        Mastery loading strategy:
        1. If topic is provided (and not "random"): Load topic-specific mastery
        2. If subject is provided but no topic: Load subject average mastery
        3. If topic is "random" or empty: Use overall mastery from MasteryOverview

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic: Topic full code for specific topic mastery. Use "random" for mixed practice.
            subject: Subject full code for subject-level mastery when no specific topic.
            recent_episodes_limit: Max recent episodic memories.
            important_episodes_limit: Max important episodic memories.
            min_pattern_samples: Min observations for procedural patterns.
            include_diagnostic: Whether to include diagnostic data (default True).
            session: Optional database session for transaction sharing.

        Returns:
            FullMemoryContext with data from all layers.
            - topic_mastery: Set when specific topic is provided
            - subject_mastery: Set when subject is provided but no topic
            - For random/mixed practice, use semantic.overall_mastery
        """
        logger.debug("Getting full memory context for student %s", student_id)

        # Get recent and important episodic memories
        recent_episodes = await self.episodic.get_recent(
            tenant_code=tenant_code,
            student_id=student_id,
            limit=recent_episodes_limit,
            session=session,
        )

        important_episodes = await self.episodic.get_important_memories(
            tenant_code=tenant_code,
            student_id=student_id,
            min_importance=0.7,
            limit=important_episodes_limit,
            session=session,
        )

        # Merge and deduplicate episodes
        episode_ids = set()
        all_episodes: list[EpisodicMemoryResponse] = []
        for episode in recent_episodes + important_episodes:
            if episode.id not in episode_ids:
                episode_ids.add(episode.id)
                all_episodes.append(episode)

        # Get semantic mastery overview
        mastery = await self.semantic.get_mastery_overview(
            tenant_code=tenant_code,
            student_id=student_id,
            session=session,
        )

        # Get procedural learning patterns
        patterns = await self.procedural.get_learning_patterns(
            tenant_code=tenant_code,
            student_id=student_id,
            min_samples=min_pattern_samples,
            session=session,
        )

        # Get associative interests
        interests = await self.associative.get_student_interests(
            tenant_code=tenant_code,
            student_id=student_id,
            session=session,
        )

        # Get diagnostic context (optional, gracefully handles missing data)
        diagnostic_context: DiagnosticContext | None = None
        if include_diagnostic:
            diagnostic_context = await self._get_diagnostic_context(
                tenant_code=tenant_code,
                student_id=student_id,
                session=session,
            )

        # Load mastery based on context:
        # 1. topic provided (not "random") → topic-specific mastery
        # 2. subject provided, no topic → subject average mastery
        # 3. random/empty → use overall mastery from MasteryOverview
        topic_mastery_data = None
        subject_mastery_value = None

        is_random_mode = topic and topic.lower() == "random"

        if topic and not is_random_mode:
            # Scenario 1: Specific topic - load topic mastery with all details
            try:
                topic_mastery_data = await self.semantic.get_by_entity_full_code(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    entity_full_code=topic,
                    session=session,
                )
                if topic_mastery_data:
                    logger.debug(
                        "Loaded topic-specific mastery: topic=%s, mastery=%.2f, "
                        "attempts=%d, streak=%d",
                        topic,
                        topic_mastery_data.mastery_level,
                        topic_mastery_data.attempts_total,
                        topic_mastery_data.current_streak,
                    )
                else:
                    logger.debug("No existing mastery for topic %s (new topic)", topic)
            except Exception as e:
                logger.warning("Failed to load topic mastery for %s: %s", topic, str(e))

        elif subject and not topic:
            # Scenario 2: Subject-level practice - calculate subject average
            try:
                subject_mastery_value = await self.semantic.get_subject_average_mastery(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    subject_full_code=subject,
                    session=session,
                )
                logger.debug(
                    "Loaded subject average mastery: subject=%s, mastery=%.2f",
                    subject,
                    subject_mastery_value if subject_mastery_value else 0.5,
                )
            except Exception as e:
                logger.warning("Failed to load subject mastery for %s: %s", subject, str(e))

        # Scenario 3: Random/mixed mode - overall_mastery from MasteryOverview is used
        # No additional loading needed, semantic.overall_mastery is already available

        return FullMemoryContext(
            student_id=student_id,
            episodic=all_episodes,
            semantic=mastery,
            procedural=patterns,
            associative=interests,
            diagnostic=diagnostic_context,
            topic_mastery=topic_mastery_data,
            subject_mastery=subject_mastery_value,
            retrieved_at=datetime.now(timezone.utc),
        )

    def get_full_context_sync(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic: str | None = None,
        recent_episodes_limit: int = 10,
        min_pattern_samples: int = 5,
    ) -> FullMemoryContext:
        """Get complete memory context for AI tutor personalization (synchronous version).

        This synchronous version is designed for use in LangGraph workflow nodes
        that run in thread pool executors where async greenlet context is not available.

        Retrieves data from all four memory layers to provide a comprehensive
        understanding of the student's learning history, knowledge state,
        preferences, and interests.

        Note: This version does not include diagnostic context as that requires
        async database operations. Use the async version for complete context.

        Mastery loading strategy:
        1. If topic is provided (and not "random"): Load topic-specific mastery
        2. Otherwise: Use overall mastery from MasteryOverview

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic: Topic full code for specific topic mastery. Use "random" for mixed practice.
            recent_episodes_limit: Max recent episodic memories.
            min_pattern_samples: Min observations for procedural patterns.

        Returns:
            FullMemoryContext with data from all layers (excluding diagnostic).
            - topic_mastery: Set when specific topic is provided
            - For random/mixed practice, use semantic.overall_mastery
        """
        logger.debug("Getting full memory context (sync) for student %s", student_id)

        # Get recent episodic memories
        recent_episodes = self.episodic.get_recent_sync(
            tenant_code=tenant_code,
            student_id=student_id,
            limit=recent_episodes_limit,
        )

        # Get semantic mastery overview
        mastery = self.semantic.get_mastery_overview_sync(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Get procedural learning patterns
        patterns = self.procedural.get_learning_patterns_sync(
            tenant_code=tenant_code,
            student_id=student_id,
            min_samples=min_pattern_samples,
        )

        # Get associative interests
        interests = self.associative.get_student_interests_sync(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Load topic-specific mastery if requested
        topic_mastery_data = None
        is_random_mode = topic and topic.lower() == "random"

        if topic and not is_random_mode:
            try:
                topic_mastery_data = self.semantic.get_by_entity_full_code_sync(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    entity_full_code=topic,
                )
                if topic_mastery_data:
                    logger.debug(
                        "Loaded topic-specific mastery (sync): topic=%s, mastery=%.2f, "
                        "attempts=%d, streak=%d",
                        topic,
                        topic_mastery_data.mastery_level,
                        topic_mastery_data.attempts_total,
                        topic_mastery_data.current_streak,
                    )
                else:
                    logger.debug("No existing mastery for topic %s (new topic)", topic)
            except Exception as e:
                logger.warning("Failed to load topic mastery for %s: %s", topic, str(e))

        return FullMemoryContext(
            student_id=student_id,
            episodic=recent_episodes,
            semantic=mastery,
            procedural=patterns,
            associative=interests,
            diagnostic=None,  # Diagnostic not available in sync version
            topic_mastery=topic_mastery_data,
            subject_mastery=None,  # Not implemented in sync version
            retrieved_at=datetime.now(timezone.utc),
        )

    async def _get_diagnostic_context(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        session: AsyncSession | None = None,
    ) -> DiagnosticContext | None:
        """Get diagnostic context from the latest completed scan.

        Fetches the most recent completed diagnostic scan for the student
        and extracts indicator scores into a DiagnosticContext.

        This method is designed to be fault-tolerant:
        - Returns None if no scan exists (new student)
        - Returns None if scan is in progress or failed
        - Returns None on any error (logged but not raised)
        - Never raises exceptions to caller

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            session: Optional database session for transaction sharing.

        Returns:
            DiagnosticContext with indicator scores, or None if unavailable.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from src.infrastructure.database.models.tenant.diagnostic import (
            DiagnosticScan,
        )

        async def _execute(db: AsyncSession) -> DiagnosticContext | None:
            # Get the latest completed scan
            result = await db.execute(
                select(DiagnosticScan)
                .where(
                    DiagnosticScan.student_id == str(student_id),
                    DiagnosticScan.status == "completed",
                )
                .options(selectinload(DiagnosticScan.indicators))
                .order_by(DiagnosticScan.completed_at.desc())
                .limit(1)
            )
            scan = result.scalar_one_or_none()

            if not scan:
                # No completed scan exists - this is normal for new students
                logger.debug(
                    "No diagnostic scan found for student %s", student_id
                )
                return None

            # Extract indicator scores
            indicator_scores = {
                "dyslexia": 0.0,
                "dyscalculia": 0.0,
                "attention": 0.0,
                "auditory": 0.0,
                "visual": 0.0,
            }

            for indicator in scan.indicators:
                indicator_type = indicator.indicator_type
                if indicator_type in indicator_scores:
                    indicator_scores[indicator_type] = float(
                        indicator.risk_score or 0.0
                    )

            return DiagnosticContext(
                dyslexia_risk=indicator_scores["dyslexia"],
                dyscalculia_risk=indicator_scores["dyscalculia"],
                attention_risk=indicator_scores["attention"],
                auditory_risk=indicator_scores["auditory"],
                visual_risk=indicator_scores["visual"],
                last_scan_id=scan.id,
                last_scan_at=scan.completed_at,
                has_concerns=scan.has_concerns if scan.has_concerns else False,
            )

        try:
            if session:
                return await _execute(session)
            async with self._tenant_db.get_session(tenant_code) as db:
                return await _execute(db)
        except Exception as e:
            # Log error but don't fail - diagnostic is optional
            logger.warning(
                "Failed to get diagnostic context for student %s: %s",
                student_id,
                str(e),
            )
            return None

    async def get_context_for_topic(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        topic_full_code: str,
        topic_description: str,
    ) -> dict[str, Any]:
        """Get memory context relevant to a specific topic.

        Retrieves topic-specific memories for personalized tutoring.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            topic_description: Description for semantic search.

        Returns:
            Dictionary with topic-relevant memories from each layer.
        """
        # Get episodic memories related to topic
        topic_episodes = await self.episodic.search_with_params(
            tenant_code=tenant_code,
            student_id=student_id,
            params=EpisodicMemorySearchParams(
                topic_full_code=topic_full_code,
                limit=10,
            ),
        )

        # Search for semantically similar episodes
        similar_episodes = await self.episodic.search(
            tenant_code=tenant_code,
            student_id=student_id,
            query=topic_description,
            limit=5,
        )

        # Get semantic memory for topic
        from src.models.memory import EntityType

        topic_mastery = await self.semantic.get_by_entity(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
        )

        # Get effective strategies for topic
        topic_strategies = await self.procedural.get_effective_strategies_for_topic(
            tenant_code=tenant_code,
            student_id=student_id,
            topic_full_code=topic_full_code,
        )

        # Find relevant interests/analogies
        relevant_associations = await self.associative.find_relevant_for_topic(
            tenant_code=tenant_code,
            student_id=student_id,
            topic_description=topic_description,
        )

        return {
            "topic_episodes": topic_episodes,
            "similar_episodes": [ep for ep, _ in similar_episodes],
            "topic_mastery": topic_mastery,
            "effective_strategies": topic_strategies,
            "relevant_associations": [assoc for assoc, _ in relevant_associations],
        }

    async def search_all(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        """Search across all memory layers.

        Performs semantic search in episodic and associative layers,
        and keyword matching in semantic and procedural layers.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            request: Search request with query and layer filters.

        Returns:
            MemorySearchResponse with results from each layer.
        """
        episodic_results: list[EpisodicMemoryResponse] = []
        semantic_results: list[SemanticMemoryResponse] = []
        associative_results: list[AssociativeMemoryResponse] = []

        # Search episodic (vector search)
        if "episodic" in request.layers:
            results = await self.episodic.search(
                tenant_code=tenant_code,
                student_id=student_id,
                query=request.query,
                limit=request.limit_per_layer,
            )
            episodic_results = [memory for memory, _ in results]

        # Search semantic (structured query - search by entity name would require joins)
        # For now, return recent memories as semantic doesn't have text content
        if "semantic" in request.layers:
            semantic_results = await self.semantic.get_all_for_student(
                tenant_code=tenant_code,
                student_id=student_id,
            )
            semantic_results = semantic_results[: request.limit_per_layer]

        # Search associative (vector search)
        if "associative" in request.layers:
            results = await self.associative.search(
                tenant_code=tenant_code,
                student_id=student_id,
                query=request.query,
                limit=request.limit_per_layer,
            )
            associative_results = [memory for memory, _ in results]

        total = (
            len(episodic_results)
            + len(semantic_results)
            + len(associative_results)
        )

        return MemorySearchResponse(
            query=request.query,
            episodic_results=episodic_results,
            semantic_results=semantic_results,
            associative_results=associative_results,
            total_results=total,
        )

    async def get_learning_summary(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Get a high-level learning summary for a student.

        Provides statistics and insights across all memory layers.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.

        Returns:
            Dictionary with learning statistics and insights.
        """
        # Get mastery overview
        mastery = await self.semantic.get_mastery_overview(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Get event type statistics
        event_stats = await self.episodic.get_event_type_stats(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Count memories
        episodic_count = await self.episodic.count_by_student(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        associative_count = await self.associative.count_by_student(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Get learning patterns
        patterns = await self.procedural.get_learning_patterns(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Calculate engagement ratio
        positive_events = sum(
            event_stats.get(et.value, 0)
            for et in [
                EpisodicEventType.BREAKTHROUGH,
                EpisodicEventType.MASTERY,
                EpisodicEventType.CORRECT_ANSWER,
                EpisodicEventType.ENGAGEMENT,
            ]
        )
        negative_events = sum(
            event_stats.get(et.value, 0)
            for et in [
                EpisodicEventType.STRUGGLE,
                EpisodicEventType.CONFUSION,
                EpisodicEventType.FRUSTRATION,
                EpisodicEventType.INCORRECT_ANSWER,
            ]
        )
        total_events = positive_events + negative_events
        engagement_ratio = (
            positive_events / total_events if total_events > 0 else 0.5
        )

        return {
            "student_id": str(student_id),
            "mastery": {
                "overall": mastery.overall_mastery,
                "topics_mastered": mastery.topics_mastered,
                "topics_learning": mastery.topics_learning,
                "topics_struggling": mastery.topics_struggling,
                "total_topics": mastery.total_topics,
            },
            "engagement": {
                "total_episodes": episodic_count,
                "event_distribution": event_stats,
                "positive_ratio": round(engagement_ratio, 3),
            },
            "personalization": {
                "interests_recorded": associative_count,
                "preferred_time": patterns.best_time_of_day,
                "preferred_format": patterns.preferred_content_format,
                "vark_profile": (
                    patterns.vark_profile.model_dump()
                    if patterns.vark_profile
                    else None
                ),
            },
        }

    async def record_learning_event(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        event_type: str,
        topic: str,
        data: dict[str, Any] | None = None,
        session_id: str | uuid.UUID | None = None,
        importance: float = 0.5,
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> EpisodicMemoryResponse:
        """Record a learning event in episodic memory.

        This is a convenience method that wraps EpisodicMemoryLayer.store()
        with appropriate defaults for learning events.

        Supported event types:
        - Practice: question_answered, hint_used
        - Learning states: struggle, breakthrough, confusion, mastery
        - Sessions: learning_session_completed, learning_session_progress,
                   tutoring_session_completed, tutoring_interaction
        - Workflows: companion_session, companion_handoff,
                    practice_helper_session, practice_helper_escalated
        - Gaming: game_move, game_completed

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            event_type: Type of event.
            topic: Topic being studied.
            data: Additional event data.
            session_id: Related practice session ID.
            importance: Event importance (0-1).
            topic_full_code: Full topic code for curriculum reference.
            session: Optional database session for transaction sharing.

        Returns:
            Created episodic memory record.
        """
        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        # Convert session_id to UUID if string
        session_uuid = None
        if session_id:
            session_uuid = (
                uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            )

        # Extract common data fields
        data = data or {}
        is_correct = data.get("is_correct", False)
        score = data.get("score", 0.0)

        # Map event type string to enum
        event_type_map = {
            # Practice events
            "question_answered": (
                EpisodicEventType.CORRECT_ANSWER
                if is_correct
                else EpisodicEventType.INCORRECT_ANSWER
            ),
            "hint_used": EpisodicEventType.HINT_USED,

            # Learning state events
            "struggle": EpisodicEventType.STRUGGLE,
            "breakthrough": EpisodicEventType.BREAKTHROUGH,
            "confusion": EpisodicEventType.CONFUSION,
            "mastery": EpisodicEventType.MASTERY,
            "struggle_overcome": EpisodicEventType.STRUGGLE_OVERCOME,
            "concept_learned": EpisodicEventType.CONCEPT_LEARNED,

            # Session completion events
            "learning_session_completed": EpisodicEventType.SESSION_COMPLETED,
            "tutoring_session_completed": EpisodicEventType.SESSION_COMPLETED,
            "session_completion": EpisodicEventType.SESSION_COMPLETED,

            # Session progress events
            "learning_session_progress": EpisodicEventType.SESSION_PROGRESS,
            "tutoring_interaction": EpisodicEventType.SESSION_PROGRESS,

            # Workflow events - companion
            "companion_session": EpisodicEventType.ENGAGEMENT,
            "companion_handoff": EpisodicEventType.HANDOFF,

            # Workflow events - practice helper
            "practice_helper_session": EpisodicEventType.ENGAGEMENT,
            "practice_helper_escalated": EpisodicEventType.HELPER_ESCALATED,

            # Gaming events
            "game_move": EpisodicEventType.GAME_MOVE,
            "game_completed": EpisodicEventType.GAME_COMPLETED,
        }

        episodic_event_type = event_type_map.get(
            event_type, EpisodicEventType.ENGAGEMENT
        )

        # Build summary based on event type
        summary = self._build_event_summary(
            event_type=event_type,
            episodic_type=episodic_event_type,
            topic=topic,
            data=data,
        )

        # Adjust importance based on event type and result
        importance = self._calculate_event_importance(
            event_type=event_type,
            episodic_type=episodic_event_type,
            data=data,
            base_importance=importance,
        )

        return await self.episodic.store(
            tenant_code=tenant_code,
            student_id=student_id,
            event_type=episodic_event_type,
            summary=summary,
            details=data,
            importance=importance,
            session_id=session_uuid,
            topic_full_code=topic_full_code,
            session=session,
        )

    def _build_event_summary(
        self,
        event_type: str,
        episodic_type: EpisodicEventType,
        topic: str,
        data: dict[str, Any],
    ) -> str:
        """Build a human-readable summary for an event.

        Args:
            event_type: Original event type string.
            episodic_type: Mapped episodic event type.
            topic: Topic name.
            data: Event data dictionary.

        Returns:
            Human-readable summary string.
        """
        # Session completion events
        if episodic_type == EpisodicEventType.SESSION_COMPLETED:
            understanding = data.get("understanding_progress", 0.0)
            reason = data.get("completion_reason", "completed")
            turn_count = data.get("turn_count", 0)
            understood = data.get("understood", False)

            if understood:
                return f"Completed learning session on {topic} with understanding ({understanding:.0%})"
            elif reason == "user_ended":
                return f"Ended learning session on {topic} after {turn_count} turns (understanding: {understanding:.0%})"
            else:
                return f"Session on {topic} ended: {reason} (understanding: {understanding:.0%})"

        # Session progress events
        if episodic_type == EpisodicEventType.SESSION_PROGRESS:
            understanding = data.get("understanding_progress", 0.0)
            mode = data.get("mode", data.get("learning_mode", "learning"))
            return f"Learning progress on {topic}: {understanding:.0%} in {mode} mode"

        # Question answered events
        if episodic_type in (EpisodicEventType.CORRECT_ANSWER, EpisodicEventType.INCORRECT_ANSWER):
            is_correct = data.get("is_correct", False)
            score = data.get("score", 0.0)
            return f"{'Correct' if is_correct else 'Incorrect'} answer on {topic} (score: {score:.0%})"

        # Handoff events
        if episodic_type == EpisodicEventType.HANDOFF:
            target = data.get("target_workflow", data.get("target", "unknown"))
            return f"Handoff to {target} workflow for {topic}"

        # Helper escalation events
        if episodic_type == EpisodicEventType.HELPER_ESCALATED:
            escalated_to = data.get("escalated_to", "learning_tutor")
            mode_escalations = data.get("mode_escalations", 0)
            return f"Practice helper escalated to {escalated_to} after {mode_escalations} mode changes on {topic}"

        # Gaming events
        if episodic_type == EpisodicEventType.GAME_MOVE:
            game_type = data.get("game_type", "game")
            move_quality = data.get("move_quality", "unknown")
            return f"Made a {move_quality} move in {game_type}"

        if episodic_type == EpisodicEventType.GAME_COMPLETED:
            game_type = data.get("game_type", "game")
            result = data.get("result", "completed")
            return f"Completed {game_type} game: {result}"

        # Learning state events
        state_summaries = {
            EpisodicEventType.STRUGGLE: f"Struggled with {topic}",
            EpisodicEventType.BREAKTHROUGH: f"Breakthrough moment on {topic}",
            EpisodicEventType.CONFUSION: f"Expressed confusion about {topic}",
            EpisodicEventType.MASTERY: f"Demonstrated mastery of {topic}",
            EpisodicEventType.STRUGGLE_OVERCOME: f"Overcame struggle with {topic}",
            EpisodicEventType.CONCEPT_LEARNED: f"Learned concept in {topic}",
            EpisodicEventType.HINT_USED: f"Used hint while studying {topic}",
            EpisodicEventType.FRUSTRATION: f"Showed frustration with {topic}",
            EpisodicEventType.INTEREST: f"Showed interest in {topic}",
        }

        if episodic_type in state_summaries:
            return state_summaries[episodic_type]

        # Default engagement summary
        return f"Engaged with {topic}"

    def _calculate_event_importance(
        self,
        event_type: str,
        episodic_type: EpisodicEventType,
        data: dict[str, Any],
        base_importance: float,
    ) -> float:
        """Calculate importance score for an event.

        Args:
            event_type: Original event type string.
            episodic_type: Mapped episodic event type.
            data: Event data dictionary.
            base_importance: Base importance from caller.

        Returns:
            Calculated importance score (0.0-1.0).
        """
        # High importance events (0.7-0.9)
        high_importance_types = {
            EpisodicEventType.BREAKTHROUGH,
            EpisodicEventType.MASTERY,
            EpisodicEventType.SESSION_COMPLETED,
            EpisodicEventType.STRUGGLE_OVERCOME,
            EpisodicEventType.CONCEPT_LEARNED,
        }

        if episodic_type in high_importance_types:
            # Further adjust based on understanding
            understanding = data.get("understanding_progress", 0.5)
            if understanding >= 0.8:
                return max(base_importance, 0.85)
            elif understanding >= 0.5:
                return max(base_importance, 0.75)
            return max(base_importance, 0.7)

        # Medium importance events (0.5-0.7)
        medium_importance_types = {
            EpisodicEventType.STRUGGLE,
            EpisodicEventType.CONFUSION,
            EpisodicEventType.HELPER_ESCALATED,
            EpisodicEventType.HANDOFF,
            EpisodicEventType.GAME_COMPLETED,
        }

        if episodic_type in medium_importance_types:
            return max(base_importance, 0.6)

        # Question answered - adjust by score
        if episodic_type in (EpisodicEventType.CORRECT_ANSWER, EpisodicEventType.INCORRECT_ANSWER):
            is_correct = data.get("is_correct", False)
            score = data.get("score", 0.0)
            if is_correct and score >= 0.9:
                return max(base_importance, 0.7)
            elif not is_correct and score < 0.3:
                return max(base_importance, 0.6)

        # Default - use base importance
        return base_importance

    async def record_practice_attempt(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic_full_code: str,
        is_correct: bool,
        time_seconds: int = 0,
        session: AsyncSession | None = None,
    ) -> SemanticMemoryResponse | None:
        """Record a practice attempt for a topic.

        Records the attempt in semantic memory, updating:
        - attempts_total (incremented)
        - attempts_correct (incremented if correct)
        - current_streak (incremented if correct, reset if incorrect)
        - best_streak (updated if current > best)
        - total_time_seconds (accumulated)
        - mastery_level (recalculated based on accuracy, streaks)
        - last_practiced_at, last_correct_at, last_incorrect_at

        This is the preferred method for practice sessions as it maintains
        proper attempt tracking and mastery calculation.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MATHS.Y5.UNIT.TOPIC").
            is_correct: Whether the answer was correct.
            time_seconds: Time spent on this attempt.
            session: Optional database session for transaction sharing.

        Returns:
            Updated semantic memory record, or None if operation fails.
        """
        from src.models.memory import EntityType

        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        if not topic_full_code:
            logger.warning("topic_full_code required for record_practice_attempt")
            return None

        return await self.semantic.record_attempt(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
            is_correct=is_correct,
            time_seconds=time_seconds,
            update_mastery=True,
            session=session,
        )

    async def update_topic_mastery(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic: str,
        score_delta: float,
        topic_full_code: str | None = None,
    ) -> SemanticMemoryResponse | None:
        """Update topic mastery by a delta value (legacy method).

        DEPRECATED: Prefer record_practice_attempt() for practice sessions
        as it properly tracks attempts, streaks, and time.

        This method directly adjusts mastery without tracking attempts.
        Use only for admin overrides or external mastery adjustments.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic: Topic name (for logging).
            score_delta: Amount to add/subtract from mastery.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            Updated semantic memory record, or None if operation fails.
        """
        from src.models.memory import EntityType

        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        if not topic_full_code:
            logger.warning("topic_full_code required for update_topic_mastery")
            return None

        # Get current mastery
        current = await self.semantic.get_by_entity(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
        )

        if current:
            # Calculate new mastery
            new_mastery = max(0.0, min(1.0, current.mastery_level + score_delta))

            # Update via set_mastery
            return await self.semantic.set_mastery(
                tenant_code=tenant_code,
                student_id=student_id,
                entity_type=EntityType.TOPIC,
                entity_full_code=topic_full_code,
                mastery_level=new_mastery,
                reason=f"Delta update: {score_delta:+.2f}",
            )
        else:
            # Create new record with initial mastery based on delta direction
            initial_mastery = max(0.0, min(1.0, 0.5 + score_delta))

            return await self.semantic.set_mastery(
                tenant_code=tenant_code,
                student_id=student_id,
                entity_type=EntityType.TOPIC,
                entity_full_code=topic_full_code,
                mastery_level=initial_mastery,
                reason=f"Initial mastery from delta: {score_delta:+.2f}",
            )

    async def record_learning_session_completion(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic_full_code: str,
        understanding_progress: float,
        session_completed: bool = True,
        understanding_verified: bool = False,
        verification_method: str | None = None,
    ) -> SemanticMemoryResponse | None:
        """Record a learning tutor session completion and update mastery.

        Updates topic mastery based on the student's understanding progress
        achieved during the learning session. Unlike practice attempts which
        track correct/incorrect, learning sessions track understanding level.

        When understanding is AI-verified (via comprehension evaluation),
        mastery gains are increased and the verification is recorded.

        Mastery update logic:
        - Verified understanding >= 0.8: Increase mastery by 0.20-0.25
        - understanding_progress >= 0.7: Increase mastery by up to 0.15
        - understanding_progress >= 0.4: Increase mastery by up to 0.08
        - understanding_progress >= 0.2: Small increase of 0.03 (effort acknowledged)
        - understanding_progress < 0.2: No change (session too short/incomplete)

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MATHS.Y2.UNIT.TOPIC").
            understanding_progress: Understanding achieved (0.0-1.0).
            session_completed: Whether session was completed (vs abandoned).
            understanding_verified: Whether understanding was AI-verified.
            verification_method: Method used for verification (e.g., "comprehension_evaluation").

        Returns:
            Updated semantic memory record, or None if operation fails.
        """
        from src.models.memory import EntityType

        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        if not topic_full_code:
            logger.warning("topic_full_code required for record_learning_session_completion")
            return None

        # Only update mastery if session was completed with some understanding
        if not session_completed or understanding_progress < 0.2:
            logger.debug(
                "Skipping mastery update: completed=%s, understanding=%.2f",
                session_completed,
                understanding_progress,
            )
            return None

        # Calculate mastery delta based on understanding progress
        # Higher understanding = larger mastery increase
        # Verified understanding gets bonus multiplier
        if understanding_verified and understanding_progress >= 0.8:
            # AI-verified strong understanding - maximum mastery gain
            mastery_delta = 0.20 + (understanding_progress - 0.8) * 0.25
        elif understanding_progress >= 0.7:
            # Strong understanding - significant mastery gain
            mastery_delta = 0.10 + (understanding_progress - 0.7) * 0.15
            if understanding_verified:
                mastery_delta *= 1.3  # 30% bonus for verified
        elif understanding_progress >= 0.4:
            # Moderate understanding - modest mastery gain
            mastery_delta = 0.05 + (understanding_progress - 0.4) * 0.10
        else:
            # Low understanding but completed - small acknowledgment
            mastery_delta = 0.03

        # Get current mastery
        current = await self.semantic.get_by_entity(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
        )

        if current:
            # Update existing mastery
            new_mastery = max(0.0, min(1.0, current.mastery_level + mastery_delta))
            reason = f"Learning session: understanding={understanding_progress:.0%}, delta={mastery_delta:+.2f}"
            if understanding_verified:
                reason += f", verified via {verification_method or 'comprehension_evaluation'}"
        else:
            # Create new record - start from understanding progress
            new_mastery = max(0.0, min(1.0, understanding_progress * 0.5))
            if understanding_verified:
                new_mastery = max(new_mastery, understanding_progress * 0.7)  # Higher initial for verified
            reason = f"Learning session (first): understanding={understanding_progress:.0%}"
            if understanding_verified:
                reason += " (verified)"

        result = await self.semantic.set_mastery(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
            mastery_level=new_mastery,
            reason=reason,
        )

        logger.info(
            "Updated mastery for topic %s: understanding=%.2f, new_mastery=%.2f",
            topic_full_code,
            understanding_progress,
            new_mastery,
        )

        return result

    async def get_topic_mastery(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic: str,
        topic_full_code: str | None = None,
    ) -> float | None:
        """Get current mastery level for a topic.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic: Topic name (for logging).
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            Current mastery level (0.0-1.0), or None if no record exists.
        """
        from src.models.memory import EntityType

        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        if not topic_full_code:
            logger.warning("topic_full_code required for get_topic_mastery")
            return None

        # Get semantic memory for topic
        memory = await self.semantic.get_by_entity(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
        )

        if memory:
            return memory.mastery_level

        return None

    async def set_topic_mastery(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic: str,
        mastery: float,
        topic_full_code: str | None = None,
    ) -> SemanticMemoryResponse | None:
        """Set mastery level for a topic.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic: Topic name (for logging).
            mastery: New mastery level (0.0-1.0).
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            Updated semantic memory record, or None if operation fails.
        """
        from src.models.memory import EntityType

        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        if not topic_full_code:
            logger.warning("topic_full_code required for set_topic_mastery")
            return None

        # Clamp mastery to valid range
        mastery = max(0.0, min(1.0, mastery))

        # Set mastery via semantic layer
        return await self.semantic.set_mastery(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
            mastery_level=mastery,
            reason="Assessment mastery update",
        )

    async def record_procedural_observation(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        observation: dict[str, Any],
        topic_full_code: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Record learning pattern observation in procedural memory.

        This method extracts strategy observations from practice or learning tutor
        session data and records them in procedural memory. The observations are
        used to build the student's learning profile (VARK, optimal times, etc.).

        Strategies recorded:
        - TIME_OF_DAY: When the student learns best (morning/afternoon/evening/night)
        - CONTENT_FORMAT: What format works best (text/visual/interactive)
        - HINT_USAGE: Whether hints help learning (with_hints/no_hints)
        - LEARNING_MODE: Which learning modes are most effective (learning tutor only)

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            observation: Observation data containing:
                For practice sessions:
                    - time_of_day: str (morning/afternoon/evening/night)
                    - content_format: str (text/visual/interactive)
                    - hints_used: int (number of hints used)
                    - is_correct: bool (whether answer was correct)
                    - score: float (answer score 0-1)
                    - topic: str (topic being practiced)
                For learning tutor sessions:
                    - session_type: "learning_tutor"
                    - time_of_day: str (morning/afternoon/evening/night)
                    - learning_mode: str (discovery/explanation/worked_example/guided_practice/assessment)
                    - understanding_progress: float (0.0-1.0)
                    - topic: str (topic being learned)
            topic_full_code: Full topic code for topic-specific observations
                (e.g., "UK-NC-2014.MATHS.Y5.MATHS-Y5-FRAC.MATHS-Y5-FRAC-CALC").
            session: Optional database session for transaction sharing.
        """
        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        session_type = observation.get("session_type", "practice")
        time_of_day = observation.get("time_of_day")

        # Determine effectiveness based on session type
        if session_type == "learning_tutor":
            # For learning tutor, use understanding_progress (> 0.3 is considered effective)
            understanding_progress = observation.get("understanding_progress", 0.0)
            was_effective = understanding_progress >= 0.3
            learning_mode = observation.get("learning_mode")
        else:
            # For practice sessions, use is_correct (original behavior)
            content_format = observation.get("content_format")
            hints_used = observation.get("hints_used", 0)
            is_correct = observation.get("is_correct", False)
            was_effective = is_correct
            learning_mode = None

        # Record time of day strategy
        if time_of_day:
            try:
                await self.procedural.record_observation(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    strategy_type=StrategyType.TIME_OF_DAY,
                    strategy_value=time_of_day,
                    was_effective=was_effective,
                    topic_full_code=topic_full_code,
                    session=session,
                )
            except Exception as e:
                logger.warning(
                    "Failed to record time_of_day observation: %s", str(e)
                )

        # Session-type specific strategies
        if session_type == "learning_tutor":
            # Record learning mode strategy for learning tutor sessions
            if learning_mode:
                try:
                    await self.procedural.record_observation(
                        tenant_code=tenant_code,
                        student_id=student_id,
                        strategy_type=StrategyType.LEARNING_MODE,
                        strategy_value=learning_mode,
                        was_effective=was_effective,
                        topic_full_code=topic_full_code,
                        session=session,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to record learning_mode observation: %s", str(e)
                    )

            logger.debug(
                "Recorded procedural observations for student %s (learning_tutor): "
                "time=%s, mode=%s, understanding=%.2f, effective=%s",
                student_id,
                time_of_day,
                learning_mode,
                observation.get("understanding_progress", 0.0),
                was_effective,
            )
        else:
            # Record content format strategy for practice sessions
            content_format = observation.get("content_format")
            if content_format:
                try:
                    await self.procedural.record_observation(
                        tenant_code=tenant_code,
                        student_id=student_id,
                        strategy_type=StrategyType.CONTENT_FORMAT,
                        strategy_value=content_format,
                        was_effective=was_effective,
                        topic_full_code=topic_full_code,
                        session=session,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to record content_format observation: %s", str(e)
                    )

            # Record hint usage strategy for practice sessions
            # If hints were used and answer was correct, hints were helpful
            # If no hints used and answer was correct, no hints needed
            hints_used = observation.get("hints_used", 0)
            hint_value = "with_hints" if hints_used > 0 else "no_hints"
            hint_effective = was_effective  # Hints help if answer was correct
            try:
                await self.procedural.record_observation(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    strategy_type=StrategyType.HINT_USAGE,
                    strategy_value=hint_value,
                    was_effective=hint_effective,
                    topic_full_code=topic_full_code,
                    session=session,
                )
            except Exception as e:
                logger.warning(
                    "Failed to record hint_usage observation: %s", str(e)
                )

            logger.debug(
                "Recorded procedural observations for student %s (practice): "
                "time=%s, format=%s, hints=%s, effective=%s",
                student_id,
                time_of_day,
                content_format,
                hint_value,
                was_effective,
            )

    async def cleanup_old_memories(
        self,
        tenant_code: str,
        student_id: uuid.UUID,
        max_episodic_age_days: int = 365,
        min_importance_to_keep: float = 0.7,
    ) -> dict[str, int]:
        """Clean up old and low-importance memories.

        Removes old episodic memories while preserving important ones.
        This helps manage storage and maintain relevant context.

        Args:
            tenant_code: Unique tenant identifier.
            student_id: Student's unique identifier.
            max_episodic_age_days: Maximum age for episodic memories.
            min_importance_to_keep: Keep memories above this importance.

        Returns:
            Dictionary with counts of cleaned up memories.
        """
        # This would require implementing delete methods with age/importance filters
        # For now, return empty counts as placeholder
        logger.info(
            "Cleanup requested for student %s (age=%d days, min_importance=%.2f)",
            student_id,
            max_episodic_age_days,
            min_importance_to_keep,
        )

        return {
            "episodic_deleted": 0,
            "associative_deleted": 0,
        }

    async def record_comprehension_evaluation(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic_full_code: str,
        evaluation_result: dict[str, Any],
        session_id: str | uuid.UUID | None = None,
    ) -> EpisodicMemoryResponse | None:
        """Record a comprehension evaluation result as episodic memory.

        Creates a rich episodic memory record from a ComprehensionEvaluationResult
        that captures the student's demonstrated understanding, misconceptions,
        and whether the understanding was verified.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic_full_code: Topic full code.
            evaluation_result: ComprehensionEvaluationResult as dict.
            session_id: Related learning session ID.

        Returns:
            EpisodicMemoryResponse for the stored evaluation.
        """
        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        # Convert session_id to UUID if string
        session_uuid = None
        if session_id:
            session_uuid = (
                uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            )

        try:
            return await self.episodic.store_comprehension_evaluation(
                tenant_code=tenant_code,
                student_id=student_id,
                topic_full_code=topic_full_code,
                evaluation_result=evaluation_result,
                session_id=session_uuid,
            )
        except Exception as e:
            logger.error(
                "Failed to record comprehension evaluation: %s", str(e)
            )
            return None

    async def get_student_progress_report_data(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive data for generating student progress reports.

        Aggregates data from all memory layers to provide a complete picture
        of the student's learning progress during a time period.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            date_from: Start date for the report period.
            date_to: End date for the report period.

        Returns:
            Dictionary containing:
            - reportable_events: High-importance learning events
            - mastery_changes: Topics with mastery changes
            - learning_patterns: Effective strategies and patterns
            - interests: Student interests and preferences
            - summary_stats: Aggregate statistics
        """
        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        # Default to last 7 days if no dates provided
        if not date_to:
            date_to = datetime.now(timezone.utc)
        if not date_from:
            from datetime import timedelta
            date_from = date_to - timedelta(days=7)

        # Get reportable events from episodic memory
        reportable_events = await self.episodic.get_reportable_events(
            tenant_code=tenant_code,
            student_id=student_id,
            date_from=date_from,
            date_to=date_to,
            limit=50,
        )

        # Get all recent events for statistics
        all_events = await self.episodic.get_recent(
            tenant_code=tenant_code,
            student_id=student_id,
            limit=100,
        )

        # Filter events within date range
        events_in_range = [
            e for e in all_events
            if date_from <= e.occurred_at <= date_to
        ]

        # Get mastery overview
        mastery_overview = await self.semantic.get_mastery_overview(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Get learning patterns
        learning_patterns = await self.procedural.get_learning_patterns(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Get interests
        interests = await self.associative.get_interests(
            tenant_code=tenant_code,
            student_id=student_id,
        )

        # Calculate summary statistics
        total_events = len(events_in_range)
        breakthroughs = len([
            e for e in events_in_range
            if e.event_type.value in ["breakthrough", "mastery"]
        ])
        struggles = len([
            e for e in events_in_range
            if e.event_type.value in ["struggle", "confusion"]
        ])

        # Count comprehension checks
        comprehension_checks = len([
            e for e in events_in_range
            if e.details.get("comprehension_score") is not None
        ])
        verified_understanding = len([
            e for e in events_in_range
            if e.details.get("verified") is True
        ])

        return {
            "reportable_events": [e.model_dump() for e in reportable_events],
            "mastery_overview": mastery_overview.model_dump() if mastery_overview else None,
            "learning_patterns": learning_patterns.model_dump() if learning_patterns else None,
            "interests": interests.model_dump() if interests else None,
            "summary_stats": {
                "total_events": total_events,
                "breakthroughs": breakthroughs,
                "struggles": struggles,
                "comprehension_checks": comprehension_checks,
                "verified_understanding": verified_understanding,
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
            },
        }

    async def get_topic_learning_journey(
        self,
        tenant_code: str,
        student_id: str | uuid.UUID,
        topic_full_code: str,
    ) -> dict[str, Any]:
        """Get the complete learning journey for a specific topic.

        Retrieves all learning events, mastery changes, and patterns
        for a particular topic to understand the student's learning
        progression.

        Args:
            tenant_code: Tenant code.
            student_id: Student identifier.
            topic_full_code: Full topic code.

        Returns:
            Dictionary containing:
            - events: All episodic events for this topic
            - mastery: Current mastery level and history
            - patterns: Effective strategies for this topic
            - comprehension_evaluations: All comprehension checks
        """
        # Convert student_id to UUID if string
        if isinstance(student_id, str):
            student_id = uuid.UUID(student_id)

        # Get all events for this topic
        events = await self.episodic.get_events_for_topic(
            tenant_code=tenant_code,
            student_id=student_id,
            topic_full_code=topic_full_code,
            limit=50,
        )

        # Get mastery for this topic
        from src.models.memory import EntityType
        mastery = await self.semantic.get_by_entity(
            tenant_code=tenant_code,
            student_id=student_id,
            entity_type=EntityType.TOPIC,
            entity_full_code=topic_full_code,
        )

        # Extract comprehension evaluations from events
        comprehension_evaluations = [
            {
                "occurred_at": e.occurred_at.isoformat(),
                "score": e.details.get("comprehension_score", 0.0),
                "verified": e.details.get("verified", False),
                "parroting_detected": e.details.get("parroting_detected", False),
                "concepts_understood": e.details.get("concepts_understood", []),
                "misconceptions": e.details.get("misconceptions", []),
            }
            for e in events
            if e.details.get("comprehension_score") is not None
        ]

        # Summarize event types
        event_summary = {}
        for e in events:
            event_type = e.event_type.value
            event_summary[event_type] = event_summary.get(event_type, 0) + 1

        return {
            "topic_full_code": topic_full_code,
            "events": [e.model_dump() for e in events],
            "event_summary": event_summary,
            "mastery": mastery.model_dump() if mastery else None,
            "comprehension_evaluations": comprehension_evaluations,
            "total_sessions": len(set(e.session_id for e in events if e.session_id)),
        }


# Import for use in search
from src.models.memory import (
    AssociativeMemoryResponse,
    EpisodicMemorySearchParams,
)
