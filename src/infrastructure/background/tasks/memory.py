# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory background tasks for EduSynapseOS.

Tasks for managing episodic and semantic memory operations.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.tasks.base import run_async

# Setup broker before defining actors
setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.MEMORY,
    max_retries=1,
    time_limit=60000,  # 1 minute
    priority=Priority.NORMAL,
)
def record_learning_event(
    tenant_code: str,
    student_id: str,
    event_type: str,
    topic: str,
    data: dict[str, Any] | None = None,
    session_id: str | None = None,
    importance: float = 0.5,
) -> dict[str, Any]:
    """Record a learning event in episodic memory.

    Args:
        tenant_code: Tenant code.
        student_id: Student identifier.
        event_type: Type of event (e.g., "question_answered").
        topic: Topic being studied.
        data: Additional event data.
        session_id: Related practice session ID.
        importance: Event importance (0-1).

    Returns:
        Created memory record info.
    """

    async def _record() -> dict[str, Any]:
        from src.core.config import get_settings
        from src.core.intelligence.embeddings import EmbeddingService
        from src.core.memory import MemoryManager
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.vectors import get_qdrant_client

        try:
            settings = get_settings()
            tenant_db = get_worker_db_manager()
            embedding_service = EmbeddingService(settings)
            qdrant = get_qdrant_client()

            manager = MemoryManager(
                tenant_db_manager=tenant_db,
                embedding_service=embedding_service,
                qdrant_client=qdrant,
            )

            memory = await manager.record_learning_event(
                tenant_code=tenant_code,
                student_id=student_id,
                event_type=event_type,
                topic=topic,
                data=data,
                session_id=session_id,
                importance=importance,
            )

            logger.debug(
                "Learning event recorded: %s (student: %s, type: %s)",
                memory.id,
                student_id,
                event_type,
            )

            return {
                "memory_id": str(memory.id),
                "student_id": student_id,
                "event_type": event_type,
                "topic": topic,
                "recorded": True,
            }

        except Exception as e:
            logger.error("Failed to record learning event: %s", str(e), exc_info=True)
            return {
                "student_id": student_id,
                "event_type": event_type,
                "recorded": False,
                "error": str(e),
            }

    return run_async(_record())


@dramatiq.actor(
    queue_name=Queues.MEMORY,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.LOW,
)
def cleanup_old_memories(
    tenant_code: str,
    max_age_days: int = 365,
    min_importance: float = 0.7,
) -> dict[str, Any]:
    """Clean up old episodic memories.

    Removes old low-importance memories while keeping important ones.

    Args:
        tenant_code: Tenant code.
        max_age_days: Maximum age for memories to keep.
        min_importance: Keep memories above this importance regardless of age.

    Returns:
        Cleanup result with counts.
    """

    async def _cleanup() -> dict[str, Any]:
        from sqlalchemy import and_, delete, func, select

        from src.infrastructure.database.models.tenant.memory import EpisodicMemory
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Count memories to delete
                count_stmt = select(func.count(EpisodicMemory.id)).where(
                    and_(
                        EpisodicMemory.created_at < cutoff_date,
                        EpisodicMemory.importance < min_importance,
                    )
                )

                result = await session.execute(count_stmt)
                delete_count = result.scalar() or 0

                if delete_count > 0:
                    # Delete old low-importance memories
                    delete_stmt = delete(EpisodicMemory).where(
                        and_(
                            EpisodicMemory.created_at < cutoff_date,
                            EpisodicMemory.importance < min_importance,
                        )
                    )

                    await session.execute(delete_stmt)
                    await session.commit()

                logger.info(
                    "Memory cleanup for %s: deleted %d memories older than %d days",
                    tenant_code,
                    delete_count,
                    max_age_days,
                )

                return {
                    "tenant_code": tenant_code,
                    "deleted_count": delete_count,
                    "cutoff_date": cutoff_date.isoformat(),
                    "min_importance": min_importance,
                }

        except Exception as e:
            logger.error("Failed to cleanup memories: %s", str(e), exc_info=True)
            return {
                "tenant_code": tenant_code,
                "error": str(e),
            }

    return run_async(_cleanup())


@dramatiq.actor(
    queue_name=Queues.MEMORY,
    max_retries=1,
    time_limit=300000,  # 5 minutes
    priority=Priority.LOW,
)
def create_mastery_snapshots(
    tenant_code: str,
) -> dict[str, Any]:
    """Create daily mastery snapshots for all students.

    Takes a snapshot of each student's current mastery levels from
    SemanticMemory and stores in MasterySnapshot table.

    Args:
        tenant_code: Tenant code.

    Returns:
        Snapshot creation result.
    """

    async def _create_snapshots() -> dict[str, Any]:
        from sqlalchemy import and_, select

        from src.infrastructure.database.models.tenant.analytics import MasterySnapshot
        from src.infrastructure.database.models.tenant.memory import SemanticMemory
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        snapshot_date = datetime.now(timezone.utc).date()

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Get all unique students with semantic memories
                stmt = select(SemanticMemory.student_id).distinct()
                result = await session.execute(stmt)
                student_ids = [row[0] for row in result.fetchall()]

                snapshots_created = 0
                snapshots_skipped = 0

                for student_id in student_ids:
                    # Check if snapshot already exists for today
                    existing_stmt = select(MasterySnapshot).where(
                        and_(
                            MasterySnapshot.student_id == student_id,
                            MasterySnapshot.snapshot_date == snapshot_date,
                        )
                    )
                    existing_result = await session.execute(existing_stmt)
                    if existing_result.scalar_one_or_none():
                        snapshots_skipped += 1
                        continue

                    # Get current mastery levels for this student
                    mastery_stmt = select(SemanticMemory).where(
                        SemanticMemory.student_id == student_id
                    )
                    mastery_result = await session.execute(mastery_stmt)
                    memories = mastery_result.scalars().all()

                    if not memories:
                        continue

                    # Build topic mastery dictionary
                    # Note: SemanticMemory entity_type is "topic" or "objective"
                    # Subject mastery is calculated from topic mastery via curriculum sync
                    subject_mastery: dict[str, float] = {}
                    topic_mastery: dict[str, float] = {}

                    for memory in memories:
                        entity_full_code = memory.entity_full_code
                        mastery_level = float(memory.mastery_level)

                        if memory.entity_type == "topic":
                            topic_mastery[entity_full_code] = mastery_level
                            # Extract subject from topic full code (first 2 parts)
                            parts = entity_full_code.split(".")
                            if len(parts) >= 2:
                                subject_full_code = f"{parts[0]}.{parts[1]}"
                                if subject_full_code not in subject_mastery:
                                    subject_mastery[subject_full_code] = mastery_level
                                else:
                                    # Average with existing subject mastery
                                    subject_mastery[subject_full_code] = (
                                        subject_mastery[subject_full_code] + mastery_level
                                    ) / 2

                    # Calculate overall mastery
                    all_levels = list(subject_mastery.values()) + list(topic_mastery.values())
                    overall_mastery = sum(all_levels) / len(all_levels) if all_levels else 0.0

                    # Create snapshot using existing model structure
                    snapshot = MasterySnapshot(
                        id=str(uuid4()),
                        student_id=student_id,
                        snapshot_date=snapshot_date,
                        subject_mastery=subject_mastery,
                        topic_mastery=topic_mastery,
                        overall_mastery=overall_mastery,
                        created_at=datetime.now(timezone.utc),
                    )

                    session.add(snapshot)
                    snapshots_created += 1

                await session.commit()

                logger.info(
                    "Mastery snapshots for %s: %d created, %d skipped (already exist)",
                    tenant_code,
                    snapshots_created,
                    snapshots_skipped,
                )

                return {
                    "tenant_code": tenant_code,
                    "date": str(snapshot_date),
                    "students_processed": len(student_ids),
                    "snapshots_created": snapshots_created,
                    "snapshots_skipped": snapshots_skipped,
                }

        except Exception as e:
            logger.error("Failed to create mastery snapshots: %s", str(e), exc_info=True)
            return {
                "tenant_code": tenant_code,
                "error": str(e),
            }

    return run_async(_create_snapshots())


def get_memory_actors() -> list:
    """Get all memory actors.

    Returns:
        List of memory actor functions.
    """
    return [
        record_learning_event,
        cleanup_old_memories,
        create_mastery_snapshots,
    ]
