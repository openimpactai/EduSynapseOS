# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MCP resource definitions for EduSynapseOS.

This module defines the MCP resources that provide data access to external
LLM clients. Resources are like GET endpoints - they expose data for reading.

Resources:
- student://{tenant_code}/{student_id}/profile: Student profile and learning state
- curriculum://{tenant_code}/{subject}/{topic}: Curriculum content for a topic
- memory://{tenant_code}/{student_id}/episodic: Recent learning events
- analytics://{tenant_code}/{student_id}/dashboard: Learning analytics summary

All resources use tenant_code for multi-tenant isolation and return JSON-formatted
data suitable for LLM consumption.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.intelligence.mcp.server import EduSynapseMCPServer

logger = logging.getLogger(__name__)


def register_resources(server: EduSynapseMCPServer) -> None:
    """Register all MCP resources with the server.

    Args:
        server: The EduSynapseMCPServer instance.
    """
    mcp = server.mcp

    @mcp.resource("student://{tenant_code}/{student_id}/profile")
    async def get_student_profile(tenant_code: str, student_id: str) -> str:
        """Get student profile and learning state.

        Provides comprehensive information about a student including:
        - Basic profile information
        - Current mastery levels
        - Learning preferences (VARK profile)
        - Recent activity summary

        Args:
            tenant_code: Tenant identifier for isolation.
            student_id: The student's unique identifier (UUID string).

        Returns:
            JSON string containing the student profile and learning state.
        """
        logger.debug(
            "Resource access: student profile, tenant=%s, student=%s",
            tenant_code,
            student_id,
        )

        try:
            student_uuid = uuid.UUID(student_id)

            # Get full context from memory manager
            context = await server.memory_manager.get_full_context(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            # Get learning summary for additional metrics
            summary = await server.memory_manager.get_learning_summary(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            profile_data = {
                "student_id": student_id,
                "tenant_code": tenant_code,
                "mastery": {
                    "overall": context.semantic.overall_mastery,
                    "topics_mastered": context.semantic.topics_mastered,
                    "topics_learning": context.semantic.topics_learning,
                    "topics_struggling": context.semantic.topics_struggling,
                    "total_topics": context.semantic.total_topics,
                },
                "learning_preferences": {
                    "vark_profile": (
                        context.procedural.vark_profile.model_dump()
                        if context.procedural.vark_profile
                        else None
                    ),
                    "preferred_time": context.procedural.best_time_of_day,
                    "preferred_format": context.procedural.preferred_content_format,
                    "average_session_minutes": context.procedural.average_session_duration_minutes,
                },
                "engagement": summary.get("engagement", {}),
                "interests_count": len(context.associative.interests),
                "retrieved_at": context.retrieved_at.isoformat(),
            }

            return json.dumps(profile_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Invalid student_id format: %s", student_id)
            return json.dumps({
                "error": f"Invalid student_id format: {e}",
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("Failed to get student profile: %s", str(e))
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.resource("curriculum://{tenant_code}/{subject}/{topic}")
    async def get_curriculum_content(
        tenant_code: str,
        subject: str,
        topic: str,
    ) -> str:
        """Get curriculum content for a specific topic.

        Retrieves educational content from the curriculum knowledge base
        for the specified subject and topic. Uses RAG to find the most
        relevant content chunks.

        Args:
            tenant_code: Tenant identifier for isolation.
            subject: Subject area (e.g., "math", "science", "history").
            topic: Specific topic within the subject.

        Returns:
            JSON string containing curriculum content and metadata.
        """
        logger.debug(
            "Resource access: curriculum content, tenant=%s, subject=%s, topic=%s",
            tenant_code,
            subject,
            topic,
        )

        try:
            # Search for curriculum content using RAG
            query = f"{subject}: {topic}"

            results = await server.rag_retriever._retrieve_curriculum(
                tenant_code=tenant_code,
                query=query,
                limit=10,
                min_score=0.3,
            )

            # Filter results by subject if available in metadata
            filtered_results = []
            for result in results:
                result_subject = result.metadata.get("subject", "").lower()
                if not result_subject or result_subject == subject.lower():
                    filtered_results.append({
                        "content": result.content,
                        "score": round(result.score, 3),
                        "metadata": {
                            "subject": result.metadata.get("subject"),
                            "topic": result.metadata.get("topic"),
                            "chunk_type": result.metadata.get("chunk_type"),
                            "source_id": result.metadata.get("source_id"),
                        },
                    })

            curriculum_data = {
                "subject": subject,
                "topic": topic,
                "tenant_code": tenant_code,
                "content_count": len(filtered_results),
                "content": filtered_results,
            }

            return json.dumps(curriculum_data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("Failed to get curriculum content: %s", str(e))
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.resource("memory://{tenant_code}/{student_id}/episodic")
    async def get_episodic_memories(tenant_code: str, student_id: str) -> str:
        """Get recent learning events for a student.

        Retrieves the student's episodic memories which capture specific
        learning events including:
        - Success and failure moments
        - Emotional states during learning
        - Session activities
        - Breakthroughs and struggles

        Args:
            tenant_code: Tenant identifier for isolation.
            student_id: The student's unique identifier (UUID string).

        Returns:
            JSON string containing recent episodic memories.
        """
        logger.debug(
            "Resource access: episodic memories, tenant=%s, student=%s",
            tenant_code,
            student_id,
        )

        try:
            student_uuid = uuid.UUID(student_id)

            # Get recent episodic memories
            recent_memories = await server.memory_manager.episodic.get_recent(
                tenant_code=tenant_code,
                student_id=student_uuid,
                limit=20,
            )

            # Get important memories (high importance)
            important_memories = await server.memory_manager.episodic.get_important_memories(
                tenant_code=tenant_code,
                student_id=student_uuid,
                min_importance=0.7,
                limit=10,
            )

            # Get event type statistics
            event_stats = await server.memory_manager.episodic.get_event_type_stats(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            episodic_data = {
                "student_id": student_id,
                "tenant_code": tenant_code,
                "recent_memories": [
                    {
                        "id": str(mem.id),
                        "event_type": mem.event_type.value,
                        "summary": mem.summary,
                        "importance": mem.importance,
                        "emotional_state": (
                            mem.emotional_state.value if mem.emotional_state else None
                        ),
                        "occurred_at": mem.occurred_at.isoformat(),
                    }
                    for mem in recent_memories
                ],
                "important_memories": [
                    {
                        "id": str(mem.id),
                        "event_type": mem.event_type.value,
                        "summary": mem.summary,
                        "importance": mem.importance,
                        "occurred_at": mem.occurred_at.isoformat(),
                    }
                    for mem in important_memories
                ],
                "event_statistics": event_stats,
            }

            return json.dumps(episodic_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Invalid student_id format: %s", student_id)
            return json.dumps({
                "error": f"Invalid student_id format: {e}",
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("Failed to get episodic memories: %s", str(e))
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.resource("analytics://{tenant_code}/{student_id}/dashboard")
    async def get_analytics_dashboard(tenant_code: str, student_id: str) -> str:
        """Get learning analytics dashboard for a student.

        Provides a comprehensive analytics summary including:
        - Mastery progression
        - Engagement metrics
        - Learning patterns
        - Performance trends

        Args:
            tenant_code: Tenant identifier for isolation.
            student_id: The student's unique identifier (UUID string).

        Returns:
            JSON string containing analytics dashboard data.
        """
        logger.debug(
            "Resource access: analytics dashboard, tenant=%s, student=%s",
            tenant_code,
            student_id,
        )

        try:
            student_uuid = uuid.UUID(student_id)

            # Get learning summary (includes mastery, engagement, personalization)
            summary = await server.memory_manager.get_learning_summary(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            # Get full context for additional data
            context = await server.memory_manager.get_full_context(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            # Calculate additional analytics
            mastery_data = summary.get("mastery", {})
            engagement_data = summary.get("engagement", {})
            personalization_data = summary.get("personalization", {})

            # Determine learning status
            overall_mastery = mastery_data.get("overall", 0.5)
            if overall_mastery >= 0.8:
                status = "excelling"
            elif overall_mastery >= 0.6:
                status = "on_track"
            elif overall_mastery >= 0.4:
                status = "needs_attention"
            else:
                status = "struggling"

            dashboard_data = {
                "student_id": student_id,
                "tenant_code": tenant_code,
                "status": status,
                "mastery": {
                    "overall": overall_mastery,
                    "topics_mastered": mastery_data.get("topics_mastered", 0),
                    "topics_learning": mastery_data.get("topics_learning", 0),
                    "topics_struggling": mastery_data.get("topics_struggling", 0),
                    "total_topics": mastery_data.get("total_topics", 0),
                },
                "engagement": {
                    "total_sessions": engagement_data.get("total_episodes", 0),
                    "positive_ratio": engagement_data.get("positive_ratio", 0.5),
                    "event_distribution": engagement_data.get("event_distribution", {}),
                },
                "learning_style": {
                    "vark_profile": personalization_data.get("vark_profile"),
                    "preferred_time": personalization_data.get("preferred_time"),
                    "preferred_format": personalization_data.get("preferred_format"),
                    "interests_count": personalization_data.get("interests_recorded", 0),
                },
                "recent_activity": {
                    "last_sessions": [
                        {
                            "event_type": ep.event_type.value,
                            "summary": ep.summary[:100],
                            "occurred_at": ep.occurred_at.isoformat(),
                        }
                        for ep in context.episodic[:5]
                    ],
                },
            }

            return json.dumps(dashboard_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Invalid student_id format: %s", student_id)
            return json.dumps({
                "error": f"Invalid student_id format: {e}",
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("Failed to get analytics dashboard: %s", str(e))
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False)

    logger.debug("Registered 4 MCP resources: student profile, curriculum content, episodic memories, analytics dashboard")
