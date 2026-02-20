# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MCP tool definitions for EduSynapseOS.

This module defines the MCP tools that expose EduSynapseOS capabilities
to external LLM clients. Each tool is a callable function that performs
a specific operation.

Tools:
- knowledge_lookup: Search curriculum knowledge base via RAG
- get_student_context: Retrieve 4-layer memory context for a student
- evaluate_answer: Semantically evaluate a student's answer
- generate_question: Generate an educational question
- get_mastery_report: Get a student's mastery/progress report

All tools require tenant_code for multi-tenant isolation and use the
existing core services (MemoryManager, RAGRetriever, AgentFactory).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from src.core.agents.context import AgentExecutionContext
from src.models.common import BloomLevel
from src.models.practice import QuestionType

if TYPE_CHECKING:
    from src.core.intelligence.mcp.server import EduSynapseMCPServer

logger = logging.getLogger(__name__)


def register_tools(server: EduSynapseMCPServer) -> None:
    """Register all MCP tools with the server.

    Args:
        server: The EduSynapseMCPServer instance.
    """
    mcp = server.mcp

    @mcp.tool()
    async def knowledge_lookup(
        query: str,
        tenant_code: str,
        subject: str | None = None,
        limit: int = 5,
    ) -> str:
        """Search curriculum knowledge base using RAG.

        Retrieves relevant curriculum content based on semantic similarity
        to the query. Useful for finding educational content, explanations,
        and reference material.

        Args:
            query: The search query text.
            tenant_code: Tenant identifier for isolation.
            subject: Optional subject filter (e.g., "math", "science").
            limit: Maximum number of results to return (default: 5).

        Returns:
            JSON string containing search results with content and metadata.

        Example:
            >>> result = await knowledge_lookup(
            ...     query="How do fractions work?",
            ...     tenant_code="acme",
            ...     subject="math",
            ...     limit=5,
            ... )
        """
        logger.info(
            "knowledge_lookup: query=%s, tenant=%s, subject=%s",
            query[:50],
            tenant_code,
            subject,
        )

        try:
            # Use a placeholder student_id for curriculum-only search
            # RAGRetriever can search curriculum without student context
            results = await server.rag_retriever._retrieve_curriculum(
                tenant_code=tenant_code,
                query=query,
                limit=limit,
                min_score=0.3,
            )

            # Format results for output
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.content,
                    "score": round(result.score, 3),
                    "source": result.source.value,
                    "metadata": {
                        "subject": result.metadata.get("subject"),
                        "topic": result.metadata.get("topic"),
                        "chunk_type": result.metadata.get("chunk_type"),
                    },
                })

            return json.dumps({
                "success": True,
                "query": query,
                "result_count": len(formatted_results),
                "results": formatted_results,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("knowledge_lookup failed: %s", str(e))
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": query,
            }, ensure_ascii=False)

    @mcp.tool()
    async def get_student_context(
        student_id: str,
        tenant_code: str,
        topic: str | None = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        include_associative: bool = True,
    ) -> str:
        """Get full learning context for a student.

        Retrieves the complete 4-layer memory context including:
        - Episodic: Recent learning events and sessions
        - Semantic: Knowledge state and mastery levels
        - Procedural: Learning patterns and VARK profile
        - Associative: Interests and effective analogies

        Args:
            student_id: The student's unique identifier (UUID string).
            tenant_code: Tenant identifier for isolation.
            topic: Optional topic to filter context (reserved for future use).
            include_episodic: Include episodic memories (default: True).
            include_semantic: Include semantic mastery (default: True).
            include_procedural: Include procedural patterns (default: True).
            include_associative: Include interests (default: True).

        Returns:
            JSON string containing the full memory context.

        Example:
            >>> context = await get_student_context(
            ...     student_id="550e8400-e29b-41d4-a716-446655440000",
            ...     tenant_code="acme",
            ...     topic="fractions",
            ... )
        """
        logger.info(
            "get_student_context: student=%s, tenant=%s, topic=%s",
            student_id,
            tenant_code,
            topic,
        )

        try:
            student_uuid = uuid.UUID(student_id)

            # Get full context from memory manager
            context = await server.memory_manager.get_full_context(
                tenant_code=tenant_code,
                student_id=student_uuid,
                topic=topic,
            )

            # Build response based on requested layers
            response_data: dict = {
                "success": True,
                "student_id": student_id,
                "retrieved_at": context.retrieved_at.isoformat(),
            }

            if include_episodic:
                response_data["episodic"] = [
                    {
                        "id": str(ep.id),
                        "event_type": ep.event_type.value,
                        "summary": ep.summary,
                        "importance": ep.importance,
                        "occurred_at": ep.occurred_at.isoformat(),
                    }
                    for ep in context.episodic[:10]  # Limit for response size
                ]

            if include_semantic:
                response_data["semantic"] = {
                    "overall_mastery": context.semantic.overall_mastery,
                    "topics_mastered": context.semantic.topics_mastered,
                    "topics_learning": context.semantic.topics_learning,
                    "topics_struggling": context.semantic.topics_struggling,
                    "total_topics": context.semantic.total_topics,
                }

            if include_procedural:
                response_data["procedural"] = {
                    "best_time_of_day": context.procedural.best_time_of_day,
                    "preferred_content_format": context.procedural.preferred_content_format,
                    "average_session_duration_minutes": context.procedural.average_session_duration_minutes,
                    "vark_profile": (
                        context.procedural.vark_profile.model_dump()
                        if context.procedural.vark_profile
                        else None
                    ),
                }

            if include_associative:
                response_data["associative"] = {
                    "interests": [
                        {
                            "category": interest.category,
                            "content": interest.content,
                            "strength": interest.strength,
                        }
                        for interest in context.associative.interests[:10]
                    ],
                    "effective_analogies": context.associative.effective_analogies[:5],
                }

            return json.dumps(response_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Invalid student_id format: %s", student_id)
            return json.dumps({
                "success": False,
                "error": f"Invalid student_id format: {e}",
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("get_student_context failed: %s", str(e))
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.tool()
    async def evaluate_answer(
        question: str,
        student_answer: str,
        expected_answer: str,
        tenant_code: str,
        student_id: str | None = None,
        topic: str = "",
        question_type: str = "short_answer",
        partial_credit: bool = True,
        language: str = "tr",
    ) -> str:
        """Semantically evaluate a student's answer.

        Uses AI to evaluate the answer with semantic understanding,
        not just exact matching. Provides detailed feedback, score,
        and identifies any misconceptions.

        Args:
            question: The original question text.
            student_answer: The student's answer to evaluate.
            expected_answer: The correct/expected answer.
            tenant_code: Tenant identifier for isolation.
            student_id: Optional student ID for personalized evaluation.
            topic: Topic of the question.
            question_type: Type of question (short_answer, multiple_choice, etc.).
            partial_credit: Whether to award partial credit (default: True).
            language: Language for feedback (default: "tr").

        Returns:
            JSON string containing evaluation result with score, feedback,
            and any detected misconceptions.

        Example:
            >>> result = await evaluate_answer(
            ...     question="What is 2 + 2?",
            ...     student_answer="4",
            ...     expected_answer="4",
            ...     tenant_code="acme",
            ...     topic="math",
            ... )
        """
        logger.info(
            "evaluate_answer: tenant=%s, topic=%s, question=%s...",
            tenant_code,
            topic,
            question[:50],
        )

        try:
            # Get the assessor agent
            agent = server.agent_factory.get("assessor")

            # Map question_type string to enum
            q_type = QuestionType.SHORT_ANSWER
            try:
                q_type = QuestionType(question_type)
            except ValueError:
                pass

            # Create execution context
            context = AgentExecutionContext(
                tenant_id=tenant_code,
                student_id=student_id or "anonymous",
                topic=topic,
                intent="answer_evaluation",
                params={
                    "question_content": question,
                    "student_answer": student_answer,
                    "expected_answer": expected_answer,
                    "question_type": q_type.value,
                    "partial_credit": partial_credit,
                    "language": language,
                    "topic": topic,
                },
            )

            # Execute the capability
            response = await agent.execute(context)

            if response.success and response.result:
                result = response.result

                # Handle both dict and object results
                if hasattr(result, "model_dump"):
                    result_data = result.model_dump()
                elif isinstance(result, dict):
                    result_data = result
                else:
                    result_data = {"raw": str(result)}

                return json.dumps({
                    "success": True,
                    "is_correct": result_data.get("is_correct", False),
                    "score": result_data.get("score", 0.0),
                    "feedback": result_data.get("feedback", ""),
                    "misconceptions": result_data.get("misconceptions", []),
                    "improvement_suggestions": result_data.get("improvement_suggestions", []),
                    "confidence": result_data.get("confidence", 0.9),
                }, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": "Evaluation failed",
                }, ensure_ascii=False)

        except Exception as e:
            logger.exception("evaluate_answer failed: %s", str(e))
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.tool()
    async def generate_question(
        topic: str,
        tenant_code: str,
        difficulty: float = 0.5,
        bloom_level: str = "understand",
        question_type: str = "multiple_choice",
        language: str = "tr",
        include_hints: bool = True,
        include_explanation: bool = True,
    ) -> str:
        """Generate an educational question using AI.

        Creates a pedagogically sound question tailored to the specified
        parameters including difficulty, Bloom's taxonomy level, and type.
        Includes hints and explanation for learning support.

        Args:
            topic: The topic for the question.
            tenant_code: Tenant identifier for isolation.
            difficulty: Target difficulty level 0.0-1.0 (default: 0.5).
            bloom_level: Bloom's taxonomy level (remember, understand, apply,
                        analyze, evaluate, create). Default: "understand".
            question_type: Type of question (multiple_choice, short_answer,
                          true_false, fill_blank). Default: "multiple_choice".
            language: Language for the question (default: "tr").
            include_hints: Generate hints (default: True).
            include_explanation: Include answer explanation (default: True).

        Returns:
            JSON string containing the generated question with content,
            options (if applicable), correct answer, hints, and explanation.

        Example:
            >>> result = await generate_question(
            ...     topic="Fractions",
            ...     tenant_code="acme",
            ...     difficulty=0.6,
            ...     bloom_level="apply",
            ...     question_type="multiple_choice",
            ... )
        """
        logger.info(
            "generate_question: topic=%s, tenant=%s, difficulty=%.1f, bloom=%s",
            topic,
            tenant_code,
            difficulty,
            bloom_level,
        )

        try:
            # Get the assessor agent (has question_generation capability)
            agent = server.agent_factory.get("assessor")

            # Map bloom_level string to enum
            b_level = BloomLevel.UNDERSTAND
            try:
                b_level = BloomLevel(bloom_level)
            except ValueError:
                pass

            # Map question_type string to enum
            q_type = QuestionType.MULTIPLE_CHOICE
            try:
                q_type = QuestionType(question_type)
            except ValueError:
                pass

            # Create execution context
            context = AgentExecutionContext(
                tenant_id=tenant_code,
                student_id="anonymous",
                topic=topic,
                intent="question_generation",
                params={
                    "topic": topic,
                    "difficulty": difficulty,
                    "bloom_level": b_level.value,
                    "question_type": q_type.value,
                    "language": language,
                    "include_hints": include_hints,
                    "hint_count": 3 if include_hints else 0,
                    "include_explanation": include_explanation,
                },
            )

            # Execute the capability
            response = await agent.execute(context)

            if response.success and response.result:
                result = response.result

                # Handle both dict and object results
                if hasattr(result, "model_dump"):
                    result_data = result.model_dump()
                elif isinstance(result, dict):
                    result_data = result
                else:
                    result_data = {"raw": str(result)}

                return json.dumps({
                    "success": True,
                    "content": result_data.get("content", ""),
                    "question_type": result_data.get("question_type", question_type),
                    "difficulty": result_data.get("difficulty", difficulty),
                    "bloom_level": result_data.get("bloom_level", bloom_level),
                    "options": result_data.get("options"),
                    "correct_answer": result_data.get("correct_answer", ""),
                    "hints": result_data.get("hints", []),
                    "explanation": result_data.get("explanation"),
                    "topic": topic,
                    "language": language,
                }, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": "Question generation failed",
                }, ensure_ascii=False)

        except Exception as e:
            logger.exception("generate_question failed: %s", str(e))
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False)

    @mcp.tool()
    async def get_mastery_report(
        student_id: str,
        tenant_code: str,
        subject: str | None = None,
        include_recommendations: bool = True,
    ) -> str:
        """Get a student's mastery and progress report.

        Provides a comprehensive overview of the student's learning
        progress including mastery levels across topics, engagement
        metrics, and personalization data.

        Args:
            student_id: The student's unique identifier (UUID string).
            tenant_code: Tenant identifier for isolation.
            subject: Optional subject filter for focused report.
            include_recommendations: Include learning recommendations (default: True).

        Returns:
            JSON string containing mastery summary, engagement metrics,
            and learning recommendations.

        Example:
            >>> report = await get_mastery_report(
            ...     student_id="550e8400-e29b-41d4-a716-446655440000",
            ...     tenant_code="acme",
            ...     subject="math",
            ... )
        """
        logger.info(
            "get_mastery_report: student=%s, tenant=%s, subject=%s",
            student_id,
            tenant_code,
            subject,
        )

        try:
            student_uuid = uuid.UUID(student_id)

            # Get learning summary from memory manager
            summary = await server.memory_manager.get_learning_summary(
                tenant_code=tenant_code,
                student_id=student_uuid,
            )

            # Build response
            response_data = {
                "success": True,
                "student_id": student_id,
                "mastery": summary.get("mastery", {}),
                "engagement": summary.get("engagement", {}),
                "personalization": summary.get("personalization", {}),
            }

            # Add recommendations if requested
            if include_recommendations:
                mastery_data = summary.get("mastery", {})
                overall = mastery_data.get("overall", 0.5)
                struggling = mastery_data.get("topics_struggling", 0)

                recommendations = []

                if overall < 0.5:
                    recommendations.append({
                        "type": "focus_area",
                        "message": "Overall mastery is below 50%. Consider reviewing foundational concepts.",
                        "priority": "high",
                    })

                if struggling > 0:
                    recommendations.append({
                        "type": "struggling_topics",
                        "message": f"Student is struggling with {struggling} topic(s). Extra practice recommended.",
                        "priority": "high" if struggling > 2 else "medium",
                    })

                engagement = summary.get("engagement", {})
                if engagement.get("positive_ratio", 0.5) < 0.4:
                    recommendations.append({
                        "type": "engagement",
                        "message": "Engagement ratio is low. Consider adjusting difficulty or trying different approaches.",
                        "priority": "medium",
                    })

                response_data["recommendations"] = recommendations

            return json.dumps(response_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Invalid student_id format: %s", student_id)
            return json.dumps({
                "success": False,
                "error": f"Invalid student_id format: {e}",
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception("get_mastery_report failed: %s", str(e))
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False)

    logger.debug("Registered 5 MCP tools: knowledge_lookup, get_student_context, evaluate_answer, generate_question, get_mastery_report")
