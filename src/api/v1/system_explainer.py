# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System Explainer API endpoints.

This module provides public endpoints for learning about EduSynapseOS
through an AI-powered explainer agent.

Endpoints:
- POST /chat - Chat with the explainer agent
- POST /quick-explain - Get a quick explanation of a specific topic
- GET /topics - List available topics for explanation
- GET /sessions/{session_id} - Get session information
- DELETE /sessions/{session_id} - Clear a session

These endpoints are PUBLIC (no authentication required) to allow
potential customers and investors to explore the platform.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.domains.system_explainer import (
    AVAILABLE_TOPICS,
    AudienceType,
    ExplainerChatRequest,
    ExplainerChatResponse,
    ExplainerTopicsResponse,
    QuickExplainRequest,
    QuickExplainResponse,
    get_system_explainer_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ExplainerChatResponse,
    summary="Chat with the System Explainer",
    description="""
Chat with SYNAPSE, the EduSynapseOS expert AI agent.

Ask any question about EduSynapseOS - architecture, features, educational
theories, business model, technical details, etc.

The agent automatically adapts its communication style based on your
questions, or you can specify an audience type for more focused responses.

**Audience Types:**
- `auto` (default) - Let the agent detect from your questions
- `investor` - Business-focused, metrics, market opportunity
- `educator` - Pedagogical focus, theory foundations, outcomes
- `technical` - Architecture, implementation, integration details
- `general` - Accessible explanations with analogies

**Examples:**
- "What is EduSynapseOS?"
- "How does the 4-layer memory system work?"
- "What's your competitive advantage?"
- "Explain the educational theory orchestration"
""",
)
async def chat(
    request: ExplainerChatRequest,
) -> ExplainerChatResponse:
    """Chat with the system explainer agent.

    This endpoint provides conversational access to learn about EduSynapseOS.
    Sessions maintain conversation history for contextual responses.

    Args:
        request: Chat request with message and optional session ID.

    Returns:
        Agent's response with session info and suggested topics.
    """
    try:
        service = get_system_explainer_service()
        response = await service.chat(request)
        return response

    except Exception as e:
        logger.exception("Error in explainer chat: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your question. Please try again.",
        )


@router.post(
    "/chat/stream",
    summary="Stream Chat with the System Explainer",
    description="""
Stream a chat response from SYNAPSE using Server-Sent Events (SSE).

Returns a stream of events:
- `start`: Initial event with session info
- `token`: Each token as it's generated
- `done`: Final event with metadata and suggested topics
- `error`: Error event if something goes wrong

Use this endpoint for real-time, ChatGPT-like streaming responses.
""",
)
async def chat_stream(request: ExplainerChatRequest) -> StreamingResponse:
    """Stream chat response using SSE.

    Provides real-time token-by-token streaming for a ChatGPT-like experience.

    Args:
        request: Chat request with message and optional session ID.

    Returns:
        StreamingResponse with SSE events.
    """

    async def event_generator():
        """Generate SSE events from the streaming response."""
        try:
            service = get_system_explainer_service()
            async for event in service.chat_stream(request):
                event_type = event.get("event", "message")
                data = json.dumps(event.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception as e:
            logger.exception("Error in streaming chat: %s", e)
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post(
    "/quick-explain",
    response_model=QuickExplainResponse,
    summary="Get Quick Explanation",
    description="""
Get a focused explanation of a specific topic without conversation context.

Useful for tooltips, info panels, or quick reference.

**Available Topics:**
- `elevator_pitch` - 30-second overview
- `memory_system` - 4-layer memory architecture
- `educational_theories` - 7-theory orchestration
- `agent_system` - Specialized AI agents
- `business_model` - B2B SaaS model
- `technical_architecture` - Tech stack details
- `competitive_advantages` - What makes us different
- `lumi_companion` - The LUMI learning companion
""",
)
async def quick_explain(
    request: QuickExplainRequest,
) -> QuickExplainResponse:
    """Get a quick explanation of a specific topic.

    Args:
        request: Request with topic ID and audience type.

    Returns:
        Focused explanation of the requested topic.
    """
    try:
        service = get_system_explainer_service()
        response = await service.quick_explain(request)
        return response

    except Exception as e:
        logger.exception("Error in quick explain: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred generating the explanation.",
        )


@router.get(
    "/topics",
    response_model=ExplainerTopicsResponse,
    summary="List Available Topics",
    description="Get a list of all topics that can be explained.",
)
async def list_topics() -> ExplainerTopicsResponse:
    """List available topics for explanation.

    Returns:
        List of topics with IDs and descriptions.
    """
    topics = [
        {"id": topic_id, **topic_info}
        for topic_id, topic_info in AVAILABLE_TOPICS.items()
    ]
    return ExplainerTopicsResponse(topics=topics)


@router.get(
    "/sessions/{session_id}",
    summary="Get Session Info",
    description="Get information about a chat session.",
)
async def get_session(session_id: UUID) -> dict:
    """Get session information.

    Args:
        session_id: Session identifier.

    Returns:
        Session information including message count and timestamps.

    Raises:
        HTTPException: If session not found.
    """
    service = get_system_explainer_service()
    session_info = service.get_session_info(session_id)

    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return session_info


@router.delete(
    "/sessions/{session_id}",
    summary="Clear Session",
    description="Clear a chat session and its history.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_session(session_id: UUID) -> None:
    """Clear a session.

    Args:
        session_id: Session to clear.

    Raises:
        HTTPException: If session not found.
    """
    service = get_system_explainer_service()
    cleared = service.clear_session(session_id)

    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )


@router.get(
    "/demo-questions",
    summary="Get Demo Questions",
    description="Get suggested questions for demo/exploration.",
)
async def get_demo_questions() -> dict:
    """Get suggested questions for exploring EduSynapseOS.

    Returns a curated list of questions organized by audience type.
    """
    return {
        "investor_questions": [
            "What is EduSynapseOS and what problem does it solve?",
            "What's your competitive moat against other EdTech companies?",
            "How does the 4-layer memory system create defensibility?",
            "What's the business model and target market?",
            "How do you measure learning effectiveness?",
        ],
        "educator_questions": [
            "How does EduSynapseOS apply educational theories?",
            "What is the Zone of Proximal Development and how do you use it?",
            "How does the mastery learning approach work?",
            "How does the AI adapt to different learning styles?",
            "Can you explain the Socratic method implementation?",
        ],
        "technical_questions": [
            "What's the technical architecture of EduSynapseOS?",
            "How does the multi-tenant isolation work?",
            "How do you handle the 4-layer memory system at scale?",
            "What AI/ML models power the platform?",
            "How do the specialized agents coordinate?",
        ],
        "general_questions": [
            "What is EduSynapseOS?",
            "How is this different from Khan Academy or other learning apps?",
            "What is LUMI and what does it do?",
            "How does the AI remember each student?",
            "Can you give me the 30-second pitch?",
        ],
    }


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the explainer service is healthy and knowledge base is indexed.",
)
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Service health status including knowledge base index status.
    """
    try:
        service = get_system_explainer_service()
        is_indexed = await service.is_knowledge_indexed()
        return {
            "status": "healthy",
            "service": "system_explainer",
            "topics_available": len(AVAILABLE_TOPICS),
            "knowledge_indexed": is_indexed,
            "rag_enabled": True,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy",
        )


@router.post(
    "/index-knowledge",
    summary="Index Knowledge Base",
    description="""
Index the knowledge base documents into the vector database for RAG retrieval.

This endpoint should be called:
- Once during initial setup
- When knowledge base documents are updated

**Note:** This is an admin operation and may take a few seconds to complete.
""",
)
async def index_knowledge_base(force_reindex: bool = False) -> dict:
    """Index knowledge base documents into Qdrant.

    Args:
        force_reindex: If True, delete existing index and recreate.

    Returns:
        Indexing result with number of chunks indexed.
    """
    try:
        service = get_system_explainer_service()
        chunks_indexed = await service.index_knowledge_base(force_reindex=force_reindex)
        return {
            "success": True,
            "chunks_indexed": chunks_indexed,
            "force_reindex": force_reindex,
            "message": f"Successfully indexed {chunks_indexed} chunks" if chunks_indexed > 0 else "Knowledge base already indexed",
        }
    except Exception as e:
        logger.exception("Error indexing knowledge base: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index knowledge base: {str(e)}",
        )
