# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning and tutoring API endpoints.

This module provides endpoints for tutoring conversations:
- POST /start - Start a new conversation
- GET /{conversation_id} - Get conversation details
- GET / - List conversations
- POST /{conversation_id}/message - Send a message
- WebSocket /{conversation_id}/ws - Real-time learning session
- GET /{conversation_id}/history - Get message history
- POST /{conversation_id}/archive - Archive conversation

Example:
    POST /api/v1/learning/start
    {
        "conversation_type": "tutoring",
        "topic_framework_code": "UK-NC-2014",
        "topic_subject_code": "MAT",
        "topic_grade_code": "Y4",
        "topic_unit_code": "NPV",
        "topic_code": "001",
        "persona_id": "mentor"
    }
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_checkpointer,
    get_tenant_db,
    get_tenant_db_manager,
    require_auth,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.core.agents import AgentFactory
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.personas.manager import PersonaManager
from src.domains.conversation.service import (
    ConversationService,
    ConversationNotFoundError,
    ConversationNotActiveError,
)
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.models.conversation import (
    StartConversationRequest,
    ConversationResponse,
    ConversationSummary,
    SendMessageRequest,
    MessageResponse,
    MessageListResponse,
    ConversationStatus,
    ConversationType,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_conversation_service(
    db: AsyncSession,
    tenant: TenantContext,
    tenant_db_manager: TenantDatabaseManager,
    checkpointer: BaseCheckpointSaver | None,
) -> ConversationService:
    """Get conversation service instance.

    Args:
        db: Tenant database session (for ConversationService DB operations).
        tenant: Tenant context.
        tenant_db_manager: Tenant database manager (for MemoryManager).
        checkpointer: Workflow checkpointer.

    Returns:
        Configured ConversationService instance.
    """
    from src.core.agents.capabilities.registry import get_default_registry
    from src.core.intelligence.embeddings.service import EmbeddingService
    from src.core.intelligence.llm.client import LLMClient
    from src.infrastructure.vectors import get_qdrant

    # Get global Qdrant client (initialized at startup)
    qdrant_client = get_qdrant()

    # Create per-request services
    embedding_service = EmbeddingService()
    llm_client = LLMClient()

    # Create managers with correct parameters
    memory_manager = MemoryManager(
        tenant_db_manager=tenant_db_manager,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )

    rag_retriever = RAGRetriever(
        memory_manager=memory_manager,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
    )

    theory_orchestrator = TheoryOrchestrator()
    persona_manager = PersonaManager()

    # Create agent factory with correct parameters
    capability_registry = get_default_registry()
    agent_factory = AgentFactory(
        llm_client=llm_client,
        capability_registry=capability_registry,
        persona_manager=persona_manager,
    )

    return ConversationService(
        db=db,
        agent_factory=agent_factory,
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        theory_orchestrator=theory_orchestrator,
        persona_manager=persona_manager,
        checkpointer=checkpointer,
    )


class StartConversationResponse(ConversationResponse):
    """Response when starting a conversation."""

    first_response: MessageResponse | None = None


@router.post(
    "/start",
    response_model=StartConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start learning session",
    description="Start a new tutoring conversation. Optionally send initial message.",
)
async def start_conversation(
    data: StartConversationRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> StartConversationResponse:
    """Start a new conversation.

    Args:
        data: Conversation configuration.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        StartConversationResponse with conversation details and optional first response.
    """
    logger.info(
        "Starting learning session: user=%s, type=%s",
        current_user.id,
        data.conversation_type.value,
    )

    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    conversation, first_response = await service.start_conversation(
        user_id=UUID(current_user.id),
        tenant_id=UUID(tenant.id),
        tenant_code=tenant.code,
        request=data,
    )

    # Build response with first response included
    response_data = conversation.model_dump()
    response_data["first_response"] = first_response

    return StartConversationResponse(**response_data)


@router.get(
    "",
    response_model=list[ConversationSummary],
    summary="List learning sessions",
    description="Get list of learning sessions for the current user.",
)
async def list_conversations(
    status: Annotated[ConversationStatus | None, Query()] = None,
    conversation_type: Annotated[ConversationType | None, Query(alias="type")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> list[ConversationSummary]:
    """List learning sessions for current user.

    Args:
        status: Optional status filter.
        conversation_type: Optional type filter.
        limit: Maximum results.
        offset: Pagination offset.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        List of conversation summaries.
    """
    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    return await service.list_conversations(
        user_id=UUID(current_user.id),
        status=status,
        conversation_type=conversation_type,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get learning session",
    description="Get learning session details by ID.",
)
async def get_conversation(
    conversation_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> ConversationResponse:
    """Get conversation details.

    Args:
        conversation_id: The conversation ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        ConversationResponse with conversation details.

    Raises:
        HTTPException: If conversation not found.
    """
    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.get_conversation(
            conversation_id=conversation_id,
            user_id=UUID(current_user.id),
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session not found",
        )


@router.get(
    "/{conversation_id}/history",
    response_model=MessageListResponse,
    summary="Get message history",
    description="Get message history for a learning session with pagination.",
)
async def get_message_history(
    conversation_id: UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    before_id: Annotated[UUID | None, Query()] = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> MessageListResponse:
    """Get message history for a learning session.

    Args:
        conversation_id: The conversation ID.
        limit: Maximum messages to return.
        before_id: Get messages before this ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        MessageListResponse with messages.

    Raises:
        HTTPException: If conversation not found.
    """
    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.get_messages(
            conversation_id=conversation_id,
            user_id=UUID(current_user.id),
            limit=limit,
            before_id=before_id,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session not found",
        )


@router.post(
    "/{conversation_id}/message",
    response_model=MessageResponse,
    summary="Send message",
    description="Send a message and get AI response. For real-time, use WebSocket.",
)
async def send_message(
    conversation_id: UUID,
    data: SendMessageRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> MessageResponse:
    """Send a message and get response.

    Args:
        conversation_id: The conversation ID.
        data: Message request.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        MessageResponse with AI response.

    Raises:
        HTTPException: If conversation not found or not active.
    """
    logger.info(
        "Sending message: conversation=%s, length=%d",
        conversation_id,
        len(data.content),
    )

    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.send_message(
            conversation_id=conversation_id,
            user_id=UUID(current_user.id),
            request=data,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session not found",
        )
    except ConversationNotActiveError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Learning session is not active",
        )


@router.post(
    "/{conversation_id}/archive",
    response_model=ConversationResponse,
    summary="Archive learning session",
    description="Archive a learning session. Can be unarchived later.",
)
async def archive_conversation(
    conversation_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    tenant_db_manager: TenantDatabaseManager = Depends(get_tenant_db_manager),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> ConversationResponse:
    """Archive a learning session.

    Args:
        conversation_id: The conversation ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.
        tenant_db_manager: Tenant database manager.
        checkpointer: Workflow checkpointer.

    Returns:
        Updated conversation response.

    Raises:
        HTTPException: If conversation not found.
    """
    service = _get_conversation_service(db, tenant, tenant_db_manager, checkpointer)

    try:
        return await service.archive_conversation(
            conversation_id=conversation_id,
            user_id=UUID(current_user.id),
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session not found",
        )


@router.websocket("/{conversation_id}/ws")
async def learning_websocket(
    websocket: WebSocket,
    conversation_id: UUID,
) -> None:
    """WebSocket endpoint for real-time learning.

    This endpoint provides streaming responses for messages.
    Authentication is done via query parameter or first message.

    Args:
        websocket: WebSocket connection.
        conversation_id: The conversation ID.
    """
    await websocket.accept()

    try:
        # In a real implementation, we would:
        # 1. Authenticate the connection (via query param or first message)
        # 2. Get tenant context
        # 3. Create service and stream responses

        # For now, send a placeholder message
        await websocket.send_json({
            "type": "connected",
            "conversation_id": str(conversation_id),
            "message": "WebSocket connected. Send messages as JSON with 'content' field.",
        })

        while True:
            data = await websocket.receive_json()

            if "content" not in data:
                await websocket.send_json({
                    "type": "error",
                    "message": "Message must include 'content' field",
                })
                continue

            # In a real implementation, we would:
            # 1. Process the message through the service
            # 2. Stream the response in chunks

            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": f"Received: {data['content']}",
                "is_complete": True,
            })

    except Exception as e:
        logger.warning("WebSocket error: %s", str(e))

    finally:
        try:
            await websocket.close()
        except Exception:
            pass
