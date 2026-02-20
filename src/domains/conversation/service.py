# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conversation and tutoring service.

This service manages tutoring conversations by orchestrating the TutoringWorkflow.
It handles conversation lifecycle, message exchanges, and session state.

The service does NOT make LLM calls directly - all AI interactions happen
through the workflow which uses the Agent layer.

Architecture:
    - Uses TutoringWorkflow with interrupt_before=["wait_for_message"]
    - On start: Workflow runs to generate greeting, then pauses
    - On message: Uses aupdate_state + ainvoke(None) to resume
    - Matches Practice service patterns for consistency
"""

import logging
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.agents import AgentFactory
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.emotional import EmotionalStateService, EmotionalSignalSource
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.orchestration.states.tutoring import (
    create_initial_tutoring_state,
    TutoringState,
)
from src.core.orchestration.workflows.tutoring import TutoringWorkflow
from src.core.personas.manager import PersonaManager
from src.infrastructure.database.models.tenant.conversation import (
    Conversation as ConversationModel,
    ConversationMessage as MessageModel,
    ConversationSummary as SummaryModel,
)
from src.infrastructure.database.models.tenant.curriculum import Topic as TopicModel
from src.models.conversation import (
    StartConversationRequest,
    ConversationResponse,
    ConversationSummary,
    SendMessageRequest,
    MessageResponse,
    MessageListResponse,
    StreamingChunk,
    ConversationStatus,
    ConversationType,
    MessageRole,
)

logger = logging.getLogger(__name__)


class ConversationServiceError(Exception):
    """Base exception for conversation service errors."""

    pass


class ConversationNotFoundError(ConversationServiceError):
    """Raised when a conversation is not found."""

    pass


class ConversationNotActiveError(ConversationServiceError):
    """Raised when conversation is not in active state."""

    pass


class ConversationService:
    """Service for managing tutoring conversations.

    This service orchestrates tutoring conversations using the TutoringWorkflow.
    It manages conversation state, message exchanges, and AI interactions.

    The workflow pattern matches Practice service:
    - Uses interrupt_before for pause/resume
    - Uses aupdate_state + ainvoke(None) for message handling
    - Full 4-layer memory integration
    - Theory-driven personalization

    The service does NOT make LLM calls - all AI interactions are handled
    by the TutoringWorkflow through the Agent layer.

    Attributes:
        _db: Async database session.
        _workflow: The tutoring workflow instance.
        _memory_manager: Memory manager for student context.

    Example:
        >>> service = ConversationService(db, agent_factory, memory_manager, ...)
        >>> conv, greeting = await service.start_conversation(user_id, tenant_id, request)
        >>> response = await service.send_message(conv.id, user_id, message)
    """

    def __init__(
        self,
        db: AsyncSession,
        agent_factory: AgentFactory,
        memory_manager: MemoryManager,
        rag_retriever: RAGRetriever,
        theory_orchestrator: TheoryOrchestrator,
        persona_manager: PersonaManager,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: EmotionalStateService | None = None,
    ) -> None:
        """Initialize the conversation service.

        Args:
            db: Async database session.
            agent_factory: Factory for creating agents.
            memory_manager: Manager for 4-layer memory operations.
            rag_retriever: Retriever for RAG context.
            theory_orchestrator: Orchestrator for educational theories.
            persona_manager: Manager for personas.
            checkpointer: Checkpointer for workflow state persistence (required).
            emotional_service: Service for emotional signal recording.
        """
        self._db = db
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager

        # Initialize emotional service (create if not provided)
        if emotional_service is None:
            emotional_service = EmotionalStateService(db=db)
        self._emotional_service = emotional_service

        # Initialize workflow with checkpointer and emotional service
        self._workflow = TutoringWorkflow(
            agent_factory=agent_factory,
            memory_manager=memory_manager,
            rag_retriever=rag_retriever,
            theory_orchestrator=theory_orchestrator,
            persona_manager=persona_manager,
            checkpointer=checkpointer,
            emotional_service=self._emotional_service,
        )

    async def start_conversation(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: StartConversationRequest,
    ) -> tuple[ConversationResponse, MessageResponse | None]:
        """Start a new conversation with proactive greeting.

        Creates a new conversation and runs the workflow to generate
        a personalized greeting. The workflow:
        1. Initializes session
        2. Loads 4-layer memory context
        3. Gets theory recommendations
        4. Generates personalized greeting
        5. Pauses at wait_for_message (interrupt point)

        If initial_message is provided, it's processed as the first
        student message after the greeting.

        Args:
            user_id: The user's ID.
            tenant_id: The tenant's ID (for database operations).
            tenant_code: The tenant code (for MemoryManager, RAG, etc.).
            request: Conversation configuration request.

        Returns:
            Tuple of (conversation response, first AI greeting or None).
        """
        conversation_id = uuid4()
        thread_id = f"tutoring_{conversation_id}"

        logger.info(
            "Starting conversation: id=%s, user=%s, type=%s, persona=%s",
            conversation_id,
            user_id,
            request.conversation_type.value,
            request.persona_id,
        )

        # Get persona name if specified
        persona_name = None
        if request.persona_id:
            try:
                persona = self._persona_manager.get_persona(request.persona_id)
                persona_name = persona.identity.name if persona.identity else None
            except Exception:
                pass

        # Query topic name if topic codes are provided
        topic_name = None
        if all([
            request.topic_framework_code,
            request.topic_subject_code,
            request.topic_grade_code,
            request.topic_unit_code,
            request.topic_code,
        ]):
            topic_stmt = select(TopicModel.name).where(
                TopicModel.framework_code == request.topic_framework_code,
                TopicModel.subject_code == request.topic_subject_code,
                TopicModel.grade_code == request.topic_grade_code,
                TopicModel.unit_code == request.topic_unit_code,
                TopicModel.code == request.topic_code,
            )
            topic_result = await self._db.execute(topic_stmt)
            topic_name = topic_result.scalar_one_or_none()

        # Create database record with composite topic keys
        db_conversation = ConversationModel(
            id=conversation_id,
            user_id=user_id,
            conversation_type=request.conversation_type.value,
            topic_framework_code=request.topic_framework_code,
            topic_subject_code=request.topic_subject_code,
            topic_grade_code=request.topic_grade_code,
            topic_unit_code=request.topic_unit_code,
            topic_code=request.topic_code,
            persona_id=request.persona_id,
            title=request.title,
            status=ConversationStatus.ACTIVE.value,
            message_count=0,
        )
        self._db.add(db_conversation)
        await self._db.flush()

        # Create initial workflow state with topic name
        initial_state = create_initial_tutoring_state(
            session_id=str(conversation_id),
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            student_id=str(user_id),
            topic=topic_name or request.title or "General",
            subtopic=None,
            persona_id=request.persona_id or "tutor",
        )

        # Run workflow - generates greeting and pauses at wait_for_message
        state = await self._workflow.run(initial_state, thread_id)

        # Get the generated greeting from workflow state
        first_greeting = state.get("first_greeting", "")
        first_response = None

        if first_greeting:
            # Save greeting as first AI message
            now = datetime.utcnow()
            greeting_message = MessageModel(
                id=uuid4(),
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT.value,
                content=first_greeting,
                created_at=now,
            )
            self._db.add(greeting_message)
            db_conversation.message_count += 1
            db_conversation.last_message_at = now
            await self._db.flush()  # Ensure message is persisted

            first_response = self._build_message_response(greeting_message)

            logger.info(
                "Generated greeting: conversation=%s, length=%d",
                conversation_id,
                len(first_greeting),
            )

        # If initial message provided, process it after greeting
        if request.initial_message:
            logger.info(
                "Processing initial message: conversation=%s",
                conversation_id,
            )
            # Send the initial message through the workflow
            response = await self._process_message(
                conversation_id=conversation_id,
                user_id=user_id,
                thread_id=thread_id,
                content=request.initial_message,
                db_conversation=db_conversation,
            )
            # Return the response to initial message instead of greeting
            first_response = response

        await self._db.commit()

        # Build response (topic_name was already queried at the beginning)
        conversation_response = self._build_conversation_response(
            db_conversation, persona_name, topic_name
        )

        return conversation_response, first_response

    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> ConversationResponse:
        """Get a conversation by ID.

        Args:
            conversation_id: The conversation ID.
            user_id: The user ID (for authorization).

        Returns:
            Conversation response.

        Raises:
            ConversationNotFoundError: If conversation not found.
        """
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.topic))
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
        )
        result = await self._db.execute(stmt)
        db_conversation = result.scalar_one_or_none()

        if not db_conversation:
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} not found"
            )

        return self._build_conversation_response(db_conversation)

    async def list_conversations(
        self,
        user_id: UUID,
        status: ConversationStatus | None = None,
        conversation_type: ConversationType | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ConversationSummary]:
        """List conversations for a user.

        Args:
            user_id: The user ID.
            status: Optional status filter.
            conversation_type: Optional type filter.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of conversation summaries.
        """
        stmt = select(ConversationModel).where(
            ConversationModel.user_id == user_id,
        )

        if status:
            stmt = stmt.where(ConversationModel.status == status.value)

        if conversation_type:
            stmt = stmt.where(
                ConversationModel.conversation_type == conversation_type.value
            )

        stmt = stmt.order_by(ConversationModel.updated_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._db.execute(stmt)
        conversations = result.scalars().all()

        return [self._build_conversation_summary(conv) for conv in conversations]

    async def send_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        request: SendMessageRequest,
    ) -> MessageResponse:
        """Send a message and get AI response.

        Uses the workflow's send_message which properly:
        1. Updates state with message via aupdate_state
        2. Resumes workflow with ainvoke(None)
        3. Processes through analyze_intent → retrieve_context → generate_response → update_memory

        Args:
            conversation_id: The conversation ID.
            user_id: The user ID.
            request: Message request.

        Returns:
            AI message response.

        Raises:
            ConversationNotFoundError: If conversation not found.
            ConversationNotActiveError: If conversation is not active.
        """
        # Get conversation
        stmt = select(ConversationModel).where(
            ConversationModel.id == conversation_id,
            ConversationModel.user_id == user_id,
        )
        result = await self._db.execute(stmt)
        db_conversation = result.scalar_one_or_none()

        if not db_conversation:
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} not found"
            )

        if db_conversation.status != ConversationStatus.ACTIVE.value:
            raise ConversationNotActiveError(
                f"Conversation {conversation_id} is not active"
            )

        thread_id = f"tutoring_{conversation_id}"

        response = await self._process_message(
            conversation_id=conversation_id,
            user_id=user_id,
            thread_id=thread_id,
            content=request.content,
            db_conversation=db_conversation,
        )

        await self._db.commit()

        return response

    async def stream_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        content: str,
    ) -> AsyncIterator[StreamingChunk]:
        """Send a message and stream the AI response.

        Args:
            conversation_id: The conversation ID.
            user_id: The user ID.
            content: Message content.

        Yields:
            Streaming chunks of the response.
        """
        # For now, we don't have streaming implemented in workflow
        # Return the full response as a single chunk
        request = SendMessageRequest(content=content)
        response = await self.send_message(conversation_id, user_id, request)

        # Simulate streaming with single chunk
        yield StreamingChunk(
            conversation_id=conversation_id,
            message_id=response.id,
            chunk=response.content,
            is_complete=True,
            tokens_so_far=response.tokens_output,
        )

    async def get_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit: int = 50,
        before_id: UUID | None = None,
    ) -> MessageListResponse:
        """Get message history for a conversation.

        Args:
            conversation_id: The conversation ID.
            user_id: The user ID (for authorization).
            limit: Maximum messages to return.
            before_id: Get messages before this ID.

        Returns:
            Message list response.
        """
        # Verify conversation access
        await self.get_conversation(conversation_id, user_id)

        stmt = select(MessageModel).where(
            MessageModel.conversation_id == conversation_id,
        )

        if before_id:
            # Get the created_at of the before message
            before_stmt = select(MessageModel.created_at).where(
                MessageModel.id == before_id
            )
            before_result = await self._db.execute(before_stmt)
            before_time = before_result.scalar_one_or_none()

            if before_time:
                stmt = stmt.where(MessageModel.created_at < before_time)

        stmt = stmt.order_by(MessageModel.created_at.desc())
        stmt = stmt.limit(limit + 1)  # Get one extra to check if more exist

        result = await self._db.execute(stmt)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        # Reverse to get chronological order
        messages.reverse()

        return MessageListResponse(
            messages=[self._build_message_response(msg) for msg in messages],
            has_more=has_more,
            oldest_id=messages[0].id if messages else None,
        )

    async def archive_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> ConversationResponse:
        """Archive a conversation.

        Args:
            conversation_id: The conversation ID.
            user_id: The user ID.

        Returns:
            Updated conversation response.
        """
        stmt = (
            update(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
            .values(status=ConversationStatus.ARCHIVED.value)
            .returning(ConversationModel)
        )

        result = await self._db.execute(stmt)
        db_conversation = result.scalar_one_or_none()

        if not db_conversation:
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} not found"
            )

        await self._db.commit()

        # Get topic name if topic codes exist
        topic_name = None
        if all([
            db_conversation.topic_framework_code,
            db_conversation.topic_subject_code,
            db_conversation.topic_grade_code,
            db_conversation.topic_unit_code,
            db_conversation.topic_code,
        ]):
            topic_stmt = select(TopicModel.name).where(
                TopicModel.framework_code == db_conversation.topic_framework_code,
                TopicModel.subject_code == db_conversation.topic_subject_code,
                TopicModel.grade_code == db_conversation.topic_grade_code,
                TopicModel.unit_code == db_conversation.topic_unit_code,
                TopicModel.code == db_conversation.topic_code,
            )
            topic_result = await self._db.execute(topic_stmt)
            topic_name = topic_result.scalar_one_or_none()

        return self._build_conversation_response(db_conversation, topic_name=topic_name)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _process_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        thread_id: str,
        content: str,
        db_conversation: ConversationModel,
    ) -> MessageResponse:
        """Process a user message and get AI response.

        Uses workflow.send_message() which:
        1. Injects message via aupdate_state
        2. Resumes workflow with ainvoke(None)
        3. Full processing: analyze → retrieve → respond → update memory

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.
            thread_id: Workflow thread ID.
            content: Message content.
            db_conversation: Database conversation model.

        Returns:
            AI message response.
        """
        logger.info(
            "Processing message: conversation=%s, length=%d",
            conversation_id,
            len(content),
        )

        # Save user message
        now = datetime.utcnow()
        user_message = MessageModel(
            id=uuid4(),
            conversation_id=conversation_id,
            role=MessageRole.USER.value,
            content=content,
            created_at=now,
        )
        self._db.add(user_message)
        db_conversation.message_count += 1
        db_conversation.last_message_at = now

        # Send to workflow and get response
        # Workflow uses aupdate_state + ainvoke(None) pattern
        state = await self._workflow.send_message(thread_id, content)

        # Record emotional signal from workflow's analysis (fire-and-forget)
        await self._record_emotional_signal_from_analysis(
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
            analysis=state.get("last_message_analysis"),
            topic=db_conversation.title,
        )

        # Extract response from state
        ai_response_text = state.get("last_tutor_response", "")

        if not ai_response_text:
            # Fallback if no response generated
            ai_response_text = "I'm sorry, I couldn't generate a response. Please try again."
            logger.warning(
                "No response generated: conversation=%s, error=%s",
                conversation_id,
                state.get("error"),
            )

        # Save AI message
        ai_now = datetime.utcnow()
        ai_message = MessageModel(
            id=uuid4(),
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT.value,
            content=ai_response_text,
            model=state.get("model_used"),
            tokens_input=state.get("tokens_input"),
            tokens_output=state.get("tokens_output"),
            created_at=ai_now,
        )
        self._db.add(ai_message)
        db_conversation.message_count += 1
        db_conversation.last_message_at = ai_now

        # Update workflow state in conversation (for session summary)
        db_conversation.workflow_state = {
            "metrics": state.get("metrics", {}),
            "concepts_covered": state.get("metrics", {}).get("concepts_covered", []),
            "mastery_updates": state.get("mastery_updates", {}),
        }

        await self._db.flush()

        return self._build_message_response(ai_message)

    def _build_conversation_response(
        self,
        db_conversation: ConversationModel,
        persona_name: str | None = None,
        topic_name: str | None = None,
    ) -> ConversationResponse:
        """Build conversation response from database model.

        Args:
            db_conversation: The conversation database model.
            persona_name: Optional persona name (if known).
            topic_name: Optional topic name. If not provided and topic relationship
                        is loaded, uses topic.name. Pass explicitly to avoid lazy loading.
        """
        # Use provided topic_name or get from eagerly-loaded relationship
        resolved_topic_name = topic_name
        if resolved_topic_name is None and db_conversation.topic is not None:
            # Only access if already loaded (selectinload was used)
            resolved_topic_name = db_conversation.topic.name

        return ConversationResponse(
            id=db_conversation.id,
            user_id=db_conversation.user_id,
            conversation_type=ConversationType(db_conversation.conversation_type),
            topic_full_code=db_conversation.topic_full_code,
            topic_name=resolved_topic_name,
            persona_id=db_conversation.persona_id,
            persona_name=persona_name,
            title=db_conversation.title,
            status=ConversationStatus(db_conversation.status),
            message_count=db_conversation.message_count,
            last_message_at=db_conversation.last_message_at,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
        )

    def _build_conversation_summary(
        self,
        db_conversation: ConversationModel,
    ) -> ConversationSummary:
        """Build conversation summary from database model."""
        return ConversationSummary(
            id=db_conversation.id,
            conversation_type=ConversationType(db_conversation.conversation_type),
            title=db_conversation.title,
            topic_name=None,
            persona_name=None,
            status=ConversationStatus(db_conversation.status),
            message_count=db_conversation.message_count,
            last_message_preview=None,  # Would need last message
            last_message_at=db_conversation.last_message_at,
            created_at=db_conversation.created_at,
        )

    def _build_message_response(
        self,
        db_message: MessageModel,
    ) -> MessageResponse:
        """Build message response from database model."""
        return MessageResponse(
            id=db_message.id,
            conversation_id=db_message.conversation_id,
            role=MessageRole(db_message.role),
            content=db_message.content,
            attachments=[],
            tokens_input=db_message.tokens_input,
            tokens_output=db_message.tokens_output,
            model=db_message.model,
            parent_id=db_message.parent_id,
            created_at=db_message.created_at,
        )

    async def _record_emotional_signal_from_analysis(
        self,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
        analysis: dict[str, Any] | None,
        topic: str | None = None,
    ) -> None:
        """Record emotional signal using pre-analyzed data from workflow.

        The emotional analysis is performed by the emotional_analyzer agent
        in the TutoringWorkflow._analyze_intent node. This method simply
        records that analysis as an emotional signal.

        This is a fire-and-forget operation. Errors are logged but do not
        affect the main chat flow.

        Args:
            user_id: The student's user ID.
            conversation_id: The conversation ID.
            content: The user's message content.
            analysis: Pre-analyzed emotional data from workflow (MessageAnalysis).
            topic: Optional topic being discussed.
        """
        if not analysis:
            logger.debug(
                "No emotional analysis available for conversation=%s",
                conversation_id,
            )
            return

        try:
            context = {}
            if topic:
                context["topic"] = topic
            context["conversation_id"] = str(conversation_id)
            context["analysis_source"] = "emotional_analyzer_agent"

            # Use record_analyzed_signal instead of record_text_signal
            await self._emotional_service.record_analyzed_signal(
                student_id=user_id,
                source=EmotionalSignalSource.LEARNING,
                emotional_state=analysis.get("emotional_state", "neutral"),
                intensity=analysis.get("intensity", "low"),
                confidence=analysis.get("sentiment_confidence", 0.8),
                triggers=analysis.get("triggers", []),
                activity_id=str(conversation_id),
                activity_type="tutoring_conversation",
                context=context,
            )

            logger.debug(
                "Recorded emotional signal from analysis: conversation=%s, state=%s",
                conversation_id,
                analysis.get("emotional_state"),
            )
        except Exception as e:
            logger.warning(
                "Failed to record emotional signal: conversation=%s, error=%s",
                conversation_id,
                str(e),
            )
