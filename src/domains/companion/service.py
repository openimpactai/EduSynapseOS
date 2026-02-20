# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion service using LangGraph workflow.

This service manages companion conversations by orchestrating the CompanionWorkflow.
It handles session lifecycle, message exchanges, conversation persistence,
and emotional signal recording.

Architecture:
    - Uses CompanionWorkflow with interrupt_before=["wait_for_message"]
    - On start: Workflow runs to generate greeting, then pauses
    - On message: Uses aupdate_state + ainvoke(None) to resume
    - All messages persisted to conversation_messages table
    - Emotional signals recorded via fire-and-forget pattern
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.emotional import EmotionalStateService
from src.core.intelligence.llm import LLMClient
from src.core.memory.manager import MemoryManager
from src.core.orchestration.states.companion import create_initial_companion_state
from src.core.orchestration.workflows.companion import CompanionWorkflow
from src.core.personas.manager import PersonaManager
from src.core.proactive.service import ProactiveService
from src.domains.companion.schemas import (
    CompanionChatRequest,
    CompanionChatResponse,
    CompanionMessageResponse,
    CompanionMessagesResponse,
    CompanionSessionResponse,
    EmotionalState,
)
from src.core.tools import ToolRegistry, UIElement
from src.tools import get_default_tool_registry
from src.infrastructure.database.models.tenant.companion import CompanionSession
from src.infrastructure.database.models.tenant.conversation import (
    Conversation,
    ConversationMessage,
)

logger = logging.getLogger(__name__)


class CompanionServiceError(Exception):
    """Base exception for companion service errors."""

    pass


class SessionNotFoundError(CompanionServiceError):
    """Raised when a session is not found."""

    pass


class SessionNotActiveError(CompanionServiceError):
    """Raised when session is not in active state."""

    pass


class CompanionService:
    """Service for managing companion conversations.

    This service orchestrates companion conversations using the CompanionWorkflow.
    It manages session state, message persistence, and AI interactions.

    The workflow pattern matches ConversationService (tutoring):
    - Uses interrupt_before for pause/resume
    - Uses aupdate_state + ainvoke(None) for message handling
    - Full 4-layer memory integration
    - Tool calling for actions
    - ProactiveService integration for pending alerts

    Attributes:
        _db: Async database session.
        _workflow: The companion workflow instance.
        _memory_manager: Memory manager for student context.
        _emotional_service: Service for emotional signal recording.
        _proactive_service: Service for proactive alerts.
    """

    def __init__(
        self,
        db: AsyncSession,
        llm_client: LLMClient,
        memory_manager: MemoryManager,
        persona_manager: PersonaManager,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: EmotionalStateService | None = None,
        proactive_service: ProactiveService | None = None,
        tool_registry: ToolRegistry | None = None,
        event_tracker: "EventTracker | None" = None,
    ) -> None:
        """Initialize the companion service.

        Args:
            db: Async database session.
            llm_client: LLM client for completions.
            memory_manager: Manager for 4-layer memory operations.
            persona_manager: Manager for personas.
            checkpointer: Checkpointer for workflow state persistence.
            emotional_service: Service for emotional signal recording.
            proactive_service: Service for proactive alerts.
            tool_registry: Registry of companion tools.
            event_tracker: EventTracker for publishing analytics events.
        """
        self._db = db
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager
        self._proactive_service = proactive_service

        # Initialize emotional service
        if emotional_service is None:
            emotional_service = EmotionalStateService(db=db)
        self._emotional_service = emotional_service

        # Initialize tool registry
        # NOTE: If None, let CompanionWorkflow create from its YAML config
        # Do NOT use get_default_tool_registry() here - that returns ALL 29 tools!
        self._tool_registry = tool_registry

        # Initialize workflow
        # If tool_registry is None, workflow will create from companion.yaml config
        self._workflow = CompanionWorkflow(
            llm_client=llm_client,
            memory_manager=memory_manager,
            persona_manager=persona_manager,
            tool_registry=tool_registry,  # None = use config, not default 29 tools
            checkpointer=checkpointer,
            emotional_service=emotional_service,
            proactive_service=proactive_service,
            event_tracker=event_tracker,
        )

    async def chat(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: CompanionChatRequest,
        grade_level: int = 5,
        framework_code: str | None = None,
        grade_code: str | None = None,
        language: str = "en",
        student_name: str = "there",
    ) -> CompanionChatResponse:
        """Unified chat endpoint for companion interactions.

        Handles both new sessions and continuing conversations.
        For new sessions, generates a proactive greeting.
        For continuing, processes the student message and generates response.

        Args:
            user_id: Student's user ID.
            tenant_id: Tenant UUID.
            tenant_code: Tenant code for memory operations.
            request: Chat request with session_id, message, trigger.
            grade_level: Student's grade level sequence (1-12).
            framework_code: Curriculum framework code (e.g., "UK-NC-2014").
            grade_code: Grade code within framework (e.g., "Y5").
            language: Student's language preference.
            student_name: Student's first name for personalization.

        Returns:
            CompanionChatResponse with message and actions.
        """
        print(f"[DEBUG] chat: session_id={request.session_id}, message={request.message[:50] if request.message else None}", flush=True)
        if request.session_id and request.message:
            # Continuing conversation
            print(f"[DEBUG] Continuing conversation for session {request.session_id}", flush=True)
            return await self._continue_conversation(
                session_id=UUID(request.session_id),
                user_id=user_id,
                message=request.message,
                context=request.context,
            )
        else:
            # New session
            return await self._start_session(
                user_id=user_id,
                tenant_id=tenant_id,
                tenant_code=tenant_code,
                trigger=request.trigger,
                context=request.context,
                grade_level=grade_level,
                framework_code=framework_code,
                grade_code=grade_code,
                language=language,
                student_name=student_name,
            )

    async def _start_session(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        trigger: str,
        context: dict[str, Any] | None,
        grade_level: int,
        framework_code: str | None,
        grade_code: str | None,
        language: str,
        student_name: str = "there",
    ) -> CompanionChatResponse:
        """Start a new companion session with proactive greeting.

        Creates database records and runs workflow to generate greeting.

        Args:
            user_id: Student's user ID.
            tenant_id: Tenant UUID.
            tenant_code: Tenant code.
            trigger: What triggered this session.
            context: Additional context.
            grade_level: Student's grade level sequence.
            framework_code: Curriculum framework code (e.g., "UK-NC-2014").
            grade_code: Grade code within framework (e.g., "Y5").
            language: Student's language preference.
            student_name: Student's first name for personalization.

        Returns:
            CompanionChatResponse with greeting.
        """
        session_id = uuid4()
        conversation_id = uuid4()
        thread_id = f"companion_{session_id}"

        logger.info(
            "Starting companion session: session=%s, user=%s, trigger=%s",
            session_id,
            user_id,
            trigger,
        )

        now = datetime.utcnow()

        # Create conversation record
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            conversation_type="companion",
            persona_id="companion",
            title="Companion Conversation",
            status="active",
            message_count=0,
        )
        self._db.add(conversation)

        # Create companion session record
        session = CompanionSession(
            id=session_id,
            student_id=user_id,
            conversation_id=conversation_id,
            session_type=trigger,
            emotional_state_start=None,
            status="active",
            started_at=now,
        )
        self._db.add(session)
        await self._db.flush()

        # Create initial workflow state
        initial_state = create_initial_companion_state(
            session_id=str(session_id),
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            student_id=str(user_id),
            grade_level=grade_level,
            framework_code=framework_code,
            grade_code=grade_code,
            language=language,
            persona_id="companion",
            student_name=student_name,
        )

        # Set database session for tool execution
        self._workflow.set_db_session(self._db)

        # Run workflow - generates greeting and pauses at wait_for_message
        print(f"[DEBUG] Running workflow with thread_id={thread_id}", flush=True)
        state = await self._workflow.run(initial_state, thread_id)
        print(f"[DEBUG] Workflow returned state keys: {list(state.keys()) if state else 'None'}", flush=True)
        print(f"[DEBUG] Workflow state type: {type(state)}", flush=True)

        # Get greeting from workflow state
        greeting = state.get("first_greeting", "Hello! How are you doing today?")
        print(f"[DEBUG] Extracted greeting: {greeting[:100] if greeting else 'None'}...", flush=True)

        # Save greeting as first message
        greeting_message = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=greeting,
            created_at=now,
        )
        self._db.add(greeting_message)
        conversation.message_count += 1
        conversation.last_message_at = now

        await self._db.commit()

        logger.info(
            "Session started: session=%s, greeting_length=%d",
            session_id,
            len(greeting),
        )

        # Build response with any suggestions from greeting
        suggestions = self._extract_ui_elements(state)
        tool_data = self._extract_tool_data(state)
        handoff = self._extract_handoff(state)

        return CompanionChatResponse(
            session_id=str(session_id),
            message=greeting,
            suggestions=suggestions,
            tool_data=tool_data,
            handoff=handoff,
            emotional_state=None,
            metadata={"trigger": trigger},
        )

    async def _continue_conversation(
        self,
        session_id: UUID,
        user_id: UUID,
        message: str,
        context: dict[str, Any] | None,
    ) -> CompanionChatResponse:
        """Continue an existing companion conversation.

        Processes student message through workflow and generates response.

        Args:
            session_id: Session ID.
            user_id: Student's user ID.
            message: Student's message.
            context: Additional context.

        Returns:
            CompanionChatResponse with companion's response.
        """
        # Get session
        stmt = select(CompanionSession).where(
            CompanionSession.id == str(session_id),
            CompanionSession.student_id == str(user_id),
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        if session.status != "active":
            raise SessionNotActiveError(f"Session {session_id} is not active")

        thread_id = f"companion_{session_id}"
        now = datetime.utcnow()

        logger.info(
            "Continuing conversation: session=%s, message_length=%d",
            session_id,
            len(message),
        )

        # Save student message
        student_message = ConversationMessage(
            id=uuid4(),
            conversation_id=session.conversation_id,
            role="user",
            content=message,
            created_at=now,
        )
        self._db.add(student_message)

        # Get conversation for updating
        conv_stmt = select(Conversation).where(
            Conversation.id == session.conversation_id
        )
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one()
        conversation.message_count += 1
        conversation.last_message_at = now

        # Set database session for tool execution
        self._workflow.set_db_session(self._db)

        # Send message to workflow
        print(f"[DEBUG] Service calling workflow.send_message for thread={thread_id}", flush=True)
        state = await self._workflow.send_message(thread_id, message)
        print(f"[DEBUG] Workflow returned state keys: {list(state.keys()) if state else 'None'}", flush=True)

        # Get response from workflow
        response_text = state.get(
            "last_companion_response",
            "I'm here for you. Is there anything you'd like to talk about?",
        )
        print(f"[DEBUG] last_companion_response: {response_text[:100] if response_text else 'None'}...", flush=True)

        # Save companion response
        companion_message = ConversationMessage(
            id=uuid4(),
            conversation_id=session.conversation_id,
            role="assistant",
            content=response_text,
            created_at=datetime.utcnow(),
        )
        self._db.add(companion_message)
        conversation.message_count += 1
        conversation.last_message_at = datetime.utcnow()

        # Update session
        session.interaction_count = (session.interaction_count or 0) + 1
        session.last_interaction_at = datetime.utcnow()

        await self._db.commit()

        logger.info(
            "Conversation continued: session=%s, response_length=%d",
            session_id,
            len(response_text),
        )

        # Extract suggestions, emotional state, and handoff
        suggestions = self._extract_ui_elements(state)
        emotional_state = self._extract_emotional_state(state)
        tool_data = self._extract_tool_data(state)
        handoff = self._extract_handoff(state)

        return CompanionChatResponse(
            session_id=str(session_id),
            message=response_text,
            suggestions=suggestions,
            tool_data=tool_data,
            handoff=handoff,
            emotional_state=emotional_state,
            metadata={
                "tool_calls": state.get("tool_call_count", 0),
            },
        )

    async def get_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> CompanionSessionResponse:
        """Get session information.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.

        Returns:
            CompanionSessionResponse.

        Raises:
            SessionNotFoundError: If session not found.
        """
        stmt = select(CompanionSession).where(
            CompanionSession.id == str(session_id),
            CompanionSession.student_id == str(user_id),
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        # Get conversation for message count
        conv_stmt = select(Conversation).where(
            Conversation.id == session.conversation_id
        )
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        return CompanionSessionResponse(
            session_id=str(session.id),
            conversation_id=str(session.conversation_id) if session.conversation_id else "",
            status=session.status,
            message_count=conversation.message_count if conversation else 0,
            started_at=session.started_at,
            last_message_at=conversation.last_message_at if conversation else None,
        )

    async def get_messages(
        self,
        session_id: UUID,
        user_id: UUID,
        limit: int = 50,
        before_id: UUID | None = None,
    ) -> CompanionMessagesResponse:
        """Get message history for a session.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.
            limit: Maximum messages to return.
            before_id: Get messages before this ID.

        Returns:
            CompanionMessagesResponse with messages.

        Raises:
            SessionNotFoundError: If session not found.
        """
        # Verify session access
        session_stmt = select(CompanionSession).where(
            CompanionSession.id == str(session_id),
            CompanionSession.student_id == str(user_id),
        )
        session_result = await self._db.execute(session_stmt)
        session = session_result.scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        # Get messages
        msg_stmt = select(ConversationMessage).where(
            ConversationMessage.conversation_id == session.conversation_id,
        )

        if before_id:
            before_stmt = select(ConversationMessage.created_at).where(
                ConversationMessage.id == str(before_id)
            )
            before_result = await self._db.execute(before_stmt)
            before_time = before_result.scalar_one_or_none()
            if before_time:
                msg_stmt = msg_stmt.where(
                    ConversationMessage.created_at < before_time
                )

        msg_stmt = msg_stmt.order_by(ConversationMessage.created_at.desc())
        msg_stmt = msg_stmt.limit(limit + 1)

        result = await self._db.execute(msg_stmt)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        messages.reverse()

        return CompanionMessagesResponse(
            messages=[
                CompanionMessageResponse(
                    id=str(msg.id),
                    role="companion" if msg.role == "assistant" else "student",
                    content=msg.content,
                    emotional_context=msg.emotional_context,
                    created_at=msg.created_at,
                )
                for msg in messages
            ],
            has_more=has_more,
        )

    async def end_session(
        self,
        session_id: UUID,
        user_id: UUID,
        final_mood: str | None = None,
    ) -> None:
        """End a companion session.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.
            final_mood: Optional final emotional state.

        Raises:
            SessionNotFoundError: If session not found.
        """
        stmt = select(CompanionSession).where(
            CompanionSession.id == str(session_id),
            CompanionSession.student_id == str(user_id),
        )
        result = await self._db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        session.end_session(final_emotional_state=final_mood)

        # Also end the conversation
        if session.conversation_id:
            conv_stmt = select(Conversation).where(
                Conversation.id == session.conversation_id
            )
            conv_result = await self._db.execute(conv_stmt)
            conversation = conv_result.scalar_one_or_none()
            if conversation:
                conversation.status = "completed"

        await self._db.commit()

        logger.info("Session ended: session=%s", session_id)

    def _extract_handoff(self, state: dict) -> dict[str, Any] | None:
        """Extract handoff information from workflow state.

        Handoff occurs when the companion transfers to another agent or module.
        Returns data in the format expected by the frontend:
        {
            "target": "practice" | "tutor" | "learning" | "game",
            "params": { topic_code, topic_name, subject_code, ... },
            "message": "optional message"
        }

        Args:
            state: Workflow state.

        Returns:
            Handoff dict formatted for frontend, or None.
        """
        pending_actions = state.get("pending_actions", [])

        for action in pending_actions:
            if action.get("type") == "handoff":
                params = action.get("params", {})

                # Determine handoff target from params
                target_module = params.get("target_module", "")
                target_agent = params.get("target_agent", "")

                # Map to frontend expected targets
                target = "practice"  # default
                if target_module == "practice":
                    target = "practice"
                elif target_module == "learning" or target_agent == "learning_tutor":
                    target = "learning"
                elif target_agent == "tutor":
                    target = "tutor"
                elif target_module == "game":
                    target = "game"

                # Build params dict (exclude target_module/target_agent)
                handoff_params = {
                    k: v for k, v in params.items()
                    if k not in ("target_module", "target_agent")
                }

                # Return in frontend expected format
                return {
                    "target": target,
                    "params": handoff_params if handoff_params else None,
                }

        return None

    def _extract_emotional_state(self, state: dict) -> EmotionalState | None:
        """Extract emotional state from workflow state.

        Args:
            state: Workflow state.

        Returns:
            EmotionalState or None.
        """
        pending_signals = state.get("pending_emotional_signals", [])

        if not pending_signals:
            return None

        # Use the first signal
        signal = pending_signals[0]
        return EmotionalState(
            emotion=signal.get("emotion", "neutral"),
            intensity=signal.get("intensity", "medium"),
            triggers=signal.get("triggers", []),
        )

    def _extract_ui_elements(self, state: dict) -> list[UIElement]:
        """Extract UI elements from workflow state.

        UI elements are returned by tools for structured frontend interactions
        (selections, confirmations, etc.).

        Args:
            state: Workflow state.

        Returns:
            List of UIElement objects for frontend rendering.
        """
        ui_elements_data = state.get("ui_elements", [])

        if not ui_elements_data:
            return []

        ui_elements = []
        for element_data in ui_elements_data:
            try:
                # element_data is already a dict from DynamicAgent
                ui_elements.append(UIElement.model_validate(element_data))
            except Exception as e:
                logger.warning("Failed to parse UI element: %s", e)

        return ui_elements

    def _extract_tool_data(self, state: dict) -> dict[str, Any]:
        """Extract tool data from workflow state.

        Tool data contains raw data from tools that the frontend may need
        for display or further processing (e.g., full list of subjects).

        Args:
            state: Workflow state.

        Returns:
            Dictionary of tool data for frontend.
        """
        return state.get("tool_data", {})
