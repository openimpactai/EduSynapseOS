# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation service.

This service manages content creation sessions and coordinates
with the LangGraph workflow for interactive content generation.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from src.core.orchestration.states.content import create_initial_content_state
from src.core.orchestration.workflows.content import ContentCreationWorkflow
from src.domains.content_creation.schemas import (
    ContentChatRequest,
    ContentChatResponse,
    ContentMessageResponse,
    ContentMessagesResponse,
    ContentSessionResponse,
    ContentTypeInfo,
    ContentTypesResponse,
    GeneratedContentResponse,
)
from src.services.h5p.converters import ConverterRegistry

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.intelligence.llm import LLMClient

logger = logging.getLogger(__name__)


class ContentSessionNotFoundError(Exception):
    """Raised when a content creation session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Content session not found: {session_id}")


class ContentCreationService:
    """Service for managing content creation sessions.

    This service provides:
    - Starting new content creation sessions
    - Continuing conversations with the workflow
    - Managing generated content
    - Exporting to H5P format

    Attributes:
        db: Database session.
        llm_client: LLM client for agent completions.
        checkpointer: LangGraph checkpointer for state.
        converter_registry: H5P converter registry.
    """

    def __init__(
        self,
        db: "AsyncSession",
        llm_client: "LLMClient",
        checkpointer: "BaseCheckpointSaver | None" = None,
    ):
        """Initialize the content creation service.

        Args:
            db: Database session.
            llm_client: LLM client for agent completions.
            checkpointer: LangGraph checkpointer for state persistence.
        """
        self.db = db
        self.llm_client = llm_client
        self.checkpointer = checkpointer
        self.converter_registry = ConverterRegistry()

        # Create workflow instance
        self._workflow = ContentCreationWorkflow(
            llm_client=llm_client,
            checkpointer=checkpointer,
            db_session=db,
        )

    async def chat(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: ContentChatRequest,
        user_role: str | None = None,
        language: str | None = None,
        country_code: str | None = None,
        framework_code: str | None = None,
        subject_code: str | None = None,
        topic_code: str | None = None,
        grade_level: int | None = None,
    ) -> ContentChatResponse:
        """Process a content creation chat message.

        All curriculum-related parameters (framework_code, subject_code,
        topic_code, grade_level, country_code) should come from external
        sources (user profile, tenant settings, request context) - never
        use hardcoded defaults.

        Args:
            user_id: User making the request.
            tenant_id: Tenant identifier.
            tenant_code: Tenant code.
            request: Chat request with message.
            user_role: User's role (teacher, student, parent).
            language: Content language code.
            country_code: User's country code.
            framework_code: Curriculum framework code from tenant/user.
            subject_code: Subject code if already known.
            topic_code: Topic code if already known.
            grade_level: Grade level if already known.

        Returns:
            Chat response with assistant message and generated content.
        """
        session_id = request.session_id

        if session_id:
            # Continue existing session
            result = await self._continue_session(
                session_id=session_id,
                message=request.message,
            )
        else:
            # Start new session
            result = await self._start_session(
                user_id=user_id,
                tenant_id=tenant_id,
                tenant_code=tenant_code,
                message=request.message,
                user_role=user_role,
                language=language or request.language,
                country_code=country_code,
                framework_code=framework_code,
                subject_code=subject_code,
                topic_code=topic_code,
                grade_level=grade_level,
                context=request.context,
            )

        # Build response
        return self._build_chat_response(result)

    async def _start_session(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        message: str,
        user_role: str | None,
        language: str | None,
        country_code: str | None,
        framework_code: str | None,
        subject_code: str | None,
        topic_code: str | None,
        grade_level: int | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Start a new content creation session.

        All curriculum parameters come from external sources (user profile,
        tenant settings, request context) - never hardcoded.

        Args:
            user_id: User making the request.
            tenant_id: Tenant identifier.
            tenant_code: Tenant code.
            message: Initial message.
            user_role: User's role (teacher, student, parent).
            language: Content language code.
            country_code: User's country code.
            framework_code: Curriculum framework from tenant/user settings.
            subject_code: Subject code if already known.
            topic_code: Topic code if already known.
            grade_level: Grade level if already known.
            context: Additional context.

        Returns:
            Workflow result with session state.
        """
        session_id = str(uuid4())

        logger.info(
            "Starting content creation session: session=%s, user=%s, tenant=%s",
            session_id,
            user_id,
            tenant_code,
        )

        # Extract additional context values if provided
        ctx_user_role = context.get("user_role") if context else None
        ctx_language = context.get("language") if context else None
        ctx_country = context.get("country_code") if context else None
        ctx_framework = context.get("framework_code") if context else None
        ctx_subject = context.get("subject_code") if context else None
        ctx_topic = context.get("topic_code") if context else None
        ctx_grade = context.get("grade_level") if context else None

        # Create initial state - all curriculum values from external sources
        initial_state = create_initial_content_state(
            session_id=session_id,
            tenant_code=tenant_code,
            user_id=str(user_id),
            user_role=user_role or ctx_user_role,
            language=language or ctx_language,
            country_code=country_code or ctx_country,
            framework_code=framework_code or ctx_framework,
            subject_code=subject_code or ctx_subject,
            topic_code=topic_code or ctx_topic,
            grade_level=grade_level or ctx_grade,
        )

        # If user provided initial message, add it
        if message:
            initial_state["messages"] = [{"role": "user", "content": message}]

        # Run workflow
        result = await self._workflow.run(
            initial_state=initial_state,
            thread_id=session_id,
        )

        # If user provided message, continue to process it
        if message:
            result = await self._workflow.send_message(
                thread_id=session_id,
                message=message,
            )

        return result

    async def _continue_session(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Continue an existing content creation session.

        Args:
            session_id: Session to continue.
            message: User message.

        Returns:
            Workflow result with updated state.

        Raises:
            ContentSessionNotFoundError: If session not found.
        """
        logger.debug(
            "Continuing content creation session: session=%s",
            session_id,
        )

        try:
            result = await self._workflow.send_message(
                thread_id=session_id,
                message=message,
            )
            return result
        except ValueError as e:
            if "No active session" in str(e):
                raise ContentSessionNotFoundError(session_id) from e
            raise

    def _build_chat_response(self, result: dict[str, Any]) -> ContentChatResponse:
        """Build chat response from workflow result.

        Args:
            result: Workflow result.

        Returns:
            ContentChatResponse with formatted data.
        """
        state = result.get("state", {})

        # Get generated content
        generated_content = None
        current_content = result.get("generated_content") or state.get("current_content")
        if current_content:
            # Parse exported_at if present
            exported_at = None
            if current_content.get("exported_at"):
                exported_at = datetime.fromisoformat(current_content["exported_at"])

            generated_content = GeneratedContentResponse(
                id=current_content.get("id", ""),
                content_type=current_content.get("content_type", ""),
                title=current_content.get("title", ""),
                status=current_content.get("status", "draft"),
                preview_url=current_content.get("preview_url"),
                h5p_id=current_content.get("h5p_id"),
                ai_content=current_content.get("ai_content", {}),
                quality_score=current_content.get("quality_score"),
                created_at=datetime.fromisoformat(
                    current_content.get("created_at", datetime.utcnow().isoformat())
                ),
                exported_at=exported_at,
            )

        # Get recommended types from content_types in state
        recommended_types = []
        selected_types = state.get("content_types", [])
        for ct in selected_types:
            converter = self.converter_registry.get(ct)
            if converter:
                info = converter.get_content_info()
                recommended_types.append(
                    ContentTypeInfo(
                        content_type=info["content_type"],
                        library=info["library"],
                        name=info.get("name", info["content_type"]),
                        description=info.get("description", ""),
                        category=info.get("category", ""),
                        ai_support=info.get("ai_support", "partial"),
                        bloom_levels=info.get("bloom_levels", []),
                        requires_media=info.get("requires_media", False),
                    )
                )

        # Map current_phase to workflow_phase for API compatibility
        current_phase = state.get("current_phase", "initialization")
        phase_map = {
            "initialization": "gathering_requirements",
            "role_detection": "gathering_requirements",
            "requirements": "gathering_requirements",
            "selection": "awaiting_confirmation",
            "generation": "generating",
            "review": "reviewing",
            "export": "exporting",
            "complete": "completed",
            "error_recovery": "gathering_requirements",
        }
        workflow_phase = phase_map.get(current_phase, "gathering_requirements")

        return ContentChatResponse(
            session_id=result.get("session_id", ""),
            message=result.get("message", ""),
            workflow_phase=workflow_phase,
            current_agent=state.get("active_agent"),
            generated_content=generated_content,
            recommended_types=recommended_types,
            suggestions=[],  # UI suggestions from tool results
            metadata={
                "thread_id": result.get("thread_id"),
                "current_phase": current_phase,
                "requires_input": state.get("requires_input", False),
                "subject_code": state.get("subject_code"),
                "topic_code": state.get("topic_code"),
                "grade_level": state.get("grade_level"),
                "framework_code": state.get("framework_code"),
                "country_code": state.get("country_code"),
                "language": state.get("language"),
                "generated_count": len(state.get("generated_contents", [])),
                "exported_count": len(state.get("exported_content_ids", [])),
            },
        )

    async def get_session(
        self,
        session_id: str,
        user_id: UUID,
    ) -> ContentSessionResponse:
        """Get session information.

        Args:
            session_id: Session identifier.
            user_id: User making the request.

        Returns:
            Session information.

        Raises:
            ContentSessionNotFoundError: If session not found.
        """
        # Get state from checkpointer
        if not self.checkpointer:
            raise ContentSessionNotFoundError(session_id)

        config = {"configurable": {"thread_id": session_id}}
        state = await self._workflow._graph.aget_state(config)

        if not state or not state.values:
            raise ContentSessionNotFoundError(session_id)

        values = state.values

        # Count messages
        conversation_history = values.get("conversation_history", [])
        message_count = len(conversation_history)

        # Get last message timestamp
        last_message_at = None
        if conversation_history:
            last_turn = conversation_history[-1]
            if isinstance(last_turn, dict) and last_turn.get("timestamp"):
                last_message_at = datetime.fromisoformat(last_turn["timestamp"])

        # Map current_phase to workflow_phase for API compatibility
        current_phase = values.get("current_phase", "initialization")
        phase_map = {
            "initialization": "gathering_requirements",
            "role_detection": "gathering_requirements",
            "requirements": "gathering_requirements",
            "selection": "awaiting_confirmation",
            "generation": "generating",
            "review": "reviewing",
            "export": "exporting",
            "complete": "completed",
            "error_recovery": "gathering_requirements",
        }
        workflow_phase = phase_map.get(current_phase, "gathering_requirements")

        return ContentSessionResponse(
            session_id=session_id,
            thread_id=session_id,
            status="active" if not values.get("should_end") else "completed",
            workflow_phase=workflow_phase,
            message_count=message_count,
            generated_count=len(values.get("generated_contents", [])),
            exported_count=len(values.get("exported_content_ids", [])),
            language=values.get("language") or "en",
            started_at=datetime.fromisoformat(
                values.get("created_at", datetime.utcnow().isoformat())
            ),
            last_message_at=last_message_at,
        )

    async def get_messages(
        self,
        session_id: str,
        user_id: UUID,
        limit: int = 50,
        before_id: UUID | None = None,
    ) -> ContentMessagesResponse:
        """Get message history for a session.

        Args:
            session_id: Session identifier.
            user_id: User making the request.
            limit: Maximum messages to return.
            before_id: Get messages before this ID.

        Returns:
            Message history.

        Raises:
            ContentSessionNotFoundError: If session not found.
        """
        if not self.checkpointer:
            raise ContentSessionNotFoundError(session_id)

        config = {"configurable": {"thread_id": session_id}}
        state = await self._workflow._graph.aget_state(config)

        if not state or not state.values:
            raise ContentSessionNotFoundError(session_id)

        values = state.values
        conversation_history = values.get("conversation_history", [])

        messages = []
        for i, turn in enumerate(conversation_history):
            if isinstance(turn, dict):
                messages.append(
                    ContentMessageResponse(
                        id=f"{session_id}-{i}",
                        role=turn.get("role", "assistant"),
                        content=turn.get("content", ""),
                        agent_id=turn.get("agent_id"),
                        created_at=datetime.fromisoformat(
                            turn.get("timestamp", datetime.utcnow().isoformat())
                        ),
                    )
                )

        # Apply limit
        has_more = len(messages) > limit
        messages = messages[-limit:]

        return ContentMessagesResponse(
            messages=messages,
            has_more=has_more,
        )

    async def get_content_types(
        self,
        category: str | None = None,
        ai_support: str | None = None,
    ) -> ContentTypesResponse:
        """Get available content types.

        Args:
            category: Filter by category.
            ai_support: Filter by AI support level.

        Returns:
            List of content types.
        """
        all_info = self.converter_registry.get_all_info()

        content_types = []
        categories = set()

        for info in all_info:
            # Apply filters
            if category and info.get("category") != category:
                continue
            if ai_support and info.get("ai_support") != ai_support:
                continue

            categories.add(info.get("category", ""))

            content_types.append(
                ContentTypeInfo(
                    content_type=info["content_type"],
                    library=info["library"],
                    name=info.get("name", info["content_type"]),
                    description=info.get("description", ""),
                    category=info.get("category", ""),
                    ai_support=info.get("ai_support", "partial"),
                    bloom_levels=info.get("bloom_levels", []),
                    requires_media=info.get("requires_media", False),
                )
            )

        return ContentTypesResponse(
            content_types=content_types,
            categories=list(categories),
            total=len(content_types),
        )

    async def end_session(
        self,
        session_id: str,
        user_id: UUID,
    ) -> dict[str, Any]:
        """End a content creation session.

        Args:
            session_id: Session to end.
            user_id: User making the request.

        Returns:
            Summary of the session.

        Raises:
            ContentSessionNotFoundError: If session not found.
        """
        if not self.checkpointer:
            raise ContentSessionNotFoundError(session_id)

        config = {"configurable": {"thread_id": session_id}}
        state = await self._workflow._graph.aget_state(config)

        if not state or not state.values:
            raise ContentSessionNotFoundError(session_id)

        values = state.values

        logger.info(
            "Ending content creation session: session=%s, generated=%d, exported=%d",
            session_id,
            len(values.get("generated_contents", [])),
            len(values.get("exported_content_ids", [])),
        )

        return {
            "session_id": session_id,
            "generated_count": len(values.get("generated_contents", [])),
            "exported_count": len(values.get("exported_content_ids", [])),
            "ended_at": datetime.utcnow().isoformat(),
        }
