# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher companion service using LangGraph workflow.

This service manages teacher assistant conversations by orchestrating
the TeacherCompanionWorkflow. It handles session lifecycle, message
exchanges, and conversation persistence.

Architecture:
    - Uses TeacherCompanionWorkflow with interrupt_before=["wait_for_message"]
    - On start: Workflow runs to generate greeting, then pauses
    - On message: Uses aupdate_state + ainvoke(None) to resume
    - All messages persisted to conversation_messages table
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.intelligence.llm import LLMClient
from src.core.orchestration.states.teacher_companion import (
    AlertSummary,
    ClassSummary,
    create_initial_teacher_companion_state,
)
from src.core.orchestration.workflows.teacher_companion import TeacherCompanionWorkflow
from src.core.tools import ToolRegistry, UIElement
from src.domains.teacher.schemas import (
    TeacherChatRequest,
    TeacherChatResponse,
    TeacherMessageResponse,
    TeacherMessagesResponse,
    TeacherSessionResponse,
)
from src.infrastructure.database.models.tenant.conversation import (
    Conversation,
    ConversationMessage,
)
from src.tools import get_default_tool_registry

if TYPE_CHECKING:
    from src.core.personas.manager import PersonaManager

logger = logging.getLogger(__name__)


class TeacherServiceError(Exception):
    """Base exception for teacher service errors."""

    pass


class TeacherSessionNotFoundError(TeacherServiceError):
    """Raised when a session is not found."""

    pass


class TeacherSessionNotActiveError(TeacherServiceError):
    """Raised when session is not in active state."""

    pass


class TeacherCompanionService:
    """Service for managing teacher assistant conversations.

    This service orchestrates teacher conversations using TeacherCompanionWorkflow.
    It manages session state, message persistence, and AI interactions.

    Unlike the student companion:
    - No 4-layer memory integration
    - No emotional context
    - No persona selection
    - Different tool set focused on class/student monitoring

    Attributes:
        _db: Async database session.
        _workflow: The teacher companion workflow instance.
    """

    def __init__(
        self,
        db: AsyncSession,
        llm_client: LLMClient,
        persona_manager: "PersonaManager",
        checkpointer: BaseCheckpointSaver | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """Initialize the teacher companion service.

        Args:
            db: Async database session.
            llm_client: LLM client for completions.
            persona_manager: Manager for assistant personas.
            checkpointer: Checkpointer for workflow state persistence.
            tool_registry: Registry of teacher tools.
        """
        self._db = db

        # Initialize tool registry
        if tool_registry is None:
            tool_registry = get_default_tool_registry()
        self._tool_registry = tool_registry

        # Initialize workflow
        self._workflow = TeacherCompanionWorkflow(
            llm_client=llm_client,
            persona_manager=persona_manager,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
        )

    async def chat(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: TeacherChatRequest,
        language: str = "en",
    ) -> TeacherChatResponse:
        """Unified chat endpoint for teacher interactions.

        Handles both new sessions and continuing conversations.
        For new sessions, generates a professional greeting.
        For continuing, processes the teacher message and generates response.

        Args:
            user_id: Teacher's user ID.
            tenant_id: Tenant UUID.
            tenant_code: Tenant code for database operations.
            request: Chat request with session_id, message, trigger.
            language: Teacher's language preference.

        Returns:
            TeacherChatResponse with message and data.
        """
        logger.debug(
            "teacher chat: session_id=%s, message=%s",
            request.session_id,
            request.message[:50] if request.message else None,
        )

        if request.session_id and request.message:
            # Continuing conversation
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
                language=language,
            )

    async def _start_session(
        self,
        user_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        trigger: str,
        context: dict[str, Any] | None,
        language: str,
    ) -> TeacherChatResponse:
        """Start a new teacher assistant session with greeting.

        Creates database records and runs workflow to generate greeting.

        Args:
            user_id: Teacher's user ID.
            tenant_id: Tenant UUID.
            tenant_code: Tenant code.
            trigger: What triggered this session.
            context: Additional context.
            language: Teacher's language preference.

        Returns:
            TeacherChatResponse with greeting.
        """
        session_id = uuid4()
        conversation_id = uuid4()
        thread_id = f"teacher_{session_id}"

        logger.info(
            "Starting teacher session: session=%s, user=%s, trigger=%s",
            session_id,
            user_id,
            trigger,
        )

        now = datetime.utcnow()

        # Create conversation record
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            conversation_type="teacher_companion",
            persona_id="teacher_assistant",
            title="Teacher Assistant Conversation",
            status="active",
            message_count=0,
        )
        self._db.add(conversation)
        await self._db.flush()

        # Load teacher context BEFORE workflow execution for transaction isolation
        # This prevents raw SQL queries in workflow from corrupting the transaction
        class_summary = await self._load_class_summary(user_id)
        alert_summary = await self._load_alert_summary(user_id)

        logger.debug(
            "Pre-loaded teacher context: classes=%d, alerts=%s",
            len(class_summary),
            alert_summary["total_count"] if alert_summary else 0,
        )

        # Create initial workflow state with pre-loaded context
        initial_state = create_initial_teacher_companion_state(
            session_id=str(session_id),
            conversation_id=str(conversation_id),
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            teacher_id=str(user_id),
            language=language,
            class_summary=class_summary,
            alert_summary=alert_summary,
        )

        # Set database session for tool execution
        self._workflow.set_db_session(self._db)

        # Run workflow - generates greeting and pauses at wait_for_message
        logger.debug("Running teacher workflow with thread_id=%s", thread_id)
        state = await self._workflow.run(initial_state, thread_id)

        # Get greeting from workflow state
        greeting = state.get("first_greeting", "Hello! How can I help you today?")

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
            "Teacher session started: session=%s, greeting_length=%d",
            session_id,
            len(greeting),
        )

        # Build response with any suggestions from greeting
        suggestions = self._extract_ui_elements(state)
        tool_data = self._extract_tool_data(state)

        return TeacherChatResponse(
            session_id=str(session_id),
            message=greeting,
            suggestions=suggestions,
            tool_data=tool_data,
            metadata={
                "trigger": trigger,
                "conversation_id": str(conversation_id),
            },
        )

    async def _continue_conversation(
        self,
        session_id: UUID,
        user_id: UUID,
        message: str,
        context: dict[str, Any] | None,
    ) -> TeacherChatResponse:
        """Continue an existing teacher conversation.

        Processes teacher message through workflow and generates response.

        Args:
            session_id: Session ID.
            user_id: Teacher's user ID.
            message: Teacher's message.
            context: Additional context.

        Returns:
            TeacherChatResponse with assistant's response.
        """
        thread_id = f"teacher_{session_id}"

        # Get conversation by looking up from thread_id pattern
        # For teacher, we store conversation_id in the state
        # We need to get the current state to find the conversation_id
        current_state = await self._workflow.get_state(thread_id)

        if not current_state:
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        conversation_id_str = current_state.get("conversation_id")
        if not conversation_id_str:
            raise TeacherSessionNotFoundError(
                f"Session {session_id} has no conversation"
            )

        conversation_id = UUID(conversation_id_str)

        # Get conversation for updating
        conv_stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise TeacherSessionNotFoundError(
                f"Conversation {conversation_id} not found"
            )

        if conversation.status != "active":
            raise TeacherSessionNotActiveError(
                f"Session {session_id} is not active"
            )

        # Verify user owns this conversation
        if str(conversation.user_id) != str(user_id):
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        now = datetime.utcnow()

        logger.info(
            "Continuing teacher conversation: session=%s, message_length=%d",
            session_id,
            len(message),
        )

        # Save teacher message
        teacher_message = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            role="user",
            content=message,
            created_at=now,
        )
        self._db.add(teacher_message)
        conversation.message_count += 1
        conversation.last_message_at = now

        # Set database session for tool execution
        self._workflow.set_db_session(self._db)

        # Send message to workflow with proper transaction handling
        logger.debug(
            "Service calling teacher workflow.send_message for thread=%s", thread_id
        )
        try:
            state = await self._workflow.send_message(thread_id, message)
        except Exception as e:
            # Rollback transaction on workflow failure to prevent
            # "current transaction is aborted" errors on subsequent requests
            logger.error("Workflow execution failed, rolling back transaction: %s", e)
            await self._db.rollback()
            raise

        # Get response from workflow
        response_text = state.get(
            "last_assistant_response",
            "I'm here to help. What would you like to know?",
        )

        # Save assistant response
        assistant_message = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            created_at=datetime.utcnow(),
        )
        self._db.add(assistant_message)
        conversation.message_count += 1
        conversation.last_message_at = datetime.utcnow()

        await self._db.commit()

        logger.info(
            "Teacher conversation continued: session=%s, response_length=%d",
            session_id,
            len(response_text),
        )

        # Extract suggestions and tool data
        suggestions = self._extract_ui_elements(state)
        tool_data = self._extract_tool_data(state)

        return TeacherChatResponse(
            session_id=str(session_id),
            message=response_text,
            suggestions=suggestions,
            tool_data=tool_data,
            metadata={
                "tool_calls": state.get("tool_call_count", 0),
            },
        )

    async def get_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> TeacherSessionResponse:
        """Get session information.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.

        Returns:
            TeacherSessionResponse.

        Raises:
            TeacherSessionNotFoundError: If session not found.
        """
        thread_id = f"teacher_{session_id}"
        current_state = await self._workflow.get_state(thread_id)

        if not current_state:
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        conversation_id_str = current_state.get("conversation_id")
        if not conversation_id_str:
            raise TeacherSessionNotFoundError(
                f"Session {session_id} has no conversation"
            )

        # Get conversation
        conv_stmt = select(Conversation).where(
            Conversation.id == UUID(conversation_id_str)
        )
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if not conversation:
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        # Verify user owns this conversation
        if str(conversation.user_id) != str(user_id):
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        return TeacherSessionResponse(
            session_id=str(session_id),
            conversation_id=str(conversation.id),
            status=conversation.status,
            message_count=conversation.message_count,
            started_at=conversation.created_at,
            last_message_at=conversation.last_message_at,
        )

    async def get_messages(
        self,
        session_id: UUID,
        user_id: UUID,
        limit: int = 50,
        before_id: UUID | None = None,
    ) -> TeacherMessagesResponse:
        """Get message history for a session.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.
            limit: Maximum messages to return.
            before_id: Get messages before this ID.

        Returns:
            TeacherMessagesResponse with messages.

        Raises:
            TeacherSessionNotFoundError: If session not found.
        """
        thread_id = f"teacher_{session_id}"
        current_state = await self._workflow.get_state(thread_id)

        if not current_state:
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        conversation_id_str = current_state.get("conversation_id")
        if not conversation_id_str:
            raise TeacherSessionNotFoundError(
                f"Session {session_id} has no conversation"
            )

        conversation_id = UUID(conversation_id_str)

        # Verify user owns this conversation
        conv_stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if not conversation or str(conversation.user_id) != str(user_id):
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        # Get messages
        msg_stmt = select(ConversationMessage).where(
            ConversationMessage.conversation_id == conversation_id,
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

        return TeacherMessagesResponse(
            messages=[
                TeacherMessageResponse(
                    id=str(msg.id),
                    role="assistant" if msg.role == "assistant" else "teacher",
                    content=msg.content,
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
    ) -> None:
        """End a teacher assistant session.

        Args:
            session_id: Session ID.
            user_id: User ID for authorization.

        Raises:
            TeacherSessionNotFoundError: If session not found.
        """
        thread_id = f"teacher_{session_id}"
        current_state = await self._workflow.get_state(thread_id)

        if not current_state:
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        conversation_id_str = current_state.get("conversation_id")
        if not conversation_id_str:
            raise TeacherSessionNotFoundError(
                f"Session {session_id} has no conversation"
            )

        # Get and verify conversation ownership
        conv_stmt = select(Conversation).where(
            Conversation.id == UUID(conversation_id_str)
        )
        conv_result = await self._db.execute(conv_stmt)
        conversation = conv_result.scalar_one_or_none()

        if not conversation or str(conversation.user_id) != str(user_id):
            raise TeacherSessionNotFoundError(f"Session {session_id} not found")

        # End the conversation
        conversation.status = "completed"
        await self._db.commit()

        logger.info("Teacher session ended: session=%s", session_id)

    async def _load_class_summary(self, teacher_id: UUID) -> list[ClassSummary]:
        """Load class summary for teacher from database.

        This method is called before workflow execution to ensure
        proper transaction isolation. If the query fails, it returns
        an empty list without corrupting the transaction.

        Args:
            teacher_id: Teacher's user ID.

        Returns:
            List of ClassSummary for teacher's assigned classes.
        """
        class_summary: list[ClassSummary] = []

        try:
            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
                ClassTeacher,
            )
            from src.infrastructure.database.models.tenant.curriculum import Subject

            query = (
                select(
                    ClassTeacher.class_id,
                    Class.name.label("class_name"),
                    ClassTeacher.is_homeroom,
                    Subject.name.label("subject_name"),
                    func.count(ClassStudent.id).label("student_count"),
                )
                .join(Class, ClassTeacher.class_id == Class.id)
                .outerjoin(
                    Subject,
                    (ClassTeacher.subject_framework_code == Subject.framework_code)
                    & (ClassTeacher.subject_code == Subject.code),
                )
                .outerjoin(
                    ClassStudent,
                    (ClassStudent.class_id == Class.id) & (ClassStudent.status == "active"),
                )
                .where(ClassTeacher.teacher_id == str(teacher_id))
                .where(ClassTeacher.ended_at.is_(None))
                .where(Class.is_active == True)  # noqa: E712
                .group_by(
                    ClassTeacher.class_id,
                    Class.name,
                    ClassTeacher.is_homeroom,
                    Subject.name,
                )
            )

            result = await self._db.execute(query)
            rows = result.all()

            for row in rows:
                class_summary.append(
                    ClassSummary(
                        class_id=str(row.class_id),
                        class_name=row.class_name,
                        student_count=row.student_count or 0,
                        subject_name=row.subject_name,
                        is_homeroom=row.is_homeroom,
                    )
                )

            logger.debug(
                "Loaded %d classes for teacher %s",
                len(class_summary),
                teacher_id,
            )

        except Exception as e:
            logger.warning("Failed to load class summary: %s", str(e))
            # Rollback to clean the dirty transaction state
            try:
                await self._db.rollback()
                logger.debug("Rolled back after class summary load failure")
            except Exception as rollback_err:
                logger.error("Rollback failed: %s", rollback_err)

        return class_summary

    async def _load_alert_summary(self, teacher_id: UUID) -> AlertSummary | None:
        """Load alert summary for teacher's students from database.

        This method is called before workflow execution to ensure
        proper transaction isolation. If the query fails, it returns
        None without corrupting the transaction.

        Args:
            teacher_id: Teacher's user ID.

        Returns:
            AlertSummary or None if no alerts or query fails.
        """
        alert_summary: AlertSummary | None = None

        try:
            from src.infrastructure.database.models.tenant.school import (
                Class,
                ClassStudent,
                ClassTeacher,
            )
            from src.infrastructure.database.models.tenant.notification import Alert

            # Get all student IDs from teacher's classes
            student_ids_query = (
                select(ClassStudent.student_id)
                .join(Class, ClassStudent.class_id == Class.id)
                .join(ClassTeacher, ClassTeacher.class_id == Class.id)
                .where(ClassTeacher.teacher_id == str(teacher_id))
                .where(ClassTeacher.ended_at.is_(None))
                .where(ClassStudent.status == "active")
                .distinct()
            )

            student_ids_result = await self._db.execute(student_ids_query)
            student_ids = [str(row[0]) for row in student_ids_result.all()]

            if student_ids:
                # Count alerts by severity
                alert_counts_query = (
                    select(
                        Alert.severity,
                        func.count(Alert.id).label("count"),
                    )
                    .where(Alert.student_id.in_(student_ids))
                    .where(Alert.status == "active")
                    .group_by(Alert.severity)
                )

                alert_counts_result = await self._db.execute(alert_counts_query)
                alert_counts = {row.severity: row.count for row in alert_counts_result.all()}

                # Get recent alerts
                recent_alerts_query = (
                    select(Alert)
                    .where(Alert.student_id.in_(student_ids))
                    .where(Alert.status == "active")
                    .order_by(Alert.created_at.desc())
                    .limit(5)
                )

                recent_alerts_result = await self._db.execute(recent_alerts_query)
                recent_alerts = [
                    {
                        "id": str(alert.id),
                        "student_id": str(alert.student_id),
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "title": alert.title,
                        "created_at": alert.created_at.isoformat() if alert.created_at else None,
                    }
                    for alert in recent_alerts_result.scalars().all()
                ]

                alert_summary = AlertSummary(
                    total_count=sum(alert_counts.values()),
                    critical_count=alert_counts.get("critical", 0),
                    warning_count=alert_counts.get("warning", 0),
                    info_count=alert_counts.get("info", 0),
                    recent_alerts=recent_alerts,
                )

                logger.debug(
                    "Loaded alert summary: total=%d, critical=%d",
                    alert_summary["total_count"],
                    alert_summary["critical_count"],
                )

        except Exception as e:
            logger.warning("Failed to load alert summary: %s", str(e))
            # Rollback to clean the dirty transaction state
            try:
                await self._db.rollback()
                logger.debug("Rolled back after alert summary load failure")
            except Exception as rollback_err:
                logger.error("Rollback failed: %s", rollback_err)

        return alert_summary

    def _extract_ui_elements(self, state: dict) -> list[UIElement]:
        """Extract UI elements from workflow state.

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
                ui_elements.append(UIElement.model_validate(element_data))
            except Exception as e:
                logger.warning("Failed to parse UI element: %s", e)

        return ui_elements

    def _extract_tool_data(self, state: dict) -> dict[str, Any]:
        """Extract tool data from workflow state.

        Args:
            state: Workflow state.

        Returns:
            Dictionary of tool data for frontend.
        """
        return state.get("tool_data", {})
