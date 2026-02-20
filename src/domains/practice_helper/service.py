# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice Helper Tutor service.

This service manages practice helper tutoring sessions:
- Start a tutoring session when student clicks "Get Help"
- Send/receive messages during the tutoring conversation
- Complete the session and return to practice

The service integrates with:
- Practice session data (question context)
- Memory system (personalization)
- Subject-specific tutor agents
- LangGraph workflow (interrupt/resume)
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.orchestration.states.practice_helper import (
    create_initial_practice_helper_state,
    select_agent_id,
    select_tutoring_mode,
    TutoringMode,
)
from src.core.orchestration.workflows import PracticeHelperWorkflow
from src.infrastructure.database.models.tenant import (
    PracticeHelperMessage,
    PracticeHelperSession,
    PracticeQuestion,
    PracticeSession,
    User,
)
from src.models.practice_helper import (
    CompleteSessionRequest,
    get_suggested_actions,
    HandoffInfo,
    MessageAction,
    MessageInfo,
    MessageResponse,
    PracticeHelperSessionResponse,
    PracticeHelperSessionStatus,
    ReturnToPracticeInfo,
    SessionCompletionResponse,
    SessionHistoryResponse,
    SendMessageRequest,
    StartPracticeHelperRequest,
    StartPracticeHelperResponse,
    SuggestedAction,
    TutoringMode as TutoringModeDTO,
)

logger = logging.getLogger(__name__)


class PracticeHelperService:
    """Service for practice helper tutoring sessions.

    Manages the lifecycle of practice helper sessions:
    1. Start session - creates workflow and generates first message
    2. Send message - processes student message and returns tutor response
    3. Get session - retrieves session status
    4. Complete session - ends session and returns to practice

    Attributes:
        workflow: PracticeHelperWorkflow instance.

    Example:
        >>> service = PracticeHelperService(workflow)
        >>> response = await service.start_session(db, student_id, tenant_id, tenant_code, request)
        >>> message_response = await service.send_message(db, session_id, student_id, request)
    """

    def __init__(self, workflow: PracticeHelperWorkflow):
        """Initialize the service.

        Args:
            workflow: PracticeHelperWorkflow instance.
        """
        self._workflow = workflow

    async def start_session(
        self,
        db: AsyncSession,
        student_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: StartPracticeHelperRequest,
    ) -> StartPracticeHelperResponse:
        """Start a practice helper tutoring session.

        Called when student clicks "Get Help" after answering incorrectly.
        Retrieves question context from practice session, creates the
        tutoring session, and generates the first tutor message.

        Args:
            db: Database session.
            student_id: Student's ID.
            tenant_id: Tenant's ID.
            tenant_code: Tenant code for memory operations.
            request: Request with practice session and question IDs.

        Returns:
            StartPracticeHelperResponse with session ID and first message.

        Raises:
            ValueError: If practice session or question not found.
        """
        logger.info(
            "Starting practice helper session: student=%s, practice_session=%s, question=%s",
            student_id,
            request.practice_session_id,
            request.question_id,
        )

        # Get practice session
        practice_session = await db.get(PracticeSession, str(request.practice_session_id))
        if not practice_session:
            raise ValueError(f"Practice session not found: {request.practice_session_id}")

        # Verify student owns the session
        if practice_session.student_id != str(student_id):
            raise ValueError("Practice session does not belong to this student")

        # Get the question
        question = await db.get(PracticeQuestion, str(request.question_id))
        if not question:
            raise ValueError(f"Question not found: {request.question_id}")

        # Verify question belongs to session
        if question.session_id != str(request.practice_session_id):
            raise ValueError("Question does not belong to this practice session")

        # Get student info for personalization
        student = await db.get(User, str(student_id))
        student_info = self._extract_student_info(student)

        # Get topic context from practice session
        topic_context = await self._get_topic_context(db, practice_session)

        # Get emotional state (from student's recent signals or default)
        emotional_state = await self._get_emotional_state(db, student_id, student_info)

        # Get topic mastery using topic full code from practice session
        topic_mastery = await self._get_topic_mastery(db, student_id, practice_session.topic_full_code or "")

        # Determine tutoring mode
        mode = select_tutoring_mode(emotional_state, topic_mastery)

        # Determine agent
        subject_code = topic_context.get("subject_code", "other")
        agent_id = select_agent_id(subject_code)

        # Create workflow session ID
        import uuid
        session_id = str(uuid.uuid4())

        # Create initial state
        initial_state = create_initial_practice_helper_state(
            session_id=session_id,
            practice_session_id=str(request.practice_session_id),
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            # Question context
            question_id=str(request.question_id),
            question_text=question.content,
            question_type=question.display_hint or "multiple_choice",
            correct_answer=question.correct_answer,
            student_answer=request.student_answer,
            options=question.data.get("options") if question.data else None,
            explanation=question.explanation,
            # Student context
            student_id=str(student_id),
            student_age=student_info.get("age"),
            student_gender=student_info.get("gender"),
            grade_level=topic_context.get("grade_level"),
            grade_level_code=topic_context.get("grade_level_code"),
            language=student_info.get("language", "en"),
            topic_mastery=topic_mastery,
            emotional_state=emotional_state,
            interests=student_info.get("interests", []),
            # Subject context
            subject=topic_context.get("subject", "other"),
            subject_code=subject_code,
            topic_name=topic_context.get("topic_name", ""),
        )

        # Run workflow to get first message
        result = await self._workflow.run(initial_state, thread_id=session_id)

        # Create database record
        db_session = PracticeHelperSession(
            id=session_id,
            student_id=str(student_id),
            practice_session_id=str(request.practice_session_id),
            practice_question_id=str(request.question_id),
            agent_id=agent_id,
            initial_mode=mode.value,
            status="active",
            subject=topic_context.get("subject", "other"),
            topic_name=topic_context.get("topic_name", ""),
            question_type=question.display_hint or "multiple_choice",
            turn_count=1,
            current_step=result.get("current_step", 0),
        )
        db.add(db_session)

        # Save first tutor message
        first_message = result.get("last_tutor_response", "")
        if first_message:
            db_message = PracticeHelperMessage(
                session_id=session_id,
                sequence=1,
                role="tutor",
                content=first_message,
                mode_at_time=mode.value,
                step_at_time=result.get("current_step"),
            )
            db.add(db_message)

        await db.commit()

        # Get suggested actions
        suggested_actions = get_suggested_actions(
            PracticeHelperSessionStatus.ACTIVE,
            TutoringModeDTO(mode.value),
        )

        logger.info(
            "Practice helper session started: session=%s, mode=%s, agent=%s",
            session_id,
            mode.value,
            agent_id,
        )

        return StartPracticeHelperResponse(
            session_id=UUID(session_id),
            mode=TutoringModeDTO(mode.value),
            message=first_message,
            suggested_actions=suggested_actions,
        )

    async def send_message(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: SendMessageRequest,
    ) -> MessageResponse:
        """Send a message in a practice helper session.

        Processes the student's message/action and returns the tutor's response.
        Handles mode escalation and step progression.

        Args:
            db: Database session.
            session_id: Practice helper session ID.
            student_id: Student's ID.
            request: Request with message and action.

        Returns:
            MessageResponse with tutor's response.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        logger.info(
            "Sending message: session=%s, action=%s, message=%s...",
            session_id,
            request.action.value,
            request.message[:50] if request.message else None,
        )

        # Get session
        session = await db.get(PracticeHelperSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Verify student owns session
        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Verify session is active
        if session.status != "active":
            raise ValueError(f"Session is not active: {session.status}")

        # Record student message if any
        if request.message:
            session.turn_count += 1
            db_message = PracticeHelperMessage(
                session_id=str(session_id),
                sequence=session.turn_count,
                role="student",
                content=request.message,
                action=request.action.value,
                mode_at_time=session.final_mode or session.initial_mode,
                step_at_time=session.current_step,
            )
            db.add(db_message)

        # Send message through workflow
        result = await self._workflow.send_message(
            thread_id=str(session_id),
            message=request.message,
            action=request.action.value,
        )

        # Update session from workflow result
        new_mode = result.get("tutoring_mode", session.final_mode or session.initial_mode)
        if new_mode != (session.final_mode or session.initial_mode):
            session.mode_escalations += 1
        session.final_mode = new_mode
        session.current_step = result.get("current_step", session.current_step)
        session.total_steps = result.get("total_steps", session.total_steps)
        session.understanding_progress = Decimal(str(result.get("understanding_progress", 0.0)))

        # Check if workflow ended
        handoff_info = None
        if result.get("status") == "completed":
            session.status = "completed"
            session.ended_at = datetime.now(timezone.utc)
            session.understood = result.get("understood")
        elif result.get("status") == "escalating":
            # Practice Helper is escalating to Learning Tutor
            session.status = "escalated"
            session.ended_at = datetime.now(timezone.utc)
            session.understood = False
            session.completion_reason = result.get("completion_reason", "escalated_to_learning_tutor")

            # Build handoff info from workflow result
            ui_action = result.get("ui_action")
            if ui_action:
                handoff_info = HandoffInfo(
                    type=ui_action.get("type", "handoff"),
                    target_workflow=ui_action.get("target_workflow", "learning_tutor"),
                    label=ui_action.get("label", "Learn this topic"),
                    params=ui_action.get("params", {}),
                    route=ui_action.get("route", "/learn"),
                )

            logger.info(
                "Practice helper escalating to Learning Tutor: session=%s",
                session_id,
            )

        # Record tutor response
        tutor_response = result.get("last_tutor_response", "")
        if tutor_response:
            session.turn_count += 1
            db_message = PracticeHelperMessage(
                session_id=str(session_id),
                sequence=session.turn_count,
                role="tutor",
                content=tutor_response,
                mode_at_time=new_mode,
                step_at_time=session.current_step,
            )
            db.add(db_message)

        await db.commit()

        # Get suggested actions
        status = PracticeHelperSessionStatus(session.status)
        mode = TutoringModeDTO(new_mode)
        suggested_actions = get_suggested_actions(status, mode)

        return MessageResponse(
            message=tutor_response,
            mode=mode,
            turn_number=session.turn_count,
            current_step=session.current_step if new_mode == "step_by_step" else None,
            total_steps=session.total_steps if new_mode == "step_by_step" else None,
            status=status,
            suggested_actions=suggested_actions,
            handoff=handoff_info,
        )

    async def get_session(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> PracticeHelperSessionResponse:
        """Get practice helper session details.

        Args:
            db: Database session.
            session_id: Practice helper session ID.
            student_id: Student's ID.

        Returns:
            PracticeHelperSessionResponse with session details.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        session = await db.get(PracticeHelperSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        return PracticeHelperSessionResponse(
            id=UUID(session.id),
            practice_session_id=UUID(session.practice_session_id),
            practice_question_id=UUID(session.practice_question_id),
            status=PracticeHelperSessionStatus(session.status),
            mode=TutoringModeDTO(session.final_mode or session.initial_mode),
            turn_count=session.turn_count,
            current_step=session.current_step if session.final_mode == "step_by_step" else None,
            total_steps=session.total_steps if session.final_mode == "step_by_step" else None,
            understanding_progress=float(session.understanding_progress),
            started_at=session.started_at,
            last_activity_at=session.updated_at or session.started_at,
            ended_at=session.ended_at,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def get_session_history(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> SessionHistoryResponse:
        """Get conversation history for a session.

        Args:
            db: Database session.
            session_id: Practice helper session ID.
            student_id: Student's ID.

        Returns:
            SessionHistoryResponse with message list.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        session = await db.get(PracticeHelperSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Get messages ordered by sequence
        query = (
            select(PracticeHelperMessage)
            .where(PracticeHelperMessage.session_id == str(session_id))
            .order_by(PracticeHelperMessage.sequence)
        )
        result = await db.execute(query)
        messages = result.scalars().all()

        return SessionHistoryResponse(
            session_id=UUID(session.id),
            messages=[
                MessageInfo(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at,
                    action=msg.action,
                )
                for msg in messages
            ],
        )

    async def complete_session(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: CompleteSessionRequest,
    ) -> SessionCompletionResponse:
        """Complete a practice helper session.

        Marks the session as completed and returns information for
        returning to the practice session.

        Args:
            db: Database session.
            session_id: Practice helper session ID.
            student_id: Student's ID.
            request: Request with completion details.

        Returns:
            SessionCompletionResponse with completion details.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        logger.info(
            "Completing session: session=%s, understood=%s, wants_retry=%s",
            session_id,
            request.understood,
            request.wants_retry,
        )

        session = await db.get(PracticeHelperSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Update session
        session.status = "completed"
        session.ended_at = datetime.now(timezone.utc)
        session.understood = request.understood
        session.wants_retry = request.wants_retry

        # Calculate time spent
        if session.started_at and session.ended_at:
            session.time_spent_seconds = int((session.ended_at - session.started_at).total_seconds())

        # Determine completion reason
        if request.understood:
            session.completion_reason = "understood"
        else:
            session.completion_reason = "user_ended"

        await db.commit()

        return SessionCompletionResponse(
            session_id=UUID(session.id),
            status=PracticeHelperSessionStatus.COMPLETED,
            total_turns=session.turn_count,
            mode_used=TutoringModeDTO(session.final_mode or session.initial_mode),
            mode_escalations=session.mode_escalations,
            understood=request.understood,
            return_to_practice=ReturnToPracticeInfo(
                practice_session_id=UUID(session.practice_session_id),
                question_id=UUID(session.practice_question_id),
                can_retry=request.wants_retry,
            ),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_student_info(self, student: User | None) -> dict[str, Any]:
        """Extract student info for personalization.

        Args:
            student: User model or None.

        Returns:
            Dict with student info.
        """
        if not student:
            return {"language": "en", "interests": []}

        extra_data = student.extra_data or {}

        return {
            "age": extra_data.get("age"),
            "gender": extra_data.get("gender"),
            "language": student.preferred_language or "en",
            "interests": extra_data.get("interests", []),
        }

    async def _get_topic_context(
        self,
        db: AsyncSession,
        practice_session: PracticeSession,
    ) -> dict[str, str]:
        """Get topic context from practice session.

        Args:
            db: Database session.
            practice_session: Practice session model.

        Returns:
            Dict with topic context.
        """
        # Initialize with defaults
        context = {
            "topic_name": "",
            "subject": "other",
            "subject_code": "other",
            "grade_level": "",
            "grade_level_code": "",
        }

        # PracticeSession has composite topic key fields
        if not practice_session.topic_full_code:
            return context

        try:
            from src.infrastructure.database.models.tenant.curriculum import Topic, Subject, GradeLevel

            # Get topic using composite key
            topic_query = select(Topic).where(
                Topic.framework_code == practice_session.topic_framework_code,
                Topic.subject_code == practice_session.topic_subject_code,
                Topic.grade_code == practice_session.topic_grade_code,
                Topic.unit_code == practice_session.topic_unit_code,
                Topic.code == practice_session.topic_code,
            )
            result = await db.execute(topic_query)
            topic = result.scalar_one_or_none()

            if not topic:
                return context

            context["topic_name"] = topic.name or ""

            # Get subject using composite key from topic
            subject_query = select(Subject).where(
                Subject.framework_code == topic.framework_code,
                Subject.code == topic.subject_code,
            )
            subject_result = await db.execute(subject_query)
            subject = subject_result.scalar_one_or_none()

            if subject:
                context["subject"] = subject.name or "other"
                context["subject_code"] = subject.code or "other"

            # Get grade level using composite key from topic
            # GradeLevel has (framework_code, stage_code, code) composite key
            # We look up by framework_code and code, taking first match
            grade_query = select(GradeLevel).where(
                GradeLevel.framework_code == topic.framework_code,
                GradeLevel.code == topic.grade_code,
            )
            grade_result = await db.execute(grade_query)
            grade = grade_result.scalars().first()

            if grade:
                context["grade_level"] = grade.name or ""
                context["grade_level_code"] = grade.code or ""

        except Exception as e:
            logger.warning("Failed to get topic context: %s", str(e))

        return context

    async def _get_emotional_state(
        self,
        db: AsyncSession,
        student_id: UUID,
        student_info: dict,
    ) -> str:
        """Get student's current emotional state.

        Args:
            db: Database session.
            student_id: Student's ID.
            student_info: Extracted student info.

        Returns:
            Emotional state string.
        """
        # Try to get from recent emotional signals
        try:
            from src.infrastructure.database.models.tenant import EmotionalSignal
            from sqlalchemy import desc

            query = (
                select(EmotionalSignal)
                .where(EmotionalSignal.student_id == str(student_id))
                .order_by(desc(EmotionalSignal.created_at))
                .limit(1)
            )
            result = await db.execute(query)
            signal = result.scalar_one_or_none()

            if signal:
                return signal.emotional_state

        except Exception as e:
            logger.debug("Could not get emotional state: %s", str(e))

        return "neutral"

    async def _get_topic_mastery(
        self,
        db: AsyncSession,
        student_id: UUID,
        topic_full_code: str,
    ) -> float:
        """Get student's mastery on the topic.

        Args:
            db: Database session.
            student_id: Student's ID.
            topic_full_code: Topic full code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            Mastery level (0.0-1.0).
        """
        if not topic_full_code:
            return 0.5

        # Try to get from semantic memory
        try:
            from src.infrastructure.database.models.tenant.memory import SemanticMemory

            query = (
                select(SemanticMemory)
                .where(
                    SemanticMemory.student_id == str(student_id),
                    SemanticMemory.entity_type == "topic",
                    SemanticMemory.entity_full_code == topic_full_code,
                )
            )
            result = await db.execute(query)
            memory = result.scalar_one_or_none()

            if memory:
                return float(memory.mastery_level)

        except Exception as e:
            logger.debug("Could not get topic mastery: %s", str(e))

        return 0.5
