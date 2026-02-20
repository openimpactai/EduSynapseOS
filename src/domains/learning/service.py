# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning Tutor service.

This service manages learning tutor sessions:
- Start a learning session for a topic
- Send/receive messages during the teaching conversation
- Get session details and history
- Complete sessions

The service integrates with:
- Memory system (personalization)
- Educational theory (TheoryOrchestrator)
- Subject-specific tutor agents
- LangGraph workflow (interrupt/resume)
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.orchestration.states.learning_tutor import (
    create_initial_learning_tutor_state,
    select_agent_id,
    select_learning_mode,
    LearningMode,
)
from src.core.orchestration.workflows.learning_tutor import LearningTutorWorkflow
from src.domains.curriculum import CurriculumLookup
from src.infrastructure.database.models.tenant import (
    LearningSession,
    LearningSessionMessage,
    Topic,
    User,
    SemanticMemory,
    EmotionalSignal,
)
from src.models.learning import (
    CompleteSessionRequest,
    get_suggested_actions,
    LearningHandoffContext,
    LearningMode as LearningModeDTO,
    LearningSessionStatus,
    MessageInfo,
    MessageResponse,
    LearningSessionResponse,
    ProgressInfo,
    SessionCompletionResponse,
    SessionHistoryResponse,
    FinalSummary,
    SendMessageRequest,
    StartLearningRequest,
    StartLearningResponse,
    SuggestedAction,
    TopicInfo,
)

logger = logging.getLogger(__name__)


class LearningService:
    """Service for learning tutor sessions.

    Manages the lifecycle of learning tutor sessions:
    1. Start session - creates workflow and generates first message
    2. Send message - processes student message and returns tutor response
    3. Get session - retrieves session details
    4. Get history - retrieves conversation history
    5. Complete session - ends session and records completion

    Attributes:
        workflow: LearningTutorWorkflow instance.

    Example:
        >>> service = LearningService(workflow)
        >>> response = await service.start_session(db, student_id, tenant_id, tenant_code, request)
        >>> message_response = await service.send_message(db, session_id, student_id, request)
    """

    def __init__(self, workflow: LearningTutorWorkflow):
        """Initialize the service.

        Args:
            workflow: LearningTutorWorkflow instance.
        """
        self._workflow = workflow

    async def start_session(
        self,
        db: AsyncSession,
        student_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: StartLearningRequest,
        grade_level: int = 5,
        language: str = "en",
    ) -> StartLearningResponse:
        """Start a learning tutor session.

        Creates a new learning session for the specified topic.
        Determines initial learning mode based on student context.

        If the student has an existing active session, it will be
        auto-completed with reason "new_session_started" before
        starting the new session. This ensures proper memory recording
        and prevents orphaned sessions.

        Args:
            db: Database session.
            student_id: Student's ID.
            tenant_id: Tenant's ID.
            tenant_code: Tenant code for memory operations.
            request: Request with topic details and entry point.
            grade_level: Student's grade level.
            language: Student's language preference.

        Returns:
            StartLearningResponse with session ID and first message.

        Raises:
            ValueError: If topic not found.
        """
        logger.info(
            "Starting learning session: student=%s, topic=%s, entry=%s",
            student_id,
            request.topic_full_code,
            request.entry_point,
        )

        # Check for existing active session and auto-complete if found
        existing_session = await self._get_active_session(db, student_id)
        if existing_session:
            logger.info(
                "Auto-completing existing session %s before starting new session",
                existing_session.id,
            )
            await self._auto_complete_session(
                db=db,
                session=existing_session,
                student_id=student_id,
                tenant_code=tenant_code,
            )

        # Verify topic exists using CurriculumLookup with composite keys
        lookup = CurriculumLookup(db)
        topic = await lookup.get_topic(
            request.topic_framework_code,
            request.topic_subject_code,
            request.topic_grade_code,
            request.topic_unit_code,
            request.topic_code,
        )
        if not topic:
            raise ValueError(f"Topic not found: {request.topic_full_code}")

        # Get student info for personalization
        student = await db.get(User, str(student_id))
        student_info = self._extract_student_info(student)

        # Get topic context using composite keys
        topic_context = await lookup.get_topic_context(
            request.topic_framework_code,
            request.topic_subject_code,
            request.topic_grade_code,
            request.topic_unit_code,
            request.topic_code,
        )

        # Get emotional state
        emotional_state = await self._get_emotional_state(db, student_id)

        # Get topic mastery using full topic code
        topic_mastery = await self._get_topic_mastery(
            db, student_id, request.topic_full_code
        )

        # Determine learning mode
        mode = select_learning_mode(
            emotional_state=emotional_state,
            topic_mastery=topic_mastery,
            entry_point=request.entry_point,
        )

        # Determine agent and subject info from topic_context
        subject_code = topic_context.subject_code if topic_context else request.topic_subject_code
        subject_name = topic_context.subject_name if topic_context else None
        grade_level_name = topic_context.grade_name if topic_context else None
        agent_id = select_agent_id(subject_code)

        # Create workflow session ID
        session_id = str(uuid4())

        # Create initial state
        subject_full_code = f"{request.topic_framework_code}.{request.topic_subject_code}" if request.has_topic() else ""
        initial_state = create_initial_learning_tutor_state(
            session_id=session_id,
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            # Topic context using Central Curriculum composite keys
            topic_full_code=request.topic_full_code or "",
            topic_name=request.topic_name or "",
            subject_full_code=subject_full_code,
            subject_name=subject_name or "General",
            subject_code=subject_code or "general",
            # Student context
            student_id=str(student_id),
            student_age=student_info.get("age"),
            grade_level=grade_level_name or str(grade_level),
            language=student_info.get("language", language),
            topic_mastery=topic_mastery,
            emotional_state=emotional_state,
            interests=student_info.get("interests", []),
            # Entry point
            entry_point=request.entry_point,
        )

        # Run workflow to get first message
        result = await self._workflow.run(initial_state, thread_id=session_id)

        # Create database record with composite key fields
        db_session = LearningSession(
            id=session_id,
            student_id=str(student_id),
            topic_framework_code=request.topic_framework_code,
            topic_subject_code=request.topic_subject_code,
            topic_grade_code=request.topic_grade_code,
            topic_unit_code=request.topic_unit_code,
            topic_code=request.topic_code,
            topic_name=request.topic_name,
            subject=subject_name,
            subject_code=subject_code,
            agent_id=agent_id,
            entry_point=request.entry_point,
            status="active",
            initial_mode=mode.value,
            current_mode=mode.value,
            initial_mastery=Decimal(str(topic_mastery)),
            started_at=datetime.utcnow(),
            turn_count=1,
        )
        db.add(db_session)

        # Save first tutor message
        first_message = result.get("first_message", "")
        if first_message:
            db_message = LearningSessionMessage(
                session_id=session_id,
                sequence=1,
                role="tutor",
                content=first_message,
                learning_mode=mode.value,
            )
            db.add(db_message)

        await db.commit()

        # Get suggested actions
        suggested_actions = get_suggested_actions(
            LearningSessionStatus.ACTIVE,
            LearningModeDTO(mode.value),
        )

        logger.info(
            "Learning session started: session=%s, mode=%s, agent=%s",
            session_id,
            mode.value,
            agent_id,
        )

        return StartLearningResponse(
            session_id=session_id,
            status="active",
            learning_mode=mode.value,
            message=first_message,
            topic=TopicInfo(
                full_code=request.topic_full_code,
                name=request.topic_name,
                subject_code=subject_code,
            ),
            suggested_actions=suggested_actions,
        )

    async def send_message(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: SendMessageRequest,
    ) -> MessageResponse:
        """Send a message in a learning session.

        Processes the student's message/action and returns the tutor's response.
        Handles mode transitions based on actions and conversation flow.

        Args:
            db: Database session.
            session_id: Learning session ID.
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
        session = await db.get(LearningSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Verify student owns session
        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Verify session is active
        if session.status not in ("active", "paused"):
            raise ValueError(f"Session is not active: {session.status}")

        # Record student message if any
        if request.message:
            session.turn_count += 1
            db_message = LearningSessionMessage(
                session_id=str(session_id),
                sequence=session.turn_count,
                role="student",
                content=request.message,
                action=request.action.value,
                learning_mode=session.current_mode,
            )
            db.add(db_message)

        previous_mode = session.current_mode

        # Send message through workflow
        result = await self._workflow.send_message(
            thread_id=str(session_id),
            message=request.message,
            action=request.action.value,
        )

        # Update session from workflow result
        new_mode = result.get("learning_mode", session.current_mode)
        if new_mode != session.current_mode:
            session.mode_transition_count += 1
        session.current_mode = new_mode
        session.understanding_progress = Decimal(
            str(result.get("understanding_progress", 0.0))
        )

        # Update practice tracking if applicable
        metrics = result.get("metrics", {})
        session.practice_questions_attempted = metrics.get("practice_attempted", 0)
        session.practice_questions_correct = metrics.get("practice_correct", 0)

        # Check if workflow ended
        if result.get("status") == "completed":
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            session.completion_reason = result.get("completion_reason", "user_ended")
            session.final_mastery = Decimal(
                str(result.get("understanding_progress", 0.0))
            )

        # Record tutor response
        tutor_response = result.get("last_tutor_response", "")
        if tutor_response:
            session.turn_count += 1
            db_message = LearningSessionMessage(
                session_id=str(session_id),
                sequence=session.turn_count,
                role="tutor",
                content=tutor_response,
                learning_mode=new_mode,
            )
            db.add(db_message)

        await db.commit()

        # Get suggested actions
        status = LearningSessionStatus(session.status)
        mode = LearningModeDTO(new_mode)
        suggested_actions = get_suggested_actions(status, mode)

        mode_changed = new_mode != previous_mode

        return MessageResponse(
            message=tutor_response,
            mode=new_mode,
            turn_number=session.turn_count,
            status=session.status,
            suggested_actions=suggested_actions,
            mode_changed=mode_changed,
            previous_mode=previous_mode if mode_changed else None,
            understanding_progress=float(session.understanding_progress),
        )

    async def get_session(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
    ) -> LearningSessionResponse:
        """Get learning session details.

        Args:
            db: Database session.
            session_id: Learning session ID.
            student_id: Student's ID.

        Returns:
            LearningSessionResponse with session details.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        session = await db.get(LearningSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        return LearningSessionResponse(
            session_id=str(session.id),
            status=session.status,
            topic=TopicInfo(
                full_code=session.topic_full_code,
                name=session.topic_name,
                subject_code=session.subject_code,
            ),
            learning_mode=session.current_mode,
            initial_mode=session.initial_mode,
            entry_point=session.entry_point,
            progress=ProgressInfo(
                turn_count=session.turn_count,
                understanding_progress=float(session.understanding_progress),
                mode_transitions=session.mode_transition_count,
                practice_attempted=session.practice_questions_attempted,
                practice_correct=session.practice_questions_correct,
            ),
            started_at=session.started_at,
            last_activity_at=session.updated_at or session.started_at,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def get_history(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> SessionHistoryResponse:
        """Get conversation history for a session.

        Args:
            db: Database session.
            session_id: Learning session ID.
            student_id: Student's ID.
            limit: Maximum messages to return.
            offset: Offset for pagination.

        Returns:
            SessionHistoryResponse with message list.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        session = await db.get(LearningSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Get total count
        count_query = (
            select(LearningSessionMessage)
            .where(LearningSessionMessage.session_id == str(session_id))
        )
        count_result = await db.execute(count_query)
        total = len(list(count_result.scalars().all()))

        # Get messages ordered by sequence with pagination
        query = (
            select(LearningSessionMessage)
            .where(LearningSessionMessage.session_id == str(session_id))
            .order_by(LearningSessionMessage.sequence)
            .offset(offset)
            .limit(limit + 1)
        )
        result = await db.execute(query)
        messages = list(result.scalars().all())

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        return SessionHistoryResponse(
            session_id=str(session.id),
            messages=[
                MessageInfo(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at,
                    action=msg.action,
                    learning_mode=msg.learning_mode,
                )
                for msg in messages
            ],
            total=total,
            has_more=has_more,
        )

    async def complete_session(
        self,
        db: AsyncSession,
        session_id: UUID,
        student_id: UUID,
        request: CompleteSessionRequest,
        tenant_code: str | None = None,
    ) -> SessionCompletionResponse:
        """Complete a learning session.

        Marks the session as completed, records completion details,
        and stores episodic memory for the session completion.

        Args:
            db: Database session.
            session_id: Learning session ID.
            student_id: Student's ID.
            request: Request with completion details.
            tenant_code: Tenant code for memory recording.

        Returns:
            SessionCompletionResponse with completion details.

        Raises:
            ValueError: If session not found or not owned by student.
        """
        logger.info(
            "Completing session: session=%s, reason=%s, understood=%s",
            session_id,
            request.reason,
            request.understood,
        )

        session = await db.get(LearningSession, str(session_id))
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.student_id != str(student_id):
            raise ValueError("Session does not belong to this student")

        # Update session
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.completion_reason = request.reason

        # Calculate practice accuracy
        practice_accuracy = None
        if session.practice_questions_attempted > 0:
            practice_accuracy = (
                session.practice_questions_correct
                / session.practice_questions_attempted
            )

        # Calculate mastery change
        mastery_change = None
        if session.initial_mastery is not None:
            final_mastery = float(session.understanding_progress)
            mastery_change = final_mastery - float(session.initial_mastery)
            session.final_mastery = Decimal(str(final_mastery))

        await db.commit()

        # Record episodic memory for session completion
        await self._record_session_completion_memory(
            tenant_code=tenant_code,
            student_id=student_id,
            session=session,
            request=request,
        )

        # Build handoff context for Practice transition
        handoff_context = await self._build_handoff_context(
            session_id=str(session_id),
            session=session,
        )

        return SessionCompletionResponse(
            session_id=str(session.id),
            status="completed",
            completion_reason=request.reason,
            final_summary=FinalSummary(
                turn_count=session.turn_count,
                mode_transitions=session.mode_transition_count,
                final_mode=session.current_mode,
                understanding_progress=float(session.understanding_progress),
                practice_accuracy=practice_accuracy,
                mastery_change=mastery_change,
            ),
            handoff_context=handoff_context,
        )

    async def _record_session_completion_memory(
        self,
        tenant_code: str | None,
        student_id: UUID,
        session: LearningSession,
        request: CompleteSessionRequest,
    ) -> None:
        """Record episodic memory for session completion.

        Args:
            tenant_code: Tenant code for memory recording.
            student_id: Student's ID.
            session: The completed learning session.
            request: Completion request with reason and understood flag.
        """
        if not tenant_code:
            logger.warning("Cannot record session memory: tenant_code not provided")
            return

        memory_manager = getattr(self._workflow, "_memory_manager", None)
        if not memory_manager:
            logger.warning("Cannot record session memory: memory_manager not available")
            return

        try:
            # Build topic full code from session
            topic_full_code = None
            if session.topic_framework_code and session.topic_code:
                topic_full_code = (
                    f"{session.topic_framework_code}."
                    f"{session.topic_subject_code}."
                    f"{session.topic_grade_code}."
                    f"{session.topic_unit_code}."
                    f"{session.topic_code}"
                )

            # Record episodic memory - session completion
            await memory_manager.record_learning_event(
                tenant_code=tenant_code,
                student_id=student_id,
                event_type="learning_session_completed",
                topic=session.topic_name or "",
                data={
                    "session_id": str(session.id),
                    "topic_full_code": topic_full_code,
                    "initial_mode": session.initial_mode,
                    "final_mode": session.current_mode,
                    "mode_transitions": session.mode_transition_count,
                    "turn_count": session.turn_count,
                    "understanding_progress": float(session.understanding_progress),
                    "understood": request.understood,
                    "completion_reason": request.reason,
                    "entry_point": session.entry_point,
                    "workflow_type": "learning_tutor",
                    "reportable": True,
                },
                importance=0.7,
                topic_full_code=topic_full_code,
            )

            # Update semantic memory - topic mastery based on understanding
            if topic_full_code:
                understanding_progress = float(session.understanding_progress)
                await memory_manager.record_learning_session_completion(
                    tenant_code=tenant_code,
                    student_id=student_id,
                    topic_full_code=topic_full_code,
                    understanding_progress=understanding_progress,
                    session_completed=True,
                )

            logger.info(
                "Recorded session completion memory: session=%s, topic=%s",
                session.id,
                session.topic_name,
            )

        except Exception as e:
            logger.warning("Failed to record session completion memory: %s", str(e))

    async def _build_handoff_context(
        self,
        session_id: str,
        session: LearningSession,
    ) -> LearningHandoffContext | None:
        """Build handoff context from workflow state for Practice transition.

        Extracts comprehension evaluation results, key concepts, and weak concepts
        from the workflow state to enable informed question selection in Practice.

        Args:
            session_id: The session ID.
            session: The learning session database model.

        Returns:
            LearningHandoffContext with learning session data, or None on error.
        """
        try:
            # Get workflow state
            compiled = self._workflow.compile()
            config = {"configurable": {"thread_id": session_id}}
            state_snapshot = await compiled.aget_state(config)

            if not state_snapshot or not state_snapshot.values:
                logger.warning(
                    "No workflow state found for session %s, building basic handoff",
                    session_id,
                )
                # Return basic handoff with DB data only
                return LearningHandoffContext(
                    source="learning_tutor",
                    session_id=session_id,
                    topic_full_code=session.topic_full_code,
                    verified_understanding=float(session.understanding_progress),
                    understanding_verified=False,
                    concepts_verified=[],
                    concepts_weak=[],
                    misconceptions_addressed=[],
                    preferred_learning_style=session.current_mode,
                )

            state = state_snapshot.values

            # Extract key concepts and concepts needing clarification
            key_concepts: list[str] = state.get("key_concepts", [])
            concepts_to_clarify: list[str] = state.get("_concepts_to_clarify", [])

            # Concepts verified = key_concepts minus concepts_to_clarify
            concepts_verified = [c for c in key_concepts if c not in concepts_to_clarify]

            # Extract misconceptions addressed (flatten list of dicts to list of strings)
            misconceptions_raw: list[dict] = state.get("_misconceptions_to_address", [])
            misconceptions_addressed = [
                m.get("misconception", str(m))
                for m in misconceptions_raw
                if isinstance(m, dict)
            ]

            # Get understanding verification status
            understanding_verified = state.get("understanding_verified", False)

            # Get understanding progress from state or fall back to DB
            understanding_progress = state.get(
                "understanding_progress",
                float(session.understanding_progress),
            )

            # Build topic full code from session composite keys
            topic_full_code = session.topic_full_code

            return LearningHandoffContext(
                source="learning_tutor",
                session_id=session_id,
                topic_full_code=topic_full_code,
                verified_understanding=understanding_progress,
                understanding_verified=understanding_verified,
                concepts_verified=concepts_verified,
                concepts_weak=concepts_to_clarify,
                misconceptions_addressed=misconceptions_addressed,
                preferred_learning_style=state.get("learning_mode", session.current_mode),
            )

        except Exception as e:
            logger.warning(
                "Failed to build handoff context for session %s: %s",
                session_id,
                str(e),
            )
            # Return basic handoff on error
            return LearningHandoffContext(
                source="learning_tutor",
                session_id=session_id,
                topic_full_code=session.topic_full_code,
                verified_understanding=float(session.understanding_progress),
                understanding_verified=False,
                concepts_verified=[],
                concepts_weak=[],
                misconceptions_addressed=[],
                preferred_learning_style=session.current_mode,
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_active_session(
        self,
        db: AsyncSession,
        student_id: UUID,
    ) -> LearningSession | None:
        """Get active learning session for student.

        Args:
            db: Database session.
            student_id: Student's ID.

        Returns:
            Active session or None.
        """
        query = (
            select(LearningSession)
            .where(
                LearningSession.student_id == str(student_id),
                LearningSession.status.in_(["pending", "active", "paused"]),
            )
            .limit(1)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def _auto_complete_session(
        self,
        db: AsyncSession,
        session: LearningSession,
        student_id: UUID,
        tenant_code: str | None,
    ) -> None:
        """Auto-complete an existing session when starting a new one.

        This ensures proper memory recording and state cleanup when a student
        starts a new session without explicitly ending the previous one.

        Args:
            db: Database session.
            session: The existing active session to complete.
            student_id: Student's ID.
            tenant_code: Tenant code for memory recording.
        """
        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.completion_reason = "new_session_started"

        await db.commit()

        # Record episodic memory for the auto-completed session
        if tenant_code:
            memory_manager = getattr(self._workflow, "_memory_manager", None)
            if memory_manager:
                try:
                    # Build topic full code from session
                    topic_full_code = None
                    if session.topic_framework_code and session.topic_code:
                        topic_full_code = (
                            f"{session.topic_framework_code}."
                            f"{session.topic_subject_code}."
                            f"{session.topic_grade_code}."
                            f"{session.topic_unit_code}."
                            f"{session.topic_code}"
                        )

                    await memory_manager.record_learning_event(
                        tenant_code=tenant_code,
                        student_id=student_id,
                        event_type="learning_session_completed",
                        topic=session.topic_name or "",
                        data={
                            "session_id": str(session.id),
                            "topic_full_code": topic_full_code,
                            "initial_mode": session.initial_mode,
                            "final_mode": session.current_mode,
                            "mode_transitions": session.mode_transition_count,
                            "turn_count": session.turn_count,
                            "understanding_progress": float(session.understanding_progress),
                            "understood": None,  # Unknown - auto-completed
                            "completion_reason": "new_session_started",
                            "entry_point": session.entry_point,
                            "workflow_type": "learning_tutor",
                            "reportable": True,
                        },
                        importance=0.5,  # Lower importance since auto-completed
                        topic_full_code=topic_full_code,
                    )

                    logger.debug(
                        "Recorded memory for auto-completed session %s",
                        session.id,
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to record memory for auto-completed session: %s",
                        str(e),
                    )

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


    async def _get_emotional_state(
        self,
        db: AsyncSession,
        student_id: UUID,
    ) -> str:
        """Get student's current emotional state.

        Args:
            db: Database session.
            student_id: Student's ID.

        Returns:
            Emotional state string.
        """
        try:
            query = (
                select(EmotionalSignal)
                .where(EmotionalSignal.student_id == str(student_id))
                .order_by(desc(EmotionalSignal.created_at))
                .limit(1)
            )
            result = await db.execute(query)
            signal = result.scalar_one_or_none()

            if signal and signal.detected_emotion:
                return signal.detected_emotion

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
            topic_full_code: Full topic code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").

        Returns:
            Mastery level (0.0-1.0).
        """
        if not topic_full_code:
            return 0.5

        try:
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
