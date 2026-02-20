# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice session service.

This service manages practice sessions by orchestrating the PracticeWorkflow.
It handles session lifecycle, answer submissions, and session completion.

The service does NOT make LLM calls directly - all AI interactions happen
through the workflow which uses the Agent layer.

Note on database operations:
Memory context loading uses sync database methods via asyncio.to_thread()
to avoid SQLAlchemy greenlet context issues. This is consistent with how
workflow nodes handle database reads.
See: /docs/memory-theory-validation/LANGGRAPH-ASYNC-DB-ISSUE.md
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.agents import AgentFactory
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.emotional import EmotionalStateService
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.orchestration.states.practice import (
    create_initial_practice_state,
    PracticeState,
)
from src.core.orchestration.workflows.practice import PracticeWorkflow
from src.core.personas.manager import PersonaManager
from src.domains.curriculum import (
    CurriculumLookup,
    TopicContext,
    COUNTRY_TO_LANGUAGE,
    DEFAULT_LANGUAGE,
)
from src.infrastructure.database.models.tenant.curriculum import (
    Topic as TopicModel,
    Unit as UnitModel,
    Subject as SubjectModel,
    LearningObjective as LearningObjectiveModel,
)
from src.infrastructure.database.models.tenant.practice import (
    PracticeSession as PracticeSessionModel,
    PracticeQuestion as PracticeQuestionModel,
    StudentAnswer as StudentAnswerModel,
    EvaluationResult as EvaluationResultModel,
)
from src.models.practice import (
    StartPracticeRequest,
    PracticeSessionResponse,
    QuestionResponse,
    QuestionOption,
    SubmitAnswerRequest,
    AnswerResultResponse,
    EvaluationResponse,
    SessionProgressResponse,
    SessionCompletionResponse,
    MasteryUpdateInfo,
    SessionStatus,
    SessionType,
    EvaluationStrategy,
    EvaluationConfig,
    QuestionType,
)

logger = logging.getLogger(__name__)


class PracticeServiceError(Exception):
    """Base exception for practice service errors."""

    pass


class SessionNotFoundError(PracticeServiceError):
    """Raised when a session is not found."""

    pass


class SessionNotActiveError(PracticeServiceError):
    """Raised when session is not in active state."""

    pass


class TopicNotFoundError(PracticeServiceError):
    """Raised when a topic is not found in curriculum."""

    pass


class PracticeService:
    """Service for managing practice sessions.

    This service orchestrates practice sessions using the PracticeWorkflow.
    It manages session state, handles answer submissions, and tracks progress.

    The service does NOT make LLM calls - all AI interactions are handled
    by the PracticeWorkflow through the Agent layer.

    Attributes:
        _db: Async database session.
        _workflow: The practice workflow instance.
        _memory_manager: Memory manager for student context.

    Example:
        >>> service = PracticeService(db, agent_factory, memory_manager, ...)
        >>> session = await service.start_session(student_id, tenant_id, request)
        >>> result = await service.submit_answer(session.id, answer_request)
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
        """Initialize the practice service.

        Args:
            db: Async database session.
            agent_factory: Factory for creating agents.
            memory_manager: Manager for memory operations.
            rag_retriever: Retriever for RAG context.
            theory_orchestrator: Orchestrator for educational theories.
            persona_manager: Manager for personas.
            checkpointer: Checkpointer for workflow state persistence.
            emotional_service: Service for emotional signal recording.
        """
        self._db = db
        self._memory_manager = memory_manager
        self._theory_orchestrator = theory_orchestrator

        # Create EmotionalStateService if not provided
        if emotional_service is None:
            emotional_service = EmotionalStateService(db=db)
        self._emotional_service = emotional_service

        # Initialize workflow with checkpointer and emotional service
        self._workflow = PracticeWorkflow(
            agent_factory=agent_factory,
            memory_manager=memory_manager,
            rag_retriever=rag_retriever,
            theory_orchestrator=theory_orchestrator,
            persona_manager=persona_manager,
            checkpointer=checkpointer,
            emotional_service=emotional_service,
            db_session=db,
        )

    async def start_session(
        self,
        student_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: StartPracticeRequest,
    ) -> tuple[PracticeSessionResponse, QuestionResponse | None]:
        """Start a new practice session.

        Creates a new session and generates the first question using the workflow.

        Args:
            student_id: The student's ID.
            tenant_id: The tenant's ID.
            tenant_code: The tenant's code for memory operations.
            request: Session configuration request with explicit composite key parts.

        Returns:
            Tuple of (session response, first question or None).
        """
        session_id = uuid4()
        thread_id = f"practice_{session_id}"

        logger.info(
            "Starting practice session: session=%s, student=%s, topic=%s",
            session_id,
            student_id,
            request.topic_full_code,
        )

        # Resolve topic using explicit composite key parts from request
        lookup = CurriculumLookup(self._db)
        topic: TopicModel | None = None

        if request.has_topic():
            topic = await lookup.get_topic(
                request.topic_framework_code,
                request.topic_subject_code,
                request.topic_grade_code,
                request.topic_unit_code,
                request.topic_code,
            )
            if not topic:
                raise TopicNotFoundError(
                    f"Topic not found: {request.topic_full_code}"
                )

        # Resolve subject for RANDOM mode using explicit composite key parts
        subject: SubjectModel | None = None
        if request.has_subject():
            subject = await lookup.get_subject(
                request.subject_framework_code,
                request.subject_code,
            )
            if not subject:
                logger.warning(
                    "Subject not found for random mode: %s.%s",
                    request.subject_framework_code,
                    request.subject_code,
                )

        # Resolve learning objective using explicit composite key parts
        learning_objective: LearningObjectiveModel | None = None
        if request.has_objective():
            learning_objective = await lookup.get_objective(
                request.objective_framework_code,
                request.objective_subject_code,
                request.objective_grade_code,
                request.objective_unit_code,
                request.objective_topic_code,
                request.objective_code,
            )
            if learning_objective:
                logger.debug(
                    "Learning objective loaded: %s, text=%s",
                    request.objective_full_code,
                    learning_objective.objective[:50] if learning_objective.objective else None,
                )

        # Extract learning_session_id from handoff context (if from Learning Tutor)
        learning_session_id = None
        if request.handoff_context:
            source = request.handoff_context.get("source", "")
            if source == "learning_tutor":
                learning_session_id = request.handoff_context.get("session_id")

        # Create database record with composite key fields
        db_session = PracticeSessionModel(
            id=session_id,
            student_id=student_id,
            learning_session_id=learning_session_id,
            topic_framework_code=request.topic_framework_code,
            topic_subject_code=request.topic_subject_code,
            topic_grade_code=request.topic_grade_code,
            topic_unit_code=request.topic_unit_code,
            topic_code=request.topic_code,
            session_type=request.session_type.value,
            persona_id=request.persona_id,
            status=SessionStatus.ACTIVE.value,
            questions_total=0,
            questions_answered=0,
            questions_correct=0,
            time_spent_seconds=0,
        )
        self._db.add(db_session)
        await self._db.flush()

        # Map session type to practice mode
        mode_map = {
            "quick": "quick",
            "focused": "deep",
            "deep": "deep",
            "review": "weakness_focus",
            "random": "random",
            "assessment": "exam_prep",
            "diagnostic": "quick",
        }
        practice_mode = mode_map.get(request.session_type.value, "quick")

        # For RANDOM mode, load topics from subject
        available_topics: list[dict] = []
        subject_context: dict | None = None
        if practice_mode == "random" and subject:
            available_topics, subject_context = await self._get_topics_from_subject(
                subject.framework_code, subject.code
            )
            if not available_topics:
                logger.warning(
                    "No topics found for subject %s.%s, falling back to quick mode",
                    subject.framework_code,
                    subject.code,
                )
                practice_mode = "quick"
            else:
                logger.info(
                    "RANDOM mode: Loaded %d topics from subject %s",
                    len(available_topics),
                    subject.name,
                )

        # Get topic context from curriculum hierarchy
        topic_context: TopicContext | None = None
        topic_name = request.topic_name or "general"

        # For RANDOM mode, use subject name as initial topic name
        if practice_mode == "random" and subject:
            topic_name = f"Mixed Topics from {subject.name}"

        if topic:
            topic_context = await lookup.get_topic_context(
                topic.framework_code,
                topic.subject_code,
                topic.grade_code,
                topic.unit_code,
                topic.code,
            )
            if topic_context:
                topic_name = topic_context.topic_name
                logger.info(
                    "Topic context loaded: topic=%s, grade=%s, subject=%s, language=%s",
                    topic_context.topic_name,
                    topic_context.grade_name,
                    topic_context.subject_name,
                    topic_context.language,
                )
            else:
                topic_name = request.topic_name or topic.name or "general"
                logger.warning(
                    "Topic context not found, using fallback: topic=%s",
                    topic_name,
                )

        # Determine educational context from topic_context or subject_context
        if topic_context:
            edu_subject_name = topic_context.subject_name
            edu_subject_code = topic_context.subject_code
            edu_grade_level = topic_context.grade_name
            edu_grade_level_code = topic_context.grade_code
            edu_age_range = topic_context.age_description
            edu_curriculum = topic_context.framework_name
            edu_curriculum_code = topic_context.framework_code
            edu_language = topic_context.language
            edu_unit_name = topic_context.unit_name
        elif subject_context:
            edu_subject_name = subject_context.get("name")
            edu_subject_code = subject_context.get("code")
            edu_grade_level = subject_context.get("grade_level_name")
            edu_grade_level_code = subject_context.get("grade_level_code")
            edu_age_range = subject_context.get("age_range")
            edu_curriculum = subject_context.get("curriculum_name")
            edu_curriculum_code = subject_context.get("curriculum_code")
            edu_language = subject_context.get("language", "en")
            edu_unit_name = None
        else:
            edu_subject_name = None
            edu_subject_code = None
            edu_grade_level = None
            edu_grade_level_code = None
            edu_age_range = None
            edu_curriculum = None
            edu_curriculum_code = None
            edu_language = "en"
            edu_unit_name = None

        # Pre-load memory context BEFORE workflow execution
        # Uses sync database methods via asyncio.to_thread() to avoid greenlet context issues
        # See: /docs/memory-theory-validation/LANGGRAPH-ASYNC-DB-ISSUE.md
        memory_context_data = {}
        theory_recommendations_data = {}
        full_context = None

        # Determine topic full code for mastery loading
        # topic_full_code: for topic-specific mastery (e.g., "UK-NC-2014.MATHS.Y4.FRAC.EQUIV")
        topic_full_code_for_mastery = request.topic_full_code

        # For random mode, don't use specific topic mastery
        if practice_mode == "random":
            topic_full_code_for_mastery = "random"

        try:
            # Use sync method via asyncio.to_thread to avoid greenlet issues
            full_context = await asyncio.to_thread(
                self._memory_manager.get_full_context_sync,
                tenant_code,
                student_id,
                topic_full_code_for_mastery,
            )
            if full_context:
                memory_context_data = full_context.model_dump()
                # Log what mastery source was loaded
                if full_context.topic_mastery:
                    logger.info(
                        "Pre-loaded TOPIC mastery: student=%s, topic=%s, mastery=%.2f, attempts=%d",
                        student_id,
                        topic_full_code_for_mastery,
                        full_context.topic_mastery.mastery_level,
                        full_context.topic_mastery.attempts_total,
                    )
                elif full_context.semantic and full_context.semantic.overall_mastery > 0:
                    logger.info(
                        "Pre-loaded OVERALL mastery: student=%s, mastery=%.2f",
                        student_id,
                        full_context.semantic.overall_mastery,
                    )
                else:
                    logger.info(
                        "No existing mastery data for student=%s (new student or topic)",
                        student_id,
                    )
        except Exception as e:
            logger.warning("Failed to pre-load memory context: %s", str(e))
            # Continue with empty context - don't rollback as it would undo session creation

        # Pre-load emotional context (prefer handoff context if available)
        emotional_context = None
        if request.handoff_context and request.handoff_context.get("source") == "companion":
            emotional_context = request.handoff_context.get("emotional_state")
        elif self._emotional_service:
            try:
                emotional_state = await self._emotional_service.get_current_state(
                    student_id=student_id,
                )
                if emotional_state:
                    emotional_context = emotional_state.to_dict()
            except Exception as e:
                logger.debug("Emotional context not available: %s", str(e))

        # Pre-load theory recommendations
        try:
            theory_recs = await self._theory_orchestrator.get_recommendations(
                tenant_code=tenant_code,
                student_id=str(student_id),
                topic=topic_name,
                memory_context=full_context,
                emotional_context=emotional_context,
            )
            if theory_recs:
                theory_recommendations_data = theory_recs.model_dump()
                logger.debug(
                    "Pre-loaded theory recommendations: student=%s, difficulty=%.2f",
                    student_id,
                    theory_recs.difficulty if theory_recs.difficulty else 0.5,
                )
        except Exception as e:
            logger.warning("Failed to pre-load theory recommendations: %s", str(e))
            # Continue with empty recommendations - don't rollback as it would undo session creation

        # Create initial workflow state with educational context
        initial_state = create_initial_practice_state(
            session_id=str(session_id),
            tenant_id=str(tenant_id),
            tenant_code=tenant_code,
            student_id=str(student_id),
            topic_full_code=request.topic_full_code,
            topic=topic_name,
            learning_objective_full_code=request.objective_full_code,
            learning_objective_text=learning_objective.objective if learning_objective else None,
            difficulty=request.difficulty or 0.5,
            target_question_count=request.question_count,
            mode=practice_mode,
            persona_id=request.persona_id or "tutor",
            question_type=request.question_type.value if request.question_type else None,
            # Handoff context from Learning Tutor or Companion
            handoff_context=request.handoff_context,
            # Random mode configuration
            subject_full_code=request.subject_full_code,
            available_topics=available_topics,
            # Educational context
            subject_name=edu_subject_name,
            grade_level=edu_grade_level,
            grade_level_code=edu_grade_level_code,
            age_range=edu_age_range,
            curriculum=edu_curriculum,
            curriculum_code=edu_curriculum_code,
            language=edu_language,
            unit_name=edu_unit_name,
            # Pre-loaded context (loaded before workflow for transaction isolation)
            memory_context=memory_context_data,
            theory_recommendations=theory_recommendations_data,
        )

        # Run workflow to generate first question
        state = await self._workflow.run(initial_state, thread_id)

        # Extract question from state
        first_question = None
        current_q = state.get("current_question")
        if current_q:
            question_id = uuid4()
            first_question = self._state_question_to_response(current_q, 1, question_id)

            # Save question to database with composite key fields
            await self._save_question_to_db(
                session_id=session_id,
                question=current_q,
                sequence=1,
                question_id=question_id,
                topic_codes={
                    "framework_code": request.topic_framework_code,
                    "subject_code": request.topic_subject_code,
                    "grade_code": request.topic_grade_code,
                    "unit_code": request.topic_unit_code,
                    "code": request.topic_code,
                } if request.has_topic() else None,
                objective_codes={
                    "framework_code": request.objective_framework_code,
                    "subject_code": request.objective_subject_code,
                    "grade_code": request.objective_grade_code,
                    "unit_code": request.objective_unit_code,
                    "topic_code": request.objective_topic_code,
                    "code": request.objective_code,
                } if request.has_objective() else None,
            )

        # Update session with question count
        db_session.questions_total = state.get("target_question_count", 10)

        await self._db.commit()
        await self._db.refresh(db_session)

        # Build response
        session_response = self._build_session_response(
            db_session,
            topic_full_code=request.topic_full_code,
            topic_name=topic_name,
            subject_full_code=request.subject_full_code or (
                f"{edu_curriculum_code}.{edu_subject_code}" if edu_curriculum_code and edu_subject_code else None
            ),
        )

        return session_response, first_question

    async def get_session(
        self,
        session_id: UUID,
        student_id: UUID,
    ) -> PracticeSessionResponse:
        """Get a practice session by ID.

        Args:
            session_id: The session ID.
            student_id: The student ID (for authorization).

        Returns:
            Session response with code-based identification.

        Raises:
            SessionNotFoundError: If session not found.
        """
        # Query session with topic relationship using composite key
        stmt = (
            select(PracticeSessionModel)
            .options(selectinload(PracticeSessionModel.topic))
            .where(
                PracticeSessionModel.id == session_id,
                PracticeSessionModel.student_id == student_id,
            )
        )
        result = await self._db.execute(stmt)
        db_session = result.scalar_one_or_none()

        if not db_session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        # Extract topic info from relationship or composite keys
        topic_name = None
        if db_session.topic:
            topic_name = db_session.topic.name

        return self._build_session_response(
            db_session,
            topic_full_code=db_session.topic_full_code,
            topic_name=topic_name,
            subject_full_code=f"{db_session.topic_framework_code}.{db_session.topic_subject_code}" if db_session.topic_framework_code and db_session.topic_subject_code else None,
        )

    async def get_current_question(
        self,
        session_id: UUID,
        student_id: UUID,
    ) -> QuestionResponse | None:
        """Get the current question for a session.

        Retrieves the current unanswered question from the database.

        Args:
            session_id: The session ID.
            student_id: The student ID (for authorization).

        Returns:
            Current question or None if session is complete.

        Raises:
            SessionNotFoundError: If session not found.
            SessionNotActiveError: If session is not active.
        """
        # Verify session exists and is active
        stmt = select(PracticeSessionModel).where(
            PracticeSessionModel.id == session_id,
            PracticeSessionModel.student_id == student_id,
        )
        result = await self._db.execute(stmt)
        db_session = result.scalar_one_or_none()

        if not db_session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        if db_session.status != SessionStatus.ACTIVE.value:
            raise SessionNotActiveError(f"Session {session_id} is not active")

        # Get the latest question from DB (current unanswered question)
        sequence = db_session.questions_answered + 1
        stmt = select(PracticeQuestionModel).where(
            PracticeQuestionModel.session_id == session_id,
            PracticeQuestionModel.sequence == sequence,
        )
        result = await self._db.execute(stmt)
        db_question = result.scalar_one_or_none()

        if not db_question:
            return None

        # Build response from DB record
        options = None
        if db_question.data and db_question.data.get("options"):
            options = [
                QuestionOption(
                    key=opt.get("key", chr(97 + i)),
                    text=opt.get("text", ""),
                    is_correct=None,  # Never expose to student
                )
                for i, opt in enumerate(db_question.data["options"])
            ]

        return QuestionResponse(
            id=db_question.id,
            sequence=db_question.sequence,
            content=db_question.content,
            question_type=db_question.display_hint,
            options=options,
            difficulty=db_question.difficulty,
            bloom_level=db_question.bloom_level,
            topic_name=None,
            hints_available=len(db_question.hints or []),
            time_limit_seconds=None,
        )

    async def submit_answer(
        self,
        session_id: UUID,
        student_id: UUID,
        tenant_id: UUID,
        tenant_code: str,
        request: SubmitAnswerRequest,
    ) -> AnswerResultResponse:
        """Submit an answer for evaluation.

        Args:
            session_id: The session ID.
            student_id: The student ID.
            tenant_id: The tenant ID.
            tenant_code: The tenant code for memory operations.
            request: Answer submission request.

        Returns:
            Answer result with evaluation and next question.

        Raises:
            SessionNotFoundError: If session not found.
            SessionNotActiveError: If session is not active.
        """
        # Get session
        stmt = select(PracticeSessionModel).where(
            PracticeSessionModel.id == session_id,
            PracticeSessionModel.student_id == student_id,
        )
        result = await self._db.execute(stmt)
        db_session = result.scalar_one_or_none()

        if not db_session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        if db_session.status != SessionStatus.ACTIVE.value:
            raise SessionNotActiveError(f"Session {session_id} is not active")

        logger.info(
            "Submitting answer: session=%s, question=%s",
            session_id,
            request.question_id,
        )

        # Submit to workflow
        thread_id = f"practice_{session_id}"
        state = await self._workflow.submit_answer(
            thread_id=thread_id,
            answer=request.answer,
            time_spent=request.time_spent_seconds,
            hints_used=request.hints_viewed,
        )

        # Extract evaluation from state
        last_evaluation = state.get("last_evaluation", {})
        is_correct = last_evaluation.get("is_correct", False)
        score = last_evaluation.get("score", 0.0)
        feedback = last_evaluation.get("feedback", "")

        # Save answer to database
        answer_id = uuid4()
        answer_record = StudentAnswerModel(
            id=answer_id,
            question_id=request.question_id,
            student_id=student_id,
            answer={"value": request.answer},
            time_spent_seconds=request.time_spent_seconds,
            hints_viewed=request.hints_viewed,
        )
        self._db.add(answer_record)

        # Save evaluation result to database
        # Get evaluation_strategy from workflow result and convert to DB-compatible value
        eval_strategy = last_evaluation.get("evaluation_strategy")
        if isinstance(eval_strategy, EvaluationStrategy):
            db_eval_method = eval_strategy.db_value
        else:
            db_eval_method = "semantic"  # Default for LLM-based

        evaluation_record = EvaluationResultModel(
            id=uuid4(),
            answer_id=answer_id,
            is_correct=is_correct,
            score=score * 100,  # Convert 0-1 to 0-100 scale
            feedback=feedback,
            detailed_feedback=last_evaluation.get("detailed_feedback"),
            evaluation_method=db_eval_method,
            confidence=last_evaluation.get("confidence"),
            reasoning=last_evaluation.get("reasoning"),
            misconceptions=last_evaluation.get("misconceptions", []),
        )
        self._db.add(evaluation_record)

        # Update session stats
        db_session.questions_answered += 1
        if is_correct:
            db_session.questions_correct += 1
        db_session.time_spent_seconds += request.time_spent_seconds

        # Save theory state to checkpoint_data for intelligence endpoints
        theory_recs = state.get("theory_recommendations", {})
        memory_context = state.get("memory_context", {})
        if theory_recs:
            db_session.checkpoint_data = db_session.checkpoint_data or {}
            db_session.checkpoint_data["theory_state"] = {
                "bloom_level": theory_recs.get("bloom_level"),
                "difficulty": theory_recs.get("difficulty"),
                "scaffold_level": theory_recs.get("scaffold_level"),
                "content_format": theory_recs.get("content_format"),
                "questioning_style": theory_recs.get("questioning_style"),
                "guide_vs_tell_ratio": theory_recs.get("guide_vs_tell_ratio"),
                "hints_enabled": theory_recs.get("hints_enabled"),
                # VARK profile from theory
                "vark_visual": theory_recs.get("vark_visual", 0.25),
                "vark_auditory": theory_recs.get("vark_auditory", 0.25),
                "vark_reading": theory_recs.get("vark_reading", 0.25),
                "vark_kinesthetic": theory_recs.get("vark_kinesthetic", 0.25),
                # Additional data
                "mastery_threshold": theory_recs.get("mastery_threshold", 0.8),
            }
            # Add current mastery from memory context
            semantic = memory_context.get("semantic", {})
            topic_mastery = memory_context.get("topic_mastery", {})
            if topic_mastery:
                db_session.checkpoint_data["theory_state"]["mastery"] = topic_mastery.get("mastery_level", 0.5)
            elif semantic:
                db_session.checkpoint_data["theory_state"]["mastery"] = semantic.get("overall_mastery", 0.5)

            # Log decision for decision history with ALL theory outputs
            if "theory_decisions" not in db_session.checkpoint_data:
                db_session.checkpoint_data["theory_decisions"] = []
            db_session.checkpoint_data["theory_decisions"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "trigger": "answer_evaluated",
                "inputs": {
                    "is_correct": is_correct,
                    "score": score,
                    "question_index": db_session.questions_answered,
                },
                "output": {
                    # Core adaptive parameters
                    "difficulty": theory_recs.get("difficulty"),
                    "bloom_level": theory_recs.get("bloom_level"),
                    "scaffold_level": theory_recs.get("scaffold_level"),
                    # VARK/Content format
                    "content_format": theory_recs.get("content_format"),
                    # Socratic method
                    "guide_vs_tell_ratio": theory_recs.get("guide_vs_tell_ratio"),
                    "questioning_style": theory_recs.get("questioning_style"),
                    # Support settings
                    "hints_enabled": theory_recs.get("hints_enabled"),
                    # Mastery
                    "advance_to_next": theory_recs.get("advance_to_next"),
                },
                "confidence": theory_recs.get("overall_confidence", theory_recs.get("confidence", 0.8)),
            })

        await self._db.flush()

        # Build evaluation response
        # Get evaluation_strategy from workflow result
        if isinstance(eval_strategy, EvaluationStrategy):
            api_eval_method = eval_strategy
        else:
            api_eval_method = EvaluationStrategy.SEMANTIC

        evaluation = EvaluationResponse(
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            correct_answer=last_evaluation.get("correct_answer"),
            explanation=last_evaluation.get("explanation"),
            misconceptions=[],
            evaluation_method=api_eval_method,
            confidence=last_evaluation.get("confidence"),
        )

        # Build progress response
        # Session is complete if workflow says so OR all questions are answered
        is_complete = (
            state.get("status") == "completed"
            or db_session.questions_answered >= db_session.questions_total
        )
        progress = SessionProgressResponse(
            questions_total=db_session.questions_total,
            questions_answered=db_session.questions_answered,
            questions_correct=db_session.questions_correct,
            questions_remaining=max(
                0, db_session.questions_total - db_session.questions_answered
            ),
            current_score=self._calculate_score(db_session),
            time_spent_seconds=db_session.time_spent_seconds,
            is_complete=is_complete,
        )

        # Get next question if available
        next_question = None
        if not is_complete:
            current_q = state.get("current_question")
            if current_q:
                sequence = db_session.questions_answered + 1
                # Generate a single question_id to use for both response and DB
                next_question_id = uuid4()

                next_question = self._state_question_to_response(current_q, sequence, next_question_id)

                # Save next question to database with composite key fields from session
                # Build topic_codes dict from session's composite key fields
                topic_codes = None
                if db_session.topic_framework_code:
                    topic_codes = {
                        "framework_code": db_session.topic_framework_code,
                        "subject_code": db_session.topic_subject_code,
                        "grade_code": db_session.topic_grade_code,
                        "unit_code": db_session.topic_unit_code,
                        "code": db_session.topic_code,
                    }

                # Get objective codes from state if available
                objective_codes = None
                obj_full_code = state.get("learning_objective_full_code")
                if obj_full_code:
                    parts = obj_full_code.split(".")
                    if len(parts) == 6:
                        objective_codes = {
                            "framework_code": parts[0],
                            "subject_code": parts[1],
                            "grade_code": parts[2],
                            "unit_code": parts[3],
                            "topic_code": parts[4],
                            "code": parts[5],
                        }

                await self._save_question_to_db(
                    session_id,
                    current_q,
                    sequence,
                    next_question_id,
                    topic_codes=topic_codes,
                    objective_codes=objective_codes,
                )
        else:
            # Session is complete - update status and calculate final score
            db_session.status = SessionStatus.COMPLETED.value
            db_session.ended_at = datetime.utcnow()
            db_session.score = self._calculate_score(db_session)

        await self._db.commit()

        return AnswerResultResponse(
            answer_id=answer_id,
            evaluation=evaluation,
            session_progress=progress,
            next_question=next_question,
        )

    async def complete_session(
        self,
        session_id: UUID,
        student_id: UUID,
    ) -> SessionCompletionResponse:
        """Complete a practice session.

        Args:
            session_id: The session ID.
            student_id: The student ID.

        Returns:
            Session completion response with summary.

        Raises:
            SessionNotFoundError: If session not found.
        """
        # Get session with topic relationship eager loaded
        stmt = (
            select(PracticeSessionModel)
            .options(selectinload(PracticeSessionModel.topic))
            .where(
                PracticeSessionModel.id == session_id,
                PracticeSessionModel.student_id == student_id,
            )
        )
        result = await self._db.execute(stmt)
        db_session = result.scalar_one_or_none()

        if not db_session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        logger.info("Completing practice session: session=%s", session_id)

        # Complete workflow if still running
        thread_id = f"practice_{session_id}"
        compiled = self._workflow.compile()
        state_snapshot = await compiled.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )

        # Get final state metrics
        metrics = {}
        if state_snapshot and state_snapshot.values:
            metrics = state_snapshot.values.get("metrics", {})

        # Update database
        db_session.status = SessionStatus.COMPLETED.value
        db_session.ended_at = datetime.utcnow()
        db_session.score = self._calculate_score(db_session)

        await self._db.commit()

        # Get topic name from the eager-loaded relationship
        topic_name = db_session.topic.name if db_session.topic else None

        # Build response
        session_response = self._build_session_response(
            db_session,
            topic_full_code=db_session.topic_full_code,
            topic_name=topic_name,
            subject_full_code=f"{db_session.topic_framework_code}.{db_session.topic_subject_code}" if db_session.topic_framework_code and db_session.topic_subject_code else None,
        )

        return SessionCompletionResponse(
            session=session_response,
            summary=metrics.get("summary"),
            performance_breakdown={
                "accuracy": session_response.accuracy,
                "questions_answered": db_session.questions_answered,
                "questions_correct": db_session.questions_correct,
            },
            recommendations=metrics.get("recommendations", []),
            mastery_updates=[],  # Would come from memory updates
        )

    async def pause_session(
        self,
        session_id: UUID,
        student_id: UUID,
    ) -> PracticeSessionResponse:
        """Pause a practice session.

        Args:
            session_id: The session ID.
            student_id: The student ID.

        Returns:
            Updated session response.
        """
        # Update session status
        update_stmt = (
            update(PracticeSessionModel)
            .where(
                PracticeSessionModel.id == session_id,
                PracticeSessionModel.student_id == student_id,
                PracticeSessionModel.status == SessionStatus.ACTIVE.value,
            )
            .values(status=SessionStatus.PAUSED.value)
        )
        result = await self._db.execute(update_stmt)

        if result.rowcount == 0:
            raise SessionNotFoundError(f"Active session {session_id} not found")

        await self._db.commit()

        # Fetch session with topic relationship eager loaded
        select_stmt = (
            select(PracticeSessionModel)
            .options(selectinload(PracticeSessionModel.topic))
            .where(PracticeSessionModel.id == session_id)
        )
        result = await self._db.execute(select_stmt)
        db_session = result.scalar_one()

        # Get topic name from the eager-loaded relationship
        topic_name = db_session.topic.name if db_session.topic else None

        return self._build_session_response(
            db_session,
            topic_full_code=db_session.topic_full_code,
            topic_name=topic_name,
            subject_full_code=f"{db_session.topic_framework_code}.{db_session.topic_subject_code}" if db_session.topic_framework_code and db_session.topic_subject_code else None,
        )

    async def resume_session(
        self,
        session_id: UUID,
        student_id: UUID,
    ) -> tuple[PracticeSessionResponse, QuestionResponse | None]:
        """Resume a paused session.

        Args:
            session_id: The session ID.
            student_id: The student ID.

        Returns:
            Tuple of (session response, current question).
        """
        # Update session status
        update_stmt = (
            update(PracticeSessionModel)
            .where(
                PracticeSessionModel.id == session_id,
                PracticeSessionModel.student_id == student_id,
                PracticeSessionModel.status == SessionStatus.PAUSED.value,
            )
            .values(status=SessionStatus.ACTIVE.value)
        )
        result = await self._db.execute(update_stmt)

        if result.rowcount == 0:
            raise SessionNotFoundError(f"Paused session {session_id} not found")

        await self._db.commit()

        # Fetch session with topic relationship eager loaded
        select_stmt = (
            select(PracticeSessionModel)
            .options(selectinload(PracticeSessionModel.topic))
            .where(PracticeSessionModel.id == session_id)
        )
        result = await self._db.execute(select_stmt)
        db_session = result.scalar_one()

        # Get topic name from the eager-loaded relationship
        topic_name = db_session.topic.name if db_session.topic else None

        # Get current question from workflow state
        current_question = await self.get_current_question(session_id, student_id)

        return self._build_session_response(
            db_session,
            topic_full_code=db_session.topic_full_code,
            topic_name=topic_name,
            subject_full_code=f"{db_session.topic_framework_code}.{db_session.topic_subject_code}" if db_session.topic_framework_code and db_session.topic_subject_code else None,
        ), current_question

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _get_topic_context(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
    ) -> TopicContext | None:
        """Get educational context for a topic from curriculum hierarchy.

        Delegates to CurriculumLookup for centralized curriculum resolution.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").
            subject_code: Subject code (e.g., "MAT").
            grade_code: Grade code (e.g., "Y4").
            unit_code: Unit code (e.g., "NPV").
            topic_code: Topic code (e.g., "001").

        Returns:
            TopicContext with full curriculum hierarchy, or None if topic not found.
        """
        lookup = CurriculumLookup(self._db)
        return await lookup.get_topic_context(
            framework_code, subject_code, grade_code, unit_code, topic_code
        )

    async def _get_topics_from_subject(
        self,
        framework_code: str,
        subject_code: str,
    ) -> tuple[list[dict], dict | None]:
        """Get all topics from a subject for RANDOM mode practice.

        Returns all topics under the subject along with subject context
        for educational metadata.

        Args:
            framework_code: Framework code (e.g., "UK-NC-2014").
            subject_code: Subject code (e.g., "MAT").

        Returns:
            Tuple of (topics_list, subject_context):
            - topics_list: List of topic dicts with full_code, name, unit_name
            - subject_context: Dict with subject educational context
        """
        # Query subject with its units and topics using composite key
        stmt = (
            select(SubjectModel)
            .options(
                selectinload(SubjectModel.units).selectinload(UnitModel.topics),
                selectinload(SubjectModel.framework),
            )
            .where(
                SubjectModel.framework_code == framework_code,
                SubjectModel.code == subject_code,
            )
        )

        result = await self._db.execute(stmt)
        subject = result.scalar_one_or_none()

        if not subject:
            logger.warning(
                "Subject not found: framework=%s, subject=%s",
                framework_code,
                subject_code,
            )
            return [], None

        # Build topics list with full composite key info
        topics_list = []
        for unit in subject.units or []:
            for topic in unit.topics or []:
                topics_list.append({
                    "full_code": topic.full_code,
                    "framework_code": topic.framework_code,
                    "subject_code": topic.subject_code,
                    "grade_code": topic.grade_code,
                    "unit_code": topic.unit_code,
                    "code": topic.code,
                    "name": topic.name,
                    "unit_name": unit.name,
                    "description": topic.description,
                })

        # Get framework for language/country info
        framework = subject.framework

        # Derive language from country code
        country_code = framework.country_code if framework else None
        language = COUNTRY_TO_LANGUAGE.get(country_code, DEFAULT_LANGUAGE) if country_code else DEFAULT_LANGUAGE

        subject_context = {
            "full_code": subject.full_code,
            "framework_code": framework_code,
            "code": subject_code,
            "name": subject.name,
            "framework_name": framework.name if framework else None,
            "country_code": country_code,
            "language": language,
        }

        logger.debug(
            "Loaded %d topics from subject '%s' for RANDOM mode",
            len(topics_list),
            subject.name,
        )

        return topics_list, subject_context

    async def _get_learning_objective(
        self,
        framework_code: str,
        subject_code: str,
        grade_code: str,
        unit_code: str,
        topic_code: str,
        objective_code: str,
    ) -> LearningObjectiveModel | None:
        """Get learning objective by composite key.

        Args:
            framework_code: Framework code.
            subject_code: Subject code.
            grade_code: Grade code.
            unit_code: Unit code.
            topic_code: Topic code.
            objective_code: Objective code.

        Returns:
            LearningObjectiveModel or None if not found.
        """
        stmt = select(LearningObjectiveModel).where(
            LearningObjectiveModel.framework_code == framework_code,
            LearningObjectiveModel.subject_code == subject_code,
            LearningObjectiveModel.grade_code == grade_code,
            LearningObjectiveModel.unit_code == unit_code,
            LearningObjectiveModel.topic_code == topic_code,
            LearningObjectiveModel.code == objective_code,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    def _build_session_response(
        self,
        db_session: PracticeSessionModel,
        topic_full_code: str | None = None,
        topic_name: str | None = None,
        subject_full_code: str | None = None,
    ) -> PracticeSessionResponse:
        """Build session response from database model.

        Args:
            db_session: Database session model.
            topic_full_code: Full topic code (e.g., "UK-NC-2014.MAT.Y4.NPV.001").
            topic_name: Topic name (optional, for display).
            subject_full_code: Full subject code (e.g., "UK-NC-2014.MAT").

        Returns:
            PracticeSessionResponse with code-based identification.
        """
        return PracticeSessionResponse(
            id=db_session.id,
            student_id=db_session.student_id,
            topic_code=topic_full_code,
            topic_name=topic_name,
            subject_code=subject_full_code,
            session_type=SessionType(db_session.session_type),
            persona_id=db_session.persona_id,
            status=SessionStatus(db_session.status),
            questions_total=db_session.questions_total,
            questions_answered=db_session.questions_answered,
            questions_correct=db_session.questions_correct,
            time_spent_seconds=db_session.time_spent_seconds,
            score=db_session.score,
            started_at=db_session.created_at,
            ended_at=db_session.ended_at,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
        )

    def _state_question_to_response(
        self,
        state_question: dict[str, Any] | Any,
        sequence: int,
        question_id: UUID,
    ) -> QuestionResponse:
        """Convert workflow state question to response DTO.

        Args:
            state_question: Question from workflow state (dict or GeneratedQuestion).
            sequence: Question sequence number in session.
            question_id: The UUID to use for this question (must match DB record).

        Returns:
            QuestionResponse DTO for API response.
        """
        # Helper to get value from dict or object
        def get_val(key: str, default: Any = None) -> Any:
            if hasattr(state_question, key):
                return getattr(state_question, key, default)
            elif isinstance(state_question, dict):
                return state_question.get(key, default)
            return default

        options = None
        raw_options = get_val("options")
        if raw_options:
            options = [
                QuestionOption(
                    key=opt.get("key", chr(97 + i)) if isinstance(opt, dict) else getattr(opt, "key", chr(97 + i)),
                    text=opt.get("text", "") if isinstance(opt, dict) else getattr(opt, "text", ""),
                    is_correct=None,  # Never expose to student
                )
                for i, opt in enumerate(raw_options)
            ]

        return QuestionResponse(
            id=question_id,
            sequence=sequence,
            content=get_val("content", ""),
            question_type=get_val("question_type", "short_answer"),
            options=options,
            difficulty=get_val("difficulty"),
            bloom_level=get_val("bloom_level"),
            topic_name=get_val("topic_name") or get_val("topic"),
            hints_available=len(get_val("hints", []) or []),
            time_limit_seconds=get_val("time_limit_seconds"),
        )

    async def _save_question_to_db(
        self,
        session_id: UUID,
        question: dict[str, Any] | Any,
        sequence: int,
        question_id: UUID,
        topic_codes: dict[str, str | None] | None = None,
        objective_codes: dict[str, str | None] | None = None,
    ) -> None:
        """Save a generated question to the database.

        Args:
            session_id: The practice session ID.
            question: Question from workflow state (dict or GeneratedQuestion).
            sequence: Question sequence number in session.
            question_id: The UUID to use for this question (must match API response).
            topic_codes: Topic composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, code.
            objective_codes: Objective composite key dict with keys:
                framework_code, subject_code, grade_code, unit_code, topic_code, code.
        """
        # Helper to get value from dict or object
        def get_val(key: str, default: Any = None) -> Any:
            if hasattr(question, key):
                return getattr(question, key, default)
            elif isinstance(question, dict):
                return question.get(key, default)
            return default

        # Convert options to serializable format
        raw_options = get_val("options", []) or []
        serialized_options = []
        for opt in raw_options:
            if hasattr(opt, "model_dump"):
                serialized_options.append(opt.model_dump())
            elif hasattr(opt, "dict"):
                serialized_options.append(opt.dict())
            elif isinstance(opt, dict):
                serialized_options.append(opt)
            else:
                serialized_options.append({"text": str(opt)})

        # Convert hints to serializable format
        raw_hints = get_val("hints", []) or []
        serialized_hints = []
        for hint in raw_hints:
            if hasattr(hint, "model_dump"):
                serialized_hints.append(hint.model_dump())
            elif hasattr(hint, "dict"):
                serialized_hints.append(hint.dict())
            elif isinstance(hint, dict):
                serialized_hints.append(hint)
            else:
                serialized_hints.append({"text": str(hint)})

        # Generate evaluation_config from question_type
        question_type_str = get_val("question_type", "short_answer")
        try:
            question_type_enum = QuestionType(question_type_str)
        except ValueError:
            question_type_enum = QuestionType.SHORT_ANSWER
        eval_config = EvaluationConfig.from_question_type(question_type_enum)

        # Extract topic composite key fields
        topic_framework_code = topic_codes.get("framework_code") if topic_codes else None
        topic_subject_code = topic_codes.get("subject_code") if topic_codes else None
        topic_grade_code = topic_codes.get("grade_code") if topic_codes else None
        topic_unit_code = topic_codes.get("unit_code") if topic_codes else None
        topic_code = topic_codes.get("code") if topic_codes else None

        # Extract objective composite key fields
        objective_framework_code = objective_codes.get("framework_code") if objective_codes else None
        objective_subject_code = objective_codes.get("subject_code") if objective_codes else None
        objective_grade_code = objective_codes.get("grade_code") if objective_codes else None
        objective_unit_code = objective_codes.get("unit_code") if objective_codes else None
        objective_topic_code = objective_codes.get("topic_code") if objective_codes else None
        objective_code = objective_codes.get("code") if objective_codes else None

        db_question = PracticeQuestionModel(
            id=question_id,
            session_id=session_id,
            sequence=sequence,
            content=get_val("content", ""),
            display_hint=question_type_str,
            data={"options": serialized_options},
            correct_answer={"value": get_val("correct_answer")},
            explanation=get_val("explanation"),
            hints=serialized_hints,
            difficulty=get_val("difficulty"),
            bloom_level=get_val("bloom_level"),
            # Topic composite key fields
            topic_framework_code=topic_framework_code,
            topic_subject_code=topic_subject_code,
            topic_grade_code=topic_grade_code,
            topic_unit_code=topic_unit_code,
            topic_code=topic_code,
            # Objective composite key fields
            objective_framework_code=objective_framework_code,
            objective_subject_code=objective_subject_code,
            objective_grade_code=objective_grade_code,
            objective_unit_code=objective_unit_code,
            objective_topic_code=objective_topic_code,
            objective_code=objective_code,
            evaluation_config=eval_config.model_dump(),
        )
        self._db.add(db_question)

    def _calculate_score(self, db_session: PracticeSessionModel) -> float | None:
        """Calculate session score."""
        if db_session.questions_answered == 0:
            return None
        return db_session.questions_correct / db_session.questions_answered
