# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning Tutor workflow using LangGraph.

This workflow proactively teaches students new concepts using a hybrid
execution model that combines tool calling and conversational execution.

Workflow Structure:
    initialize
        ↓
    load_context (memory + emotional state)
        ↓
    load_curriculum_content (get_learning_objectives tool)
        ↓
    apply_theory (TheoryOrchestrator recommendations)
        ↓
    generate_opening (first teaching message)
        ↓
    wait_for_message [INTERRUPT POINT]
        ↓
    analyze_message (check for mode transitions, understanding)
        ↓
    generate_response (subject-specific tutor)
        ↓
    update_progress (update metrics, record events)
        ↓
    check_end
        ↓
    [conditional: continue → wait_for_message, end → end_session]

Learning Modes:
    - DISCOVERY: Socratic questioning, student-led exploration
    - EXPLANATION: Clear concept explanation with examples
    - WORKED_EXAMPLE: Step-by-step problem demonstration
    - GUIDED_PRACTICE: Practice with scaffolded support
    - ASSESSMENT: Check understanding with questions

Entry Points:
    - companion_handoff: From Companion when student asks to learn
    - practice_help: "I need to learn this" button in practice
    - direct: Direct access from learning menu
    - lms: External LMS deep link
    - review: Spaced repetition review trigger
    - weakness: Suggested based on mastery gaps
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.agents import AgentFactory
from src.core.memory.manager import MemoryManager
from src.core.personas.manager import PersonaManager
from src.core.tools.base import ToolContext
from src.core.orchestration.states.learning_tutor import (
    LearningTutorState,
    LearningTutorTurn,
    LearningTutorMetrics,
    LearningMode,
    escalate_mode,
    advance_mode,
    get_mode_actions,
)
from src.core.agents.capabilities.comprehension_evaluation import (
    ComprehensionEvaluationCapability,
)
from src.models.comprehension import (
    ComprehensionEvaluationParams,
    ComprehensionTrigger,
)

if TYPE_CHECKING:
    from src.core.educational.orchestrator import TheoryOrchestrator
    from src.core.emotional import EmotionalStateService
    from src.domains.analytics.events import EventTracker
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Maximum conversation turns before auto-ending
MAX_TURNS = 50


class LearningTutorWorkflow:
    """LangGraph workflow for learning tutor.

    This workflow proactively teaches students new concepts using
    subject-specific tutor agents. It uses a hybrid execution model
    that combines tool calling for data retrieval and conversational
    execution for teaching dialogue.

    Agents:
        - learning_tutor_math: Mathematics tutoring
        - learning_tutor_science: Science tutoring
        - learning_tutor_history: History tutoring
        - learning_tutor_geography: Geography tutoring
        - learning_tutor_general: All other subjects

    Features:
        - 5 learning modes with smooth transitions
        - Educational theory integration (TheoryOrchestrator)
        - Memory integration for personalization
        - Tool calling for curriculum data
        - Conversational execution for teaching

    Example:
        >>> workflow = LearningTutorWorkflow(agent_factory, memory_manager, ...)
        >>> initial_state = create_initial_learning_tutor_state(...)
        >>> result = await workflow.run(initial_state, thread_id="session_123")
        >>> # Workflow is now paused at wait_for_message
        >>>
        >>> # To send a message:
        >>> result = await workflow.send_message(thread_id, "I understand", "respond")
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        memory_manager: MemoryManager,
        persona_manager: PersonaManager,
        db_session: "AsyncSession | None" = None,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: "EmotionalStateService | None" = None,
        theory_orchestrator: "TheoryOrchestrator | None" = None,
        event_tracker: "EventTracker | None" = None,
    ):
        """Initialize the learning tutor workflow.

        Args:
            agent_factory: Factory for creating agents.
            memory_manager: Manager for memory operations.
            persona_manager: Manager for tutor personas.
            db_session: Database session for tool execution.
            checkpointer: Checkpointer for state persistence.
            emotional_service: Service for emotional state.
            theory_orchestrator: Orchestrator for educational theories.
            event_tracker: Tracker for publishing analytics events.
        """
        self._agent_factory = agent_factory
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager
        self._db_session = db_session
        self._checkpointer = checkpointer
        self._emotional_service = emotional_service
        self._theory_orchestrator = theory_orchestrator
        self._event_tracker = event_tracker

        # Build the workflow graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Graph Structure:
            initialize → load_context → load_curriculum_content →
            apply_theory → extract_key_concepts → generate_opening →
            wait_for_message [INTERRUPT] → analyze_message →
            [conditional: comprehension_check needed?]
                → ask_comprehension_check → wait_for_comprehension_response [INTERRUPT]
                → evaluate_comprehension → process_evaluation →
            generate_response → update_progress → check_end →
            [continue/end]

        The comprehension check flow is triggered when:
        - Student says "I understand" or similar patterns (SELF_REPORTED)
        - Mode transition is about to happen (MODE_TRANSITION)
        - Session is ending (SESSION_END)
        - Periodic checkpoint every N turns (CHECKPOINT)
        - Explicit understanding check requested (EXPLICIT)

        Returns:
            StateGraph configured for learning tutor.
        """
        graph = StateGraph(LearningTutorState)

        # Add nodes
        graph.add_node("initialize", self._initialize)
        graph.add_node("load_context", self._load_context)
        graph.add_node("load_curriculum_content", self._load_curriculum_content)
        graph.add_node("apply_theory", self._apply_theory)
        graph.add_node("extract_key_concepts", self._extract_key_concepts)
        graph.add_node("generate_opening", self._generate_opening)
        graph.add_node("wait_for_message", self._wait_for_message)
        graph.add_node("analyze_message", self._analyze_message)
        graph.add_node("ask_comprehension_check", self._ask_comprehension_check_question)
        graph.add_node("wait_for_comprehension_response", self._wait_for_comprehension_response)
        graph.add_node("evaluate_comprehension", self._evaluate_comprehension)
        graph.add_node("process_evaluation", self._process_evaluation)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_progress", self._update_progress)
        graph.add_node("check_end", self._check_end)
        graph.add_node("end_session", self._end_session)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges - linear flow through initialization
        graph.add_edge("initialize", "load_context")
        graph.add_edge("load_context", "load_curriculum_content")
        graph.add_edge("load_curriculum_content", "apply_theory")
        graph.add_edge("apply_theory", "extract_key_concepts")
        graph.add_edge("extract_key_concepts", "generate_opening")
        graph.add_edge("generate_opening", "wait_for_message")

        # Main conversation loop
        graph.add_edge("wait_for_message", "analyze_message")

        # Conditional edge from analyze_message: check comprehension or continue
        graph.add_conditional_edges(
            "analyze_message",
            self._should_check_comprehension,
            {
                "check_comprehension": "ask_comprehension_check",
                "continue": "generate_response",
                "end": "end_session",
            },
        )

        # Comprehension check flow
        graph.add_edge("ask_comprehension_check", "wait_for_comprehension_response")
        graph.add_edge("wait_for_comprehension_response", "evaluate_comprehension")
        graph.add_edge("evaluate_comprehension", "process_evaluation")

        # After comprehension evaluation, route based on result
        graph.add_conditional_edges(
            "process_evaluation",
            self._route_after_evaluation,
            {
                "continue_learning": "generate_response",
                "ready_for_practice": "generate_response",
                "end": "end_session",
            },
        )

        # Normal response flow
        graph.add_edge("generate_response", "update_progress")
        graph.add_edge("update_progress", "check_end")

        # Conditional edge from check_end
        graph.add_conditional_edges(
            "check_end",
            self._should_continue,
            {
                "continue": "wait_for_message",
                "end": "end_session",
            },
        )

        graph.add_edge("end_session", END)

        return graph

    def compile(self) -> Any:
        """Compile the workflow graph with interrupt support.

        Uses interrupt_before to pause workflow when waiting for student input:
        - wait_for_message: Normal conversation flow
        - wait_for_comprehension_response: Waiting for explanation during comprehension check

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_message", "wait_for_comprehension_response"],
        )

    async def run(
        self,
        initial_state: LearningTutorState,
        thread_id: str,
    ) -> LearningTutorState:
        """Run the workflow from initial state.

        Executes initialization and pauses at wait_for_message.

        Args:
            initial_state: Starting state for the workflow.
            thread_id: Thread ID for checkpointing.

        Returns:
            Workflow state with first tutor message.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        result = await compiled.ainvoke(initial_state, config=config)
        return result

    async def send_message(
        self,
        thread_id: str,
        message: str | None,
        action: str = "respond",
    ) -> LearningTutorState:
        """Send a message and get response.

        Uses aupdate_state + ainvoke(None) pattern for proper resume.

        Args:
            thread_id: Thread ID for the conversation.
            message: Student message (can be None for actions).
            action: Action type (respond, more_examples, let_me_try, etc.).

        Returns:
            Updated workflow state with tutor response.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow is paused
        state_snapshot = await compiled.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No state found for thread {thread_id}")

        # Update state with message and action
        await compiled.aupdate_state(
            config,
            {
                "_pending_message": message,
                "_pending_action": action,
                "awaiting_input": False,
            },
        )

        logger.info(
            "Resuming learning tutor: thread=%s, action=%s, message=%s...",
            thread_id,
            action,
            message[:50] if message else None,
        )

        # Resume workflow by passing None
        result = await compiled.ainvoke(None, config=config)

        return result

    # =========================================================================
    # Node Implementations
    # =========================================================================

    async def _initialize(self, state: LearningTutorState) -> dict:
        """Initialize the learning tutor session.

        Sets status to active and logs session start.

        Args:
            state: Current workflow state.

        Returns:
            State updates with active status.
        """
        logger.info(
            "Initializing learning tutor: session=%s, topic=%s, mode=%s, agent=%s, entry=%s",
            state.get("session_id"),
            state.get("topic_name"),
            state.get("learning_mode"),
            state.get("agent_id"),
            state.get("entry_point"),
        )

        # Publish session started event (non-blocking)
        asyncio.create_task(self._publish_session_started(state))

        return {
            "status": "active",
            "initial_mode": state.get("learning_mode"),
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: LearningTutorState) -> dict:
        """Load memory context and emotional state.

        Args:
            state: Current workflow state.

        Returns:
            State updates with loaded context.
        """
        logger.info(
            "Loading context: student=%s, topic=%s",
            state.get("student_id"),
            state.get("topic_name"),
        )

        memory_context = {}
        emotional_context = None

        try:
            student_id = state.get("student_id")
            if student_id:
                student_uuid = UUID(student_id) if isinstance(student_id, str) else student_id

                # Load full memory context
                full_context = await self._memory_manager.get_full_context(
                    tenant_code=state["tenant_code"],
                    student_id=student_uuid,
                    topic=state.get("topic_name", ""),
                    session=self._db_session,
                )

                if full_context:
                    memory_context = full_context.model_dump()
                    logger.debug("Loaded memory context for student %s", student_id)

        except Exception as e:
            logger.warning("Failed to load memory context: %s", str(e))

        try:
            # Load current emotional context if service available
            student_id = state.get("student_id")
            if self._emotional_service and student_id:
                emotional_state = await self._emotional_service.get_current_state(
                    student_id=UUID(student_id),
                )
                if emotional_state:
                    emotional_context = emotional_state.to_dict()
                    logger.debug("Loaded emotional context: %s", emotional_state.current_state)

        except Exception as e:
            logger.warning("Failed to load emotional context: %s", str(e))

        # Extract interests from memory context
        interests = state.get("interests", [])
        if memory_context:
            associative = memory_context.get("associative", {})
            if associative and "interests" in associative:
                memory_interests = associative.get("interests", [])
                if memory_interests:
                    interests = [i.get("name", i) if isinstance(i, dict) else i for i in memory_interests]

        return {
            "memory_context": memory_context,
            "emotional_context": emotional_context,
            "interests": interests,
        }

    async def _load_curriculum_content(self, state: LearningTutorState) -> dict:
        """Load curriculum content using get_learning_objectives tool.

        Args:
            state: Current workflow state.

        Returns:
            State updates with learning objectives and topic description.
        """
        logger.info(
            "Loading curriculum content: topic_full_code=%s",
            state.get("topic_full_code"),
        )

        learning_objectives = []
        topic_description = None

        try:
            # Import and instantiate the tool
            from src.tools.learning.get_learning_objectives import GetLearningObjectivesTool

            tool = GetLearningObjectivesTool()

            # Build tool context
            student_id = state.get("student_id", "")
            context = ToolContext(
                tenant_code=state["tenant_code"],
                user_id=UUID(student_id) if student_id else UUID("00000000-0000-0000-0000-000000000000"),
                user_type="student",
                grade_level=int(state.get("grade_level", 5)) if state.get("grade_level") else 5,
                language=state.get("language", "en"),
                session=self._db_session,
            )

            # Execute tool (uses topic_full_code from Central Curriculum)
            result = await tool.execute(
                {"topic_full_code": state.get("topic_full_code")},
                context,
            )

            if result.success:
                learning_objectives = result.data.get("learning_objectives", [])
                topic_description = result.data.get("topic_description")
                logger.info(
                    "Loaded %d learning objectives for topic",
                    len(learning_objectives),
                )
            else:
                logger.warning("Failed to load learning objectives: %s", result.error)

        except Exception as e:
            logger.exception("Failed to load curriculum content: %s", str(e))

        return {
            "learning_objectives": learning_objectives,
            "topic_description": topic_description,
        }

    async def _apply_theory(self, state: LearningTutorState) -> dict:
        """Apply educational theory recommendations.

        Uses TheoryOrchestrator to get recommendations based on
        student context and learning context.

        Args:
            state: Current workflow state.

        Returns:
            State updates with theory recommendations.
        """
        logger.info("Applying educational theory")

        theory_recommendation = state.get("theory_recommendation")

        if self._theory_orchestrator and not theory_recommendation:
            try:
                recommendation = await self._theory_orchestrator.get_recommendations(
                    tenant_code=state.get("tenant_code", ""),
                    student_id=state.get("student_id", ""),
                    topic=state.get("topic_name", ""),
                    memory_context=state.get("memory_context"),
                    emotional_context=None,  # Could be enhanced to pass emotional state
                )

                if recommendation:
                    theory_recommendation = {
                        "difficulty": recommendation.difficulty,
                        "bloom_level": recommendation.bloom_level.value if recommendation.bloom_level else None,
                        "content_format": recommendation.content_format.value if recommendation.content_format else None,
                        "scaffold_level": recommendation.scaffold_level.value if recommendation.scaffold_level else None,
                        "guide_ratio": recommendation.guide_vs_tell_ratio,
                        "questioning_style": recommendation.questioning_style.value if recommendation.questioning_style else None,
                    }
                    logger.debug("Theory recommendation: %s", theory_recommendation)

            except Exception as e:
                logger.warning("Failed to get theory recommendations: %s", str(e))

        return {
            "theory_recommendation": theory_recommendation,
        }

    async def _extract_key_concepts(self, state: LearningTutorState) -> dict:
        """Extract key concepts for the topic.

        Key concepts are used for comprehension evaluation. They come from:
        1. Curriculum metadata (preferred)
        2. Learning objectives extraction
        3. AI-based extraction (fallback)

        Args:
            state: Current workflow state.

        Returns:
            State updates with key concepts.
        """
        logger.info(
            "Extracting key concepts: topic=%s",
            state.get("topic_name"),
        )

        key_concepts: list[str] = []

        # Try to get from curriculum metadata
        learning_objectives = state.get("learning_objectives", [])
        topic_description = state.get("topic_description", "")

        # Extract from learning objectives
        if learning_objectives:
            for obj in learning_objectives:
                if isinstance(obj, dict):
                    # Try to get concept from objective
                    description = obj.get("description", obj.get("text", ""))
                    if description:
                        # Extract the core concept from the objective
                        key_concepts.append(description)
                elif isinstance(obj, str):
                    key_concepts.append(obj)

            if key_concepts:
                # Limit to top 5 concepts
                key_concepts = key_concepts[:5]
                logger.info(
                    "Extracted %d key concepts from learning objectives",
                    len(key_concepts),
                )
                return {"key_concepts": key_concepts}

        # Fallback: Use AI to extract key concepts
        if topic_description or state.get("topic_name"):
            try:
                key_concepts = await self._ai_extract_key_concepts(
                    topic_name=state.get("topic_name", ""),
                    topic_description=topic_description,
                    subject=state.get("subject_name", ""),
                )
                if key_concepts:
                    logger.info(
                        "AI extracted %d key concepts",
                        len(key_concepts),
                    )
                    return {"key_concepts": key_concepts}
            except Exception as e:
                logger.warning("Failed to extract key concepts via AI: %s", str(e))

        # Default fallback: Use topic name as single concept
        topic_name = state.get("topic_name", "the topic")
        key_concepts = [f"Understanding of {topic_name}"]
        logger.info("Using fallback key concept: %s", key_concepts[0])

        return {"key_concepts": key_concepts}

    async def _ai_extract_key_concepts(
        self,
        topic_name: str,
        topic_description: str,
        subject: str,
    ) -> list[str]:
        """Use AI to extract key concepts from topic description.

        Args:
            topic_name: Name of the topic.
            topic_description: Description of the topic.
            subject: Subject area.

        Returns:
            List of key concepts.
        """
        try:
            # Use a simple agent to extract concepts
            agent_id = "learning_tutor_general"
            agent = self._agent_factory.get(agent_id)

            prompt = f"""Extract 3-5 key concepts that a student should understand about this topic.

Topic: {topic_name}
Subject: {subject}
Description: {topic_description}

List only the key concepts, one per line, without numbering or bullets.
Focus on the most important ideas a student needs to grasp."""

            messages = [{"role": "user", "content": prompt}]
            response = await agent.execute_conversational(
                messages=messages,
                runtime_context={"topic_name": topic_name, "subject": subject},
            )

            if response.get("success") and response.get("content"):
                content = response["content"]
                # Parse response into list
                concepts = [
                    line.strip().lstrip("•-123456789. ")
                    for line in content.split("\n")
                    if line.strip() and len(line.strip()) > 5
                ]
                return concepts[:5]

        except Exception as e:
            logger.warning("AI key concept extraction failed: %s", str(e))

        return []

    async def _generate_opening(self, state: LearningTutorState) -> dict:
        """Generate the first teaching message.

        Uses the subject-specific agent with execute_conversational.

        Args:
            state: Current workflow state.

        Returns:
            State updates with first message and conversation history.
        """
        logger.info(
            "Generating opening: mode=%s, agent=%s",
            state.get("learning_mode"),
            state.get("agent_id"),
        )

        try:
            # Get the subject-specific agent
            agent_id = state.get("agent_id", "learning_tutor_general")
            agent = self._agent_factory.get(agent_id)

            # Build runtime context
            runtime_context = self._build_runtime_context(state)

            # Build initial message - student entering the learning session
            entry_messages = {
                "companion_handoff": "I want to learn about this topic.",
                "practice_help": "I need to learn this before practicing.",
                "direct": "I want to learn about this topic.",
                "lms": "I'm here to learn about this topic.",
                "review": "I need to review this topic.",
                "weakness": "I need to work on this topic.",
            }

            entry_point = state.get("entry_point", "direct")
            initial_message = entry_messages.get(entry_point, "I want to learn about this topic.")

            messages = [
                {"role": "user", "content": initial_message}
            ]

            # Call agent with conversational mode
            response = await agent.execute_conversational(
                messages=messages,
                runtime_context=runtime_context,
            )

            if response.get("success") and response.get("content"):
                message_text = response["content"]

                # Create conversation history
                history = [LearningTutorTurn(
                    role="tutor",
                    content=message_text,
                    timestamp=datetime.now().isoformat(),
                    learning_mode=state.get("learning_mode"),
                )]

                # Initialize metrics
                metrics = LearningTutorMetrics(
                    turn_count=1,
                    mode_transitions=0,
                    practice_attempted=0,
                    practice_correct=0,
                    understanding_progress=0.0,
                )

                logger.info("Generated opening message: %d chars", len(message_text))

                return {
                    "first_message": message_text,
                    "last_tutor_response": message_text,
                    "conversation_history": history,
                    "metrics": metrics,
                    "awaiting_input": True,
                }

        except Exception as e:
            logger.exception("Failed to generate opening: %s", str(e))

        # Default opening if agent fails
        topic = state.get("topic_name", "this topic")
        mode = state.get("learning_mode", "explanation")

        default_messages = {
            LearningMode.DISCOVERY.value: f"Let's explore {topic} together! I have some interesting questions for you.",
            LearningMode.EXPLANATION.value: f"Let's learn about {topic}! I'll explain this concept step by step.",
            LearningMode.WORKED_EXAMPLE.value: f"Let me show you how {topic} works with some examples.",
            LearningMode.GUIDED_PRACTICE.value: f"Let's practice {topic} together! I'll guide you through it.",
            LearningMode.ASSESSMENT.value: f"Let's check your understanding of {topic}.",
        }

        default_message = default_messages.get(
            mode,
            f"Let's learn about {topic}!"
        )

        history = [LearningTutorTurn(
            role="tutor",
            content=default_message,
            timestamp=datetime.now().isoformat(),
            learning_mode=mode,
        )]

        return {
            "first_message": default_message,
            "last_tutor_response": default_message,
            "conversation_history": history,
            "metrics": LearningTutorMetrics(
                turn_count=1,
                mode_transitions=0,
                practice_attempted=0,
                practice_correct=0,
                understanding_progress=0.0,
            ),
            "awaiting_input": True,
        }

    async def _wait_for_message(self, state: LearningTutorState) -> dict:
        """Wait for student message (interrupt point).

        Args:
            state: Current workflow state.

        Returns:
            State updates extracting pending message.
        """
        pending_message = state.get("_pending_message")
        pending_action = state.get("_pending_action", "respond")

        logger.info(
            "wait_for_message: pending_message=%s, action=%s",
            pending_message[:50] if pending_message else None,
            pending_action,
        )

        if pending_message or pending_action:
            # Add student message to conversation history
            history = list(state.get("conversation_history", []))

            if pending_message:
                history.append(LearningTutorTurn(
                    role="student",
                    content=pending_message,
                    timestamp=datetime.now().isoformat(),
                    action=pending_action,
                ))

            return {
                "last_student_message": pending_message,
                "last_action": pending_action,
                "_pending_message": None,
                "_pending_action": None,
                "conversation_history": history,
                "awaiting_input": False,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "awaiting_input": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _analyze_message(self, state: LearningTutorState) -> dict:
        """Analyze student message for mode transitions and comprehension triggers.

        This method has been refactored to:
        1. Remove pattern-based understanding increment (now done via AI evaluation)
        2. Detect comprehension check triggers
        3. Set flags for comprehension evaluation flow
        4. Store intended mode for post-verification transitions

        Comprehension check triggers:
        - SELF_REPORTED: Student says "I understand" or similar
        - MODE_TRANSITION: Before advancing to practice/assessment
        - CHECKPOINT: Every 5 turns of meaningful interaction
        - EXPLICIT: "Check my understanding" request

        Args:
            state: Current workflow state.

        Returns:
            State updates with analysis results and trigger flags.
        """
        pending_action = state.get("last_action") or "respond"
        message = state.get("last_student_message", "")
        current_mode = state.get("learning_mode", LearningMode.EXPLANATION.value)

        logger.debug(
            "Analyzing message: action=%s, mode=%s",
            pending_action,
            current_mode,
        )

        updates: dict[str, Any] = {}
        metrics = dict(state.get("metrics", {}))

        # Handle explicit actions
        mode_changed = False
        new_mode = current_mode
        comprehension_trigger: ComprehensionTrigger | None = None
        intended_mode_after: str | None = None

        # Actions that escalate (more support) - no comprehension check needed
        if pending_action in ("show_me", "simpler", "show_solution"):
            new_mode = escalate_mode(current_mode)
            if new_mode != current_mode:
                mode_changed = True
                logger.info("Mode escalated: %s -> %s", current_mode, new_mode)

        # Actions that advance (toward practice/assessment) - REQUIRE comprehension check
        elif pending_action in ("let_me_try", "i_understand"):
            # Don't advance mode yet - trigger comprehension check first
            intended_mode_after = advance_mode(current_mode)
            if intended_mode_after != current_mode:
                comprehension_trigger = ComprehensionTrigger.MODE_TRANSITION
                logger.info(
                    "Mode transition requested: %s -> %s (pending verification)",
                    current_mode,
                    intended_mode_after,
                )

        # Explicit check request
        elif pending_action == "check_understanding":
            comprehension_trigger = ComprehensionTrigger.EXPLICIT
            logger.info("Explicit comprehension check requested")

        # End actions
        elif pending_action == "end":
            # Don't trigger comprehension check for end action
            updates["understood"] = False

        # Analyze message content for comprehension triggers
        if message and not comprehension_trigger:
            message_lower = message.lower()

            # Self-reported understanding patterns - trigger comprehension check
            understanding_patterns = [
                "i get it", "i understand", "makes sense",
                "oh i see", "got it", "anladim", "tamam",
                "that's clear", "now i understand", "i think i got it",
            ]

            for pattern in understanding_patterns:
                if pattern in message_lower:
                    # Don't increment progress - trigger comprehension check instead
                    comprehension_trigger = ComprehensionTrigger.SELF_REPORTED
                    logger.info("Self-reported understanding detected, triggering check")
                    break

            # Confusion signals - escalate mode
            if not comprehension_trigger:
                confusion_patterns = [
                    "i don't understand", "i dont understand",
                    "confused", "not making sense", "lost",
                    "anlamiyorum", "anlamadim",
                ]

                for pattern in confusion_patterns:
                    if pattern in message_lower and not mode_changed:
                        new_mode = escalate_mode(current_mode)
                        if new_mode != current_mode:
                            mode_changed = True
                            logger.info(
                                "Mode escalated due to confusion: %s -> %s",
                                current_mode,
                                new_mode,
                            )
                        break

        # Check for periodic checkpoint (every 5 turns)
        turn_count = metrics.get("turn_count", 0) + 1
        last_check_turn = state.get("_last_comprehension_check_turn", 0)
        if (
            not comprehension_trigger
            and turn_count > 0
            and turn_count - last_check_turn >= 5
            and not state.get("understanding_verified", False)
        ):
            comprehension_trigger = ComprehensionTrigger.CHECKPOINT
            logger.info("Checkpoint comprehension check triggered at turn %d", turn_count)

        # Update mode if changed (not waiting for verification)
        if mode_changed and not comprehension_trigger:
            updates["learning_mode"] = new_mode
            updates["mode_transition_count"] = state.get("mode_transition_count", 0) + 1
            metrics["mode_transitions"] = metrics.get("mode_transitions", 0) + 1
            asyncio.create_task(self._publish_mode_changed(state, current_mode, new_mode))

        # Set comprehension check flags
        if comprehension_trigger:
            updates["_comprehension_check_pending"] = True
            updates["_comprehension_trigger"] = comprehension_trigger.value
            if intended_mode_after:
                updates["_intended_mode_after_verification"] = intended_mode_after

            # Store the last AI explanation for parroting detection
            last_tutor_response = state.get("last_tutor_response", "")
            if last_tutor_response:
                updates["_ai_explanation_for_comparison"] = last_tutor_response

            # Publish comprehension requested event
            asyncio.create_task(self._publish_comprehension_requested(state, comprehension_trigger))

        # Update turn count
        metrics["turn_count"] = turn_count
        updates["metrics"] = metrics

        return updates

    def _should_check_comprehension(
        self,
        state: LearningTutorState,
    ) -> Literal["check_comprehension", "continue", "end"]:
        """Determine if comprehension check should be performed.

        This is a conditional edge function that routes the workflow
        to either the comprehension check flow or normal response generation.

        Args:
            state: Current workflow state.

        Returns:
            "check_comprehension", "continue", or "end".
        """
        # Check for end action first
        if state.get("last_action") == "end":
            return "end"

        # Check if comprehension check is pending
        if state.get("_comprehension_check_pending", False):
            # Skip if we're already awaiting comprehension response
            if state.get("_awaiting_comprehension_response", False):
                return "continue"
            logger.info("Routing to comprehension check")
            return "check_comprehension"

        return "continue"

    async def _ask_comprehension_check_question(self, state: LearningTutorState) -> dict:
        """Ask the student to explain their understanding.

        This node generates a comprehension check question that asks the
        student to explain the concept in their own words. The question
        is tailored to the subject and topic.

        Args:
            state: Current workflow state.

        Returns:
            State updates with comprehension check question.
        """
        logger.info(
            "Asking comprehension check question: trigger=%s",
            state.get("_comprehension_trigger"),
        )

        try:
            # Get the comprehension evaluation capability for question templates
            capability = ComprehensionEvaluationCapability()

            # Get appropriate question for subject
            topic_name = state.get("topic_name", "this topic")
            subject = state.get("subject_name", "general")
            question = capability.get_comprehension_check_question(topic_name, subject)

            # Personalize based on trigger
            trigger = state.get("_comprehension_trigger", "self_reported")

            trigger_intros = {
                "self_reported": "That's great to hear! To make sure you've really got it, ",
                "mode_transition": "Before we move on to practice, let me check your understanding. ",
                "checkpoint": "Let's pause and check how well you're understanding. ",
                "explicit": "Sure, let me check your understanding. ",
                "session_end": "Before we wrap up, I'd like to see how much you've learned. ",
            }

            intro = trigger_intros.get(trigger, "Let me check your understanding. ")
            full_question = f"{intro}{question}"

            # Add to conversation history
            history = list(state.get("conversation_history", []))
            history.append(LearningTutorTurn(
                role="tutor",
                content=full_question,
                timestamp=datetime.now().isoformat(),
                learning_mode=state.get("learning_mode"),
            ))

            # Update metrics
            metrics = dict(state.get("metrics", {}))
            metrics["comprehension_checks"] = metrics.get("comprehension_checks", 0) + 1

            logger.info("Generated comprehension check question")

            return {
                "last_tutor_response": full_question,
                "conversation_history": history,
                "_awaiting_comprehension_response": True,
                "awaiting_input": True,
                "metrics": metrics,
            }

        except Exception as e:
            logger.exception("Failed to generate comprehension check question: %s", str(e))

            # Fallback question
            topic = state.get("topic_name", "this topic")
            fallback = f"Can you explain {topic} in your own words?"

            history = list(state.get("conversation_history", []))
            history.append(LearningTutorTurn(
                role="tutor",
                content=fallback,
                timestamp=datetime.now().isoformat(),
                learning_mode=state.get("learning_mode"),
            ))

            return {
                "last_tutor_response": fallback,
                "conversation_history": history,
                "_awaiting_comprehension_response": True,
                "awaiting_input": True,
            }

    async def _wait_for_comprehension_response(self, state: LearningTutorState) -> dict:
        """Wait for student's comprehension check response (interrupt point).

        This is similar to wait_for_message but specifically for the
        comprehension check flow.

        Args:
            state: Current workflow state.

        Returns:
            State updates extracting the student's explanation.
        """
        pending_message = state.get("_pending_message")
        pending_action = state.get("_pending_action", "respond")

        logger.info(
            "wait_for_comprehension_response: message=%s...",
            pending_message[:50] if pending_message else None,
        )

        if pending_message:
            # Add student explanation to conversation history
            history = list(state.get("conversation_history", []))
            history.append(LearningTutorTurn(
                role="student",
                content=pending_message,
                timestamp=datetime.now().isoformat(),
                action="comprehension_response",
            ))

            return {
                "last_student_message": pending_message,
                "last_action": pending_action,
                "_pending_message": None,
                "_pending_action": None,
                "conversation_history": history,
                "awaiting_input": False,
                "_awaiting_comprehension_response": False,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "awaiting_input": True,
            "_awaiting_comprehension_response": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _evaluate_comprehension(self, state: LearningTutorState) -> dict:
        """Evaluate student's comprehension using AI.

        Uses the ComprehensionEvaluationCapability to assess the student's
        explanation against key concepts, detecting parroting and misconceptions.

        Args:
            state: Current workflow state.

        Returns:
            State updates with evaluation results.
        """
        student_explanation = state.get("last_student_message") or ""
        key_concepts = state.get("key_concepts") or []

        logger.info(
            "Evaluating comprehension: explanation_length=%d, concepts=%d",
            len(student_explanation),
            len(key_concepts),
        )

        try:
            # Build evaluation parameters
            params = ComprehensionEvaluationParams(
                student_explanation=student_explanation,
                key_concepts=key_concepts,
                topic_name=state.get("topic_name", ""),
                topic_description=state.get("topic_description", ""),
                learning_objectives=[
                    obj.get("description", obj) if isinstance(obj, dict) else obj
                    for obj in state.get("learning_objectives", [])
                ],
                ai_explanation=state.get("_ai_explanation_for_comparison", ""),
                conversation_context=[
                    {"role": turn.get("role", ""), "content": turn.get("content", "")}
                    for turn in state.get("conversation_history", [])[-5:]
                ],
                trigger=ComprehensionTrigger(state.get("_comprehension_trigger", "self_reported")),
                subject=state.get("subject_name", ""),
                grade_level=state.get("grade_level"),
                language=state.get("language", "en"),
                current_mastery=state.get("topic_mastery", 0.0),
                session_turn_count=state.get("metrics", {}).get("turn_count", 0),
            )

            # Get the comprehension evaluator agent
            capability = ComprehensionEvaluationCapability()

            # Build context for capability
            from src.core.agents.capabilities.base import CapabilityContext

            memory_context = state.get("memory_context", {})
            context = CapabilityContext(
                memory=memory_context,
                theory=state.get("theory_recommendation"),
                rag_results=None,
                persona=None,
            )

            # Build prompt
            prompt_messages = capability.build_prompt(params.model_dump(), context)

            # Get agent and execute
            agent = self._agent_factory.get("comprehension_evaluator")
            response = await agent.execute_conversational(
                messages=prompt_messages,
                runtime_context=self._build_runtime_context(state),
            )

            if response.get("success") and response.get("content"):
                # Parse the response
                result = capability.parse_response(response["content"])

                # Store evaluation result
                evaluation_dict = result.model_dump()

                # Add to all evaluations list
                all_evaluations = list(state.get("_all_comprehension_evaluations", []))
                all_evaluations.append(evaluation_dict)

                # Update metrics
                metrics = dict(state.get("metrics", {}))
                metrics["turn_count"] = metrics.get("turn_count", 0) + 1
                if result.verified:
                    metrics["comprehension_verified"] = metrics.get("comprehension_verified", 0) + 1

                logger.info(
                    "Comprehension evaluation: score=%.2f, verified=%s, parroting=%s",
                    result.understanding_score,
                    result.verified,
                    result.parroting_detected,
                )

                # Publish evaluation event
                asyncio.create_task(self._publish_comprehension_evaluated(state, result))

                # If misconceptions detected, publish event
                if result.misconceptions:
                    for misconception in result.misconceptions:
                        asyncio.create_task(
                            self._publish_misconception_detected(state, misconception)
                        )

                return {
                    "_comprehension_evaluation": evaluation_dict,
                    "last_comprehension_evaluation": evaluation_dict,
                    "_all_comprehension_evaluations": all_evaluations,
                    "_last_comprehension_check_turn": metrics.get("turn_count", 0),
                    "_comprehension_check_pending": False,
                    "metrics": metrics,
                }

        except Exception as e:
            logger.exception("Failed to evaluate comprehension: %s", str(e))

        # Fallback: Basic evaluation
        return {
            "_comprehension_evaluation": {
                "understanding_score": 0.5,
                "verified": False,
                "error": "Evaluation failed",
            },
            "_comprehension_check_pending": False,
        }

    async def _process_evaluation(self, state: LearningTutorState) -> dict:
        """Process the comprehension evaluation results.

        Updates understanding progress, handles mode transitions,
        and prepares response based on evaluation.

        Args:
            state: Current workflow state.

        Returns:
            State updates based on evaluation.
        """
        evaluation = state.get("_comprehension_evaluation", {})

        logger.info(
            "Processing evaluation: score=%.2f, verified=%s",
            evaluation.get("understanding_score", 0.0),
            evaluation.get("verified", False),
        )

        updates: dict[str, Any] = {}

        understanding_score = evaluation.get("understanding_score", 0.0)
        verified = evaluation.get("verified", False)
        misconceptions = evaluation.get("misconceptions", [])
        concepts_to_clarify = evaluation.get("clarification_needed", [])

        # Update understanding progress
        updates["understanding_progress"] = understanding_score

        # Handle verified understanding
        if verified:
            updates["understanding_verified"] = True

            # Publish understanding verified event
            asyncio.create_task(self._publish_understanding_verified(state, evaluation))

            # Check if mode transition was intended
            intended_mode = state.get("_intended_mode_after_verification")
            if intended_mode:
                current_mode = state.get("learning_mode")
                updates["learning_mode"] = intended_mode
                updates["mode_transition_count"] = state.get("mode_transition_count", 0) + 1
                updates["_intended_mode_after_verification"] = None

                # Update metrics
                metrics = dict(state.get("metrics", {}))
                metrics["mode_transitions"] = metrics.get("mode_transitions", 0) + 1
                updates["metrics"] = metrics

                # Publish mode changed event
                asyncio.create_task(self._publish_mode_changed(state, current_mode, intended_mode))

                logger.info("Mode transitioned after verification: %s -> %s", current_mode, intended_mode)

        else:
            # Not verified - store items needing attention
            if misconceptions:
                updates["_misconceptions_to_address"] = misconceptions
            if concepts_to_clarify:
                updates["_concepts_to_clarify"] = concepts_to_clarify

            # Clear intended mode - not ready yet
            updates["_intended_mode_after_verification"] = None

        # Clear evaluation trigger
        updates["_comprehension_trigger"] = None

        return updates

    def _route_after_evaluation(
        self,
        state: LearningTutorState,
    ) -> Literal["continue_learning", "ready_for_practice", "end"]:
        """Route workflow after comprehension evaluation.

        Args:
            state: Current workflow state.

        Returns:
            "continue_learning", "ready_for_practice", or "end".
        """
        evaluation = state.get("_comprehension_evaluation", {})

        if evaluation.get("ready_for_assessment", False):
            # High comprehension - could end or continue
            if state.get("understanding_progress", 0.0) >= 0.95:
                return "end"

        if evaluation.get("ready_for_practice", False):
            return "ready_for_practice"

        return "continue_learning"

    async def _generate_response(self, state: LearningTutorState) -> dict:
        """Generate tutor response.

        Args:
            state: Current workflow state.

        Returns:
            State updates with tutor response.
        """
        pending_action = state.get("last_action") or "respond"
        student_message = state.get("last_student_message", "")

        # Don't generate response for end action
        if pending_action == "end":
            return {}

        logger.info(
            "Generating response: mode=%s, action=%s",
            state.get("learning_mode"),
            pending_action,
        )

        try:
            # Get the subject-specific agent
            agent_id = state.get("agent_id", "learning_tutor_general")
            agent = self._agent_factory.get(agent_id)

            # Build runtime context
            runtime_context = self._build_runtime_context(state)

            # Build messages from conversation history
            messages = []
            for turn in state.get("conversation_history", [])[-10:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role and content:
                    llm_role = "assistant" if role == "tutor" else "user"
                    messages.append({"role": llm_role, "content": content})

            # Add current student message
            if student_message:
                messages.append({"role": "user", "content": student_message})

            # Call agent with conversational mode
            response = await agent.execute_conversational(
                messages=messages,
                runtime_context=runtime_context,
            )

            if response.get("success") and response.get("content"):
                message_text = response["content"]

                # Add to conversation history
                history = list(state.get("conversation_history", []))
                history.append(LearningTutorTurn(
                    role="tutor",
                    content=message_text,
                    timestamp=datetime.now().isoformat(),
                    learning_mode=state.get("learning_mode"),
                ))

                logger.info("Generated response: %d chars", len(message_text))

                return {
                    "last_tutor_response": message_text,
                    "conversation_history": history,
                    "awaiting_input": True,
                    "error": None,
                }

        except Exception as e:
            logger.exception("Failed to generate response: %s", str(e))

        # Default response
        mode = state.get("learning_mode", LearningMode.EXPLANATION.value)
        topic = state.get("topic_name", "this topic")

        default_responses = {
            LearningMode.DISCOVERY.value: f"That's interesting! Let me ask you another question about {topic}.",
            LearningMode.EXPLANATION.value: f"Let me explain that part of {topic} in more detail.",
            LearningMode.WORKED_EXAMPLE.value: f"Let me show you another example of {topic}.",
            LearningMode.GUIDED_PRACTICE.value: f"Good effort! Let's try another practice problem.",
            LearningMode.ASSESSMENT.value: f"Let me check your understanding with another question.",
        }

        default_response = default_responses.get(mode, "Let me help you understand this better.")

        history = list(state.get("conversation_history", []))
        history.append(LearningTutorTurn(
            role="tutor",
            content=default_response,
            timestamp=datetime.now().isoformat(),
            learning_mode=mode,
        ))

        return {
            "last_tutor_response": default_response,
            "conversation_history": history,
            "awaiting_input": True,
        }

    async def _update_progress(self, state: LearningTutorState) -> dict:
        """Update progress metrics and record events.

        Args:
            state: Current workflow state.

        Returns:
            State updates with progress.
        """
        # Record learning event if significant
        try:
            student_id = state.get("student_id")
            if student_id and self._memory_manager:
                turn_count = state.get("metrics", {}).get("turn_count", 0)

                # Record event every 5 turns or on mode transition
                if turn_count % 5 == 0 or state.get("mode_transition_count", 0) > 0:
                    await self._memory_manager.record_learning_event(
                        tenant_code=state["tenant_code"],
                        student_id=student_id,
                        event_type="learning_session_progress",
                        topic=state.get("topic_name", ""),
                        data={
                            "session_id": state["session_id"],
                            "turn_count": turn_count,
                            "mode": state.get("learning_mode"),
                            "understanding_progress": state.get("understanding_progress", 0.0),
                        },
                        importance=0.3,
                    )

        except Exception as e:
            logger.warning("Failed to record progress event: %s", str(e))

        return {}

    async def _check_end(self, state: LearningTutorState) -> dict:
        """Check if session should end and update metrics.

        Args:
            state: Current workflow state.

        Returns:
            State updates with duration calculated.
        """
        # Calculate total duration
        started = state.get("started_at", "")
        if started:
            try:
                start_time = datetime.fromisoformat(started)
                duration = (datetime.now() - start_time).total_seconds()

                metrics = dict(state.get("metrics", {}))
                metrics["total_duration_seconds"] = duration

                return {"metrics": metrics}
            except Exception:
                pass

        return {}

    def _should_continue(self, state: LearningTutorState) -> Literal["continue", "end"]:
        """Determine if workflow should continue or end.

        Args:
            state: Current workflow state.

        Returns:
            "continue" or "end".
        """
        # Check for error
        if state.get("status") == "error":
            return "end"

        # Check if student understood
        if state.get("understood") is True:
            return "end"

        # Check for end action
        last_action = state.get("last_action")
        if last_action == "end":
            return "end"

        # Check max turns
        turn_count = state.get("metrics", {}).get("turn_count", 0)
        if turn_count >= MAX_TURNS:
            logger.info("Max turns reached: %d", turn_count)
            return "end"

        # Check understanding progress (auto-end if very high)
        if state.get("understanding_progress", 0.0) >= 0.95:
            return "end"

        return "continue"

    async def _end_session(self, state: LearningTutorState) -> dict:
        """End the learning tutor session.

        Args:
            state: Current workflow state.

        Returns:
            State updates marking completion.
        """
        # Determine completion reason
        completion_reason = "user_ended"

        if state.get("understood") is True:
            completion_reason = "understood"
        elif state.get("understanding_progress", 0.0) >= 0.95:
            completion_reason = "understood"
        elif state.get("metrics", {}).get("turn_count", 0) >= MAX_TURNS:
            completion_reason = "max_turns"
        elif state.get("last_action") == "end":
            completion_reason = "user_ended"

        logger.info(
            "Learning tutor session ended: session=%s, turns=%d, mode=%s, reason=%s",
            state.get("session_id"),
            state.get("metrics", {}).get("turn_count", 0),
            state.get("learning_mode"),
            completion_reason,
        )

            # Record in memory
        try:
            student_id = state.get("student_id")
            if student_id and self._memory_manager:
                # Record episodic memory - session completion
                await self._memory_manager.record_learning_event(
                    tenant_code=state["tenant_code"],
                    student_id=student_id,
                    event_type="learning_session_completed",
                    topic=state.get("topic_name", ""),
                    data={
                        "session_id": state["session_id"],
                        "topic_full_code": state.get("topic_full_code"),
                        "initial_mode": state.get("initial_mode"),
                        "final_mode": state.get("learning_mode"),
                        "mode_transitions": state.get("mode_transition_count", 0),
                        "turn_count": state.get("metrics", {}).get("turn_count", 0),
                        "understanding_progress": state.get("understanding_progress", 0.0),
                        "understood": completion_reason == "understood",
                        "completion_reason": completion_reason,
                        "entry_point": state.get("entry_point"),
                    },
                    importance=0.7,
                )

                # Update semantic memory - topic mastery based on understanding
                topic_full_code = state.get("topic_full_code")
                understanding_progress = state.get("understanding_progress", 0.0)
                if topic_full_code:
                    await self._memory_manager.record_learning_session_completion(
                        tenant_code=state["tenant_code"],
                        student_id=student_id,
                        topic_full_code=topic_full_code,
                        understanding_progress=understanding_progress,
                        session_completed=True,
                    )

                # Record procedural observation (non-blocking)
                asyncio.create_task(self._record_procedural_observation(state))

        except Exception as e:
            logger.warning("Failed to record session completion: %s", str(e))

        # Build final state for event publishing
        final_state = dict(state)
        final_state["completion_reason"] = completion_reason
        final_state["understood"] = completion_reason == "understood"

        # Publish session completed event (non-blocking)
        asyncio.create_task(self._publish_session_completed(final_state))

        return {
            "status": "completed",
            "completion_reason": completion_reason,
            "understood": completion_reason == "understood",
            "completed_at": datetime.now().isoformat(),
            "awaiting_input": False,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_runtime_context(self, state: LearningTutorState) -> dict[str, Any]:
        """Build runtime context for agent execution.

        Args:
            state: Current workflow state.

        Returns:
            Runtime context dictionary for YAML interpolation.
        """
        # Format interests
        interests = state.get("interests", [])
        interests_str = ", ".join(interests) if interests else "not specified"

        # Get theory recommendation values
        theory = state.get("theory_recommendation") or {}

        # Build context
        context = {
            # Topic context (using Central Curriculum composite keys)
            "topic_name": state.get("topic_name", ""),
            "topic_full_code": state.get("topic_full_code", ""),
            "subject_name": state.get("subject_name", "General"),
            "subject_full_code": state.get("subject_full_code", ""),
            "subject_code": state.get("subject_code", "general"),
            "topic_description": state.get("topic_description", ""),

            # Student context
            "student_age": state.get("student_age", 10),
            "grade_level": state.get("grade_level", 5),
            "language": state.get("language", "en"),
            "topic_mastery": state.get("topic_mastery", 0.5),
            "emotional_state": state.get("emotional_state", "neutral"),
            "interests": interests_str,

            # Learning mode
            "learning_mode": state.get("learning_mode", "explanation"),
            "current_turn": state.get("metrics", {}).get("turn_count", 1),

            # Theory recommendations
            "recommended_difficulty": theory.get("difficulty", 0.5),
            "bloom_level": theory.get("bloom_level", "understand"),
            "scaffold_level": theory.get("scaffold_level", "medium"),
            "guide_ratio": theory.get("guide_ratio", 0.5),

            # Curriculum content
            "has_curriculum": bool(state.get("topic_description")),
            "has_theory": bool(theory),
        }

        return context

    # =========================================================================
    # Event Publishing Methods
    # =========================================================================

    async def _publish_session_started(self, state: LearningTutorState) -> None:
        """Publish learning tutor session started event.

        Args:
            state: Current workflow state.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            # Remove None values
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await tracker.track_event(
                event_type="learning_tutor.session.started",
                student_id=state["student_id"],
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "learning_tutor",
                    "topic_name": state.get("topic_name"),
                    "topic_full_code": state.get("topic_full_code"),
                    "subject_name": state.get("subject_name"),
                    "entry_point": state.get("entry_point"),
                    "initial_mode": state.get("learning_mode"),
                    "agent_id": state.get("agent_id"),
                },
            )
            logger.debug("Published learning_tutor.session.started event")
        except Exception as e:
            logger.warning("Failed to publish session started event: %s", str(e))

    async def _publish_session_completed(self, state: LearningTutorState) -> None:
        """Publish learning tutor session completed event.

        Args:
            state: Current workflow state.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            metrics = state.get("metrics", {})
            await tracker.track_event(
                event_type="learning_tutor.session.completed",
                student_id=state["student_id"],
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "learning_tutor",
                    "topic_name": state.get("topic_name"),
                    "topic_full_code": state.get("topic_full_code"),
                    "initial_mode": state.get("initial_mode"),
                    "final_mode": state.get("learning_mode"),
                    "mode_transitions": state.get("mode_transition_count", 0),
                    "turn_count": metrics.get("turn_count", 0),
                    "understanding_progress": state.get("understanding_progress", 0.0),
                    "understood": state.get("understood", False),
                    "completion_reason": state.get("completion_reason"),
                },
            )
            logger.debug("Published learning_tutor.session.completed event")
        except Exception as e:
            logger.warning("Failed to publish session completed event: %s", str(e))

    async def _publish_understanding_updated(self, state: LearningTutorState) -> None:
        """Publish understanding progress update event.

        Args:
            state: Current workflow state.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            metrics = state.get("metrics", {})
            await tracker.track_event(
                event_type="learning_tutor.understanding.updated",
                student_id=state["student_id"],
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "learning_tutor",
                    "understanding_score": state.get("understanding_progress", 0.0),
                    "turn_count": metrics.get("turn_count", 0),
                    "current_mode": state.get("learning_mode"),
                },
            )
            logger.debug("Published learning_tutor.understanding.updated event")
        except Exception as e:
            logger.warning("Failed to publish understanding updated event: %s", str(e))

    async def _publish_mode_changed(
        self,
        state: LearningTutorState,
        from_mode: str,
        to_mode: str,
    ) -> None:
        """Publish learning mode changed event.

        Args:
            state: Current workflow state.
            from_mode: Previous learning mode.
            to_mode: New learning mode.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            metrics = state.get("metrics", {})
            await tracker.track_event(
                event_type="learning_tutor.mode.changed",
                student_id=state["student_id"],
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "learning_tutor",
                    "from_mode": from_mode,
                    "to_mode": to_mode,
                    "transition_count": state.get("mode_transition_count", 0),
                    "turn_count": metrics.get("turn_count", 0),
                    "understanding_progress": state.get("understanding_progress", 0.0),
                },
            )
            logger.debug("Published learning_tutor.mode.changed event: %s -> %s", from_mode, to_mode)
        except Exception as e:
            logger.warning("Failed to publish mode changed event: %s", str(e))

    async def _publish_comprehension_requested(
        self,
        state: LearningTutorState,
        trigger: ComprehensionTrigger,
    ) -> None:
        """Publish comprehension check requested event.

        Args:
            state: Current workflow state.
            trigger: What triggered the comprehension check.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            metrics = state.get("metrics", {})
            await tracker.track_event(
                event_type="learning_tutor.comprehension.requested",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "learning_tutor",
                    "trigger": trigger.value,
                    "topic_full_code": state.get("topic_full_code"),
                    "turn_number": metrics.get("turn_count", 0),
                    "current_mode": state.get("learning_mode"),
                },
            )
            logger.debug("Published learning_tutor.comprehension.requested event")
        except Exception as e:
            logger.warning("Failed to publish comprehension requested event: %s", str(e))

    async def _publish_comprehension_evaluated(
        self,
        state: LearningTutorState,
        result: Any,
    ) -> None:
        """Publish comprehension evaluation completed event.

        Args:
            state: Current workflow state.
            result: ComprehensionEvaluationResult.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            await tracker.track_event(
                event_type="learning_tutor.comprehension.evaluated",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "learning_tutor",
                    "comprehension_score": result.understanding_score,
                    "understanding_level": result.understanding_level.value,
                    "verified": result.verified,
                    "parroting_detected": result.parroting_detected,
                    "concepts_understood_count": len(result.concepts_understood),
                    "concepts_total": len(state.get("key_concepts", [])),
                    "misconceptions_count": len(result.misconceptions),
                    "ready_for_practice": result.ready_for_practice,
                    "recommended_action": result.recommended_action,
                    "confidence": result.confidence,
                },
            )
            logger.debug("Published learning_tutor.comprehension.evaluated event")
        except Exception as e:
            logger.warning("Failed to publish comprehension evaluated event: %s", str(e))

    async def _publish_understanding_verified(
        self,
        state: LearningTutorState,
        evaluation: dict,
    ) -> None:
        """Publish understanding verified event.

        Args:
            state: Current workflow state.
            evaluation: Evaluation result dictionary.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            previous_progress = state.get("understanding_progress", 0.0)
            new_score = evaluation.get("understanding_score", 0.0)

            await tracker.track_event(
                event_type="learning_tutor.understanding.verified",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "learning_tutor",
                    "previous_score": previous_progress,
                    "new_score": new_score,
                    "verification_method": "comprehension_evaluation",
                    "concepts_verified": evaluation.get("concepts_understood", []),
                    "topic_full_code": state.get("topic_full_code"),
                },
            )
            logger.debug("Published learning_tutor.understanding.verified event")
        except Exception as e:
            logger.warning("Failed to publish understanding verified event: %s", str(e))

    async def _publish_misconception_detected(
        self,
        state: LearningTutorState,
        misconception: Any,
    ) -> None:
        """Publish misconception detected event.

        Args:
            state: Current workflow state.
            misconception: DetectedMisconception or dict.
        """
        try:
            tracker = self._event_tracker
            if tracker is None:
                from src.domains.analytics.events import EventTracker
                tracker = EventTracker(tenant_code=state["tenant_code"])

            # Handle both object and dict
            if hasattr(misconception, "model_dump"):
                misc_data = misconception.model_dump()
            else:
                misc_data = misconception

            await tracker.track_event(
                event_type="learning_tutor.misconception.detected",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "learning_tutor",
                    "misconception_id": misc_data.get("id", "unknown"),
                    "misconception_description": misc_data.get("description", ""),
                    "severity": misc_data.get("severity", "significant"),
                    "related_concept": misc_data.get("related_concept", ""),
                    "topic_full_code": state.get("topic_full_code"),
                },
            )
            logger.debug("Published learning_tutor.misconception.detected event")
        except Exception as e:
            logger.warning("Failed to publish misconception detected event: %s", str(e))

    # =========================================================================
    # Memory Recording Methods
    # =========================================================================

    async def _record_procedural_observation(
        self,
        state: LearningTutorState,
    ) -> None:
        """Record learning pattern observation in procedural memory.

        Tracks behavioral patterns that inform personalization:
        - When the student learns best (time of day)
        - Which learning modes are most effective
        - Understanding progress patterns

        Args:
            state: Current workflow state.
        """
        try:
            student_id = state.get("student_id")
            if not student_id or not self._memory_manager:
                return

            student_uuid = (
                UUID(student_id)
                if isinstance(student_id, str)
                else student_id
            )

            # Determine time of day bucket
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                time_of_day = "morning"
            elif 12 <= current_hour < 17:
                time_of_day = "afternoon"
            elif 17 <= current_hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            metrics = state.get("metrics", {})
            observation = {
                "session_type": "learning_tutor",
                "time_of_day": time_of_day,
                "learning_mode": state.get("learning_mode"),
                "understanding_progress": state.get("understanding_progress", 0.0),
                "turn_count": metrics.get("turn_count", 0),
                "mode_transitions": state.get("mode_transition_count", 0),
                "topic": state.get("topic_name"),
                "subject": state.get("subject_name"),
                "entry_point": state.get("entry_point"),
                "session_id": state["session_id"],
            }

            await self._memory_manager.record_procedural_observation(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                observation=observation,
                topic_full_code=state.get("topic_full_code"),
            )

            logger.debug(
                "Recorded procedural observation: time=%s, mode=%s",
                time_of_day,
                state.get("learning_mode"),
            )

        except Exception as e:
            logger.warning("Failed to record procedural observation: %s", str(e))
