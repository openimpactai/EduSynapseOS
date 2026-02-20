# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Practice Helper Tutor workflow using LangGraph.

This workflow helps students understand concepts when they answer incorrectly
during practice and click "Get Help". It uses subject-specific tutor agents
and adapts the tutoring mode based on the student's needs.

Workflow Structure:
    initialize
        ↓
    load_context (memory + emotional state)
        ↓
    generate_first_message (based on selected mode)
        ↓
    wait_for_message [INTERRUPT POINT]
        ↓
    analyze_message (check for mode escalation, understanding)
        ↓
    generate_response (subject-specific tutor)
        ↓
    check_end
        ↓
    [conditional: continue → wait_for_message, end → end_session]

Tutoring Modes:
    - HINT: Short hints, guide without giving answer
    - GUIDED: Discovery through Q&A, Socratic method
    - STEP_BY_STEP: Show solution step by step

Mode Selection:
    - emotional_state in [frustrated, anxious] → STEP_BY_STEP
    - topic_mastery < 0.5 → GUIDED
    - default → HINT
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
from src.core.orchestration.states.practice_helper import (
    PracticeHelperState,
    PracticeHelperTurn,
    PracticeHelperMetrics,
    TutoringMode,
    escalate_mode,
    create_initial_practice_helper_state,
)

if TYPE_CHECKING:
    from src.core.emotional import EmotionalStateService
    from src.domains.analytics.events import EventTracker

logger = logging.getLogger(__name__)

# Maximum conversation turns before auto-ending
MAX_TURNS = 15


class PracticeHelperWorkflow:
    """LangGraph workflow for practice helper tutoring.

    This workflow helps students understand concepts when they answer
    incorrectly during practice. It uses subject-specific tutor agents
    that adapt their teaching approach based on the subject matter.

    Agents:
        - practice_helper_tutor_math: Mathematics tutoring
        - practice_helper_tutor_science: Science tutoring
        - practice_helper_tutor_history: History tutoring
        - practice_helper_tutor_geography: Geography tutoring
        - practice_helper_tutor_general: All other subjects

    Features:
        - Mode escalation (HINT → GUIDED → STEP_BY_STEP)
        - Step tracking for STEP_BY_STEP mode
        - Understanding detection
        - Memory integration for personalization

    Attributes:
        agent_factory: Factory for creating agents.
        memory_manager: Manager for memory operations.
        persona_manager: Manager for tutor personas.
        checkpointer: Checkpointer for state persistence.

    Example:
        >>> workflow = PracticeHelperWorkflow(agent_factory, memory_manager, ...)
        >>> initial_state = create_initial_practice_helper_state(...)
        >>> result = await workflow.run(initial_state, thread_id="session_123")
        >>> # Workflow is now paused at wait_for_message
        >>>
        >>> # To send a message:
        >>> result = await workflow.send_message(thread_id, "I don't understand", "respond")
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        memory_manager: MemoryManager,
        persona_manager: PersonaManager,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: "EmotionalStateService | None" = None,
        event_tracker: "EventTracker | None" = None,
    ):
        """Initialize the practice helper workflow.

        Args:
            agent_factory: Factory for creating agents.
            memory_manager: Manager for memory operations.
            persona_manager: Manager for tutor personas.
            checkpointer: Checkpointer for state persistence.
            emotional_service: Service for emotional state.
            event_tracker: Tracker for publishing analytics events.
        """
        self._agent_factory = agent_factory
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager
        self._checkpointer = checkpointer
        self._emotional_service = emotional_service
        self._event_tracker = event_tracker

        # Build the workflow graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Graph Structure:
            initialize → load_context → generate_first_message →
            wait_for_message [INTERRUPT] → analyze_message →
            generate_response → check_end → [continue/end]

        Returns:
            StateGraph configured for practice helper tutoring.
        """
        graph = StateGraph(PracticeHelperState)

        # Add nodes
        graph.add_node("initialize", self._initialize)
        graph.add_node("load_context", self._load_context)
        graph.add_node("generate_first_message", self._generate_first_message)
        graph.add_node("wait_for_message", self._wait_for_message)
        graph.add_node("analyze_message", self._analyze_message)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("check_end", self._check_end)
        graph.add_node("escalate_to_learning_tutor", self._escalate_to_learning_tutor)
        graph.add_node("end_session", self._end_session)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges
        graph.add_edge("initialize", "load_context")
        graph.add_edge("load_context", "generate_first_message")
        graph.add_edge("generate_first_message", "wait_for_message")
        graph.add_edge("wait_for_message", "analyze_message")

        # After analyze_message, check if we need to escalate to Learning Tutor
        graph.add_conditional_edges(
            "analyze_message",
            self._route_after_analysis,
            {
                "escalate": "escalate_to_learning_tutor",
                "continue": "generate_response",
            },
        )

        graph.add_edge("generate_response", "check_end")
        graph.add_edge("escalate_to_learning_tutor", END)

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

        Uses interrupt_before=["wait_for_message"] to pause workflow
        when waiting for student input.

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_message"],
        )

    async def run(
        self,
        initial_state: PracticeHelperState,
        thread_id: str,
    ) -> PracticeHelperState:
        """Run the workflow from initial state.

        Executes initialize → load_context → generate_first_message, then
        pauses at wait_for_message (interrupt point).

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
        action: str,
    ) -> PracticeHelperState:
        """Send a message and get response.

        Uses aupdate_state + ainvoke(None) pattern for proper resume.

        Args:
            thread_id: Thread ID for the conversation.
            message: Student message (can be None for actions).
            action: Action type (respond, next_step, show_me, i_understand, end).

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
            "Resuming workflow: thread=%s, action=%s, message=%s...",
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

    async def _initialize(self, state: PracticeHelperState) -> dict:
        """Initialize the practice helper session.

        Sets status to active and logs session start.

        Args:
            state: Current workflow state.

        Returns:
            State updates with active status.
        """
        logger.info(
            "Initializing practice helper: session=%s, subject=%s, mode=%s, agent=%s",
            state.get("session_id"),
            state.get("subject"),
            state.get("tutoring_mode"),
            state.get("agent_id"),
        )

        # Publish session started event (non-blocking)
        asyncio.create_task(self._publish_session_started(state))

        return {
            "status": "active",
            "initial_mode": state.get("tutoring_mode"),
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: PracticeHelperState) -> dict:
        """Load memory context and emotional state.

        Loads student's memory context for personalization.
        The emotional state and topic mastery were already used
        for mode selection during state creation.

        Args:
            state: Current workflow state.

        Returns:
            State updates with loaded context.
        """
        logger.info(
            "Loading context: student=%s, topic=%s",
            state.get("student_context", {}).get("student_id"),
            state.get("topic_name"),
        )

        memory_context = {}
        emotional_context = None

        try:
            student_context = state.get("student_context", {})
            student_id = student_context.get("student_id")

            if student_id:
                student_uuid = UUID(student_id) if isinstance(student_id, str) else student_id

                # Load full memory context
                full_context = await self._memory_manager.get_full_context(
                    tenant_code=state["tenant_code"],
                    student_id=student_uuid,
                    topic=state.get("topic_name", ""),
                )

                if full_context:
                    memory_context = full_context.model_dump()
                    logger.debug("Loaded memory context for student %s", student_id)

        except Exception as e:
            logger.warning("Failed to load memory context: %s", str(e))

        try:
            # Load current emotional context if service available
            student_context = state.get("student_context", {})
            student_id = student_context.get("student_id")

            if self._emotional_service and student_id:
                emotional_state = await self._emotional_service.get_current_state(
                    student_id=UUID(student_id),
                )
                if emotional_state:
                    emotional_context = emotional_state.to_dict()
                    logger.debug("Loaded emotional context: %s", emotional_state.current_state)

        except Exception as e:
            logger.warning("Failed to load emotional context: %s", str(e))

        return {
            "memory_context": memory_context,
            "emotional_context": emotional_context,
        }

    async def _generate_first_message(self, state: PracticeHelperState) -> dict:
        """Generate the first tutor message based on selected mode.

        Uses the subject-specific agent with execute_conversational to generate
        an appropriate first message for helping with the incorrectly answered question.

        Args:
            state: Current workflow state.

        Returns:
            State updates with first message and conversation history.
        """
        logger.info(
            "Generating first message: mode=%s, agent=%s",
            state.get("tutoring_mode"),
            state.get("agent_id"),
        )

        try:
            # Get the subject-specific agent
            agent_id = state.get("agent_id", "practice_helper_tutor_general")
            agent = self._agent_factory.get(agent_id)

            # Get question context
            question_context = state.get("question_context", {})
            student_context = state.get("student_context", {})

            # Format options for display
            options = question_context.get("options")
            if isinstance(options, dict):
                options_str = "\n".join(f"  {k}) {v}" for k, v in options.items())
            elif isinstance(options, list):
                options_str = "\n".join(f"  {chr(97+i)}) {opt}" for i, opt in enumerate(options))
            else:
                options_str = str(options) if options else ""

            # Format interests for display
            interests = student_context.get("interests", [])
            if isinstance(interests, list):
                interests_str = ", ".join(interests) if interests else "not specified"
            else:
                interests_str = str(interests) if interests else "not specified"

            # Build runtime_context for YAML prompt interpolation
            runtime_context = {
                # Tutoring context
                "tutoring_mode": state.get("tutoring_mode", "hint"),
                "current_turn": 1,
                "current_step": 1 if state.get("tutoring_mode") == TutoringMode.STEP_BY_STEP.value else 0,
                # Student information
                "student_age": student_context.get("student_age", 10),
                "grade_level": student_context.get("grade_level", 5),
                "language": student_context.get("language", "en"),
                "topic_mastery": student_context.get("topic_mastery", 0.5),
                "emotional_state": student_context.get("emotional_state", "neutral"),
                "interests": interests_str,
                # Question information
                "topic_name": state.get("topic_name", ""),
                "subject": state.get("subject", "Mathematics"),
                "question_text": question_context.get("question_text", ""),
                "question_type": question_context.get("question_type", "multiple_choice"),
                "options": options_str,
                "correct_answer": question_context.get("correct_answer", ""),
                "student_answer": question_context.get("student_answer", ""),
            }

            # Build initial message - student asking for help
            messages = [
                {
                    "role": "user",
                    "content": "I got this question wrong and I don't understand why. Can you help me?",
                }
            ]

            # Call agent with conversational mode
            response = await agent.execute_conversational(
                messages=messages,
                runtime_context=runtime_context,
            )

            if response.get("success") and response.get("content"):
                message_text = response["content"]

                # Create conversation history
                history = [PracticeHelperTurn(
                    role="tutor",
                    content=message_text,
                    timestamp=datetime.now().isoformat(),
                )]

                # Initialize metrics
                metrics = PracticeHelperMetrics(
                    turn_count=1,
                    current_step=1 if state.get("tutoring_mode") == TutoringMode.STEP_BY_STEP.value else 0,
                    total_steps=None,
                    mode_escalations=0,
                    understanding_progress=0.0,
                )

                logger.info("Generated first message: %d chars", len(message_text))

                return {
                    "last_tutor_response": message_text,
                    "conversation_history": history,
                    "metrics": metrics,
                    "current_step": 1 if state.get("tutoring_mode") == TutoringMode.STEP_BY_STEP.value else 0,
                    "awaiting_input": True,
                }

        except Exception as e:
            logger.exception("Failed to generate first message: %s", str(e))

        # Default message if agent fails
        topic = state.get("topic_name", "this concept")
        mode = state.get("tutoring_mode", "hint")

        if mode == TutoringMode.STEP_BY_STEP.value:
            default_message = f"Let's work through this step by step. First, let's look at what the question is asking about {topic}. Can you tell me what you understand so far?"
        elif mode == TutoringMode.GUIDED.value:
            default_message = f"Let's think about this together! When looking at this {topic} question, what part made you choose your answer?"
        else:
            default_message = f"I see you're working on {topic}. Think about what key concept this question is testing. What comes to mind?"

        history = [PracticeHelperTurn(
            role="tutor",
            content=default_message,
            timestamp=datetime.now().isoformat(),
        )]

        return {
            "last_tutor_response": default_message,
            "conversation_history": history,
            "metrics": PracticeHelperMetrics(
                turn_count=1,
                current_step=1 if mode == TutoringMode.STEP_BY_STEP.value else 0,
                total_steps=None,
                mode_escalations=0,
                understanding_progress=0.0,
            ),
            "current_step": 1 if mode == TutoringMode.STEP_BY_STEP.value else 0,
            "awaiting_input": True,
        }

    async def _wait_for_message(self, state: PracticeHelperState) -> dict:
        """Wait for student message (interrupt point).

        This node is the interrupt point. When workflow reaches here,
        it pauses and waits for send_message() to inject a message.

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

            # Only add to history if there's actual content
            if pending_message:
                history.append(PracticeHelperTurn(
                    role="student",
                    content=pending_message,
                    timestamp=datetime.now().isoformat(),
                    action=pending_action,
                ))

            return {
                "last_student_message": pending_message,
                "last_action": pending_action,  # Save action for _analyze_message
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

    async def _analyze_message(self, state: PracticeHelperState) -> dict:
        """Analyze the student message and action for mode escalation.

        Checks for:
        - Explicit actions (show_me, i_understand, end, next_step)
        - Understanding signals
        - Confusion signals (triggers mode escalation)
        - "I don't know" count

        Args:
            state: Current workflow state.

        Returns:
            State updates with analysis results.
        """
        # Read action from last_action (set by _wait_for_message)
        pending_action = state.get("last_action") or "respond"
        message = state.get("last_student_message", "")
        current_mode = state.get("tutoring_mode", TutoringMode.HINT.value)

        logger.debug("Analyzing message: action=%s, message=%s...", pending_action, message[:50] if message else None)

        updates: dict[str, Any] = {}
        metrics = dict(state.get("metrics", {}))

        # Handle explicit actions
        if pending_action == "show_me":
            # Escalate directly to STEP_BY_STEP
            new_mode = TutoringMode.STEP_BY_STEP.value
            if current_mode != new_mode:
                updates["tutoring_mode"] = new_mode
                updates["mode_escalation_count"] = state.get("mode_escalation_count", 0) + 1
                metrics["mode_escalations"] = metrics.get("mode_escalations", 0) + 1
                updates["current_step"] = 1
                logger.info("Mode escalated to STEP_BY_STEP via show_me action")

                # Publish mode escalated event (non-blocking)
                asyncio.create_task(self._publish_mode_escalated(state, current_mode, new_mode))

        elif pending_action == "next_step":
            # Advance step counter
            current_step = state.get("current_step", 0)
            updates["current_step"] = current_step + 1
            metrics["current_step"] = current_step + 1

        elif pending_action in ("i_understand", "end"):
            # Mark for ending - will be handled in check_end
            if pending_action == "i_understand":
                updates["understood"] = True
                updates["understanding_progress"] = 1.0
                metrics["understanding_progress"] = 1.0

        # Analyze message content for escalation triggers
        if message:
            message_lower = message.lower()

            # Check for "I don't know" patterns
            i_dont_know_patterns = [
                "i don't know", "i dont know", "idk",
                "no idea", "not sure", "don't understand",
                "bilmiyorum", "anlamadim", "anlamıyorum"
            ]

            for pattern in i_dont_know_patterns:
                if pattern in message_lower:
                    new_count = state.get("i_dont_know_count", 0) + 1
                    updates["i_dont_know_count"] = new_count

                    # Escalate mode after 2 "I don't know"s
                    if new_count >= 2:
                        new_mode = escalate_mode(current_mode)
                        if current_mode != new_mode:
                            updates["tutoring_mode"] = new_mode
                            updates["mode_escalation_count"] = state.get("mode_escalation_count", 0) + 1
                            metrics["mode_escalations"] = metrics.get("mode_escalations", 0) + 1
                            updates["i_dont_know_count"] = 0  # Reset counter
                            logger.info("Mode escalated to %s after I don't know count", new_mode)

                            # Publish mode escalated event (non-blocking)
                            asyncio.create_task(self._publish_mode_escalated(state, current_mode, new_mode))
                    break

            # Check for frustration/confusion triggers
            frustration_patterns = [
                "i give up", "this is impossible", "i hate this",
                "so confused", "makes no sense", "vazgectim"
            ]

            for pattern in frustration_patterns:
                if pattern in message_lower:
                    new_mode = TutoringMode.STEP_BY_STEP.value
                    if current_mode != new_mode:
                        updates["tutoring_mode"] = new_mode
                        updates["mode_escalation_count"] = state.get("mode_escalation_count", 0) + 1
                        metrics["mode_escalations"] = metrics.get("mode_escalations", 0) + 1
                        logger.info("Mode escalated to STEP_BY_STEP due to frustration")

                        # Publish mode escalated event (non-blocking)
                        asyncio.create_task(self._publish_mode_escalated(state, current_mode, new_mode))

                    # Track confusion patterns for Learning Tutor escalation
                    confusion_count = state.get("confusion_pattern_count", 0) + 1
                    updates["confusion_pattern_count"] = confusion_count
                    break

            # Check for explicit "teach me" / "explain from beginning" patterns
            full_lesson_patterns = [
                "i need a full lesson", "teach me from scratch",
                "start from the beginning", "i don't understand any of this",
                "can you teach me", "explain everything", "basından anlat"
            ]

            for pattern in full_lesson_patterns:
                if pattern in message_lower:
                    updates["student_wants_full_lesson"] = True
                    logger.info("Student requested full lesson")
                    break

            # Check for understanding signals
            understanding_patterns = [
                "i get it", "i understand", "makes sense",
                "oh i see", "got it", "anladim", "tamam"
            ]

            for pattern in understanding_patterns:
                if pattern in message_lower:
                    # Increase understanding progress
                    current_progress = state.get("understanding_progress", 0.0)
                    new_progress = min(1.0, current_progress + 0.3)
                    updates["understanding_progress"] = new_progress
                    metrics["understanding_progress"] = new_progress
                    logger.debug("Understanding progress: %.2f", new_progress)
                    break

        # Update turn count
        metrics["turn_count"] = metrics.get("turn_count", 0) + 1
        updates["metrics"] = metrics

        # Check if escalation to Learning Tutor is needed
        if self._should_escalate_to_learning_tutor(state, updates):
            updates["_escalate_to_learning_tutor"] = True
            logger.info("Flagging for Learning Tutor escalation")

        return updates

    def _route_after_analysis(
        self,
        state: PracticeHelperState,
    ) -> Literal["escalate", "continue"]:
        """Route after message analysis.

        Checks if we need to escalate to Learning Tutor instead of continuing.

        Args:
            state: Current workflow state.

        Returns:
            "escalate" to go to Learning Tutor, "continue" to generate response.
        """
        if state.get("_escalate_to_learning_tutor"):
            return "escalate"
        return "continue"

    def _should_escalate_to_learning_tutor(
        self,
        state: PracticeHelperState,
        updates: dict[str, Any],
    ) -> bool:
        """Determine if we should escalate to Learning Tutor.

        Escalation triggers:
        - Explicit student request for full lesson
        - 2+ mode escalations (reached STEP_BY_STEP from lower mode)
        - At STEP_BY_STEP mode with 3+ confusion patterns
        - Very low mastery (<0.3) + 2+ confusion patterns

        Note: Mode can only escalate twice max (HINT→GUIDED→STEP_BY_STEP),
        so we use 2+ as the threshold for mode escalations.

        Args:
            state: Current workflow state.
            updates: Pending state updates.

        Returns:
            True if escalation to Learning Tutor is warranted.
        """
        # Check for explicit student request
        if updates.get("student_wants_full_lesson") or state.get("student_wants_full_lesson"):
            logger.info("Escalating: student requested full lesson")
            return True

        # Check mode escalation count (2+ means reached highest support level)
        mode_escalations = updates.get("mode_escalation_count") or state.get("mode_escalation_count", 0)
        if mode_escalations >= 2:
            logger.info("Escalating: 2+ mode escalations indicate fundamental gap")
            return True

        # Check if at STEP_BY_STEP with multiple confusion patterns
        current_mode = updates.get("tutoring_mode") or state.get("tutoring_mode", "hint")
        confusion_count = updates.get("confusion_pattern_count") or state.get("confusion_pattern_count", 0)

        if current_mode == "step_by_step" and confusion_count >= 3:
            logger.info(
                "Escalating: at step_by_step mode with %d confusion patterns",
                confusion_count,
            )
            return True

        # Check for very low mastery + confusion patterns
        student_context = state.get("student_context", {})
        topic_mastery = student_context.get("topic_mastery", 0.5)

        if topic_mastery < 0.3 and confusion_count >= 2:
            logger.info(
                "Escalating: low mastery (%.2f) + confusion patterns (%d)",
                topic_mastery,
                confusion_count,
            )
            return True

        return False

    async def _escalate_to_learning_tutor(self, state: PracticeHelperState) -> dict:
        """Escalate to Learning Tutor workflow.

        Prepares handoff context for Learning Tutor and updates state
        to indicate escalation is happening.

        Args:
            state: Current workflow state.

        Returns:
            State updates with escalation status and handoff context.
        """
        logger.info(
            "Escalating to Learning Tutor: session=%s, mode_escalations=%d",
            state.get("session_id"),
            state.get("mode_escalation_count", 0),
        )

        question_context = state.get("question_context", {})
        student_context = state.get("student_context", {})

        # Build handoff context for Learning Tutor
        handoff_context = {
            "source": "practice_helper",
            "helper_session_id": state.get("session_id"),
            "practice_session_id": state.get("practice_session_id"),
            "topic_code": state.get("topic_full_code") or state.get("topic_code"),
            "original_question": {
                "question_id": question_context.get("question_id"),
                "question_text": question_context.get("question_text"),
                "correct_answer": question_context.get("correct_answer"),
                "student_answer": question_context.get("student_answer"),
            },
            "student_struggle_points": {
                "mode_escalations": state.get("mode_escalation_count", 0),
                "i_dont_know_count": state.get("i_dont_know_count", 0),
                "confusion_patterns": state.get("confusion_pattern_count", 0),
            },
            "misconceptions": state.get("discovered_misconceptions", []),
            "concepts_to_focus": state.get("concepts_student_lacks", []),
            "suggested_learning_mode": "review",  # Start with review since they've seen it
            "topic_mastery": student_context.get("topic_mastery", 0.5),
            "conversation_summary": self._summarize_conversation(state),
        }

        # Create UI action for frontend navigation
        ui_action = {
            "type": "handoff",
            "target_workflow": "learning_tutor",
            "label": "Let's learn this topic properly",
            "params": {
                "topic_full_code": state.get("topic_full_code") or state.get("topic_code"),
                "topic_name": state.get("topic_name"),
                "handoff_context": handoff_context,
            },
            "route": "/learn",
        }

        # Record escalation in memory
        try:
            student_id = student_context.get("student_id")
            if student_id:
                await self._memory_manager.record_learning_event(
                    tenant_code=state["tenant_code"],
                    student_id=student_id,
                    event_type="practice_helper_escalated",
                    topic=state.get("topic_name", ""),
                    data={
                        "session_id": state["session_id"],
                        "practice_session_id": state.get("practice_session_id"),
                        "question_id": question_context.get("question_id"),
                        "mode_escalations": state.get("mode_escalation_count", 0),
                        "escalated_to": "learning_tutor",
                    },
                    importance=0.7,
                )
        except Exception as e:
            logger.warning("Failed to record escalation event: %s", str(e))

        # Publish escalation event (non-blocking)
        asyncio.create_task(self._publish_escalation_to_learning_tutor(state))

        return {
            "status": "escalating",
            "completion_reason": "escalated_to_learning_tutor",
            "handoff_context": handoff_context,
            "ui_action": ui_action,
            "awaiting_input": False,
            "completed_at": datetime.now().isoformat(),
        }

    def _summarize_conversation(self, state: PracticeHelperState) -> str:
        """Summarize the Practice Helper conversation for handoff.

        Args:
            state: Current workflow state.

        Returns:
            Summary string of key points from the conversation.
        """
        history = state.get("conversation_history", [])
        if not history:
            return "Student requested help but no progress was made."

        # Get key points from conversation
        mode_escalations = state.get("mode_escalation_count", 0)
        turns = len(history)
        current_mode = state.get("tutoring_mode", "hint")

        summary = f"Practice Helper session with {turns} turns. "
        summary += f"Started with {state.get('initial_mode', 'hint')} mode, "

        if mode_escalations > 0:
            summary += f"escalated {mode_escalations} time(s) to {current_mode}. "
        else:
            summary += f"stayed in {current_mode} mode. "

        understanding_progress = state.get("understanding_progress", 0.0)
        summary += f"Understanding progress: {int(understanding_progress * 100)}%. "

        if state.get("confusion_pattern_count", 0) > 0:
            summary += "Student showed signs of confusion. "

        return summary

    async def _generate_response(self, state: PracticeHelperState) -> dict:
        """Generate tutor response using the subject-specific agent.

        Uses execute_conversational with full context including:
        - Question information (in runtime_context for YAML interpolation)
        - Student information
        - Current tutoring mode
        - Conversation history (as messages)
        - Step number (for STEP_BY_STEP mode)

        In STEP_BY_STEP mode, automatically advances current_step after each
        response to ensure varied context for each turn. This prevents
        repetitive responses that occur when the step stays constant.
        The step is capped at max_steps (5) to prevent infinite progression.

        Note: Step is NOT advanced when action is "next_step" since it's
        already advanced in _analyze_message for that action.

        Args:
            state: Current workflow state.

        Returns:
            State updates with tutor response.
        """
        pending_action = state.get("last_action") or "respond"
        student_message = state.get("last_student_message", "")

        # Don't generate response for end/i_understand actions
        if pending_action in ("i_understand", "end"):
            return {}

        logger.info(
            "Generating response: mode=%s, step=%s, action=%s",
            state.get("tutoring_mode"),
            state.get("current_step"),
            pending_action,
        )

        try:
            # Get the subject-specific agent
            agent_id = state.get("agent_id", "practice_helper_tutor_general")
            agent = self._agent_factory.get(agent_id)

            # Get contexts
            question_context = state.get("question_context", {})
            student_context = state.get("student_context", {})
            metrics = state.get("metrics", {})

            # Format options for display
            options = question_context.get("options")
            if isinstance(options, dict):
                options_str = "\n".join(f"  {k}) {v}" for k, v in options.items())
            elif isinstance(options, list):
                options_str = "\n".join(f"  {chr(97+i)}) {opt}" for i, opt in enumerate(options))
            else:
                options_str = str(options) if options else ""

            # Format interests for display
            interests = student_context.get("interests", [])
            if isinstance(interests, list):
                interests_str = ", ".join(interests) if interests else "not specified"
            else:
                interests_str = str(interests) if interests else "not specified"

            # Build runtime_context for YAML prompt interpolation
            runtime_context = {
                # Tutoring context
                "tutoring_mode": state.get("tutoring_mode", "hint"),
                "current_turn": metrics.get("turn_count", 1) + 1,
                "current_step": state.get("current_step", 0),
                # Student information
                "student_age": student_context.get("student_age", 10),
                "grade_level": student_context.get("grade_level", 5),
                "language": student_context.get("language", "en"),
                "topic_mastery": student_context.get("topic_mastery", 0.5),
                "emotional_state": student_context.get("emotional_state", "neutral"),
                "interests": interests_str,
                # Question information
                "topic_name": state.get("topic_name", ""),
                "subject": state.get("subject", "Mathematics"),
                "question_text": question_context.get("question_text", ""),
                "question_type": question_context.get("question_type", "multiple_choice"),
                "options": options_str,
                "correct_answer": question_context.get("correct_answer", ""),
                "student_answer": question_context.get("student_answer", ""),
            }

            # Build messages from conversation history
            messages = []
            for turn in state.get("conversation_history", [])[-10:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role and content:
                    # Map tutor -> assistant, student -> user
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
                history.append(PracticeHelperTurn(
                    role="tutor",
                    content=message_text,
                    timestamp=datetime.now().isoformat(),
                ))

                logger.info("Generated response: %d chars", len(message_text))

                updates = {
                    "last_tutor_response": message_text,
                    "conversation_history": history,
                    "awaiting_input": True,
                    "error": None,
                }

                # In STEP_BY_STEP mode, advance the step after each response
                # This ensures varied context for each turn, preventing repetitive responses
                # Note: Don't advance if action was "next_step" (already advanced in _analyze_message)
                current_mode = state.get("tutoring_mode", "hint")
                if current_mode == TutoringMode.STEP_BY_STEP.value and pending_action != "next_step":
                    current_step = state.get("current_step", 1)
                    # Cap at max_steps (5) to prevent infinite progression
                    max_steps = 5
                    new_step = min(current_step + 1, max_steps)
                    updates["current_step"] = new_step

                    # Also update metrics
                    metrics = dict(state.get("metrics", {}))
                    metrics["current_step"] = new_step
                    updates["metrics"] = metrics

                    logger.debug("Advanced step: %d -> %d", current_step, new_step)

                return updates

        except Exception as e:
            logger.exception("Failed to generate response: %s", str(e))

        # Default response if agent fails
        mode = state.get("tutoring_mode", TutoringMode.HINT.value)
        current_step = state.get("current_step", 1)

        if mode == TutoringMode.STEP_BY_STEP.value:
            default_response = f"**Step {current_step}:** Let's continue working through this. Can you try applying what we've discussed so far?"
        elif mode == TutoringMode.GUIDED.value:
            default_response = "That's a good attempt! Let me ask you a guiding question to help you think through this differently."
        else:
            default_response = "Good thinking! Here's a hint to help you: think about what the question is really asking."

        history = list(state.get("conversation_history", []))
        history.append(PracticeHelperTurn(
            role="tutor",
            content=default_response,
            timestamp=datetime.now().isoformat(),
        ))

        updates = {
            "last_tutor_response": default_response,
            "conversation_history": history,
            "awaiting_input": True,
        }

        # In STEP_BY_STEP mode, advance the step even in fallback case
        # Note: Don't advance if action was "next_step" (already advanced in _analyze_message)
        if mode == TutoringMode.STEP_BY_STEP.value and pending_action != "next_step":
            max_steps = 5
            new_step = min(current_step + 1, max_steps)
            updates["current_step"] = new_step

            metrics = dict(state.get("metrics", {}))
            metrics["current_step"] = new_step
            updates["metrics"] = metrics

        return updates

    async def _check_end(self, state: PracticeHelperState) -> dict:
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

    def _should_continue(self, state: PracticeHelperState) -> Literal["continue", "end"]:
        """Determine if workflow should continue or end.

        Ends on:
        - Error status
        - Student understood (i_understand action)
        - User requested end
        - Max turns exceeded
        - Understanding progress >= 0.9

        Args:
            state: Current workflow state.

        Returns:
            "continue" to wait for more messages, "end" to finish.
        """
        # Check for error
        if state.get("status") == "error":
            return "end"

        # Check if student understood
        if state.get("understood") is True:
            return "end"

        # Check for end action
        last_action = state.get("last_action")
        if last_action in ("i_understand", "end"):
            return "end"

        # Check max turns
        turn_count = state.get("metrics", {}).get("turn_count", 0)
        if turn_count >= MAX_TURNS:
            logger.info("Max turns reached: %d", turn_count)
            return "end"

        # Check understanding progress
        if state.get("understanding_progress", 0.0) >= 0.9:
            return "end"

        return "continue"

    async def _end_session(self, state: PracticeHelperState) -> dict:
        """End the practice helper session.

        Records session completion, determines completion reason, and
        prepares handoff context for returning to Practice workflow.

        Args:
            state: Current workflow state.

        Returns:
            State updates marking completion with handoff context.
        """
        # Determine completion reason
        completion_reason = "user_ended"

        if state.get("understood") is True:
            completion_reason = "understood"
        elif state.get("understanding_progress", 0.0) >= 0.9:
            completion_reason = "understood"
        elif state.get("metrics", {}).get("turn_count", 0) >= MAX_TURNS:
            completion_reason = "max_turns"
        elif state.get("last_action") == "end":
            completion_reason = "user_ended"
        elif state.get("last_action") == "i_understand":
            completion_reason = "understood"

        understood = state.get("understood") or (completion_reason == "understood")

        logger.info(
            "Practice helper session ended: session=%s, turns=%d, mode=%s, reason=%s",
            state.get("session_id"),
            state.get("metrics", {}).get("turn_count", 0),
            state.get("tutoring_mode"),
            completion_reason,
        )

        # Build handoff context for returning to Practice
        question_context = state.get("question_context", {})
        handoff_context = {
            "source": "practice_helper",
            "helper_session_id": state.get("session_id"),
            "practice_session_id": state.get("practice_session_id"),
            "understood": understood,
            "mode_that_helped": state.get("tutoring_mode") if understood else None,
            "escalations_needed": state.get("mode_escalation_count", 0),
            "question_id": question_context.get("question_id"),
            "topic_code": state.get("topic_full_code") or state.get("topic_code"),
        }

        # Create UI action for returning to Practice
        ui_action = None
        if understood and state.get("practice_session_id"):
            ui_action = {
                "type": "handoff",
                "target_workflow": "practice",
                "label": "Try the question again" if understood else "Continue practicing",
                "params": {
                    "practice_session_id": state.get("practice_session_id"),
                    "retry_question": understood,
                    "question_id": question_context.get("question_id"),
                    "handoff_context": handoff_context,
                },
                "route": "/practice",
            }

        # Build final state for event publishing
        final_state = dict(state)
        final_state["completion_reason"] = completion_reason
        final_state["understood"] = understood

        # Publish understood event if student understood
        if understood:
            asyncio.create_task(self._publish_understood(final_state))

        # Publish session completed event (non-blocking)
        asyncio.create_task(self._publish_session_completed(final_state))

        # Record in memory if we have access
        try:
            student_context = state.get("student_context", {})
            student_id = student_context.get("student_id")

            if student_id:
                await self._memory_manager.record_learning_event(
                    tenant_code=state["tenant_code"],
                    student_id=student_id,
                    event_type="practice_helper_session",
                    topic=state.get("topic_name", ""),
                    data={
                        "session_id": state["session_id"],
                        "practice_session_id": state.get("practice_session_id"),
                        "question_id": question_context.get("question_id"),
                        "initial_mode": state.get("initial_mode"),
                        "final_mode": state.get("tutoring_mode"),
                        "mode_escalations": state.get("mode_escalation_count", 0),
                        "turn_count": state.get("metrics", {}).get("turn_count", 0),
                        "understood": understood,
                        "completion_reason": completion_reason,
                    },
                    importance=0.6,
                )

                # Record procedural observation (non-blocking)
                asyncio.create_task(self._record_procedural_observation(final_state))

        except Exception as e:
            logger.warning("Failed to record session completion: %s", str(e))

        return {
            "status": "completed",
            "completion_reason": completion_reason,
            "understood": understood,
            "handoff_context": handoff_context,
            "ui_action": ui_action,
            "completed_at": datetime.now().isoformat(),
            "awaiting_input": False,
        }

    # =========================================================================
    # Event Publishing Methods
    # =========================================================================

    async def _publish_session_started(self, state: PracticeHelperState) -> None:
        """Publish practice helper session started event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            student_context = state.get("student_context", {})
            question_context = state.get("question_context", {})

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await self._event_tracker.track_event(
                event_type="practice_helper.session.started",
                student_id=student_context.get("student_id"),
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "practice_helper",
                    "practice_session_id": state.get("practice_session_id"),
                    "question_id": question_context.get("question_id"),
                    "initial_mode": state.get("tutoring_mode"),
                    "topic_name": state.get("topic_name"),
                    "subject": state.get("subject"),
                },
            )
            logger.debug("Published practice helper session started event")
        except Exception as e:
            logger.warning("Failed to publish session started event: %s", str(e))

    async def _publish_session_completed(self, state: PracticeHelperState) -> None:
        """Publish practice helper session completed event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            student_context = state.get("student_context", {})
            metrics = state.get("metrics", {})

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await self._event_tracker.track_event(
                event_type="practice_helper.session.completed",
                student_id=student_context.get("student_id"),
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "practice_helper",
                    "initial_mode": state.get("initial_mode"),
                    "final_mode": state.get("tutoring_mode"),
                    "mode_escalations": state.get("mode_escalation_count", 0),
                    "turn_count": metrics.get("turn_count", 0),
                    "understood": state.get("understood", False),
                    "completion_reason": state.get("completion_reason"),
                    "understanding_progress": state.get("understanding_progress", 0.0),
                },
            )
            logger.debug("Published practice helper session completed event")
        except Exception as e:
            logger.warning("Failed to publish session completed event: %s", str(e))

    async def _publish_mode_escalated(
        self,
        state: PracticeHelperState,
        from_mode: str,
        to_mode: str,
    ) -> None:
        """Publish mode escalation event - important struggle signal.

        Args:
            state: Current workflow state.
            from_mode: Previous tutoring mode.
            to_mode: New tutoring mode.
        """
        if self._event_tracker is None:
            return

        try:
            student_context = state.get("student_context", {})
            metrics = state.get("metrics", {})

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await self._event_tracker.track_event(
                event_type="practice_helper.mode.escalated",
                student_id=student_context.get("student_id"),
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "practice_helper",
                    "from_mode": from_mode,
                    "to_mode": to_mode,
                    "escalation_count": state.get("mode_escalation_count", 0) + 1,
                    "turn_count": metrics.get("turn_count", 0),
                    "understanding_progress": state.get("understanding_progress", 0.0),
                },
            )
            logger.debug("Published mode escalated event: %s -> %s", from_mode, to_mode)
        except Exception as e:
            logger.warning("Failed to publish mode escalated event: %s", str(e))

    async def _publish_understood(self, state: PracticeHelperState) -> None:
        """Publish understanding achieved event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            student_context = state.get("student_context", {})
            metrics = state.get("metrics", {})

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await self._event_tracker.track_event(
                event_type="practice_helper.understood",
                student_id=student_context.get("student_id"),
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "practice_helper",
                    "mode_that_helped": state.get("tutoring_mode"),
                    "escalations_needed": state.get("mode_escalation_count", 0),
                    "turn_count": metrics.get("turn_count", 0),
                },
            )
            logger.debug("Published understood event")
        except Exception as e:
            logger.warning("Failed to publish understood event: %s", str(e))

    async def _publish_escalation_to_learning_tutor(
        self,
        state: PracticeHelperState,
    ) -> None:
        """Publish escalation to Learning Tutor event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            student_context = state.get("student_context", {})
            metrics = state.get("metrics", {})

            topic_codes = {
                "framework_code": state.get("topic_framework_code"),
                "subject_code": state.get("topic_subject_code"),
                "grade_code": state.get("topic_grade_code"),
                "unit_code": state.get("topic_unit_code"),
                "code": state.get("topic_code"),
            }
            topic_codes = {k: v for k, v in topic_codes.items() if v is not None}

            await self._event_tracker.track_event(
                event_type="practice_helper.escalated_to_learning_tutor",
                student_id=student_context.get("student_id"),
                session_id=state["session_id"],
                topic_codes=topic_codes if topic_codes else None,
                data={
                    "session_type": "practice_helper",
                    "practice_session_id": state.get("practice_session_id"),
                    "mode_escalations": state.get("mode_escalation_count", 0),
                    "confusion_patterns": state.get("confusion_pattern_count", 0),
                    "turn_count": metrics.get("turn_count", 0),
                    "topic_mastery": student_context.get("topic_mastery", 0.5),
                    "student_requested_lesson": state.get("student_wants_full_lesson", False),
                },
            )
            logger.debug("Published escalation to Learning Tutor event")
        except Exception as e:
            logger.warning("Failed to publish escalation event: %s", str(e))

    # =========================================================================
    # Memory Recording Methods
    # =========================================================================

    async def _record_procedural_observation(
        self,
        state: PracticeHelperState,
    ) -> None:
        """Record tutoring pattern observation in procedural memory.

        Tracks behavioral patterns that inform personalization:
        - When the student needs help (time of day)
        - Which tutoring modes are most effective
        - Mode escalation patterns

        Args:
            state: Current workflow state.
        """
        try:
            student_context = state.get("student_context", {})
            student_id = student_context.get("student_id")
            if not student_id:
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
                "session_type": "practice_helper",
                "time_of_day": time_of_day,
                "initial_mode": state.get("initial_mode"),
                "final_mode": state.get("tutoring_mode"),
                "mode_escalations": state.get("mode_escalation_count", 0),
                "turn_count": metrics.get("turn_count", 0),
                "understood": state.get("understood", False),
                "understanding_progress": state.get("understanding_progress", 0.0),
                "topic": state.get("topic_name"),
                "subject": state.get("subject"),
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
                state.get("tutoring_mode"),
            )

        except Exception as e:
            logger.warning("Failed to record procedural observation: %s", str(e))
