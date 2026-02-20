# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tutoring conversation workflow using LangGraph.

This workflow manages an interactive tutoring conversation with full
personalization support through 4-layer memory and educational theory
integration.

Workflow Structure:
    initialize
        ↓
    load_context (4-layer memory + theory recommendations)
        ↓
    generate_greeting (proactive first message)
        ↓
    wait_for_message [INTERRUPT POINT]
        ↓
    analyze_intent (emotional + intent analysis)
        ↓
    retrieve_context (RAG retrieval)
        ↓
    generate_response (theory-informed response)
        ↓
    update_memory (all 4 layers + FSRS scheduling)
        ↓
    check_end
        ↓
    [conditional: continue → wait_for_message, end → end_session]

The workflow uses checkpointing with interrupt_before pattern,
matching the Practice workflow architecture for reliable pause/resume.
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.agents import AgentFactory, AgentExecutionContext
from src.core.emotional import EmotionalSignalSource

if TYPE_CHECKING:
    from src.core.emotional import EmotionalStateService
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.core.educational.orchestrator import TheoryOrchestrator
from src.core.personas.manager import PersonaManager
from src.core.orchestration.states.tutoring import (
    TutoringState,
    ConversationTurn,
    ExplanationRecord,
    MessageAnalysis,
    create_initial_tutoring_state,
)

logger = logging.getLogger(__name__)


class TutoringWorkflow:
    """LangGraph workflow for tutoring conversations.

    This workflow orchestrates a multi-turn tutoring conversation
    with full personalization support:

    Memory Integration (4 Layers):
        - Episodic: Recent learning events for context
        - Semantic: Topic mastery levels for ZPD
        - Procedural: Learning patterns for VARK
        - Associative: Student interests for analogies

    Theory Integration (7 Theories):
        - ZPD: Optimal difficulty level
        - Bloom's: Cognitive complexity
        - VARK: Content format preference
        - Scaffolding: Support level
        - Mastery: Readiness to advance
        - Socratic: Questioning style
        - Spaced Repetition: Review scheduling

    The workflow uses interrupt_before=["wait_for_message"] pattern
    for reliable pause/resume, matching Practice workflow architecture.

    Attributes:
        agent_factory: Factory for creating agents.
        memory_manager: Manager for 4-layer memory operations.
        rag_retriever: Retriever for context via RAG.
        theory_orchestrator: Orchestrator for 7 educational theories.
        persona_manager: Manager for tutor personas.

    Example:
        >>> workflow = TutoringWorkflow(agent_factory, memory_manager, ...)
        >>> initial_state = create_initial_tutoring_state(...)
        >>> result = await workflow.run(initial_state, thread_id="session_123")
        >>> # result contains first_greeting from generate_greeting node
        >>> # Workflow is now paused at wait_for_message
        >>>
        >>> # To send a message:
        >>> result = await workflow.send_message(thread_id, "What is algebra?")
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        memory_manager: MemoryManager,
        rag_retriever: RAGRetriever,
        theory_orchestrator: TheoryOrchestrator,
        persona_manager: PersonaManager,
        checkpointer: BaseCheckpointSaver | None = None,
        emotional_service: "EmotionalStateService | None" = None,
    ):
        """Initialize the tutoring workflow.

        Args:
            agent_factory: Factory for creating agents.
            memory_manager: Manager for 4-layer memory operations.
            rag_retriever: Retriever for context via RAG.
            theory_orchestrator: Orchestrator for educational theories.
            persona_manager: Manager for tutor personas.
            checkpointer: Checkpointer for state persistence (required).
            emotional_service: Service for recording emotional signals.
        """
        self._agent_factory = agent_factory
        self._memory_manager = memory_manager
        self._rag_retriever = rag_retriever
        self._theory_orchestrator = theory_orchestrator
        self._persona_manager = persona_manager
        self._checkpointer = checkpointer
        self._emotional_service = emotional_service

        # Build the workflow graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Graph Structure:
            initialize → load_context → generate_greeting →
            wait_for_message [INTERRUPT] → analyze_intent →
            retrieve_context → generate_response → update_memory →
            check_end → [continue/end]

        Returns:
            StateGraph configured for tutoring conversations.
        """
        graph = StateGraph(TutoringState)

        # Add nodes
        graph.add_node("initialize", self._initialize)
        graph.add_node("load_context", self._load_context)
        graph.add_node("generate_greeting", self._generate_greeting)
        graph.add_node("wait_for_message", self._wait_for_message)
        graph.add_node("analyze_intent", self._analyze_intent)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_memory", self._update_memory)
        graph.add_node("check_end", self._check_end)
        graph.add_node("end_session", self._end_session)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges
        graph.add_edge("initialize", "load_context")
        graph.add_edge("load_context", "generate_greeting")
        graph.add_edge("generate_greeting", "wait_for_message")
        graph.add_edge("wait_for_message", "analyze_intent")
        graph.add_edge("analyze_intent", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", "check_end")

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
        when waiting for student input, matching Practice pattern.

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_message"],
        )

    async def run(
        self,
        initial_state: TutoringState,
        thread_id: str,
    ) -> TutoringState:
        """Run the workflow from initial state.

        Executes initialize → load_context → generate_greeting, then
        pauses at wait_for_message (interrupt point).

        Args:
            initial_state: Starting state for the workflow.
            thread_id: Thread ID for checkpointing.

        Returns:
            Workflow state with first_greeting populated.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        result = await compiled.ainvoke(initial_state, config=config)
        return result

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> TutoringState:
        """Send a message and get response.

        Uses aupdate_state + ainvoke(None) pattern for proper resume,
        matching Practice workflow architecture.

        Args:
            thread_id: Thread ID for the conversation.
            message: Student message.

        Returns:
            Updated workflow state with tutor response.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow is paused
        state_snapshot = await compiled.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No state found for thread {thread_id}")

        # Update state with student message using aupdate_state
        # NOTE: Do NOT use as_node here - we want wait_for_message to run
        # so it can convert _pending_message into last_student_message
        await compiled.aupdate_state(
            config,
            {
                "_pending_message": message,
                "awaiting_input": False,
            },
        )

        logger.info("Resuming workflow for thread=%s with message: %s...", thread_id, message[:50])

        # Resume workflow by passing None
        # This continues from interrupt point (BEFORE wait_for_message)
        # So wait_for_message runs first and processes the _pending_message
        result = await compiled.ainvoke(None, config=config)

        logger.info(
            "Workflow resumed: last_tutor_response=%s...",
            str(result.get("last_tutor_response", ""))[:50] if result else "None"
        )
        return result

    # =========================================================================
    # Node Implementations
    # =========================================================================

    async def _initialize(self, state: TutoringState) -> dict:
        """Initialize the tutoring session.

        Sets status to active and prepares for context loading.

        Args:
            state: Current workflow state.

        Returns:
            State updates with active status.
        """
        logger.info(
            "Initializing tutoring session: session=%s, topic=%s, persona=%s",
            state.get("session_id"),
            state.get("topic"),
            state.get("persona_id"),
        )

        return {
            "status": "active",
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: TutoringState) -> dict:
        """Load full 4-layer memory context and theory recommendations.

        This node mirrors Practice workflow's _load_context, loading:
        - Full memory context (episodic, semantic, procedural, associative)
        - Emotional context (current emotional state)
        - Theory recommendations (all 7 theories combined)

        Args:
            state: Current workflow state.

        Returns:
            State updates with loaded context.
        """
        logger.info(
            "Loading context: student=%s, topic=%s",
            state["student_id"],
            state["topic"],
        )

        memory_context = {}
        theory_recommendations = {}
        emotional_context = None

        try:
            # Convert student_id to UUID if needed
            student_uuid = UUID(state["student_id"]) if isinstance(state["student_id"], str) else state["student_id"]

            # Load full 4-layer memory context
            full_context = await self._memory_manager.get_full_context(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                topic=state["topic"],
            )

            if full_context:
                memory_context = full_context.model_dump()
                logger.debug(
                    "Loaded memory context: episodic=%d, semantic_topics=%d",
                    len(full_context.episodic),
                    len(full_context.semantic.topics) if full_context.semantic else 0,
                )

        except Exception as e:
            logger.warning("Failed to load memory context: %s", str(e))

        try:
            # Load current emotional context
            if self._emotional_service:
                emotional_state = await self._emotional_service.get_current_state(
                    student_id=UUID(state["student_id"]),
                )
                if emotional_state:
                    # EmotionalContext is a dataclass, use to_dict()
                    emotional_context = emotional_state.to_dict()
                    logger.debug(
                        "Loaded emotional context: state=%s",
                        emotional_state.current_state,
                    )

        except Exception as e:
            logger.warning("Failed to load emotional context: %s", str(e))

        try:
            # Get theory recommendations
            from src.models.memory import FullMemoryContext

            memory_obj = None
            if memory_context:
                try:
                    memory_obj = FullMemoryContext.model_validate(memory_context)
                except Exception:
                    pass

            theory_recs = await self._theory_orchestrator.get_recommendations(
                tenant_code=state["tenant_code"],
                student_id=state["student_id"],
                topic=state["topic"],
                memory_context=memory_obj,
            )

            if theory_recs:
                theory_recommendations = theory_recs.model_dump()
                logger.debug(
                    "Theory recommendations: difficulty=%.2f, scaffold=%s, style=%s",
                    theory_recs.difficulty,
                    theory_recs.scaffold_level.value if theory_recs.scaffold_level else "unknown",
                    theory_recs.questioning_style.value if theory_recs.questioning_style else "unknown",
                )

        except Exception as e:
            logger.warning("Failed to load theory recommendations: %s", str(e))

        return {
            "memory_context": memory_context,
            "theory_recommendations": theory_recommendations,
            "emotional_context": emotional_context,
        }

    async def _generate_greeting(self, state: TutoringState) -> dict:
        """Generate proactive first greeting from tutor.

        Creates a personalized greeting based on:
        - Student's memory context (prior knowledge, interests)
        - Current emotional state
        - Topic being discussed
        - Selected persona

        This enables a warm, personalized start to the tutoring session.

        Args:
            state: Current workflow state.

        Returns:
            State updates with first_greeting and conversation history.
        """
        logger.info("Generating greeting for topic: %s", state["topic"])

        try:
            # Get tutor agent
            agent = self._agent_factory.get("tutor")

            # Set persona
            persona_id = state.get("persona_id", "tutor")
            try:
                persona = self._persona_manager.get_persona(persona_id)
                agent.set_persona(persona)
            except Exception:
                pass

            # Build greeting context from memory
            memory_context = state.get("memory_context", {})
            theory_recs = state.get("theory_recommendations", {})
            emotional_context = state.get("emotional_context")

            # Extract student info for personalization
            student_interests = []
            if memory_context.get("associative"):
                interests = memory_context["associative"].get("interests", [])
                student_interests = [i.get("name", "") for i in interests[:3]]

            prior_knowledge = []
            if memory_context.get("semantic"):
                topics = memory_context["semantic"].get("topics", [])
                prior_knowledge = [
                    t.get("topic", "") for t in topics
                    if t.get("mastery_level", 0) > 0.5
                ][:3]

            # Build greeting request
            context = AgentExecutionContext(
                tenant_id=state["tenant_id"],
                student_id=state["student_id"],
                topic=state["topic"],
                intent="greeting_generation",
                params={
                    "topic": state["topic"],
                    "subtopic": state.get("subtopic"),
                    "student_interests": student_interests,
                    "prior_knowledge": prior_knowledge,
                    "emotional_state": emotional_context.get("current_state") if emotional_context else None,
                    "scaffold_level": theory_recs.get("scaffold_level", "moderate"),
                    "questioning_style": theory_recs.get("questioning_style", "guided"),
                },
            )

            response = await agent.execute(context)

            if response.success and response.result:
                # Extract greeting text
                if hasattr(response.result, "greeting"):
                    greeting_text = response.result.greeting
                elif hasattr(response.result, "content"):
                    greeting_text = response.result.content
                elif isinstance(response.result, dict):
                    greeting_text = response.result.get("greeting") or response.result.get("content", "")
                else:
                    greeting_text = str(response.result)

                # Add to conversation history
                history = [ConversationTurn(
                    role="tutor",
                    content=greeting_text,
                    timestamp=datetime.now().isoformat(),
                    intent="greeting",
                )]

                logger.info("Generated greeting: %s...", greeting_text[:50])

                return {
                    "first_greeting": greeting_text,
                    "last_tutor_response": greeting_text,
                    "conversation_history": history,
                    "awaiting_input": True,
                }

        except Exception as e:
            logger.warning("Failed to generate greeting, using default: %s", str(e))

        # Default greeting if agent fails
        topic = state.get("topic", "this topic")
        default_greeting = f"Hello! I'm here to help you learn about {topic}. What would you like to know?"

        return {
            "first_greeting": default_greeting,
            "last_tutor_response": default_greeting,
            "conversation_history": [ConversationTurn(
                role="tutor",
                content=default_greeting,
                timestamp=datetime.now().isoformat(),
                intent="greeting",
            )],
            "awaiting_input": True,
        }

    async def _wait_for_message(self, state: TutoringState) -> dict:
        """Wait for student message (interrupt point).

        This node is the interrupt point. When workflow reaches here,
        it pauses and waits for send_message() to inject a message
        via aupdate_state.

        Args:
            state: Current workflow state.

        Returns:
            State updates extracting pending message.
        """
        pending_message = state.get("_pending_message")
        logger.info(
            "wait_for_message node: pending_message=%s, awaiting=%s",
            pending_message[:50] if pending_message else None,
            state.get("awaiting_input")
        )

        if pending_message:
            # Add student message to conversation history
            history = list(state.get("conversation_history", []))
            history.append(ConversationTurn(
                role="student",
                content=pending_message,
                timestamp=datetime.now().isoformat(),
            ))

            return {
                "last_student_message": pending_message,
                "_pending_message": None,  # Clear pending
                "conversation_history": history,
                "awaiting_input": False,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "awaiting_input": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _analyze_intent(self, state: TutoringState) -> dict:
        """Analyze the intent and emotional state of student message.

        Uses the emotional_analyzer agent to detect:
        - Message intent (question, clarification, acknowledgment, etc.)
        - Emotional state (frustrated, confused, engaged, etc.)
        - Whether emotional support is needed
        - Understanding signals (did student show they understood?)

        Args:
            state: Current workflow state.

        Returns:
            State updates with message analysis and metrics.
        """
        message = state.get("last_student_message", "")

        if not message:
            return {}

        logger.debug("Analyzing message: %s...", message[:50])

        try:
            # Get emotional_analyzer agent
            agent = self._agent_factory.get("emotional_analyzer")

            # Build conversation history for context
            history_for_context = [
                {"role": turn.get("role", ""), "content": turn.get("content", "")}
                for turn in state.get("conversation_history", [])[-5:]
            ]

            # Create execution context
            context = AgentExecutionContext(
                tenant_id=state["tenant_id"],
                student_id=state["student_id"],
                topic=state.get("topic", ""),
                intent="message_analysis",
                params={
                    "message": message,
                    "conversation_history": history_for_context,
                    "language": state.get("language", "en"),
                },
            )

            response = await agent.execute(context)

            if response.success and response.result:
                result = response.result

                analysis = MessageAnalysis(
                    intent=getattr(result, "intent", "question"),
                    intent_confidence=getattr(result, "intent_confidence", 0.8),
                    emotional_state=getattr(result, "emotional_state", "neutral"),
                    intensity=getattr(result, "intensity", "low"),
                    sentiment_confidence=getattr(result, "sentiment_confidence", 0.8),
                    triggers=getattr(result, "triggers", []),
                    requires_support=getattr(result, "requires_support", False),
                    suggested_response_tone=getattr(result, "suggested_response_tone", "neutral"),
                    understanding_signal=getattr(result, "understanding_signal", None),
                )

                logger.info(
                    "Message analysis: intent=%s, emotional_state=%s, intensity=%s",
                    analysis["intent"],
                    analysis["emotional_state"],
                    analysis["intensity"],
                )

                # Fire-and-forget: Record emotional signal
                if self._emotional_service is not None:
                    asyncio.create_task(
                        self._record_emotional_signal(
                            student_id=state["student_id"],
                            analysis=analysis,
                            conversation_id=state.get("session_id"),
                        )
                    )

                # Update metrics based on analysis
                metrics = dict(state.get("metrics", {}))
                metrics["turns_count"] = metrics.get("turns_count", 0) + 1

                intent = analysis["intent"]
                if intent in ["question", "clarification", "help_request"]:
                    metrics["student_questions"] = metrics.get("student_questions", 0) + 1
                if intent == "clarification":
                    metrics["clarifications_requested"] = metrics.get("clarifications_requested", 0) + 1

                # Track understanding/confusion signals
                if analysis.get("understanding_signal") is True:
                    metrics["understanding_signals"] = metrics.get("understanding_signals", 0) + 1
                if analysis["emotional_state"] in ["confused", "frustrated"]:
                    metrics["confusion_signals"] = metrics.get("confusion_signals", 0) + 1
                if analysis.get("requires_support"):
                    metrics["support_interventions"] = metrics.get("support_interventions", 0) + 1

                # Update conversation history with analysis
                history = list(state.get("conversation_history", []))
                if history and history[-1]["role"] == "student":
                    history[-1]["intent"] = intent
                    history[-1]["emotional_state"] = analysis["emotional_state"]

                return {
                    "last_message_analysis": analysis,
                    "metrics": metrics,
                    "conversation_history": history,
                    "current_focus": self._extract_focus(message, state.get("topic", "")),
                }

        except Exception as e:
            logger.warning("Message analysis failed: %s", str(e))

        # Fallback: minimal analysis
        metrics = dict(state.get("metrics", {}))
        metrics["turns_count"] = metrics.get("turns_count", 0) + 1

        return {
            "last_message_analysis": MessageAnalysis(
                intent="question",
                intent_confidence=0.5,
                emotional_state="neutral",
                intensity="low",
                sentiment_confidence=0.5,
            ),
            "metrics": metrics,
        }

    async def _record_emotional_signal(
        self,
        student_id: str,
        analysis: MessageAnalysis,
        conversation_id: str | None = None,
    ) -> None:
        """Record emotional signal from message analysis (fire-and-forget).

        Args:
            student_id: Student ID.
            analysis: Message analysis with emotional data.
            conversation_id: Optional conversation ID.
        """
        try:
            await self._emotional_service.record_analyzed_signal(
                student_id=UUID(student_id),
                source=EmotionalSignalSource.LEARNING,
                emotional_state=analysis["emotional_state"],
                intensity=analysis["intensity"],
                confidence=analysis["sentiment_confidence"],
                triggers=analysis.get("triggers", []),
                activity_id=conversation_id,
                activity_type="tutoring_conversation",
                context={
                    "intent": analysis["intent"],
                    "requires_support": analysis.get("requires_support", False),
                },
            )
            logger.debug(
                "Recorded emotional signal: student=%s, state=%s",
                student_id,
                analysis["emotional_state"],
            )
        except Exception as e:
            logger.warning(
                "Failed to record emotional signal: student=%s, error=%s",
                student_id,
                str(e),
            )

    async def _retrieve_context(self, state: TutoringState) -> dict:
        """Retrieve relevant context via RAG.

        Retrieves from multiple sources:
        - Curriculum content
        - Student's episodic memories
        - Student's interests (for analogies)

        Args:
            state: Current workflow state.

        Returns:
            State updates with RAG context.
        """
        message = state.get("last_student_message", "")
        topic = state.get("topic", "")
        current_focus = state.get("current_focus", topic)

        if not message:
            return {}

        logger.debug("Retrieving context for: %s", message[:50])

        try:
            # Combine message with focus for better retrieval
            query = f"{current_focus}: {message}"

            results = await self._rag_retriever.retrieve(
                tenant_id=state["tenant_id"],
                query=query,
                top_k=5,
            )

            # Convert to serializable format
            rag_context = [
                {
                    "content": r.content,
                    "source": r.source.value if hasattr(r.source, "value") else str(r.source),
                    "score": r.score,
                }
                for r in results
            ]

            logger.debug("Retrieved %d context items", len(rag_context))

            return {"rag_context": rag_context}

        except Exception as e:
            logger.warning("RAG retrieval failed: %s", str(e))
            return {"rag_context": []}

    async def _generate_response(self, state: TutoringState) -> dict:
        """Generate tutor response using theory-informed approach.

        Uses full context for personalization:
        - Memory context for prior knowledge
        - Theory recommendations for pedagogy
        - RAG context for content
        - Emotional analysis for tone

        Args:
            state: Current workflow state.

        Returns:
            State updates with tutor response.
        """
        message = state.get("last_student_message", "")
        logger.info(
            "generate_response node: last_student_message=%s, pending=%s",
            message[:50] if message else None,
            state.get("_pending_message")
        )

        if not message:
            logger.warning("generate_response: No message to respond to!")
            return {}

        logger.info("Generating response for: %s...", message[:50])

        try:
            # Get tutor agent
            agent = self._agent_factory.get("tutor")

            # Set persona
            persona_id = state.get("persona_id", "tutor")
            persona = None
            try:
                persona = self._persona_manager.get_persona(persona_id)
                agent.set_persona(persona)
            except Exception:
                pass

            # Get dicts from state
            memory_context_dict = state.get("memory_context", {})
            theory_recs_dict = state.get("theory_recommendations", {})
            analysis = state.get("last_message_analysis", {})

            # Convert dict to object (following Practice pattern)
            # If dict is empty or conversion fails, use None
            memory_obj = None
            if memory_context_dict:
                try:
                    from src.models.memory import FullMemoryContext
                    memory_obj = FullMemoryContext.model_validate(memory_context_dict)
                except Exception as e:
                    logger.warning("Failed to convert memory_context to object: %s", str(e))

            theory_obj = None
            if theory_recs_dict:
                try:
                    from src.core.educational.orchestrator import CombinedRecommendation
                    theory_obj = CombinedRecommendation.model_validate(theory_recs_dict)
                except Exception as e:
                    logger.warning("Failed to convert theory_recommendations to object: %s", str(e))

            # Build context from RAG results
            rag_context = state.get("rag_context", [])
            context_text = "\n".join([r.get("content", "") for r in rag_context[:3]])

            # Determine response approach based on analysis
            response_tone = analysis.get("suggested_response_tone", "neutral")
            requires_support = analysis.get("requires_support", False)

            # Create execution context with full personalization
            # Use .get() with defaults for theory-driven params (works with empty dict too)
            context = AgentExecutionContext(
                tenant_id=state["tenant_id"],
                student_id=state["student_id"],
                topic=state["topic"],
                intent="concept_explanation",
                params={
                    "concept": state.get("current_focus", state["topic"]),
                    "student_question": message,
                    "context": context_text,
                    # Theory-driven parameters (with defaults)
                    "target_level": theory_recs_dict.get("scaffold_level", "moderate"),
                    "bloom_level": theory_recs_dict.get("bloom_level", "understand"),
                    "content_format": theory_recs_dict.get("content_format", "text"),
                    "questioning_style": theory_recs_dict.get("questioning_style", "guided"),
                    "guide_vs_tell_ratio": theory_recs_dict.get("guide_vs_tell_ratio", 0.5),
                    # Emotional adjustment
                    "response_tone": response_tone,
                    "requires_support": requires_support,
                    "emotional_state": analysis.get("emotional_state", "neutral"),
                },
                memory=memory_obj,   # FullMemoryContext object or None
                theory=theory_obj,   # CombinedRecommendation object or None
                persona=persona,
            )

            response = await agent.execute(context)

            if response.success and response.result:
                # Extract response text
                if hasattr(response.result, "explanation"):
                    response_text = response.result.explanation
                elif hasattr(response.result, "content"):
                    response_text = response.result.content
                elif isinstance(response.result, dict):
                    response_text = response.result.get("explanation") or response.result.get("content", "")
                else:
                    response_text = str(response.result)

                # Add to conversation history
                history = list(state.get("conversation_history", []))
                history.append(ConversationTurn(
                    role="tutor",
                    content=response_text,
                    timestamp=datetime.now().isoformat(),
                ))

                # Track explanation
                explanations = list(state.get("concepts_explained", []))
                current_focus = state.get("current_focus", state["topic"])

                explanations.append(ExplanationRecord(
                    concept=current_focus,
                    explanation=response_text[:500],
                    bloom_level=theory_recs_dict.get("bloom_level", "understand"),
                    scaffold_level=theory_recs_dict.get("scaffold_level", "moderate"),
                    student_understood=None,  # Will be determined by next message
                    mastery_delta=0.0,  # Will be calculated in update_memory
                ))

                # Update metrics
                metrics = dict(state.get("metrics", {}))
                metrics["explanations_given"] = metrics.get("explanations_given", 0) + 1

                concepts = list(metrics.get("concepts_covered", []))
                if current_focus and current_focus not in concepts:
                    concepts.append(current_focus)
                metrics["concepts_covered"] = concepts

                # Track concepts for review scheduling
                concepts_for_review = list(state.get("concepts_for_review", []))
                if current_focus and current_focus not in concepts_for_review:
                    concepts_for_review.append(current_focus)

                logger.info("Generated response: %d chars", len(response_text))

                return {
                    "last_tutor_response": response_text,
                    "conversation_history": history,
                    "concepts_explained": explanations,
                    "metrics": metrics,
                    "concepts_for_review": concepts_for_review,
                    "awaiting_input": True,
                    "error": None,
                }

        except Exception as e:
            logger.exception("Response generation failed")
            return {
                "error": str(e),
                "awaiting_input": True,
            }

        return {
            "error": "Failed to generate response",
            "awaiting_input": True,
        }

    async def _update_memory(self, state: TutoringState) -> dict:
        """Update all 4 memory layers with the interaction.

        Updates:
        - Episodic: Records the tutoring interaction
        - Semantic: Updates topic mastery based on understanding signals
        - Procedural: Records learning patterns
        - Spaced Repetition: Schedules concepts for review

        This mirrors Practice workflow's comprehensive memory updates.

        Args:
            state: Current workflow state.

        Returns:
            State updates with memory changes tracked.
        """
        logger.debug("Updating memory layers")

        mastery_updates = dict(state.get("mastery_updates", {}))

        try:
            # 1. Record episodic memory (tutoring interaction)
            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=state["student_id"],
                event_type="tutoring_interaction",
                topic=state["topic"],
                data={
                    "session_id": state["session_id"],
                    "student_message": state.get("last_student_message", "")[:500],
                    "tutor_response": state.get("last_tutor_response", "")[:500],
                    "current_focus": state.get("current_focus"),
                    "intent": state.get("last_message_analysis", {}).get("intent"),
                    "emotional_state": state.get("last_message_analysis", {}).get("emotional_state"),
                },
            )
            logger.debug("Recorded episodic memory")

        except Exception as e:
            logger.warning("Failed to record episodic memory: %s", str(e))

        try:
            # 2. Update semantic memory (mastery) based on understanding signals
            analysis = state.get("last_message_analysis", {})
            current_focus = state.get("current_focus", state["topic"])

            # Determine mastery delta based on understanding signals
            if analysis.get("understanding_signal") is True:
                # Student showed understanding - increase mastery
                delta = 0.03  # Smaller than Practice (0.05) since explanation != assessment
                await self._memory_manager.update_topic_mastery(
                    tenant_code=state["tenant_code"],
                    student_id=state["student_id"],
                    topic=current_focus,
                    delta=delta,
                )
                mastery_updates[current_focus] = mastery_updates.get(current_focus, 0) + delta
                logger.debug("Updated mastery for %s: +%.2f", current_focus, delta)

            elif analysis.get("emotional_state") in ["confused", "frustrated"]:
                # Student is struggling - might indicate misunderstanding
                # Don't decrease mastery, but track it
                struggling = list(state.get("concepts_struggling", []))
                if current_focus and current_focus not in struggling:
                    struggling.append(current_focus)

        except Exception as e:
            logger.warning("Failed to update semantic memory: %s", str(e))

        try:
            # 3. Record procedural memory (learning patterns)
            now = datetime.now()
            hour = now.hour

            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            theory_recs = state.get("theory_recommendations", {})

            await self._memory_manager.record_procedural_observation(
                tenant_code=state["tenant_code"],
                student_id=state["student_id"],
                strategy_type="learning_session",
                parameters={
                    "time_of_day": time_of_day,
                    "content_format": theory_recs.get("content_format", "text"),
                    "session_type": "tutoring",
                    "topic": state["topic"],
                    "persona": state.get("persona_id"),
                },
                effectiveness=1.0 if analysis.get("understanding_signal") else 0.5,
            )
            logger.debug("Recorded procedural observation")

        except Exception as e:
            logger.warning("Failed to record procedural memory: %s", str(e))

        try:
            # 4. Schedule spaced repetition for discussed concepts
            concepts_for_review = state.get("concepts_for_review", [])

            for concept in concepts_for_review:
                await self._memory_manager.schedule_concept_review(
                    tenant_code=state["tenant_code"],
                    student_id=state["student_id"],
                    concept=concept,
                    rating=3,  # "Good" - explained but not tested
                )
            logger.debug("Scheduled %d concepts for review", len(concepts_for_review))

        except Exception as e:
            logger.warning("Failed to schedule spaced repetition: %s", str(e))

        return {
            "last_activity_at": datetime.now().isoformat(),
            "mastery_updates": mastery_updates,
        }

    async def _check_end(self, state: TutoringState) -> dict:
        """Check if session should end and update duration.

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

    def _should_continue(self, state: TutoringState) -> Literal["continue", "end"]:
        """Determine if workflow should continue or end.

        Ends on:
        - Error status
        - "farewell" intent detected
        - Explicit session end request

        Args:
            state: Current workflow state.

        Returns:
            "continue" to wait for more messages, "end" to finish.
        """
        # Check for error
        if state.get("status") == "error":
            return "end"

        # Check agent-analyzed intent for farewell
        analysis = state.get("last_message_analysis")
        if analysis:
            intent = analysis.get("intent", "")
            if intent in ["farewell", "goodbye", "end_session"]:
                return "end"

        return "continue"

    async def _end_session(self, state: TutoringState) -> dict:
        """End the tutoring session with final memory updates.

        Performs final updates:
        - Sets status to completed
        - Applies any pending mastery updates
        - Records session completion in episodic memory

        Args:
            state: Current workflow state.

        Returns:
            State updates marking completion.
        """
        logger.info(
            "Tutoring session ended: session=%s, turns=%d, concepts=%d",
            state.get("session_id"),
            state.get("metrics", {}).get("turns_count", 0),
            len(state.get("metrics", {}).get("concepts_covered", [])),
        )

        try:
            # Record session completion in episodic memory
            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=state["student_id"],
                event_type="tutoring_session_completed",
                topic=state["topic"],
                data={
                    "session_id": state["session_id"],
                    "turns_count": state.get("metrics", {}).get("turns_count", 0),
                    "concepts_covered": state.get("metrics", {}).get("concepts_covered", []),
                    "mastery_updates": state.get("mastery_updates", {}),
                    "duration_seconds": state.get("metrics", {}).get("total_duration_seconds", 0),
                },
                importance=0.7,  # Session completions are moderately important
            )
        except Exception as e:
            logger.warning("Failed to record session completion: %s", str(e))

        return {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "awaiting_input": False,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_focus(self, message: str, topic: str) -> str:
        """Extract the focus of the message.

        Args:
            message: Student message.
            topic: Current topic.

        Returns:
            Extracted focus or topic.
        """
        # Simple extraction - could be enhanced with NER
        # For now, return the topic
        return topic
