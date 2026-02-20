# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation workflow using LangGraph.

This workflow manages interactive content creation conversations with
deterministic state transitions for H5P content generation.

Workflow Structure:
    initialize
        ↓
    gather_requirements (orchestrator agent)
        ↓
    wait_for_input [INTERRUPT POINT]
        ↓
    process_input
        ↓
    [conditional: handoff → execute_handoff, generate → agent_generate]
        ↓
    agent_generate (specialized generator agent)
        ↓
    review_content
        ↓
    [conditional: approve → export, modify → agent_generate, reject → gather_requirements]
        ↓
    export_content
        ↓
    check_end
        ↓
    [conditional: continue → gather_requirements, end → end_session]

The workflow uses checkpointing with interrupt_before pattern for
reliable pause/resume during user interactions.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from src.core.config import get_settings
from src.core.orchestration.states.content import (
    ContentCreationState,
    ContentTurn,
    GeneratedContent,
    HandoffContext,
    ReviewResult,
    UserAction,
    UserMessageInference,
    create_initial_content_state,
)
from src.services.h5p import (
    ConverterRegistry,
    H5PClient,
    H5PAPIError,
    H5PConversionError,
    H5PValidationError,
)
from src.services.h5p.schema_loader import H5PSchemaLoader, get_ai_prompt_schema
from src.tools.content.inference import ExtractUserIntentTool
from src.tools.content.media.generate_audio_gemini import GenerateAudioGeminiTool
from src.tools.content.media.generate_image_gemini import GenerateImageGeminiTool
from src.tools.content.media.generate_video_gemini import GenerateVideoGeminiTool
from src.core.tools import ToolContext

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.intelligence.llm import LLMClient
    from src.core.tools import ToolRegistry

logger = logging.getLogger(__name__)


class ContentCreationWorkflow:
    """LangGraph workflow for content creation conversations.

    This workflow orchestrates content creation through multiple agents:
    - Orchestrator: Gathers requirements and routes to generators
    - Quiz Generator: Creates assessment content
    - Vocabulary Generator: Creates vocabulary learning content
    - Game Generator: Creates game-based content
    - Media Generator: Generates images and charts

    The workflow supports:
    - Multi-turn conversations for requirement gathering
    - Agent handoffs for specialized generation
    - Content review and modification
    - Export to H5P format

    Attributes:
        llm_client: LLM client for agent completions.
        tool_registry: Registry of content creation tools.
        checkpointer: Checkpointer for state persistence.
        db_session: Database session for tool execution.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        tool_registry: "ToolRegistry | None" = None,
        checkpointer: BaseCheckpointSaver | None = None,
        db_session: "AsyncSession | None" = None,
    ):
        """Initialize the content creation workflow.

        Args:
            llm_client: LLM client for agent completions.
            tool_registry: Registry of content creation tools.
            checkpointer: Checkpointer for state persistence.
            db_session: Database session for tool execution.
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.checkpointer = checkpointer
        self.db_session = db_session

        # Build the workflow graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph.

        Returns:
            Compiled StateGraph with all nodes and edges.
        """
        graph = StateGraph(ContentCreationState)

        # Add nodes
        graph.add_node("initialize", self._initialize)
        graph.add_node("gather_requirements", self._gather_requirements)
        graph.add_node("wait_for_input", self._wait_for_input)
        graph.add_node("process_input", self._process_input)
        graph.add_node("execute_handoff", self._execute_handoff)
        graph.add_node("agent_generate", self._agent_generate)
        graph.add_node("review_content", self._review_content)
        graph.add_node("export_content", self._export_content)
        graph.add_node("end_session", self._end_session)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges
        graph.add_edge("initialize", "gather_requirements")
        graph.add_edge("gather_requirements", "wait_for_input")
        graph.add_edge("wait_for_input", "process_input")

        # Conditional routing from process_input
        graph.add_conditional_edges(
            "process_input",
            self._route_after_input,
            {
                "handoff": "execute_handoff",
                "generate": "agent_generate",
                "export": "export_content",  # Added: route to export when user approves
                "continue": "wait_for_input",
                "end": "end_session",
            },
        )

        graph.add_edge("execute_handoff", "agent_generate")
        graph.add_edge("agent_generate", "review_content")

        # Conditional routing from review_content
        graph.add_conditional_edges(
            "review_content",
            self._route_after_review,
            {
                "export": "export_content",
                "modify": "agent_generate",
                "restart": "gather_requirements",
                "continue": "wait_for_input",
            },
        )

        graph.add_conditional_edges(
            "export_content",
            self._route_after_export,
            {
                "continue": "gather_requirements",
                "end": "end_session",
            },
        )

        graph.add_edge("end_session", END)

        # Compile with checkpointer and interrupt points
        return graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["wait_for_input"],
        )

    async def run(
        self,
        initial_state: ContentCreationState | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the workflow from initial state.

        Args:
            initial_state: Initial state. Created if not provided.
            thread_id: Thread ID for checkpointing.
            **kwargs: Additional arguments for state creation.

        Returns:
            Workflow result with initial response.
        """
        if initial_state is None:
            initial_state = create_initial_content_state(**kwargs)

        thread_id = thread_id or initial_state.get("session_id", str(uuid4()))

        config = {"configurable": {"thread_id": thread_id}}

        # Run until first interrupt (wait_for_input)
        result = await self._graph.ainvoke(initial_state, config)

        return {
            "session_id": result.get("session_id"),
            "thread_id": thread_id,
            "message": self._get_last_assistant_message(result),
            "workflow_phase": result.get("workflow_phase"),
            "state": result,
        }

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a user message and continue the workflow.

        Uses LangGraph's aupdate_state + ainvoke(None) pattern to properly
        resume from interrupt_before checkpoint.

        Args:
            thread_id: Thread ID for the session.
            message: User message to process.

        Returns:
            Workflow result with assistant response.
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        state = await self._graph.aget_state(config)
        if not state or not state.values:
            raise ValueError(f"No active session found for thread: {thread_id}")

        current_state = state.values

        # Add user message to state
        now = datetime.utcnow().isoformat()
        user_turn = ContentTurn(
            role="user",
            content=message,
            timestamp=now,
        )

        # Prepare state update with user message
        state_update = {
            "conversation_history": [
                *current_state.get("conversation_history", []),
                user_turn,
            ],
            "messages": [{"role": "user", "content": message}],
            "updated_at": now,
        }

        # Update state using LangGraph's aupdate_state (properly handles add_messages reducer)
        await self._graph.aupdate_state(config, state_update)

        # Resume workflow from interrupt point with None input
        result = await self._graph.ainvoke(None, config)

        return {
            "session_id": result.get("session_id"),
            "thread_id": thread_id,
            "message": self._get_last_assistant_message(result),
            "workflow_phase": result.get("workflow_phase"),
            "generated_content": result.get("current_content"),
            "state": result,
        }

    # Node implementations

    async def _initialize(self, state: ContentCreationState) -> ContentCreationState:
        """Initialize the workflow session."""
        logger.info(
            "Initializing content creation workflow: session=%s, language=%s, country=%s, framework=%s",
            state.get("session_id"),
            state.get("language"),
            state.get("country_code"),
            state.get("framework_code"),
        )

        return {
            **state,
            "current_phase": "initialization",
            "requires_input": False,
            "updated_at": datetime.utcnow().isoformat(),
        }

    async def _gather_requirements(self, state: ContentCreationState) -> ContentCreationState:
        """Gather content requirements from user via orchestrator.

        This node uses curriculum tools to fetch available subjects, topics,
        and learning objectives from external sources - never hardcoded.
        """
        logger.debug(
            "Gathering requirements: session=%s, subject=%s, topic=%s, grade=%s",
            state.get("session_id"),
            state.get("subject_code"),
            state.get("topic_code"),
            state.get("grade_level"),
        )

        # Generate initial greeting/prompt from orchestrator
        system_prompt = self._get_orchestrator_system_prompt(state)

        messages = [
            {"role": "system", "content": system_prompt},
            *self._convert_messages_to_dicts(state.get("messages", [])),
        ]

        # If no messages yet, generate welcome message
        if not state.get("conversation_history"):
            welcome = await self._generate_welcome_message(state)
            messages.append({"role": "assistant", "content": welcome})

            return {
                **state,
                "messages": messages,
                "conversation_history": [
                    ContentTurn(
                        role="assistant",
                        content=welcome,
                        timestamp=datetime.utcnow().isoformat(),
                        agent_id="content_creation_orchestrator",
                    )
                ],
                "active_agent": "content_creation_orchestrator",
                "current_phase": "requirements",
                "requires_input": True,
            }

        return state

    async def _wait_for_input(self, state: ContentCreationState) -> ContentCreationState:
        """Wait for user input (interrupt point)."""
        # This node is an interrupt point - workflow pauses here
        return state

    # Category → generator mapping for deterministic handoffs
    CATEGORY_TO_GENERATOR = {
        "assessment": "quiz_generator",
        "vocabulary": "vocabulary_generator",
        "game": "game_content_generator",
        "learning": "learning_content_generator",
        "media": "media_generator",
    }

    async def _process_input(self, state: ContentCreationState) -> ContentCreationState:
        """Process user input with deterministic state transitions.

        Two-layer processing replaces stochastic tool-calling:
        1. _extract_user_intent → content type / media inference (existing)
        2. _classify_user_action → structured JSON classification (new)
        3. Deterministic if/elif sets workflow flags (new)
        4. _generate_response_message → conversational reply (new)
        """
        logger.debug("Processing input: session=%s", state.get("session_id"))

        # Step 1: Extract user intent from message (smart inference - existing)
        user_message = self._get_last_user_message(state)
        user_inference = await self._extract_user_intent(user_message, state)

        logger.debug(
            "User inference: content_type=%s, wants_media=%s, confidence=%.2f",
            user_inference.get("content_type"),
            user_inference.get("wants_media"),
            user_inference.get("confidence", 0),
        )

        # Step 2: Classify user action deterministically
        action = await self._classify_user_action(user_message, state, user_inference)

        logger.info(
            "Classified action: action=%s, content_type=%s, confidence=%.2f, phase=%s",
            action.get("action"),
            action.get("content_type"),
            action.get("confidence", 0),
            state.get("current_phase"),
        )

        # Step 3: Deterministic state transitions based on classified action
        content_type_confirmed = state.get("content_type_confirmed", False)
        include_images = state.get("include_images", False)
        pending_handoff = None
        handoff_context = None
        approved = state.get("approved", False)
        should_end = state.get("should_end", False)

        if user_inference.get("wants_media"):
            include_images = True

        action_type = action.get("action", "unclear")
        # Resolve content type: action > current inference > previous inference
        prev_inference = state.get("user_inference")
        action_content_type = (
            action.get("content_type")
            or user_inference.get("content_type")
            or (prev_inference.get("content_type") if prev_inference else None)
        )

        if action_type == "end":
            should_end = True

        elif action_type == "approve" and state.get("current_phase") == "review":
            approved = True

        elif action_type == "confirm_content_type" and action_content_type:
            content_type_confirmed = True
            # Build deterministic handoff
            pending_handoff, handoff_context = self._build_handoff(
                content_type=action_content_type,
                include_media=include_images or user_inference.get("wants_media", False),
                media_description=user_inference.get("media_description"),
            )

        elif action_type == "request_generation" and content_type_confirmed:
            # User wants to generate — re-trigger handoff with confirmed type
            ct = action_content_type or (
                state.get("handoff_context", {}).get("content_type") if state.get("handoff_context") else None
            )
            if ct:
                pending_handoff, handoff_context = self._build_handoff(
                    content_type=ct,
                    include_media=include_images or user_inference.get("wants_media", False),
                    media_description=user_inference.get("media_description"),
                )

        elif action_type == "modify" and state.get("current_phase") == "review":
            # Modification: re-trigger generation with modification detail
            ct = (
                state.get("current_content", {}).get("content_type")
                if state.get("current_content")
                else action_content_type
            )
            if ct:
                mod_detail = action.get("modification_detail", "")
                pending_handoff, handoff_context = self._build_handoff(
                    content_type=ct,
                    include_media=include_images,
                    media_description=user_inference.get("media_description"),
                    extra_prompt=f"Modify the previous content: {mod_detail}" if mod_detail else None,
                )

        # action_type == "provide_info" or "unclear" → no flag changes, just respond

        # Step 4: Generate conversational response
        assistant_message = await self._generate_response_message(
            state, action, user_inference,
        )

        # Update conversation history
        conversation_history = list(state.get("conversation_history", []))
        if assistant_message:
            conversation_history.append(
                ContentTurn(
                    role="assistant",
                    content=assistant_message,
                    timestamp=datetime.utcnow().isoformat(),
                    agent_id="content_creation_orchestrator",
                )
            )

        return {
            **state,
            "messages": [*state.get("messages", []), {"role": "assistant", "content": assistant_message}] if assistant_message else state.get("messages", []),
            "conversation_history": conversation_history,
            "pending_handoff": pending_handoff,
            "handoff_context": handoff_context,
            "user_inference": user_inference,
            "content_type_confirmed": content_type_confirmed,
            "include_images": include_images,
            "approved": approved,
            "should_end": should_end,
            "updated_at": datetime.utcnow().isoformat(),
        }

    async def _classify_user_action(
        self,
        user_message: str,
        state: ContentCreationState,
        user_inference: UserMessageInference,
    ) -> UserAction:
        """Classify user message into a discrete action using structured JSON output.

        Single LLM call with response_format=json_object and temperature=0.1
        for deterministic classification. The prompt is phase-aware so only
        valid actions are considered for the current workflow phase.

        Args:
            user_message: The user's message.
            state: Current workflow state.
            user_inference: Extracted intent from user's message.

        Returns:
            UserAction with classified action and metadata.
        """
        current_phase = state.get("current_phase", "requirements")
        content_type_confirmed = state.get("content_type_confirmed", False)
        # Use current inference, or fall back to previously stored inference
        inferred_type = user_inference.get("content_type")
        prev_inference = state.get("user_inference")
        prev_inferred_type = prev_inference.get("content_type") if prev_inference else None
        effective_inferred_type = inferred_type or prev_inferred_type
        has_content = state.get("current_content") is not None

        # Build phase-specific valid actions
        if current_phase == "review" and has_content:
            valid_actions = '"approve", "modify", "end", "provide_info", "unclear"'
            phase_context = (
                "The user is reviewing generated content. "
                "'approve' = user accepts the content for export. "
                "'modify' = user wants changes to the generated content. "
                "'end' = user wants to stop the session."
            )
        elif current_phase == "requirements" and content_type_confirmed:
            valid_actions = '"request_generation", "provide_info", "end", "unclear"'
            phase_context = (
                "Content type is already confirmed. "
                "'request_generation' = user wants to proceed with generation. "
                "'provide_info' = user is providing additional requirements."
            )
        elif current_phase == "requirements" and effective_inferred_type:
            valid_actions = '"confirm_content_type", "provide_info", "end", "unclear"'
            phase_context = (
                f"Content type '{effective_inferred_type}' was inferred. "
                "'confirm_content_type' = user confirms or agrees to create this content type. "
                "If user says 'yes', 'ok', 'generate', 'go ahead', etc. this counts as confirm. "
                "'provide_info' = user is providing info but NOT confirming a type yet."
            )
        else:
            valid_actions = '"confirm_content_type", "provide_info", "end", "unclear"'
            phase_context = (
                "We are gathering requirements. "
                "'confirm_content_type' = user explicitly states what content type to create. "
                "'provide_info' = user is providing requirements or context."
            )

        # Recent conversation for context
        recent_turns = state.get("conversation_history", [])[-4:]
        conversation_context = "\n".join(
            f"{t.get('role', 'unknown')}: {t.get('content', '')[:200]}"
            for t in recent_turns
        )

        classification_prompt = f"""Classify the user's message into exactly one action.

CURRENT PHASE: {current_phase}
CONTENT TYPE CONFIRMED: {content_type_confirmed}
INFERRED CONTENT TYPE: {effective_inferred_type or "none"}
HAS GENERATED CONTENT: {has_content}

PHASE CONTEXT: {phase_context}

VALID ACTIONS: {valid_actions}

RECENT CONVERSATION:
{conversation_context}

USER MESSAGE: {user_message}

Return ONLY a JSON object with these fields:
- "action": one of [{valid_actions}]
- "content_type": string or null (the H5P content type if relevant)
- "modification_detail": string or null (what to change, only for "modify")
- "confidence": float between 0.0 and 1.0"""

        try:
            import json

            response = await self.llm_client.complete_with_messages(
                messages=[{"role": "system", "content": classification_prompt}],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.content)
            # Normalize content_type: LLM may return underscores (mark_the_words)
            # but config/converters use dashes (mark-words)
            raw_ct = result.get("content_type")
            if raw_ct:
                # Strip H5P library prefix if present (e.g. "H5P.Summary 1.10" → "summary")
                if raw_ct.startswith("H5P."):
                    raw_ct = raw_ct.split(" ")[0].replace("H5P.", "")
                raw_ct = raw_ct.replace("_", "-").lower()
                # Also normalize common LLM variants
                ct_aliases = {
                    "mark-the-words": "mark-words",
                    "markthewords": "mark-words",
                    "drag-the-words": "drag-words",
                    "dragtext": "drag-words",
                    "fill-the-blanks": "fill-blanks",
                    "fill-in-the-blanks": "fill-blanks",
                    "blanks": "fill-blanks",
                    "multiple-choice-question": "multiple-choice",
                    "multichoice": "multiple-choice",
                    "truefalse": "true-false",
                    "singlechoiceset": "single-choice-set",
                }
                raw_ct = ct_aliases.get(raw_ct, raw_ct)
            return UserAction(
                action=result.get("action", "unclear"),
                content_type=raw_ct,
                modification_detail=result.get("modification_detail"),
                confidence=float(result.get("confidence", 0.5)),
            )
        except Exception as e:
            logger.warning("Action classification failed, defaulting to 'unclear': %s", e)
            return UserAction(
                action="unclear",
                content_type=None,
                modification_detail=None,
                confidence=0.0,
            )

    async def _generate_response_message(
        self,
        state: ContentCreationState,
        action: UserAction,
        user_inference: UserMessageInference,
    ) -> str:
        """Generate a natural language response based on the classified action.

        Replaces complete_with_tools for response generation. The LLM knows
        what action is about to happen and generates an appropriate message.

        Args:
            state: Current workflow state.
            action: The classified user action.
            user_inference: Extracted intent from user's message.

        Returns:
            Assistant message string.
        """
        action_type = action.get("action", "unclear")
        content_type = action.get("content_type") or user_inference.get("content_type")
        language = state.get("language") or "en"
        current_phase = state.get("current_phase", "requirements")

        # Build action-specific context for the response LLM
        if action_type == "confirm_content_type" and content_type:
            action_context = (
                f"You confirmed the content type as '{content_type}' and are about to "
                f"start generating it. Tell the user you'll begin generating the content now."
            )
        elif action_type == "request_generation":
            action_context = "The user wants to generate content. Tell them generation is starting."
        elif action_type == "approve":
            action_context = "The user approved the content. Tell them you're exporting it now."
        elif action_type == "modify":
            detail = action.get("modification_detail", "")
            action_context = (
                f"The user wants to modify the content: '{detail}'. "
                f"Acknowledge the change and tell them you're regenerating."
            )
        elif action_type == "end":
            action_context = "The user wants to end the session. Say goodbye politely."
        elif action_type == "provide_info":
            action_context = (
                "The user provided additional information or requirements. "
                "Acknowledge what they said and ask clarifying questions if needed, "
                "or suggest a content type if you can infer one."
            )
        else:  # unclear
            action_context = (
                "The user's intent is unclear. Ask them to clarify what they'd like to do. "
                "Suggest options based on the current phase."
            )

        # Curriculum context
        context = {
            "language": language,
            "subject": state.get("subject_name") or state.get("subject_code"),
            "topic": state.get("topic_name") or state.get("topic_code"),
            "grade_level": state.get("grade_level"),
            "current_phase": current_phase,
        }
        context = {k: v for k, v in context.items() if v}

        system_prompt = f"""You are a content creation assistant for H5P educational content.
RESPOND IN THE USER'S LANGUAGE. Language hint: {language}

Context: {context}

WHAT IS HAPPENING: {action_context}

Generate a brief, natural response. Do NOT use any tool syntax or function calls.
Keep it concise (1-3 sentences)."""

        messages = [
            {"role": "system", "content": system_prompt},
            *self._convert_messages_to_dicts(state.get("messages", [])[-6:]),
        ]

        try:
            response = await self.llm_client.complete_with_messages(messages=messages)
            return response.content or "..."
        except Exception as e:
            logger.warning("Response generation failed: %s", e)
            return "..."

    def _build_handoff(
        self,
        content_type: str,
        include_media: bool = False,
        media_description: str | None = None,
        extra_prompt: str | None = None,
    ) -> tuple[str, HandoffContext]:
        """Build deterministic handoff context from content type.

        Uses H5PSchemaLoader to map content type → category → generator.

        Args:
            content_type: H5P content type identifier.
            include_media: Whether to include media generation.
            media_description: User's media description.
            extra_prompt: Additional prompt text (e.g., modification instructions).

        Returns:
            Tuple of (target_agent, HandoffContext).
        """
        schema_loader = H5PSchemaLoader()
        schema = schema_loader.get_schema(content_type)
        category = schema.get("category", "assessment") if schema else "assessment"
        target_agent = self.CATEGORY_TO_GENERATOR.get(category, "quiz_generator")

        generation_prompt = extra_prompt or f"Generate {content_type} content based on the conversation context."

        handoff_context = HandoffContext(
            source_agent="content_creation_orchestrator",
            target_agent=target_agent,
            task_type="generate",
            content_type=content_type,
            generation_prompt=generation_prompt,
            additional_data={
                "content_type": content_type,
                "generation_prompt": generation_prompt,
                "include_media": include_media,
                "media_description": media_description,
            },
        )

        return target_agent, handoff_context

    async def _execute_handoff(self, state: ContentCreationState) -> ContentCreationState:
        """Execute handoff to specialized generator agent."""
        handoff_context = state.get("handoff_context")
        if not handoff_context:
            logger.warning("No handoff context found")
            return state

        target_agent = handoff_context.get("target_agent", "")
        logger.info(
            "Executing handoff: session=%s, target=%s",
            state.get("session_id"),
            target_agent,
        )

        return {
            **state,
            "active_agent": target_agent,
            "current_phase": "generation",
            "pending_handoff": None,  # Clear pending handoff
        }

    async def _agent_generate(self, state: ContentCreationState) -> ContentCreationState:
        """Run specialized generator agent to create content.

        Uses curriculum context (subject, topic, grade, framework) from state
        which was populated via curriculum tools - never hardcoded.
        """
        active_agent = state.get("active_agent", "")
        handoff_context = state.get("handoff_context")

        logger.info(
            "Agent generating: session=%s, agent=%s, subject=%s, topic=%s",
            state.get("session_id"),
            active_agent,
            state.get("subject_code"),
            state.get("topic_code"),
        )

        # Get generator system prompt with curriculum context
        system_prompt = self._get_generator_system_prompt(active_agent, handoff_context, state)

        # Build generation prompt with conversation context
        generation_prompt = handoff_context.get("generation_prompt", "") if handoff_context else ""

        # Include conversation history so generator knows what user actually asked for
        conversation_history = state.get("conversation_history", [])
        conversation_summary = ""
        if conversation_history:
            user_messages = [
                turn.get("content", "") if isinstance(turn, dict) else getattr(turn, "content", "")
                for turn in conversation_history
                if (turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", "")) == "user"
            ]
            if user_messages:
                conversation_summary = "\n\nUser's conversation (generate content based on these requests):\n" + "\n".join(
                    f"- {msg}" for msg in user_messages
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generation_prompt + conversation_summary},
        ]

        # Call LLM to generate content
        response = await self.llm_client.complete_with_messages(
            messages=messages,
        )

        # Parse generated content (extract JSON from markdown code blocks if present)
        try:
            import json
            import re
            content_text = response.content
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_text)
            if json_match:
                content_text = json_match.group(1).strip()
            ai_content = json.loads(content_text)
        except (json.JSONDecodeError, AttributeError):
            ai_content = {"raw_response": response.content}

        # If LLM returned a bare list, wrap it in a dict with the correct key
        # for the content type. This happens when the schema example was a list.
        if isinstance(ai_content, list):
            content_type_key = handoff_context.get("content_type", "") if handoff_context else ""
            list_key_map = {
                "flashcards": "cards",
                "crossword": "words",
                "dialog-cards": "dialogs",
                "accordion": "panels",
                "find-the-words": "words",
                "word-search": "words",
                "memory-game": "pairs",
                "image-pairing": "pairs",
                "image-sequencing": "images",
                "sort-paragraphs": "paragraphs",
            }
            key = list_key_map.get(content_type_key, "items")
            ai_content = {key: ai_content}
            logger.warning(
                "LLM returned bare list for %s, wrapped under '%s' key",
                content_type_key, key,
            )

        # Create GeneratedContent record
        generated = GeneratedContent(
            id=str(uuid4()),
            content_type=handoff_context.get("content_type", "") if handoff_context else "",
            title=ai_content.get("title", "Generated Content"),
            ai_content=ai_content,
            h5p_params=None,  # Will be converted on export
            status="draft",
            generated_by=active_agent,
            quality_score=None,
            created_at=datetime.utcnow().isoformat(),
        )

        # Add to generated contents
        generated_contents = list(state.get("generated_contents", []))
        generated_contents.append(generated)

        return {
            **state,
            "generated_contents": generated_contents,
            "current_content": generated,
            "pending_content": generated,
            "current_phase": "review",
        }

    async def _review_content(self, state: ContentCreationState) -> ContentCreationState:
        """Review generated content and get user feedback."""
        current_content = state.get("current_content")
        if not current_content:
            return {
                **state,
                "current_phase": "review",
                "approved": True,
            }

        # Format content for user review (LLM generates in user's language)
        review_message = await self._format_content_for_review(current_content, state)

        # Add review message to conversation
        conversation_history = list(state.get("conversation_history", []))
        conversation_history.append(
            ContentTurn(
                role="assistant",
                content=review_message,
                timestamp=datetime.utcnow().isoformat(),
                agent_id="content_creation_orchestrator",
            )
        )

        return {
            **state,
            "messages": [*state.get("messages", []), {"role": "assistant", "content": review_message}],
            "conversation_history": conversation_history,
            "current_phase": "review",
            "requires_input": True,
        }

    async def _export_content(self, state: ContentCreationState) -> ContentCreationState:
        """Export approved content to H5P format.

        Uses ConverterRegistry to convert AI content to H5P params format,
        then creates content via H5P API (Creatiq).

        The function:
        1. Validates ai_content is in proper format (not raw_response fallback)
        2. Uses safe_convert() for validation and conversion
        3. Creates H5P content via API
        4. Updates content with h5p_id, preview_url, h5p_params, exported_at
        """
        current_content = state.get("current_content")
        if not current_content:
            return state

        content_type = current_content.get("content_type", "")
        ai_content = current_content.get("ai_content", {})
        title = current_content.get("title", "Untitled")
        language = state.get("language") or "en"

        logger.info(
            "Exporting content: session=%s, content_id=%s, type=%s",
            state.get("session_id"),
            current_content.get("id"),
            content_type,
        )

        # Validate ai_content is not in fallback format (raw_response)
        if "raw_response" in ai_content and len(ai_content) == 1:
            error_message = (
                "AI content is in raw format and could not be parsed. "
                "The LLM response may not have been valid JSON."
            )
            logger.error(
                "Cannot export content with raw_response format: session=%s, content_id=%s",
                state.get("session_id"),
                current_content.get("id"),
            )
            current_content["status"] = "export_failed"
            return await self._build_export_error_response(
                state, current_content, error_message, language, title
            )

        # Auto-wrap: if a standalone assessment type produced multiple items,
        # wrap them into a QuestionSet so they export as a multi-question quiz
        content_type, ai_content = self._auto_wrap_to_question_set(
            content_type, ai_content, title
        )
        # Update current_content so downstream code sees the wrapped version
        current_content["content_type"] = content_type
        current_content["ai_content"] = ai_content

        # Get converter for content type
        registry = ConverterRegistry()
        converter = registry.get(content_type)

        export_success = False
        h5p_id = None
        preview_url = None
        h5p_params = None
        error_message = None

        if not converter:
            error_message = f"No converter found for content type: {content_type}"
            logger.warning(error_message)
            current_content["status"] = "export_failed"
            return await self._build_export_error_response(
                state, current_content, error_message, language, title
            )

        try:
            # Process media prompt fields: generate images/video/audio and replace with URLs
            include_images = state.get("include_images", False)
            if include_images:
                ai_content = await self._process_media_prompts(
                    ai_content=ai_content,
                    state=state,
                )

            # Use safe_convert for validation and conversion
            h5p_params = converter.safe_convert(ai_content, language)

            # Create H5P content via API
            settings = get_settings()
            h5p_client = H5PClient(
                api_url=settings.h5p.api_url,
                api_key=settings.h5p.api_key.get_secret_value(),
                timeout=settings.h5p.timeout,
            )

            h5p_id = await h5p_client.create_content(
                library=converter.library,
                params=h5p_params,
                metadata={"title": title, "language": language},
                tenant_code=state.get("tenant_code"),
            )

            # Attach generated media files to content directory
            if include_images and h5p_id:
                # Collect media filenames by type from all item arrays
                media_by_type: dict[str, list[str]] = {
                    "images": [], "videos": [], "audios": [],
                }
                all_items = (
                    ai_content.get("questions", [])
                    + ai_content.get("statements", [])
                    + ai_content.get("exercises", [])
                )
                for q in all_items:
                    for prefix, media_type in [("images/", "images"), ("videos/", "videos"), ("audios/", "audios")]:
                        for url_field in ["image_url", "video_url", "audio_url"]:
                            url = q.get(url_field, "")
                            if not url and "params" in q:
                                url = q.get("params", {}).get(url_field, "")
                            if url.startswith(prefix):
                                media_by_type[media_type].append(url.replace(prefix, "", 1))

                for media_type, filenames in media_by_type.items():
                    if filenames:
                        await h5p_client.attach_media(
                            content_id=str(h5p_id),
                            filenames=filenames,
                            media_type=media_type,
                        )

            preview_url = h5p_client.get_preview_url(h5p_id)
            export_success = True

            logger.info(
                "H5P content created: h5p_id=%s, type=%s, library=%s",
                h5p_id,
                content_type,
                converter.library,
            )

        except H5PValidationError as e:
            logger.error("H5P validation error during export: %s", e)
            error_message = f"Content validation failed: {e.message}"
        except H5PConversionError as e:
            logger.error("H5P conversion error during export: %s", e)
            error_message = f"Content conversion failed: {e.message}"
        except H5PAPIError as e:
            logger.error("H5P API error during export: %s", e)
            error_message = f"H5P API error: {e.message}"
        except Exception as e:
            logger.exception("Unexpected error during H5P export")
            error_message = f"Unexpected error: {str(e)}"

        # Update content status and metadata
        now = datetime.utcnow().isoformat()
        current_content["status"] = "exported" if export_success else "export_failed"
        if export_success:
            current_content["h5p_id"] = h5p_id
            current_content["preview_url"] = preview_url
            current_content["h5p_params"] = h5p_params
            current_content["exported_at"] = now

        exported_ids = list(state.get("exported_content_ids", []))
        if export_success and h5p_id:
            exported_ids.append(h5p_id)

        # LLM generates export message in user's language
        if export_success:
            prompt_content = f"""Generate a brief export success message.
Language: {language}
Content title: {title}
Preview URL: {preview_url}

Include:
1. Confirmation that content was exported successfully
2. The preview URL
3. Ask if they want to create more content

Keep it concise."""
        else:
            prompt_content = f"""Generate a brief export failure message.
Language: {language}
Content title: {title}
Error: {error_message}

Include:
1. Apologize for the export failure
2. Suggest trying again or modifying the content
3. Ask what they'd like to do next

Keep it concise and helpful."""

        messages = [{"role": "system", "content": prompt_content}]
        response = await self.llm_client.complete_with_messages(messages=messages)
        export_message = response.content or f"Content '{title}' {'exported' if export_success else 'export failed'}."

        conversation_history = list(state.get("conversation_history", []))
        conversation_history.append(
            ContentTurn(
                role="assistant",
                content=export_message,
                timestamp=datetime.utcnow().isoformat(),
                agent_id="content_creation_orchestrator",
            )
        )

        return {
            **state,
            "messages": [*state.get("messages", []), {"role": "assistant", "content": export_message}],
            "conversation_history": conversation_history,
            "exported_content_ids": exported_ids,
            "current_content": current_content,
            "current_phase": "complete" if export_success else "review",
            "requires_input": True,
            "approved": False if not export_success else state.get("approved"),
        }

    async def _process_media_prompts(
        self,
        ai_content: dict[str, Any],
        state: ContentCreationState,
    ) -> dict[str, Any]:
        """Extract media prompt fields and generate media (images, audio, video).

        Walks through ai_content arrays looking for 'image_prompt', 'video_prompt',
        and 'audio_prompt' fields. Generates media via respective tools and replaces
        prompt fields with URL fields.

        Args:
            ai_content: The AI-generated content dict.
            state: Current workflow state (for tenant_code, grade_level).

        Returns:
            Updated ai_content with prompts replaced by URLs.
        """
        from uuid import UUID

        image_tool = GenerateImageGeminiTool()
        audio_tool = GenerateAudioGeminiTool()
        video_tool = GenerateVideoGeminiTool()
        tenant_code = state.get("tenant_code", "default")
        grade_level = state.get("grade_level", 5)
        tool_context = ToolContext(
            tenant_code=tenant_code,
            user_id=UUID(state.get("user_id", "00000000-0000-0000-0000-000000000000")),
            user_type=state.get("user_type", "teacher"),
            grade_level=grade_level,
            language=state.get("language", "en"),
        )

        # Media types: (prompt_field, url_field, alt_field, tool, tool_params_builder)
        media_specs = [
            ("image_prompt", "image_url", "image_alt", image_tool, lambda p: {
                "prompt": p, "style": "illustration", "purpose": "quiz illustration",
                "grade_level": grade_level,
            }),
            ("video_prompt", "video_url", "video_alt", video_tool, lambda p: {
                "prompt": p, "grade_level": grade_level,
            }),
            ("audio_prompt", "audio_url", "audio_alt", audio_tool, lambda p: {
                "text": p, "language": state.get("language", "en"),
            }),
        ]

        # Process all item arrays (questions, statements, exercises)
        all_items = (
            ai_content.get("questions", [])
            + ai_content.get("statements", [])
            + ai_content.get("exercises", [])
        )

        for i, question in enumerate(all_items):
            for prompt_field, url_field, alt_field, tool, params_builder in media_specs:
                # Support flat and nested (QuestionSet) formats
                nested = False
                prompt = question.get(prompt_field)
                if not prompt and "params" in question:
                    prompt = question.get("params", {}).get(prompt_field)
                    if prompt:
                        nested = True
                if not prompt:
                    continue

                logger.info(
                    "Generating %s for question %d: prompt=%s...",
                    prompt_field.replace("_prompt", ""),
                    i + 1,
                    prompt[:80],
                )

                try:
                    result = await tool.execute(
                        params=params_builder(prompt),
                        context=tool_context,
                    )

                    if result.success and result.data:
                        media_url = result.data.get(url_field, "")
                        alt_text = result.data.get("alt_text", prompt[:100])
                        target = question["params"] if nested else question
                        target[url_field] = media_url
                        target[alt_field] = alt_text
                        logger.info(
                            "%s generated for question %d: url=%s",
                            prompt_field.replace("_prompt", "").title(),
                            i + 1,
                            media_url,
                        )
                    else:
                        logger.warning(
                            "%s generation failed for question %d: %s",
                            prompt_field.replace("_prompt", "").title(),
                            i + 1,
                            result.error or "unknown error",
                        )
                except Exception as e:
                    logger.warning(
                        "%s generation error for question %d: %s",
                        prompt_field.replace("_prompt", "").title(),
                        i + 1,
                        str(e),
                    )

                # Remove prompt regardless of success
                if nested:
                    question.get("params", {}).pop(prompt_field, None)
                else:
                    question.pop(prompt_field, None)

        return ai_content

    @staticmethod
    def _auto_wrap_to_question_set(
        content_type: str,
        ai_content: dict[str, Any],
        title: str,
    ) -> tuple[str, dict[str, Any]]:
        """Auto-wrap standalone assessment content into QuestionSet when multiple items exist.

        If a standalone assessment type (MC, TF, Fill Blanks, Drag Words, Mark Words)
        produced 2+ items, convert the AI content to QuestionSet format so all items
        are exported as a multi-question quiz instead of losing all but the first.

        Returns:
            Tuple of (content_type, ai_content) — possibly transformed.
        """
        # Mapping: content_type -> (array_field, question_set_type)
        WRAP_MAP = {
            "multiple-choice": ("questions", "multiple_choice"),
            "true-false": ("statements", "true_false"),
            "fill-blanks": ("exercises", "fill_blanks"),
            "drag-words": ("exercises", "drag_words"),
            "mark-words": ("exercises", "mark_words"),
        }

        if content_type not in WRAP_MAP:
            return content_type, ai_content

        array_field, qs_type = WRAP_MAP[content_type]
        items = ai_content.get(array_field, [])

        if len(items) <= 1:
            return content_type, ai_content

        # Transform each item into QuestionSet sub-question format
        # Carry top-level instruction into each item so converters can use it
        top_instruction = ai_content.get("instruction")
        qs_questions = []
        for item in items:
            params = dict(item)
            if top_instruction and "instruction" not in params:
                params["instruction"] = top_instruction
            qs_questions.append({
                "type": qs_type,
                "params": params,
            })

        wrapped_content = {
            "title": ai_content.get("title", title),
            "introPage": {
                "showIntroPage": False,
                "title": "",
                "introduction": "",
            },
            "questions": qs_questions,
            "passPercentage": 50,
            "randomQuestions": False,
        }

        logger.info(
            "Auto-wrapped %d %s items into QuestionSet",
            len(items),
            content_type,
        )

        return "question-set", wrapped_content

    async def _build_export_error_response(
        self,
        state: ContentCreationState,
        current_content: GeneratedContent,
        error_message: str,
        language: str,
        title: str,
    ) -> ContentCreationState:
        """Build response state for export failure with early return.

        This helper generates an error message via LLM and returns
        the updated state for export failures that occur before
        the main export try/except block.

        Args:
            state: Current workflow state.
            current_content: The content that failed to export.
            error_message: Description of the error.
            language: Language for error message.
            title: Content title.

        Returns:
            Updated state with error message and review phase.
        """
        prompt_content = f"""Generate a brief export failure message.
Language: {language}
Content title: {title}
Error: {error_message}

Include:
1. Apologize for the export failure
2. Suggest trying again or modifying the content
3. Ask what they'd like to do next

Keep it concise and helpful."""

        messages = [{"role": "system", "content": prompt_content}]
        response = await self.llm_client.complete_with_messages(messages=messages)
        export_message = response.content or f"Content '{title}' export failed: {error_message}"

        conversation_history = list(state.get("conversation_history", []))
        conversation_history.append(
            ContentTurn(
                role="assistant",
                content=export_message,
                timestamp=datetime.utcnow().isoformat(),
                agent_id="content_creation_orchestrator",
            )
        )

        return {
            **state,
            "messages": [*state.get("messages", []), {"role": "assistant", "content": export_message}],
            "conversation_history": conversation_history,
            "current_content": current_content,
            "current_phase": "review",
            "requires_input": True,
            "approved": False,
        }

    async def _end_session(self, state: ContentCreationState) -> ContentCreationState:
        """End the workflow session."""
        logger.info(
            "Ending content creation session: session=%s, generated=%d, exports=%d",
            state.get("session_id"),
            len(state.get("generated_contents", [])),
            len(state.get("exported_content_ids", [])),
        )

        return {
            **state,
            "should_end": True,
            "current_phase": "complete",
            "requires_input": False,
        }

    # Routing functions

    def _route_after_input(self, state: ContentCreationState) -> str:
        """Route after processing user input.

        Routing is determined by deterministic flags set from classified action.
        """
        if state.get("should_end"):
            return "end"
        # Check if user approved content during review phase
        if state.get("approved") and state.get("current_phase") == "review":
            return "export"
        if state.get("pending_handoff"):
            return "handoff"
        if self._requirements_complete(state):
            return "generate"
        return "continue"

    def _route_after_review(self, state: ContentCreationState) -> str:
        """Route after content review."""
        if state.get("approved"):
            return "export"

        review_result = state.get("review_result")
        if review_result:
            if review_result.get("approved"):
                return "export"
            if review_result.get("critical_issues"):
                return "modify"

        # Default: wait for user decision (LLM will set flags)
        return "continue"

    def _route_after_export(self, state: ContentCreationState) -> str:
        """Route after content export."""
        if state.get("should_end"):
            return "end"
        # Default to continue - LLM will set should_end if user wants to stop
        return "continue"

    def _requirements_complete(self, state: ContentCreationState) -> bool:
        """Check if we have enough requirements to generate content."""
        has_topic = bool(state.get("topic_code") or state.get("topic_name"))
        has_types = bool(state.get("content_types"))
        return has_topic and has_types

    # Helper methods

    def _get_orchestrator_system_prompt(
        self,
        state: ContentCreationState,
        user_inference: UserMessageInference | None = None,
    ) -> str:
        """Get system prompt for orchestrator agent.

        System prompt is language-agnostic. LLM responds in user's language.
        Curriculum context comes from state (populated via curriculum tools).
        User inference provides smart extraction from user's message.

        Args:
            state: Current workflow state.
            user_inference: Extracted intent from user's message (optional).
        """
        # Build context from external sources (never hardcoded)
        context = {
            "user_role": state.get("user_role"),
            "language": state.get("language"),
            "country_code": state.get("country_code"),
            "framework_code": state.get("framework_code"),
            "subject": state.get("subject_name") or state.get("subject_code"),
            "topic": state.get("topic_name") or state.get("topic_code"),
            "grade_level": state.get("grade_level"),
        }
        # Filter out None values
        context = {k: v for k, v in context.items() if v is not None}

        # Build inference context if available
        inference_section = ""
        if user_inference:
            inferred_type = user_inference.get("content_type")
            wants_media = user_inference.get("wants_media", False)
            media_desc = user_inference.get("media_description")
            confidence = user_inference.get("confidence", 0)

            if inferred_type or wants_media:
                inference_section = f"""
USER INTENT INFERENCE (extracted from their message):
- Inferred content type: {inferred_type or "not detected"}
- Wants media/images: {wants_media}
- Media description: {media_desc or "none"}
- Confidence: {confidence:.0%}

IMPORTANT INSTRUCTIONS based on inference:
"""
                if inferred_type and not state.get("content_type_confirmed"):
                    inference_section += f"""- Content type "{inferred_type}" was inferred from user's message
- Confirm with user: Ask if they want to create {inferred_type} content
- If user confirms, proceed with handoff to appropriate generator
- If user wants a different type, ask what they prefer
"""
                if wants_media:
                    inference_section += f"""- User requested images/media in their content
- Include this requirement when handing off to generator
- Media description: "{media_desc or 'user wants images'}"
"""

        return f"""You are a content creation orchestrator for H5P educational content.

RESPOND IN THE USER'S LANGUAGE (detected from messages or state.language).

Current Context: {context or "None - gather from user"}
{inference_section}
Tasks:
1. Detect user's language from their messages
2. Understand content requirements
3. If content type is inferred, confirm with user before proceeding
4. Recommend appropriate H5P content types

Content Categories:
- Assessment: multiple_choice, true_false, fill_blanks, drag_words, question_set
- Vocabulary: flashcards, dialog_cards, crossword, word_search
- Game: memory_game, timeline, image_pairing, image_sequencing
- Learning: course_presentation, interactive_book, branching_scenario
- Media: chart, image_hotspots

Quality Rules:
- Respond naturally in user's language
- Keep responses concise"""

    def _get_generator_system_prompt(
        self,
        agent_id: str,
        handoff_context: HandoffContext | None,
        state: ContentCreationState,
    ) -> str:
        """Get system prompt for generator agent.

        Uses curriculum context from state (populated via curriculum tools).
        Includes content-type specific JSON schema to ensure LLM generates correct format.
        Handles media requirements based on content type and user request.
        """
        # Build curriculum context from state
        context = {
            "language": state.get("language"),
            "subject": state.get("subject_name") or state.get("subject_code"),
            "topic": state.get("topic_name") or state.get("topic_code"),
            "grade_level": state.get("grade_level"),
            "difficulty": state.get("difficulty"),
            "learning_objectives": state.get("learning_objectives", [])[:3],
            "content_type": handoff_context.get("content_type") if handoff_context else None,
        }
        context = {k: v for k, v in context.items() if v}

        # Get content-type specific schema from config files
        content_type = handoff_context.get("content_type", "") if handoff_context else ""
        schema_info = get_ai_prompt_schema(content_type)

        # Check media requirements
        media_section = ""
        if handoff_context:
            additional_data = handoff_context.get("additional_data", {})
            include_media = additional_data.get("include_media", False)
            media_description = additional_data.get("media_description")

            # Check if content type supports/requires media
            schema_loader = H5PSchemaLoader()
            requires_media = schema_loader.requires_media(content_type)
            supports_media = schema_loader.supports_media(content_type)

            if requires_media:
                media_section = """
MEDIA REQUIREMENTS:
- This content type REQUIRES media to function
- You MUST include media prompt fields in your content
- Available media prompt fields (use as appropriate):
  - "image_prompt": "detailed description of an image to generate"
  - "video_prompt": "description of a short video clip to generate"
  - "audio_prompt": "text to be spoken aloud as audio narration"
- Media will be generated after content creation
"""
            elif supports_media and include_media:
                media_section = f"""
MEDIA REQUIREMENTS:
- User requested media for this content
- User's media description: "{media_description or 'include relevant media'}"
- Available media prompt fields (include whichever the user asked for):
  - "image_prompt": "description of an image to generate" (for images)
  - "video_prompt": "description of a short video clip to generate" (for video)
  - "audio_prompt": "text to be spoken aloud as audio narration" (for audio)
- Use the media type that matches what the user asked for
- If user asked for video, use "video_prompt" NOT "image_prompt"
- If user asked for audio, use "audio_prompt" NOT "image_prompt"
- Media will be generated after content creation
"""

        return f"""You are a specialized content generator: {agent_id}

Generate educational content in the language specified in context.
You MUST return valid JSON that matches the exact schema for the content type.

Context: {context}
{media_section}
{schema_info}

Requirements:
- Generate content appropriate for the grade level
- Align with learning objectives if provided
- Match the specified difficulty level
- Use culturally appropriate examples for the country/language
- Return ONLY valid JSON - no markdown, no code blocks, just the JSON object"""

    async def _generate_welcome_message(self, state: ContentCreationState) -> str:
        """Generate welcome message for new session using LLM.

        LLM generates appropriate greeting in user's language.
        No hardcoded language-specific strings.
        """
        context = {
            "user_role": state.get("user_role"),
            "language": state.get("language"),
            "subject": state.get("subject_name") or state.get("subject_code"),
            "grade_level": state.get("grade_level"),
        }
        context = {k: v for k, v in context.items() if v}

        # LLM generates welcome in appropriate language
        messages = [
            {
                "role": "system",
                "content": f"""Generate a brief welcome message for a content creation assistant.
Context: {context}
Language: {state.get('language') or 'detect from context or use English'}

Include:
- Brief greeting
- What you can help with (H5P educational content)
- Ask what they'd like to create

Keep it concise. No emojis unless user's culture commonly uses them."""
            }
        ]

        response = await self.llm_client.complete_with_messages(messages=messages)
        return response.content or "Hello! How can I help you create educational content today?"

    async def _format_content_for_review(self, content: GeneratedContent, state: ContentCreationState) -> str:
        """Format generated content for user review using LLM.

        LLM generates review summary in user's language.
        No hardcoded language-specific strings.
        """
        title = content.get("title", "Content")
        content_type = content.get("content_type", "unknown")
        ai_content = content.get("ai_content", {})

        # Build content summary
        summary_data = {
            "title": title,
            "content_type": content_type,
        }

        if "questions" in ai_content:
            questions = ai_content["questions"]
            summary_data["item_count"] = len(questions)
            summary_data["item_type"] = "questions"
            summary_data["samples"] = [
                q.get("question", q.get("text", ""))[:100]
                for q in questions[:3]
            ]
        elif "cards" in ai_content:
            cards = ai_content["cards"]
            summary_data["item_count"] = len(cards)
            summary_data["item_type"] = "cards"
            summary_data["samples"] = [
                c.get("term", c.get("front", ""))[:50]
                for c in cards[:3]
            ]
        elif "words" in ai_content:
            words = ai_content["words"]
            if words and isinstance(words[0], dict):
                summary_data["samples"] = [w.get("word", "") for w in words[:5]]
            elif words:
                summary_data["samples"] = words[:5]
            summary_data["item_type"] = "words"

        # LLM generates review message in appropriate language
        messages = [
            {
                "role": "system",
                "content": f"""Generate a brief content review summary for the user.
Language: {state.get('language') or 'same as content'}

Content: {summary_data}

Format:
1. Show title and type
2. Show count of items generated
3. Show 2-3 sample items
4. Ask user to approve/export, modify, or start over

Keep it concise. Use markdown formatting."""
            }
        ]

        response = await self.llm_client.complete_with_messages(messages=messages)
        return response.content or f"Generated: {title} ({content_type})"

    def _convert_messages_to_dicts(self, raw_messages: list) -> list[dict[str, str]]:
        """Convert state messages to dict format for LLM calls.

        LangGraph's add_messages reducer may convert dict messages to LangChain
        message objects (HumanMessage, AIMessage, etc.). This method ensures
        all messages are in dict format compatible with LiteLLM.

        Following the same pattern used in companion.py workflow.

        Args:
            raw_messages: Messages from state (may be dicts or LangChain objects).

        Returns:
            List of messages in OpenAI dict format.
        """
        messages = []
        for msg in raw_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
            else:
                # LangGraph message objects (HumanMessage, AIMessage, SystemMessage)
                role = getattr(msg, "type", "assistant")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = getattr(msg, "content", "")
                if content:
                    messages.append({"role": role, "content": content})
        return messages

    def _get_last_user_message(self, state: ContentCreationState) -> str:
        """Get the last user message from state.

        Handles both dict format and LangChain message objects.
        """
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    return msg.get("content", "")
            else:
                if getattr(msg, "type", None) == "human":
                    return getattr(msg, "content", "")
        return ""

    def _build_tool_context(self, state: ContentCreationState) -> ToolContext:
        """Build ToolContext from workflow state."""
        from uuid import UUID as UUIDType

        # Get user_id, convert to UUID if string
        user_id_raw = state.get("user_id") or "00000000-0000-0000-0000-000000000000"
        if isinstance(user_id_raw, str):
            user_id = UUIDType(user_id_raw)
        else:
            user_id = user_id_raw

        return ToolContext(
            tenant_code=state.get("tenant_code") or "",
            user_id=user_id,
            user_type=state.get("user_role") or "teacher",
            grade_level=state.get("grade_level") or 5,
            language=state.get("language") or "en",
            framework_code=state.get("framework_code"),
            session=self.db_session,
        )

    async def _extract_user_intent(
        self,
        user_message: str,
        state: ContentCreationState,
    ) -> UserMessageInference:
        """Extract content type and media request from user message.

        Uses LLM-based semantic analysis to understand what the user
        wants to create without requiring explicit structured input.

        Args:
            user_message: The user's message to analyze.
            state: Current workflow state.

        Returns:
            UserMessageInference with extracted intent.
        """
        if not user_message:
            return UserMessageInference(
                content_type=None,
                wants_media=False,
                media_description=None,
                confidence=0.0,
            )

        try:
            tool = ExtractUserIntentTool()
            context = self._build_tool_context(state)

            result = await tool.execute(
                params={
                    "user_message": user_message,
                    "language": state.get("language") or "en",
                },
                context=context,
            )

            if result.success:
                return UserMessageInference(
                    content_type=result.data.get("content_type"),
                    wants_media=result.data.get("wants_media", False),
                    media_description=result.data.get("media_description"),
                    confidence=result.data.get("confidence", 0.0),
                )
        except Exception as e:
            logger.warning("Failed to extract user intent: %s", e)

        return UserMessageInference(
            content_type=None,
            wants_media=False,
            media_description=None,
            confidence=0.0,
        )

    def _get_last_assistant_message(self, state: ContentCreationState) -> str:
        """Get the last assistant message from state.

        Handles both dict format and LangChain message objects (AIMessage, etc.)
        since add_messages reducer may convert dicts to LangChain messages.
        """
        messages = state.get("messages", [])
        for msg in reversed(messages):
            # Handle dict format
            if isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            else:
                # LangChain message object (AIMessage has type="ai")
                if getattr(msg, "type", None) == "ai":
                    return getattr(msg, "content", "")
        return ""
