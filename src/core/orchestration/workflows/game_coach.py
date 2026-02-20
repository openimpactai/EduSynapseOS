# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Coach workflow using LangGraph.

This workflow manages an interactive game coaching session where
students play strategy games (chess, connect4) with AI coaching.

Workflow Structure:
    initialize
        ↓
    load_context (4-layer memory + persona)
        ↓
    setup_game (initialize engine, get starting position)
        ↓
    generate_greeting (coach welcome message)
        ↓
    [conditional: AI's turn first → make_ai_move → wait_for_move]
    wait_for_move [INTERRUPT POINT]
        ↓
    process_move (validate move, analyze, generate coach message)
        ↓
    [conditional: game over → end_game, else → make_ai_move]
    make_ai_move
        ↓
    check_game_end
        ↓
    [conditional: game over → end_game, else → wait_for_move]

The workflow uses checkpointing with interrupt_before pattern,
matching the Practice workflow architecture for reliable pause/resume.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from src.core.agents.capabilities.registry import get_default_registry
from src.core.agents.context import AgentConfig
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.intelligence.llm import LLMClient
from src.core.orchestration.states.game_coach import (
    GameCoachState,
    GameStats,
    MoveRecord,
    create_initial_game_coach_state,
)
from src.domains.gaming.engines.registry import get_engine_registry
from src.domains.gaming.models import GameDifficulty, GameMode, GameType, Position

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.memory.manager import MemoryManager
    from src.core.personas.manager import PersonaManager
    from src.domains.analytics.events import EventTracker

logger = logging.getLogger(__name__)


class GameCoachWorkflow:
    """LangGraph workflow for game coaching sessions.

    This workflow orchestrates a game coaching session where a student
    plays strategy games with AI opponents and receives coaching feedback.

    The workflow:
    - Uses game engines for move validation and AI play
    - Generates age-appropriate coaching messages via LLM
    - Tracks performance and identifies learning points
    - Supports interruption and resumption

    Attributes:
        llm_client: LLM client for generating coach messages.
        memory_manager: Manager for memory operations.
        persona_manager: Manager for personas.
        checkpointer: Checkpointer for state persistence.

    Example:
        >>> workflow = GameCoachWorkflow(llm_client, memory_manager, ...)
        >>> initial_state = create_initial_game_coach_state(...)
        >>> result = await workflow.run(initial_state, thread_id="session_123")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_manager: "MemoryManager",
        persona_manager: "PersonaManager",
        checkpointer: BaseCheckpointSaver | None = None,
        db_session: "AsyncSession | None" = None,
        event_tracker: "EventTracker | None" = None,
    ):
        """Initialize the game coach workflow.

        Args:
            llm_client: LLM client for generating coach messages.
            memory_manager: Manager for memory operations.
            persona_manager: Manager for personas.
            checkpointer: Checkpointer for state persistence.
            db_session: Database session for operations.
            event_tracker: Tracker for publishing analytics events.
        """
        self._llm_client = llm_client
        self._memory_manager = memory_manager
        self._persona_manager = persona_manager
        self._checkpointer = checkpointer
        self._db_session = db_session
        self._event_tracker = event_tracker

        # Load agent config
        self._agent_config = self._load_agent_config()

        # Create DynamicAgent for coach message generation
        self._agent = DynamicAgent(
            config=self._agent_config,
            llm_client=llm_client,
            capability_registry=get_default_registry(),
        )

        # Engine registry for game engines
        self._engine_registry = get_engine_registry()

        # Build the workflow graph
        self._graph = self._build_graph()

    def _load_agent_config(self) -> AgentConfig:
        """Load game coach agent configuration from YAML.

        Returns:
            AgentConfig loaded from config/agents/game_coach.yaml.

        Raises:
            FileNotFoundError: If game_coach.yaml is not found.
        """
        config_path = Path("config/agents/game_coach.yaml")
        if not config_path.exists():
            logger.error("Game coach agent config not found: %s", config_path)
            raise FileNotFoundError(f"Game coach config not found: {config_path}")

        return AgentConfig.from_yaml(config_path)

    def set_db_session(self, session: "AsyncSession") -> None:
        """Set database session for operations.

        Args:
            session: AsyncSession for database operations.
        """
        self._db_session = session

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            StateGraph configured for game coaching.
        """
        graph = StateGraph(GameCoachState)

        # Add nodes
        graph.add_node("initialize", self._initialize)
        graph.add_node("load_context", self._load_context)
        graph.add_node("setup_game", self._setup_game)
        graph.add_node("generate_greeting", self._generate_greeting)
        graph.add_node("wait_for_move", self._wait_for_move)
        graph.add_node("process_move", self._process_move)
        graph.add_node("make_ai_move", self._make_ai_move)
        graph.add_node("check_game_end", self._check_game_end)
        graph.add_node("end_game", self._end_game)

        # Set entry point
        graph.set_entry_point("initialize")

        # Linear edges
        graph.add_edge("initialize", "load_context")
        graph.add_edge("load_context", "setup_game")
        graph.add_edge("setup_game", "generate_greeting")

        # After greeting, check if it's player's turn or AI's turn
        graph.add_conditional_edges(
            "generate_greeting",
            self._route_after_greeting,
            {
                "wait": "wait_for_move",
                "ai_first": "make_ai_move",
            },
        )

        # After wait_for_move, process the move
        graph.add_edge("wait_for_move", "process_move")

        # After processing player move, either end game or AI responds
        graph.add_conditional_edges(
            "process_move",
            self._route_after_player_move,
            {
                "game_over": "end_game",
                "ai_turn": "make_ai_move",
            },
        )

        # After AI move, check if game ended
        graph.add_edge("make_ai_move", "check_game_end")

        # Check game end routes
        graph.add_conditional_edges(
            "check_game_end",
            self._route_after_check,
            {
                "game_over": "end_game",
                "continue": "wait_for_move",
            },
        )

        graph.add_edge("end_game", END)

        return graph

    def compile(self) -> Any:
        """Compile the workflow graph with interrupt support.

        Returns:
            Compiled workflow that can be executed.
        """
        return self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["wait_for_move"],
        )

    async def run(
        self,
        initial_state: GameCoachState,
        thread_id: str,
    ) -> GameCoachState:
        """Run the workflow from initial state.

        Executes until the first interrupt point (wait_for_move).

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

    async def send_move(
        self,
        thread_id: str,
        move: str,
        time_spent_ms: int | None = None,
    ) -> GameCoachState:
        """Send a move and get response.

        Args:
            thread_id: Thread ID for the game.
            move: Move notation (e.g., "e2e4" for chess, "3" for connect4).
            time_spent_ms: Time spent thinking in milliseconds.

        Returns:
            Updated workflow state with move results.
        """
        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}

        # Verify workflow is paused
        state_snapshot = await compiled.aget_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No state found for thread {thread_id}")

        # Update state with move via aupdate_state
        await compiled.aupdate_state(
            config,
            {
                "_pending_move": move,
                "_pending_time_spent": (time_spent_ms // 1000) if time_spent_ms else 0,
                "your_turn": False,
            },
        )

        logger.info(
            "Resuming workflow for thread=%s with move: %s",
            thread_id,
            move,
        )

        # Resume workflow
        result = await compiled.ainvoke(None, config=config)
        return result

    # =========================================================================
    # Node Implementations
    # =========================================================================

    async def _initialize(self, state: GameCoachState) -> dict:
        """Initialize the game coaching session.

        Args:
            state: Current workflow state.

        Returns:
            State updates with active status.
        """
        logger.info(
            "Initializing game coach session: session=%s, game=%s, mode=%s",
            state.get("session_id"),
            state.get("game_type"),
            state.get("game_mode"),
        )

        # Publish game started event (non-blocking)
        asyncio.create_task(self._publish_game_started(state))

        return {
            "status": "active",
            "consecutive_mistakes": 0,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _load_context(self, state: GameCoachState) -> dict:
        """Load memory context for personalization.

        Args:
            state: Current workflow state.

        Returns:
            State updates with memory context.
        """
        logger.info(
            "Loading context: student=%s",
            state["student_id"],
        )

        memory_context = {}

        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            full_context = await self._memory_manager.get_full_context(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                session=self._db_session,
            )

            if full_context:
                memory_context = full_context.model_dump()
                logger.debug("Loaded memory context for student")

        except Exception as e:
            logger.warning("Failed to load memory context: %s", str(e))

        return {"memory_context": memory_context}

    async def _setup_game(self, state: GameCoachState) -> dict:
        """Set up the game with initial position.

        Args:
            state: Current workflow state.

        Returns:
            State updates with initial position.
        """
        logger.info("Setting up game: %s", state["game_type"])

        try:
            # Get game engine
            game_type = GameType(state["game_type"])
            engine = self._engine_registry.get_engine(game_type)

            # Get initial position
            if state.get("current_position"):
                # Custom starting position (for puzzles)
                position = Position(**state["current_position"])
            else:
                position = engine.get_initial_position()

            # Set persona on agent
            persona_id = state.get("persona_id", "game_coach")
            persona_name = None
            try:
                persona = self._persona_manager.get_persona(persona_id)
                self._agent.set_persona(persona)
                persona_name = getattr(persona, "name", persona_id)
            except Exception as e:
                logger.warning("Failed to load persona '%s': %s", persona_id, str(e))

            return {
                "current_position": position.model_dump(),
                "persona_name": persona_name,
            }

        except Exception as e:
            logger.exception("Failed to setup game")
            return {
                "error": f"Failed to setup game: {str(e)}",
                "status": "error",
            }

    async def _generate_greeting(self, state: GameCoachState) -> dict:
        """Generate coach greeting message.

        Args:
            state: Current workflow state.

        Returns:
            State updates with greeting.
        """
        logger.info("Generating greeting for session: %s", state["session_id"])

        try:
            # Build greeting based on game mode
            game_mode = state.get("game_mode", "practice")
            student_name = state.get("student_name", "there")
            game_type = state.get("game_type", "chess")
            language = state.get("language", "en")
            grade_level = state.get("grade_level", 5)

            # Use LLM to generate personalized greeting
            system_prompt = f"""You are a friendly game coach for {game_type}.
Your student is {student_name}, grade {grade_level}.
Generate a warm, encouraging greeting to start the game.
Game mode: {game_mode}
Speak in {language}.
Keep it SHORT (2-3 sentences max).
Just output the greeting message, no JSON."""

            user_prompt = f"Generate a greeting for {student_name} starting a {game_mode} mode {game_type} game."

            response = await self._llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=150,
            )

            greeting = response.content.strip()

            if not greeting:
                greeting = f"Hi {student_name}! Ready to play {game_type}? Let's have fun!"

            return {
                "first_greeting": greeting,
                "last_coach_message": greeting,
                "messages": [{"role": "assistant", "content": greeting}],
            }

        except Exception as e:
            logger.warning("Failed to generate greeting: %s", str(e))
            student_name = state.get("student_name", "there")
            default_greeting = f"Hi {student_name}! Let's play! Your move."
            return {
                "first_greeting": default_greeting,
                "last_coach_message": default_greeting,
                "messages": [{"role": "assistant", "content": default_greeting}],
            }

    async def _wait_for_move(self, state: GameCoachState) -> dict:
        """Wait for player move (interrupt point).

        Args:
            state: Current workflow state.

        Returns:
            State updates.
        """
        logger.debug("Waiting for player move")

        pending_move = state.get("_pending_move")

        if pending_move:
            return {
                "your_turn": False,
                "_pending_move": None,
                "last_activity_at": datetime.now().isoformat(),
            }

        return {
            "your_turn": True,
            "last_activity_at": datetime.now().isoformat(),
        }

    async def _process_move(self, state: GameCoachState) -> dict:
        """Process the player's move.

        Validates the move, updates position, analyzes quality,
        and generates coach feedback.

        Args:
            state: Current workflow state.

        Returns:
            State updates with move result and coach message.
        """
        pending_move = state.get("_pending_move")
        if not pending_move:
            return {"error": "No move to process"}

        logger.info("Processing move: %s", pending_move)

        try:
            # Get game engine
            game_type = GameType(state["game_type"])
            engine = self._engine_registry.get_engine(game_type)
            difficulty = GameDifficulty(state.get("difficulty", "medium"))

            # Get current position
            position = Position(**state["current_position"])

            # Validate the move
            from src.domains.gaming.models import Move

            move = Move(notation=pending_move, player="player")
            validation = engine.validate_move(position, move)

            if not validation.is_valid:
                # Invalid move - ask player to try again
                error_msg = validation.error_message or "Invalid move. Please try again."
                return {
                    "last_coach_message": error_msg,
                    "your_turn": True,
                    "_pending_move": None,
                    "messages": [{"role": "assistant", "content": error_msg}],
                }

            # Analyze the move quality
            analysis = engine.analyze_position(position)
            best_move = analysis.best_moves[0] if analysis.best_moves else None

            # Determine move quality
            move_quality = None
            if best_move and pending_move == best_move.notation:
                move_quality = "excellent"
            elif validation.captures_piece or validation.gives_check:
                move_quality = "good"
            else:
                # Simple quality assessment based on evaluation change
                old_eval = analysis.evaluation or 0.0
                new_position = validation.resulting_position
                if new_position:
                    new_analysis = engine.analyze_position(new_position)
                    new_eval = new_analysis.evaluation or 0.0
                    eval_diff = new_eval - old_eval

                    if eval_diff > 0.5:
                        move_quality = "good"
                    elif eval_diff < -2.0:
                        move_quality = "blunder"
                    elif eval_diff < -1.0:
                        move_quality = "mistake"
                    elif eval_diff < -0.5:
                        move_quality = "inaccuracy"
                    else:
                        move_quality = "good"

            # Update stats
            stats = dict(state.get("stats", {}))
            stats["total_moves"] = stats.get("total_moves", 0) + 1
            if move_quality == "excellent":
                stats["excellent_moves"] = stats.get("excellent_moves", 0) + 1
            elif move_quality == "good":
                stats["good_moves"] = stats.get("good_moves", 0) + 1
            elif move_quality == "inaccuracy":
                stats["inaccuracies"] = stats.get("inaccuracies", 0) + 1
            elif move_quality == "mistake":
                stats["mistakes"] = stats.get("mistakes", 0) + 1
            elif move_quality == "blunder":
                stats["blunders"] = stats.get("blunders", 0) + 1

            time_spent = state.get("_pending_time_spent", 0)
            stats["time_spent_seconds"] = stats.get("time_spent_seconds", 0) + time_spent

            # Record move
            move_history = list(state.get("move_history", []))
            move_record = MoveRecord(
                move_number=state["move_number"],
                player="player",
                notation=pending_move,
                position_before=state["current_position"],
                position_after=validation.resulting_position.model_dump() if validation.resulting_position else {},
                evaluation_before=analysis.evaluation,
                quality=move_quality,
                is_best_move=best_move and pending_move == best_move.notation,
                best_move=best_move.notation if best_move else None,
                time_spent_seconds=time_spent,
            )
            move_history.append(move_record)

            # Generate coach message
            coach_message = await self._generate_move_comment(
                state=state,
                move=pending_move,
                move_quality=move_quality,
                analysis=analysis,
                best_move=best_move.notation if best_move else None,
            )

            # Check if game is over
            game_over = validation.is_checkmate or validation.is_stalemate
            game_result = None
            result_reason = None

            if validation.is_checkmate:
                game_over = True
                game_result = "win"
                result_reason = "checkmate"
            elif validation.is_stalemate:
                game_over = True
                game_result = "draw"
                result_reason = "stalemate"
            elif validation.is_draw:
                game_over = True
                game_result = "draw"
                result_reason = "draw"

            # Track consecutive mistakes for attention detection
            consecutive_mistakes = state.get("consecutive_mistakes", 0)
            if move_quality in ["mistake", "blunder"]:
                consecutive_mistakes += 1

                # Publish mistake detected event if threshold reached
                if consecutive_mistakes >= 2:
                    asyncio.create_task(
                        self._publish_mistake_detected(state, consecutive_mistakes)
                    )
            else:
                consecutive_mistakes = 0

            # Publish move made event for significant moves
            is_best = best_move and pending_move == best_move.notation
            asyncio.create_task(
                self._publish_move_made(
                    state=state,
                    move=pending_move,
                    move_quality=move_quality,
                    time_spent=time_spent,
                    is_best_move=is_best,
                )
            )

            # Record to memory (non-blocking)
            asyncio.create_task(
                self._record_move_memory(
                    state=state,
                    move=pending_move,
                    move_quality=move_quality,
                    time_spent=time_spent,
                    is_best_move=is_best,
                )
            )
            asyncio.create_task(
                self._record_procedural_observation(
                    state=state,
                    move_quality=move_quality,
                    time_spent=time_spent,
                )
            )

            return {
                "current_position": validation.resulting_position.model_dump() if validation.resulting_position else state["current_position"],
                "move_history": move_history,
                "move_number": state["move_number"] + 1,
                "stats": stats,
                "consecutive_mistakes": consecutive_mistakes,
                "last_move_quality": move_quality,
                "last_best_move": best_move.notation if best_move else None,
                "last_analysis": analysis.model_dump(),
                "last_coach_message": coach_message,
                "game_over": game_over,
                "game_result": game_result,
                "result_reason": result_reason,
                "your_turn": False,
                "_pending_move": None,
                "_pending_time_spent": None,
                "messages": [{"role": "assistant", "content": coach_message}],
            }

        except Exception as e:
            logger.exception("Failed to process move")
            return {
                "error": f"Failed to process move: {str(e)}",
                "last_coach_message": "Oops, something went wrong. Please try again.",
                "your_turn": True,
                "_pending_move": None,
            }

    async def _make_ai_move(self, state: GameCoachState) -> dict:
        """Make the AI's move.

        Args:
            state: Current workflow state.

        Returns:
            State updates with AI move.
        """
        logger.info("Making AI move")

        try:
            # Get game engine
            game_type = GameType(state["game_type"])
            engine = self._engine_registry.get_engine(game_type)
            difficulty = GameDifficulty(state.get("difficulty", "medium"))

            # Get current position
            position = Position(**state["current_position"])

            # Get AI move
            ai_move = engine.get_ai_move(position, difficulty)

            # Validate and apply the move
            from src.domains.gaming.models import Move

            move = Move(notation=ai_move.move, player="ai")
            validation = engine.validate_move(position, move)

            if not validation.is_valid:
                logger.error("AI generated invalid move: %s", ai_move.move)
                return {"error": "AI generated invalid move"}

            # Record AI move
            move_history = list(state.get("move_history", []))
            move_record = MoveRecord(
                move_number=state["move_number"],
                player="ai",
                notation=ai_move.move,
                position_before=state["current_position"],
                position_after=validation.resulting_position.model_dump() if validation.resulting_position else {},
                time_spent_seconds=0,
            )
            move_history.append(move_record)

            # Generate AI move explanation (for tutorial mode)
            ai_comment = ""
            if state.get("game_mode") == "tutorial":
                ai_comment = await self._generate_ai_move_explanation(
                    state=state,
                    move=ai_move.move,
                    reasoning=ai_move.reasoning,
                )

            # Check if game is over
            game_over = validation.is_checkmate or validation.is_stalemate
            game_result = None
            result_reason = None

            if validation.is_checkmate:
                game_over = True
                game_result = "loss"
                result_reason = "checkmate"
            elif validation.is_stalemate:
                game_over = True
                game_result = "draw"
                result_reason = "stalemate"
            elif validation.is_draw:
                game_over = True
                game_result = "draw"
                result_reason = "draw"

            return {
                "current_position": validation.resulting_position.model_dump() if validation.resulting_position else state["current_position"],
                "move_history": move_history,
                "move_number": state["move_number"] + 1,
                "last_coach_message": ai_comment if ai_comment else state.get("last_coach_message"),
                "game_over": game_over,
                "game_result": game_result,
                "result_reason": result_reason,
                "your_turn": not game_over,
                "current_turn": "player" if not game_over else "ai",
            }

        except Exception as e:
            logger.exception("Failed to make AI move")
            return {
                "error": f"Failed to make AI move: {str(e)}",
                "status": "error",
            }

    async def _check_game_end(self, state: GameCoachState) -> dict:
        """Check if the game has ended.

        Args:
            state: Current workflow state.

        Returns:
            State updates.
        """
        return {}

    async def _end_game(self, state: GameCoachState) -> dict:
        """End the game and generate summary.

        Args:
            state: Current workflow state.

        Returns:
            State updates with game summary.
        """
        logger.info(
            "Game ended: session=%s, result=%s, reason=%s",
            state.get("session_id"),
            state.get("game_result"),
            state.get("result_reason"),
        )

        # Publish game completed event (non-blocking)
        asyncio.create_task(self._publish_game_completed(state))

        # Record game completion in memory (non-blocking)
        asyncio.create_task(self._record_game_completion_memory(state))

        try:
            # Generate end game message
            summary_message = await self._generate_game_summary(state)

            return {
                "status": "completed",
                "last_coach_message": summary_message,
                "ended_at": datetime.now().isoformat(),
                "messages": [{"role": "assistant", "content": summary_message}],
            }

        except Exception as e:
            logger.warning("Failed to generate game summary: %s", str(e))
            return {
                "status": "completed",
                "ended_at": datetime.now().isoformat(),
            }

    # =========================================================================
    # Routing Functions
    # =========================================================================

    def _route_after_greeting(
        self, state: GameCoachState
    ) -> Literal["wait", "ai_first"]:
        """Route after greeting based on who moves first.

        Args:
            state: Current workflow state.

        Returns:
            "wait" if player moves first, "ai_first" if AI moves first.
        """
        if state.get("your_turn", True):
            return "wait"
        return "ai_first"

    def _route_after_player_move(
        self, state: GameCoachState
    ) -> Literal["game_over", "ai_turn"]:
        """Route after player move.

        Args:
            state: Current workflow state.

        Returns:
            "game_over" if game ended, "ai_turn" otherwise.
        """
        if state.get("game_over"):
            return "game_over"
        return "ai_turn"

    def _route_after_check(
        self, state: GameCoachState
    ) -> Literal["game_over", "continue"]:
        """Route after checking game end.

        Args:
            state: Current workflow state.

        Returns:
            "game_over" if game ended, "continue" otherwise.
        """
        if state.get("game_over"):
            return "game_over"
        if state.get("status") == "error":
            return "game_over"
        return "continue"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _generate_move_comment(
        self,
        state: GameCoachState,
        move: str,
        move_quality: str | None,
        analysis: Any,
        best_move: str | None,
    ) -> str:
        """Generate coach comment for a player move.

        Args:
            state: Current workflow state.
            move: The move played.
            move_quality: Quality assessment.
            analysis: Position analysis.
            best_move: Best move according to engine.

        Returns:
            Coach comment string.
        """
        try:
            game_mode = state.get("game_mode", "practice")
            student_name = state.get("student_name", "there")
            grade_level = state.get("grade_level", 5)
            language = state.get("language", "en")

            # Build prompt based on move quality
            quality_context = ""
            if move_quality == "excellent":
                quality_context = "The student played the BEST move. Be celebratory!"
            elif move_quality == "good":
                quality_context = "Good solid move. Brief praise."
            elif move_quality == "inaccuracy":
                quality_context = f"Slight inaccuracy. Best was {best_move}. Gently mention."
            elif move_quality == "mistake":
                quality_context = f"Mistake made. Best was {best_move}. Be supportive, explain kindly."
            elif move_quality == "blunder":
                quality_context = f"Blunder. Best was {best_move}. Be very supportive, this is a learning moment."

            system_prompt = f"""You are a friendly game coach.
Student: {student_name}, grade {grade_level}
Game mode: {game_mode}
Language: {language}

{quality_context}

Rules:
- Keep it SHORT (1-2 sentences)
- Be encouraging, NEVER discouraging
- For tutorial mode, explain more
- For challenge mode, be brief
- Just output the message, no JSON."""

            user_prompt = f"Generate a coach comment for move {move} (quality: {move_quality})."

            response = await self._llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=100,
            )

            return response.content.strip() or "Nice move!"

        except Exception as e:
            logger.warning("Failed to generate move comment: %s", str(e))
            if move_quality in ["excellent", "good"]:
                return "Good move!"
            return "Interesting choice. Your turn!"

    async def _generate_ai_move_explanation(
        self,
        state: GameCoachState,
        move: str,
        reasoning: str | None,
    ) -> str:
        """Generate explanation for AI's move (tutorial mode).

        Args:
            state: Current workflow state.
            move: The AI's move.
            reasoning: Engine's reasoning if available.

        Returns:
            Explanation string.
        """
        try:
            student_name = state.get("student_name", "there")
            grade_level = state.get("grade_level", 5)
            language = state.get("language", "en")

            system_prompt = f"""You are a friendly game coach explaining your move.
Student: {student_name}, grade {grade_level}
Language: {language}

Explain why you played this move in simple terms.
Keep it SHORT (1-2 sentences).
Just output the explanation, no JSON."""

            user_prompt = f"Explain the move {move}. Reasoning: {reasoning or 'strategic'}"

            response = await self._llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=100,
            )

            return response.content.strip() or f"I played {move}. Your turn!"

        except Exception as e:
            logger.warning("Failed to generate AI explanation: %s", str(e))
            return f"I played {move}. Your turn!"

    async def _generate_game_summary(self, state: GameCoachState) -> str:
        """Generate end-of-game summary.

        Args:
            state: Current workflow state.

        Returns:
            Summary message.
        """
        try:
            student_name = state.get("student_name", "there")
            grade_level = state.get("grade_level", 5)
            language = state.get("language", "en")
            game_result = state.get("game_result", "completed")
            result_reason = state.get("result_reason", "")
            stats = state.get("stats", {})

            # Build context
            if game_result == "win":
                result_context = "The student WON! Celebrate!"
            elif game_result == "loss":
                result_context = "The student lost. Be supportive and encouraging."
            else:
                result_context = "The game was a draw. Acknowledge the good battle."

            system_prompt = f"""You are a friendly game coach summarizing the game.
Student: {student_name}, grade {grade_level}
Language: {language}

{result_context}

Stats: {stats.get('total_moves', 0)} moves, {stats.get('excellent_moves', 0)} excellent, {stats.get('blunders', 0)} blunders

Rules:
- Keep it SHORT (2-3 sentences)
- Highlight positives
- If they lost, encourage them to try again
- Suggest what they did well
- Just output the message, no JSON."""

            user_prompt = f"Summarize the game that ended in {game_result} by {result_reason}."

            response = await self._llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=150,
            )

            return response.content.strip() or "Good game! Want to play again?"

        except Exception as e:
            logger.warning("Failed to generate game summary: %s", str(e))
            return "Good game! Want to play again?"

    # =========================================================================
    # Event Publishing Methods
    # =========================================================================

    async def _publish_game_started(self, state: GameCoachState) -> None:
        """Publish game started event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            await self._event_tracker.track_event(
                event_type="gaming.session.started",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "gaming",
                    "game_type": state.get("game_type"),
                    "game_mode": state.get("game_mode"),
                    "difficulty": state.get("difficulty"),
                    "player_color": state.get("player_color"),
                },
            )
            logger.debug("Published gaming.session.started event")
        except Exception as e:
            logger.warning("Failed to publish game started event: %s", str(e))

    async def _publish_game_completed(self, state: GameCoachState) -> None:
        """Publish game completed event.

        Args:
            state: Current workflow state.
        """
        if self._event_tracker is None:
            return

        try:
            stats = state.get("stats", {})
            await self._event_tracker.track_event(
                event_type="gaming.session.completed",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "gaming",
                    "game_type": state.get("game_type"),
                    "game_mode": state.get("game_mode"),
                    "difficulty": state.get("difficulty"),
                    "result": state.get("game_result"),
                    "result_reason": state.get("result_reason"),
                    "total_moves": stats.get("total_moves", 0),
                    "time_spent_seconds": stats.get("time_spent_seconds", 0),
                    "excellent_moves": stats.get("excellent_moves", 0),
                    "blunders": stats.get("blunders", 0),
                },
            )
            logger.debug("Published gaming.session.completed event")
        except Exception as e:
            logger.warning("Failed to publish game completed event: %s", str(e))

    async def _publish_move_made(
        self,
        state: GameCoachState,
        move: str,
        move_quality: str | None,
        time_spent: int,
        is_best_move: bool,
    ) -> None:
        """Publish move made event for significant moves.

        Only publishes for excellent, good, mistake, or blunder moves
        to avoid event spam.

        Args:
            state: Current workflow state.
            move: The move notation.
            move_quality: Quality of the move.
            time_spent: Time spent in seconds.
            is_best_move: Whether it was the best move.
        """
        if self._event_tracker is None:
            return

        # Only publish significant moves
        if move_quality not in ["excellent", "good", "mistake", "blunder"]:
            return

        try:
            await self._event_tracker.track_event(
                event_type="gaming.move.made",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "gaming",
                    "game_type": state.get("game_type"),
                    "move_number": state.get("move_number"),
                    "move": move,
                    "quality": move_quality,
                    "is_best_move": is_best_move,
                    "time_spent_seconds": time_spent,
                },
            )
            logger.debug("Published move made event: %s (%s)", move, move_quality)
        except Exception as e:
            logger.warning("Failed to publish move made event: %s", str(e))

    async def _publish_mistake_detected(
        self,
        state: GameCoachState,
        consecutive_count: int,
    ) -> None:
        """Publish mistake detected event for attention pattern detection.

        Args:
            state: Current workflow state.
            consecutive_count: Number of consecutive mistakes.
        """
        if self._event_tracker is None:
            return

        try:
            await self._event_tracker.track_event(
                event_type="gaming.mistake.detected",
                student_id=state["student_id"],
                session_id=state["session_id"],
                data={
                    "session_type": "gaming",
                    "game_type": state.get("game_type"),
                    "consecutive_mistakes": consecutive_count,
                    "move_number": state.get("move_number"),
                },
            )
            logger.debug("Published gaming.mistake.detected event: %d consecutive", consecutive_count)
        except Exception as e:
            logger.warning("Failed to publish mistake detected event: %s", str(e))

    # =========================================================================
    # Memory Recording Methods
    # =========================================================================

    async def _record_move_memory(
        self,
        state: GameCoachState,
        move: str,
        move_quality: str | None,
        time_spent: int,
        is_best_move: bool,
    ) -> None:
        """Record significant move in episodic memory.

        Only records excellent, good, mistake, or blunder moves to avoid
        memory spam.

        Args:
            state: Current workflow state.
            move: The move notation.
            move_quality: Quality of the move.
            time_spent: Time spent in seconds.
            is_best_move: Whether it was the best move.
        """
        # Only record significant moves
        if move_quality not in ["excellent", "good", "mistake", "blunder"]:
            return

        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            # Determine importance based on move quality
            importance = 0.4
            if move_quality == "excellent":
                importance = 0.7
            elif move_quality in ["mistake", "blunder"]:
                importance = 0.6

            game_topic = f"game_{state.get('game_type', 'chess')}"

            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                event_type="game_move",
                topic=game_topic,
                data={
                    "game_type": state.get("game_type"),
                    "move": move,
                    "move_quality": move_quality,
                    "move_number": state.get("move_number"),
                    "is_best_move": is_best_move,
                    "time_spent_seconds": time_spent,
                    "session_id": state["session_id"],
                },
                importance=importance,
            )

            logger.debug("Recorded move to episodic memory: %s (%s)", move, move_quality)

        except Exception as e:
            logger.warning("Failed to record move memory: %s", str(e))

    async def _record_game_completion_memory(self, state: GameCoachState) -> None:
        """Record game completion in episodic memory.

        Args:
            state: Current workflow state.
        """
        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
            )

            stats = state.get("stats", {})
            game_topic = f"game_{state.get('game_type', 'chess')}"

            # Record game completion event
            await self._memory_manager.record_learning_event(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                event_type="game_completed",
                topic=game_topic,
                data={
                    "game_type": state.get("game_type"),
                    "game_mode": state.get("game_mode"),
                    "difficulty": state.get("difficulty"),
                    "result": state.get("game_result"),
                    "result_reason": state.get("result_reason"),
                    "total_moves": stats.get("total_moves", 0),
                    "time_spent_seconds": stats.get("time_spent_seconds", 0),
                    "excellent_moves": stats.get("excellent_moves", 0),
                    "good_moves": stats.get("good_moves", 0),
                    "mistakes": stats.get("mistakes", 0),
                    "blunders": stats.get("blunders", 0),
                    "session_id": state["session_id"],
                },
                importance=0.7,
            )

            logger.debug("Recorded game completion to episodic memory")

        except Exception as e:
            logger.warning("Failed to record game completion memory: %s", str(e))

    async def _record_procedural_observation(
        self,
        state: GameCoachState,
        move_quality: str | None,
        time_spent: int,
    ) -> None:
        """Record gaming pattern observation in procedural memory.

        Tracks behavioral patterns that inform personalization:
        - When the student plays best (time of day)
        - Thinking time patterns by move quality
        - Game mode preferences

        Args:
            state: Current workflow state.
            move_quality: Quality of the move.
            time_spent: Time spent thinking in seconds.
        """
        try:
            student_uuid = (
                UUID(state["student_id"])
                if isinstance(state["student_id"], str)
                else state["student_id"]
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

            observation = {
                "game_type": state.get("game_type"),
                "game_mode": state.get("game_mode"),
                "difficulty": state.get("difficulty"),
                "time_of_day": time_of_day,
                "thinking_time_seconds": time_spent,
                "move_quality": move_quality,
                "is_good_move": move_quality in ["excellent", "good"] if move_quality else False,
                "session_id": state["session_id"],
            }

            await self._memory_manager.record_procedural_observation(
                tenant_code=state["tenant_code"],
                student_id=student_uuid,
                observation=observation,
                topic_full_code=None,
            )

            logger.debug(
                "Recorded procedural observation: time=%s, quality=%s",
                time_of_day,
                move_quality,
            )

        except Exception as e:
            logger.warning("Failed to record procedural observation: %s", str(e))
