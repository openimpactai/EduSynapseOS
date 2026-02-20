# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game Coach Agent for LLM-based game coaching.

This module provides the GameCoachAgent class that wraps DynamicAgent
to provide game coaching functionality. Each game type has its own
YAML configuration with game-specific prompts, examples, and terminology.

Supported games:
- chess: Uses config/agents/games/chess_coach.yaml
- connect4: Uses config/agents/games/connect4_coach.yaml
- gomoku: Uses config/agents/games/gomoku_coach.yaml
- othello: Uses config/agents/games/othello_coach.yaml
- checkers: Uses config/agents/games/checkers_coach.yaml

The agent supports multiple intents:
- move_comment: Commentary on player moves
- ai_move_explain: Explanation of AI moves
- greeting: Game start greetings
- hint: Graduated hints (level 1-3)
- game_end: Game end messages
- invalid_move: Invalid move feedback
- analysis: Full game analysis

Example:
    agent = GameCoachAgent(llm_client, game_type=GameType.CHESS)
    context = context_builder.build_move_context(...)
    message = await agent.generate_move_comment(context)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.core.agents.capabilities.base import CapabilityContext, CapabilityResult
from src.core.agents.capabilities.registry import CapabilityRegistry
from src.core.agents.context import AgentConfig, AgentExecutionContext
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.intelligence.llm.client import LLMClient
from src.core.personas.loader import load_persona
from src.domains.gaming.context import CoachIntent, GameCoachContext
from src.domains.gaming.models import GameType

logger = logging.getLogger(__name__)

# Mapping of game types to their config file names
GAME_CONFIG_FILES = {
    GameType.CHESS: "chess_coach.yaml",
    GameType.CONNECT4: "connect4_coach.yaml",
    GameType.GOMOKU: "gomoku_coach.yaml",
    GameType.OTHELLO: "othello_coach.yaml",
    GameType.CHECKERS: "checkers_coach.yaml",
}


def get_agent_config_path(game_type: GameType) -> Path:
    """Get the path to the game-specific coach agent configuration.

    Args:
        game_type: The type of game (chess, connect4, etc.)

    Returns:
        Path to the game-specific config file in config/agents/games/

    Raises:
        ValueError: If game type is not supported.
    """
    config_file = GAME_CONFIG_FILES.get(game_type)
    if not config_file:
        raise ValueError(f"No coach configuration found for game type: {game_type}")

    return (
        Path(__file__).parent.parent.parent.parent
        / "config"
        / "agents"
        / "games"
        / config_file
    )


class GameCoachAgent:
    """Game coaching agent using DynamicAgent infrastructure.

    This agent wraps the DynamicAgent with game-specific YAML configuration
    to provide intelligent, context-aware game coaching. Each game type
    has its own configuration with specialized prompts and examples.

    Attributes:
        game_type: The type of game this agent coaches.
        config: Loaded agent configuration.
        agent: Underlying DynamicAgent instance.

    Example:
        agent = GameCoachAgent(llm_client, game_type=GameType.CHESS)
        context = context_builder.build_move_context(
            position_before=pos1,
            position_after=pos2,
            move="e2e4",
            player="player",
            analysis=analysis,
            student=StudentContext(name="Oliver", grade_level=4),
            game_mode=GameMode.TUTORIAL,
        )
        message = await agent.generate_move_comment(context)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        game_type: GameType,
        capability_registry: CapabilityRegistry | None = None,
    ):
        """Initialize the game coach agent for a specific game type.

        Args:
            llm_client: LLM client for text generation.
            game_type: The type of game (chess, connect4, etc.)
            capability_registry: Registry of capabilities (uses default if None).
        """
        self._llm_client = llm_client
        self._game_type = game_type
        self._capability_registry = capability_registry or CapabilityRegistry.default()

        # Load game-specific agent configuration
        config_path = get_agent_config_path(game_type)
        self._config = AgentConfig.from_yaml(config_path)

        # Create DynamicAgent
        self._agent = DynamicAgent(
            config=self._config,
            llm_client=llm_client,
            capability_registry=self._capability_registry,
        )

        # Load and set persona
        persona_id = self._config.default_persona
        try:
            persona = load_persona(persona_id)
            self._agent.set_persona(persona)
            logger.info(
                "GameCoachAgent initialized for %s with persona: %s",
                game_type.value,
                persona_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to load persona '%s': %s. Agent will work without persona.",
                persona_id,
                str(e),
            )

    @property
    def game_type(self) -> GameType:
        """Get the game type this agent coaches."""
        return self._game_type

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config

    async def generate_move_comment(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate coach commentary for a player's move.

        Args:
            context: Game context with position, move, and analysis data.

        Returns:
            Coach message string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_move_analysis",
        )

    async def generate_ai_move_explanation(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate explanation for AI's move (tutorial mode only).

        Args:
            context: Game context with AI move data.

        Returns:
            AI move explanation string.
        """
        # Only explain AI moves in tutorial mode
        logger.info("generate_ai_move_explanation called with game_mode=%s", context.game_mode)
        if context.game_mode != "tutorial":
            logger.info("Skipping AI move explanation in %s mode", context.game_mode)
            return ""

        logger.info("Proceeding to generate AI move explanation")
        return await self._execute_capability(
            context=context,
            intent="game_move_analysis",
            intent_override="ai_move_explain",
        )

    async def generate_greeting(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate game start greeting.

        Args:
            context: Game context with student and game info.

        Returns:
            Greeting message string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_coach_response",
            intent_override="greeting",
        )

    async def generate_hint(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate hint at specified level.

        Args:
            context: Game context with hint_level set (1-3).

        Returns:
            Hint message string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_hint_generation",
            intent_override="hint",
        )

    async def generate_game_end_message(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate game end message.

        Args:
            context: Game context with game_result and result_reason.

        Returns:
            Game end message string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_coach_response",
            intent_override="game_end",
        )

    async def generate_invalid_move_message(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate feedback for invalid move attempt.

        Args:
            context: Game context with invalid_move and invalid_reason.

        Returns:
            Invalid move feedback string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_coach_response",
            intent_override="invalid_move",
        )

    async def generate_analysis_summary(
        self,
        context: GameCoachContext,
    ) -> str:
        """Generate full game analysis summary.

        Args:
            context: Game context with session stats and analysis.

        Returns:
            Analysis summary string.
        """
        return await self._execute_capability(
            context=context,
            intent="game_move_analysis",
            intent_override="analysis",
        )

    async def generate_improvement_tips(
        self,
        context: GameCoachContext,
    ) -> list[str]:
        """Generate improvement tips based on game analysis.

        Args:
            context: Game context with session stats.

        Returns:
            List of improvement tip strings.
        """
        result = await self._execute_capability(
            context=context,
            intent="game_move_analysis",
            intent_override="improvement_tips",
            return_raw=True,
        )

        # Parse tips from result
        if isinstance(result, str):
            # Split by newlines and filter empty lines
            tips = [tip.strip() for tip in result.split("\n") if tip.strip()]
            # Remove numbering if present (1., 2., etc.)
            cleaned_tips = []
            for tip in tips:
                if tip and tip[0].isdigit() and len(tip) > 2 and tip[1] in ".)":
                    cleaned_tips.append(tip[2:].strip())
                elif tip.startswith("-"):
                    cleaned_tips.append(tip[1:].strip())
                else:
                    cleaned_tips.append(tip)
            return cleaned_tips[:5]  # Max 5 tips

        return []

    async def _execute_capability(
        self,
        context: GameCoachContext,
        intent: str,
        intent_override: str | None = None,
        return_raw: bool = False,
    ) -> str:
        """Execute a capability with the given context.

        Args:
            context: Game coach context.
            intent: Capability name (e.g., 'game_move_analysis').
            intent_override: Override intent for prompt selection.
            return_raw: If True, return raw response without extraction.

        Returns:
            Extracted coach message or raw response.
        """
        try:
            # Build runtime context for YAML prompt interpolation
            runtime_context = context.to_runtime_context()

            # Add intent override if specified
            if intent_override:
                runtime_context["intent"] = intent_override

            # Build capability params
            params = context.get_capability_params()

            # Create execution context
            exec_context = AgentExecutionContext(
                tenant_id="gaming",  # Gaming domain
                student_id=str(context.student.name),  # Use name as identifier
                topic=context.game_type,
                intent=intent,
                params=params,
            )

            logger.debug(
                "Executing capability: intent=%s, override=%s, game_mode=%s",
                intent,
                intent_override,
                context.game_mode,
            )

            # Execute agent
            response = await self._agent.execute(
                context=exec_context,
                runtime_context=runtime_context,
            )

            if not response.success:
                logger.warning(
                    "Agent execution failed: %s",
                    response.raw_response,
                )
                return self._get_fallback_message(context)

            # Extract message from result
            result = response.result

            if return_raw:
                return response.raw_response

            if isinstance(result, CapabilityResult):
                # Try to get coach_message from result
                if hasattr(result, "coach_message") and result.coach_message:
                    return self._parse_coach_response(result.coach_message)

            # Fallback to raw response - parse JSON if present
            return self._parse_coach_response(response.raw_response)

        except Exception as e:
            logger.exception(
                "Error executing capability %s: %s",
                intent,
                str(e),
            )
            return self._get_fallback_message(context)

    def _parse_coach_response(self, response: str) -> str:
        """Parse LLM response to extract coach message.

        The LLM may return responses in various formats:
        - Plain text message
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Raw JSON object

        This method extracts the actual message text from any format.

        Args:
            response: Raw LLM response string.

        Returns:
            Extracted message string.
        """
        if not response:
            return response

        text = response.strip()

        # Check if response is wrapped in markdown code blocks
        # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(code_block_pattern, text)
        if match:
            text = match.group(1).strip()

        # Try to parse as JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Look for common message field names
                for field in ["message", "coach_message", "text", "response", "content"]:
                    if field in data and data[field]:
                        message = data[field]
                        # Optionally append encouragement if present
                        if "encouragement" in data and data["encouragement"]:
                            message = f"{message} {data['encouragement']}"
                        return message
                # If no known field found, return first string value
                for value in data.values():
                    if isinstance(value, str) and value:
                        return value
        except (json.JSONDecodeError, TypeError):
            # Not JSON, return as-is
            pass

        return text

    def _get_fallback_message(self, context: GameCoachContext) -> str:
        """Get a fallback message when agent execution fails.

        Args:
            context: Game coach context.

        Returns:
            Simple fallback message.
        """
        intent = context.intent

        if intent == CoachIntent.GREETING:
            return f"Hi {context.student.name}! Let's play!"

        if intent == CoachIntent.AI_MOVE_EXPLAIN:
            return "I made that move to improve my position."

        if intent == CoachIntent.HINT:
            # Level-specific fallback hints
            hint_level = context.hint_level or 1
            if hint_level == 1:
                return "Think about how to improve your piece activity."
            elif hint_level == 2:
                # Try to give piece-specific hint
                if context.analysis and context.analysis.best_move:
                    return f"Look at what moves are available in the center area."
                return "Consider developing your minor pieces to more active squares."
            else:  # Level 3
                # Reveal the best move
                if context.analysis and context.analysis.best_move:
                    best = context.analysis.best_move
                    reason = context.analysis.best_move_reason or "it improves your position"
                    return f"The best move is {best}. {reason}"
                return "The best move controls the center and develops your pieces."

        if intent == CoachIntent.GAME_END:
            if context.game_result == "win":
                return f"Congratulations {context.student.name}! Great game!"
            return "Good game! Let's play again!"

        if intent == CoachIntent.INVALID_MOVE:
            return "That move isn't allowed. Try another one!"

        # Default: move comment
        if context.last_move and context.last_move.quality:
            quality = context.last_move.quality
            if quality == "excellent":
                return "Excellent move!"
            if quality in ["mistake", "blunder"]:
                return "Interesting choice. Let's see how it plays out."

        return "Your move!"
