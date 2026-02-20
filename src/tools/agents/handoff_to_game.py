# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to game tool.

This tool creates a handoff action to transfer the student
to the games module for educational gaming sessions.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


class HandoffToGameTool(BaseTool):
    """Tool to hand off student to games module.

    Creates navigation action for frontend to direct student
    to the game interface with game type and settings.

    This tool should be called when:
    - Student explicitly says they want to play a game
    - Student selects a specific game (chess, connect4)
    - Student wants to take a brain break with games
    """

    @property
    def name(self) -> str:
        return "handoff_to_game"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_game",
                "description": (
                    "Hand off to games module for an educational game session. "
                    "Use this when:\n"
                    "- Student wants to play chess or connect4\n"
                    "- Student says they want to play a game\n"
                    "- Student needs a brain break\n"
                    "- You want to suggest a strategy game for learning\n\n"
                    "Available games: chess, connect4"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_type": {
                            "type": "string",
                            "enum": ["chess", "connect4"],
                            "description": (
                                "Type of game to play:\n"
                                "- chess: Strategic chess game with coaching\n"
                                "- connect4: Connect Four game\n"
                                "If not specified, will show game selection."
                            ),
                        },
                        "game_mode": {
                            "type": "string",
                            "enum": ["tutorial", "practice", "challenge"],
                            "description": (
                                "Game mode:\n"
                                "- tutorial: Full explanations, best for learning\n"
                                "- practice: Normal play with hints available\n"
                                "- challenge: Minimal help, competitive\n"
                                "Default: practice"
                            ),
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["beginner", "easy", "medium", "hard", "expert"],
                            "description": (
                                "AI difficulty level. Default: medium. "
                                "Adjust based on student's experience."
                            ),
                        },
                        "player_color": {
                            "type": "string",
                            "enum": ["white", "black", "random"],
                            "description": (
                                "Player's color/side for chess. "
                                "Default: white (moves first)."
                            ),
                        },
                    },
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the handoff_to_game tool.

        Creates a navigation action for the frontend to start
        a game session with the specified parameters.

        Args:
            params: Tool parameters from LLM.
                - game_type: Type of game (chess, connect4)
                - game_mode: Game mode (tutorial, practice, challenge)
                - difficulty: AI difficulty level
                - player_color: Player's color for chess
            context: Execution context.

        Returns:
            ToolResult with handoff action.
        """
        game_type = params.get("game_type")
        game_mode = params.get("game_mode", "practice")
        difficulty = params.get("difficulty", "medium")
        player_color = params.get("player_color", "white")

        # Validate game_type
        valid_game_types = {"chess", "connect4"}
        if game_type and game_type not in valid_game_types:
            game_type = None

        # Validate game_mode
        valid_modes = {"tutorial", "practice", "challenge"}
        if game_mode not in valid_modes:
            game_mode = "practice"

        # Validate difficulty
        valid_difficulties = {"beginner", "easy", "medium", "hard", "expert"}
        if difficulty not in valid_difficulties:
            difficulty = "medium"

        # Validate player_color
        valid_colors = {"white", "black", "random"}
        if player_color not in valid_colors:
            player_color = "white"

        # Build action params
        action_params = {
            "target_module": "games",
            "game_mode": game_mode,
            "difficulty": difficulty,
        }

        if game_type:
            action_params["game_type"] = game_type

        if game_type == "chess":
            action_params["player_color"] = player_color

        # Include emotional context if available
        if context.emotional_context:
            ec = context.emotional_context
            if hasattr(ec, "current_state") and ec.current_state:
                action_params["emotional_state"] = str(ec.current_state)

        # Build the route with query parameters
        route = "/games"
        query_parts = []

        if game_type:
            query_parts.append(f"game={game_type}")
        if game_mode != "practice":
            query_parts.append(f"mode={game_mode}")
        if difficulty != "medium":
            query_parts.append(f"difficulty={difficulty}")
        if game_type == "chess" and player_color != "white":
            query_parts.append(f"color={player_color}")

        if query_parts:
            route = f"{route}?{'&'.join(query_parts)}"

        # Build human-readable message
        if game_type:
            game_names = {"chess": "Chess", "connect4": "Connect Four"}
            game_name = game_names.get(game_type, game_type)

            if game_mode == "tutorial":
                message = f"Let's learn {game_name} together! I'll explain everything as we play."
            elif game_mode == "challenge":
                message = f"Ready for a {game_name} challenge? Let's see what you've got!"
            else:
                message = f"Let's play {game_name}! I'll give you tips along the way."
        else:
            message = "Let's play a game! I'll show you what games are available."

        # Build handoff action
        action = {
            "type": "game",
            "label": f"Play {game_type.title() if game_type else 'Game'}",
            "description": f"Start {game_type or 'game'} session",
            "icon": "🎮",
            "params": action_params,
            "route": route,
            "requires_confirmation": False,
        }

        return ToolResult(
            success=True,
            data={
                "action": action,
                "message": message,
            },
            stop_chaining=True,  # Handoff completes the conversation flow
        )
