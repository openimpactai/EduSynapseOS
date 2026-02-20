# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get games tool.

This tool retrieves available games for a student based on their grade level.
Games are stored in the companion_activities table with category='fun'.
"""

from typing import Any

from sqlalchemy import select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.infrastructure.database.models.tenant import CompanionActivity


class GetGamesTool(BaseTool):
    """Tool to get available games for the student.

    Queries games from companion_activities table filtered by:
    - category = 'fun'
    - is_enabled = True
    - Grade level within min_grade/max_grade range

    Used when student asks to play a game or agent wants to
    suggest a fun activity. Call this BEFORE navigate to show
    available game options.
    """

    @property
    def name(self) -> str:
        return "get_games"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_games",
                "description": (
                    "Get available games for the student. "
                    "CRITICAL: Call this when the student wants to play a game.\n\n"
                    "Use cases:\n"
                    "- Student says 'I want to play a game' → Call get_games first\n"
                    "- Student seems bored → Suggest games using get_games\n"
                    "- You want to offer fun activities → Call get_games\n\n"
                    "After getting games, ask which game they prefer:\n"
                    "- Math games (arithmetic challenges)\n"
                    "- Word games (vocabulary puzzles)\n"
                    "- Other games based on availability\n\n"
                    "Returns a list of available games with types and descriptions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["fun", "educational", "all"],
                            "description": "Game category filter. 'fun' for entertainment, 'educational' for learning games, 'all' for both.",
                            "default": "all",
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
        """Execute the get_games tool.

        Queries games from companion_activities filtered by grade level.

        Args:
            params: Tool parameters from LLM.
                - category: Category filter (fun, educational, all)
            context: Execution context with grade_level.

        Returns:
            ToolResult with list of available games.
        """
        category = params.get("category", "all")
        grade_level = context.grade_level

        try:
            # Build query for games (fun category activities)
            conditions = [
                CompanionActivity.is_enabled == True,  # noqa: E712
                CompanionActivity.min_grade <= grade_level,
                CompanionActivity.max_grade >= grade_level,
            ]

            # Filter by category
            if category == "fun":
                conditions.append(CompanionActivity.category == "fun")
            elif category == "educational":
                # Educational games might be in learning category
                # For now, include both fun and learning with "game" in code
                from sqlalchemy import or_
                conditions.append(
                    or_(
                        CompanionActivity.category == "fun",
                        CompanionActivity.code.like("game_%"),
                    )
                )
            # 'all' doesn't add category filter

            stmt = (
                select(CompanionActivity)
                .where(*conditions)
                .order_by(
                    CompanionActivity.category,
                    CompanionActivity.display_order,
                )
            )

            result = await context.session.execute(stmt)
            games = result.scalars().all()

            # Format games for LLM
            games_data = []
            for game in games:
                # Extract game type from code (e.g., "game_math" → "math")
                game_type = game.code
                if game.code.startswith("game_"):
                    game_type = game.code[5:]  # Remove "game_" prefix

                games_data.append({
                    "code": game.code,
                    "name": game.name,
                    "description": game.description,
                    "icon": game.icon,
                    "game_type": game_type,
                    "difficulty": game.difficulty,
                    "category": game.category,
                    "route": game.route,
                })

            if not games_data:
                return ToolResult(
                    success=True,
                    data={
                        "games": [],
                        "count": 0,
                        "grade_level": grade_level,
                        "message": f"No games available for grade {grade_level}. You might want to suggest other activities instead.",
                    },
                )

            # Build human-readable message
            game_names = [g["name"] for g in games_data]
            message = f"Found {len(games_data)} games: {', '.join(game_names)}"

            # Build UI element for frontend selection
            ui_options = [
                UIElementOption(
                    id=g["code"],
                    label=g["name"],
                    description=g.get("description"),
                    icon=g.get("icon"),
                    metadata={
                        "game_type": g["game_type"],
                        "difficulty": g.get("difficulty"),
                        "route": g.get("route"),
                    },
                )
                for g in games_data
            ]

            ui_element = UIElement(
                type=UIElementType.SINGLE_SELECT,
                id="game_selection",
                title="Choose a Game",
                options=ui_options,
                allow_text_input=False,
            )

            # Passthrough data for frontend with navigation context
            passthrough_data = {
                "games": games_data,
                "grade_level": grade_level,
                "intent": "game",
                "navigation": {
                    "type": "game",
                    "ready": False,  # Game selection still needed
                    "route": "/games",
                    "params": {},
                    "awaiting": "game_selection",
                },
            }

            return ToolResult(
                success=True,
                data={
                    "games": games_data,
                    "count": len(games_data),
                    "grade_level": grade_level,
                    "message": message,
                },
                ui_element=ui_element,
                passthrough_data=passthrough_data,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get games: {str(e)}",
            )
