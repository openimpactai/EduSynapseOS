# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Record interest tool.

This tool records student interests detected from conversation.
Used by the companion when students mention hobbies, favorites,
or activities they enjoy.

Examples:
- Student mentions they love playing Minecraft
- Student talks about their favorite sport
- Student shares interest in music or art
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.models.memory import AssociationType


# Valid interest categories
INTEREST_CATEGORIES = frozenset({
    "gaming",
    "sports",
    "music",
    "art",
    "science",
    "nature",
    "technology",
    "reading",
    "movies",
    "crafts",
    "animals",
    "food",
    "travel",
    "social",
    "other",
})


class RecordInterestTool(BaseTool):
    """Tool to record student interests from conversations.

    When the companion detects a student expressing interest in
    something, this tool stores it in associative memory for
    future personalization.

    The tool requires memory_manager in ToolContext.extra to function.
    Interests are stored with embeddings for semantic search.
    """

    @property
    def name(self) -> str:
        return "record_interest"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "record_interest",
                "description": (
                    "Record a student interest detected from conversation. "
                    "Use when student mentions hobbies, favorites, or things they enjoy. "
                    "This helps personalize future interactions and explanations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interest": {
                            "type": "string",
                            "description": (
                                "Description of the interest. Be specific but concise. "
                                "Examples: 'Playing Minecraft and building games', "
                                "'Soccer and sports statistics', 'Drawing anime characters'"
                            ),
                        },
                        "category": {
                            "type": "string",
                            "enum": list(INTEREST_CATEGORIES),
                            "description": (
                                "Category of interest:\n"
                                "- gaming: Video games, board games\n"
                                "- sports: Physical activities, teams\n"
                                "- music: Playing, listening, dancing\n"
                                "- art: Drawing, painting, crafts\n"
                                "- science: Experiments, nature, space\n"
                                "- nature: Animals, plants, outdoors\n"
                                "- technology: Coding, robots, gadgets\n"
                                "- reading: Books, stories, comics\n"
                                "- movies: Films, TV shows, cartoons\n"
                                "- crafts: Building, making things\n"
                                "- animals: Pets, wildlife\n"
                                "- food: Cooking, baking, eating\n"
                                "- travel: Places, exploring\n"
                                "- social: Friends, family activities\n"
                                "- other: Anything else"
                            ),
                        },
                        "strength": {
                            "type": "number",
                            "description": (
                                "How strongly the student expressed this interest (0.1-1.0). "
                                "0.3=mentioned briefly, 0.5=discussed, 0.8=very enthusiastic"
                            ),
                            "minimum": 0.1,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                    },
                    "required": ["interest", "category"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the record_interest tool.

        Stores the interest in associative memory with embeddings
        for future semantic search.

        Args:
            params: Tool parameters from LLM.
                - interest: Description of the interest
                - category: Interest category
                - strength: Optional strength (default 0.5)
            context: Execution context with memory_manager in extra.

        Returns:
            ToolResult indicating success/failure.
        """
        interest = params.get("interest", "").strip()
        category = params.get("category", "").lower()
        strength = min(max(params.get("strength", 0.5), 0.1), 1.0)

        # Validate interest
        if not interest:
            return ToolResult(
                success=False,
                error="Missing required parameter: interest",
            )

        if len(interest) < 3:
            return ToolResult(
                success=False,
                error="Interest description too short (minimum 3 characters)",
            )

        # Validate category
        if not category:
            return ToolResult(
                success=False,
                error="Missing required parameter: category",
            )

        if category not in INTEREST_CATEGORIES:
            return ToolResult(
                success=False,
                error=f"Invalid category: {category}. Valid: {', '.join(INTEREST_CATEGORIES)}",
            )

        # Get memory_manager from context
        memory_manager = context.extra.get("memory_manager")
        if not memory_manager:
            return ToolResult(
                success=False,
                error="Memory manager not available",
            )

        try:
            # Store the interest
            memory = await memory_manager.associative.store(
                tenant_code=context.tenant_code,
                student_id=context.student_id,
                association_type=AssociationType.INTEREST,
                content=interest,
                strength=strength,
                tags=[category],
            )

            return ToolResult(
                success=True,
                data={
                    "message": f"Recorded interest: {interest}",
                    "interest_id": str(memory.id),
                    "category": category,
                    "strength": strength,
                    "recorded": True,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to record interest: {str(e)}",
            )
