# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Search interests tool.

This tool searches student interests using semantic similarity.
Used by the companion to personalize interactions based on what
the student is interested in.

Examples:
- Finding relevant interests for a math topic
- Discovering connections between topics and hobbies
- Personalizing explanations using student's interests
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.models.memory import AssociationType


class SearchInterestsTool(BaseTool):
    """Tool to search student interests using semantic search.

    Searches the student's recorded interests using vector similarity
    to find interests relevant to a given query. Useful for personalizing
    tutoring sessions or finding analogies based on student hobbies.

    The tool requires memory_manager in ToolContext.extra to function.
    Returns top matching interests with relevance scores.
    """

    @property
    def name(self) -> str:
        return "search_interests"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_interests",
                "description": (
                    "Search for student interests relevant to a topic or query. "
                    "Use this to find connections between what the student likes "
                    "and what they are learning. Returns interests with relevance scores."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query to find relevant interests. "
                                "Examples: 'fractions and division', 'sports', "
                                "'creative building', 'music and rhythm'"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of interests to return (1-10)",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the search_interests tool.

        Performs semantic search in the associative memory layer
        to find interests relevant to the query.

        Args:
            params: Tool parameters from LLM.
                - query: Search query text
                - limit: Optional max results (default 5)
            context: Execution context with memory_manager in extra.

        Returns:
            ToolResult with matching interests and scores.
        """
        query = params.get("query", "").strip()
        limit = min(max(params.get("limit", 5), 1), 10)

        # Validate query
        if not query:
            return ToolResult(
                success=False,
                error="Missing required parameter: query",
            )

        # Get memory_manager from context
        memory_manager = context.extra.get("memory_manager")
        if not memory_manager:
            return ToolResult(
                success=False,
                error="Memory manager not available",
            )

        try:
            # Search for interests
            results = await memory_manager.associative.search(
                tenant_code=context.tenant_code,
                student_id=context.student_id,
                query=query,
                limit=limit,
                min_score=0.3,
                association_types=[AssociationType.INTEREST],
            )

            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "message": "No interests found matching this query.",
                        "interests": [],
                        "count": 0,
                    },
                )

            # Format results
            interests = [
                {
                    "content": memory.content,
                    "relevance": round(score, 2),
                    "strength": round(memory.strength, 2),
                    "tags": memory.tags,
                }
                for memory, score in results
            ]

            return ToolResult(
                success=True,
                data={
                    "message": f"Found {len(interests)} interest(s) related to '{query}'.",
                    "interests": interests,
                    "count": len(interests),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to search interests: {str(e)}",
            )
