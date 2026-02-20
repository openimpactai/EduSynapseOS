# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Directly answer tool.

This tool allows the LLM to explicitly signal that it wants to respond
directly without calling any other tools. This prevents hallucination
of non-existent tools like "directly-answer" or "respond".

Based on Cohere Command-R best practices.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


class DirectlyAnswerTool(BaseTool):
    """Tool for direct text responses without other tool calls.

    Use this tool when:
    - User is just chatting or saying hello
    - No data retrieval is needed
    - Previous tools returned empty/no data
    - You want to ask a clarifying question
    - You want to respond with encouragement or support

    This tool simply passes through your message - it doesn't
    perform any action. It exists to give the LLM an explicit
    way to "not use tools" while still following the tool-calling pattern.
    """

    @property
    def name(self) -> str:
        return "directly_answer"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "directly_answer",
                "description": (
                    "Use this tool to respond directly to the user without calling other tools. "
                    "Call this when: greeting the user, chatting casually, asking clarifying questions, "
                    "or when other tools returned empty results. "
                    "Pass your complete response message as the 'message' parameter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Your complete response message to the user.",
                        },
                    },
                    "required": ["message"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the directly_answer tool.

        Simply returns the message as the response.

        Args:
            params: Tool parameters with 'message' field.
            context: Execution context (unused).

        Returns:
            ToolResult with the message.
        """
        message = params.get("message", "")

        if not message:
            return ToolResult(
                success=False,
                error="Message parameter is required.",
            )

        return ToolResult(
            success=True,
            data={
                "message": message,
                "direct_response": True,
            },
            stop_chaining=True,  # No more tool calls needed
        )
