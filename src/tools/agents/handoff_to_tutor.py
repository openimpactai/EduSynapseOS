# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to tutor tool.

This tool creates a handoff action to transfer the student
to the tutor agent for academic questions.

CRITICAL: Agents must use this tool for ANY academic question.
The companion is NOT a teacher and should NEVER explain concepts.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


class HandoffToTutorTool(BaseTool):
    """Tool to hand off academic questions to the tutor.

    Agents should NOT explain academic concepts directly.
    Instead, they use this tool to transfer the conversation
    to the tutor agent with appropriate context.

    This tool should be called IMMEDIATELY when a student asks:
    - How to solve any problem
    - Explanation of any concept
    - Help with homework
    - Why something works a certain way
    - Any "teach me" or "explain" request
    """

    @property
    def name(self) -> str:
        return "handoff_to_tutor"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_tutor",
                "description": (
                    "REQUIRED: Hand off to tutor for ANY academic/homework question. "
                    "You MUST use this tool IMMEDIATELY when student asks:\n"
                    "- How to solve a math problem (fractions, equations, etc.)\n"
                    "- Explanation of any concept (science, history, grammar)\n"
                    "- Help with homework or practice problems\n"
                    "- Why something works a certain way\n"
                    "- 'Can you teach me...' or 'Explain...'\n\n"
                    "CRITICAL: You are NOT a teacher. NEVER explain academic content yourself. "
                    "ALWAYS use this tool to transfer to the tutor who can properly teach.\n\n"
                    "Unlike navigate, this tool does NOT require prior clarification - "
                    "pass the question directly to the tutor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The academic question to hand off",
                        },
                        "topic": {
                            "type": "string",
                            "description": (
                                "Related topic if identifiable. Examples:\n"
                                "- 'fractions'\n"
                                "- 'photosynthesis'\n"
                                "- 'quadratic equations'\n"
                                "- 'grammar'"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Additional context for the tutor. Examples:\n"
                                "- 'Student was frustrated with this topic'\n"
                                "- 'Asked during homework help'\n"
                                "- 'Follow-up from previous session'"
                            ),
                        },
                    },
                    "required": ["question"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the handoff_to_tutor tool.

        Creates a handoff action for the frontend to navigate
        to the tutoring interface with context.

        Args:
            params: Tool parameters from LLM.
                - question: The academic question
                - topic: Related topic (optional)
                - context: Additional context (optional)
            context: Execution context.

        Returns:
            ToolResult with handoff action.
        """
        question = params.get("question")
        topic = params.get("topic")
        additional_context = params.get("context")

        if not question:
            return ToolResult(
                success=False,
                error="Missing required parameter: question",
            )

        # Build action params
        action_params = {
            "target_agent": "tutor",
            "question": question,
        }

        if topic:
            action_params["topic"] = topic

        if additional_context:
            action_params["context"] = additional_context

        # Include emotional context if student was struggling
        emotional_state = None
        if context.emotional_context:
            ec = context.emotional_context
            if hasattr(ec, "current_state"):
                emotional_state = str(ec.current_state)
                action_params["emotional_state"] = emotional_state

        # Build the route
        route = "/tutor"
        query = []
        if question:
            # URL encode the question for query string
            import urllib.parse
            query.append(f"q={urllib.parse.quote(question[:200])}")  # Limit length
        if topic:
            query.append(f"topic={topic}")
        if query:
            route = f"{route}?{'&'.join(query)}"

        # Build handoff action with new structure
        action = {
            "type": "handoff",
            "label": "Connect with Tutor",
            "description": f"Get help with: {topic or 'your question'}",
            "icon": "👨‍🏫",
            "params": action_params,
            "route": route,
            "requires_confirmation": False,
        }

        return ToolResult(
            success=True,
            data={
                "action": action,
                "message": (
                    "I'll connect you with our tutor who can explain this really well. "
                    "They're great at breaking down concepts!"
                ),
            },
            stop_chaining=True,  # Handoff completes the conversation flow
        )
