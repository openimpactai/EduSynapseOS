# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Navigate tool.

This tool creates navigation actions with typed parameters that the frontend
can use to redirect the student to specific pages or activities.

The navigate tool is the FINAL step in the clarification flow:
1. Student expresses intent (e.g., "I want to practice")
2. Agent uses get_subjects to show options
3. Student selects subject (e.g., "Mathematics")
4. Agent uses get_topics to show topic options
5. Student selects topic or chooses random
6. Agent uses navigate with all clarified parameters

CRITICAL: Do NOT call navigate until ALL required parameters are clarified.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


# Action type icons for UI
ACTION_ICONS = {
    "practice": "📝",
    "learning": "📚",
    "game": "🎮",
    "review": "🔄",
    "break": "☕",
    "creative": "🎨",
    "navigate": "➡️",
    "handoff": "👋",
}


class NavigateTool(BaseTool):
    """Tool to navigate student to a specific page or activity.

    Creates a navigation action with typed parameters that the frontend
    interprets to redirect the student. The workflow includes
    these actions in the response for frontend handling.

    IMPORTANT: This tool should only be called AFTER all required
    parameters have been clarified through conversation.

    Required parameters by action_type:
    - practice: subject_full_code (unless random=true), optionally topic_full_code
    - learning: subject_full_code, optionally topic_full_code
    - game: game_type
    - review: (none required, but subject_full_code recommended)
    - break: (none required)
    - creative: activity_type
    - navigate: destination
    - handoff: Use handoff_to_tutor tool instead
    """

    @property
    def name(self) -> str:
        return "navigate"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "navigate",
                "description": (
                    "Navigate the student to an activity page. "
                    "CRITICAL: Only call this tool AFTER you have clarified ALL required parameters.\n\n"
                    "Required parameters by action_type:\n"
                    "- practice: subject_full_code (unless random=true), optionally topic_full_code\n"
                    "- learning: subject_full_code, optionally topic_full_code\n"
                    "- game: game_type (math, word, puzzle)\n"
                    "- review: none required, but subject_full_code is helpful\n"
                    "- break: none required\n"
                    "- creative: activity_type (drawing, story, music)\n"
                    "- navigate: destination (dashboard, settings, etc.)\n\n"
                    "WORKFLOW:\n"
                    "1. Student says 'I want to practice' → Call get_subjects, ask which subject\n"
                    "2. Student says 'Math' → Call get_topics, ask which topic or random\n"
                    "3. Student says 'Fractions' → NOW call navigate with all params\n\n"
                    "DO NOT call navigate if required parameters are missing!"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["practice", "learning", "game", "review", "break", "creative", "navigate"],
                            "description": "Type of activity/destination",
                        },
                        "label": {
                            "type": "string",
                            "description": "Button label for UI (e.g., 'Start Fractions Practice')",
                        },
                        "subject_full_code": {
                            "type": "string",
                            "description": "Subject full code (from get_subjects, e.g., 'UK-NC-2014.MAT'). Required for practice/learning unless random.",
                        },
                        "subject_name": {
                            "type": "string",
                            "description": "Subject display name (for UI)",
                        },
                        "topic_full_code": {
                            "type": "string",
                            "description": "Topic full code (from get_topics, e.g., 'UK-NC-2014.MAT.Y4.NPV.001'). Optional for specific topic.",
                        },
                        "topic_name": {
                            "type": "string",
                            "description": "Topic display name (for UI)",
                        },
                        "game_type": {
                            "type": "string",
                            "enum": ["math", "word", "puzzle"],
                            "description": "Game type (required for game action)",
                        },
                        "activity_type": {
                            "type": "string",
                            "enum": ["drawing", "story", "music"],
                            "description": "Creative activity type (required for creative action)",
                        },
                        "break_type": {
                            "type": "string",
                            "enum": ["short", "breathing", "stretch"],
                            "description": "Break activity type",
                            "default": "short",
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination for general navigation (dashboard, settings)",
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Difficulty level preference",
                        },
                        "random": {
                            "type": "boolean",
                            "description": "Random practice across all subjects/topics",
                            "default": False,
                        },
                    },
                    "required": ["action_type", "label"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the navigate tool.

        Creates a navigation action with validation and route computation.

        Args:
            params: Tool parameters from LLM.
            context: Execution context.

        Returns:
            ToolResult with navigation action or error.
        """
        action_type = params.get("action_type")
        label = params.get("label")

        if not action_type:
            return ToolResult(
                success=False,
                error="Missing required parameter: action_type",
            )

        if not label:
            return ToolResult(
                success=False,
                error="Missing required parameter: label",
            )

        # Validate required params based on action_type
        validation_error = self._validate_params(action_type, params)
        if validation_error:
            return ToolResult(
                success=False,
                error=validation_error,
                data={"missing_params": True, "action_type": action_type},
            )

        # Build the action
        action = self._build_action(action_type, params, label)

        return ToolResult(
            success=True,
            data={
                "action": action,
                "message": f"Navigation to {action_type} prepared: {label}",
            },
            stop_chaining=True,  # Navigation completes the conversation flow
        )

    def _validate_params(self, action_type: str, params: dict) -> str | None:
        """Validate required params for action type.

        Args:
            action_type: The action type.
            params: The parameters provided.

        Returns:
            Error message if validation fails, None if valid.
        """
        match action_type:
            case "practice":
                if not params.get("random") and not params.get("subject_full_code"):
                    return (
                        "subject_full_code is required for practice (unless random=true). "
                        "Call get_subjects first to get available subjects, then ask the student which one."
                    )

            case "learning":
                if not params.get("subject_full_code"):
                    return (
                        "subject_full_code is required for learning. "
                        "Call get_subjects first to get available subjects, then ask the student which one."
                    )

            case "game":
                if not params.get("game_type"):
                    return (
                        "game_type is required for game (math, word, or puzzle). "
                        "Call get_games first to show available games, then ask which one."
                    )

            case "creative":
                if not params.get("activity_type"):
                    return (
                        "activity_type is required for creative (drawing, story, or music). "
                        "Ask the student which creative activity they want."
                    )

            case "navigate":
                if not params.get("destination"):
                    return (
                        "destination is required for navigate (e.g., dashboard, settings). "
                        "Ask the student where they want to go."
                    )

        return None

    def _build_action(self, action_type: str, params: dict, label: str) -> dict:
        """Build the action dictionary with proper structure.

        Args:
            action_type: The action type.
            params: The parameters provided.
            label: The button label.

        Returns:
            Action dictionary for frontend.
        """
        # Build action params based on action type
        action_params = {}

        # Copy relevant params based on action type
        param_mapping = {
            "practice": ["subject_full_code", "subject_name", "topic_full_code", "topic_name", "difficulty", "random"],
            "learning": ["subject_full_code", "subject_name", "topic_full_code", "topic_name", "initial_question", "emotional_context"],
            "game": ["game_type", "difficulty", "subject_full_code"],
            "review": ["subject_full_code", "subject_name", "topic_full_code", "topic_name", "all_pending"],
            "break": ["break_type", "duration_minutes"],
            "creative": ["activity_type", "theme"],
            "navigate": ["destination"],
        }

        relevant_keys = param_mapping.get(action_type, [])
        for key in relevant_keys:
            if params.get(key) is not None:
                action_params[key] = params[key]

        # Build the action
        action = {
            "type": action_type,
            "label": label,
            "icon": ACTION_ICONS.get(action_type, "➡️"),
            "params": action_params,
            "route": self._compute_route(action_type, action_params),
            "requires_confirmation": False,
        }

        # Add description if we have subject/topic names
        if action_params.get("subject_name"):
            desc_parts = [action_params["subject_name"]]
            if action_params.get("topic_name"):
                desc_parts.append(action_params["topic_name"])
            action["description"] = " - ".join(desc_parts)

        return action

    def _compute_route(self, action_type: str, params: dict) -> str:
        """Compute frontend route based on action type and params.

        Args:
            action_type: The action type.
            params: The action parameters.

        Returns:
            Frontend route string with query parameters.
        """
        match action_type:
            case "practice":
                route = "/practice"
                query = []
                if params.get("subject_full_code"):
                    query.append(f"subject={params['subject_full_code']}")
                if params.get("topic_full_code"):
                    query.append(f"topic={params['topic_full_code']}")
                if params.get("difficulty"):
                    query.append(f"difficulty={params['difficulty']}")
                if params.get("random"):
                    query.append("random=true")
                return f"{route}?{'&'.join(query)}" if query else route

            case "learning":
                route = "/learn"
                query = []
                if params.get("subject_full_code"):
                    query.append(f"subject={params['subject_full_code']}")
                if params.get("topic_full_code"):
                    query.append(f"topic={params['topic_full_code']}")
                return f"{route}?{'&'.join(query)}" if query else route

            case "game":
                game_type = params.get("game_type", "math")
                route = f"/games/{game_type}"
                if params.get("difficulty"):
                    route += f"?difficulty={params['difficulty']}"
                return route

            case "review":
                route = "/review"
                query = []
                if params.get("subject_full_code"):
                    query.append(f"subject={params['subject_full_code']}")
                if params.get("topic_full_code"):
                    query.append(f"topic={params['topic_full_code']}")
                if params.get("all_pending"):
                    query.append("all=true")
                return f"{route}?{'&'.join(query)}" if query else route

            case "break":
                break_type = params.get("break_type", "short")
                if break_type == "short":
                    return "/break"
                return f"/break/{break_type}"

            case "creative":
                activity_type = params.get("activity_type", "drawing")
                return f"/creative/{activity_type}"

            case "navigate":
                return params.get("destination", "/dashboard")

            case _:
                return "/dashboard"
