# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to practice tool.

This tool creates a handoff action to transfer the student
to the practice module for topic-specific practice sessions.
"""

from typing import Any
import urllib.parse

from src.core.tools import BaseTool, ToolContext, ToolResult


class HandoffToPracticeTool(BaseTool):
    """Tool to hand off student to practice module.

    Creates navigation action for frontend to direct student
    to the practice interface with optional topic pre-selection.

    This tool should be called when:
    - Student explicitly says they want to practice
    - Student selects a topic from the list
    - Student wants to work on a weak area
    """

    @property
    def name(self) -> str:
        return "handoff_to_practice"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_practice",
                "description": (
                    "Hand off to practice module for a practice session. "
                    "Use this when:\n"
                    "- Student has selected a topic to practice\n"
                    "- Student confirms they want to practice a specific topic\n"
                    "- You want to start a practice session\n\n"
                    "IMPORTANT: Always use topic_full_code (from get_topics result) "
                    "instead of just topic_code. The full code format is: "
                    "'framework.subject.grade.unit.topic' (e.g., 'UK-NC-2014.MATHS.Y5.MATHS-Y5-NPV.MATHS-Y5-NPV-PV'). "
                    "This enables direct navigation to practice without additional selection steps.\n\n"
                    "NOTE: Only use this AFTER topic selection is confirmed. "
                    "Use get_subjects → get_topics flow first if topic is not specified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_full_code": {
                            "type": "string",
                            "description": (
                                "Full topic code from get_topics result "
                                "(e.g., 'UK-NC-2014.MATHS.Y5.MATHS-Y5-NPV.MATHS-Y5-NPV-PV'). "
                                "PREFERRED - use this when available as it contains all navigation info."
                            ),
                        },
                        "topic_code": {
                            "type": "string",
                            "description": (
                                "Short topic code (e.g., 'MATHS-Y5-NPV-PV'). "
                                "Use topic_full_code instead when available."
                            ),
                        },
                        "topic_name": {
                            "type": "string",
                            "description": (
                                "Topic name for display in the message."
                            ),
                        },
                        "subject_code": {
                            "type": "string",
                            "description": (
                                "Subject code for context. Optional."
                            ),
                        },
                        "session_type": {
                            "type": "string",
                            "enum": ["quick", "deep", "review", "random"],
                            "description": (
                                "Type of practice session:\n"
                                "- quick: Short 5-question session\n"
                                "- deep: Longer focused session\n"
                                "- review: Spaced repetition review\n"
                                "- random: Mixed topics from subject\n"
                                "Default: quick"
                            ),
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "adaptive", "challenging"],
                            "description": (
                                "Difficulty level. Default: adaptive (adjusts based on performance)"
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
        """Execute the handoff_to_practice tool.

        Creates a navigation action for the frontend to start
        a practice session with the specified parameters.

        Args:
            params: Tool parameters from LLM.
                - topic_full_code: Full topic code (preferred)
                - topic_code: Short topic code to practice
                - topic_name: Topic name for display
                - subject_code: Subject code for context
                - session_type: Type of practice session
                - difficulty: Difficulty level
            context: Execution context.

        Returns:
            ToolResult with handoff action.
        """
        topic_full_code = params.get("topic_full_code")
        topic_code = params.get("topic_code")
        topic_name = params.get("topic_name", "selected topic")
        subject_code = params.get("subject_code")
        session_type = params.get("session_type", "quick")
        difficulty = params.get("difficulty", "adaptive")

        # Parse full_code to extract framework, subject, grade, unit, topic
        # Format: framework.subject.grade.unit.topic
        framework_code = None
        grade_code = None
        unit_code = None

        if topic_full_code:
            parts = topic_full_code.split(".")
            if len(parts) == 5:
                framework_code = parts[0]
                subject_code = parts[1]  # Override subject_code from full_code
                grade_code = parts[2]
                unit_code = parts[3]
                topic_code = parts[4]  # Override topic_code from full_code

        # Validate session_type
        valid_session_types = {"quick", "deep", "review", "random"}
        if session_type not in valid_session_types:
            session_type = "quick"

        # Validate difficulty
        valid_difficulties = {"easy", "adaptive", "challenging"}
        if difficulty not in valid_difficulties:
            difficulty = "adaptive"

        # Build action params with all curriculum navigation data
        action_params = {
            "target_module": "practice",
            "session_type": session_type,
            "difficulty": difficulty,
        }

        # Include full topic code if available (allows frontend to parse if needed)
        if topic_full_code:
            action_params["topic_full_code"] = topic_full_code

        # Include all parsed curriculum components for direct store population
        if framework_code:
            action_params["framework_code"] = framework_code

        if subject_code:
            action_params["subject_code"] = subject_code

        if grade_code:
            action_params["grade_code"] = grade_code

        if unit_code:
            action_params["unit_code"] = unit_code

        if topic_code:
            action_params["topic_code"] = topic_code

        if topic_name:
            action_params["topic_name"] = topic_name

        # Get session_id from context.extra (set by DynamicAgent from runtime_context)
        companion_session_id = context.extra.get("session_id") if context.extra else None

        # Build handoff context for Practice workflow
        handoff_context = {
            "source": "companion",
            "session_id": companion_session_id,
            "topic_code": topic_code,
            "student_request": f"Practice {topic_name}" if topic_name else "Practice session",
            "suggested_mode": session_type,
        }

        # Include emotional context for personalization
        if context.emotional_context:
            ec = context.emotional_context
            if hasattr(ec, "current_state") and ec.current_state:
                action_params["emotional_state"] = str(ec.current_state)
                handoff_context["emotional_state"] = {
                    "current_state": str(ec.current_state),
                    "intensity": getattr(ec, "intensity", "medium"),
                }

        # Include conversation context if available
        if hasattr(context, "conversation_history"):
            recent_turns = context.conversation_history[-5:] if context.conversation_history else []
            handoff_context["conversation_context"] = recent_turns

        # Include weak concepts from memory for practice focus
        # Use memory_manager.semantic from context.extra to query weak areas if available
        memory_manager = context.extra.get("memory_manager") if context.extra else None
        if memory_manager and hasattr(memory_manager, "semantic"):
            try:
                weak_areas = await memory_manager.semantic.get_weak_areas(
                    tenant_code=context.tenant_code,
                    student_id=context.user_id,
                    max_mastery=0.5,
                    limit=5,
                )
                if weak_areas:
                    # Extract entity_full_code from weak areas
                    handoff_context["weak_concepts"] = [
                        area.entity_full_code for area in weak_areas
                    ]
            except Exception:
                # Silently ignore errors in weak area retrieval
                pass

        action_params["handoff_context"] = handoff_context

        # Build the route with query parameters
        route = "/practice"
        query_parts = []

        if topic_code and topic_code != "random":
            query_parts.append(f"topic={topic_code}")
        if subject_code:
            query_parts.append(f"subject={subject_code}")
        if session_type:
            query_parts.append(f"type={session_type}")
        if difficulty != "adaptive":
            query_parts.append(f"difficulty={difficulty}")

        if query_parts:
            route = f"{route}?{'&'.join(query_parts)}"

        # Build human-readable message based on session type
        if session_type == "random":
            message = "Let's start a random practice session! I'll mix questions from different topics."
        elif session_type == "review":
            message = f"Starting a review session for {topic_name}. Let's strengthen your memory!"
        elif session_type == "deep":
            message = f"Starting a deep practice session on {topic_name}. Let's master this topic!"
        else:
            message = f"Let's practice {topic_name}! I'll get some questions ready for you."

        # Build handoff action
        action = {
            "type": "handoff",
            "label": "Start Practice",
            "description": f"Practice: {topic_name}" if topic_name else "Practice Session",
            "icon": "📝",
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
