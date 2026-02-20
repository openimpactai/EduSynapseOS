# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Handoff to Learning Tutor tool.

This tool creates a handoff action to transfer the student
to the Learning Tutor for proactive concept teaching.

Use this when student says:
- "I want to learn about [topic]"
- "Teach me [concept]"
- "Can you explain [subject]?"
- "I need to understand [topic]"

IMPORTANT: This is different from handoff_to_tutor which is for
reactive academic questions. handoff_to_learning is for proactive
teaching sessions where the student wants to learn a new concept.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


class HandoffToLearningTool(BaseTool):
    """Tool to hand off to Learning Tutor for proactive teaching.

    The Learning Tutor provides structured, proactive teaching sessions
    with multiple learning modes (discovery, explanation, worked examples,
    guided practice, assessment).

    Use this tool when a student:
    - Wants to learn a new concept
    - Asks to be taught something
    - Needs to understand a topic before practice
    - Requests explanation of subject matter

    This creates a handoff to start a Learning Tutor session
    with the specified topic and context.
    """

    @property
    def name(self) -> str:
        return "handoff_to_learning"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff_to_learning",
                "description": (
                    "Start a proactive Learning Tutor session to teach a concept. "
                    "Use when student says 'I want to learn about...', 'Teach me...', "
                    "'I need to understand...', or similar learning requests.\n\n"
                    "Different from handoff_to_tutor: This starts a structured teaching "
                    "session with modes (discovery, explanation, examples, practice).\n\n"
                    "The Learning Tutor will:\n"
                    "- Determine the best teaching approach based on student context\n"
                    "- Use engaging explanations and examples\n"
                    "- Adapt to student's emotional state and mastery level\n"
                    "- Progress through learning modes as student understands"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": (
                                "Name of the topic to learn. Examples:\n"
                                "- 'Fractions'\n"
                                "- 'Photosynthesis'\n"
                                "- 'World War 1'\n"
                                "- 'Grammar rules'"
                            ),
                        },
                        "topic_code": {
                            "type": "string",
                            "description": (
                                "Optional topic code if known from curriculum lookup. "
                                "If not provided, will be looked up from topic_name."
                            ),
                        },
                        "subject": {
                            "type": "string",
                            "description": (
                                "Subject area. Examples:\n"
                                "- 'Mathematics'\n"
                                "- 'Science'\n"
                                "- 'History'\n"
                                "- 'Geography'"
                            ),
                        },
                        "subject_code": {
                            "type": "string",
                            "description": (
                                "Subject code for agent selection. Examples:\n"
                                "- 'mathematics'\n"
                                "- 'science'\n"
                                "- 'history'\n"
                                "- 'geography'"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Additional context about why the student wants to learn. "
                                "Examples:\n"
                                "- 'Preparing for a test'\n"
                                "- 'Didn't understand in class'\n"
                                "- 'Curious about this topic'\n"
                                "- 'Struggled during practice'"
                            ),
                        },
                    },
                    "required": ["topic_name"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the handoff_to_learning tool.

        Creates a handoff action for the frontend to navigate
        to the Learning Tutor with the specified topic.

        Args:
            params: Tool parameters from LLM.
                - topic_name: Name of the topic to learn (required)
                - topic_code: Topic code if known (optional)
                - subject: Subject name (optional)
                - subject_code: Subject code (optional)
                - context: Additional context (optional)
            context: Execution context.

        Returns:
            ToolResult with handoff action.
        """
        topic_name = params.get("topic_name")
        topic_code = params.get("topic_code")
        subject = params.get("subject")
        subject_code = params.get("subject_code")
        additional_context = params.get("context")

        if not topic_name:
            return ToolResult(
                success=False,
                error="Missing required parameter: topic_name",
            )

        # Build action params for Learning Tutor
        action_params = {
            "target_agent": "learning_tutor",
            "topic_name": topic_name,
            "entry_point": "companion_handoff",
        }

        if topic_code:
            action_params["topic_code"] = topic_code

        if subject:
            action_params["subject"] = subject

        if subject_code:
            action_params["subject_code"] = subject_code

        if additional_context:
            action_params["context"] = additional_context

        # Build handoff context for Learning Tutor workflow
        handoff_context = {
            "source": "companion",
            "session_id": context.session_id if hasattr(context, "session_id") else None,
            "topic_code": topic_code,
            "topic_name": topic_name,
            "student_request": additional_context or f"Learn about {topic_name}",
        }

        # Include emotional context for personalization
        emotional_state = None
        if context.emotional_context:
            ec = context.emotional_context
            if hasattr(ec, "current_state"):
                emotional_state = str(ec.current_state)
                action_params["emotional_state"] = emotional_state
                handoff_context["emotional_state"] = {
                    "current_state": emotional_state,
                    "intensity": getattr(ec, "intensity", "medium"),
                }

        # Include conversation context if available
        if hasattr(context, "conversation_history"):
            recent_turns = context.conversation_history[-5:] if context.conversation_history else []
            handoff_context["conversation_context"] = recent_turns

        # Include memory insights for learning focus
        if hasattr(context, "memory_context") and context.memory_context:
            mc = context.memory_context
            # Extract mastery level for this topic if available
            if hasattr(mc, "semantic") and mc.semantic and topic_code:
                topic_data = getattr(mc.semantic, "topics", {}).get(topic_code, {})
                if isinstance(topic_data, dict):
                    handoff_context["topic_mastery"] = topic_data.get("mastery", 0.5)
            # Include learning patterns from procedural memory
            if hasattr(mc, "procedural") and mc.procedural:
                patterns = getattr(mc.procedural, "patterns", {})
                if patterns:
                    handoff_context["learning_patterns"] = {
                        "preferred_mode": patterns.get("preferred_learning_mode"),
                        "optimal_time": patterns.get("optimal_learning_time"),
                    }

        action_params["handoff_context"] = handoff_context

        # Build the route with query parameters
        route = "/learning-tutor/start"
        query_parts = []

        import urllib.parse
        if topic_name:
            query_parts.append(f"topic={urllib.parse.quote(topic_name[:100])}")
        if subject:
            query_parts.append(f"subject={urllib.parse.quote(subject)}")
        if topic_code:
            query_parts.append(f"topic_code={topic_code}")

        if query_parts:
            route = f"{route}?{'&'.join(query_parts)}"

        # Build handoff action
        action = {
            "type": "handoff",
            "label": "Start Learning Session",
            "description": f"Learn about: {topic_name}",
            "icon": "📚",
            "params": action_params,
            "route": route,
            "requires_confirmation": False,
        }

        # Customize message based on context
        if emotional_state in ("frustrated", "anxious", "confused"):
            message = (
                f"I can see you'd like to understand {topic_name} better. "
                "Let me connect you with our Learning Tutor who will guide you "
                "through this step by step in a way that makes sense!"
            )
        else:
            message = (
                f"Great! Let's learn about {topic_name}! "
                "I'll connect you with our Learning Tutor who will teach you "
                "this in an engaging way with examples and practice."
            )

        return ToolResult(
            success=True,
            data={
                "action": action,
                "message": message,
            },
            stop_chaining=True,  # Handoff completes the conversation flow
        )
