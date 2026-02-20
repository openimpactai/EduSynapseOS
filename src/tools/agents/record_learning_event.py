# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Record learning event tool.

This tool records significant learning events in episodic memory.
Used by agents to capture memorable moments during learning sessions
that should be remembered for future personalization.

Examples:
- Student had a breakthrough understanding fractions
- Student struggled with negative numbers
- Student showed confusion about place value
- Student mastered multiplication tables
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.models.memory import EpisodicEventType


# Valid event types with descriptions
EVENT_TYPES = {
    "breakthrough": "Student had an 'aha!' moment, suddenly understanding something",
    "struggle": "Student had difficulty with a concept or problem",
    "confusion": "Student expressed confusion or uncertainty",
    "mastery": "Student demonstrated mastery of a skill or concept",
    "engagement": "Student showed high engagement or enthusiasm",
    "frustration": "Student expressed frustration or discouragement",
    "correct_answer": "Student answered correctly (notable achievement)",
    "incorrect_answer": "Student answered incorrectly (learning opportunity)",
    "hint_used": "Student used a hint to solve a problem",
}


class RecordLearningEventTool(BaseTool):
    """Tool to record significant learning events.

    When agents observe notable learning moments (breakthroughs,
    struggles, achievements), this tool stores them in episodic
    memory for future reference and personalization.

    The tool requires memory_manager in ToolContext.extra to function.
    Events are stored with embeddings for semantic retrieval.
    """

    @property
    def name(self) -> str:
        return "record_learning_event"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "record_learning_event",
                "description": (
                    "Record a significant learning event for future reference. "
                    "Use for notable moments like breakthroughs, struggles, or achievements. "
                    "These memories help personalize future tutoring sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "enum": list(EVENT_TYPES.keys()),
                            "description": (
                                "Type of learning event:\n"
                                + "\n".join(
                                    f"- {k}: {v}" for k, v in EVENT_TYPES.items()
                                )
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": (
                                "Brief summary of what happened. Be specific but concise. "
                                "Examples: 'Understood how to add fractions with different denominators', "
                                "'Struggled with negative number subtraction', "
                                "'Finally got the concept of place value'"
                            ),
                        },
                        "topic": {
                            "type": "string",
                            "description": (
                                "The topic or subject area involved. "
                                "Examples: 'fractions', 'multiplication', 'reading comprehension'"
                            ),
                        },
                        "importance": {
                            "type": "number",
                            "description": (
                                "How important is this event (0.1-1.0). "
                                "0.3=minor, 0.5=moderate, 0.8=significant, 1.0=major milestone"
                            ),
                            "minimum": 0.1,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                    },
                    "required": ["event_type", "summary"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the record_learning_event tool.

        Stores the event in episodic memory with embeddings
        for future semantic retrieval.

        Args:
            params: Tool parameters from LLM.
                - event_type: Type of learning event
                - summary: Brief description of what happened
                - topic: Optional topic/subject area
                - importance: Optional importance score (default 0.5)
            context: Execution context with memory_manager in extra.

        Returns:
            ToolResult indicating success/failure.
        """
        event_type = params.get("event_type", "").lower()
        summary = params.get("summary", "").strip()
        topic = params.get("topic", "").strip()
        importance = min(max(params.get("importance", 0.5), 0.1), 1.0)

        # Validate event_type
        if not event_type:
            return ToolResult(
                success=False,
                error="Missing required parameter: event_type",
            )

        if event_type not in EVENT_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid event_type: {event_type}. Valid: {', '.join(EVENT_TYPES.keys())}",
            )

        # Validate summary
        if not summary:
            return ToolResult(
                success=False,
                error="Missing required parameter: summary",
            )

        if len(summary) < 10:
            return ToolResult(
                success=False,
                error="Summary too short (minimum 10 characters)",
            )

        # Get memory_manager from context
        memory_manager = context.extra.get("memory_manager")
        if not memory_manager:
            return ToolResult(
                success=False,
                error="Memory manager not available",
            )

        # Map string event type to enum
        event_type_map = {
            "breakthrough": EpisodicEventType.BREAKTHROUGH,
            "struggle": EpisodicEventType.STRUGGLE,
            "confusion": EpisodicEventType.CONFUSION,
            "mastery": EpisodicEventType.MASTERY,
            "engagement": EpisodicEventType.ENGAGEMENT,
            "frustration": EpisodicEventType.FRUSTRATION,
            "correct_answer": EpisodicEventType.CORRECT_ANSWER,
            "incorrect_answer": EpisodicEventType.INCORRECT_ANSWER,
            "hint_used": EpisodicEventType.HINT_USED,
        }

        try:
            # Build full summary with topic if provided
            full_summary = summary
            if topic:
                full_summary = f"[{topic}] {summary}"

            # Store the event
            memory = await memory_manager.episodic.store(
                tenant_code=context.tenant_code,
                student_id=context.student_id,
                event_type=event_type_map[event_type],
                summary=full_summary,
                details={
                    "topic": topic,
                    "original_summary": summary,
                    "recorded_via": "record_learning_event_tool",
                },
                importance=importance,
            )

            return ToolResult(
                success=True,
                data={
                    "message": f"Recorded {event_type} event: {summary[:50]}...",
                    "event_id": str(memory.id),
                    "event_type": event_type,
                    "importance": importance,
                    "recorded": True,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to record learning event: {str(e)}",
            )
