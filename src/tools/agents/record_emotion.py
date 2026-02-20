# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Record emotion tool.

This tool captures detected emotional states from the conversation.
The actual recording to the database is handled by the workflow
in a fire-and-forget manner using EmotionalStateService.
"""

from typing import Any

from src.core.tools import BaseTool, ToolContext, ToolResult


# Valid emotional states that can be recorded
VALID_EMOTIONS = frozenset({
    "happy",
    "excited",
    "confident",
    "curious",
    "neutral",
    "bored",
    "confused",
    "frustrated",
    "anxious",
    "tired",
})

# Valid intensity levels
VALID_INTENSITIES = frozenset({"low", "moderate", "high"})


class RecordEmotionTool(BaseTool):
    """Tool to record detected emotional state.

    Captures emotional signals detected during conversation.
    The tool validates the emotion data and returns it for
    the workflow to record asynchronously (fire-and-forget).

    This approach keeps the tool stateless while allowing
    the workflow to handle the actual persistence.
    """

    @property
    def name(self) -> str:
        return "record_emotion"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "record_emotion",
                "description": (
                    "Record detected emotional state from conversation. "
                    "Use this when:\n"
                    "- Student expresses clear emotion (frustrated, happy, anxious)\n"
                    "- Emotional state changes during conversation\n"
                    "- Student explicitly shares feelings\n"
                    "This is fire-and-forget - doesn't affect your response."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "enum": list(VALID_EMOTIONS),
                            "description": "Detected emotional state",
                        },
                        "intensity": {
                            "type": "string",
                            "enum": list(VALID_INTENSITIES),
                            "description": "Emotional intensity level",
                        },
                        "triggers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "What triggered this emotion. Examples:\n"
                                "- 'math_difficulty'\n"
                                "- 'test_anxiety'\n"
                                "- 'achievement'\n"
                                "- 'peer_comparison'\n"
                                "- 'time_pressure'"
                            ),
                        },
                    },
                    "required": ["emotion", "intensity"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the record_emotion tool.

        Validates the emotion data and returns it for workflow recording.

        Args:
            params: Tool parameters from LLM.
                - emotion: Detected emotional state
                - intensity: Emotional intensity level
                - triggers: Optional list of triggers
            context: Execution context.

        Returns:
            ToolResult with validated emotion data for recording.
        """
        emotion = params.get("emotion")
        intensity = params.get("intensity")
        triggers = params.get("triggers", [])

        # Validate emotion
        if not emotion:
            return ToolResult(
                success=False,
                error="Missing required parameter: emotion",
            )

        if emotion not in VALID_EMOTIONS:
            return ToolResult(
                success=False,
                error=f"Invalid emotion: {emotion}. Valid values: {', '.join(VALID_EMOTIONS)}",
            )

        # Validate intensity
        if not intensity:
            return ToolResult(
                success=False,
                error="Missing required parameter: intensity",
            )

        if intensity not in VALID_INTENSITIES:
            return ToolResult(
                success=False,
                error=f"Invalid intensity: {intensity}. Valid values: {', '.join(VALID_INTENSITIES)}",
            )

        # Validate triggers (must be strings)
        if triggers:
            if not isinstance(triggers, list):
                triggers = [triggers]
            triggers = [str(t) for t in triggers]

        # Build human-readable message for LLM
        trigger_info = f" (triggers: {', '.join(triggers)})" if triggers else ""
        message = f"Emotion recorded: {emotion} ({intensity} intensity){trigger_info}"

        # Return validated data for workflow to record
        return ToolResult(
            success=True,
            data={
                "message": message,
                "recorded": True,
                "emotion": emotion,
                "intensity": intensity,
                "triggers": triggers,
                "student_id": str(context.student_id),
                # Flag for workflow to handle the actual recording
                "_action": "record_emotional_signal",
            },
        )
