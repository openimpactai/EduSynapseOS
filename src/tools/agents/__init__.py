# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agent tools for emotional tracking, alerts, memory, and agent handoffs.

This category contains tools for agent-specific actions:
- record_emotion: Record detected emotional state for async processing
- create_alert: Create proactive alerts for teachers/parents
- search_interests: Search student interests using semantic search
- record_interest: Record student interests from conversations
- record_learning_event: Record significant learning events
- handoff_to_tutor: Handoff conversation to tutor agent
- handoff_to_practice: Handoff to practice module
- handoff_to_game: Handoff to games module

These tools handle cross-cutting concerns like emotional awareness,
memory operations, stakeholder notifications, and agent-to-agent communication.
"""

from src.tools.agents.create_alert import CreateAlertTool
from src.tools.agents.handoff_to_game import HandoffToGameTool
from src.tools.agents.handoff_to_practice import HandoffToPracticeTool
from src.tools.agents.handoff_to_tutor import HandoffToTutorTool
from src.tools.agents.record_emotion import RecordEmotionTool
from src.tools.agents.record_interest import RecordInterestTool
from src.tools.agents.record_learning_event import RecordLearningEventTool
from src.tools.agents.search_interests import SearchInterestsTool

__all__ = [
    "CreateAlertTool",
    "RecordEmotionTool",
    "RecordInterestTool",
    "RecordLearningEventTool",
    "SearchInterestsTool",
    "HandoffToTutorTool",
    "HandoffToPracticeTool",
    "HandoffToGameTool",
]
