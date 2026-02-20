# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Emotional Intelligence System for EduSynapse.

This package provides emotional detection, state management, and response
for the educational platform. It integrates with:

- Tutoring workflow: Analyzes sentiment via emotional_analyzer agent
- Educational Theories: Provides EmotionalContext for adaptive difficulty
- Proactive Monitors: Detects distress patterns for alerts
- Companion Agent: Enables emotional support interactions
- Parent Portal: Accepts parent inputs and generates reports

Key Components:
- EmotionalStateService: Central service for all emotional operations
- EmotionalContext: Data structure passed to consumers
- emotional_analyzer agent: LLM-based detection for all messages

All emotional analysis is performed by LLM through the emotional_analyzer
agent with MessageAnalysisCapability. The service only stores and
aggregates the analysis results.

Example usage:

    from src.core.emotional import EmotionalStateService, EmotionalContext

    # Get current emotional state
    service = EmotionalStateService(db)
    context = await service.get_current_state(student_id)
    if context.needs_support:
        # Trigger support intervention
        pass

    # For chat messages, use emotional_analyzer agent in TutoringWorkflow
    # The workflow calls agent.execute(intent="message_analysis")
    # and records the result via service.record_analyzed_signal()
"""

from src.core.emotional.constants import (
    EMOTION_PRIORITY,
    STATE_TO_ACTIONS,
    EmotionalIntensity,
    EmotionalResponseAction,
    EmotionalSignalSource,
    EmotionalState,
    EmotionalThresholds,
)
from src.core.emotional.context import (
    EmotionalContext,
    EmotionalSignalData,
    EmotionalTrend,
    ParentMoodInput,
)
from src.core.emotional.service import EmotionalStateService

__all__ = [
    # Main service
    "EmotionalStateService",
    # Context and data structures
    "EmotionalContext",
    "EmotionalSignalData",
    "EmotionalTrend",
    "ParentMoodInput",
    # Constants and enums
    "EmotionalState",
    "EmotionalIntensity",
    "EmotionalSignalSource",
    "EmotionalResponseAction",
    "EmotionalThresholds",
    "EMOTION_PRIORITY",
    "STATE_TO_ACTIONS",
]
