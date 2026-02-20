# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Workflow state definitions.

This module provides TypedDict-based state definitions for LangGraph workflows.
Each state type defines the structure of data that flows through the workflow.

States:
    PracticeState: State for practice session workflows
    TutoringState: State for tutoring conversation workflows
    CompanionState: State for companion conversation workflows
    TeacherCompanionState: State for teacher assistant workflows
    GameCoachState: State for game coaching workflows
"""

from src.core.orchestration.states.companion import (
    CompanionAction,
    CompanionState,
    CompanionTurn,
    EmotionalSignalRecord,
    ToolCallRecord,
    create_initial_companion_state,
)
from src.core.orchestration.states.practice import (
    PracticeState,
    QuestionRecord,
    SessionMetrics,
    create_initial_practice_state,
)
from src.core.orchestration.states.practice_helper import (
    PracticeHelperMetrics,
    PracticeHelperState,
    PracticeHelperTurn,
    QuestionContext,
    StudentContext,
    TutoringMode,
    create_initial_practice_helper_state,
    escalate_mode,
    select_agent_id,
    select_tutoring_mode,
)
from src.core.orchestration.states.teacher_companion import (
    AlertSummary,
    ClassSummary,
    TeacherAction,
    TeacherCompanionState,
    TeacherTurn,
    create_initial_teacher_companion_state,
)
from src.core.orchestration.states.tutoring import (
    ConversationTurn,
    ExplanationRecord,
    TutoringMetrics,
    TutoringState,
    create_initial_tutoring_state,
)
from src.core.orchestration.states.game_coach import (
    GameCoachState,
    GameStats,
    MoveRecord,
    create_initial_game_coach_state,
)

__all__ = [
    # Companion (Student)
    "CompanionAction",
    "CompanionState",
    "CompanionTurn",
    "EmotionalSignalRecord",
    "ToolCallRecord",
    "create_initial_companion_state",
    # Teacher Companion
    "AlertSummary",
    "ClassSummary",
    "TeacherAction",
    "TeacherCompanionState",
    "TeacherTurn",
    "create_initial_teacher_companion_state",
    # Practice
    "PracticeState",
    "QuestionRecord",
    "SessionMetrics",
    "create_initial_practice_state",
    # Practice Helper Tutor
    "PracticeHelperState",
    "PracticeHelperTurn",
    "PracticeHelperMetrics",
    "QuestionContext",
    "StudentContext",
    "TutoringMode",
    "create_initial_practice_helper_state",
    "select_tutoring_mode",
    "select_agent_id",
    "escalate_mode",
    # Tutoring
    "TutoringState",
    "ConversationTurn",
    "ExplanationRecord",
    "TutoringMetrics",
    "create_initial_tutoring_state",
    # Game Coach
    "GameCoachState",
    "GameStats",
    "MoveRecord",
    "create_initial_game_coach_state",
]
