# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LangGraph workflow implementations.

This module provides workflow classes that orchestrate educational interactions:

PracticeWorkflow:
    Question generation → wait for answer → evaluate → update memory → repeat

TutoringWorkflow:
    Receive message → process → generate response → update memory → loop

CompanionWorkflow:
    Greeting → wait for message → agent with tools → execute tools → respond → loop

TeacherCompanionWorkflow:
    Greeting → wait for message → agent with tools → execute tools → respond → loop
    (For teachers: class management, student monitoring, analytics)

GameCoachWorkflow:
    Setup game → greeting → wait for move → process move → AI move → repeat
    (For students: chess, connect4 with coaching feedback)

Each workflow:
- Uses LangGraph StateGraph for flow control
- Supports checkpointing for interruption/resumption
- Integrates with the agent system for LLM operations
- Updates memory layers as learning progresses
"""

from src.core.orchestration.workflows.companion import CompanionWorkflow
from src.core.orchestration.workflows.game_coach import GameCoachWorkflow
from src.core.orchestration.workflows.practice import PracticeWorkflow
from src.core.orchestration.workflows.practice_helper import PracticeHelperWorkflow
from src.core.orchestration.workflows.teacher_companion import TeacherCompanionWorkflow
from src.core.orchestration.workflows.tutoring import TutoringWorkflow

__all__ = [
    "CompanionWorkflow",
    "GameCoachWorkflow",
    "PracticeWorkflow",
    "PracticeHelperWorkflow",
    "TeacherCompanionWorkflow",
    "TutoringWorkflow",
]
