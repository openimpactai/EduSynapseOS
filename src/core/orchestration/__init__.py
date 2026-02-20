# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Workflow orchestration using LangGraph.

This package provides LangGraph-based workflow orchestration for:
- Practice sessions: Question generation, evaluation, memory updates
- Tutoring conversations: Multi-turn dialogue with context

Each workflow:
- Uses typed state for safe state management
- Supports checkpointing to PostgreSQL for resumption
- Integrates with the agent system for LLM operations
- Updates memory layers as learning progresses

Architecture:
    API -> Workflow -> Agent -> Capabilities -> LLM
              |
           State (checkpointed)
              |
           Memory Updates

Usage:
    from src.core.orchestration import PracticeWorkflow

    workflow = PracticeWorkflow(agent_factory, memory_manager)
    result = await workflow.run(initial_state)
"""

from src.core.orchestration.checkpointer import (
    init_checkpointer,
    get_checkpointer_instance,
    close_checkpointer,
    reset_checkpointer,
    create_thread_config,
    create_session_thread_id,
)
from src.core.orchestration.states import (
    PracticeState,
    QuestionRecord,
    SessionMetrics,
    create_initial_practice_state,
    TutoringState,
    ConversationTurn,
    ExplanationRecord,
    TutoringMetrics,
    create_initial_tutoring_state,
)
from src.core.orchestration.workflows import (
    PracticeWorkflow,
    TutoringWorkflow,
)

__all__ = [
    # Checkpointer
    "init_checkpointer",
    "get_checkpointer_instance",
    "close_checkpointer",
    "reset_checkpointer",
    "create_thread_config",
    "create_session_thread_id",
    # Practice State
    "PracticeState",
    "QuestionRecord",
    "SessionMetrics",
    "create_initial_practice_state",
    # Tutoring State
    "TutoringState",
    "ConversationTurn",
    "ExplanationRecord",
    "TutoringMetrics",
    "create_initial_tutoring_state",
    # Workflows
    "PracticeWorkflow",
    "TutoringWorkflow",
]
