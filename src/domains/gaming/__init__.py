# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gaming domain for educational game integration.

This domain provides:
- Game engine abstractions (Chess, Connect4, etc.)
- Game session management via GameCoachService
- Move validation and analysis
- AI opponent integration
- Coaching feedback generation

The gaming domain works with the Game Coach agent
and LangGraph workflow for educational gaming experiences.

Usage:
    from src.domains.gaming import GameCoachService
    from src.domains.gaming.engines import get_engine_registry

    # Get engine for a game type
    registry = get_engine_registry()
    engine = registry.get("chess")

    # Use service for session management
    service = GameCoachService(db, workflow, ...)
    session = await service.start_session(student_id, request)
"""

from src.domains.gaming.models import (
    GameType,
    GameMode,
    GameDifficulty,
    GameStatus,
    MoveResult,
    GameState,
    Position,
    Move,
    MoveValidation,
    PositionAnalysis,
    SuggestedMove,
    HintResponse,
    AIMove,
)

__all__ = [
    # Enums
    "GameType",
    "GameMode",
    "GameDifficulty",
    "GameStatus",
    "MoveResult",
    # Models
    "GameState",
    "Position",
    "Move",
    "MoveValidation",
    "PositionAnalysis",
    "SuggestedMove",
    "HintResponse",
    "AIMove",
]
