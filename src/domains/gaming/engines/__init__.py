# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game engines module.

This module provides game engine implementations:
- GameEngine: Abstract base class for all engines
- GameEngineClient: HTTP client for remote game engine service
- ChessEngine: Stockfish-based chess engine (local)
- Connect4Engine: Minimax-based Connect 4 engine (local)
- GomokuEngine: Minimax-based Gomoku engine (local)
- OthelloEngine: Minimax-based Othello engine (local)
- CheckersEngine: Minimax-based Checkers engine (local)
- EngineRegistry: Registry for engine instances

The registry automatically uses the remote Game Engine service when enabled
and available, falling back to local engines otherwise.

Usage:
    from src.domains.gaming.engines import get_engine_registry, GameType

    registry = get_engine_registry()
    chess_engine = registry.get(GameType.CHESS)
    result = chess_engine.validate_move(position, move)
"""

from src.domains.gaming.engines.base import (
    GameEngine,
    EngineError,
    EngineNotAvailableError,
    InvalidPositionError,
    InvalidMoveError,
)
from src.domains.gaming.engines.registry import (
    EngineRegistry,
    EngineNotRegisteredError,
    get_engine_registry,
    reset_engine_registry,
)
from src.domains.gaming.engines.client import (
    GameEngineClient,
    get_game_engine_client,
)
from src.domains.gaming.engines.chess import ChessEngine
from src.domains.gaming.engines.connect4 import Connect4Engine
from src.domains.gaming.engines.gomoku import GomokuEngine
from src.domains.gaming.engines.othello import OthelloEngine
from src.domains.gaming.engines.checkers import CheckersEngine

__all__ = [
    # Base
    "GameEngine",
    "EngineError",
    "EngineNotAvailableError",
    "InvalidPositionError",
    "InvalidMoveError",
    # Registry
    "EngineRegistry",
    "EngineNotRegisteredError",
    "get_engine_registry",
    "reset_engine_registry",
    # Remote client
    "GameEngineClient",
    "get_game_engine_client",
    # Local implementations
    "ChessEngine",
    "Connect4Engine",
    "GomokuEngine",
    "OthelloEngine",
    "CheckersEngine",
]
