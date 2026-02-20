# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Game engine registry for managing engine instances.

This module provides:
- EngineRegistry: Central registry for game engines
- get_engine_registry: Factory function for default registry

The registry allows registration and retrieval of game engines by type.

Usage:
    from src.domains.gaming.engines import get_engine_registry

    registry = get_engine_registry()
    chess_engine = registry.get(GameType.CHESS)
    connect4_engine = registry.get(GameType.CONNECT4)
"""

import logging
from typing import Iterator

from src.domains.gaming.models import GameType
from src.domains.gaming.engines.base import GameEngine

logger = logging.getLogger(__name__)


class EngineNotRegisteredError(Exception):
    """Raised when attempting to get an unregistered engine.

    Attributes:
        game_type: The game type that was not found.
        available: List of available game types.
    """

    def __init__(self, game_type: GameType, available: list[GameType]) -> None:
        self.game_type = game_type
        self.available = available
        available_str = ", ".join(t.value for t in available)
        message = (
            f"Engine for '{game_type.value}' not registered. "
            f"Available: {available_str or 'none'}"
        )
        super().__init__(message)


class EngineRegistry:
    """Registry for managing game engine instances.

    Provides centralized management of game engines with registration,
    lookup by game type, and listing functionality.

    The registry is designed to be singleton-like per application,
    with all engines registered at startup.

    Attributes:
        _engines: Dictionary mapping game types to engine instances.

    Example:
        registry = EngineRegistry()
        registry.register(ChessEngine())
        registry.register(Connect4Engine())

        chess = registry.get(GameType.CHESS)
        validation = chess.validate_move(position, move)
    """

    def __init__(self) -> None:
        """Initialize an empty engine registry."""
        self._engines: dict[GameType, GameEngine] = {}

    def register(self, engine: GameEngine) -> None:
        """Register an engine in the registry.

        Args:
            engine: Engine instance to register.

        Raises:
            ValueError: If an engine for this game type already exists.
        """
        game_type = engine.game_type

        if game_type in self._engines:
            raise ValueError(
                f"Engine for '{game_type.value}' is already registered. "
                f"Use replace() to override."
            )

        self._engines[game_type] = engine
        logger.info(
            "Registered game engine: %s (%s)",
            engine.name,
            game_type.value,
        )

    def replace(self, engine: GameEngine) -> None:
        """Register or replace an engine in the registry.

        Unlike register(), this method will overwrite an existing
        engine for the same game type.

        Args:
            engine: Engine instance to register or replace.
        """
        game_type = engine.game_type

        if game_type in self._engines:
            logger.info(
                "Replacing game engine for %s: %s",
                game_type.value,
                engine.name,
            )
        else:
            logger.info(
                "Registering game engine: %s (%s)",
                engine.name,
                game_type.value,
            )

        self._engines[game_type] = engine

    def unregister(self, game_type: GameType) -> None:
        """Remove an engine from the registry.

        Args:
            game_type: Type of game to unregister.

        Raises:
            KeyError: If no engine is registered for this game type.
        """
        if game_type not in self._engines:
            raise KeyError(f"No engine registered for '{game_type.value}'")

        del self._engines[game_type]
        logger.info("Unregistered game engine for: %s", game_type.value)

    def get(self, game_type: GameType) -> GameEngine:
        """Get an engine by game type.

        Args:
            game_type: Type of game.

        Returns:
            The registered engine instance.

        Raises:
            EngineNotRegisteredError: If no engine is registered.
        """
        if game_type not in self._engines:
            raise EngineNotRegisteredError(
                game_type=game_type,
                available=list(self._engines.keys()),
            )

        return self._engines[game_type]

    def get_optional(self, game_type: GameType) -> GameEngine | None:
        """Get an engine by game type, returning None if not found.

        Args:
            game_type: Type of game.

        Returns:
            The engine instance or None if not registered.
        """
        return self._engines.get(game_type)

    def has(self, game_type: GameType) -> bool:
        """Check if an engine is registered for a game type.

        Args:
            game_type: Type of game to check.

        Returns:
            True if an engine is registered.
        """
        return game_type in self._engines

    def list_types(self) -> list[GameType]:
        """List all registered game types.

        Returns:
            List of registered game types.
        """
        return list(self._engines.keys())

    def list_all(self) -> list[GameEngine]:
        """List all registered engine instances.

        Returns:
            List of engine instances.
        """
        return list(self._engines.values())

    def get_info(self) -> list[dict[str, str]]:
        """Get information about all registered engines.

        Returns:
            List of dicts with 'type' and 'name' for each engine.
        """
        return [
            {
                "type": engine.game_type.value,
                "name": engine.name,
            }
            for engine in self._engines.values()
        ]

    def clear(self) -> None:
        """Remove all engines from the registry."""
        self._engines.clear()
        logger.info("Engine registry cleared")

    def __len__(self) -> int:
        """Get number of registered engines."""
        return len(self._engines)

    def __contains__(self, game_type: GameType) -> bool:
        """Check if a game type is registered."""
        return game_type in self._engines

    def __iter__(self) -> Iterator[GameType]:
        """Iterate over registered game types."""
        return iter(self._engines)

    def __repr__(self) -> str:
        """Return string representation."""
        types = ", ".join(t.value for t in self._engines.keys())
        return f"EngineRegistry([{types}])"


# Global default registry instance (lazy-loaded)
_default_registry: EngineRegistry | None = None


def get_engine_registry() -> EngineRegistry:
    """Get or create the global default engine registry.

    Creates a registry with all available game engines registered.

    Returns:
        The default engine registry.
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = _create_default_registry()

    return _default_registry


def _create_default_registry() -> EngineRegistry:
    """Create the default registry with all engines.

    Uses remote Game Engine service when enabled and available,
    falls back to local engines otherwise.

    Returns:
        Configured engine registry.
    """
    from src.core.config.settings import get_settings
    from src.domains.gaming.engines.chess import ChessEngine
    from src.domains.gaming.engines.connect4 import Connect4Engine
    from src.domains.gaming.engines.gomoku import GomokuEngine
    from src.domains.gaming.engines.othello import OthelloEngine
    from src.domains.gaming.engines.checkers import CheckersEngine

    registry = EngineRegistry()
    settings = get_settings()

    # Check if remote game engine service is enabled
    use_remote = settings.game_engine.enabled
    remote_available = False

    if use_remote:
        try:
            from src.domains.gaming.engines.client import GameEngineClient
            # Test if service is available
            test_client = GameEngineClient(GameType.CHESS)
            remote_available = test_client.is_available()
            if remote_available:
                logger.info(
                    "Game Engine service available at %s",
                    settings.game_engine.url,
                )
            else:
                logger.warning(
                    "Game Engine service not available at %s, using local engines",
                    settings.game_engine.url,
                )
        except Exception as e:
            logger.warning("Failed to connect to Game Engine service: %s", e)

    # Register engines based on availability
    if remote_available:
        # Use remote Game Engine clients for all game types
        from src.domains.gaming.engines.client import GameEngineClient

        for game_type in GameType:
            try:
                client = GameEngineClient(game_type)
                registry.register(client)
                logger.info("Registered remote engine for %s", game_type.value)
            except Exception as e:
                logger.warning("Failed to register remote %s engine: %s", game_type.value, e)
                # Fall back to local engine
                _register_local_engine(registry, game_type)
    else:
        # Use local engines
        _register_local_engine(registry, GameType.CHESS)
        _register_local_engine(registry, GameType.CONNECT4)
        _register_local_engine(registry, GameType.GOMOKU)
        _register_local_engine(registry, GameType.OTHELLO)
        _register_local_engine(registry, GameType.CHECKERS)

    logger.info(
        "Created default EngineRegistry with %d engines: %s",
        len(registry),
        [e.name for e in registry.list_all()],
    )

    return registry


def _register_local_engine(registry: EngineRegistry, game_type: GameType) -> None:
    """Register a local engine for the given game type.

    Args:
        registry: Engine registry to register to.
        game_type: Type of game to register.
    """
    try:
        if game_type == GameType.CHESS:
            from src.domains.gaming.engines.chess import ChessEngine
            registry.register(ChessEngine())
        elif game_type == GameType.CONNECT4:
            from src.domains.gaming.engines.connect4 import Connect4Engine
            registry.register(Connect4Engine())
        elif game_type == GameType.GOMOKU:
            from src.domains.gaming.engines.gomoku import GomokuEngine
            registry.register(GomokuEngine())
        elif game_type == GameType.OTHELLO:
            from src.domains.gaming.engines.othello import OthelloEngine
            registry.register(OthelloEngine())
        elif game_type == GameType.CHECKERS:
            from src.domains.gaming.engines.checkers import CheckersEngine
            registry.register(CheckersEngine())
    except Exception as e:
        logger.warning("Failed to initialize local %s engine: %s", game_type.value, e)


def reset_engine_registry() -> None:
    """Reset the global default engine registry.

    Useful for testing or reconfiguration.
    """
    global _default_registry
    _default_registry = None
    logger.info("Engine registry reset")
