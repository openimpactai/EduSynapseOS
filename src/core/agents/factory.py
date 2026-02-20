# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agent factory for creating DynamicAgent instances from YAML configuration.

This module provides the AgentFactory which:
- Loads agent configurations from YAML files
- Creates DynamicAgent instances with proper dependencies
- Caches created agents for reuse
- Attaches default personas automatically

Usage:
    # Get the factory singleton
    factory = get_agent_factory()

    # Get or create an agent
    agent = factory.get("assessor")

    # Set a different persona
    agent.set_persona(coach_persona)

    # Execute
    response = await agent.execute(context)
"""

import logging
from pathlib import Path
from typing import Optional

from src.core.agents.capabilities.registry import (
    CapabilityRegistry,
    get_default_registry,
)
from src.core.agents.context import AgentConfig, AgentError
from src.core.agents.dynamic_agent import DynamicAgent
from src.core.intelligence.llm.client import LLMClient
from src.core.personas.manager import PersonaManager, get_persona_manager

logger = logging.getLogger(__name__)

# Default path to agent configurations
DEFAULT_AGENTS_DIR = Path("config/agents")


class AgentNotFoundError(Exception):
    """Raised when a requested agent type is not found.

    Attributes:
        agent_type: The agent type that was not found.
        available: List of available agent types.
    """

    def __init__(self, agent_type: str, available: list[str]):
        self.agent_type = agent_type
        self.available = available
        message = (
            f"Agent type '{agent_type}' not found. "
            f"Available: {', '.join(available) if available else 'none'}"
        )
        super().__init__(message)


class AgentFactory:
    """Factory for creating DynamicAgent instances.

    The AgentFactory loads agent configurations from YAML files and
    creates DynamicAgent instances with the proper dependencies.
    It also caches created agents for reuse.

    Attributes:
        agents_dir: Path to agent configuration files.
        llm_client: Shared LLM client for all agents.
        capability_registry: Shared capability registry.
        persona_manager: Shared persona manager.

    Example:
        >>> factory = AgentFactory(llm_client, registry, persona_manager)
        >>> agent = factory.get("assessor")
        >>> # Agent is cached and reused on subsequent calls
        >>> same_agent = factory.get("assessor")
        >>> assert agent is same_agent
    """

    def __init__(
        self,
        llm_client: LLMClient,
        capability_registry: CapabilityRegistry,
        persona_manager: PersonaManager,
        agents_dir: Optional[Path] = None,
    ):
        """Initialize the AgentFactory.

        Args:
            llm_client: LLM client for agent LLM calls.
            capability_registry: Registry of available capabilities.
            persona_manager: Manager for loading personas.
            agents_dir: Path to agent configuration directory.
        """
        self._llm_client = llm_client
        self._capability_registry = capability_registry
        self._persona_manager = persona_manager
        self._agents_dir = agents_dir or DEFAULT_AGENTS_DIR
        self._agents: dict[str, DynamicAgent] = {}
        self._configs: dict[str, AgentConfig] = {}

        # Discover available agent configs
        self._discover_agents()

        logger.info(
            "AgentFactory initialized: agents_dir=%s, available=%s",
            self._agents_dir,
            list(self._configs.keys()),
        )

    def _discover_agents(self) -> None:
        """Discover available agent configurations."""
        if not self._agents_dir.exists():
            logger.warning(
                "Agents directory does not exist: %s",
                self._agents_dir,
            )
            return

        for yaml_file in self._agents_dir.glob("*.yaml"):
            try:
                config = AgentConfig.from_yaml(yaml_file)
                self._configs[config.id] = config
                logger.debug("Discovered agent config: %s", config.id)
            except Exception as e:
                logger.warning(
                    "Failed to load agent config %s: %s",
                    yaml_file,
                    str(e),
                )

    def get(self, agent_type: str) -> DynamicAgent:
        """Get or create an agent by type.

        If the agent has been created before, returns the cached instance.
        Otherwise, creates a new instance, attaches the default persona,
        and caches it.

        Args:
            agent_type: Agent type identifier (e.g., "assessor", "tutor").

        Returns:
            DynamicAgent instance.

        Raises:
            AgentNotFoundError: If the agent type is not found.
            AgentError: If agent creation fails.

        Example:
            >>> agent = factory.get("assessor")
            >>> response = await agent.execute(context)
        """
        if agent_type in self._agents:
            return self._agents[agent_type]

        agent = self.create(agent_type)
        self._agents[agent_type] = agent
        return agent

    def create(self, agent_type: str) -> DynamicAgent:
        """Create a new agent instance (not cached).

        This method always creates a new instance, even if one already
        exists in the cache. Use `get()` for cached access.

        Args:
            agent_type: Agent type identifier.

        Returns:
            New DynamicAgent instance.

        Raises:
            AgentNotFoundError: If the agent type is not found.
            AgentError: If agent creation fails.
        """
        if agent_type not in self._configs:
            # Try to load it
            config_path = self._agents_dir / f"{agent_type}.yaml"
            if config_path.exists():
                try:
                    config = AgentConfig.from_yaml(config_path)
                    self._configs[config.id] = config
                except Exception as e:
                    raise AgentError(
                        message=f"Failed to load agent config: {e}",
                        agent_id=agent_type,
                    ) from e
            else:
                raise AgentNotFoundError(
                    agent_type=agent_type,
                    available=list(self._configs.keys()),
                )

        config = self._configs[agent_type]

        try:
            # Create the agent
            agent = DynamicAgent(
                config=config,
                llm_client=self._llm_client,
                capability_registry=self._capability_registry,
            )

            # Attach default persona if available
            if config.default_persona:
                try:
                    persona = self._persona_manager.get_persona(config.default_persona)
                    agent.set_persona(persona)
                except Exception as e:
                    logger.warning(
                        "Could not attach default persona %s to agent %s: %s",
                        config.default_persona,
                        agent_type,
                        str(e),
                    )

            logger.info(
                "Created agent: type=%s, capabilities=%s, persona=%s",
                agent_type,
                config.capabilities,
                config.default_persona,
            )

            return agent

        except Exception as e:
            if isinstance(e, (AgentError, AgentNotFoundError)):
                raise
            raise AgentError(
                message=f"Failed to create agent: {e}",
                agent_id=agent_type,
                original_error=e,
            ) from e

    def list_available(self) -> list[str]:
        """List all available agent types.

        Returns:
            List of agent type identifiers.
        """
        return list(self._configs.keys())

    def get_config(self, agent_type: str) -> AgentConfig:
        """Get the configuration for an agent type.

        Args:
            agent_type: Agent type identifier.

        Returns:
            Agent configuration.

        Raises:
            AgentNotFoundError: If the agent type is not found.
        """
        if agent_type not in self._configs:
            raise AgentNotFoundError(
                agent_type=agent_type,
                available=list(self._configs.keys()),
            )
        return self._configs[agent_type]

    def clear_cache(self) -> None:
        """Clear all cached agent instances.

        This is useful for testing or when agent configurations change.
        """
        self._agents.clear()
        logger.debug("Agent cache cleared")

    def reload_configs(self) -> None:
        """Reload all agent configurations from disk.

        Also clears the agent cache since configs may have changed.
        """
        self._configs.clear()
        self._agents.clear()
        self._discover_agents()
        logger.info(
            "Agent configs reloaded: available=%s",
            list(self._configs.keys()),
        )

    def set_persona_for_agent(
        self,
        agent_type: str,
        persona_id: str,
    ) -> DynamicAgent:
        """Get an agent with a specific persona attached.

        This is a convenience method that gets the agent and
        sets a specific persona.

        Args:
            agent_type: Agent type identifier.
            persona_id: Persona to attach.

        Returns:
            Agent with the specified persona attached.

        Raises:
            AgentNotFoundError: If agent or persona not found.
        """
        agent = self.get(agent_type)
        persona = self._persona_manager.get_persona(persona_id)
        agent.set_persona(persona)
        return agent

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String showing available agents.
        """
        agents = ", ".join(self._configs.keys())
        return f"AgentFactory(agents=[{agents}])"


# Module-level singleton instance
_factory: Optional[AgentFactory] = None


def get_agent_factory(
    llm_client: Optional[LLMClient] = None,
    capability_registry: Optional[CapabilityRegistry] = None,
    persona_manager: Optional[PersonaManager] = None,
    agents_dir: Optional[Path] = None,
) -> AgentFactory:
    """Get the singleton AgentFactory instance.

    On first call, all parameters must be provided or defaults will be used.
    Subsequent calls return the cached instance, ignoring parameters.

    Args:
        llm_client: LLM client for agent LLM calls.
        capability_registry: Registry of available capabilities.
        persona_manager: Manager for loading personas.
        agents_dir: Path to agent configuration directory.

    Returns:
        The AgentFactory singleton instance.
    """
    global _factory

    if _factory is None:
        # Create dependencies if not provided
        if llm_client is None:
            llm_client = LLMClient()

        if capability_registry is None:
            capability_registry = get_default_registry()

        if persona_manager is None:
            persona_manager = get_persona_manager()

        _factory = AgentFactory(
            llm_client=llm_client,
            capability_registry=capability_registry,
            persona_manager=persona_manager,
            agents_dir=agents_dir,
        )

    return _factory


def reset_agent_factory() -> None:
    """Reset the singleton AgentFactory instance.

    This is primarily useful for testing.
    """
    global _factory
    _factory = None
