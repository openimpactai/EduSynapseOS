# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Persona manager for EduSynapseOS.

This module provides the PersonaManager class which handles persona selection,
caching, and application to agent responses. It serves as the main interface
for the persona system.
"""

from pathlib import Path
from typing import Optional

from src.core.personas.loader import load_all_personas, load_persona
from src.core.personas.models import Persona
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaNotFoundError(Exception):
    """Raised when a requested persona is not found."""

    pass


class PersonaManager:
    """Manages persona loading, caching, and selection.

    The PersonaManager is responsible for:
    - Loading personas from YAML configuration files
    - Caching loaded personas for efficient access
    - Selecting appropriate personas based on user preferences
    - Providing persona data for agent prompt construction

    Attributes:
        _personas: Cache of loaded personas
        _default_persona_id: ID of the default persona to use
        _personas_dir: Path to the personas configuration directory
    """

    def __init__(
        self,
        personas_dir: Optional[Path] = None,
        default_persona_id: str = "tutor",
        auto_load: bool = True,
    ):
        """Initialize the PersonaManager.

        Args:
            personas_dir: Optional path to personas configuration directory
            default_persona_id: ID of the default persona to use
            auto_load: Whether to automatically load all personas on init
        """
        self._personas: dict[str, Persona] = {}
        self._default_persona_id = default_persona_id
        self._personas_dir = personas_dir
        self._loaded = False

        if auto_load:
            self._load_personas()

    def _load_personas(self) -> None:
        """Load all personas from the configuration directory."""
        try:
            self._personas = load_all_personas(self._personas_dir)
            self._loaded = True
            logger.info(
                "persona_manager_initialized",
                persona_count=len(self._personas),
                default_persona=self._default_persona_id,
            )
        except Exception as e:
            logger.error(
                "persona_manager_init_failed",
                error=str(e),
            )
            self._personas = {}
            self._loaded = False

    def reload(self) -> None:
        """Reload all personas from disk.

        This is useful when persona configurations are updated at runtime.
        """
        logger.info("reloading_personas")
        self._load_personas()

    def get_persona(self, persona_id: str) -> Persona:
        """Get a persona by its ID.

        Args:
            persona_id: The unique identifier of the persona

        Returns:
            The requested Persona object

        Raises:
            PersonaNotFoundError: If the persona is not found
        """
        if not self._loaded:
            self._load_personas()

        if persona_id in self._personas:
            return self._personas[persona_id]

        # Try to load it directly if not in cache
        try:
            persona = load_persona(persona_id, self._personas_dir)
            if persona.enabled:
                self._personas[persona_id] = persona
                return persona
            raise PersonaNotFoundError(
                f"Persona '{persona_id}' exists but is disabled"
            )
        except Exception as e:
            raise PersonaNotFoundError(
                f"Persona '{persona_id}' not found: {e}"
            )

    def get_default_persona(self) -> Persona:
        """Get the default persona.

        Returns:
            The default Persona object

        Raises:
            PersonaNotFoundError: If the default persona is not found
        """
        return self.get_persona(self._default_persona_id)

    def get_persona_for_user(
        self,
        user_type: Optional[str] = None,
        preferred_persona_id: Optional[str] = None,
        age_group: Optional[str] = None,
    ) -> Persona:
        """Get the most suitable persona for a user.

        This method selects a persona based on user preferences and characteristics.
        Priority order:
        1. User's preferred persona (if valid and enabled)
        2. Persona matching user type
        3. Persona matching age group
        4. Default persona

        Args:
            user_type: Type of user (e.g., "student", "young_learner", "advanced")
            preferred_persona_id: User's preferred persona ID
            age_group: User's age group (e.g., "child", "teen", "adult")

        Returns:
            The most suitable Persona object
        """
        # Try user's preferred persona first
        if preferred_persona_id:
            try:
                return self.get_persona(preferred_persona_id)
            except PersonaNotFoundError:
                logger.warning(
                    "preferred_persona_not_found",
                    preferred_id=preferred_persona_id,
                )

        if not self._loaded:
            self._load_personas()

        # Try to find a persona suitable for the user type
        if user_type:
            for persona in self._personas.values():
                if persona.suitable_for and user_type in persona.suitable_for:
                    logger.debug(
                        "selected_persona_by_user_type",
                        persona_id=persona.id,
                        user_type=user_type,
                    )
                    return persona

        # Try to find a persona suitable for the age group
        if age_group:
            for persona in self._personas.values():
                if persona.suitable_for and age_group in persona.suitable_for:
                    logger.debug(
                        "selected_persona_by_age_group",
                        persona_id=persona.id,
                        age_group=age_group,
                    )
                    return persona

        # Fall back to default
        return self.get_default_persona()

    def list_personas(self, include_disabled: bool = False) -> list[Persona]:
        """List all available personas.

        Args:
            include_disabled: Whether to include disabled personas

        Returns:
            List of Persona objects
        """
        if not self._loaded:
            self._load_personas()

        if include_disabled:
            return list(self._personas.values())
        return [p for p in self._personas.values() if p.enabled]

    def list_persona_ids(self) -> list[str]:
        """List all available persona IDs.

        Returns:
            List of persona ID strings
        """
        if not self._loaded:
            self._load_personas()

        return list(self._personas.keys())

    def has_persona(self, persona_id: str) -> bool:
        """Check if a persona exists and is enabled.

        Args:
            persona_id: The persona ID to check

        Returns:
            True if the persona exists and is enabled
        """
        if not self._loaded:
            self._load_personas()

        if persona_id in self._personas:
            return self._personas[persona_id].enabled
        return False

    def get_persona_summary(self, persona_id: str) -> dict:
        """Get a summary of a persona for display purposes.

        Args:
            persona_id: The persona ID

        Returns:
            Dictionary with persona summary information

        Raises:
            PersonaNotFoundError: If the persona is not found
        """
        persona = self.get_persona(persona_id)
        return {
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "role": persona.identity.role,
            "tone": persona.voice.tone.value,
            "formality": persona.voice.formality.value,
            "suitable_for": persona.suitable_for or [],
            "tags": persona.tags or [],
        }

    def get_all_summaries(self) -> list[dict]:
        """Get summaries of all available personas.

        Returns:
            List of persona summary dictionaries
        """
        if not self._loaded:
            self._load_personas()

        return [
            self.get_persona_summary(persona_id)
            for persona_id in self._personas.keys()
        ]


# Module-level singleton instance
_manager: Optional[PersonaManager] = None


def get_persona_manager(
    personas_dir: Optional[Path] = None,
    default_persona_id: str = "tutor",
) -> PersonaManager:
    """Get the singleton PersonaManager instance.

    Args:
        personas_dir: Optional path to personas configuration directory
        default_persona_id: ID of the default persona to use

    Returns:
        The PersonaManager singleton instance
    """
    global _manager
    if _manager is None:
        _manager = PersonaManager(
            personas_dir=personas_dir,
            default_persona_id=default_persona_id,
        )
    return _manager


def reset_persona_manager() -> None:
    """Reset the singleton PersonaManager instance.

    This is primarily useful for testing.
    """
    global _manager
    _manager = None
