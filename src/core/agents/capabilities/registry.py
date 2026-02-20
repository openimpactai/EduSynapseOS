# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Capability registry for managing and accessing agent capabilities.

The registry provides:
- Central registration of all capabilities
- Lookup by name
- Capability discovery and listing
- Default capability set for agents

Usage:
    # Get default registry with all built-in capabilities
    registry = CapabilityRegistry.default()

    # Get a specific capability
    question_gen = registry.get("question_generation")

    # List all available capabilities
    for name in registry.list_names():
        cap = registry.get(name)
        print(f"{name}: {cap.description}")
"""

import logging
from typing import Iterator

from src.core.agents.capabilities.base import Capability

logger = logging.getLogger(__name__)


class CapabilityNotFoundError(Exception):
    """Raised when a requested capability is not found.

    Attributes:
        capability_name: Name of the capability that was not found.
        available: List of available capability names.
    """

    def __init__(self, capability_name: str, available: list[str]):
        self.capability_name = capability_name
        self.available = available
        message = (
            f"Capability '{capability_name}' not found. "
            f"Available: {', '.join(available)}"
        )
        super().__init__(message)


class CapabilityRegistry:
    """Registry for managing agent capabilities.

    Provides central registration and lookup of capabilities.
    Supports both built-in and custom capabilities.

    Attributes:
        _capabilities: Dictionary mapping names to capability instances.

    Example:
        # Create registry with custom capabilities
        registry = CapabilityRegistry()
        registry.register(MyCustomCapability())

        # Or use default with all built-ins
        registry = CapabilityRegistry.default()

        # Get capability by name
        cap = registry.get("question_generation")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._capabilities: dict[str, Capability] = {}

    @classmethod
    def default(cls) -> "CapabilityRegistry":
        """Create a registry with all built-in capabilities.

        Returns:
            Registry populated with default capabilities.
        """
        from src.core.agents.capabilities.answer_evaluation import (
            AnswerEvaluationCapability,
        )
        from src.core.agents.capabilities.concept_explanation import (
            ConceptExplanationCapability,
        )
        from src.core.agents.capabilities.diagnostic_analysis import (
            DiagnosticAnalysisCapability,
        )
        from src.core.agents.capabilities.feedback_generation import (
            FeedbackGenerationCapability,
        )
        from src.core.agents.capabilities.message_analysis import (
            MessageAnalysisCapability,
        )
        from src.core.agents.capabilities.question_generation import (
            QuestionGenerationCapability,
        )
        # Companion capabilities
        from src.core.agents.capabilities.wellbeing_check import (
            WellbeingCheckCapability,
        )
        from src.core.agents.capabilities.emotional_support import (
            EmotionalSupportCapability,
        )
        from src.core.agents.capabilities.activity_guidance import (
            ActivityGuidanceCapability,
        )
        from src.core.agents.capabilities.companion_decision import (
            CompanionDecisionCapability,
        )
        # Gaming capabilities
        from src.core.agents.capabilities.game_move_analysis import (
            GameMoveAnalysisCapability,
        )
        from src.core.agents.capabilities.game_coach_response import (
            GameCoachResponseCapability,
        )
        from src.core.agents.capabilities.game_hint_generation import (
            GameHintGenerationCapability,
        )

        registry = cls()

        # Register all built-in capabilities
        registry.register(QuestionGenerationCapability())
        registry.register(AnswerEvaluationCapability())
        registry.register(ConceptExplanationCapability())
        registry.register(FeedbackGenerationCapability())
        registry.register(DiagnosticAnalysisCapability())
        registry.register(MessageAnalysisCapability())
        # Companion capabilities
        registry.register(WellbeingCheckCapability())
        registry.register(EmotionalSupportCapability())
        registry.register(ActivityGuidanceCapability())
        registry.register(CompanionDecisionCapability())
        # Gaming capabilities
        registry.register(GameMoveAnalysisCapability())
        registry.register(GameCoachResponseCapability())
        registry.register(GameHintGenerationCapability())

        logger.info(
            "Created default CapabilityRegistry with %d capabilities",
            len(registry),
        )

        return registry

    def register(self, capability: Capability) -> None:
        """Register a capability in the registry.

        Args:
            capability: The capability instance to register.

        Raises:
            ValueError: If a capability with the same name exists.
        """
        name = capability.name

        if name in self._capabilities:
            raise ValueError(
                f"Capability '{name}' is already registered. "
                f"Use replace() to override."
            )

        self._capabilities[name] = capability
        logger.debug("Registered capability: %s", name)

    def replace(self, capability: Capability) -> None:
        """Replace an existing capability or register a new one.

        Args:
            capability: The capability instance to register.
        """
        name = capability.name

        if name in self._capabilities:
            logger.debug("Replacing capability: %s", name)
        else:
            logger.debug("Registering new capability: %s", name)

        self._capabilities[name] = capability

    def unregister(self, name: str) -> bool:
        """Remove a capability from the registry.

        Args:
            name: Name of the capability to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._capabilities:
            del self._capabilities[name]
            logger.debug("Unregistered capability: %s", name)
            return True
        return False

    def get(self, name: str) -> Capability:
        """Get a capability by name.

        Args:
            name: Name of the capability.

        Returns:
            The capability instance.

        Raises:
            CapabilityNotFoundError: If capability not found.
        """
        if name not in self._capabilities:
            raise CapabilityNotFoundError(
                capability_name=name,
                available=self.list_names(),
            )
        return self._capabilities[name]

    def get_optional(self, name: str) -> Capability | None:
        """Get a capability by name, returning None if not found.

        Args:
            name: Name of the capability.

        Returns:
            The capability instance or None.
        """
        return self._capabilities.get(name)

    def has(self, name: str) -> bool:
        """Check if a capability is registered.

        Args:
            name: Name of the capability.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._capabilities

    def list_names(self) -> list[str]:
        """List all registered capability names.

        Returns:
            List of capability names.
        """
        return list(self._capabilities.keys())

    def list_all(self) -> list[Capability]:
        """List all registered capability instances.

        Returns:
            List of capability instances.
        """
        return list(self._capabilities.values())

    def get_descriptions(self) -> dict[str, str]:
        """Get descriptions of all registered capabilities.

        Returns:
            Dictionary mapping names to descriptions.
        """
        return {
            name: cap.description
            for name, cap in self._capabilities.items()
        }

    def subset(self, names: list[str]) -> "CapabilityRegistry":
        """Create a new registry with a subset of capabilities.

        Args:
            names: Names of capabilities to include.

        Returns:
            New registry with only the specified capabilities.

        Raises:
            CapabilityNotFoundError: If any name is not found.
        """
        new_registry = CapabilityRegistry()

        for name in names:
            capability = self.get(name)  # Raises if not found
            new_registry._capabilities[name] = capability

        return new_registry

    def merge(self, other: "CapabilityRegistry") -> None:
        """Merge another registry into this one.

        Capabilities from the other registry will override
        capabilities with the same name in this registry.

        Args:
            other: Registry to merge from.
        """
        for name, capability in other._capabilities.items():
            self._capabilities[name] = capability

        logger.debug("Merged %d capabilities from other registry", len(other))

    def __len__(self) -> int:
        """Return the number of registered capabilities."""
        return len(self._capabilities)

    def __contains__(self, name: str) -> bool:
        """Check if a capability is registered."""
        return name in self._capabilities

    def __iter__(self) -> Iterator[str]:
        """Iterate over capability names."""
        return iter(self._capabilities)

    def __repr__(self) -> str:
        """Return string representation."""
        names = ", ".join(self._capabilities.keys())
        return f"CapabilityRegistry([{names}])"


# Global default registry instance (lazy-loaded)
_default_registry: CapabilityRegistry | None = None


def get_default_registry() -> CapabilityRegistry:
    """Get or create the global default registry.

    Returns:
        The default capability registry.
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = CapabilityRegistry.default()

    return _default_registry


def reset_default_registry() -> None:
    """Reset the global default registry.

    Useful for testing or reconfiguration.
    """
    global _default_registry
    _default_registry = None
