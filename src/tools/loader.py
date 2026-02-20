# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tool loader and registry factory.

This module provides the central factory functions for creating tool registries
from agent YAML configurations. It is the SINGLE place where tools are loaded
and registered, ensuring consistency across all agents.

Usage:
    from src.tools import create_registry_from_config

    # In any workflow
    registry = create_registry_from_config(agent_config.tools)
    agent = DynamicAgent(tool_registry=registry, ...)

The loader dynamically imports tool classes from the manifest, so no hardcoded
imports are needed. Adding a new tool only requires updating manifest.py.
"""

import importlib
import logging
from typing import TYPE_CHECKING

from src.core.tools import BaseTool, ToolRegistry
from src.tools.manifest import TOOL_MANIFEST

if TYPE_CHECKING:
    from src.core.agents.context import ToolsConfig

logger = logging.getLogger(__name__)


def load_tool_class(class_path: str) -> type[BaseTool]:
    """Dynamically load a tool class from its path.

    Args:
        class_path: Fully qualified class path in format "module.path:ClassName"

    Returns:
        The tool class (not an instance).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class doesn't exist in the module.

    Example:
        tool_class = load_tool_class("src.tools.curriculum.get_subjects:GetSubjectsTool")
        tool = tool_class()
    """
    module_path, class_name = class_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_default_tool_registry() -> ToolRegistry:
    """Get a registry with ALL available tools registered.

    This loads every tool defined in the manifest. Useful for testing
    or when you want all tools available regardless of agent config.

    For production, prefer create_registry_from_config() which only
    loads tools enabled in the agent's YAML configuration.

    Returns:
        ToolRegistry with all tools from manifest.

    Raises:
        ImportError: If any tool module cannot be imported.
        AttributeError: If any tool class doesn't exist.
    """
    registry = ToolRegistry()

    for tool_name, tool_info in TOOL_MANIFEST.items():
        try:
            tool_class = load_tool_class(tool_info["class_path"])
            tool_instance = tool_class()

            # Store category as metadata on the tool
            tool_instance.category = tool_info["category"]  # type: ignore[attr-defined]

            registry.register(tool_instance)
            logger.debug(
                "Registered tool: %s (category=%s)",
                tool_name,
                tool_info["category"],
            )
        except (ImportError, AttributeError) as e:
            logger.error(
                "Failed to load tool '%s' from '%s': %s",
                tool_name,
                tool_info["class_path"],
                e,
            )
            raise

    logger.info("Default tool registry created with %d tools", len(registry))
    return registry


def create_registry_from_config(tools_config: "ToolsConfig") -> ToolRegistry:
    """Create tool registry from agent YAML configuration.

    This is the CENTRAL factory function used by ALL agents and workflows.
    Only registers tools that are enabled in the config, in the order
    specified by their 'order' field.

    Args:
        tools_config: Tool configuration from agent YAML.
            Expected structure in YAML:
            tools:
              enabled: true
              max_iterations: 5
              definitions:
                - name: get_subjects
                  enabled: true
                  group: information_gathering
                  order: 1

    Returns:
        ToolRegistry with only enabled tools registered.

    Raises:
        ValueError: If a tool name in config doesn't exist in manifest.

    Example:
        # In CompanionWorkflow or any workflow
        from src.tools import create_registry_from_config

        registry = create_registry_from_config(self._agent_config.tools)
        self._agent = DynamicAgent(tool_registry=registry, ...)
    """
    registry = ToolRegistry()

    # Sort by order for consistent registration
    sorted_definitions = sorted(tools_config.definitions, key=lambda d: d.order)

    for tool_def in sorted_definitions:
        if not tool_def.enabled:
            logger.debug("Skipping disabled tool: %s", tool_def.name)
            continue

        if tool_def.name not in TOOL_MANIFEST:
            available = list(TOOL_MANIFEST.keys())
            raise ValueError(
                f"Unknown tool '{tool_def.name}'. "
                f"Available tools: {available}. "
                f"Add the tool to src/tools/manifest.py first."
            )

        tool_info = TOOL_MANIFEST[tool_def.name]

        try:
            tool_class = load_tool_class(tool_info["class_path"])
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to load tool '{tool_def.name}' "
                f"from '{tool_info['class_path']}': {e}"
            ) from e

        tool_instance = tool_class()

        # Store group (from YAML) and category (from manifest) on the tool
        tool_instance.group = tool_def.group  # type: ignore[attr-defined]
        tool_instance.category = tool_info["category"]  # type: ignore[attr-defined]

        registry.register(tool_instance)
        logger.debug(
            "Registered tool: %s (category=%s, group=%s, order=%d)",
            tool_def.name,
            tool_info["category"],
            tool_def.group,
            tool_def.order,
        )

    logger.info(
        "Created tool registry with %d/%d tools enabled",
        len(registry),
        len(tools_config.definitions),
    )

    return registry
