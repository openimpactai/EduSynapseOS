# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generic tool registry for managing tool instances.

This module provides the ToolRegistry class for managing tools.
The registry handles tool registration, lookup, and definition aggregation
for LLM tool calling.

The registry is agent-agnostic - it can be used by any agent that
supports tool calling. Domain-specific registries can extend this
class or provide factory functions that populate it.

Example:
    from src.core.tools import ToolRegistry, BaseTool

    # Create registry
    registry = ToolRegistry()

    # Register tools
    registry.register(MyTool())

    # Get definitions for LLM
    tools = registry.get_definitions()

    # Execute a tool
    tool = registry.get("my_tool")
    result = await tool.execute(params, context)
"""

import logging
from typing import Any

from src.core.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool instances.

    Provides centralized management of tool instances with registration,
    lookup, and definition aggregation for LLM integration.

    The registry is designed to be:
    - Agent-agnostic: Can be used by any agent workflow
    - Extensible: Domain registries can extend or wrap this class
    - Simple: Provides only essential registry operations

    Example:
        registry = ToolRegistry()
        registry.register(GetSubjectsTool())
        registry.register(NavigateTool())

        # Get all definitions for LLM
        tools = registry.get_definitions()

        # Execute a specific tool
        tool = registry.get("get_subjects")
        result = await tool.execute(params, context)
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def replace(self, tool: BaseTool) -> None:
        """Register or replace a tool in the registry.

        Unlike `register()`, this method will overwrite an existing
        tool with the same name.

        Args:
            tool: Tool instance to register or replace.
        """
        self._tools[tool.name] = tool
        logger.debug("Registered/replaced tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        del self._tools[name]
        logger.debug("Unregistered tool: %s", name)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool instance.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        return self._tools[name]

    def get_optional(self, name: str) -> BaseTool | None:
        """Get a tool by name, returning None if not found.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool instance or None if not found.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Name of the tool to check.

        Returns:
            True if the tool is registered.
        """
        return name in self._tools

    def list_names(self) -> list[str]:
        """Get names of all registered tools.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def list_all(self) -> list[BaseTool]:
        """Get all registered tool instances.

        Returns:
            List of tool instances.
        """
        return list(self._tools.values())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible definitions for all tools.

        Returns a list of tool definitions ready to be passed
        to the LLM's tools parameter.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [tool.definition for tool in self._tools.values()]

    def get_descriptions(self) -> dict[str, str]:
        """Get name-to-description mapping for all tools.

        Returns:
            Dictionary mapping tool names to their descriptions.
        """
        descriptions = {}
        for name, tool in self._tools.items():
            # Extract description from definition
            func_def = tool.definition.get("function", {})
            descriptions[name] = func_def.get("description", "No description")
        return descriptions

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        logger.debug("Registry cleared")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        return f"ToolRegistry(tools={list(self._tools.keys())})"
