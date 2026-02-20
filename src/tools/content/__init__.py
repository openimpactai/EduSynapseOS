# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Content Creation Tools.

This package provides tools for the Content Creation multi-agent system.
Tools are organized by category:

- curriculum: Subject, topic, and learning objective queries
- h5p: H5P content type information and schema tools
- media: Image and diagram generation tools
- handoff: Agent handoff tools for delegation
- export: H5P export, preview, and draft management
- quality: Content quality checking tools
- knowledge: Topic concepts and knowledge base search
- inference: User intent extraction and smart inference

Usage:
    from src.tools.content import get_content_tool_registry

    # Get registry with all content creation tools
    registry = get_content_tool_registry()

    # Or import specific tools
    from src.tools.content.h5p import GetH5PContentTypesTool
"""

from src.tools.content.manifest import CONTENT_CREATION_TOOLS, get_content_tool_names


def get_content_tool_registry():
    """Create a ToolRegistry with all content creation tools.

    Returns:
        ToolRegistry with registered content creation tools.
    """
    from src.core.tools import ToolRegistry
    from src.tools.loader import load_tool_class

    registry = ToolRegistry()

    for tool_name, tool_info in CONTENT_CREATION_TOOLS.items():
        try:
            tool_class = load_tool_class(tool_info["class_path"])
            tool_instance = tool_class()
            tool_instance.group = tool_info.get("group", "general")
            tool_instance.category = tool_info["category"]
            registry.register(tool_instance)
        except (ImportError, AttributeError):
            # Tool not yet implemented
            pass

    return registry


__all__ = [
    "CONTENT_CREATION_TOOLS",
    "get_content_tool_names",
    "get_content_tool_registry",
]
