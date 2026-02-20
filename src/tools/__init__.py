# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared tool implementations for all agents.

This package contains all tool implementations that can be used by any agent.
Tools are organized by functional category:

- curriculum: Subject and topic queries (get_subjects, get_topics)
- student: Student context, notes, reviews (get_student_context, get_parent_notes, get_review_schedule)
- activities: Games and learning activities (get_games, get_activities)
- navigation: Page navigation (navigate)
- agents: Agent interactions like emotions and handoffs (record_emotion, handoff_to_tutor)

The manifest.py file is the central registry of all available tools.
The loader.py file provides factory functions for creating tool registries.

Usage:
    from src.tools import create_registry_from_config, TOOL_MANIFEST

    # Create registry from agent's YAML config
    registry = create_registry_from_config(agent_config.tools)

    # Or get all tools
    registry = get_default_tool_registry()

Adding a New Tool:
    1. Create tool class in the appropriate category folder
    2. Add entry to TOOL_MANIFEST in manifest.py
    3. Enable in agent's YAML config (config/agents/*.yaml)
"""

from src.tools.loader import (
    create_registry_from_config,
    get_default_tool_registry,
    load_tool_class,
)
from src.tools.manifest import (
    TOOL_MANIFEST,
    get_available_tool_names,
    get_tool_info,
    get_tools_by_category,
)

__all__ = [
    # Factory functions
    "create_registry_from_config",
    "get_default_tool_registry",
    "load_tool_class",
    # Manifest access
    "TOOL_MANIFEST",
    "get_available_tool_names",
    "get_tool_info",
    "get_tools_by_category",
]
