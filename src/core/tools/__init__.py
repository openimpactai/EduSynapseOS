# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core tool infrastructure for LLM tool calling.

This package provides the foundational classes for building tools
that can be used by any agent supporting tool calling.

The tool system is agent-agnostic - tools defined here can be used
by Companion, Practice, or any future agent workflow.

Base Classes:
    BaseTool: Abstract base class for all tools
    ToolContext: Context available to tools during execution
    ToolResult: Standardized result from tool execution

Registry:
    ToolRegistry: Generic registry for managing tool instances

UI Elements:
    UIElement: Structured UI element for frontend rendering
    UIElementType: Types of UI elements (single_select, searchable_select, etc.)
    UIElementOption: Option in a selection UI element
    ConversationState: Tracks conversation flow state

Usage:
    from src.core.tools import BaseTool, ToolContext, ToolResult
    from src.core.tools import ToolRegistry
    from src.core.tools import UIElement, UIElementType

    class MyTool(BaseTool):
        @property
        def name(self) -> str:
            return "my_tool"

        @property
        def definition(self) -> dict:
            return {...}

        async def execute(self, params, context) -> ToolResult:
            return ToolResult(success=True, data={...})
"""

from src.core.tools.base import BaseTool, ToolContext, ToolResult
from src.core.tools.registry import ToolRegistry
from src.core.tools.ui_elements import (
    ConversationState,
    UIElement,
    UIElementOption,
    UIElementType,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolContext",
    "ToolResult",
    # Registry
    "ToolRegistry",
    # UI Elements
    "UIElement",
    "UIElementType",
    "UIElementOption",
    "ConversationState",
]
