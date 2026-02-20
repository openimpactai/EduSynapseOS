# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""UI element models for structured frontend interactions.

This module provides models for UI elements that tools can return
to enable structured user interactions on the frontend.

Instead of relying on free-text input parsed from LLM responses,
tools can return UI elements that the frontend renders as proper
selection controls (dropdowns, checkboxes, search inputs, etc.).

UI Element Types:
    SINGLE_SELECT: Radio button style, one selection
    MULTI_SELECT: Checkbox style, multiple selections
    SEARCHABLE_SELECT: Dropdown with search capability
    QUICK_REPLIES: Horizontal button row for quick responses
    CONFIRM: Yes/No confirmation dialog
    INFO_CARD: Display-only information card

Usage:
    from src.core.tools.ui_elements import UIElement, UIElementType, UIElementOption

    # Create a single-select UI element
    ui_element = UIElement(
        type=UIElementType.SINGLE_SELECT,
        id="subject_selection",
        title="Choose a Subject",
        options=[
            UIElementOption(id="math-uuid", label="Mathematics", icon="calculator"),
            UIElementOption(id="geo-uuid", label="Geography", icon="globe"),
        ],
        allow_text_input=True,
    )

    # Return from tool
    return ToolResult(
        success=True,
        data={"message": "Which subject?"},
        ui_element=ui_element,
    )
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UIElementType(str, Enum):
    """Types of UI elements that can be displayed by frontend.

    Each type corresponds to a specific UI control:
    - SINGLE_SELECT: Radio buttons or dropdown (one selection)
    - MULTI_SELECT: Checkboxes (multiple selections)
    - SEARCHABLE_SELECT: Dropdown with search input
    - QUICK_REPLIES: Horizontal row of buttons
    - CONFIRM: Confirmation dialog with Yes/No
    - INFO_CARD: Read-only information display
    """

    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    SEARCHABLE_SELECT = "searchable_select"
    QUICK_REPLIES = "quick_replies"
    CONFIRM = "confirm"
    INFO_CARD = "info_card"


class UIElementOption(BaseModel):
    """An option in a selection UI element.

    Represents a single selectable item in a list or grid.
    Options can have icons, descriptions, and additional metadata.

    Attributes:
        id: Unique option identifier (usually UUID from database).
        label: Display label shown to user.
        description: Optional longer description or subtitle.
        icon: Emoji or icon identifier for visual representation.
        disabled: Whether the option is disabled/unselectable.
        metadata: Additional data for frontend logic.
    """

    id: str = Field(description="Unique option identifier")
    label: str = Field(description="Display label")
    description: str | None = Field(default=None, description="Optional description")
    icon: str | None = Field(default=None, description="Emoji or icon identifier")
    disabled: bool = Field(default=False, description="Whether option is disabled")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional data"
    )

    model_config = {"extra": "forbid"}


class UIElement(BaseModel):
    """A UI element for structured user interaction.

    Frontend uses this to render appropriate selection controls
    instead of relying on free-text input. This enables:
    - Better UX with proper selection controls
    - Full data display (all options, not just LLM's subset)
    - Search capability for long lists
    - Consistent option IDs for reliable processing

    Attributes:
        type: Type of UI element to render.
        id: Element identifier for tracking selections.
        title: Optional title/label above the element.
        options: List of selectable options.
        searchable: Enable search input in options.
        allow_text_input: Allow custom text in addition to options.
        required: Whether selection is required to proceed.
        placeholder: Placeholder text for input fields.
        min_selections: Minimum selections for multi_select.
        max_selections: Maximum selections for multi_select.
    """

    type: UIElementType = Field(description="Type of UI element")
    id: str = Field(description="Element identifier for tracking")
    title: str | None = Field(default=None, description="Optional title/label")
    options: list[UIElementOption] = Field(
        default_factory=list, description="Selection options"
    )
    searchable: bool = Field(default=False, description="Enable search in options")
    allow_text_input: bool = Field(
        default=True, description="Allow custom text input"
    )
    required: bool = Field(default=False, description="Whether selection is required")
    placeholder: str | None = Field(default=None, description="Placeholder text")
    min_selections: int = Field(
        default=0, description="Minimum selections for multi_select"
    )
    max_selections: int | None = Field(
        default=None, description="Maximum selections for multi_select"
    )

    model_config = {"extra": "forbid"}


class ConversationState(BaseModel):
    """Tracks what the conversation is waiting for.

    Helps frontend understand the current flow stage and
    what input is expected from the user.

    This state is updated by tools when they return selection
    UI elements, allowing the frontend to track multi-step flows.

    Attributes:
        awaiting: What selection/input is being awaited.
            Examples: "subject_selection", "topic_selection", "confirmation"
        context: Flow context with accumulated selections.
            Examples: {"intent": "practice", "subject_full_code": "UK-NC-2014.MAT"}
        step: Current step number in multi-step flow (1-indexed).
        total_steps: Total steps if known (for progress indicator).
    """

    awaiting: str | None = Field(default=None, description="What is being awaited")
    context: dict[str, Any] = Field(default_factory=dict, description="Flow context")
    step: int = Field(default=0, description="Current step in flow")
    total_steps: int | None = Field(
        default=None, description="Total steps if known"
    )

    model_config = {"extra": "forbid"}
