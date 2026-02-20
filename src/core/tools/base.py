# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base classes for the tool system.

This module defines the foundational classes for the tool system:
- ToolContext: Context available to tools during execution
- ToolResult: Standardized result from tool execution with UI element support
- BaseTool: Abstract base class for all tools

Tools are executed by agent workflows when the LLM requests
specific actions through tool calling. The tool system is designed
to be agent-agnostic and reusable across different workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.emotional.context import EmotionalContext
    from src.core.tools.ui_elements import UIElement
    from src.models.memory import FullMemoryContext


@dataclass
class ToolContext:
    """Context available to tools during execution.

    Provides access to tenant, user, and session information
    along with pre-loaded memory and emotional context.

    This context is passed to every tool execution, allowing tools
    to access necessary information without needing to query it themselves.

    The context supports different user types (student, teacher, parent)
    through the user_id and user_type fields. For backwards compatibility
    with student tools, student_id property is provided as an alias for user_id.

    Attributes:
        tenant_code: Tenant identifier for database access.
        user_id: User performing the action (student, teacher, or parent).
        user_type: Type of user ("student", "teacher", "parent").
        grade_level: Grade level sequence number (e.g., 5 for Year 5/Grade 5).
        language: User's language preference (e.g., "en", "tr").
        framework_code: Curriculum framework code (e.g., "UK-NC-2014", "US-CCSS-2024").
            Used to filter subjects/topics to the student's curriculum.
        grade_code: Grade code within framework (e.g., "Y5", "G5", "P5", "STD5").
            Used for precise curriculum filtering.
        session: Database session for queries.
        memory_context: Pre-loaded 4-layer memory context (student only).
        emotional_context: Pre-loaded emotional state context (student only).
        extra: Additional context that tools might need.
    """

    tenant_code: str
    user_id: UUID
    user_type: str  # "student", "teacher", "parent"
    grade_level: int
    language: str
    framework_code: str | None = None
    grade_code: str | None = None
    session: "AsyncSession | None" = None
    memory_context: "FullMemoryContext | None" = None
    emotional_context: "EmotionalContext | None" = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def student_id(self) -> UUID:
        """Backwards compatibility alias for user_id.

        Student tools can continue using context.student_id.
        For teacher/parent tools, this still returns user_id but
        those tools should use user_id directly.

        Returns:
            The user_id (same as user_id field).
        """
        return self.user_id

    @property
    def is_student(self) -> bool:
        """Check if the current user is a student."""
        return self.user_type == "student"

    @property
    def is_teacher(self) -> bool:
        """Check if the current user is a teacher."""
        return self.user_type == "teacher"

    @property
    def is_parent(self) -> bool:
        """Check if the current user is a parent."""
        return self.user_type == "parent"


@dataclass
class ToolResult:
    """Result from tool execution with UI element support.

    Standardized response format for all tool executions.
    Extended to support:
    - UI element suggestions for frontend rendering
    - Data passthrough to frontend (raw tool data)
    - Conversation state updates for flow tracking
    - Tool chaining control

    Attributes:
        success: Whether the tool executed successfully.
        data: Tool-specific result data. Should include a 'message' key
            for human-readable output to LLM.
        error: Error message if success is False.
        ui_element: Optional UI element suggestion for frontend.
            When provided, frontend can render structured selection UI.
        passthrough_data: Data to pass through to frontend response.
            This data is included in the API response's tool_data field.
        state_update: Conversation state update. Used to track
            what the conversation is waiting for (e.g., "topic_selection").
        stop_chaining: If True, prevents further tool calls in this turn.
            Use this for action tools (navigate, handoff) that complete
            a conversation flow. Info tools should leave this False to
            allow the LLM to chain multiple tool calls in a single turn.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # UI element suggestion for frontend
    ui_element: "UIElement | None" = None

    # Data to pass through to frontend (raw tool data)
    passthrough_data: dict[str, Any] | None = None

    # Conversation state update
    state_update: dict[str, Any] | None = None

    # Tool chaining control - if True, stops further tool calls in this turn
    stop_chaining: bool = False

    def to_llm_message(self) -> str:
        """Convert result to message for LLM.

        Returns the human-readable message from tool execution.
        The frontend receives structured data separately via passthrough_data.

        Returns:
            Human-readable message describing the tool result.
        """
        if not self.success:
            return f"Error: {self.error}"

        message = self.data.get("message", "")
        return message if message else "Operation completed successfully."


class BaseTool(ABC):
    """Abstract base class for all tools.

    All tools must inherit from this class and implement
    the required properties and methods.

    The tool lifecycle:
    1. LLM receives tool definitions via `definition` property
    2. LLM calls tool with arguments
    3. Workflow executes tool via `execute()` method
    4. Result is sent back to LLM for response generation
    5. UI elements and passthrough data are sent to frontend

    Tools can be used by any agent that supports tool calling.
    The same tool implementation can be reused across different
    agents and workflows.

    Example:
        class GetSubjectsTool(BaseTool):
            @property
            def name(self) -> str:
                return "get_subjects"

            @property
            def definition(self) -> dict[str, Any]:
                return {
                    "type": "function",
                    "function": {
                        "name": "get_subjects",
                        "description": "Get available subjects",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }

            async def execute(
                self, params: dict[str, Any], context: ToolContext
            ) -> ToolResult:
                subjects = await self._fetch_subjects(context)
                return ToolResult(
                    success=True,
                    data={"subjects": subjects, "message": "Found 3 subjects"},
                    ui_element=UIElement(type="single_select", ...),
                    passthrough_data={"subjects": subjects},
                )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name matching the function name in definition.

        This must match the 'name' field in the function definition
        to ensure proper routing of tool calls from the LLM.

        Returns:
            Tool name as string.
        """
        pass

    @property
    @abstractmethod
    def definition(self) -> dict[str, Any]:
        """OpenAI-compatible tool definition.

        Returns the tool schema in OpenAI's function calling format.
        This is sent to the LLM so it knows how to call the tool.

        The definition should include:
        - type: "function"
        - function.name: Same as the `name` property
        - function.description: Clear description of what the tool does
        - function.parameters: JSON schema for parameters

        Returns:
            Dictionary with tool definition in OpenAI format.
        """
        pass

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with given parameters.

        Called by the workflow when the LLM requests this tool.
        Implementations should validate params and use context
        to perform the required action.

        Args:
            params: Arguments from the LLM's tool call.
            context: Execution context with tenant, student, and session.

        Returns:
            ToolResult with success status, data, and optional UI elements.
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate parameters before execution.

        Override this method to add custom validation logic.
        Raises ValueError if parameters are invalid.

        Args:
            params: Parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        pass
