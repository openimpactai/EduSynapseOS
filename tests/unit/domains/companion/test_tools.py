# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for companion tools.

Tests the companion tool implementations:
- NavigateTool
- RecordEmotionTool
- HandoffToTutorTool
- GetActivitiesTool
- GetStudentContextTool
- GetParentNotesTool
- GetReviewScheduleTool

These tests mock database dependencies where needed.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from src.domains.companion.tools.base import ToolContext, ToolResult
from src.domains.companion.tools.navigate import NavigateTool
from src.domains.companion.tools.record_emotion import (
    RecordEmotionTool,
    VALID_EMOTIONS,
    VALID_INTENSITIES,
)
from src.domains.companion.tools.handoff_to_tutor import HandoffToTutorTool
from src.domains.companion.tools.registry import (
    ToolRegistry,
    get_companion_tool_registry,
    reset_companion_tool_registry,
)


@pytest.fixture
def mock_context() -> ToolContext:
    """Create a mock tool context for testing."""
    mock_session = AsyncMock()
    return ToolContext(
        tenant_code="test_school",
        student_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        grade_level=5,
        language="en",
        session=mock_session,
    )


class TestNavigateTool:
    """Tests for NavigateTool."""

    def test_name(self):
        """Tool name matches definition."""
        tool = NavigateTool()
        assert tool.name == "navigate"

    def test_definition_structure(self):
        """Definition has correct structure."""
        tool = NavigateTool()
        definition = tool.definition

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "navigate"
        assert "description" in definition["function"]
        assert "parameters" in definition["function"]
        assert "target" in definition["function"]["parameters"]["properties"]
        assert "label" in definition["function"]["parameters"]["properties"]
        assert "target" in definition["function"]["parameters"]["required"]
        assert "label" in definition["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_context: ToolContext):
        """Navigate with valid params returns action."""
        tool = NavigateTool()
        params = {
            "target": "practice",
            "label": "Start Practice",
        }

        result = await tool.execute(params, mock_context)

        assert result.success is True
        assert result.data["action"]["type"] == "navigate"
        assert result.data["action"]["target"] == "practice"
        assert result.data["action"]["label"] == "Start Practice"

    @pytest.mark.asyncio
    async def test_execute_with_params(self, mock_context: ToolContext):
        """Navigate with route params includes them in action."""
        tool = NavigateTool()
        params = {
            "target": "practice",
            "label": "Practice Math",
            "params": {"topic_id": "math-101", "difficulty": "easy"},
        }

        result = await tool.execute(params, mock_context)

        assert result.success is True
        assert result.data["action"]["params"]["topic_id"] == "math-101"
        assert result.data["action"]["params"]["difficulty"] == "easy"

    @pytest.mark.asyncio
    async def test_execute_missing_target(self, mock_context: ToolContext):
        """Navigate without target fails."""
        tool = NavigateTool()
        params = {"label": "Go Somewhere"}

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "target" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_label(self, mock_context: ToolContext):
        """Navigate without label fails."""
        tool = NavigateTool()
        params = {"target": "practice"}

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "label" in result.error.lower()


class TestRecordEmotionTool:
    """Tests for RecordEmotionTool."""

    def test_name(self):
        """Tool name matches definition."""
        tool = RecordEmotionTool()
        assert tool.name == "record_emotion"

    def test_definition_structure(self):
        """Definition has correct structure."""
        tool = RecordEmotionTool()
        definition = tool.definition

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "record_emotion"
        assert "emotion" in definition["function"]["parameters"]["properties"]
        assert "intensity" in definition["function"]["parameters"]["properties"]
        assert "triggers" in definition["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_context: ToolContext):
        """Record emotion with valid params succeeds."""
        tool = RecordEmotionTool()
        params = {
            "emotion": "frustrated",
            "intensity": "high",
            "triggers": ["math_difficulty"],
        }

        result = await tool.execute(params, mock_context)

        assert result.success is True
        assert result.data["recorded"] is True
        assert result.data["emotion"] == "frustrated"
        assert result.data["intensity"] == "high"
        assert "math_difficulty" in result.data["triggers"]

    @pytest.mark.asyncio
    async def test_execute_without_triggers(self, mock_context: ToolContext):
        """Record emotion works without triggers."""
        tool = RecordEmotionTool()
        params = {
            "emotion": "happy",
            "intensity": "moderate",
        }

        result = await tool.execute(params, mock_context)

        assert result.success is True
        assert result.data["emotion"] == "happy"
        assert result.data["triggers"] == []

    @pytest.mark.asyncio
    async def test_execute_invalid_emotion(self, mock_context: ToolContext):
        """Record emotion with invalid emotion fails."""
        tool = RecordEmotionTool()
        params = {
            "emotion": "invalid_emotion",
            "intensity": "high",
        }

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_intensity(self, mock_context: ToolContext):
        """Record emotion with invalid intensity fails."""
        tool = RecordEmotionTool()
        params = {
            "emotion": "happy",
            "intensity": "invalid_level",
        }

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_emotion(self, mock_context: ToolContext):
        """Record emotion without emotion fails."""
        tool = RecordEmotionTool()
        params = {"intensity": "high"}

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "emotion" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_intensity(self, mock_context: ToolContext):
        """Record emotion without intensity fails."""
        tool = RecordEmotionTool()
        params = {"emotion": "happy"}

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "intensity" in result.error.lower()

    def test_all_valid_emotions(self):
        """All valid emotions are defined."""
        expected_emotions = {
            "happy", "excited", "confident", "curious",
            "neutral", "bored", "confused", "frustrated",
            "anxious", "tired"
        }
        assert VALID_EMOTIONS == expected_emotions

    def test_all_valid_intensities(self):
        """All valid intensities are defined."""
        expected_intensities = {"low", "moderate", "high"}
        assert VALID_INTENSITIES == expected_intensities


class TestHandoffToTutorTool:
    """Tests for HandoffToTutorTool."""

    def test_name(self):
        """Tool name matches definition."""
        tool = HandoffToTutorTool()
        assert tool.name == "handoff_to_tutor"

    def test_definition_structure(self):
        """Definition has correct structure."""
        tool = HandoffToTutorTool()
        definition = tool.definition

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "handoff_to_tutor"
        assert "question" in definition["function"]["parameters"]["properties"]
        assert "question" in definition["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_context: ToolContext):
        """Handoff with valid question returns action."""
        tool = HandoffToTutorTool()
        params = {
            "question": "How do I add fractions?",
            "topic": "fractions",
        }

        result = await tool.execute(params, mock_context)

        assert result.success is True
        assert result.data["action"]["type"] == "handoff"
        assert result.data["action"]["target"] == "tutor"
        assert result.data["action"]["params"]["question"] == "How do I add fractions?"

    @pytest.mark.asyncio
    async def test_execute_missing_question(self, mock_context: ToolContext):
        """Handoff without question fails."""
        tool = HandoffToTutorTool()
        params = {"topic": "fractions"}

        result = await tool.execute(params, mock_context)

        assert result.success is False
        assert "question" in result.error.lower()


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_companion_tool_registry()

    def test_register_tool(self):
        """Register adds tool to registry."""
        registry = ToolRegistry()
        tool = NavigateTool()

        registry.register(tool)

        assert registry.has("navigate")
        assert len(registry) == 1

    def test_register_duplicate_fails(self):
        """Registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool = NavigateTool()

        registry.register(tool)

        with pytest.raises(ValueError):
            registry.register(tool)

    def test_replace_tool(self):
        """Replace overwrites existing tool."""
        registry = ToolRegistry()
        tool1 = NavigateTool()
        tool2 = NavigateTool()

        registry.register(tool1)
        registry.replace(tool2)

        assert len(registry) == 1

    def test_get_tool(self):
        """Get returns registered tool."""
        registry = ToolRegistry()
        tool = NavigateTool()
        registry.register(tool)

        retrieved = registry.get("navigate")

        assert retrieved is tool

    def test_get_missing_tool_raises(self):
        """Get missing tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_optional_returns_none(self):
        """Get optional returns None for missing tool."""
        registry = ToolRegistry()

        result = registry.get_optional("nonexistent")

        assert result is None

    def test_unregister_tool(self):
        """Unregister removes tool from registry."""
        registry = ToolRegistry()
        tool = NavigateTool()
        registry.register(tool)

        registry.unregister("navigate")

        assert not registry.has("navigate")

    def test_list_names(self):
        """List names returns all tool names."""
        registry = ToolRegistry()
        registry.register(NavigateTool())
        registry.register(RecordEmotionTool())

        names = registry.list_names()

        assert "navigate" in names
        assert "record_emotion" in names

    def test_get_definitions(self):
        """Get definitions returns all tool definitions."""
        registry = ToolRegistry()
        registry.register(NavigateTool())
        registry.register(RecordEmotionTool())

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        assert all(d["type"] == "function" for d in definitions)

    def test_default_registry(self):
        """Default registry has all companion tools."""
        registry = get_companion_tool_registry()

        expected_tools = [
            "get_activities",
            "navigate",
            "record_emotion",
            "handoff_to_tutor",
            "get_student_context",
            "get_parent_notes",
            "get_review_schedule",
        ]

        for tool_name in expected_tools:
            assert registry.has(tool_name), f"Missing tool: {tool_name}"

        assert len(registry) == 7


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_to_llm_message(self):
        """Success result formats as JSON."""
        result = ToolResult(
            success=True,
            data={"action": "test", "value": 42}
        )

        message = result.to_llm_message()

        assert "action" in message
        assert "test" in message
        assert "42" in message

    def test_error_to_llm_message(self):
        """Error result formats with error prefix."""
        result = ToolResult(
            success=False,
            error="Something went wrong"
        )

        message = result.to_llm_message()

        assert message.startswith("Error:")
        assert "Something went wrong" in message
