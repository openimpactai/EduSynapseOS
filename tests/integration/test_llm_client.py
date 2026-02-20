# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for LLMClient.

Tests the LLM client functionality including:
- Completions with various providers
- Streaming responses
- Token counting
- Error handling

Note: These tests require LLM services to be running:
- Ollama at localhost:11434 for local tests
- Or valid API keys for cloud providers
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.intelligence.llm.client import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
)


@pytest.mark.integration
class TestLLMClientInit:
    """Test cases for LLMClient initialization."""

    def test_default_initialization(self) -> None:
        """Test that default initialization uses settings values."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None

            client = LLMClient()

            assert client.model == "ollama/qwen2.5:7b"
            assert client.timeout == 60.0
            assert client.max_retries == 3

    def test_custom_model_initialization(self) -> None:
        """Test initialization with custom model."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None

            client = LLMClient(model="gpt-4o")

            assert client.model == "gpt-4o"

    def test_custom_timeout_initialization(self) -> None:
        """Test initialization with custom timeout."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None

            client = LLMClient(timeout=120.0)

            assert client.timeout == 120.0


@pytest.mark.integration
class TestLLMResponse:
    """Test cases for LLMResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test creating an LLMResponse."""
        response = LLMResponse(
            content="Hello world",
            model="gpt-4o",
            tokens_input=10,
            tokens_output=5,
            finish_reason="stop",
        )

        assert response.content == "Hello world"
        assert response.model == "gpt-4o"
        assert response.tokens_input == 10
        assert response.tokens_output == 5
        assert response.finish_reason == "stop"

    def test_total_tokens_property(self) -> None:
        """Test that total_tokens calculates correctly."""
        response = LLMResponse(
            content="Hello",
            model="gpt-4o",
            tokens_input=10,
            tokens_output=5,
        )

        assert response.total_tokens == 15


@pytest.mark.integration
class TestMessage:
    """Test cases for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test creating a Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self) -> None:
        """Test converting message to dictionary."""
        msg = Message(role="assistant", content="Hi there")
        result = msg.to_dict()

        assert result == {"role": "assistant", "content": "Hi there"}


@pytest.mark.integration
class TestComplete:
    """Test cases for complete method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a mocked LLMClient."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None
            return LLMClient()

    @pytest.mark.asyncio
    async def test_complete_success(self, mock_client: LLMClient) -> None:
        """Test successful completion."""
        with patch(
            "src.core.intelligence.llm.client.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_choice = MagicMock()
            mock_choice.message.content = "4"
            mock_choice.finish_reason = "stop"

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_acompletion.return_value = mock_response

            result = await mock_client.complete("What is 2+2?")

            assert result.content == "4"
            assert result.tokens_input == 10
            assert result.tokens_output == 5
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, mock_client: LLMClient) -> None:
        """Test completion with system prompt."""
        with patch(
            "src.core.intelligence.llm.client.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Math result"
            mock_choice.finish_reason = "stop"

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 20
            mock_usage.completion_tokens = 5

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_acompletion.return_value = mock_response

            result = await mock_client.complete(
                prompt="What is 2+2?",
                system_prompt="You are a math tutor.",
            )

            # Verify system prompt was included in messages
            call_args = mock_acompletion.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a math tutor."

    @pytest.mark.asyncio
    async def test_complete_empty_prompt_raises_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await mock_client.complete("")

    @pytest.mark.asyncio
    async def test_complete_api_error_raises_llm_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that API errors are wrapped in LLMError."""
        with patch(
            "src.core.intelligence.llm.client.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API error")

            with pytest.raises(LLMError) as exc_info:
                await mock_client.complete("What is 2+2?")

            assert "Completion failed" in str(exc_info.value)
            assert exc_info.value.model == "ollama/qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, mock_client: LLMClient) -> None:
        """Test completion with model override."""
        with patch(
            "src.core.intelligence.llm.client.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Result"
            mock_choice.finish_reason = "stop"

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_acompletion.return_value = mock_response

            await mock_client.complete(
                prompt="Hello",
                model="gpt-4o",
            )

            call_args = mock_acompletion.call_args
            assert call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.integration
class TestStream:
    """Test cases for stream method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a mocked LLMClient."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None
            return LLMClient()

    @pytest.mark.asyncio
    async def test_stream_empty_prompt_raises_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            async for _ in mock_client.stream(""):
                pass


@pytest.mark.integration
class TestCountTokens:
    """Test cases for token counting methods."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a mocked LLMClient."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None
            return LLMClient()

    def test_count_tokens_success(self, mock_client: LLMClient) -> None:
        """Test successful token counting."""
        with patch(
            "src.core.intelligence.llm.client.token_counter",
        ) as mock_counter:
            mock_counter.return_value = 5

            result = mock_client.count_tokens("Hello world")

            assert result == 5
            mock_counter.assert_called_once()

    def test_count_tokens_fallback(self, mock_client: LLMClient) -> None:
        """Test token counting fallback on error."""
        with patch(
            "src.core.intelligence.llm.client.token_counter",
        ) as mock_counter:
            mock_counter.side_effect = Exception("Counter error")

            # Should use fallback calculation (len // 4)
            result = mock_client.count_tokens("Hello world!!!")

            # "Hello world!!!" is 14 chars, 14 // 4 = 3
            assert result == 3

    def test_count_message_tokens(self, mock_client: LLMClient) -> None:
        """Test counting tokens for messages."""
        with patch(
            "src.core.intelligence.llm.client.token_counter",
        ) as mock_counter:
            mock_counter.return_value = 15

            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ]

            result = mock_client.count_message_tokens(messages)

            assert result == 15


@pytest.mark.integration
class TestCompleteWithHistory:
    """Test cases for complete_with_history method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a mocked LLMClient."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None
            return LLMClient()

    @pytest.mark.asyncio
    async def test_complete_with_history(self, mock_client: LLMClient) -> None:
        """Test completion with conversation history."""
        with patch(
            "src.core.intelligence.llm.client.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_choice = MagicMock()
            mock_choice.message.content = "1991"
            mock_choice.finish_reason = "stop"

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 30
            mock_usage.completion_tokens = 5

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage

            mock_acompletion.return_value = mock_response

            history = [
                ("What is Python?", "Python is a programming language."),
                ("Who created it?", "Guido van Rossum created Python."),
            ]

            result = await mock_client.complete_with_history(
                prompt="When was it created?",
                history=history,
            )

            assert result.content == "1991"

            # Verify history was included
            call_args = mock_acompletion.call_args
            messages = call_args.kwargs["messages"]

            # Should have 5 messages: 2 pairs of history + 1 current
            assert len(messages) == 5
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[-1]["content"] == "When was it created?"


@pytest.mark.integration
class TestRepr:
    """Test cases for __repr__ method."""

    def test_repr_format(self) -> None:
        """Test that repr returns expected format."""
        with patch("src.core.intelligence.llm.client.get_settings") as mock_settings:
            mock_settings.return_value.llm.get_default_model.return_value = "ollama/qwen2.5:7b"
            mock_settings.return_value.llm.request_timeout = 60.0
            mock_settings.return_value.llm.max_retries = 3
            mock_settings.return_value.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.return_value.llm.openai_api_key = None
            mock_settings.return_value.llm.anthropic_api_key = None
            mock_settings.return_value.llm.google_api_key = None

            client = LLMClient()
            repr_str = repr(client)

            assert "LLMClient" in repr_str
            assert "ollama/qwen2.5:7b" in repr_str
            assert "60" in repr_str
            assert "3" in repr_str
