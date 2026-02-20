# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LLM client using LiteLLM for multi-provider support.

This module provides a unified LLM interface through LiteLLM,
supporting completions and streaming for 100+ providers.

Provider configuration is loaded from config/llm/providers.yaml.
API keys and endpoints are passed directly to LiteLLM's acompletion()
function rather than using environment variables, which is required
for LiteLLM 1.80+ to work correctly with Ollama and custom endpoints.

NOTE: For Ollama tool calling, we bypass LiteLLM and call Ollama API directly
due to a known LiteLLM bug where tool_calls are not properly parsed.

Supported providers:
- Ollama: Local or remote LLM inference (including Vast.ai)
- OpenAI: GPT-4, GPT-3.5, etc.
- Anthropic: Claude 3, Claude 3.5
- Google: Gemini models
- And many more through LiteLLM

Example:
    >>> from src.core.intelligence.llm import LLMClient
    >>> client = LLMClient()
    >>> response = await client.complete("Explain quantum computing")
    >>> print(response.content)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import aiohttp
import litellm
from litellm import acompletion, token_counter

from src.core.config.llm_providers import get_llm_params, get_provider_manager
from src.core.config.settings import LLMSettings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM completion.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        tokens_input: Number of input tokens used.
        tokens_output: Number of output tokens generated.
        finish_reason: Why generation stopped (stop, length, etc.).
        raw_response: Original response object from LiteLLM.
    """

    content: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[object] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used (input + output).

        Returns:
            Total token count.
        """
        return self.tokens_input + self.tokens_output


@dataclass
class ToolCall:
    """A tool call requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMToolResponse:
    """Response from an LLM completion with tool calling support.

    Attributes:
        content: The generated text content (may be empty if tool_calls present).
        model: The model that generated the response.
        tokens_input: Number of input tokens used.
        tokens_output: Number of output tokens generated.
        finish_reason: Why generation stopped (stop, tool_calls, length, etc.).
        tool_calls: List of tool calls requested by the LLM.
        raw_response: Original response object from LiteLLM.
    """

    content: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    finish_reason: str = "stop"
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Optional[object] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used (input + output)."""
        return self.tokens_input + self.tokens_output

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMError(Exception):
    """Exception raised when LLM operation fails.

    Attributes:
        message: Error description.
        model: Model that caused the error.
        error_code: Error code if available.
        original_error: Original exception if any.
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize LLMError.

        Args:
            message: Error description.
            model: Model that caused the error.
            error_code: Error code if available.
            original_error: Original exception if any.
        """
        self.message = message
        self.model = model
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message text content.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for LiteLLM.

        Returns:
            Dictionary with role and content.
        """
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Client for LLM operations via LiteLLM.

    Provides a unified interface for text generation across multiple
    LLM providers through LiteLLM's abstraction layer.

    Features:
    - Multi-provider support (Ollama, OpenAI, Anthropic, Google, etc.)
    - Async completions and streaming
    - Token counting
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Direct Ollama API for tool calling (bypasses LiteLLM bug)

    Attributes:
        model: Default model to use for completions.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Example:
        >>> client = LLMClient()
        >>> response = await client.complete(
        ...     prompt="What is machine learning?",
        ...     temperature=0.7,
        ... )
        >>> print(response.content)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        llm_settings: Optional[LLMSettings] = None,
    ):
        """Initialize the LLM client.

        Args:
            model: Default model in LiteLLM format. Falls back to provider config.
            timeout: Request timeout in seconds. Falls back to settings.
            max_retries: Maximum retry attempts. Falls back to settings.
            llm_settings: LLM configuration. Uses get_settings() if None.
        """
        settings = get_settings()
        self._settings = llm_settings or settings.llm
        self._provider_manager = get_provider_manager()

        # Use model from provider config if not specified
        self._model = model or self._provider_manager.get_default_model()
        self._timeout = timeout or self._settings.request_timeout
        self._max_retries = max_retries or self._settings.max_retries

        # Configure LiteLLM global settings
        self._configure_litellm()

        logger.info(
            "LLMClient initialized with model=%s, timeout=%.1fs, max_retries=%d",
            self._model,
            self._timeout,
            self._max_retries,
        )

    def _configure_litellm(self) -> None:
        """Configure LiteLLM global settings.

        Note: API keys and base URLs are NOT set via environment variables.
        Instead, they are passed directly to acompletion() via get_llm_params().
        This is required for LiteLLM 1.80+ to work correctly with Ollama
        and custom endpoints.
        """
        # LiteLLM global settings
        litellm.set_verbose = False
        litellm.drop_params = True

    def _get_provider_params(self, model: str) -> dict[str, Any]:
        """Get provider-specific parameters for a model.

        Returns api_base and api_key that should be passed directly
        to LiteLLM's acompletion() function.

        Args:
            model: Model string in LiteLLM format.

        Returns:
            Dictionary with api_base and/or api_key if configured (no model key).
        """
        params = get_llm_params(model)
        # Remove 'model' key as it's passed separately to acompletion()
        params.pop("model", None)
        return params

    def _is_ollama_model(self, model: str) -> bool:
        """Check if model is an Ollama model.

        Args:
            model: Model string in LiteLLM format.

        Returns:
            True if this is an Ollama model.
        """
        return model.startswith("ollama/") or model.startswith("ollama_chat/")

    def _get_ollama_model_name(self, model: str) -> str:
        """Extract raw model name from LiteLLM model string.

        Args:
            model: Model string like 'ollama_chat/command-r:35b'

        Returns:
            Raw model name like 'command-r:35b'
        """
        if model.startswith("ollama_chat/"):
            return model[len("ollama_chat/"):]
        if model.startswith("ollama/"):
            return model[len("ollama/"):]
        return model

    async def _ollama_complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        api_base: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        context_length: int = 4096,
    ) -> LLMToolResponse:
        """Call Ollama API directly for tool calling.

        LiteLLM has a known bug where it doesn't properly parse tool_calls
        from Ollama responses. This method bypasses LiteLLM and calls
        Ollama's /api/chat endpoint directly.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in OpenAI format.
            model: Model name (without ollama/ prefix).
            api_base: Ollama API base URL.
            api_key: Optional API key for authentication.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (num_predict).
            context_length: Context window size (num_ctx).

        Returns:
            LLMToolResponse with content and/or tool_calls.
        """
        # Extract raw model name
        raw_model = self._get_ollama_model_name(model)

        # Prepare messages for Ollama compatibility
        # Ollama doesn't support:
        # 1. tool_calls field in assistant messages
        # 2. role: "tool" messages
        # We need to convert these to a format Ollama understands
        ollama_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                # Collect tool results to format as user context
                tool_name = msg.get("name", "unknown")
                content = msg.get("content", "")
                pending_tool_results.append({"name": tool_name, "content": content})

            elif role == "assistant":
                # First, flush any pending tool results as user context
                if pending_tool_results:
                    results_parts = [
                        f"• {tr['name']}: {tr['content']}"
                        for tr in pending_tool_results
                    ]
                    results_context = (
                        "[Tool Execution Results]\n"
                        "The following tools were executed. "
                        "Use this information to respond:\n\n"
                        + "\n".join(results_parts)
                    )
                    ollama_messages.append({"role": "user", "content": results_context})
                    pending_tool_results = []

                # Strip tool_calls from assistant messages (Ollama doesn't support it)
                content = msg.get("content", "")
                if content:  # Only add if there's content
                    ollama_messages.append({"role": "assistant", "content": content})

            else:
                # system, user messages - pass through as-is
                ollama_messages.append({"role": role, "content": msg.get("content", "")})

        # Flush any remaining tool results
        if pending_tool_results:
            results_parts = [
                f"• {tr['name']}: {tr['content']}"
                for tr in pending_tool_results
            ]
            results_context = (
                "[Tool Execution Results]\n"
                "The following tools were executed. "
                "Use this information to respond:\n\n"
                + "\n".join(results_parts)
            )
            ollama_messages.append({"role": "user", "content": results_context})

        # Build request payload for Ollama /api/chat
        payload = {
            "model": raw_model,
            "messages": ollama_messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_length,
            },
        }

        # Build headers
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{api_base.rstrip('/')}/api/chat"

        logger.debug(
            "[OLLAMA_DIRECT] Calling %s with model=%s, "
            "original_messages=%d, ollama_messages=%d, tools=%d",
            url,
            raw_model,
            len(messages),
            len(ollama_messages),
            len(tools),
        )

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise LLMError(
                            message=f"Ollama API error: {resp.status} - {error_text}",
                            model=model,
                        )

                    data = await resp.json()

            # Parse response
            message_data = data.get("message", {})
            content = message_data.get("content", "") or ""
            tool_calls_raw = message_data.get("tool_calls", []) or []

            logger.debug(
                "[OLLAMA_DIRECT] Response: content_len=%d, tool_calls=%d",
                len(content),
                len(tool_calls_raw),
            )

            # Parse tool calls
            parsed_tool_calls: list[ToolCall] = []
            for i, tc in enumerate(tool_calls_raw):
                func_data = tc.get("function", {})
                tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                tc_name = func_data.get("name", "")
                tc_args = func_data.get("arguments", {})

                # Arguments can be string or dict
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except json.JSONDecodeError:
                        tc_args = {}

                if tc_name:  # Only add if we have a name
                    parsed_tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=tc_name,
                            arguments=tc_args,
                        )
                    )

            if parsed_tool_calls:
                logger.info(
                    "[OLLAMA_DIRECT] Parsed tool calls: %s",
                    [tc.name for tc in parsed_tool_calls],
                )

            # Get token counts from response
            tokens_input = data.get("prompt_eval_count", 0) or 0
            tokens_output = data.get("eval_count", 0) or 0
            finish_reason = "tool_calls" if parsed_tool_calls else "stop"

            return LLMToolResponse(
                content=content,
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                tool_calls=parsed_tool_calls,
                raw_response=data,
            )

        except aiohttp.ClientError as e:
            logger.error("[OLLAMA_DIRECT] Request failed: %s", str(e))
            raise LLMError(
                message=f"Ollama request failed: {str(e)}",
                model=model,
                original_error=e,
            ) from e

    @property
    def model(self) -> str:
        """Get the default model.

        Returns:
            Model identifier in LiteLLM format.
        """
        return self._model

    @property
    def timeout(self) -> float:
        """Get the request timeout.

        Returns:
            Timeout in seconds.
        """
        return self._timeout

    @property
    def max_retries(self) -> int:
        """Get the maximum retry count.

        Returns:
            Maximum number of retries.
        """
        return self._max_retries

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[list[Message]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: User prompt text.
            model: Override default model for this request.
            system_prompt: Optional system prompt to set context.
            messages: Previous conversation messages (if multi-turn).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences to end generation.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            LLMResponse with generated content and metadata.

        Raises:
            LLMError: If generation fails after retries.
            ValueError: If prompt is empty.

        Example:
            >>> response = await client.complete(
            ...     prompt="Write a haiku about coding",
            ...     temperature=0.9,
            ... )
            >>> print(response.content)
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        use_model = model or self._model

        # Build messages list
        chat_messages: list[dict[str, str]] = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        if messages:
            chat_messages.extend([m.to_dict() for m in messages])

        chat_messages.append({"role": "user", "content": prompt})

        # Get provider-specific params (api_base, api_key)
        provider_params = self._get_provider_params(use_model)

        try:
            response = await acompletion(
                model=use_model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                timeout=self._timeout,
                num_retries=self._max_retries,
                **provider_params,  # Pass api_base and api_key directly
                **kwargs,
            )

            # Extract content from response
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Get token usage if available
            tokens_input = getattr(response.usage, "prompt_tokens", 0) or 0
            tokens_output = getattr(response.usage, "completion_tokens", 0) or 0

            result = LLMResponse(
                content=content,
                model=use_model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                raw_response=response,
            )

            logger.debug(
                "Completion generated: model=%s, tokens_in=%d, tokens_out=%d",
                use_model,
                tokens_input,
                tokens_output,
            )

            return result

        except Exception as e:
            logger.error(
                "Completion failed: model=%s, prompt_length=%d, error=%s",
                use_model,
                len(prompt),
                str(e),
            )
            raise LLMError(
                message=f"Completion failed: {str(e)}",
                model=use_model,
                original_error=e,
            ) from e

    async def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[list[Message]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion token by token.

        Yields tokens as they are generated for real-time display.
        Useful for chat interfaces and long-form generation.

        Args:
            prompt: User prompt text.
            model: Override default model for this request.
            system_prompt: Optional system prompt to set context.
            messages: Previous conversation messages (if multi-turn).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences to end generation.
            **kwargs: Additional LiteLLM parameters.

        Yields:
            Generated tokens as strings.

        Raises:
            LLMError: If streaming fails.
            ValueError: If prompt is empty.

        Example:
            >>> async for token in client.stream("Tell me a story"):
            ...     print(token, end="", flush=True)
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        use_model = model or self._model

        # Build messages list
        chat_messages: list[dict[str, str]] = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        if messages:
            chat_messages.extend([m.to_dict() for m in messages])

        chat_messages.append({"role": "user", "content": prompt})

        # Get provider-specific params (api_base, api_key)
        provider_params = self._get_provider_params(use_model)

        try:
            response = await acompletion(
                model=use_model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                timeout=self._timeout,
                stream=True,
                **provider_params,  # Pass api_base and api_key directly
                **kwargs,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(
                "Streaming failed: model=%s, error=%s",
                use_model,
                str(e),
            )
            raise LLMError(
                message=f"Streaming failed: {str(e)}",
                model=use_model,
                original_error=e,
            ) from e

    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """Count the number of tokens in a text.

        Uses LiteLLM's token counter which handles different
        tokenizers based on the model.

        Args:
            text: Text to count tokens for.
            model: Model to use for tokenization.

        Returns:
            Number of tokens.

        Example:
            >>> count = client.count_tokens("Hello, world!")
            >>> print(f"Token count: {count}")
            Token count: 4
        """
        use_model = model or self._model

        try:
            count = token_counter(model=use_model, text=text)
            return count
        except Exception:
            # Fallback to simple word-based estimation
            # Roughly 4 characters per token on average
            return len(text) // 4

    def count_message_tokens(
        self,
        messages: list[Message],
        model: Optional[str] = None,
    ) -> int:
        """Count tokens for a list of messages.

        Accounts for message formatting overhead used by chat models.

        Args:
            messages: List of messages to count.
            model: Model to use for tokenization.

        Returns:
            Total token count including formatting.
        """
        use_model = model or self._model

        # Convert messages to dict format
        message_dicts = [m.to_dict() for m in messages]

        try:
            count = token_counter(model=use_model, messages=message_dicts)
            return count
        except Exception:
            # Fallback: sum up content tokens with overhead
            total = 0
            for msg in messages:
                total += self.count_tokens(msg.content, model) + 4  # overhead per message
            return total

    async def complete_with_history(
        self,
        prompt: str,
        history: list[tuple[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion with conversation history.

        Convenience method that converts history tuples to messages.

        Args:
            prompt: Current user prompt.
            history: List of (user, assistant) message tuples.
            model: Override default model.
            system_prompt: Optional system context.
            **kwargs: Additional completion parameters.

        Returns:
            LLMResponse with generated content.

        Example:
            >>> history = [
            ...     ("What is Python?", "Python is a programming language."),
            ...     ("Who created it?", "Guido van Rossum created Python."),
            ... ]
            >>> response = await client.complete_with_history(
            ...     "When was it created?",
            ...     history,
            ... )
        """
        messages: list[Message] = []

        for user_msg, assistant_msg in history:
            messages.append(Message(role="user", content=user_msg))
            messages.append(Message(role="assistant", content=assistant_msg))

        return await self.complete(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            **kwargs,
        )

    async def complete_with_messages(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.

        This is a convenience method for conversation-style completions
        where the entire context is provided as messages. Unlike complete(),
        this method doesn't require separating system prompt and user prompt.

        The messages should be in OpenAI format with 'role' and 'content' keys.
        The first message can be a system message (role='system').

        Args:
            messages: Conversation messages in OpenAI format.
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            model: Override default model for this request.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            LLMResponse with generated content and metadata.

        Raises:
            LLMError: If completion fails after retries.
            ValueError: If messages list is empty.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful tutor."},
            ...     {"role": "user", "content": "I don't understand fractions."},
            ...     {"role": "assistant", "content": "Let me help explain..."},
            ...     {"role": "user", "content": "Can you give an example?"},
            ... ]
            >>> response = await client.complete_with_messages(messages)
            >>> print(response.content)
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        use_model = model or self._model

        # Get provider-specific params (api_base, api_key)
        provider_params = self._get_provider_params(use_model)

        try:
            response = await acompletion(
                model=use_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._timeout,
                num_retries=self._max_retries,
                **provider_params,
                **kwargs,
            )

            # Extract content from response
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Get token usage if available
            tokens_input = getattr(response.usage, "prompt_tokens", 0) or 0
            tokens_output = getattr(response.usage, "completion_tokens", 0) or 0

            return LLMResponse(
                content=content,
                model=use_model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except Exception as e:
            logger.error(
                "LLM completion with messages failed: model=%s, error=%s",
                use_model,
                str(e),
            )
            raise LLMError(
                message=f"Completion failed: {str(e)}",
                model=use_model,
                original_error=e,
            ) from e

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMToolResponse:
        """Generate a completion with tool calling support.

        This method enables the LLM to request tool executions. The response
        may contain tool_calls that should be executed, with results sent
        back in a follow-up call.

        For Ollama models, this method bypasses LiteLLM and calls Ollama API
        directly to work around a known bug in LiteLLM's tool call parsing.

        Args:
            messages: Conversation messages in OpenAI format.
                Each message should have 'role' and 'content' keys.
                Tool results use role='tool' with 'tool_call_id' and 'name'.
            tools: Tool definitions in OpenAI format.
                Each tool has 'type': 'function' and 'function' with
                'name', 'description', and 'parameters' (JSON schema).
            tool_choice: How the model should use tools.
                - "auto": Model decides whether to use tools (default)
                - "none": Model will not use any tools
                - "required": Model must use at least one tool
                - {"type": "function", "function": {"name": "..."}}:
                  Force specific tool
            model: Override default model for this request.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            LLMToolResponse with content and/or tool_calls.

        Raises:
            LLMError: If completion fails after retries.
            ValueError: If messages list is empty.

        Example:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
            >>> response = await client.complete_with_tools(messages, tools)
            >>> if response.has_tool_calls:
            ...     for call in response.tool_calls:
            ...         print(f"Execute: {call.name}({call.arguments})")
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        use_model = model or self._model

        # Get provider-specific params (api_base, api_key)
        provider_params = self._get_provider_params(use_model)

        # For Ollama models, bypass LiteLLM and call Ollama API directly
        # LiteLLM has bugs:
        # 1. Doesn't parse tool_calls from Ollama responses
        # 2. Expects tool_call arguments as string but messages may have dict
        # We always use bypass for Ollama, even with empty tools, to ensure
        # consistent message handling
        if self._is_ollama_model(use_model):
            api_base = provider_params.get("api_base")
            api_key = provider_params.get("api_key")

            if not api_base:
                raise LLMError(
                    message="api_base is required for Ollama tool calling",
                    model=use_model,
                )

            # Get context_length from provider config (defaults to 4096)
            context_length = provider_params.get("context_length", 4096)

            logger.debug(
                "Using direct Ollama API for tool calling: model=%s, tools=%d, "
                "context_length=%d",
                use_model,
                len(tools),
                context_length,
            )

            return await self._ollama_complete_with_tools(
                messages=messages,
                tools=tools,
                model=use_model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                context_length=context_length,
            )

        # For non-Ollama models, use LiteLLM
        try:
            # DEBUG: Log what we're sending to LLM
            print(f"\n{'='*60}")
            print(f"[LLM CLIENT] complete_with_tools called")
            print(f"  Model: {use_model}")
            print(f"  Messages count: {len(messages)}")
            print(f"  Tools count: {len(tools)}")
            print(f"  Tool choice: {tool_choice}")
            print(f"  Tool names: {[t.get('function', {}).get('name') for t in tools]}")
            print(f"{'='*60}\n")

            response = await acompletion(
                model=use_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._timeout,
                num_retries=self._max_retries,
                **provider_params,
                **kwargs,
            )

            # Extract content from response
            if not response.choices:
                # Log the full response for debugging
                logger.error(
                    "LLM returned empty choices: model=%s, response=%s",
                    use_model,
                    str(response)[:1000],
                )
                raise LLMError(
                    message=f"LLM returned empty response with no choices",
                    model=use_model,
                )
            message = response.choices[0].message
            content = message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # DEBUG: Log response details
            print(f"\n{'='*60}")
            print(f"[LLM CLIENT] Response received")
            print(f"  Content length: {len(content)}")
            print(f"  Finish reason: {finish_reason}")
            print(f"  Has tool_calls attr: {hasattr(message, 'tool_calls')}")
            print(f"  Tool calls value: {getattr(message, 'tool_calls', None)}")
            print(f"{'='*60}\n")

            # Parse tool calls if present
            parsed_tool_calls: list[ToolCall] = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    # Parse arguments - may be dict or JSON string
                    if isinstance(tc.function.arguments, dict):
                        args = tc.function.arguments
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {}

                    parsed_tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        )
                    )

            # Get token usage if available
            tokens_input = getattr(response.usage, "prompt_tokens", 0) or 0
            tokens_output = getattr(response.usage, "completion_tokens", 0) or 0

            result = LLMToolResponse(
                content=content,
                model=use_model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                tool_calls=parsed_tool_calls,
                raw_response=response,
            )

            logger.debug(
                "Tool completion generated: model=%s, tokens_in=%d, tokens_out=%d, tool_calls=%d",
                use_model,
                tokens_input,
                tokens_output,
                len(parsed_tool_calls),
            )

            return result

        except Exception as e:
            logger.error(
                "Tool completion failed: model=%s, error=%s",
                use_model,
                str(e),
            )
            raise LLMError(
                message=f"Tool completion failed: {str(e)}",
                model=use_model,
                original_error=e,
            ) from e

    def __repr__(self) -> str:
        """Return string representation of the client.

        Returns:
            String showing model, timeout, and retry settings.
        """
        return (
            f"LLMClient(model={self._model!r}, "
            f"timeout={self._timeout}, max_retries={self._max_retries})"
        )
