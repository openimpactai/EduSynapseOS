# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LLM client and routing module using LiteLLM.

This module provides a unified interface for LLM operations through LiteLLM,
supporting 100+ LLM providers including Ollama, OpenAI, Anthropic, and Google.

Components:
- LLMClient: Main interface for completions and streaming
- ModelRouter: Intelligent model selection and fallback chains

Example:
    >>> from src.core.intelligence.llm import LLMClient, ModelRouter
    >>> client = LLMClient()
    >>> response = await client.complete("What is 2+2?")
    >>> print(response.content)
    2+2 equals 4
"""

from src.core.intelligence.llm.client import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMToolResponse,
    ToolCall,
)
from src.core.intelligence.llm.router import ModelConfig, ModelRouter

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LLMToolResponse",
    "ModelConfig",
    "ModelRouter",
    "ToolCall",
]
