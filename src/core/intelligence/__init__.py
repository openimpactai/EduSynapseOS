# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Intelligence module for AI-powered operations.

This module provides unified interfaces for:
- Embedding generation via LiteLLM (API-based, supporting 100+ providers)
- LLM completions and streaming via LiteLLM
- Model routing and fallback chain management
- MCP (Model Context Protocol) server for external LLM tool integration

The module uses LiteLLM as the unified interface for all AI operations,
enabling support for Ollama, OpenAI, Anthropic, Google, and many other providers.

The MCP server exposes EduSynapseOS capabilities to external LLM clients
(Claude Desktop, OpenAI Agents, Cursor, etc.) via the industry-standard
Model Context Protocol.

Example:
    >>> from src.core.intelligence import EmbeddingService, LLMClient
    >>> embedder = EmbeddingService()
    >>> vector = await embedder.embed_text("Hello world")
    >>> client = LLMClient()
    >>> response = await client.complete("What is 2+2?")

    # MCP is imported separately to avoid circular dependencies
    >>> from src.core.intelligence.mcp import create_mcp_server
    >>> server = create_mcp_server(memory_manager, rag_retriever, agent_factory, tenant_db_manager)
    >>> server.run(transport="stdio")
"""

from src.core.intelligence.embeddings import EmbeddingService
from src.core.intelligence.llm import LLMClient, ModelRouter

# Note: MCP module is NOT imported here to avoid circular dependencies.
# Import directly: from src.core.intelligence.mcp import create_mcp_server

__all__ = [
    # Embeddings
    "EmbeddingService",
    # LLM
    "LLMClient",
    "ModelRouter",
]
