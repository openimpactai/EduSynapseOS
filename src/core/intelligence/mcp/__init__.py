# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MCP (Model Context Protocol) server module.

This module implements an MCP server that exposes EduSynapseOS capabilities
to external LLM clients via the industry-standard Model Context Protocol.

The MCP server provides:
- Tools: Callable functions for knowledge lookup, question generation,
  answer evaluation, and student context retrieval
- Resources: Data access endpoints for student profiles, curriculum content,
  memory layers, and analytics dashboards
- Prompts: Pre-built prompt templates for common educational interactions

The server is LLM-agnostic:
- External clients (Claude Desktop, OpenAI Agents, Cursor, etc.) can connect
- Internal LLM calls use the existing LiteLLM infrastructure (Vast.AI, OpenAI, Ollama, etc.)

Example:
    # Run the MCP server
    from src.core.intelligence.mcp import create_mcp_server

    server = create_mcp_server(
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        agent_factory=agent_factory,
    )

    # Run with stdio transport (for Claude Desktop)
    server.run(transport="stdio")

    # Or with HTTP transport (for API integration)
    server.run(transport="streamable-http", port=8080)
"""

from src.core.intelligence.mcp.server import (
    EduSynapseMCPServer,
    create_mcp_server,
    get_mcp_server,
)

__all__ = [
    "EduSynapseMCPServer",
    "create_mcp_server",
    "get_mcp_server",
]
