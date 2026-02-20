# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MCP Server entry point and lifecycle management.

This module provides the EduSynapseMCPServer class which wraps FastMCP
and manages the server lifecycle, tool/resource/prompt registration,
and dependency injection.

The server supports multiple transport modes:
- stdio: For Claude Desktop and terminal-based clients
- streamable-http: For API integration and web-based clients

Example:
    # Create server with dependencies
    server = create_mcp_server(
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        agent_factory=agent_factory,
        tenant_db_manager=tenant_db_manager,
    )

    # Run with stdio transport
    server.run()

    # Run with HTTP transport on specific port
    server.run(transport="streamable-http", port=8080)
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from src.core.agents.factory import AgentFactory
from src.core.memory.manager import MemoryManager
from src.core.memory.rag.retriever import RAGRetriever
from src.infrastructure.database.tenant_manager import TenantDatabaseManager

logger = logging.getLogger(__name__)


class EduSynapseMCPServer:
    """MCP Server for EduSynapseOS.

    This class manages the MCP server lifecycle and provides access to
    EduSynapseOS capabilities through the Model Context Protocol.

    The server exposes:
    - Tools: Functions for AI operations (question generation, answer evaluation, etc.)
    - Resources: Data access for student profiles, curriculum, memory, analytics
    - Prompts: Pre-built templates for common educational interactions

    Attributes:
        mcp: The underlying FastMCP server instance.
        memory_manager: Manager for 4-layer memory system.
        rag_retriever: Retriever for RAG operations.
        agent_factory: Factory for creating DynamicAgent instances.
        tenant_db_manager: Manager for tenant database connections.

    Example:
        server = EduSynapseMCPServer(
            memory_manager=memory_manager,
            rag_retriever=rag_retriever,
            agent_factory=agent_factory,
            tenant_db_manager=tenant_db_manager,
        )
        server.run(transport="stdio")
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        rag_retriever: RAGRetriever,
        agent_factory: AgentFactory,
        tenant_db_manager: TenantDatabaseManager,
    ) -> None:
        """Initialize the MCP server.

        Args:
            memory_manager: Manager for 4-layer memory system.
            rag_retriever: Retriever for RAG operations.
            agent_factory: Factory for creating DynamicAgent instances.
            tenant_db_manager: Manager for tenant database connections.
        """
        self._memory_manager = memory_manager
        self._rag_retriever = rag_retriever
        self._agent_factory = agent_factory
        self._tenant_db_manager = tenant_db_manager

        # Create FastMCP server with lifespan
        self._mcp = FastMCP(
            name="edusynapse",
            instructions=(
                "EduSynapseOS MCP Server - An AI-native educational platform. "
                "Use the available tools to search curriculum knowledge, "
                "get student context, generate questions, evaluate answers, "
                "and access learning analytics."
            ),
        )

        # Register all tools, resources, and prompts
        self._register_tools()
        self._register_resources()
        self._register_prompts()

        logger.info("EduSynapseMCPServer initialized")

    @property
    def mcp(self) -> FastMCP:
        """Get the underlying FastMCP instance.

        Returns:
            The FastMCP server instance.
        """
        return self._mcp

    @property
    def memory_manager(self) -> MemoryManager:
        """Get the memory manager.

        Returns:
            The MemoryManager instance.
        """
        return self._memory_manager

    @property
    def rag_retriever(self) -> RAGRetriever:
        """Get the RAG retriever.

        Returns:
            The RAGRetriever instance.
        """
        return self._rag_retriever

    @property
    def agent_factory(self) -> AgentFactory:
        """Get the agent factory.

        Returns:
            The AgentFactory instance.
        """
        return self._agent_factory

    @property
    def tenant_db_manager(self) -> TenantDatabaseManager:
        """Get the tenant database manager.

        Returns:
            The TenantDatabaseManager instance.
        """
        return self._tenant_db_manager

    def _register_tools(self) -> None:
        """Register all MCP tools.

        Tools are callable functions that perform operations like
        knowledge lookup, question generation, and answer evaluation.
        """
        from src.core.intelligence.mcp.tools import register_tools

        register_tools(self)
        logger.debug("MCP tools registered")

    def _register_resources(self) -> None:
        """Register all MCP resources.

        Resources provide data access for student profiles,
        curriculum content, memory layers, and analytics.
        """
        from src.core.intelligence.mcp.resources import register_resources

        register_resources(self)
        logger.debug("MCP resources registered")

    def _register_prompts(self) -> None:
        """Register all MCP prompts.

        Prompts are pre-built templates for common
        educational interactions.
        """
        from src.core.intelligence.mcp.prompts import register_prompts

        register_prompts(self)
        logger.debug("MCP prompts registered")

    def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """Run the MCP server.

        Args:
            transport: Transport type. Options:
                - "stdio": Standard I/O (for Claude Desktop)
                - "streamable-http": HTTP transport (for API integration)
                - "sse": Server-Sent Events transport
            host: Host to bind to (for HTTP transport).
            port: Port to bind to (for HTTP transport).

        Example:
            # Run with stdio (default, for Claude Desktop)
            server.run()

            # Run with HTTP transport
            server.run(transport="streamable-http", port=8080)
        """
        logger.info(
            "Starting MCP server: transport=%s, host=%s, port=%d",
            transport,
            host,
            port,
        )

        if transport == "streamable-http":
            self._mcp.run(transport=transport, host=host, port=port)
        else:
            self._mcp.run(transport=transport)


# Module-level singleton instance
_server: EduSynapseMCPServer | None = None


def create_mcp_server(
    memory_manager: MemoryManager,
    rag_retriever: RAGRetriever,
    agent_factory: AgentFactory,
    tenant_db_manager: TenantDatabaseManager,
) -> EduSynapseMCPServer:
    """Create the MCP server singleton.

    Creates a new EduSynapseMCPServer instance with the provided dependencies.
    If a server already exists, it will be replaced.

    Args:
        memory_manager: Manager for 4-layer memory system.
        rag_retriever: Retriever for RAG operations.
        agent_factory: Factory for creating DynamicAgent instances.
        tenant_db_manager: Manager for tenant database connections.

    Returns:
        The created EduSynapseMCPServer instance.

    Example:
        server = create_mcp_server(
            memory_manager=memory_manager,
            rag_retriever=rag_retriever,
            agent_factory=agent_factory,
            tenant_db_manager=tenant_db_manager,
        )
    """
    global _server

    _server = EduSynapseMCPServer(
        memory_manager=memory_manager,
        rag_retriever=rag_retriever,
        agent_factory=agent_factory,
        tenant_db_manager=tenant_db_manager,
    )

    return _server


def get_mcp_server() -> EduSynapseMCPServer | None:
    """Get the MCP server singleton.

    Returns:
        The EduSynapseMCPServer instance, or None if not created.
    """
    return _server


def reset_mcp_server() -> None:
    """Reset the MCP server singleton.

    This is primarily useful for testing.
    """
    global _server
    _server = None
