# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""FastAPI Application Factory.

This module provides the main application factory for EduSynapseOS API.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.dependencies import init_db, close_db, get_tenant_db_manager
from src.api.middleware.api_key_auth import APIKeyAuthMiddleware
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import limiter
from src.api.middleware.tenant import TenantMiddleware
from src.api.routes import health, metrics
from src.api.v1 import router as v1_router
from src.core.config import get_settings
from src.infrastructure.background import (
    setup_dramatiq,
    shutdown_dramatiq,
    start_scheduler,
    stop_scheduler,
)
from src.infrastructure.cache import init_redis, close_redis
from src.infrastructure.database.connection import get_central_session
from src.infrastructure.events import start_event_bridge, stop_event_bridge
from src.infrastructure.telemetry import setup_telemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events for the application.
    Initializes and cleans up:
    - Database connections
    - Redis cache
    - Dramatiq broker
    - Event-to-Dramatiq bridge
    - APScheduler for periodic tasks

    Args:
        app: The FastAPI application instance.

    Yields:
        None during application runtime.
    """
    settings = get_settings()
    logger.info(
        "Starting EduSynapseOS API",
        extra={"environment": settings.environment, "debug": settings.debug},
    )

    # =========================================================================
    # Startup
    # =========================================================================

    # Setup OpenTelemetry (optional, depends on OTEL_EXPORTER_OTLP_ENDPOINT)
    try:
        setup_telemetry(service_name="edusynapse-api")
    except Exception as e:
        logger.warning("Failed to setup telemetry: %s", str(e))

    # Initialize databases
    try:
        print("[STARTUP] Initializing databases...", flush=True)
        await init_db()
        print("[STARTUP] Database connections initialized", flush=True)
        logger.info("Database connections initialized")
    except Exception as e:
        print(f"[STARTUP] Failed to initialize database connections: {e}", flush=True)
        import traceback
        traceback.print_exc()
        logger.warning("Failed to initialize database connections: %s", str(e))

    # Seed initial system admin if not exists
    try:
        from sqlalchemy import select
        from src.infrastructure.database.models.central import SystemUser
        from src.infrastructure.database.seeds.central import seed_central_database

        async with get_central_session() as session:
            result = await session.execute(
                select(SystemUser).where(SystemUser.role == "super_admin").limit(1)
            )
            admin_exists = result.scalar_one_or_none() is not None

            if not admin_exists:
                print("[STARTUP] No super admin found, seeding initial data...", flush=True)
                await seed_central_database(session)
                print("[STARTUP] Initial data seeded successfully", flush=True)
                logger.info("Initial system admin and licenses created")
            else:
                logger.info("System admin already exists, skipping seed")
    except Exception as e:
        print(f"[STARTUP] Failed to seed initial data: {e}", flush=True)
        logger.warning("Failed to seed initial data: %s", str(e))

    # Initialize Redis
    try:
        await init_redis(settings)
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.warning("Failed to initialize Redis: %s", str(e))

    # Setup Dramatiq broker
    try:
        setup_dramatiq()
        logger.info("Dramatiq broker initialized")
    except Exception as e:
        logger.warning("Failed to setup Dramatiq: %s", str(e))

    # Start Event-to-Dramatiq bridge
    try:
        await start_event_bridge()
        logger.info("Event bridge started")
    except Exception as e:
        logger.warning("Failed to start event bridge: %s", str(e))

    # Start scheduler for periodic tasks
    try:
        await start_scheduler()
        logger.info("Scheduler started")
    except Exception as e:
        logger.warning("Failed to start scheduler: %s", str(e))

    # Register companion system alert handler with ProactiveService
    try:
        from src.api.dependencies import _tenant_db_manager
        from src.core.intelligence.embeddings import EmbeddingService
        from src.core.memory.manager import MemoryManager
        from src.core.proactive import get_proactive_service
        from src.domains.companion.proactive_handler import companion_system_alert_handler
        from src.infrastructure.vectors import get_qdrant

        if _tenant_db_manager:
            # Get Qdrant client
            qdrant_client = get_qdrant()

            if qdrant_client:
                # Create embedding service
                embedding_service = EmbeddingService()

                # Create memory manager for ProactiveService
                memory_manager = MemoryManager(
                    tenant_db_manager=_tenant_db_manager,
                    embedding_service=embedding_service,
                    qdrant_client=qdrant_client,
                )

                # Get or create ProactiveService and register handler
                proactive_service = get_proactive_service(
                    memory_manager=memory_manager,
                    tenant_db_manager=_tenant_db_manager,
                )
                proactive_service.set_system_alert_handler(companion_system_alert_handler)
                logger.info("Companion system alert handler registered with ProactiveService")
            else:
                logger.warning("Qdrant not available, skipping companion alert handler registration")
    except Exception as e:
        logger.warning("Failed to register companion alert handler: %s", str(e))

    yield

    # =========================================================================
    # Shutdown
    # =========================================================================

    # Stop scheduler first (it may trigger events)
    try:
        await stop_scheduler()
        logger.info("Scheduler stopped")
    except Exception as e:
        logger.warning("Error stopping scheduler: %s", str(e))

    # Stop Event bridge
    try:
        await stop_event_bridge()
        logger.info("Event bridge stopped")
    except Exception as e:
        logger.warning("Error stopping event bridge: %s", str(e))

    # Shutdown Dramatiq
    try:
        shutdown_dramatiq()
        logger.info("Dramatiq broker shutdown")
    except Exception as e:
        logger.warning("Error shutting down Dramatiq: %s", str(e))

    # Close Redis
    try:
        await close_redis()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning("Error closing Redis: %s", str(e))

    # Close databases
    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning("Error closing database connections: %s", str(e))

    logger.info("Shutting down EduSynapseOS API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This factory function creates a new FastAPI instance with all
    middleware, routes, and configurations applied.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="EduSynapseOS API",
        description="AI-native educational platform backend",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
        # Disable automatic redirects from /path to /path/
        # This prevents 307 redirects that lose Authorization headers
        redirect_slashes=False,
    )

    # =========================================================================
    # State
    # =========================================================================
    app.state.limiter = limiter

    # =========================================================================
    # Exception handlers
    # =========================================================================
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # =========================================================================
    # Middleware (order matters - last added is first executed)
    # =========================================================================

    # CORS middleware (should be last to execute first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins_list,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    # API Key Auth middleware - validates API key/secret for LMS integration
    app.add_middleware(
        APIKeyAuthMiddleware,
        get_central_db=get_central_session,
    )

    # Auth middleware - validates JWT tokens
    app.add_middleware(AuthMiddleware)

    # Tenant middleware - resolves tenant from request
    app.add_middleware(
        TenantMiddleware,
        get_central_db=get_central_session,
        base_domain=settings.cors.origins_list[0].split("//")[-1].split(":")[0]
        if settings.cors.origins_list
        else "edusynapse.com",
    )

    # =========================================================================
    # Routes
    # =========================================================================
    app.include_router(health.router, tags=["Health"])
    app.include_router(metrics.router, tags=["Metrics"])
    app.include_router(v1_router)

    return app
