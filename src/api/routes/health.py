# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Health check endpoints.

This module provides health and readiness endpoints for the API.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Track server start time for uptime calculation
_server_start_time = time.time()


class ComponentHealth(BaseModel):
    """Individual component health status."""
    status: str = Field(description="Component status")
    latency_ms: float | None = Field(None, description="Response latency in ms")
    message: str | None = Field(None, description="Additional status message")


class LLMProviderHealth(BaseModel):
    """LLM provider health status."""
    status: str = Field(description="Provider status")
    url: str | None = Field(None, description="Provider URL")


class ComponentsHealth(BaseModel):
    """All components health status."""
    database: ComponentHealth | None = None
    redis: ComponentHealth | None = None
    qdrant: ComponentHealth | None = None
    llm_providers: dict[str, LLMProviderHealth] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(description="Current server timestamp")
    version: str = Field(description="API version")
    environment: str = Field(description="Deployment environment")
    uptime_seconds: int = Field(description="Server uptime in seconds")
    checked_at: datetime = Field(description="When health was checked")
    components: ComponentsHealth = Field(default_factory=ComponentsHealth)


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    ready: bool = Field(description="Whether the service is ready")
    checks: dict[str, Any] = Field(description="Individual check results")


async def check_database() -> ComponentHealth:
    """Check PostgreSQL database connection."""
    try:
        from sqlalchemy import text
        from src.infrastructure.database.connection import get_central_engine

        engine = get_central_engine()
        start = time.time()

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        latency = (time.time() - start) * 1000
        return ComponentHealth(status="healthy", latency_ms=round(latency, 2))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return ComponentHealth(status="unhealthy", message=str(e))


async def check_redis() -> ComponentHealth:
    """Check Redis connection."""
    try:
        import redis.asyncio as aioredis
        from src.core.config import get_settings

        settings = get_settings()
        redis_cfg = settings.redis
        start = time.time()

        client = aioredis.Redis(
            host=redis_cfg.host,
            port=redis_cfg.port,
            password=redis_cfg.password.get_secret_value() if redis_cfg.password else None,
            decode_responses=True,
        )
        await client.ping()
        await client.close()

        latency = (time.time() - start) * 1000
        return ComponentHealth(status="healthy", latency_ms=round(latency, 2))
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return ComponentHealth(status="unhealthy", message=str(e))


async def check_qdrant() -> ComponentHealth:
    """Check Qdrant vector database connection."""
    try:
        import httpx
        from src.core.config import get_settings

        settings = get_settings()
        qdrant_cfg = settings.qdrant
        start = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://{qdrant_cfg.host}:{qdrant_cfg.http_port}/readyz",
                timeout=5.0,
            )
            response.raise_for_status()

        latency = (time.time() - start) * 1000
        return ComponentHealth(status="healthy", latency_ms=round(latency, 2))
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return ComponentHealth(status="unhealthy", message=str(e))


async def check_llm_providers() -> dict[str, LLMProviderHealth]:
    """Check LLM provider configurations."""
    from src.core.config import get_settings

    settings = get_settings()
    llm = settings.llm
    providers: dict[str, LLMProviderHealth] = {}

    # Check Ollama
    if llm.ollama_base_url:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{llm.ollama_base_url}/api/tags",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    providers["ollama"] = LLMProviderHealth(
                        status="healthy",
                        url=llm.ollama_base_url,
                    )
                else:
                    providers["ollama"] = LLMProviderHealth(
                        status="degraded",
                        url=llm.ollama_base_url,
                    )
        except Exception:
            providers["ollama"] = LLMProviderHealth(
                status="configured",
                url=llm.ollama_base_url,
            )

    # Check OpenAI
    if llm.openai_api_key:
        providers["openai"] = LLMProviderHealth(
            status="configured",
            url="https://api.openai.com",
        )

    # Check Anthropic
    if llm.anthropic_api_key:
        providers["anthropic"] = LLMProviderHealth(
            status="configured",
            url="https://api.anthropic.com",
        )

    return providers


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the API is healthy with component details.

    This endpoint provides comprehensive health status including
    all infrastructure components.

    Returns:
        HealthResponse with detailed status.
    """
    from src.core.config import get_settings

    settings = get_settings()
    now = datetime.now(timezone.utc)
    uptime = int(time.time() - _server_start_time)

    # Check all components
    db_health = await check_database()
    redis_health = await check_redis()
    qdrant_health = await check_qdrant()
    llm_providers = await check_llm_providers()

    # Determine overall status
    component_statuses = [db_health.status, redis_health.status, qdrant_health.status]
    if all(s == "healthy" for s in component_statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in component_statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=now,
        version="1.0.0",
        environment=settings.environment,
        uptime_seconds=uptime,
        checked_at=now,
        components=ComponentsHealth(
            database=db_health,
            redis=redis_health,
            qdrant=qdrant_health,
            llm_providers=llm_providers,
        ),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Check if the API is ready to accept traffic.

    Returns:
        ReadinessResponse with individual check results.
    """
    checks: dict[str, Any] = {}
    all_ready = True

    # Check database
    db_health = await check_database()
    checks["database"] = {"status": db_health.status, "latency_ms": db_health.latency_ms}
    if db_health.status != "healthy":
        all_ready = False

    # Check redis
    redis_health = await check_redis()
    checks["redis"] = {"status": redis_health.status, "latency_ms": redis_health.latency_ms}
    if redis_health.status != "healthy":
        all_ready = False

    return ReadinessResponse(ready=all_ready, checks=checks)
