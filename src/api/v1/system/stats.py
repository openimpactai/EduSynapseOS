# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System statistics endpoints.

This module provides statistics endpoints for the admin dashboard:
- GET /stats - Get system-wide statistics
- GET /containers - Get Docker container status
"""

import logging
import docker
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_central_db, get_jwt_manager, get_password_hasher
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.system.auth_service import SystemAuthService
from src.infrastructure.database.models.central import Tenant, TenantAPICredential, APIKeyAuditLog

logger = logging.getLogger(__name__)

router = APIRouter()


class SystemStatsResponse(BaseModel):
    """System statistics response."""

    # Tenant counts
    total_tenants: int = Field(..., description="Total number of tenants")
    tenants_by_status: dict[str, int] = Field(
        ..., description="Tenant count by status"
    )
    tenants_by_tier: dict[str, int] = Field(..., description="Tenant count by tier")

    # API Usage
    api_calls_today: int = Field(..., description="API calls made today")
    api_calls_this_week: int = Field(..., description="API calls made this week")

    # Credentials
    total_credentials: int = Field(..., description="Total API credentials")
    active_credentials: int = Field(..., description="Active API credentials")

    # Timestamp
    generated_at: datetime = Field(..., description="When stats were generated")


async def _verify_system_admin(
    request: Request,
    db: AsyncSession,
    jwt_manager: JWTManager,
    password_hasher: PasswordHasher,
) -> None:
    """Verify the request is from a system admin."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ")[1]
    auth_service = SystemAuthService(db, jwt_manager, password_hasher)

    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics",
    description="Get system-wide statistics for the admin dashboard.",
)
async def get_system_stats(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> SystemStatsResponse:
    """Get system statistics.

    Args:
        request: HTTP request with Authorization header.
        db: Central database session.
        jwt_manager: JWT manager.
        password_hasher: Password hasher.

    Returns:
        SystemStatsResponse with system statistics.

    Raises:
        HTTPException: If not authenticated as system admin.
    """
    # Verify system admin
    await _verify_system_admin(request, db, jwt_manager, password_hasher)

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())

    # Get tenant counts
    total_tenants_result = await db.execute(
        select(func.count(Tenant.id)).where(Tenant.status != "deleted")
    )
    total_tenants = total_tenants_result.scalar() or 0

    # Tenants by status
    status_result = await db.execute(
        select(Tenant.status, func.count(Tenant.id))
        .where(Tenant.status != "deleted")
        .group_by(Tenant.status)
    )
    tenants_by_status = {row[0]: row[1] for row in status_result.fetchall()}

    # Tenants by tier
    tier_result = await db.execute(
        select(Tenant.tier, func.count(Tenant.id))
        .where(Tenant.status != "deleted")
        .group_by(Tenant.tier)
    )
    tenants_by_tier = {row[0]: row[1] for row in tier_result.fetchall()}

    # API calls today
    api_today_result = await db.execute(
        select(func.count(APIKeyAuditLog.id)).where(
            APIKeyAuditLog.created_at >= today_start
        )
    )
    api_calls_today = api_today_result.scalar() or 0

    # API calls this week
    api_week_result = await db.execute(
        select(func.count(APIKeyAuditLog.id)).where(
            APIKeyAuditLog.created_at >= week_start
        )
    )
    api_calls_this_week = api_week_result.scalar() or 0

    # Credentials count
    total_creds_result = await db.execute(select(func.count(TenantAPICredential.id)))
    total_credentials = total_creds_result.scalar() or 0

    active_creds_result = await db.execute(
        select(func.count(TenantAPICredential.id)).where(
            TenantAPICredential.is_active == True
        )
    )
    active_credentials = active_creds_result.scalar() or 0

    return SystemStatsResponse(
        total_tenants=total_tenants,
        tenants_by_status=tenants_by_status,
        tenants_by_tier=tenants_by_tier,
        api_calls_today=api_calls_today,
        api_calls_this_week=api_calls_this_week,
        total_credentials=total_credentials,
        active_credentials=active_credentials,
        generated_at=now,
    )


class ContainerInfo(BaseModel):
    """Container information."""

    name: str = Field(..., description="Container name")
    status: str = Field(..., description="Container status")
    health: str | None = Field(None, description="Health status if available")
    image: str = Field(..., description="Container image")
    ports: str | None = Field(None, description="Exposed ports")
    created: str = Field(..., description="Creation time")
    uptime: str = Field(..., description="Container uptime")


class ContainersResponse(BaseModel):
    """Containers list response."""

    containers: list[ContainerInfo] = Field(..., description="List of containers")
    total: int = Field(..., description="Total container count")
    healthy: int = Field(..., description="Healthy container count")
    unhealthy: int = Field(..., description="Unhealthy container count")
    generated_at: datetime = Field(..., description="When data was generated")


@router.get(
    "/containers",
    response_model=ContainersResponse,
    summary="Get Docker containers status",
    description="Get status of all EduSynapseOS Docker containers.",
)
async def get_containers(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> ContainersResponse:
    """Get Docker containers status.

    Args:
        request: HTTP request with Authorization header.
        db: Central database session.
        jwt_manager: JWT manager.
        password_hasher: Password hasher.

    Returns:
        ContainersResponse with container information.

    Raises:
        HTTPException: If not authenticated as system admin.
    """
    # Verify system admin
    await _verify_system_admin(request, db, jwt_manager, password_hasher)

    containers = []
    healthy_count = 0
    unhealthy_count = 0

    try:
        client = docker.from_env()
        docker_containers = client.containers.list(all=True, filters={"name": "edusynapse"})

        for c in docker_containers:
            # Get health status
            health = None
            health_status = c.attrs.get("State", {}).get("Health", {}).get("Status")
            status_str = c.status

            if health_status == "healthy":
                health = "healthy"
                healthy_count += 1
            elif health_status == "unhealthy":
                health = "unhealthy"
                unhealthy_count += 1
            elif c.status == "running":
                healthy_count += 1  # Running but no health check
                health = "running"
            else:
                unhealthy_count += 1

            # Format ports
            ports_list = []
            for port_info in c.attrs.get("NetworkSettings", {}).get("Ports", {}).items():
                port_key, bindings = port_info
                if bindings:
                    for binding in bindings:
                        host_port = binding.get("HostPort", "")
                        if host_port:
                            ports_list.append(f"{host_port}->{port_key}")
            ports_str = ", ".join(ports_list) if ports_list else None

            # Calculate uptime
            started_at = c.attrs.get("State", {}).get("StartedAt", "")
            uptime = ""
            if started_at and c.status == "running":
                try:
                    start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    delta = datetime.now(timezone.utc) - start_time
                    days = delta.days
                    hours = delta.seconds // 3600
                    if days > 0:
                        uptime = f"Up {days} days"
                    elif hours > 0:
                        uptime = f"Up {hours} hours"
                    else:
                        minutes = delta.seconds // 60
                        uptime = f"Up {minutes} minutes"
                except Exception:
                    uptime = c.status

            container = ContainerInfo(
                name=c.name,
                status=f"{status_str} ({health_status})" if health_status else status_str,
                health=health,
                image=c.image.tags[0] if c.image.tags else c.image.short_id,
                ports=ports_str,
                created=c.attrs.get("Created", "")[:19].replace("T", " "),
                uptime=uptime,
            )
            containers.append(container)

    except docker.errors.DockerException as e:
        logger.error(f"Docker error: {e}")
    except Exception as e:
        logger.error(f"Error getting container info: {e}")

    # Sort by name
    containers.sort(key=lambda x: x.name)

    return ContainersResponse(
        containers=containers,
        total=len(containers),
        healthy=healthy_count,
        unhealthy=unhealthy_count,
        generated_at=datetime.now(timezone.utc),
    )
