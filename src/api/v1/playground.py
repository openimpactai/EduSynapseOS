# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Playground API endpoints.

This module provides public endpoints for the playground/demo functionality:
- GET /users - List demo users by type and country
- GET /frameworks - List available curriculum frameworks

These endpoints use API credentials (X-API-Key, X-API-Secret) for authentication
instead of JWT tokens, allowing the frontend to fetch user lists before login.

Authentication:
    Requires valid API credentials in headers:
    - X-API-Key: The tenant's API key
    - X-API-Secret: The tenant's API secret
    - X-Tenant-Code: The tenant code (should be 'playground')
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_tenant_db, require_api_credentials
from src.api.middleware.tenant import TenantContext
from src.infrastructure.database.models.tenant.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class PlaygroundUser(BaseModel):
    """Playground user information."""

    id: str
    email: str
    first_name: str
    last_name: str
    display_name: str | None
    user_type: str
    country_code: str | None = None
    level: str | None = None


class PlaygroundUserListResponse(BaseModel):
    """Response for playground user list."""

    users: list[PlaygroundUser]
    total: int


class PlaygroundFramework(BaseModel):
    """Playground curriculum framework."""

    code: str
    name: str
    country: str
    version: str | None


class PlaygroundFrameworkListResponse(BaseModel):
    """Response for playground framework list."""

    frameworks: list[PlaygroundFramework]


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/users",
    response_model=PlaygroundUserListResponse,
    summary="List playground users",
    description="""
    List available demo users for the playground.

    This endpoint uses API credentials for authentication, not JWT tokens.
    It allows the frontend to fetch available users before the user selects
    one and performs token exchange.

    **Authentication:** Requires X-API-Key and X-API-Secret headers.
    """,
)
async def list_playground_users(
    user_type: Annotated[
        str | None,
        Query(description="Filter by user type (student, teacher, parent)"),
    ] = None,
    country_code: Annotated[
        str | None,
        Query(description="Filter by country code (GB, US, TR, RW, NG)"),
    ] = None,
    tenant: TenantContext = Depends(require_api_credentials),
    db: AsyncSession = Depends(get_tenant_db),
) -> PlaygroundUserListResponse:
    """List playground users.

    Args:
        user_type: Optional filter by user type.
        country_code: Optional filter by country code.
        tenant: Tenant context (validated via API credentials).
        db: Database session.

    Returns:
        List of playground users.
    """
    # Build query
    query = select(User).where(
        User.deleted_at.is_(None),
        User.status == "active",
    )

    # Filter by user type
    if user_type:
        query = query.where(User.user_type == user_type)
    else:
        # Only return student, teacher, parent - not admins
        query = query.where(User.user_type.in_(["student", "teacher", "parent"]))

    # Filter by country code (stored in extra_data JSONB field)
    # For playground, we store country in user's extra_data JSON
    if country_code:
        # PostgreSQL JSON query for extra_data->country_code
        query = query.where(
            User.extra_data["country_code"].astext == country_code
        )

    # Order by user type, then by name
    query = query.order_by(User.user_type, User.first_name, User.last_name)

    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()

    # Transform to response
    playground_users = []
    for user in users:
        # Extract country and level from extra_data if available
        extra_data = user.extra_data or {}
        country = extra_data.get("country_code")
        level = extra_data.get("level")

        playground_users.append(
            PlaygroundUser(
                id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                display_name=user.display_name,
                user_type=user.user_type,
                country_code=country,
                level=level,
            )
        )

    return PlaygroundUserListResponse(
        users=playground_users,
        total=len(playground_users),
    )


@router.get(
    "/frameworks",
    response_model=PlaygroundFrameworkListResponse,
    summary="List playground frameworks",
    description="""
    List available curriculum frameworks for the playground.

    **Authentication:** Requires X-API-Key and X-API-Secret headers.
    """,
)
async def list_playground_frameworks(
    tenant: TenantContext = Depends(require_api_credentials),
    db: AsyncSession = Depends(get_tenant_db),
) -> PlaygroundFrameworkListResponse:
    """List available curriculum frameworks.

    Args:
        tenant: Tenant context (validated via API credentials).
        db: Database session.

    Returns:
        List of available frameworks.
    """
    from src.infrastructure.database.models.tenant.curriculum import CurriculumFramework

    query = select(CurriculumFramework).where(
        CurriculumFramework.is_active == True,
    ).order_by(CurriculumFramework.name)

    result = await db.execute(query)
    frameworks = result.scalars().all()

    return PlaygroundFrameworkListResponse(
        frameworks=[
            PlaygroundFramework(
                code=f.code,
                name=f.name,
                country=f.country_code or "INT",
                version=f.version,
            )
            for f in frameworks
        ]
    )
