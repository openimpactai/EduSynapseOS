# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMS provisioning API endpoints.

This module provides the provisioning endpoint for LMS integration:
- POST /provision - Atomic provisioning of school, class, and student

The provisioning endpoint allows LMS systems to create or update school,
class, and student entities in a single atomic operation. It uses upsert
semantics - entities are created if they don't exist, or updated if they do.

NOTE: Curriculum data (frameworks, stages, grades, subjects, units, topics)
is NOT created through this endpoint. Curriculum data is synced from the
Central Curriculum service. Classes reference existing grade levels using
code-based composite keys.

Authentication:
    Requires authenticated user (via token exchange) and tenant context.

    Flow for LMS integration:
    1. Call POST /api/v1/auth/exchange with API key + user info
    2. Receive JWT tokens
    3. Call this endpoint with JWT token

Example:
    POST /api/v1/lms/provision
    Headers:
        Authorization: Bearer <token>
        X-Tenant-Code: test_school
    Body:
        {
            "school": {"code": "greenwood_academy", "name": "Greenwood Academy"},
            "academic_year": {"code": "2024-2025", ...},
            "class": {
                "code": "4A",
                "name": "Class 4A",
                "framework_code": "UK-NC-2014",
                "stage_code": "KS2",
                "grade_code": "Y4"
            },
            "student": {"email": "noah@example.com", ...}
        }
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_tenant_db, require_auth
from src.api.middleware.auth import CurrentUser
from src.domains.provisioning import (
    ProvisioningService,
    ProvisioningError,
    GradeLevelNotFoundError,
)
from src.models.provisioning import (
    ProvisioningRequest,
    ProvisioningResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_provisioning_service(db: AsyncSession) -> ProvisioningService:
    """Get provisioning service instance.

    Args:
        db: Tenant database session.

    Returns:
        Configured ProvisioningService instance.
    """
    return ProvisioningService(db=db)


@router.post(
    "/provision",
    response_model=ProvisioningResponse,
    status_code=status.HTTP_200_OK,
    summary="Provision LMS data",
    description="""
    Atomic provisioning of school, class, and student data.

    This endpoint uses upsert semantics:
    - Entities are created if they don't exist
    - Existing entities are updated with new values
    - The entire operation is atomic (all or nothing)

    The endpoint provisions:
    1. School and academic year
    2. Class (referencing existing grade level from Central Curriculum)
    3. Student with class enrollment

    **Important:** The class must reference an existing grade level using
    framework_code + stage_code + grade_code. Grade levels are created by
    the Central Curriculum sync service, not by this endpoint.

    **Authentication:** Requires authenticated user via token exchange.

    **Idempotent:** Safe to call multiple times with the same data.
    """,
    responses={
        200: {
            "description": "Provisioning completed successfully",
            "model": ProvisioningResponse,
        },
        400: {
            "description": "Invalid request data",
        },
        401: {
            "description": "Not authenticated",
        },
        404: {
            "description": "Referenced grade level not found",
        },
        500: {
            "description": "Provisioning failed",
        },
    },
)
async def provision(
    request: ProvisioningRequest,
    db: AsyncSession = Depends(get_tenant_db),
    current_user: CurrentUser = Depends(require_auth),
) -> ProvisioningResponse:
    """Provision school, class, and student data.

    This endpoint atomically provisions all entities required for a student
    to use EduSynapse. It's designed for LMS integration and uses upsert
    semantics for idempotent operations.

    The class must reference an existing grade level that has been synced
    from Central Curriculum.

    Args:
        request: Complete provisioning request.
        db: Tenant database session.
        current_user: Authenticated user (from token exchange).

    Returns:
        ProvisioningResponse with results.

    Raises:
        HTTPException: If provisioning fails or grade level not found.
    """
    logger.info(
        "Provisioning request received: school=%s, class=%s, student=%s, user=%s",
        request.school.code,
        request.class_.code,
        request.student.email,
        current_user.id,
    )

    service = _get_provisioning_service(db)

    try:
        result = await service.provision(request)

        logger.info(
            "Provisioning completed: student_id=%s, school=%s, class=%s",
            result.student_id,
            result.school_code,
            result.class_code,
        )

        return result

    except GradeLevelNotFoundError as e:
        logger.error("Grade level not found: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    except ProvisioningError as e:
        logger.error("Provisioning failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
