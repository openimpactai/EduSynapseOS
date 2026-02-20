# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostics API endpoints.

This module provides endpoints for learning diagnostics:
- POST /scan - Start a diagnostic scan
- GET /scans - List diagnostic scans
- GET /scans/{scan_id} - Get scan details
- GET /student/{student_id} - Get student risk summary
- POST /batch - Start batch scan (teacher only)

Example:
    POST /api/v1/diagnostics/scan
    {
        "scan_type": "targeted",
        "indicator_types": ["dyslexia", "attention"]
    }
"""

import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import (
    get_tenant_db,
    require_auth,
    require_teacher_or_admin,
    require_tenant,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.infrastructure.database.models.tenant.diagnostic import (
    DiagnosticIndicator,
    DiagnosticRecommendation,
    DiagnosticScan,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request Models
# ============================================================================


class StartScanRequest(BaseModel):
    """Request to start a diagnostic scan."""

    scan_type: str = Field(
        default="targeted",
        description="Type of scan: full, targeted, quick",
        examples=["full", "targeted", "quick"],
    )
    indicator_types: list[str] | None = Field(
        default=None,
        description="Specific indicators to check (None = all)",
        examples=[["dyslexia", "attention"]],
    )


class BatchScanRequest(BaseModel):
    """Request to start batch diagnostic scans."""

    student_ids: list[UUID] | None = Field(
        default=None,
        description="Specific student IDs (None = all in class)",
    )
    class_id: UUID | None = Field(
        default=None,
        description="Class ID to scan all students",
    )
    scan_type: str = Field(
        default="targeted",
        description="Type of scan for all students",
    )


# ============================================================================
# Response Models
# ============================================================================


class IndicatorResponse(BaseModel):
    """Diagnostic indicator response."""

    id: str = Field(description="Indicator ID")
    indicator_type: str = Field(description="Type of indicator")
    risk_score: float = Field(description="Risk score 0.0-1.0")
    confidence: float | None = Field(description="Confidence level")
    threshold_level: str | None = Field(description="Severity level")
    evidence: list[dict[str, Any]] = Field(description="Supporting evidence")
    created_at: datetime = Field(description="When detected")


class RecommendationResponse(BaseModel):
    """Diagnostic recommendation response."""

    id: str = Field(description="Recommendation ID")
    recommendation_type: str = Field(description="Type of recommendation")
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    priority: int = Field(description="Priority level")
    for_teacher: bool = Field(description="Is for teachers")
    for_parent: bool = Field(description="Is for parents")
    disclaimer: str | None = Field(description="Professional disclaimer")


class ScanSummaryResponse(BaseModel):
    """Summary response for a scan."""

    id: str = Field(description="Scan ID")
    scan_type: str = Field(description="Type of scan")
    status: str = Field(description="Scan status")
    trigger_reason: str | None = Field(description="What triggered the scan")
    risk_score: float | None = Field(description="Overall risk score")
    findings_count: int = Field(description="Number of findings")
    started_at: datetime | None = Field(description="When started")
    completed_at: datetime | None = Field(description="When completed")
    created_at: datetime = Field(description="When created")


class ScanDetailResponse(BaseModel):
    """Detailed response for a scan."""

    id: str = Field(description="Scan ID")
    student_id: str = Field(description="Student ID")
    scan_type: str = Field(description="Type of scan")
    status: str = Field(description="Scan status")
    trigger_reason: str | None = Field(description="What triggered the scan")
    risk_score: float | None = Field(description="Overall risk score")
    findings_count: int = Field(description="Number of findings")
    started_at: datetime | None = Field(description="When started")
    completed_at: datetime | None = Field(description="When completed")
    created_at: datetime = Field(description="When created")
    indicators: list[IndicatorResponse] = Field(description="Detected indicators")
    recommendations: list[RecommendationResponse] = Field(description="Recommendations")


class StudentRiskSummaryResponse(BaseModel):
    """Risk summary for a student."""

    student_id: str = Field(description="Student ID")
    overall_risk_score: float | None = Field(description="Current overall risk")
    risk_level: str = Field(description="Risk level: low, medium, high, unknown")
    last_scan_at: datetime | None = Field(description="Last scan timestamp")
    total_scans: int = Field(description="Total scans performed")
    active_indicators: list[dict[str, Any]] = Field(description="Current indicators")
    trend: str = Field(description="Trend: improving, stable, concerning")


class StartScanResponse(BaseModel):
    """Response after starting a scan."""

    scan_id: str = Field(description="Created scan ID")
    status: str = Field(description="Initial status (pending)")
    message: str = Field(description="Status message")


class BatchScanResponse(BaseModel):
    """Response after starting batch scans."""

    batch_id: str = Field(description="Batch operation ID")
    scans_queued: int = Field(description="Number of scans queued")
    message: str = Field(description="Status message")


class ScanListResponse(BaseModel):
    """Paginated scan list response."""

    items: list[ScanSummaryResponse] = Field(description="Scan items")
    total: int = Field(description="Total count")
    page: int = Field(description="Current page")
    page_size: int = Field(description="Items per page")
    pages: int = Field(description="Total pages")


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/scan",
    response_model=StartScanResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start diagnostic scan",
    description="Queue a diagnostic scan for the current user.",
)
async def start_scan(
    request: StartScanRequest,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> StartScanResponse:
    """Start a diagnostic scan for the current user.

    The scan runs asynchronously in the background.

    Args:
        request: Scan configuration.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        StartScanResponse with scan ID.
    """
    from src.infrastructure.background.tasks import run_diagnostic_scan

    logger.info(
        "Starting diagnostic scan: user=%s, type=%s",
        current_user.id,
        request.scan_type,
    )

    # Create pending scan record
    scan = DiagnosticScan(
        student_id=current_user.id,
        scan_type=request.scan_type,
        trigger_reason="api_request",
        status="pending",
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    # Queue the scan
    run_diagnostic_scan.send(
        tenant.code,
        current_user.id,
        scan_type=request.scan_type,
        trigger_reason="api_request",
        indicator_types=request.indicator_types,
    )

    return StartScanResponse(
        scan_id=scan.id,
        status="pending",
        message="Diagnostic scan queued successfully",
    )


@router.get(
    "/scans",
    response_model=ScanListResponse,
    summary="List diagnostic scans",
    description="Get list of diagnostic scans for the current user.",
)
async def list_scans(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = 10,
    status_filter: str | None = None,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ScanListResponse:
    """List diagnostic scans.

    Students see their own scans, teachers can see their students' scans.

    Args:
        page: Page number.
        page_size: Items per page.
        status_filter: Filter by status.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ScanListResponse with paginated scans.
    """
    # Base query for current user
    query = select(DiagnosticScan).where(
        DiagnosticScan.student_id == current_user.id
    )

    if status_filter:
        query = query.where(DiagnosticScan.status == status_filter)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Paginate
    query = query.order_by(desc(DiagnosticScan.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    scans = result.scalars().all()

    pages = (total + page_size - 1) // page_size if total > 0 else 0

    return ScanListResponse(
        items=[
            ScanSummaryResponse(
                id=scan.id,
                scan_type=scan.scan_type,
                status=scan.status,
                trigger_reason=scan.trigger_reason,
                risk_score=float(scan.risk_score) if scan.risk_score else None,
                findings_count=scan.findings_count,
                started_at=scan.started_at,
                completed_at=scan.completed_at,
                created_at=scan.created_at,
            )
            for scan in scans
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get(
    "/scans/{scan_id}",
    response_model=ScanDetailResponse,
    summary="Get scan details",
    description="Get detailed results of a diagnostic scan.",
)
async def get_scan_detail(
    scan_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> ScanDetailResponse:
    """Get detailed scan results.

    Args:
        scan_id: Scan ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        ScanDetailResponse with full details.

    Raises:
        HTTPException: If scan not found or unauthorized.
    """
    query = (
        select(DiagnosticScan)
        .where(DiagnosticScan.id == str(scan_id))
        .options(
            selectinload(DiagnosticScan.indicators),
            selectinload(DiagnosticScan.recommendations),
        )
    )

    result = await db.execute(query)
    scan = result.scalar_one_or_none()

    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )

    # Authorization check
    if scan.student_id != current_user.id:
        if not (current_user.is_teacher or current_user.is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this scan",
            )

    return ScanDetailResponse(
        id=scan.id,
        student_id=scan.student_id,
        scan_type=scan.scan_type,
        status=scan.status,
        trigger_reason=scan.trigger_reason,
        risk_score=float(scan.risk_score) if scan.risk_score else None,
        findings_count=scan.findings_count,
        started_at=scan.started_at,
        completed_at=scan.completed_at,
        created_at=scan.created_at,
        indicators=[
            IndicatorResponse(
                id=ind.id,
                indicator_type=ind.indicator_type,
                risk_score=float(ind.risk_score),
                confidence=float(ind.confidence) if ind.confidence else None,
                threshold_level=ind.threshold_level,
                evidence=ind.evidence or [],
                created_at=ind.created_at,
            )
            for ind in scan.indicators
        ],
        recommendations=[
            RecommendationResponse(
                id=rec.id,
                recommendation_type=rec.recommendation_type,
                title=rec.title,
                description=rec.description,
                priority=rec.priority,
                for_teacher=rec.for_teacher,
                for_parent=rec.for_parent,
                disclaimer=rec.disclaimer,
            )
            for rec in scan.recommendations
        ],
    )


@router.get(
    "/student/{student_id}",
    response_model=StudentRiskSummaryResponse,
    summary="Get student risk summary",
    description="Get risk summary for a student.",
)
async def get_student_risk_summary(
    student_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> StudentRiskSummaryResponse:
    """Get student risk summary.

    Args:
        student_id: Student ID.
        current_user: Authenticated user.
        tenant: Tenant context.
        db: Database session.

    Returns:
        StudentRiskSummaryResponse with risk summary.

    Raises:
        HTTPException: If unauthorized.
    """
    # Authorization check
    if str(student_id) != current_user.id:
        if not (current_user.is_teacher or current_user.is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this student's diagnostics",
            )

    # Get latest completed scan with indicators
    latest_scan_query = (
        select(DiagnosticScan)
        .where(
            and_(
                DiagnosticScan.student_id == str(student_id),
                DiagnosticScan.status == "completed",
            )
        )
        .order_by(desc(DiagnosticScan.completed_at))
        .limit(1)
        .options(selectinload(DiagnosticScan.indicators))
    )

    result = await db.execute(latest_scan_query)
    latest_scan = result.scalar_one_or_none()

    # Count total scans
    count_query = select(func.count()).where(
        DiagnosticScan.student_id == str(student_id)
    )
    total_scans = await db.scalar(count_query) or 0

    # Determine risk level and indicators
    if latest_scan and latest_scan.risk_score:
        risk_score = float(latest_scan.risk_score)
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        active_indicators = [
            {
                "type": ind.indicator_type,
                "risk_score": float(ind.risk_score),
                "threshold_level": ind.threshold_level,
            }
            for ind in latest_scan.indicators
            if float(ind.risk_score) >= 0.4
        ]
    else:
        risk_score = None
        risk_level = "unknown"
        active_indicators = []

    # Calculate trend (compare with previous scan)
    trend = "stable"
    if latest_scan and latest_scan.risk_score:
        prev_scan_query = (
            select(DiagnosticScan)
            .where(
                and_(
                    DiagnosticScan.student_id == str(student_id),
                    DiagnosticScan.status == "completed",
                    DiagnosticScan.id != latest_scan.id,
                )
            )
            .order_by(desc(DiagnosticScan.completed_at))
            .limit(1)
        )
        prev_result = await db.execute(prev_scan_query)
        prev_scan = prev_result.scalar_one_or_none()

        if prev_scan and prev_scan.risk_score:
            diff = float(latest_scan.risk_score) - float(prev_scan.risk_score)
            if diff <= -0.1:
                trend = "improving"
            elif diff >= 0.1:
                trend = "concerning"

    return StudentRiskSummaryResponse(
        student_id=str(student_id),
        overall_risk_score=risk_score,
        risk_level=risk_level,
        last_scan_at=latest_scan.completed_at if latest_scan else None,
        total_scans=total_scans,
        active_indicators=active_indicators,
        trend=trend,
    )


@router.post(
    "/batch",
    response_model=BatchScanResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start batch scan",
    description="Queue batch diagnostic scans for multiple students. Requires teacher role.",
)
async def start_batch_scan(
    request: BatchScanRequest,
    current_user: CurrentUser = Depends(require_teacher_or_admin),
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
) -> BatchScanResponse:
    """Start batch diagnostic scans.

    Args:
        request: Batch scan configuration.
        current_user: Authenticated teacher or admin.
        tenant: Tenant context.
        db: Database session.

    Returns:
        BatchScanResponse with batch info.

    Raises:
        HTTPException: If no students specified.
    """
    from src.infrastructure.background.tasks import run_batch_diagnostic_scans

    if not request.student_ids and not request.class_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either student_ids or class_id must be provided",
        )

    batch_id = str(uuid4())

    logger.info(
        "Starting batch diagnostic scan: batch=%s, teacher=%s",
        batch_id,
        current_user.id,
    )

    # Count students
    scans_queued = len(request.student_ids) if request.student_ids else 0

    # Queue the batch scan
    run_batch_diagnostic_scans.send(
        tenant.code,
        student_ids=[str(sid) for sid in request.student_ids] if request.student_ids else None,
        class_id=str(request.class_id) if request.class_id else None,
        scan_type=request.scan_type,
    )

    return BatchScanResponse(
        batch_id=batch_id,
        scans_queued=scans_queued,
        message=f"Batch scan queued for {scans_queued if scans_queued else 'class'} students",
    )
