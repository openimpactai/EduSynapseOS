# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic background tasks for EduSynapseOS.

Tasks for running learning difficulty detection and analysis.

IMPORTANT: These tasks identify INDICATORS only, not diagnoses.
Professional evaluation is recommended for concerning patterns.

Available actors:
- run_diagnostic_scan: Run full or targeted scan for a student
- run_batch_diagnostic_scans: Run scans for multiple students
- assess_risk_score: Assess risk for specific indicator type
- generate_diagnostic_report: Generate diagnostic report
- check_diagnostic_thresholds: Check thresholds and create alerts
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import dramatiq

from src.infrastructure.background.broker import Priority, Queues, setup_dramatiq
from src.infrastructure.background.tasks.base import run_async

# Setup broker before defining actors
setup_dramatiq()

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=300000,  # 5 minutes
    priority=Priority.DIAGNOSTIC,
)
def run_diagnostic_scan(
    tenant_code: str,
    student_id: str,
    scan_type: str = "full",
    trigger_reason: str | None = None,
    indicator_types: list[str] | None = None,
) -> dict[str, Any]:
    """Run a diagnostic scan for a student.

    IMPORTANT: This does NOT diagnose any condition.
    It identifies indicators for professional evaluation.

    Args:
        tenant_code: Tenant code.
        student_id: Student identifier.
        scan_type: Type of scan ("full", "targeted", "quick").
        trigger_reason: What triggered this scan.
        indicator_types: For targeted scan, list of indicator types to check.
            Options: "dyslexia", "dyscalculia", "attention", "auditory", "visual"

    Returns:
        Scan result with findings and risk scores.

    Example:
        # Full scan
        run_diagnostic_scan.send("test_school", "student-uuid")

        # Targeted scan
        run_diagnostic_scan.send(
            "test_school",
            "student-uuid",
            scan_type="targeted",
            indicator_types=["dyslexia", "attention"],
        )
    """

    async def _scan() -> dict[str, Any]:
        from src.core.diagnostics.detectors.base import IndicatorType
        from src.core.diagnostics.service import (
            DIAGNOSTIC_DISCLAIMER,
            get_diagnostic_service,
        )
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.events import get_event_bus

        scan_start = datetime.now(timezone.utc)

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                service = get_diagnostic_service()

                if scan_type == "targeted" and indicator_types:
                    # Convert string types to IndicatorType enum
                    types = [IndicatorType(t) for t in indicator_types]
                    scan = await service.run_targeted_scan(
                        db=session,
                        student_id=student_id,
                        indicator_types=types,
                        trigger_reason=trigger_reason,
                    )
                else:
                    scan = await service.run_scan(
                        db=session,
                        student_id=student_id,
                        scan_type=scan_type,
                        trigger_reason=trigger_reason,
                    )

                # Publish completion event
                event_bus = get_event_bus()
                await event_bus.publish(
                    "diagnostic.scan.completed",
                    {
                        "scan_id": scan.id,
                        "student_id": student_id,
                        "tenant_code": tenant_code,
                        "scan_type": scan_type,
                        "status": scan.status,
                        "risk_score": float(scan.risk_score) if scan.risk_score else 0,
                        "findings_count": scan.findings_count,
                        "has_concerns": scan.has_concerns,
                    },
                )

                scan_duration = (datetime.now(timezone.utc) - scan_start).total_seconds()

                logger.info(
                    "Diagnostic scan completed: scan_id=%s, student=%s, "
                    "risk_score=%.2f, findings=%d, duration=%.2fs",
                    scan.id,
                    student_id,
                    float(scan.risk_score) if scan.risk_score else 0,
                    scan.findings_count,
                    scan_duration,
                )

                return {
                    "scan_id": scan.id,
                    "student_id": student_id,
                    "tenant_code": tenant_code,
                    "scan_type": scan_type,
                    "status": scan.status,
                    "risk_score": float(scan.risk_score) if scan.risk_score else 0,
                    "findings_count": scan.findings_count,
                    "has_concerns": scan.has_concerns,
                    "duration_seconds": scan_duration,
                    "disclaimer": DIAGNOSTIC_DISCLAIMER,
                }

        except Exception as e:
            logger.error(
                "Diagnostic scan failed: student=%s, error=%s",
                student_id,
                str(e),
                exc_info=True,
            )
            # Import disclaimer for error response
            from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER

            return {
                "student_id": student_id,
                "tenant_code": tenant_code,
                "scan_type": scan_type,
                "status": "failed",
                "error": str(e),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            }

    return run_async(_scan())


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=600000,  # 10 minutes
    priority=Priority.LOW,
)
def run_batch_diagnostic_scans(
    tenant_code: str,
    student_ids: list[str] | None = None,
    class_id: str | None = None,
    scan_type: str = "full",
    trigger_reason: str = "batch_scan",
) -> dict[str, Any]:
    """Run batch diagnostic scans for multiple students.

    Can specify student_ids directly or provide class_id to scan all students
    in a class. If neither provided, scans all active students in tenant.

    IMPORTANT: This does NOT diagnose any condition.
    It identifies indicators for professional evaluation.

    Args:
        tenant_code: Tenant code.
        student_ids: List of student IDs to scan.
        class_id: Class ID to scan all students in class.
        scan_type: Type of scan ("full", "quick").
        trigger_reason: What triggered this batch scan.

    Returns:
        Batch scan result with statistics.

    Example:
        # Scan specific students
        run_batch_diagnostic_scans.send("test_school", student_ids=["s1", "s2", "s3"])

        # Scan entire class
        run_batch_diagnostic_scans.send("test_school", class_id="class-uuid")
    """

    async def _batch_scan() -> dict[str, Any]:
        from sqlalchemy import select

        from src.core.diagnostics.service import (
            DIAGNOSTIC_DISCLAIMER,
            get_diagnostic_service,
        )
        from src.infrastructure.database.models.tenant.user import User
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.events import get_event_bus

        batch_id = str(uuid4())
        batch_start = datetime.now(timezone.utc)

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Determine students to scan
                if student_ids:
                    ids_to_scan = student_ids
                elif class_id:
                    # Get students in class - use deleted_at for soft delete check
                    stmt = select(User.id).where(
                        User.user_type == "student",
                        User.status == "active",
                        User.deleted_at.is_(None),
                    )
                    result = await session.execute(stmt)
                    ids_to_scan = [str(row[0]) for row in result.fetchall()]
                else:
                    # Get all active students - use deleted_at for soft delete check
                    stmt = select(User.id).where(
                        User.user_type == "student",
                        User.status == "active",
                        User.deleted_at.is_(None),
                    )
                    result = await session.execute(stmt)
                    ids_to_scan = [str(row[0]) for row in result.fetchall()]

                service = get_diagnostic_service()

                scanned = 0
                skipped = 0
                high_risk_students: list[dict[str, Any]] = []
                errors: list[dict[str, str]] = []

                for sid in ids_to_scan:
                    try:
                        # Check if recent scan exists (within last 24 hours)
                        recent_scan = await service.get_latest_scan(session, sid)
                        if recent_scan and recent_scan.completed_at:
                            age_hours = (
                                datetime.now(timezone.utc) - recent_scan.completed_at
                            ).total_seconds() / 3600
                            if age_hours < 24:
                                skipped += 1
                                continue

                        # Run scan
                        scan = await service.run_scan(
                            db=session,
                            student_id=sid,
                            scan_type=scan_type,
                            trigger_reason=trigger_reason,
                        )
                        scanned += 1

                        if scan.risk_score and float(scan.risk_score) >= 0.7:
                            high_risk_students.append({
                                "student_id": sid,
                                "scan_id": scan.id,
                                "risk_score": float(scan.risk_score),
                                "findings_count": scan.findings_count,
                            })

                    except Exception as e:
                        errors.append({
                            "student_id": sid,
                            "error": str(e),
                        })

                batch_duration = (datetime.now(timezone.utc) - batch_start).total_seconds()

                # Publish batch completion event
                event_bus = get_event_bus()
                await event_bus.publish(
                    "diagnostic.batch.completed",
                    {
                        "batch_id": batch_id,
                        "tenant_code": tenant_code,
                        "total": len(ids_to_scan),
                        "scanned": scanned,
                        "skipped": skipped,
                        "high_risk_count": len(high_risk_students),
                        "error_count": len(errors),
                    },
                )

                logger.info(
                    "Batch diagnostic scan completed: batch_id=%s, "
                    "total=%d, scanned=%d, high_risk=%d, duration=%.2fs",
                    batch_id,
                    len(ids_to_scan),
                    scanned,
                    len(high_risk_students),
                    batch_duration,
                )

                return {
                    "batch_id": batch_id,
                    "tenant_code": tenant_code,
                    "total_students": len(ids_to_scan),
                    "scanned": scanned,
                    "skipped": skipped,
                    "high_risk_students": high_risk_students[:20],  # Limit for response size
                    "high_risk_count": len(high_risk_students),
                    "errors": errors[:10],
                    "error_count": len(errors),
                    "duration_seconds": batch_duration,
                    "disclaimer": DIAGNOSTIC_DISCLAIMER,
                }

        except Exception as e:
            logger.error(
                "Batch diagnostic scan failed: tenant=%s, error=%s",
                tenant_code,
                str(e),
                exc_info=True,
            )
            from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER

            return {
                "batch_id": batch_id,
                "tenant_code": tenant_code,
                "status": "failed",
                "error": str(e),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            }

    return run_async(_batch_scan())


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=2,
    time_limit=60000,  # 1 minute
    priority=Priority.DIAGNOSTIC,
)
def assess_risk_score(
    tenant_code: str,
    student_id: str,
    indicator_type: str,
) -> dict[str, Any]:
    """Assess risk score for a specific indicator type.

    Quick assessment for a single indicator without full scan.

    IMPORTANT: This does NOT diagnose any condition.
    Risk scores are indicators for professional evaluation.

    Args:
        tenant_code: Tenant code.
        student_id: Student identifier.
        indicator_type: Type of indicator to assess.
            Options: "dyslexia", "dyscalculia", "attention", "auditory", "visual"

    Returns:
        Risk assessment result.

    Example:
        assess_risk_score.send("test_school", "student-uuid", "attention")
    """

    async def _assess() -> dict[str, Any]:
        from src.core.diagnostics.detectors.attention import AttentionDetector
        from src.core.diagnostics.detectors.auditory import AuditoryDetector
        from src.core.diagnostics.detectors.dyscalculia import DyscalculiaDetector
        from src.core.diagnostics.detectors.dyslexia import DyslexiaDetector
        from src.core.diagnostics.detectors.visual import VisualDetector
        from src.core.diagnostics.service import (
            DIAGNOSTIC_DISCLAIMER,
            DiagnosticService,
        )
        from src.infrastructure.database.tenant_manager import get_worker_db_manager

        # Map indicator type to detector
        detector_map = {
            "dyslexia": DyslexiaDetector,
            "dyscalculia": DyscalculiaDetector,
            "attention": AttentionDetector,
            "auditory": AuditoryDetector,
            "visual": VisualDetector,
        }

        if indicator_type not in detector_map:
            return {
                "student_id": student_id,
                "indicator_type": indicator_type,
                "status": "failed",
                "error": f"Unknown indicator type: {indicator_type}",
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            }

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Create service and fetch data
                service = DiagnosticService()
                student_data = await service._fetch_student_data(session, student_id)

                # Run single detector
                detector = detector_map[indicator_type]()
                result = await detector.analyze(student_data)

                logger.debug(
                    "Risk assessment: student=%s, indicator=%s, score=%.2f",
                    student_id,
                    indicator_type,
                    result.risk_score,
                )

                return {
                    "student_id": student_id,
                    "tenant_code": tenant_code,
                    "indicator_type": indicator_type,
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "threshold_level": result.threshold_level.value,
                    "is_significant": result.is_significant,
                    "sample_size": result.sample_size,
                    "analysis_summary": result.analysis_summary,
                    "evidence_count": len(result.evidence),
                    "disclaimer": DIAGNOSTIC_DISCLAIMER,
                }

        except Exception as e:
            logger.error(
                "Risk assessment failed: student=%s, indicator=%s, error=%s",
                student_id,
                indicator_type,
                str(e),
                exc_info=True,
            )
            from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER

            return {
                "student_id": student_id,
                "indicator_type": indicator_type,
                "status": "failed",
                "error": str(e),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            }

    return run_async(_assess())


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=120000,  # 2 minutes
    priority=Priority.NORMAL,
)
def generate_diagnostic_report(
    tenant_code: str,
    student_id: str,
    scan_id: str | None = None,
    include_recommendations: bool = True,
    include_history: bool = False,
) -> dict[str, Any]:
    """Generate a diagnostic report for a student.

    Creates a comprehensive report based on the latest or specified scan.

    IMPORTANT: This report does NOT contain diagnoses.
    It identifies indicators for professional evaluation.

    Args:
        tenant_code: Tenant code.
        student_id: Student identifier.
        scan_id: Specific scan ID to report on. If None, uses latest.
        include_recommendations: Whether to include recommendations.
        include_history: Whether to include historical scans.

    Returns:
        Report data with findings, recommendations, and disclaimer.

    Example:
        generate_diagnostic_report.send("test_school", "student-uuid")
    """

    async def _generate() -> dict[str, Any]:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER
        from src.infrastructure.database.models.tenant.diagnostic import (
            DiagnosticScan,
        )
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.events import get_event_bus

        report_id = str(uuid4())

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                # Get scan
                if scan_id:
                    stmt = (
                        select(DiagnosticScan)
                        .where(DiagnosticScan.id == scan_id)
                        .options(
                            selectinload(DiagnosticScan.indicators),
                            selectinload(DiagnosticScan.recommendations),
                        )
                    )
                else:
                    stmt = (
                        select(DiagnosticScan)
                        .where(
                            DiagnosticScan.student_id == student_id,
                            DiagnosticScan.status == "completed",
                        )
                        .options(
                            selectinload(DiagnosticScan.indicators),
                            selectinload(DiagnosticScan.recommendations),
                        )
                        .order_by(DiagnosticScan.completed_at.desc())
                        .limit(1)
                    )

                result = await session.execute(stmt)
                scan = result.scalar_one_or_none()

                if not scan:
                    return {
                        "report_id": report_id,
                        "student_id": student_id,
                        "status": "no_scan_found",
                        "message": "No completed diagnostic scan found",
                        "disclaimer": DIAGNOSTIC_DISCLAIMER,
                    }

                # Build indicator details
                indicators = []
                for ind in scan.indicators:
                    indicators.append({
                        "type": ind.indicator_type,
                        "risk_score": float(ind.risk_score),
                        "confidence": float(ind.confidence) if ind.confidence else None,
                        "threshold_level": ind.threshold_level,
                        "evidence_count": len(ind.evidence) if ind.evidence else 0,
                    })

                # Build recommendations if requested
                recommendations = []
                if include_recommendations:
                    for rec in scan.recommendations:
                        recommendations.append({
                            "type": rec.recommendation_type,
                            "title": rec.title,
                            "description": rec.description,
                            "priority": rec.priority,
                            "for_teacher": rec.for_teacher,
                            "for_parent": rec.for_parent,
                        })

                # Get history if requested
                history = []
                if include_history:
                    history_stmt = (
                        select(DiagnosticScan)
                        .where(
                            DiagnosticScan.student_id == student_id,
                            DiagnosticScan.status == "completed",
                            DiagnosticScan.id != scan.id,
                        )
                        .order_by(DiagnosticScan.completed_at.desc())
                        .limit(5)
                    )
                    history_result = await session.execute(history_stmt)
                    for hist_scan in history_result.scalars().all():
                        history.append({
                            "scan_id": hist_scan.id,
                            "date": hist_scan.completed_at.isoformat() if hist_scan.completed_at else None,
                            "risk_score": float(hist_scan.risk_score) if hist_scan.risk_score else 0,
                            "findings_count": hist_scan.findings_count,
                        })

                # Publish report generated event
                event_bus = get_event_bus()
                await event_bus.publish(
                    "diagnostic.report.generated",
                    {
                        "report_id": report_id,
                        "scan_id": scan.id,
                        "student_id": student_id,
                        "tenant_code": tenant_code,
                    },
                )

                logger.info(
                    "Diagnostic report generated: report_id=%s, scan_id=%s",
                    report_id,
                    scan.id,
                )

                report = {
                    "report_id": report_id,
                    "student_id": student_id,
                    "tenant_code": tenant_code,
                    "scan_id": scan.id,
                    "scan_date": scan.completed_at.isoformat() if scan.completed_at else None,
                    "scan_type": scan.scan_type,
                    "overall_risk_score": float(scan.risk_score) if scan.risk_score else 0,
                    "findings_count": scan.findings_count,
                    "has_concerns": scan.has_concerns,
                    "indicators": indicators,
                    "disclaimer": DIAGNOSTIC_DISCLAIMER,
                }

                if include_recommendations:
                    report["recommendations"] = recommendations
                    report["recommendation_count"] = len(recommendations)

                if include_history:
                    report["history"] = history

                return report

        except Exception as e:
            logger.error(
                "Report generation failed: student=%s, error=%s",
                student_id,
                str(e),
                exc_info=True,
            )
            from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER

            return {
                "report_id": report_id,
                "student_id": student_id,
                "status": "failed",
                "error": str(e),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            }

    return run_async(_generate())


@dramatiq.actor(
    queue_name=Queues.DIAGNOSTICS,
    max_retries=1,
    time_limit=60000,  # 1 minute
    priority=Priority.DIAGNOSTIC,
)
def check_diagnostic_thresholds(
    tenant_code: str,
    student_id: str,
    session_accuracy: float | None = None,
    trigger_scan: bool = True,
) -> dict[str, Any]:
    """Check diagnostic thresholds and create alerts if needed.

    Called after practice sessions to check if diagnostic concerns
    should be flagged. Optionally triggers a targeted scan.

    Args:
        tenant_code: Tenant code.
        student_id: Student identifier.
        session_accuracy: Accuracy from the completed session (0-1).
        trigger_scan: Whether to trigger scan on low accuracy.

    Returns:
        Threshold check result with any alerts created.

    Example:
        # After a low-accuracy session
        check_diagnostic_thresholds.send(
            "test_school",
            "student-uuid",
            session_accuracy=0.35,
        )
    """

    async def _check() -> dict[str, Any]:
        from src.core.diagnostics.service import (
            DIAGNOSTIC_DISCLAIMER,
            get_diagnostic_service,
        )
        from src.infrastructure.database.tenant_manager import get_worker_db_manager
        from src.infrastructure.events import get_event_bus

        alerts_created = []
        scan_triggered = False

        try:
            tenant_db = get_worker_db_manager()
            async with tenant_db.get_session(tenant_code) as session:
                service = get_diagnostic_service()

                # Get latest scan results
                latest_scan = await service.get_latest_scan(session, student_id)

                if not latest_scan:
                    # No scan exists - trigger one if low accuracy
                    if trigger_scan and session_accuracy is not None and session_accuracy < 0.4:
                        run_diagnostic_scan.send(
                            tenant_code,
                            student_id,
                            scan_type="full",
                            trigger_reason=f"low_accuracy_{session_accuracy:.0%}",
                        )
                        scan_triggered = True

                    return {
                        "student_id": student_id,
                        "tenant_code": tenant_code,
                        "has_prior_scan": False,
                        "scan_triggered": scan_triggered,
                        "alerts_created": [],
                    }

                # Check thresholds from latest scan
                risk_score = float(latest_scan.risk_score) if latest_scan.risk_score else 0

                event_bus = get_event_bus()

                # High risk threshold (0.7+)
                if risk_score >= 0.7:
                    await event_bus.publish(
                        "diagnostic.high_risk.detected",
                        {
                            "student_id": student_id,
                            "tenant_code": tenant_code,
                            "risk_score": risk_score,
                            "scan_id": latest_scan.id,
                            "findings_count": latest_scan.findings_count,
                        },
                    )
                    alerts_created.append({
                        "type": "high_risk",
                        "severity": "critical",
                        "risk_score": risk_score,
                    })

                # Elevated risk threshold (0.5-0.7)
                elif risk_score >= 0.5:
                    await event_bus.publish(
                        "diagnostic.elevated_risk.detected",
                        {
                            "student_id": student_id,
                            "tenant_code": tenant_code,
                            "risk_score": risk_score,
                            "scan_id": latest_scan.id,
                        },
                    )
                    alerts_created.append({
                        "type": "elevated_risk",
                        "severity": "warning",
                        "risk_score": risk_score,
                    })

                # Check if new scan needed based on session performance
                if trigger_scan and session_accuracy is not None:
                    if session_accuracy < 0.4 and latest_scan.completed_at:
                        age_hours = (
                            datetime.now(timezone.utc) - latest_scan.completed_at
                        ).total_seconds() / 3600

                        if age_hours > 24:  # Scan is older than 24 hours
                            run_diagnostic_scan.send(
                                tenant_code,
                                student_id,
                                scan_type="quick",
                                trigger_reason=f"low_accuracy_{session_accuracy:.0%}",
                            )
                            scan_triggered = True

                return {
                    "student_id": student_id,
                    "tenant_code": tenant_code,
                    "has_prior_scan": True,
                    "last_scan_id": latest_scan.id,
                    "risk_score": risk_score,
                    "scan_triggered": scan_triggered,
                    "alerts_created": alerts_created,
                    "disclaimer": DIAGNOSTIC_DISCLAIMER,
                }

        except Exception as e:
            logger.error(
                "Threshold check failed: student=%s, error=%s",
                student_id,
                str(e),
                exc_info=True,
            )
            return {
                "student_id": student_id,
                "status": "failed",
                "error": str(e),
            }

    return run_async(_check())


def get_diagnostic_actors() -> list:
    """Get all diagnostic actors.

    Returns:
        List of diagnostic actor functions.
    """
    return [
        run_diagnostic_scan,
        run_batch_diagnostic_scans,
        assess_risk_score,
        generate_diagnostic_report,
        check_diagnostic_thresholds,
    ]
