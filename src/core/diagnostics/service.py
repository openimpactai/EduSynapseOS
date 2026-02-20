# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic service for orchestrating learning difficulty detection.

This module provides the main service for running diagnostic scans
that identify potential learning difficulty indicators.

IMPORTANT: This service identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.diagnostics.config import get_diagnostic_config
from src.core.diagnostics.detectors.attention import AttentionDetector
from src.core.diagnostics.detectors.auditory import AuditoryDetector
from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)
from src.core.diagnostics.detectors.dyscalculia import DyscalculiaDetector
from src.core.diagnostics.detectors.dyslexia import DyslexiaDetector
from src.core.diagnostics.detectors.visual import VisualDetector
from src.infrastructure.database.models.tenant.diagnostic import (
    DiagnosticIndicator,
    DiagnosticRecommendation,
    DiagnosticScan,
)
from src.infrastructure.database.models.tenant.memory import (
    EpisodicMemory,
    SemanticMemory,
)
from src.infrastructure.database.models.tenant.practice import (
    EvaluationResult,
    PracticeSession,
    StudentAnswer,
)
from src.utils.datetime import utc_now

logger = logging.getLogger(__name__)


# Standard disclaimer for all diagnostic results
DIAGNOSTIC_DISCLAIMER = (
    "These indicators are for monitoring purposes only and do not constitute "
    "a diagnosis. If patterns persist, please consult a qualified educational "
    "specialist or healthcare professional for proper evaluation."
)


class DiagnosticService:
    """Service for running diagnostic scans on student data.

    This service orchestrates all learning difficulty detectors
    to analyze student data and identify potential indicators.

    Usage:
        service = DiagnosticService()

        # Run full scan
        scan = await service.run_scan(
            db=session,
            student_id="student-uuid",
            trigger_reason="periodic_check",
        )

        # Run targeted scan
        scan = await service.run_targeted_scan(
            db=session,
            student_id="student-uuid",
            indicator_types=[IndicatorType.DYSLEXIA, IndicatorType.DYSCALCULIA],
        )
    """

    # Scan types
    SCAN_TYPE_FULL = "full"
    SCAN_TYPE_TARGETED = "targeted"
    SCAN_TYPE_QUICK = "quick"

    # Risk thresholds
    SIGNIFICANT_RISK_THRESHOLD = 0.5
    HIGH_RISK_THRESHOLD = 0.7

    def __init__(self) -> None:
        """Initialize the diagnostic service with all detectors and config."""
        self._detectors: list[BaseDetector] = [
            DyslexiaDetector(),
            DyscalculiaDetector(),
            AttentionDetector(),
            AuditoryDetector(),
            VisualDetector(),
        ]

        self._detector_map: dict[IndicatorType, BaseDetector] = {
            d.indicator_type: d for d in self._detectors
        }

        # Load configuration from YAML files
        self._config = get_diagnostic_config()

    @property
    def available_detectors(self) -> list[IndicatorType]:
        """Get list of available detector types."""
        return list(self._detector_map.keys())

    async def run_scan(
        self,
        db: AsyncSession,
        student_id: str,
        scan_type: str = SCAN_TYPE_FULL,
        trigger_reason: str | None = None,
    ) -> DiagnosticScan:
        """Run a full diagnostic scan for a student.

        Runs all available detectors and stores results.

        Args:
            db: Database session.
            student_id: Student to analyze.
            scan_type: Type of scan (full, quick).
            trigger_reason: What triggered this scan.

        Returns:
            DiagnosticScan with all results.
        """
        logger.info(
            "Starting diagnostic scan",
            extra={
                "student_id": student_id,
                "scan_type": scan_type,
                "trigger_reason": trigger_reason,
            },
        )

        # Create scan record
        scan = DiagnosticScan(
            student_id=student_id,
            scan_type=scan_type,
            trigger_reason=trigger_reason,
            status="in_progress",
            started_at=utc_now(),
        )
        db.add(scan)
        await db.flush()

        try:
            # Fetch student data
            student_data = await self._fetch_student_data(db, student_id)

            # Run all detectors
            results = await self._run_detectors(
                student_data,
                detectors=self._detectors,
            )

            # Store results
            await self._store_results(db, scan, results)

            # Generate recommendations
            await self._generate_recommendations(db, scan, results)

            # Update scan with summary
            scan.status = "completed"
            scan.completed_at = utc_now()
            scan.findings_count = sum(1 for r in results if r.is_significant)
            scan.risk_score = Decimal(str(self._calculate_overall_risk(results)))

            await db.flush()

            logger.info(
                "Diagnostic scan completed",
                extra={
                    "scan_id": scan.id,
                    "student_id": student_id,
                    "findings_count": scan.findings_count,
                    "risk_score": float(scan.risk_score) if scan.risk_score else 0,
                },
            )

            return scan

        except Exception as e:
            scan.status = "failed"
            scan.completed_at = utc_now()
            await db.flush()

            logger.error(
                "Diagnostic scan failed",
                extra={"scan_id": scan.id, "error": str(e)},
                exc_info=True,
            )
            raise

    async def run_targeted_scan(
        self,
        db: AsyncSession,
        student_id: str,
        indicator_types: list[IndicatorType],
        trigger_reason: str | None = None,
    ) -> DiagnosticScan:
        """Run a targeted scan for specific indicator types.

        Args:
            db: Database session.
            student_id: Student to analyze.
            indicator_types: Specific indicators to check.
            trigger_reason: What triggered this scan.

        Returns:
            DiagnosticScan with targeted results.
        """
        logger.info(
            "Starting targeted diagnostic scan",
            extra={
                "student_id": student_id,
                "indicator_types": [t.value for t in indicator_types],
            },
        )

        # Create scan record
        scan = DiagnosticScan(
            student_id=student_id,
            scan_type=self.SCAN_TYPE_TARGETED,
            trigger_reason=trigger_reason,
            status="in_progress",
            started_at=utc_now(),
        )
        db.add(scan)
        await db.flush()

        try:
            # Fetch student data
            student_data = await self._fetch_student_data(db, student_id)

            # Select relevant detectors
            detectors = [
                self._detector_map[t]
                for t in indicator_types
                if t in self._detector_map
            ]

            # Run selected detectors
            results = await self._run_detectors(student_data, detectors=detectors)

            # Store results
            await self._store_results(db, scan, results)

            # Generate recommendations
            await self._generate_recommendations(db, scan, results)

            # Update scan
            scan.status = "completed"
            scan.completed_at = utc_now()
            scan.findings_count = sum(1 for r in results if r.is_significant)
            scan.risk_score = Decimal(str(self._calculate_overall_risk(results)))

            await db.flush()

            return scan

        except Exception as e:
            scan.status = "failed"
            scan.completed_at = utc_now()
            await db.flush()
            raise

    async def get_latest_scan(
        self,
        db: AsyncSession,
        student_id: str,
    ) -> DiagnosticScan | None:
        """Get the most recent completed scan for a student.

        Args:
            db: Database session.
            student_id: Student ID.

        Returns:
            Latest DiagnosticScan or None.
        """
        result = await db.execute(
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
        return result.scalar_one_or_none()

    async def get_risk_summary(
        self,
        db: AsyncSession,
        student_id: str,
    ) -> dict[str, Any]:
        """Get a summary of risk indicators for a student.

        Args:
            db: Database session.
            student_id: Student ID.

        Returns:
            Risk summary dictionary.
        """
        scan = await self.get_latest_scan(db, student_id)

        if not scan:
            return {
                "has_scan": False,
                "message": "No diagnostic scan available",
            }

        indicators = {
            ind.indicator_type: {
                "risk_score": float(ind.risk_score),
                "threshold_level": ind.threshold_level,
                "is_significant": float(ind.risk_score) >= self.SIGNIFICANT_RISK_THRESHOLD,
            }
            for ind in scan.indicators
        }

        return {
            "has_scan": True,
            "scan_id": scan.id,
            "scan_date": scan.completed_at.isoformat() if scan.completed_at else None,
            "overall_risk": float(scan.risk_score) if scan.risk_score else 0,
            "findings_count": scan.findings_count,
            "has_concerns": scan.has_concerns,
            "indicators": indicators,
            "recommendation_count": len(scan.recommendations),
            "disclaimer": self._config.get_disclaimer("en"),
        }

    async def _fetch_student_data(
        self,
        db: AsyncSession,
        student_id: str,
    ) -> StudentData:
        """Fetch all relevant student data for analysis.

        Args:
            db: Database session.
            student_id: Student ID.

        Returns:
            StudentData with all learning data.
        """
        # Fetch semantic memories
        semantic_result = await db.execute(
            select(SemanticMemory)
            .where(SemanticMemory.student_id == student_id)
            .order_by(SemanticMemory.updated_at.desc())
            .limit(100)
        )
        semantic_memories = list(semantic_result.scalars().all())

        # Fetch episodic memories
        episodic_result = await db.execute(
            select(EpisodicMemory)
            .where(EpisodicMemory.student_id == student_id)
            .order_by(EpisodicMemory.occurred_at.desc())
            .limit(200)
        )
        episodic_memories = list(episodic_result.scalars().all())

        # Fetch practice sessions
        session_result = await db.execute(
            select(PracticeSession)
            .where(PracticeSession.student_id == student_id)
            .order_by(PracticeSession.started_at.desc())
            .limit(50)
        )
        practice_sessions = list(session_result.scalars().all())

        # Fetch student answers
        answer_result = await db.execute(
            select(StudentAnswer)
            .where(StudentAnswer.student_id == student_id)
            .order_by(StudentAnswer.submitted_at.desc())
            .limit(200)
        )
        student_answers = list(answer_result.scalars().all())

        # Fetch evaluation results for these answers
        answer_ids = [a.id for a in student_answers]
        if answer_ids:
            eval_result = await db.execute(
                select(EvaluationResult)
                .where(EvaluationResult.answer_id.in_(answer_ids))
            )
            evaluation_results = list(eval_result.scalars().all())
        else:
            evaluation_results = []

        return StudentData(
            student_id=student_id,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
            practice_sessions=practice_sessions,
            student_answers=student_answers,
            evaluation_results=evaluation_results,
        )

    async def _run_detectors(
        self,
        student_data: StudentData,
        detectors: list[BaseDetector],
    ) -> list[DetectorResult]:
        """Run specified detectors on student data.

        Args:
            student_data: Student data to analyze.
            detectors: List of detectors to run.

        Returns:
            List of detector results.
        """
        results: list[DetectorResult] = []

        for detector in detectors:
            try:
                result = await detector.analyze(student_data)
                results.append(result)

                logger.debug(
                    "Detector completed",
                    extra={
                        "detector": detector.name,
                        "risk_score": result.risk_score,
                        "threshold": result.threshold_level.value,
                    },
                )

            except Exception as e:
                logger.error(
                    "Detector failed",
                    extra={
                        "detector": detector.name,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                # Add a no-data result for failed detectors
                results.append(DetectorResult.no_data(detector.indicator_type))

        return results

    async def _store_results(
        self,
        db: AsyncSession,
        scan: DiagnosticScan,
        results: list[DetectorResult],
    ) -> None:
        """Store detector results as DiagnosticIndicator records.

        Args:
            db: Database session.
            scan: Parent scan record.
            results: Detector results to store.
        """
        for result in results:
            indicator = DiagnosticIndicator(
                scan_id=scan.id,
                indicator_type=result.indicator_type.value,
                risk_score=Decimal(str(round(result.risk_score, 2))),
                confidence=Decimal(str(round(result.confidence, 2))),
                evidence=[e.to_dict() for e in result.evidence],
                threshold_level=result.threshold_level.value,
            )
            db.add(indicator)

        await db.flush()

    async def _generate_recommendations(
        self,
        db: AsyncSession,
        scan: DiagnosticScan,
        results: list[DetectorResult],
    ) -> None:
        """Generate recommendations based on detector results.

        Args:
            db: Database session.
            scan: Parent scan record.
            results: Detector results.
        """
        significant_results = [r for r in results if r.is_significant]

        if not significant_results:
            return

        # Get indicator IDs for linking
        indicator_result = await db.execute(
            select(DiagnosticIndicator)
            .where(DiagnosticIndicator.scan_id == scan.id)
        )
        indicators = {
            ind.indicator_type: ind
            for ind in indicator_result.scalars().all()
        }

        for result in significant_results:
            indicator = indicators.get(result.indicator_type.value)

            # Generate recommendations based on indicator type and severity
            recommendations = self._get_recommendations_for_result(result)

            for rec in recommendations:
                db_rec = DiagnosticRecommendation(
                    scan_id=scan.id,
                    indicator_id=indicator.id if indicator else None,
                    recommendation_type=rec["type"],
                    title=rec["title"],
                    description=rec["description"],
                    priority=rec["priority"],
                    for_teacher=rec.get("for_teacher", True),
                    for_parent=rec.get("for_parent", False),
                    disclaimer=DIAGNOSTIC_DISCLAIMER,
                )
                db.add(db_rec)

        await db.flush()

    def _get_recommendations_for_result(
        self,
        result: DetectorResult,
        lang: str = "en",
    ) -> list[dict[str, Any]]:
        """Get recommendations for a specific detector result from config.

        Reads recommendations from YAML configuration based on indicator type
        and severity level.

        Args:
            result: Detector result.
            lang: Language code for recommendations (en, tr).

        Returns:
            List of recommendation dictionaries.
        """
        recommendations: list[dict[str, Any]] = []
        indicator_type = result.indicator_type.value  # e.g., "dyslexia"

        # Determine level based on config thresholds
        detector_config = self._config.get_detector_config(indicator_type)
        if detector_config:
            if result.risk_score >= detector_config.alert_threshold:
                level = "alert"
            elif result.risk_score >= detector_config.concern_threshold:
                level = "concern"
            else:
                level = None
        else:
            # Fallback to default thresholds
            if result.threshold_level == ThresholdLevel.HIGH:
                level = "alert"
            elif result.threshold_level in [ThresholdLevel.ELEVATED, ThresholdLevel.MEDIUM]:
                level = "concern"
            else:
                level = None

        if not level:
            return recommendations

        is_alert = level == "alert"

        # Get teacher recommendations from config
        teacher_recs = self._config.get_recommendations(
            indicator_type, level, "for_teacher", lang
        )
        for idx, rec_text in enumerate(teacher_recs):
            recommendations.append({
                "type": "accommodation" if not is_alert else "referral",
                "title": f"{detector_config.get_name(lang) if detector_config else indicator_type.title()} - Teacher Support",
                "description": rec_text,
                "priority": 3 if is_alert else 2,
                "for_teacher": True,
                "for_parent": False,
            })

        # Get parent recommendations from config
        parent_recs = self._config.get_recommendations(
            indicator_type, level, "for_parent", lang
        )
        for idx, rec_text in enumerate(parent_recs):
            recommendations.append({
                "type": "accommodation" if not is_alert else "referral",
                "title": f"{detector_config.get_name(lang) if detector_config else indicator_type.title()} - Parent Support",
                "description": rec_text,
                "priority": 3 if is_alert else 2,
                "for_teacher": False,
                "for_parent": True,
            })

        # Add general monitoring recommendation for all elevated results
        recommendations.append({
            "type": "monitoring",
            "title": "Continued Monitoring",
            "description": (
                f"Continue monitoring {indicator_type.replace('_', ' ')} "
                f"indicators. Current risk level: {result.threshold_level.value}. "
                "Regular check-ins can help track progress and adjust support strategies."
            ),
            "priority": 1,
            "for_teacher": True,
            "for_parent": False,
        })

        return recommendations

    def get_disclaimer(self, lang: str = "en") -> str:
        """Get the professional disclaimer for reports.

        Args:
            lang: Language code (tr, en).

        Returns:
            Disclaimer text from config.
        """
        return self._config.get_disclaimer(lang)

    def get_recommendations_for_indicator(
        self,
        indicator_type: str,
        risk_score: float,
        lang: str = "en",
    ) -> dict[str, list[str]]:
        """Get recommendations for an indicator type and score.

        Args:
            indicator_type: Type of indicator.
            risk_score: Current risk score.
            lang: Language code.

        Returns:
            Dict with "for_teacher" and "for_parent" recommendation lists.
        """
        # Determine level
        detector_config = self._config.get_detector_config(indicator_type)
        if detector_config:
            if risk_score >= detector_config.alert_threshold:
                level = "alert"
            elif risk_score >= detector_config.concern_threshold:
                level = "concern"
            else:
                level = None
        else:
            level = "concern" if risk_score >= 0.4 else None

        if not level:
            return {"for_teacher": [], "for_parent": []}

        return {
            "for_teacher": self._config.get_recommendations(
                indicator_type, level, "for_teacher", lang
            ),
            "for_parent": self._config.get_recommendations(
                indicator_type, level, "for_parent", lang
            ),
        }

    def should_notify_parent(self, risk_score: float) -> bool:
        """Check if parent should be notified based on config.

        Args:
            risk_score: Current risk score.

        Returns:
            True if parent notification threshold is exceeded.
        """
        notif_config = self._config.get_notification_config("parent")
        if notif_config:
            return risk_score >= notif_config.threshold
        return risk_score >= 0.5

    def should_refer_professional(self, risk_score: float) -> bool:
        """Check if professional referral is needed based on config.

        Args:
            risk_score: Current risk score.

        Returns:
            True if professional referral threshold is exceeded.
        """
        notif_config = self._config.get_notification_config("professional_referral")
        if notif_config:
            return risk_score >= notif_config.threshold
        return risk_score >= 0.7

    def _calculate_overall_risk(
        self,
        results: list[DetectorResult],
    ) -> float:
        """Calculate overall risk score from all detector results.

        Uses weighted average with emphasis on significant findings.

        Args:
            results: All detector results.

        Returns:
            Overall risk score (0.0-1.0).
        """
        if not results:
            return 0.0

        # Weight significant results more heavily
        weighted_sum = 0.0
        weight_total = 0.0

        for result in results:
            weight = 2.0 if result.is_significant else 1.0
            weighted_sum += result.risk_score * weight
            weight_total += weight

        if weight_total == 0:
            return 0.0

        return round(weighted_sum / weight_total, 2)


# Singleton instance
_service_instance: DiagnosticService | None = None


def get_diagnostic_service() -> DiagnosticService:
    """Get the diagnostic service singleton.

    Returns:
        DiagnosticService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = DiagnosticService()
    return _service_instance
