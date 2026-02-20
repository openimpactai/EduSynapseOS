# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic system for learning difficulty detection.

This package provides a comprehensive system for identifying potential
learning difficulty indicators in student performance data.

Architecture:
    The diagnostic system consists of:
    1. Detectors: Rule-based analyzers for specific indicator types
    2. DiagnosticService: Orchestrator that runs scans and stores results
    3. Database Models: DiagnosticScan, DiagnosticIndicator, DiagnosticRecommendation

Quick Start:
    from src.core.diagnostics import get_diagnostic_service, IndicatorType

    service = get_diagnostic_service()

    # Run full scan
    scan = await service.run_scan(
        db=session,
        student_id="student-uuid",
        trigger_reason="periodic_check",
    )

    # Check results
    print(f"Risk score: {scan.risk_score}")
    print(f"Findings: {scan.findings_count}")

    # Get risk summary
    summary = await service.get_risk_summary(db, student_id)

Available Indicator Types:
    - DYSLEXIA: Reading/writing difficulty indicators
    - DYSCALCULIA: Math difficulty indicators
    - ATTENTION: Attention-related indicators
    - AUDITORY: Auditory processing indicators
    - VISUAL: Visual processing indicators

IMPORTANT: This system identifies INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
All results include appropriate disclaimers.
"""

from src.core.diagnostics.config import (
    DetectorConfig,
    DiagnosticConfig,
    GlobalSettings,
    get_diagnostic_config,
    reload_diagnostic_config,
)
from src.core.diagnostics.detectors import (
    AttentionDetector,
    AuditoryDetector,
    BaseDetector,
    DetectorResult,
    DyscalculiaDetector,
    DyslexiaDetector,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
    VisualDetector,
)
from src.core.diagnostics.service import (
    DIAGNOSTIC_DISCLAIMER,
    DiagnosticService,
    get_diagnostic_service,
)

__all__ = [
    # Service
    "DiagnosticService",
    "get_diagnostic_service",
    "DIAGNOSTIC_DISCLAIMER",
    # Config
    "DiagnosticConfig",
    "DetectorConfig",
    "GlobalSettings",
    "get_diagnostic_config",
    "reload_diagnostic_config",
    # Types
    "IndicatorType",
    "ThresholdLevel",
    # Base classes
    "BaseDetector",
    "DetectorResult",
    "Evidence",
    "StudentData",
    # Detector implementations
    "AttentionDetector",
    "AuditoryDetector",
    "DyscalculiaDetector",
    "DyslexiaDetector",
    "VisualDetector",
]
