# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Learning difficulty detector implementations.

This package provides detectors for identifying potential learning
difficulty indicators in student performance data.

Available Detectors:
    - DyslexiaDetector: Identifies reading/writing difficulty indicators
    - DyscalculiaDetector: Identifies math difficulty indicators
    - AttentionDetector: Identifies attention-related indicators
    - AuditoryDetector: Identifies auditory processing indicators
    - VisualDetector: Identifies visual processing indicators

Base Classes:
    - BaseDetector: Abstract base class for all detectors
    - DetectorResult: Result container for detector analysis
    - StudentData: Aggregated student data for analysis
    - Evidence: Evidence supporting detected indicators

IMPORTANT: All detectors identify INDICATORS only, not diagnoses.
Professional evaluation by qualified specialists is required for diagnosis.
"""

from src.core.diagnostics.detectors.attention import AttentionDetector
from src.core.diagnostics.detectors.auditory import AuditoryDetector
from src.core.diagnostics.detectors.base import (
    BaseDetector,
    DetectorResult,
    Evidence,
    IndicatorType,
    StudentData,
    ThresholdLevel,
)
from src.core.diagnostics.detectors.dyscalculia import DyscalculiaDetector
from src.core.diagnostics.detectors.dyslexia import DyslexiaDetector
from src.core.diagnostics.detectors.visual import VisualDetector

__all__ = [
    # Base classes
    "BaseDetector",
    "DetectorResult",
    "Evidence",
    "IndicatorType",
    "StudentData",
    "ThresholdLevel",
    # Detector implementations
    "AttentionDetector",
    "AuditoryDetector",
    "DyscalculiaDetector",
    "DyslexiaDetector",
    "VisualDetector",
]
