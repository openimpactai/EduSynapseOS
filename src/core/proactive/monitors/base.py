# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base monitor classes and shared types for proactive intelligence.

This module provides the abstract base class for all proactive monitors
and shared data structures they use. Monitors detect patterns in student
behavior and generate alerts for stakeholders (teachers, parents).

Threshold values are shared with the diagnostic system for consistency:
- ELEVATED_RISK_THRESHOLD = 0.5 (same as DiagnosticIndicatorScores)
- HIGH_RISK_THRESHOLD = 0.7 (same as DiagnosticIndicatorScores)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from src.models.memory import DiagnosticContext, FullMemoryContext


# Shared thresholds - SAME as DiagnosticIndicatorScores and DiagnosticService
ELEVATED_RISK_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7


class AlertType(str, Enum):
    """Types of proactive alerts."""

    STRUGGLE_DETECTED = "struggle_detected"
    ENGAGEMENT_DROP = "engagement_drop"
    MILESTONE_ACHIEVED = "milestone_achieved"
    INACTIVITY_WARNING = "inactivity_warning"
    DIAGNOSTIC_ALERT = "diagnostic_alert"
    EMOTIONAL_DISTRESS = "emotional_distress"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertTarget(str, Enum):
    """Who should receive the alert."""

    STUDENT = "student"
    TEACHER = "teacher"
    PARENT = "parent"
    SYSTEM = "system"


@dataclass
class AlertData:
    """Data structure for a proactive alert.

    Attributes:
        alert_type: Type of alert.
        severity: Severity level.
        title: Alert title (human-readable).
        message: Detailed alert message.
        student_id: Student this alert is about.
        targets: Who should receive this alert.
        topic_codes: Related topic composite key (optional).
        session_id: Related session (optional).
        details: Additional alert details.
        suggested_actions: List of recommended actions.
        diagnostic_context: Diagnostic data if available.
    """

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    student_id: UUID
    targets: list[AlertTarget] = field(default_factory=list)
    topic_codes: dict[str, str] | None = None
    session_id: UUID | None = None
    details: dict[str, Any] = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)
    diagnostic_context: DiagnosticContext | None = None

    @property
    def topic_full_code(self) -> str | None:
        """Get the full topic code if all parts are present."""
        if self.topic_codes and all(k in self.topic_codes for k in ["framework_code", "subject_code", "grade_code", "unit_code", "code"]):
            return f"{self.topic_codes['framework_code']}.{self.topic_codes['subject_code']}.{self.topic_codes['grade_code']}.{self.topic_codes['unit_code']}.{self.topic_codes['code']}"
        return None

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage.

        Returns:
            Dictionary matching Alert model fields.
        """
        return {
            "student_id": str(self.student_id),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "details": {
                **self.details,
                "targets": [t.value for t in self.targets],
                "diagnostic_context": (
                    self.diagnostic_context.model_dump()
                    if self.diagnostic_context
                    else None
                ),
            },
            "topic_framework_code": self.topic_codes.get("framework_code") if self.topic_codes else None,
            "topic_subject_code": self.topic_codes.get("subject_code") if self.topic_codes else None,
            "topic_grade_code": self.topic_codes.get("grade_code") if self.topic_codes else None,
            "topic_unit_code": self.topic_codes.get("unit_code") if self.topic_codes else None,
            "topic_code": self.topic_codes.get("code") if self.topic_codes else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "suggested_actions": self.suggested_actions,
            "status": "active",
        }


class BaseMonitor(ABC):
    """Abstract base class for proactive monitors.

    Each monitor analyzes student context to detect specific patterns
    and generate appropriate alerts. Monitors are diagnostic-aware,
    meaning they can adjust their thresholds and recommendations
    based on diagnostic indicator scores.

    Diagnostic integration:
    - Monitors access diagnostic context via FullMemoryContext.diagnostic
    - Uses shared thresholds: ELEVATED_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
    - Adjusts severity and messaging based on diagnostic indicators
    """

    def __init__(self) -> None:
        """Initialize the monitor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the monitor name."""
        ...

    @property
    @abstractmethod
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        ...

    @abstractmethod
    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if an alert should be generated.

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if alert should be generated, None otherwise.
        """
        ...

    def has_diagnostic_context(self, context: FullMemoryContext) -> bool:
        """Check if diagnostic context is available.

        Args:
            context: Full memory context.

        Returns:
            True if diagnostic data exists.
        """
        return context.diagnostic is not None

    def has_elevated_risk(self, context: FullMemoryContext) -> bool:
        """Check if any diagnostic indicator is at elevated level.

        Args:
            context: Full memory context.

        Returns:
            True if any indicator >= ELEVATED_RISK_THRESHOLD.
        """
        if not context.diagnostic:
            return False
        return context.diagnostic.max_risk >= ELEVATED_RISK_THRESHOLD

    def has_high_risk(self, context: FullMemoryContext) -> bool:
        """Check if any diagnostic indicator is at high level.

        Args:
            context: Full memory context.

        Returns:
            True if any indicator >= HIGH_RISK_THRESHOLD.
        """
        if not context.diagnostic:
            return False
        return context.diagnostic.max_risk >= HIGH_RISK_THRESHOLD

    def get_risk_level_label(self, context: FullMemoryContext) -> str:
        """Get human-readable risk level label.

        Args:
            context: Full memory context.

        Returns:
            Risk level label string.
        """
        if not context.diagnostic:
            return "unknown"
        if self.has_high_risk(context):
            return "high"
        if self.has_elevated_risk(context):
            return "elevated"
        return "normal"

    def get_elevated_indicators(
        self, context: FullMemoryContext
    ) -> list[tuple[str, float]]:
        """Get list of elevated diagnostic indicators.

        Args:
            context: Full memory context.

        Returns:
            List of (indicator_name, score) tuples for elevated indicators.
        """
        if not context.diagnostic:
            return []

        diag = context.diagnostic
        elevated = []

        if diag.dyslexia_risk >= ELEVATED_RISK_THRESHOLD:
            elevated.append(("dyslexia", diag.dyslexia_risk))
        if diag.dyscalculia_risk >= ELEVATED_RISK_THRESHOLD:
            elevated.append(("dyscalculia", diag.dyscalculia_risk))
        if diag.attention_risk >= ELEVATED_RISK_THRESHOLD:
            elevated.append(("attention", diag.attention_risk))
        if diag.auditory_risk >= ELEVATED_RISK_THRESHOLD:
            elevated.append(("auditory", diag.auditory_risk))
        if diag.visual_risk >= ELEVATED_RISK_THRESHOLD:
            elevated.append(("visual", diag.visual_risk))

        return sorted(elevated, key=lambda x: x[1], reverse=True)

    def create_alert(
        self,
        student_id: UUID,
        severity: AlertSeverity,
        title: str,
        message: str,
        targets: list[AlertTarget] | None = None,
        topic_codes: dict[str, str] | None = None,
        session_id: UUID | None = None,
        details: dict[str, Any] | None = None,
        suggested_actions: list[str] | None = None,
        diagnostic_context: DiagnosticContext | None = None,
    ) -> AlertData:
        """Helper method to create an alert.

        Args:
            student_id: Student ID.
            severity: Alert severity.
            title: Alert title.
            message: Alert message.
            targets: Alert recipients.
            topic_codes: Related topic composite key dict.
            session_id: Related session.
            details: Additional details.
            suggested_actions: Recommended actions.
            diagnostic_context: Diagnostic data.

        Returns:
            AlertData instance.
        """
        return AlertData(
            alert_type=self.alert_type,
            severity=severity,
            title=title,
            message=message,
            student_id=student_id,
            targets=targets or [AlertTarget.TEACHER],
            topic_codes=topic_codes,
            session_id=session_id,
            details=details or {},
            suggested_actions=suggested_actions or [],
            diagnostic_context=diagnostic_context,
        )
