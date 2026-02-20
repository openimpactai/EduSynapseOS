# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Diagnostic monitor for alerting on learning difficulty indicators.

This monitor uses results from the DiagnosticService (Phase 16) to
generate alerts when significant learning difficulty indicators are
detected. It bridges the diagnostic system with the proactive
intelligence system.

Integration with DiagnosticService:
- Uses same threshold levels: ELEVATED (>= 0.5), HIGH (>= 0.7)
- Respects DIAGNOSTIC_DISCLAIMER in all alerts
- References DiagnosticRecommendations for suggested actions

IMPORTANT: This monitor creates awareness alerts only. It does NOT
diagnose learning difficulties - it alerts stakeholders to patterns
that may warrant professional evaluation.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from src.core.diagnostics.service import DIAGNOSTIC_DISCLAIMER
from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
    ELEVATED_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)
from src.models.memory import DiagnosticContext, FullMemoryContext

logger = logging.getLogger(__name__)


class DiagnosticMonitor(BaseMonitor):
    """Monitor that creates alerts based on diagnostic findings.

    Analyzes diagnostic context from FullMemoryContext to determine
    if stakeholders should be alerted about learning difficulty
    indicators. Uses the same thresholds as DiagnosticService.

    Alert triggers:
    - HIGH risk (>= 0.7): Critical alert to teacher and parent
    - ELEVATED risk (>= 0.5): Warning alert to teacher
    - Multiple elevated indicators: Combined concern alert

    All alerts include DIAGNOSTIC_DISCLAIMER emphasizing that
    professional evaluation is needed for actual diagnosis.
    """

    # Configuration constants (same as DiagnosticService)
    ELEVATED_THRESHOLD = ELEVATED_RISK_THRESHOLD  # 0.5
    HIGH_THRESHOLD = HIGH_RISK_THRESHOLD  # 0.7

    @property
    def name(self) -> str:
        """Return the monitor name."""
        return "diagnostic_monitor"

    @property
    def alert_type(self) -> AlertType:
        """Return the type of alert this monitor generates."""
        return AlertType.DIAGNOSTIC_ALERT

    async def check(
        self,
        context: FullMemoryContext,
        tenant_code: str,
    ) -> AlertData | None:
        """Check if diagnostic findings warrant an alert.

        Checks for:
        1. HIGH risk indicators (>= 0.7) - always alert
        2. ELEVATED risk indicators (>= 0.5) - alert with recommendations
        3. Multiple elevated indicators - combined concern alert

        Args:
            context: Full memory context for the student.
            tenant_code: Tenant identifier.

        Returns:
            AlertData if diagnostic alert warranted, None otherwise.
        """
        if not context.diagnostic:
            logger.debug(
                "No diagnostic context for student %s", context.student_id
            )
            return None

        diag = context.diagnostic

        # Get all elevated indicators
        elevated_indicators = self._get_all_elevated_indicators(diag)

        if not elevated_indicators:
            # No concerning indicators
            return None

        # Check for HIGH risk indicators
        high_indicators = [
            (name, score) for name, score in elevated_indicators
            if score >= self.HIGH_THRESHOLD
        ]

        if high_indicators:
            return self._create_high_risk_alert(
                context=context,
                high_indicators=high_indicators,
                all_elevated=elevated_indicators,
            )

        # Check for multiple elevated indicators (comorbidity pattern)
        if len(elevated_indicators) >= 2:
            return self._create_multiple_elevated_alert(
                context=context,
                elevated_indicators=elevated_indicators,
            )

        # Single elevated indicator
        return self._create_elevated_alert(
            context=context,
            indicator=elevated_indicators[0],
        )

    def _get_all_elevated_indicators(
        self, diag: DiagnosticContext
    ) -> list[tuple[str, float]]:
        """Get all indicators at elevated level or above.

        Args:
            diag: Diagnostic context.

        Returns:
            List of (indicator_name, score) tuples, sorted by score descending.
        """
        indicators = []

        if diag.dyslexia_risk >= self.ELEVATED_THRESHOLD:
            indicators.append(("dyslexia", diag.dyslexia_risk))

        if diag.dyscalculia_risk >= self.ELEVATED_THRESHOLD:
            indicators.append(("dyscalculia", diag.dyscalculia_risk))

        if diag.attention_risk >= self.ELEVATED_THRESHOLD:
            indicators.append(("attention", diag.attention_risk))

        if diag.auditory_risk >= self.ELEVATED_THRESHOLD:
            indicators.append(("auditory", diag.auditory_risk))

        if diag.visual_risk >= self.ELEVATED_THRESHOLD:
            indicators.append(("visual", diag.visual_risk))

        return sorted(indicators, key=lambda x: x[1], reverse=True)

    def _create_high_risk_alert(
        self,
        context: FullMemoryContext,
        high_indicators: list[tuple[str, float]],
        all_elevated: list[tuple[str, float]],
    ) -> AlertData:
        """Create alert for HIGH risk indicators.

        Args:
            context: Full memory context.
            high_indicators: Indicators at HIGH risk level.
            all_elevated: All elevated indicators.

        Returns:
            AlertData for high risk.
        """
        # Build indicator summary
        high_names = [self._format_indicator_name(name) for name, _ in high_indicators]
        primary_indicator = high_indicators[0]

        title = f"High Risk: {', '.join(high_names)} Indicator(s) Detected"

        message = self._build_high_risk_message(
            context, high_indicators, all_elevated
        )

        suggested_actions = self._build_high_risk_actions(high_indicators)

        return self.create_alert(
            student_id=context.student_id,
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            targets=[AlertTarget.TEACHER, AlertTarget.PARENT],
            details={
                "high_risk_indicators": [
                    {"name": name, "score": score}
                    for name, score in high_indicators
                ],
                "all_elevated_indicators": [
                    {"name": name, "score": score}
                    for name, score in all_elevated
                ],
                "scan_date": (
                    context.diagnostic.last_scan_at.isoformat()
                    if context.diagnostic and context.diagnostic.last_scan_at
                    else None
                ),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _create_multiple_elevated_alert(
        self,
        context: FullMemoryContext,
        elevated_indicators: list[tuple[str, float]],
    ) -> AlertData:
        """Create alert for multiple elevated indicators.

        Args:
            context: Full memory context.
            elevated_indicators: All elevated indicators.

        Returns:
            AlertData for multiple elevated.
        """
        indicator_names = [
            self._format_indicator_name(name)
            for name, _ in elevated_indicators
        ]

        title = f"Multiple Learning Indicators: {', '.join(indicator_names)}"

        message = self._build_multiple_elevated_message(context, elevated_indicators)

        suggested_actions = self._build_multiple_elevated_actions(elevated_indicators)

        return self.create_alert(
            student_id=context.student_id,
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            targets=[AlertTarget.TEACHER, AlertTarget.PARENT],
            details={
                "elevated_indicators": [
                    {"name": name, "score": score}
                    for name, score in elevated_indicators
                ],
                "indicator_count": len(elevated_indicators),
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _create_elevated_alert(
        self,
        context: FullMemoryContext,
        indicator: tuple[str, float],
    ) -> AlertData:
        """Create alert for single elevated indicator.

        Args:
            context: Full memory context.
            indicator: The elevated indicator (name, score).

        Returns:
            AlertData for single elevated.
        """
        name, score = indicator
        formatted_name = self._format_indicator_name(name)

        title = f"Learning Pattern: {formatted_name} Indicator Elevated"

        message = self._build_elevated_message(context, name, score)

        suggested_actions = self._get_indicator_specific_actions(name, score)

        return self.create_alert(
            student_id=context.student_id,
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            targets=[AlertTarget.TEACHER],
            details={
                "indicator_name": name,
                "indicator_score": score,
                "threshold_level": "elevated",
                "disclaimer": DIAGNOSTIC_DISCLAIMER,
            },
            suggested_actions=suggested_actions,
            diagnostic_context=context.diagnostic,
        )

    def _format_indicator_name(self, name: str) -> str:
        """Format indicator name for display.

        Args:
            name: Raw indicator name.

        Returns:
            Formatted display name.
        """
        display_names = {
            "dyslexia": "Reading/Writing",
            "dyscalculia": "Math Processing",
            "attention": "Attention",
            "auditory": "Auditory Processing",
            "visual": "Visual Processing",
        }
        return display_names.get(name, name.title())

    def _build_high_risk_message(
        self,
        context: FullMemoryContext,
        high_indicators: list[tuple[str, float]],
        all_elevated: list[tuple[str, float]],
    ) -> str:
        """Build message for high risk alert.

        Args:
            context: Full memory context.
            high_indicators: HIGH risk indicators.
            all_elevated: All elevated indicators.

        Returns:
            Alert message with disclaimer.
        """
        high_names = ", ".join(
            f"{self._format_indicator_name(name)} ({score:.0%})"
            for name, score in high_indicators
        )

        message = (
            f"The diagnostic system has detected HIGH risk indicators for: {high_names}. "
            "These patterns suggest the student may benefit from professional evaluation "
            "and targeted support strategies."
        )

        # Add other elevated indicators if any
        other_elevated = [
            (name, score) for name, score in all_elevated
            if score < self.HIGH_THRESHOLD
        ]
        if other_elevated:
            other_names = ", ".join(
                self._format_indicator_name(name)
                for name, _ in other_elevated
            )
            message += f"\n\nAdditional elevated indicators: {other_names}."

        # Add disclaimer
        message += f"\n\n⚠️ IMPORTANT: {DIAGNOSTIC_DISCLAIMER}"

        return message

    def _build_multiple_elevated_message(
        self,
        context: FullMemoryContext,
        elevated_indicators: list[tuple[str, float]],
    ) -> str:
        """Build message for multiple elevated indicators.

        Args:
            context: Full memory context.
            elevated_indicators: Elevated indicators.

        Returns:
            Alert message with disclaimer.
        """
        indicator_list = ", ".join(
            f"{self._format_indicator_name(name)} ({score:.0%})"
            for name, score in elevated_indicators
        )

        message = (
            f"The student shows elevated indicators across multiple areas: {indicator_list}. "
            "When multiple indicators are present, a comprehensive evaluation may be "
            "particularly beneficial to understand the student's learning profile and "
            "develop appropriate support strategies."
        )

        message += f"\n\n⚠️ IMPORTANT: {DIAGNOSTIC_DISCLAIMER}"

        return message

    def _build_elevated_message(
        self,
        context: FullMemoryContext,
        name: str,
        score: float,
    ) -> str:
        """Build message for single elevated indicator.

        Args:
            context: Full memory context.
            name: Indicator name.
            score: Indicator score.

        Returns:
            Alert message with disclaimer.
        """
        formatted_name = self._format_indicator_name(name)

        message = (
            f"The diagnostic system has detected an elevated {formatted_name} indicator "
            f"(score: {score:.0%}). This suggests the student may benefit from specific "
            "accommodations and support strategies in this area."
        )

        # Add indicator-specific context
        context_notes = self._get_indicator_context(name)
        if context_notes:
            message += f"\n\n{context_notes}"

        message += f"\n\n⚠️ {DIAGNOSTIC_DISCLAIMER}"

        return message

    def _get_indicator_context(self, name: str) -> str:
        """Get contextual information for an indicator.

        Args:
            name: Indicator name.

        Returns:
            Contextual information string.
        """
        context_map = {
            "dyslexia": (
                "Reading and writing difficulties may manifest as slower reading pace, "
                "letter reversals, spelling challenges, or reluctance to read aloud."
            ),
            "dyscalculia": (
                "Math processing difficulties may include challenges with number sense, "
                "sequencing, memorizing math facts, or understanding mathematical concepts."
            ),
            "attention": (
                "Attention patterns may show difficulty sustaining focus, inconsistent "
                "performance, or challenges with task completion."
            ),
            "auditory": (
                "Auditory processing patterns may indicate difficulty following verbal "
                "instructions or processing spoken information."
            ),
            "visual": (
                "Visual processing patterns may suggest difficulty with visual-spatial "
                "tasks, reading visual displays, or processing visual information."
            ),
        }
        return context_map.get(name, "")

    def _build_high_risk_actions(
        self, high_indicators: list[tuple[str, float]]
    ) -> list[str]:
        """Build suggested actions for high risk.

        Args:
            high_indicators: HIGH risk indicators.

        Returns:
            List of suggested actions.
        """
        actions = [
            "Schedule a meeting with parents/guardians to discuss observations",
            "Consider referral to appropriate specialist for evaluation",
            "Document specific patterns and examples for professional consultation",
            "Implement immediate accommodations while awaiting evaluation",
        ]

        # Add indicator-specific actions
        for name, _ in high_indicators:
            specific_actions = self._get_indicator_specific_actions(name, 0.7)
            for action in specific_actions[:2]:  # Add top 2 per indicator
                if action not in actions:
                    actions.append(action)

        return actions

    def _build_multiple_elevated_actions(
        self, elevated_indicators: list[tuple[str, float]]
    ) -> list[str]:
        """Build suggested actions for multiple elevated indicators.

        Args:
            elevated_indicators: Elevated indicators.

        Returns:
            List of suggested actions.
        """
        actions = [
            "Consider comprehensive learning evaluation",
            "Implement multi-sensory teaching approaches",
            "Monitor progress across all affected areas",
            "Discuss patterns with parents/guardians",
        ]

        # Add indicator-specific accommodations
        for name, score in elevated_indicators[:3]:  # Top 3
            specific_actions = self._get_indicator_specific_actions(name, score)
            if specific_actions:
                actions.append(specific_actions[0])

        return actions

    def _get_indicator_specific_actions(
        self, name: str, score: float
    ) -> list[str]:
        """Get actions specific to an indicator.

        Args:
            name: Indicator name.
            score: Indicator score.

        Returns:
            List of indicator-specific actions.
        """
        actions_map = {
            "dyslexia": [
                "Provide audio versions of text materials",
                "Use larger fonts and increased line spacing",
                "Allow extended time for reading and writing tasks",
                "Consider text-to-speech and speech-to-text tools",
            ],
            "dyscalculia": [
                "Use visual and manipulative math aids",
                "Break multi-step problems into smaller parts",
                "Allow use of number lines and reference charts",
                "Focus on conceptual understanding before procedures",
            ],
            "attention": [
                "Break tasks into shorter, focused segments",
                "Provide frequent breaks and movement opportunities",
                "Use timers and visual schedules",
                "Minimize distractions in learning environment",
            ],
            "auditory": [
                "Provide written instructions alongside verbal",
                "Use visual cues and demonstrations",
                "Allow extra processing time for verbal information",
                "Minimize background noise during instruction",
            ],
            "visual": [
                "Provide verbal descriptions of visual content",
                "Use audio alternatives where possible",
                "Allow verbal responses to visual questions",
                "Use high-contrast and uncluttered visual materials",
            ],
        }

        return actions_map.get(name, [
            "Consider appropriate accommodations for this area",
            "Monitor and document specific patterns",
        ])
