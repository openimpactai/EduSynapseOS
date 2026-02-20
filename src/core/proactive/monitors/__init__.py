# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Proactive monitors for detecting student patterns.

This package provides monitors that analyze student behavior
and generate alerts for stakeholders. All monitors are
diagnostic-aware and integrate with the Phase 16 diagnostic system.

Monitors:
- StruggleMonitor: Detects consecutive errors and low accuracy
- EngagementMonitor: Detects declining engagement patterns
- MilestoneMonitor: Celebrates achievements and progress
- InactivityMonitor: Detects prolonged absence
- DiagnosticMonitor: Alerts on diagnostic findings
- EmotionalDistressMonitor: Detects sustained emotional distress

Usage:
    from src.core.proactive.monitors import (
        StruggleMonitor,
        EngagementMonitor,
        MilestoneMonitor,
        InactivityMonitor,
        DiagnosticMonitor,
        EmotionalDistressMonitor,
    )

    # Create a monitor
    struggle_monitor = StruggleMonitor()

    # Check for alerts
    alert = await struggle_monitor.check(context, tenant_code)
"""

from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
    ELEVATED_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)
from src.core.proactive.monitors.diagnostic import DiagnosticMonitor
from src.core.proactive.monitors.emotional import EmotionalDistressMonitor
from src.core.proactive.monitors.engagement import EngagementMonitor
from src.core.proactive.monitors.inactivity import InactivityMonitor
from src.core.proactive.monitors.milestone import MilestoneMonitor
from src.core.proactive.monitors.struggle import StruggleMonitor

__all__ = [
    # Base types
    "AlertData",
    "AlertSeverity",
    "AlertTarget",
    "AlertType",
    "BaseMonitor",
    "ELEVATED_RISK_THRESHOLD",
    "HIGH_RISK_THRESHOLD",
    # Monitors
    "DiagnosticMonitor",
    "EmotionalDistressMonitor",
    "EngagementMonitor",
    "InactivityMonitor",
    "MilestoneMonitor",
    "StruggleMonitor",
]
