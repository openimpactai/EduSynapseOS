# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Proactive Intelligence System for EduSynapseOS.

This package provides proactive monitoring capabilities that detect
patterns in student behavior and generate alerts for stakeholders.
It integrates with the diagnostic system (Phase 16) and educational
theories (Phase 7, 16.5) to provide context-aware alerting.

Key Components:
- ProactiveService: Main orchestrator for all monitors
- Monitors: Specialized pattern detectors (struggle, engagement, etc.)
- AlertData: Structured alert information for storage and notification

Integration Points:
- MemoryManager: For student context retrieval
- DiagnosticService: For diagnostic indicator scores
- Alert model: For database storage
- Notification system (Phase 18): For alert delivery

Usage:
    from src.core.proactive import ProactiveService, get_proactive_service

    # Create service (typically done at startup)
    service = ProactiveService(
        memory_manager=memory_manager,
        tenant_db_manager=tenant_db_manager,
    )

    # Check all monitors for a student
    alerts = await service.check_student(
        tenant_code="acme",
        student_id=student_uuid,
    )

    # Or use singleton
    service = get_proactive_service(memory_manager, tenant_db_manager)

Shared Thresholds (consistent with DiagnosticService):
- ELEVATED_RISK_THRESHOLD = 0.5
- HIGH_RISK_THRESHOLD = 0.7
"""

from src.core.proactive.monitors import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
    DiagnosticMonitor,
    ELEVATED_RISK_THRESHOLD,
    EngagementMonitor,
    HIGH_RISK_THRESHOLD,
    InactivityMonitor,
    MilestoneMonitor,
    StruggleMonitor,
)
from src.core.proactive.service import (
    ProactiveService,
    ProactiveServiceError,
    get_proactive_service,
)

__all__ = [
    # Service
    "ProactiveService",
    "ProactiveServiceError",
    "get_proactive_service",
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
    "EngagementMonitor",
    "InactivityMonitor",
    "MilestoneMonitor",
    "StruggleMonitor",
]
