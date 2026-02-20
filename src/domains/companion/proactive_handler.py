# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Companion proactive alert handler.

This module provides the system alert handler for companion integration
with ProactiveService. When alerts with AlertTarget.SYSTEM are generated,
this handler determines if the companion should proactively intervene.

The handler is registered at application startup and runs in fire-and-forget
mode, not blocking the main ProactiveService flow.
"""

import logging
from typing import Any

from src.core.proactive.monitors.base import AlertData

logger = logging.getLogger(__name__)


async def companion_system_alert_handler(
    tenant_code: str,
    alert_data: AlertData,
) -> dict[str, Any]:
    """Handle system alerts for companion intervention decisions.

    This handler is registered with ProactiveService via set_system_alert_handler().
    It analyzes incoming alerts and determines if the companion should
    proactively reach out to the student.

    The response is logged and can be used by the frontend (via polling or
    WebSocket) to trigger companion interventions.

    Args:
        tenant_code: Tenant identifier.
        alert_data: Alert data from ProactiveService.

    Returns:
        Response dict with:
        - should_speak: Whether companion should proactively speak
        - priority: Intervention priority (high, medium, low)
        - suggested_action: Optional action type for companion
        - alert_id: Database ID of the alert
        - alert_type: Type of the alert
    """
    logger.info(
        "Companion handler received alert: type=%s, severity=%s, student=%s, tenant=%s",
        alert_data.alert_type.value,
        alert_data.severity.value,
        alert_data.student_id,
        tenant_code,
    )

    # Determine response based on alert type and severity
    should_speak = False
    priority = "low"
    suggested_action = None

    # Critical alerts always trigger companion with high priority
    if alert_data.severity.value == "critical":
        should_speak = True
        priority = "high"
        logger.info(
            "Critical alert - companion should speak immediately: %s",
            alert_data.title,
        )

    # Warning alerts trigger companion with medium priority
    elif alert_data.severity.value == "warning":
        should_speak = True
        priority = "medium"
        logger.info(
            "Warning alert - companion should check in: %s",
            alert_data.title,
        )

    # Handle specific alert types
    alert_type = alert_data.alert_type.value

    if alert_type == "emotional_distress":
        should_speak = True
        priority = "high"
        suggested_action = "emotional_support"
        logger.info(
            "Emotional distress - companion should provide support: %s",
            alert_data.title,
        )

    elif alert_type == "struggle_detected":
        should_speak = True
        if priority == "low":
            priority = "medium"
        suggested_action = "suggest_break"
        logger.info(
            "Struggle detected - companion should offer encouragement: %s",
            alert_data.title,
        )

    elif alert_type == "engagement_drop":
        should_speak = True
        suggested_action = "suggest_activity"
        logger.info(
            "Engagement drop - companion should suggest activity: %s",
            alert_data.title,
        )

    elif alert_type == "inactivity_warning":
        should_speak = True
        suggested_action = "welcome_back"
        logger.info(
            "Inactivity - companion should welcome back: %s",
            alert_data.title,
        )

    elif alert_type == "milestone_achieved":
        should_speak = True
        suggested_action = "celebrate"
        if priority == "low":
            priority = "medium"
        logger.info(
            "Milestone achieved - companion should celebrate: %s",
            alert_data.title,
        )

    response = {
        "should_speak": should_speak,
        "priority": priority,
        "suggested_action": suggested_action,
        "alert_id": alert_data.details.get("alert_db_id"),
        "alert_type": alert_type,
        "student_id": str(alert_data.student_id),
        "topic_full_code": alert_data.topic_full_code,
        "topic_codes": alert_data.topic_codes,
    }

    logger.info(
        "Companion handler response: should_speak=%s, priority=%s, action=%s",
        should_speak,
        priority,
        suggested_action,
    )

    return response
