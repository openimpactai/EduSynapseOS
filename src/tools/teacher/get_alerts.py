# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Get alerts tool for teachers.

Returns active alerts for students in the teacher's classes,
including academic struggles, engagement issues, and emotional concerns.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.core.tools.ui_elements import UIElement, UIElementOption, UIElementType
from src.tools.teacher.helpers import (
    get_teacher_class_ids,
    get_teacher_student_ids,
    verify_teacher_has_class_access,
)

logger = logging.getLogger(__name__)


class GetAlertsTool(BaseTool):
    """Tool to get alerts for teacher's students."""

    @property
    def name(self) -> str:
        return "get_alerts"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_alerts",
                "description": "Get active alerts for students in your classes, including academic struggles, engagement issues, and emotional concerns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_id": {
                            "type": "string",
                            "description": "Optional: Filter alerts to a specific class",
                        },
                        "alert_type": {
                            "type": "string",
                            "enum": ["struggle", "engagement", "milestone", "emotional", "performance"],
                            "description": "Optional: Filter by alert type",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "critical"],
                            "description": "Optional: Filter by severity level",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "acknowledged", "resolved"],
                            "description": "Alert status filter (default: active)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of alerts to return (default: 30)",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the get_alerts tool.

        Args:
            params: Tool parameters (class_id, alert_type, severity, status, limit).
            context: Tool context with user_id (teacher).

        Returns:
            ToolResult with alerts.
        """
        if not context.is_teacher:
            return ToolResult(
                success=False,
                error="This tool is only available for teachers.",
            )

        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available.",
            )

        class_id_str = params.get("class_id")
        alert_type = params.get("alert_type")
        severity = params.get("severity")
        status = params.get("status", "active")
        limit = params.get("limit", 30)
        teacher_id = context.user_id

        try:
            # Determine which students to include
            if class_id_str:
                try:
                    class_id = UUID(class_id_str)
                except ValueError:
                    return ToolResult(
                        success=False,
                        error="Invalid class_id format.",
                    )

                has_access = await verify_teacher_has_class_access(
                    context.session, teacher_id, class_id
                )
                if not has_access:
                    return ToolResult(
                        success=False,
                        error="You don't have access to this class.",
                    )

                # Get students in this class
                from src.infrastructure.database.models.tenant.school import ClassStudent

                student_ids_query = (
                    select(ClassStudent.student_id)
                    .where(ClassStudent.class_id == str(class_id))
                    .where(ClassStudent.status == "active")
                )
                student_ids_result = await context.session.execute(student_ids_query)
                student_ids = [str(row[0]) for row in student_ids_result.all()]
            else:
                # Get all students from teacher's classes
                student_ids = await get_teacher_student_ids(context.session, teacher_id)

            if not student_ids:
                return ToolResult(
                    success=True,
                    data={
                        "message": "No students found in your classes.",
                        "alerts": [],
                        "count": 0,
                    },
                )

            from src.infrastructure.database.models.tenant.notification import Alert
            from src.infrastructure.database.models.tenant.user import User

            # Build query for alerts
            query = (
                select(
                    Alert,
                    User.first_name,
                    User.last_name,
                )
                .join(User, User.id == Alert.student_id)
                .where(Alert.student_id.in_(student_ids))
                .where(Alert.status == status)
            )

            if alert_type:
                query = query.where(Alert.alert_type == alert_type)

            if severity:
                query = query.where(Alert.severity == severity)

            # Order by severity (critical first) then by created_at
            query = query.order_by(
                Alert.severity.desc(),
                Alert.created_at.desc(),
            ).limit(limit)

            result = await context.session.execute(query)
            rows = result.all()

            alerts = []
            critical_count = 0
            warning_count = 0

            for row in rows:
                alert = row[0]
                student_first = row[1]
                student_last = row[2]

                alert_data = {
                    "id": str(alert.id),
                    "student_id": str(alert.student_id),
                    "student_name": f"{student_first} {student_last}",
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "details": alert.details,
                    "suggested_actions": alert.suggested_actions,
                    "emotional_trigger": alert.emotional_trigger,
                    "status": alert.status,
                    "created_at": alert.created_at.isoformat() if alert.created_at else None,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                }
                alerts.append(alert_data)

                if alert.severity == "critical":
                    critical_count += 1
                elif alert.severity == "warning":
                    warning_count += 1

            # Group by alert type
            by_type = {}
            for alert in alerts:
                atype = alert["alert_type"]
                if atype not in by_type:
                    by_type[atype] = []
                by_type[atype].append(alert)

            # Build message
            if not alerts:
                message = f"No {status} alerts found."
            else:
                message = f"{len(alerts)} {status} alerts:\n"
                if critical_count > 0:
                    message += f"- Critical: {critical_count}\n"
                if warning_count > 0:
                    message += f"- Warning: {warning_count}\n"

                for atype, type_alerts in by_type.items():
                    message += f"- {atype}: {len(type_alerts)} alerts\n"

                # Show most critical alert
                if alerts and alerts[0]["severity"] == "critical":
                    message += f"\nMost urgent: {alerts[0]['student_name']} - {alerts[0]['title']}"

            # Build UI element for alert selection
            ui_element = None
            if alerts:
                options = [
                    UIElementOption(
                        id=a["id"],
                        label=f"{a['student_name']} - {a['title']}",
                        description=f"{a['severity'].upper()} - {a['alert_type']}",
                    )
                    for a in alerts[:10]
                ]
                ui_element = UIElement(
                    type=UIElementType.SINGLE_SELECT,
                    id="alert_selection",
                    title="Select an Alert",
                    options=options,
                    allow_text_input=False,
                )

            logger.info(
                "get_alerts: teacher=%s, alerts=%d, critical=%d",
                teacher_id,
                len(alerts),
                critical_count,
            )

            return ToolResult(
                success=True,
                data={
                    "message": message,
                    "alerts": alerts,
                    "count": len(alerts),
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "by_type": {k: len(v) for k, v in by_type.items()},
                },
                ui_element=ui_element,
                passthrough_data={
                    "alerts": alerts[:5],
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                },
            )

        except Exception as e:
            logger.exception("get_alerts failed")
            return ToolResult(
                success=False,
                error=f"Failed to get alerts: {str(e)}",
            )
