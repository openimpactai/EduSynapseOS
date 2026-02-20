# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Create alert tool.

This tool creates proactive alerts for students that can be seen by
teachers and parents. Used when the companion detects situations
requiring stakeholder attention.

Examples:
- Student expresses emotional distress
- Student asks for help repeatedly
- Student mentions concerning topics
- Student shows significant improvement (milestone)
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.core.tools import BaseTool, ToolContext, ToolResult
from src.infrastructure.database.models.tenant.notification import Alert


# Valid alert types
ALERT_TYPES = frozenset({
    "emotional_distress",    # Student showing emotional difficulty
    "help_requested",        # Student explicitly asked for help
    "struggle_detected",     # Academic struggle detected in conversation
    "milestone_achieved",    # Student achieved something noteworthy
    "concerning_topic",      # Student mentioned concerning topic
    "engagement_issue",      # Student seems disengaged
})

# Valid severity levels
SEVERITY_LEVELS = frozenset({
    "info",      # Informational, no action needed
    "warning",   # Attention recommended
    "critical",  # Immediate attention needed
})


class CreateAlertTool(BaseTool):
    """Tool to create proactive alerts for stakeholders.

    Creates alerts that teachers and parents can see in their dashboards.
    Use this when the conversation reveals something important about
    the student's wellbeing, progress, or needs.

    The tool directly persists to the database - no workflow handling needed.
    Notification delivery is handled separately by the notification system.
    """

    @property
    def name(self) -> str:
        return "create_alert"

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "create_alert",
                "description": (
                    "Create an alert for teachers/parents about this student. "
                    "Use when:\n"
                    "- Student shows emotional distress needing adult attention\n"
                    "- Student explicitly asks for help with something serious\n"
                    "- Student mentions concerning topics (bullying, family issues)\n"
                    "- Student achieves a significant milestone worth celebrating\n"
                    "- Student seems disengaged or struggling repeatedly\n\n"
                    "This notifies relevant adults - use thoughtfully."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_type": {
                            "type": "string",
                            "enum": list(ALERT_TYPES),
                            "description": (
                                "Type of alert:\n"
                                "- emotional_distress: Student showing emotional difficulty\n"
                                "- help_requested: Student explicitly asked for help\n"
                                "- struggle_detected: Academic struggle in conversation\n"
                                "- milestone_achieved: Noteworthy achievement\n"
                                "- concerning_topic: Mentioned concerning topic\n"
                                "- engagement_issue: Student seems disengaged"
                            ),
                        },
                        "severity": {
                            "type": "string",
                            "enum": list(SEVERITY_LEVELS),
                            "description": (
                                "How urgent is this?\n"
                                "- info: FYI, no action needed\n"
                                "- warning: Should check on student\n"
                                "- critical: Needs immediate attention"
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": "Short alert title (max 100 chars)",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "Detailed message for teachers/parents. "
                                "Explain what happened and any context."
                            ),
                        },
                        "suggested_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Suggested actions for the adult. Examples:\n"
                                "- 'Check in with student about math anxiety'\n"
                                "- 'Celebrate student's progress in reading'\n"
                                "- 'Discuss with school counselor'"
                            ),
                        },
                    },
                    "required": ["alert_type", "severity", "title", "message"],
                },
            },
        }

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the create_alert tool.

        Creates an alert record in the database.

        Args:
            params: Tool parameters from LLM.
                - alert_type: Type of alert
                - severity: Alert severity
                - title: Alert title
                - message: Detailed message
                - suggested_actions: Optional list of suggested actions
            context: Execution context with session.

        Returns:
            ToolResult indicating success/failure.
        """
        alert_type = params.get("alert_type")
        severity = params.get("severity")
        title = params.get("title", "")
        message = params.get("message", "")
        suggested_actions = params.get("suggested_actions", [])

        # Validate alert_type
        if not alert_type:
            return ToolResult(
                success=False,
                error="Missing required parameter: alert_type",
            )

        if alert_type not in ALERT_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid alert_type: {alert_type}. Valid: {', '.join(ALERT_TYPES)}",
            )

        # Validate severity
        if not severity:
            return ToolResult(
                success=False,
                error="Missing required parameter: severity",
            )

        if severity not in SEVERITY_LEVELS:
            return ToolResult(
                success=False,
                error=f"Invalid severity: {severity}. Valid: {', '.join(SEVERITY_LEVELS)}",
            )

        # Validate title
        if not title:
            return ToolResult(
                success=False,
                error="Missing required parameter: title",
            )

        title = str(title)[:255]  # Truncate to DB limit

        # Validate message
        if not message:
            return ToolResult(
                success=False,
                error="Missing required parameter: message",
            )

        message = str(message)

        # Validate suggested_actions
        if suggested_actions:
            if not isinstance(suggested_actions, list):
                suggested_actions = [suggested_actions]
            suggested_actions = [str(a) for a in suggested_actions]

        # Check if session is available
        if not context.session:
            return ToolResult(
                success=False,
                error="Database session not available",
            )

        try:
            # Create alert record
            alert = Alert(
                id=str(uuid4()),
                student_id=str(context.student_id),
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                details={
                    "source": "companion_agent",
                    "created_via": "create_alert_tool",
                },
                suggested_actions=suggested_actions,
                status="active",
                created_at=datetime.now(timezone.utc),
            )

            context.session.add(alert)
            await context.session.flush()

            # Build response message
            response_message = (
                f"Alert created: [{severity.upper()}] {title}. "
                f"Teachers and parents will be notified."
            )

            return ToolResult(
                success=True,
                data={
                    "message": response_message,
                    "alert_id": alert.id,
                    "alert_type": alert_type,
                    "severity": severity,
                    "created": True,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create alert: {str(e)}",
            )
