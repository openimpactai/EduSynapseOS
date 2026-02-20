# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Proactive intelligence service for orchestrating monitors.

This service coordinates all proactive monitors to detect patterns
in student behavior and generate appropriate alerts. It integrates
with the memory system for context and the database for alert storage.

After storing alerts, it triggers notifications via NotificationService
to deliver alerts to relevant stakeholders (teachers, parents).

For SYSTEM target alerts (e.g., emotional distress), the service can
trigger a registered system alert handler (e.g., CompanionService) to
provide immediate in-app intervention.

Usage:
    service = ProactiveService(
        memory_manager=memory_manager,
        tenant_db_manager=tenant_db_manager,
    )

    # Register system alert handler (e.g., for Companion)
    service.set_system_alert_handler(companion_handler)

    # Check all monitors for a student
    alerts = await service.check_student(
        tenant_code="acme",
        student_id=student_uuid,
    )

    # Process an interaction (trigger relevant monitors)
    await service.process_interaction(
        tenant_code="acme",
        student_id=student_uuid,
        interaction_type="answer_submitted",
        interaction_data={"is_correct": False},
    )
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.memory.manager import MemoryManager
from src.core.proactive.monitors.base import (
    AlertData,
    AlertSeverity,
    AlertTarget,
    AlertType,
    BaseMonitor,
)
from src.core.proactive.monitors.diagnostic import DiagnosticMonitor
from src.core.proactive.monitors.emotional import EmotionalDistressMonitor
from src.core.proactive.monitors.engagement import EngagementMonitor
from src.core.proactive.monitors.inactivity import InactivityMonitor
from src.core.proactive.monitors.milestone import MilestoneMonitor
from src.core.proactive.monitors.struggle import StruggleMonitor
from src.infrastructure.database.models.tenant.notification import Alert
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.notifications.service import (
    NotificationService,
    get_notification_service,
)

logger = logging.getLogger(__name__)


class ProactiveServiceError(Exception):
    """Exception raised for proactive service operations."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ProactiveService:
    """Service for orchestrating proactive monitoring.

    Coordinates all monitors to detect patterns in student behavior
    and generate alerts. Handles alert storage and retrieval.

    Monitors included:
    - StruggleMonitor: Detects consecutive errors and low accuracy
    - EngagementMonitor: Detects declining engagement patterns
    - MilestoneMonitor: Celebrates achievements and progress
    - InactivityMonitor: Detects prolonged absence
    - DiagnosticMonitor: Alerts on diagnostic findings

    Attributes:
        monitors: List of active monitor instances.
    """

    # Type alias for system alert handler callback
    SystemAlertHandler = Callable[[str, AlertData], Awaitable[dict[str, Any]]]

    def __init__(
        self,
        memory_manager: MemoryManager,
        tenant_db_manager: TenantDatabaseManager,
    ) -> None:
        """Initialize the proactive service.

        Args:
            memory_manager: Memory manager for context retrieval.
            tenant_db_manager: Tenant database manager for alert storage.
        """
        self._memory = memory_manager
        self._tenant_db = tenant_db_manager
        self._notification_service: NotificationService = get_notification_service(
            tenant_db_manager
        )

        # System alert handler for SYSTEM target alerts (e.g., Companion)
        self._system_alert_handler: ProactiveService.SystemAlertHandler | None = None

        # Initialize all monitors
        self.monitors: list[BaseMonitor] = [
            StruggleMonitor(),
            EngagementMonitor(),
            MilestoneMonitor(),
            InactivityMonitor(),
            DiagnosticMonitor(),
            EmotionalDistressMonitor(),
        ]

        logger.info(
            "ProactiveService initialized with %d monitors",
            len(self.monitors),
        )

    def set_system_alert_handler(
        self,
        handler: "ProactiveService.SystemAlertHandler",
    ) -> None:
        """Register a handler for SYSTEM target alerts.

        The handler will be called (fire-and-forget) when an alert
        with AlertTarget.SYSTEM is generated. This is used to trigger
        immediate in-app interventions like the Companion.

        Args:
            handler: Async callable(tenant_code, alert_data) -> response dict
        """
        self._system_alert_handler = handler
        logger.info("System alert handler registered")

    async def check_student(
        self,
        tenant_code: str,
        student_id: UUID,
        include_diagnostic: bool = True,
    ) -> list[AlertData]:
        """Run all monitors for a student and return generated alerts.

        This is the main entry point for proactive monitoring.
        It retrieves the student's full context and runs all monitors.

        Args:
            tenant_code: Tenant identifier.
            student_id: Student to check.
            include_diagnostic: Whether to include diagnostic context.

        Returns:
            List of generated AlertData objects.
        """
        logger.info(
            "Checking all monitors for student %s in tenant %s",
            student_id,
            tenant_code,
        )

        try:
            # Get full memory context
            context = await self._memory.get_full_context(
                tenant_code=tenant_code,
                student_id=student_id,
                include_diagnostic=include_diagnostic,
            )

            # Run all monitors
            alerts: list[AlertData] = []
            for monitor in self.monitors:
                try:
                    alert = await monitor.check(context, tenant_code)
                    if alert:
                        alerts.append(alert)
                        logger.info(
                            "Monitor %s generated alert: %s",
                            monitor.name,
                            alert.alert_type.value,
                        )
                except Exception as e:
                    logger.error(
                        "Monitor %s failed: %s",
                        monitor.name,
                        str(e),
                        exc_info=True,
                    )
                    # Continue with other monitors

            # Store generated alerts and trigger notifications
            if alerts:
                await self._store_alerts(tenant_code, alerts)
                await self._trigger_notifications(tenant_code, alerts)

            logger.info(
                "Generated %d alerts for student %s",
                len(alerts),
                student_id,
            )

            return alerts

        except Exception as e:
            logger.error(
                "Failed to check monitors for student %s: %s",
                student_id,
                str(e),
                exc_info=True,
            )
            raise ProactiveServiceError(
                f"Failed to check monitors: {str(e)}",
                original_error=e,
            )

    async def check_specific_monitors(
        self,
        tenant_code: str,
        student_id: UUID,
        monitor_types: list[AlertType],
    ) -> list[AlertData]:
        """Run specific monitors for a student.

        Useful when you know which monitors are relevant based
        on the type of interaction that occurred.

        Args:
            tenant_code: Tenant identifier.
            student_id: Student to check.
            monitor_types: Types of monitors to run.

        Returns:
            List of generated AlertData objects.
        """
        logger.debug(
            "Checking specific monitors %s for student %s",
            [t.value for t in monitor_types],
            student_id,
        )

        # Get full memory context
        context = await self._memory.get_full_context(
            tenant_code=tenant_code,
            student_id=student_id,
            include_diagnostic=True,
        )

        # Filter monitors by type
        relevant_monitors = [
            m for m in self.monitors
            if m.alert_type in monitor_types
        ]

        # Run selected monitors
        alerts: list[AlertData] = []
        for monitor in relevant_monitors:
            try:
                alert = await monitor.check(context, tenant_code)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(
                    "Monitor %s failed: %s",
                    monitor.name,
                    str(e),
                )

        # Store generated alerts and trigger notifications
        if alerts:
            await self._store_alerts(tenant_code, alerts)
            await self._trigger_notifications(tenant_code, alerts)

        return alerts

    async def process_interaction(
        self,
        tenant_code: str,
        student_id: UUID,
        interaction_type: str,
        interaction_data: dict[str, Any] | None = None,
    ) -> list[AlertData]:
        """Process an interaction and trigger relevant monitors.

        Called when a student interaction occurs (e.g., answer submitted).
        Determines which monitors are relevant and runs them.

        Args:
            tenant_code: Tenant identifier.
            student_id: Student who interacted.
            interaction_type: Type of interaction.
            interaction_data: Additional interaction data.

        Returns:
            List of generated AlertData objects.
        """
        logger.debug(
            "Processing interaction '%s' for student %s",
            interaction_type,
            student_id,
        )

        # Map interaction types to relevant monitors
        interaction_monitor_map = {
            "answer_submitted": [
                AlertType.STRUGGLE_DETECTED,
                AlertType.MILESTONE_ACHIEVED,
                AlertType.EMOTIONAL_DISTRESS,
            ],
            "session_started": [
                AlertType.ENGAGEMENT_DROP,
                AlertType.INACTIVITY_WARNING,
            ],
            "session_completed": [
                AlertType.ENGAGEMENT_DROP,
                AlertType.MILESTONE_ACHIEVED,
                AlertType.EMOTIONAL_DISTRESS,
            ],
            "mastery_updated": [
                AlertType.MILESTONE_ACHIEVED,
                AlertType.DIAGNOSTIC_ALERT,
            ],
            "diagnostic_scan_completed": [
                AlertType.DIAGNOSTIC_ALERT,
            ],
            "chat_message_sent": [
                AlertType.EMOTIONAL_DISTRESS,
            ],
            "emotional_signal_recorded": [
                AlertType.EMOTIONAL_DISTRESS,
            ],
        }

        # Get relevant monitor types
        monitor_types = interaction_monitor_map.get(
            interaction_type,
            list(AlertType),  # Default: check all
        )

        return await self.check_specific_monitors(
            tenant_code=tenant_code,
            student_id=student_id,
            monitor_types=monitor_types,
        )

    async def get_active_alerts(
        self,
        tenant_code: str,
        student_id: UUID | None = None,
        alert_types: list[AlertType] | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[Alert]:
        """Get active alerts from the database.

        Args:
            tenant_code: Tenant identifier.
            student_id: Filter by student (optional).
            alert_types: Filter by alert types (optional).
            severity: Filter by severity (optional).
            limit: Maximum number of alerts to return.

        Returns:
            List of Alert model instances.
        """
        async with self._tenant_db.get_session(tenant_code) as session:
            query = select(Alert).where(Alert.status == "active")

            if student_id:
                query = query.where(Alert.student_id == str(student_id))

            if alert_types:
                type_values = [t.value for t in alert_types]
                query = query.where(Alert.alert_type.in_(type_values))

            if severity:
                query = query.where(Alert.severity == severity.value)

            query = query.order_by(Alert.created_at.desc()).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_student_alert_summary(
        self,
        tenant_code: str,
        student_id: UUID,
    ) -> dict[str, Any]:
        """Get a summary of alerts for a student.

        Args:
            tenant_code: Tenant identifier.
            student_id: Student ID.

        Returns:
            Summary dictionary with counts and recent alerts.
        """
        async with self._tenant_db.get_session(tenant_code) as session:
            # Get active alerts
            active_result = await session.execute(
                select(Alert)
                .where(
                    Alert.student_id == str(student_id),
                    Alert.status == "active",
                )
                .order_by(Alert.created_at.desc())
                .limit(10)
            )
            active_alerts = list(active_result.scalars().all())

            # Count by type
            type_counts: dict[str, int] = {}
            severity_counts: dict[str, int] = {}

            for alert in active_alerts:
                type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

            return {
                "student_id": str(student_id),
                "active_count": len(active_alerts),
                "by_type": type_counts,
                "by_severity": severity_counts,
                "recent_alerts": [
                    {
                        "id": alert.id,
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "title": alert.title,
                        "created_at": alert.created_at.isoformat() if alert.created_at else None,
                    }
                    for alert in active_alerts[:5]
                ],
                "has_critical": "critical" in severity_counts,
            }

    async def acknowledge_alert(
        self,
        tenant_code: str,
        alert_id: str,
        user_id: str,
    ) -> bool:
        """Acknowledge an alert.

        Args:
            tenant_code: Tenant identifier.
            alert_id: Alert ID to acknowledge.
            user_id: User who is acknowledging.

        Returns:
            True if acknowledged, False if not found.
        """
        async with self._tenant_db.get_session(tenant_code) as session:
            result = await session.execute(
                select(Alert).where(Alert.id == alert_id)
            )
            alert = result.scalar_one_or_none()

            if not alert:
                return False

            alert.acknowledge(user_id)
            await session.commit()

            logger.info(
                "Alert %s acknowledged by user %s",
                alert_id,
                user_id,
            )

            return True

    async def resolve_alert(
        self,
        tenant_code: str,
        alert_id: str,
    ) -> bool:
        """Resolve an alert.

        Args:
            tenant_code: Tenant identifier.
            alert_id: Alert ID to resolve.

        Returns:
            True if resolved, False if not found.
        """
        async with self._tenant_db.get_session(tenant_code) as session:
            result = await session.execute(
                select(Alert).where(Alert.id == alert_id)
            )
            alert = result.scalar_one_or_none()

            if not alert:
                return False

            alert.resolve()
            await session.commit()

            logger.info("Alert %s resolved", alert_id)

            return True

    async def _store_alerts(
        self,
        tenant_code: str,
        alerts: list[AlertData],
    ) -> list[str]:
        """Store generated alerts in the database.

        Args:
            tenant_code: Tenant identifier.
            alerts: AlertData objects to store.

        Returns:
            List of created alert IDs.
        """
        if not alerts:
            return []

        async with self._tenant_db.get_session(tenant_code) as session:
            created_ids: list[str] = []

            for alert_data in alerts:
                # Check for duplicate active alerts
                existing = await self._find_similar_active_alert(
                    session, alert_data
                )
                if existing:
                    logger.debug(
                        "Similar active alert exists for student %s, skipping",
                        alert_data.student_id,
                    )
                    continue

                # Create new alert
                db_dict = alert_data.to_db_dict()
                alert = Alert(**db_dict)
                session.add(alert)
                await session.flush()

                # Store alert ID in details for notification service
                alert_data.details["alert_db_id"] = alert.id
                created_ids.append(alert.id)

                logger.info(
                    "Created alert %s: %s for student %s",
                    alert.id,
                    alert_data.alert_type.value,
                    alert_data.student_id,
                )

            await session.commit()

            return created_ids

    async def _find_similar_active_alert(
        self,
        session: AsyncSession,
        alert_data: AlertData,
    ) -> Alert | None:
        """Find a similar active alert to prevent duplicates.

        Args:
            session: Database session.
            alert_data: Alert to check for duplicates.

        Returns:
            Existing similar alert or None.
        """
        result = await session.execute(
            select(Alert)
            .where(
                Alert.student_id == str(alert_data.student_id),
                Alert.alert_type == alert_data.alert_type.value,
                Alert.status == "active",
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _trigger_notifications(
        self,
        tenant_code: str,
        alerts: list[AlertData],
    ) -> None:
        """Trigger notifications for stored alerts.

        Sends notifications to relevant stakeholders (teachers, parents)
        via the NotificationService. Each alert is processed independently
        to ensure partial failures don't block other notifications.

        For alerts with AlertTarget.SYSTEM, also triggers the registered
        system alert handler (e.g., CompanionService) for immediate
        in-app intervention.

        Args:
            tenant_code: Tenant identifier.
            alerts: AlertData objects with stored alert IDs.
        """
        for alert_data in alerts:
            # Only process alerts that were actually stored
            if not alert_data.details.get("alert_db_id"):
                continue

            # Check for SYSTEM target and trigger handler (fire-and-forget)
            if AlertTarget.SYSTEM in alert_data.targets:
                self._trigger_system_alert_handler(tenant_code, alert_data)

            try:
                result = await self._notification_service.notify_from_alert(
                    tenant_code=tenant_code,
                    alert_data=alert_data,
                )

                if result.errors:
                    logger.warning(
                        "Notification errors for alert %s: %s",
                        alert_data.details.get("alert_db_id"),
                        result.errors,
                    )
                else:
                    logger.info(
                        "Notified %d recipients for alert %s",
                        result.recipients_count,
                        alert_data.details.get("alert_db_id"),
                    )

            except Exception as e:
                # Log but don't fail - notifications are best-effort
                logger.error(
                    "Failed to send notifications for alert %s: %s",
                    alert_data.details.get("alert_db_id"),
                    str(e),
                    exc_info=True,
                )

    def _trigger_system_alert_handler(
        self,
        tenant_code: str,
        alert_data: AlertData,
    ) -> None:
        """Trigger system alert handler in fire-and-forget mode.

        Creates a background task to call the registered handler.
        This ensures the main notification flow is not blocked.

        Args:
            tenant_code: Tenant identifier.
            alert_data: Alert data to process.
        """
        if not self._system_alert_handler:
            logger.debug(
                "No system alert handler registered, skipping SYSTEM target for alert %s",
                alert_data.details.get("alert_db_id"),
            )
            return

        async def _call_handler():
            try:
                response = await self._system_alert_handler(tenant_code, alert_data)
                logger.info(
                    "System alert handler completed for alert %s: should_speak=%s",
                    alert_data.details.get("alert_db_id"),
                    response.get("should_speak", False),
                )
            except Exception as e:
                logger.error(
                    "System alert handler failed for alert %s: %s",
                    alert_data.details.get("alert_db_id"),
                    str(e),
                    exc_info=True,
                )

        # Fire-and-forget
        asyncio.create_task(_call_handler())
        logger.debug(
            "Triggered system alert handler for alert %s",
            alert_data.details.get("alert_db_id"),
        )


# Singleton instance management
_service_instance: ProactiveService | None = None


def get_proactive_service(
    memory_manager: MemoryManager,
    tenant_db_manager: TenantDatabaseManager,
) -> ProactiveService:
    """Get or create the proactive service singleton.

    Args:
        memory_manager: Memory manager instance.
        tenant_db_manager: Tenant database manager instance.

    Returns:
        ProactiveService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ProactiveService(
            memory_manager=memory_manager,
            tenant_db_manager=tenant_db_manager,
        )
    return _service_instance
