# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Notification service for orchestrating notification delivery.

This service handles the complete notification flow:
1. Finding recipients based on alert targets
2. Checking user preferences and quiet hours
3. Rendering templates for different languages
4. Sending through enabled channels

Integration with ProactiveService:
The ProactiveService calls notify_from_alert() after storing an alert,
which triggers notifications to relevant stakeholders (teachers, parents).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.proactive.monitors.base import AlertData, AlertTarget
from src.infrastructure.database.models.tenant.notification import (
    Notification,
    NotificationPreference,
    NotificationTemplate,
)
from src.infrastructure.database.models.tenant.school import (
    ClassStudent,
    ClassTeacher,
    ParentStudentRelation,
)
from src.infrastructure.database.models.tenant.user import User
from src.infrastructure.database.tenant_manager import TenantDatabaseManager
from src.infrastructure.notifications.channels import (
    ChannelResult,
    ChannelType,
    EmailChannel,
    InAppChannel,
    NotificationPayload,
    PushChannel,
)

logger = logging.getLogger(__name__)


# Alert type to notification type mapping
ALERT_TO_NOTIFICATION_TYPE = {
    "struggle_detected": "alert_struggle",
    "engagement_drop": "alert_engagement",
    "milestone_achieved": "alert_milestone",
    "inactivity_warning": "alert_inactivity",
    "diagnostic_alert": "alert_diagnostic",
}


@dataclass
class Recipient:
    """A notification recipient with their details.

    Attributes:
        user_id: User's UUID.
        email: User's email address.
        full_name: User's display name.
        user_type: Type of user (teacher, parent, student).
        language: Preferred language code.
        timezone: User's timezone.
        push_tokens: List of push token info dicts.
    """

    user_id: UUID
    email: str
    full_name: str
    user_type: str
    language: str
    timezone: str
    push_tokens: list[dict[str, str]]


@dataclass
class NotificationResult:
    """Result of sending notifications for an alert.

    Attributes:
        alert_id: Related alert ID.
        recipients_count: Number of recipients notified.
        channel_results: Results per channel per recipient.
        errors: List of error messages.
    """

    alert_id: str | None
    recipients_count: int
    channel_results: list[dict[str, Any]]
    errors: list[str]


class NotificationService:
    """Service for sending notifications to users.

    Orchestrates the notification flow:
    1. Find recipients based on alert targets
    2. Check preferences and quiet hours
    3. Render templates
    4. Send through enabled channels

    Attributes:
        channels: Dictionary of available channels.
    """

    def __init__(self, tenant_db_manager: TenantDatabaseManager) -> None:
        """Initialize the notification service.

        Args:
            tenant_db_manager: Tenant database manager.
        """
        self._tenant_db = tenant_db_manager

        # Initialize channels
        self._in_app = InAppChannel()
        self._push = PushChannel()
        self._email = EmailChannel()

        self.channels: dict[ChannelType, Any] = {
            ChannelType.IN_APP: self._in_app,
            ChannelType.PUSH: self._push,
            ChannelType.EMAIL: self._email,
        }

        logger.info("NotificationService initialized with %d channels", len(self.channels))

    async def notify_from_alert(
        self,
        tenant_code: str,
        alert_data: AlertData,
    ) -> NotificationResult:
        """Send notifications for a proactive alert.

        Main entry point for alert-triggered notifications.
        Finds recipients based on alert targets and sends
        notifications through enabled channels.

        Args:
            tenant_code: Tenant identifier.
            alert_data: Alert data from ProactiveService.

        Returns:
            NotificationResult with delivery details.
        """
        logger.info(
            "Processing notifications for alert type %s, student %s",
            alert_data.alert_type.value,
            alert_data.student_id,
        )

        async with self._tenant_db.get_session(tenant_code) as session:
            # Get student info
            student = await self._get_user(session, alert_data.student_id)
            if not student:
                logger.error("Student %s not found", alert_data.student_id)
                return NotificationResult(
                    alert_id=None,
                    recipients_count=0,
                    channel_results=[],
                    errors=[f"Student {alert_data.student_id} not found"],
                )

            # Find recipients based on targets
            recipients = await self._find_recipients(
                session=session,
                student_id=alert_data.student_id,
                targets=alert_data.targets,
            )

            if not recipients:
                logger.info("No recipients found for alert")
                return NotificationResult(
                    alert_id=None,
                    recipients_count=0,
                    channel_results=[],
                    errors=[],
                )

            # Get notification type
            notification_type = ALERT_TO_NOTIFICATION_TYPE.get(
                alert_data.alert_type.value,
                f"alert_{alert_data.alert_type.value}",
            )

            # Send to each recipient
            all_results: list[dict[str, Any]] = []
            errors: list[str] = []

            for recipient in recipients:
                try:
                    results = await self._send_to_recipient(
                        session=session,
                        recipient=recipient,
                        notification_type=notification_type,
                        alert_data=alert_data,
                        student=student,
                    )
                    all_results.extend(results)
                except Exception as e:
                    error_msg = f"Failed to notify {recipient.email}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)

            await session.commit()

            logger.info(
                "Sent notifications to %d recipients with %d channel sends",
                len(recipients),
                len(all_results),
            )

            return NotificationResult(
                alert_id=str(alert_data.details.get("alert_db_id")) if alert_data.details else None,
                recipients_count=len(recipients),
                channel_results=all_results,
                errors=errors,
            )

    async def send_direct(
        self,
        tenant_code: str,
        user_id: UUID,
        notification_type: str,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
        action_url: str | None = None,
        action_label: str | None = None,
    ) -> list[ChannelResult]:
        """Send a direct notification to a specific user.

        Used for non-alert notifications (e.g., system messages).

        Args:
            tenant_code: Tenant identifier.
            user_id: Recipient user ID.
            notification_type: Type of notification.
            title: Notification title.
            message: Notification message.
            data: Additional data.
            action_url: Action URL.
            action_label: Action button label.

        Returns:
            List of channel results.
        """
        async with self._tenant_db.get_session(tenant_code) as session:
            user = await self._get_user(session, user_id)
            if not user:
                logger.error("User %s not found for direct notification", user_id)
                return []

            recipient = self._user_to_recipient(user)

            # Get preference
            preference = await self._get_preference(
                session, user_id, notification_type
            )

            # Check quiet hours
            if self._is_in_quiet_hours(recipient.timezone, preference):
                logger.debug("User %s is in quiet hours, skipping", user_id)
                return []

            # Build payload
            payload = NotificationPayload(
                notification_type=notification_type,
                title=title,
                message=message,
                recipient_id=user_id,
                recipient_email=recipient.email,
                recipient_language=recipient.language,
                recipient_timezone=recipient.timezone,
                data=data or {},
                action_url=action_url,
                action_label=action_label,
                push_tokens=recipient.push_tokens,
            )

            # Send through enabled channels
            results = await self._send_through_channels(
                session=session,
                payload=payload,
                preference=preference,
            )

            await session.commit()
            return results

    async def _find_recipients(
        self,
        session: AsyncSession,
        student_id: UUID,
        targets: list[AlertTarget],
    ) -> list[Recipient]:
        """Find all recipients for given targets.

        Args:
            session: Database session.
            student_id: Student the alert is about.
            targets: Alert target types.

        Returns:
            List of recipient objects.
        """
        recipients: list[Recipient] = []
        seen_ids: set[str] = set()

        for target in targets:
            target_recipients = await self._find_recipients_for_target(
                session, student_id, target
            )
            for r in target_recipients:
                if str(r.user_id) not in seen_ids:
                    recipients.append(r)
                    seen_ids.add(str(r.user_id))

        return recipients

    async def _find_recipients_for_target(
        self,
        session: AsyncSession,
        student_id: UUID,
        target: AlertTarget,
    ) -> list[Recipient]:
        """Find recipients for a specific target type.

        Args:
            session: Database session.
            student_id: Student the alert is about.
            target: Target type.

        Returns:
            List of recipient objects.
        """
        if target == AlertTarget.STUDENT:
            user = await self._get_user(session, student_id)
            if user:
                return [self._user_to_recipient(user)]
            return []

        elif target == AlertTarget.TEACHER:
            return await self._find_teachers_for_student(session, student_id)

        elif target == AlertTarget.PARENT:
            return await self._find_parents_for_student(session, student_id)

        elif target == AlertTarget.SYSTEM:
            # System target for internal logging, no recipients
            return []

        return []

    async def _find_teachers_for_student(
        self,
        session: AsyncSession,
        student_id: UUID,
    ) -> list[Recipient]:
        """Find all teachers for a student's classes.

        Args:
            session: Database session.
            student_id: Student ID.

        Returns:
            List of teacher recipients.
        """
        # Get student's active class enrollments
        class_query = (
            select(ClassStudent.class_id)
            .where(
                ClassStudent.student_id == str(student_id),
                ClassStudent.status == "active",
            )
        )
        class_result = await session.execute(class_query)
        class_ids = [row[0] for row in class_result.all()]

        if not class_ids:
            return []

        # Get teachers for those classes
        teacher_query = (
            select(ClassTeacher)
            .options(selectinload(ClassTeacher.teacher))
            .where(
                ClassTeacher.class_id.in_(class_ids),
                ClassTeacher.ended_at.is_(None),
            )
        )
        teacher_result = await session.execute(teacher_query)
        class_teachers = teacher_result.scalars().all()

        # Deduplicate and convert to recipients
        seen_ids: set[str] = set()
        recipients: list[Recipient] = []

        for ct in class_teachers:
            if ct.teacher_id not in seen_ids and ct.teacher:
                recipients.append(self._user_to_recipient(ct.teacher))
                seen_ids.add(ct.teacher_id)

        return recipients

    async def _find_parents_for_student(
        self,
        session: AsyncSession,
        student_id: UUID,
    ) -> list[Recipient]:
        """Find all parents for a student who can receive notifications.

        Args:
            session: Database session.
            student_id: Student ID.

        Returns:
            List of parent recipients.
        """
        query = (
            select(ParentStudentRelation)
            .options(selectinload(ParentStudentRelation.parent))
            .where(
                ParentStudentRelation.student_id == str(student_id),
                ParentStudentRelation.can_receive_notifications == True,
            )
        )
        result = await session.execute(query)
        relations = result.scalars().all()

        recipients: list[Recipient] = []
        for rel in relations:
            if rel.parent and rel.parent.is_active:
                recipients.append(self._user_to_recipient(rel.parent))

        return recipients

    async def _get_user(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> User | None:
        """Get a user by ID.

        Args:
            session: Database session.
            user_id: User ID.

        Returns:
            User model or None.
        """
        result = await session.execute(
            select(User).where(User.id == str(user_id))
        )
        return result.scalar_one_or_none()

    def _user_to_recipient(self, user: User) -> Recipient:
        """Convert User model to Recipient dataclass.

        Args:
            user: User model.

        Returns:
            Recipient dataclass.
        """
        # Extract push tokens from extra_data
        push_tokens = user.extra_data.get("push_tokens", [])

        return Recipient(
            user_id=UUID(user.id),
            email=user.email,
            full_name=user.full_name,
            user_type=user.user_type,
            language=user.preferred_language,
            timezone=user.timezone,
            push_tokens=push_tokens,
        )

    async def _send_to_recipient(
        self,
        session: AsyncSession,
        recipient: Recipient,
        notification_type: str,
        alert_data: AlertData,
        student: User,
    ) -> list[dict[str, Any]]:
        """Send notification to a single recipient.

        Args:
            session: Database session.
            recipient: Recipient info.
            notification_type: Notification type.
            alert_data: Alert data.
            student: Student user object.

        Returns:
            List of result dictionaries.
        """
        # Get preference
        preference = await self._get_preference(
            session, recipient.user_id, notification_type
        )

        # Check if notifications enabled
        if preference and not preference.is_enabled:
            logger.debug(
                "Notifications disabled for user %s, type %s",
                recipient.user_id,
                notification_type,
            )
            return []

        # Check quiet hours
        if self._is_in_quiet_hours(recipient.timezone, preference):
            logger.debug(
                "User %s is in quiet hours, skipping",
                recipient.user_id,
            )
            return []

        # Get template and render
        title, message = await self._render_notification(
            session=session,
            notification_type=notification_type,
            language=recipient.language,
            alert_data=alert_data,
            student_name=student.full_name,
        )

        # Build payload
        payload = NotificationPayload(
            notification_type=notification_type,
            title=title,
            message=message,
            recipient_id=recipient.user_id,
            recipient_email=recipient.email,
            recipient_language=recipient.language,
            recipient_timezone=recipient.timezone,
            student_id=alert_data.student_id,
            student_name=student.full_name,
            alert_id=str(alert_data.details.get("alert_db_id")) if alert_data.details else None,
            data={
                "alert_type": alert_data.alert_type.value,
                "severity": alert_data.severity.value,
            },
            action_url=f"/students/{alert_data.student_id}/alerts",
            action_label="View Details",
            push_tokens=recipient.push_tokens,
            priority="high" if alert_data.severity.value == "critical" else "normal",
        )

        # Send through channels
        results = await self._send_through_channels(
            session=session,
            payload=payload,
            preference=preference,
        )

        return [
            {
                "recipient_id": str(recipient.user_id),
                "recipient_type": recipient.user_type,
                **r.to_dict(),
            }
            for r in results
        ]

    async def _get_preference(
        self,
        session: AsyncSession,
        user_id: UUID,
        notification_type: str,
    ) -> NotificationPreference | None:
        """Get user's notification preference.

        Args:
            session: Database session.
            user_id: User ID.
            notification_type: Notification type.

        Returns:
            Preference or None for defaults.
        """
        result = await session.execute(
            select(NotificationPreference).where(
                NotificationPreference.user_id == str(user_id),
                NotificationPreference.notification_type == notification_type,
            )
        )
        return result.scalar_one_or_none()

    def _is_in_quiet_hours(
        self,
        timezone_str: str,
        preference: NotificationPreference | None,
    ) -> bool:
        """Check if current time is in user's quiet hours.

        Args:
            timezone_str: User's timezone string.
            preference: User's preference.

        Returns:
            True if in quiet hours.
        """
        if not preference or not preference.quiet_start or not preference.quiet_end:
            return False

        try:
            tz = ZoneInfo(timezone_str)
            current_time = datetime.now(tz).time()

            quiet_start = preference.quiet_start
            quiet_end = preference.quiet_end

            # Handle overnight quiet hours
            if quiet_start > quiet_end:
                return current_time >= quiet_start or current_time <= quiet_end
            else:
                return quiet_start <= current_time <= quiet_end

        except Exception as e:
            logger.warning("Error checking quiet hours: %s", str(e))
            return False

    async def _render_notification(
        self,
        session: AsyncSession,
        notification_type: str,
        language: str,
        alert_data: AlertData,
        student_name: str,
    ) -> tuple[str, str]:
        """Render notification title and message.

        Uses NotificationTemplate if available, otherwise defaults.

        Args:
            session: Database session.
            notification_type: Notification type.
            language: Language code.
            alert_data: Alert data.
            student_name: Student's name.

        Returns:
            Tuple of (title, message).
        """
        # Try to find template
        result = await session.execute(
            select(NotificationTemplate).where(
                NotificationTemplate.notification_type == notification_type,
                NotificationTemplate.language_code == language,
            )
        )
        template = result.scalar_one_or_none()

        # Build context
        context = {
            "student_name": student_name,
            "alert_title": alert_data.title,
            "alert_message": alert_data.message,
            "severity": alert_data.severity.value,
        }

        if template:
            try:
                return template.render(context)
            except KeyError as e:
                logger.warning(
                    "Template render error for %s: %s",
                    notification_type,
                    str(e),
                )

        # Default: use alert's title and message
        return alert_data.title, alert_data.message

    async def _send_through_channels(
        self,
        session: AsyncSession,
        payload: NotificationPayload,
        preference: NotificationPreference | None,
    ) -> list[ChannelResult]:
        """Send notification through all enabled channels.

        Args:
            session: Database session.
            payload: Notification payload.
            preference: User's preference.

        Returns:
            List of channel results.
        """
        results: list[ChannelResult] = []

        # In-app channel (requires session)
        if self._in_app.is_enabled_for_preference(preference):
            self._in_app.set_session(session)
            result = await self._in_app.send(payload)
            results.append(result)

        # Push channel
        if self._push.is_enabled_for_preference(preference):
            result = await self._push.send(payload)
            results.append(result)

        # Email channel
        if self._email.is_enabled_for_preference(preference):
            result = await self._email.send(payload)
            results.append(result)

        return results


# Singleton instance management
_service_instance: NotificationService | None = None


def get_notification_service(
    tenant_db_manager: TenantDatabaseManager,
) -> NotificationService:
    """Get or create the notification service singleton.

    Args:
        tenant_db_manager: Tenant database manager.

    Returns:
        NotificationService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = NotificationService(tenant_db_manager)
    return _service_instance
