# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""In-app notification channel.

This channel creates notification records in the database
that are displayed within the application UI. This is the
primary and most reliable notification channel.
"""

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models.tenant.notification import Notification
from src.infrastructure.notifications.channels.base import (
    BaseChannel,
    ChannelResult,
    ChannelType,
    DeliveryStatus,
    NotificationPayload,
)

logger = logging.getLogger(__name__)


class InAppChannel(BaseChannel):
    """In-app notification channel.

    Creates notification records in the notifications table.
    These are displayed in the application's notification center.

    This channel requires a database session to be set before
    sending notifications via set_session().
    """

    # Default expiration time for notifications (30 days)
    DEFAULT_EXPIRATION_DAYS = 30

    def __init__(self) -> None:
        """Initialize the in-app channel."""
        super().__init__()
        self._session: AsyncSession | None = None

    @property
    def channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.IN_APP

    def set_session(self, session: AsyncSession) -> None:
        """Set the database session for this channel.

        Must be called before send() when using this channel.

        Args:
            session: Async database session.
        """
        self._session = session

    async def send(self, payload: NotificationPayload) -> ChannelResult:
        """Create an in-app notification record.

        Args:
            payload: The notification payload.

        Returns:
            ChannelResult with delivery status.
        """
        if self._session is None:
            return self.create_failure_result(
                "Database session not set. Call set_session() first."
            )

        try:
            # Calculate expiration time
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=self.DEFAULT_EXPIRATION_DAYS
            )

            # Build notification data
            notification_data = {
                "student_id": str(payload.student_id) if payload.student_id else None,
                "student_name": payload.student_name,
                "alert_id": payload.alert_id,
                **payload.data,
            }

            # Create notification record
            notification = Notification(
                id=str(uuid4()),
                user_id=str(payload.recipient_id),
                notification_type=payload.notification_type,
                title=payload.title,
                message=payload.message,
                data=notification_data,
                channels=[ChannelType.IN_APP.value],
                delivery_status={
                    ChannelType.IN_APP.value: {
                        "status": DeliveryStatus.SENT.value,
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
                action_url=payload.action_url,
                action_label=payload.action_label,
                expires_at=expires_at,
            )

            self._session.add(notification)
            await self._session.flush()

            self.logger.info(
                "Created in-app notification %s for user %s",
                notification.id,
                payload.recipient_id,
            )

            return self.create_success_result(
                message_id=notification.id,
                metadata={"notification_id": notification.id},
            )

        except Exception as e:
            self.logger.error(
                "Failed to create in-app notification for user %s: %s",
                payload.recipient_id,
                str(e),
                exc_info=True,
            )
            return self.create_failure_result(
                f"Database error: {str(e)}"
            )
