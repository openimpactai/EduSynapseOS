# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base classes for notification channels.

This module defines the abstract base class and shared types
for all notification channels. Each channel handles delivery
through a specific medium (in-app, push, email, SMS).

Channel implementations must be async and handle their own
error recovery and retry logic where applicable.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any
from uuid import UUID

from src.infrastructure.database.models.tenant.notification import NotificationPreference


class ChannelType(str, Enum):
    """Available notification channel types."""

    IN_APP = "in_app"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"


class DeliveryStatus(str, Enum):
    """Delivery status for a channel."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NotificationPayload:
    """Payload for sending a notification.

    Contains all information needed to send a notification
    through any channel.

    Attributes:
        notification_type: Type of notification (maps to alert_type).
        title: Notification title.
        message: Notification message body.
        recipient_id: User ID of the recipient.
        recipient_email: Email address (for email channel).
        recipient_language: Preferred language code.
        recipient_timezone: User's timezone.
        student_id: Student this notification is about.
        student_name: Student's display name.
        alert_id: Related alert ID (if applicable).
        data: Additional data for the notification.
        action_url: URL to open when notification is clicked.
        action_label: Label for the action button.
        push_tokens: List of push tokens for push channel.
        priority: Notification priority (low, normal, high).
    """

    notification_type: str
    title: str
    message: str
    recipient_id: UUID
    recipient_email: str
    recipient_language: str = "tr"
    recipient_timezone: str = "Europe/Istanbul"
    student_id: UUID | None = None
    student_name: str | None = None
    alert_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    action_url: str | None = None
    action_label: str | None = None
    push_tokens: list[dict[str, str]] = field(default_factory=list)
    priority: str = "normal"


@dataclass
class ChannelResult:
    """Result of a channel send operation.

    Attributes:
        channel: Which channel was used.
        status: Delivery status.
        message_id: External message ID (if available).
        error_message: Error message if failed.
        sent_at: When message was sent.
        metadata: Additional result metadata.
    """

    channel: ChannelType
    status: DeliveryStatus
    message_id: str | None = None
    error_message: str | None = None
    sent_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary representation for JSONB storage.
        """
        return {
            "channel": self.channel.value,
            "status": self.status.value,
            "message_id": self.message_id,
            "error_message": self.error_message,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "metadata": self.metadata,
        }


class BaseChannel(ABC):
    """Abstract base class for notification channels.

    Each channel implementation handles delivery through
    a specific medium. Channels must implement the send
    method and handle their own error recovery.

    Attributes:
        channel_type: The type of this channel.
    """

    def __init__(self) -> None:
        """Initialize the channel."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type."""
        ...

    @abstractmethod
    async def send(self, payload: NotificationPayload) -> ChannelResult:
        """Send a notification through this channel.

        Args:
            payload: The notification payload to send.

        Returns:
            ChannelResult with delivery status.
        """
        ...

    def is_enabled_for_preference(
        self, preference: NotificationPreference | None
    ) -> bool:
        """Check if this channel is enabled in user preferences.

        Args:
            preference: User's notification preference.

        Returns:
            True if channel is enabled.
        """
        if preference is None:
            # Default: in_app and push enabled, email disabled
            return self.channel_type in (ChannelType.IN_APP, ChannelType.PUSH)

        if not preference.is_enabled:
            return False

        channel_map = {
            ChannelType.IN_APP: preference.in_app,
            ChannelType.PUSH: preference.push,
            ChannelType.EMAIL: preference.email,
            ChannelType.SMS: preference.sms,
        }
        return channel_map.get(self.channel_type, False)

    def is_in_quiet_hours(
        self,
        preference: NotificationPreference | None,
        current_time: time,
    ) -> bool:
        """Check if current time is within quiet hours.

        Args:
            preference: User's notification preference.
            current_time: Current time in user's timezone.

        Returns:
            True if in quiet hours (should not send).
        """
        if preference is None:
            return False

        if preference.quiet_start is None or preference.quiet_end is None:
            return False

        quiet_start = preference.quiet_start
        quiet_end = preference.quiet_end

        # Handle overnight quiet hours (e.g., 22:00 to 07:00)
        if quiet_start > quiet_end:
            # Quiet hours span midnight
            return current_time >= quiet_start or current_time <= quiet_end
        else:
            # Normal range
            return quiet_start <= current_time <= quiet_end

    def create_success_result(
        self,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelResult:
        """Create a successful channel result.

        Args:
            message_id: External message ID.
            metadata: Additional metadata.

        Returns:
            ChannelResult with SENT status.
        """
        return ChannelResult(
            channel=self.channel_type,
            status=DeliveryStatus.SENT,
            message_id=message_id,
            sent_at=datetime.now(),
            metadata=metadata or {},
        )

    def create_failure_result(
        self,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelResult:
        """Create a failed channel result.

        Args:
            error_message: Error description.
            metadata: Additional metadata.

        Returns:
            ChannelResult with FAILED status.
        """
        return ChannelResult(
            channel=self.channel_type,
            status=DeliveryStatus.FAILED,
            error_message=error_message,
            sent_at=datetime.now(),
            metadata=metadata or {},
        )

    def create_skipped_result(
        self,
        reason: str,
    ) -> ChannelResult:
        """Create a skipped channel result.

        Args:
            reason: Why the send was skipped.

        Returns:
            ChannelResult with SKIPPED status.
        """
        return ChannelResult(
            channel=self.channel_type,
            status=DeliveryStatus.SKIPPED,
            error_message=reason,
            sent_at=datetime.now(),
        )
