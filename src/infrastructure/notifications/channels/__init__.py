# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Notification channels for delivering notifications.

This package provides channel implementations for sending
notifications through various delivery mechanisms:

- InAppChannel: Creates notification records in the database
- PushChannel: Sends push notifications via Firebase Cloud Messaging
- EmailChannel: Sends email notifications via SMTP

Usage:
    from src.infrastructure.notifications.channels import (
        InAppChannel,
        PushChannel,
        EmailChannel,
        ChannelType,
        NotificationPayload,
    )

    # Create channels
    in_app = InAppChannel()
    push = PushChannel()
    email = EmailChannel()

    # Send notification
    payload = NotificationPayload(
        notification_type="alert_struggle",
        title="Attention Needed",
        message="Your student needs help",
        recipient_id=user_uuid,
        recipient_email="parent@example.com",
    )

    result = await in_app.send(payload)
"""

from src.infrastructure.notifications.channels.base import (
    BaseChannel,
    ChannelResult,
    ChannelType,
    DeliveryStatus,
    NotificationPayload,
)
from src.infrastructure.notifications.channels.email import EmailChannel
from src.infrastructure.notifications.channels.in_app import InAppChannel
from src.infrastructure.notifications.channels.push import PushChannel

__all__ = [
    # Base types
    "BaseChannel",
    "ChannelResult",
    "ChannelType",
    "DeliveryStatus",
    "NotificationPayload",
    # Channels
    "EmailChannel",
    "InAppChannel",
    "PushChannel",
]
