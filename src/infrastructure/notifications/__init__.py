# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Notification System for EduSynapseOS.

This package provides a multi-channel notification system that
delivers alerts and messages to users through various channels:
- In-app notifications (database records)
- Push notifications (Firebase Cloud Messaging)
- Email notifications (SMTP)

Integration with Proactive Intelligence (Phase 17):
The NotificationService.notify_from_alert() method is called by
ProactiveService after storing an alert, which triggers notifications
to relevant stakeholders (teachers, parents) based on alert targets.

Key Components:
- NotificationService: Main orchestrator for notification delivery
- Channels: InAppChannel, PushChannel, EmailChannel
- NotificationPayload: Data structure for notification content

Usage:
    from src.infrastructure.notifications import (
        NotificationService,
        get_notification_service,
    )

    # Get service (typically at startup)
    service = get_notification_service(tenant_db_manager)

    # Notify from alert (called by ProactiveService)
    result = await service.notify_from_alert(
        tenant_code="acme",
        alert_data=alert_data,
    )

    # Send direct notification
    results = await service.send_direct(
        tenant_code="acme",
        user_id=user_uuid,
        notification_type="system_message",
        title="Welcome!",
        message="Welcome to EduSynapseOS.",
    )

Configuration (environment variables):
- FIREBASE_CREDENTIALS_PATH: Path to Firebase service account JSON
- FIREBASE_PROJECT_ID: Firebase project ID
- SMTP_HOST: SMTP server hostname
- SMTP_PORT: SMTP server port (default: 587)
- SMTP_USERNAME: SMTP authentication username
- SMTP_PASSWORD: SMTP authentication password
- SMTP_FROM_EMAIL: Sender email address
- SMTP_FROM_NAME: Sender display name
"""

from src.infrastructure.notifications.channels import (
    BaseChannel,
    ChannelResult,
    ChannelType,
    DeliveryStatus,
    EmailChannel,
    InAppChannel,
    NotificationPayload,
    PushChannel,
)
from src.infrastructure.notifications.service import (
    NotificationResult,
    NotificationService,
    Recipient,
    get_notification_service,
)

__all__ = [
    # Service
    "NotificationService",
    "NotificationResult",
    "Recipient",
    "get_notification_service",
    # Channel types
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
