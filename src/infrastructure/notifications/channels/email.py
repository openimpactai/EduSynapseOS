# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Email notification channel using async SMTP.

This channel sends email notifications using aiosmtplib for
async SMTP communication. It supports both plain text and
HTML email formats.

Configuration (via environment variables):
- SMTP_HOST: SMTP server hostname
- SMTP_PORT: SMTP server port (default: 587)
- SMTP_USERNAME: SMTP authentication username
- SMTP_PASSWORD: SMTP authentication password
- SMTP_USE_TLS: Use STARTTLS (default: true)
- SMTP_FROM_EMAIL: Sender email address
- SMTP_FROM_NAME: Sender display name
"""

import logging
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from src.infrastructure.notifications.channels.base import (
    BaseChannel,
    ChannelResult,
    ChannelType,
    NotificationPayload,
)

logger = logging.getLogger(__name__)


class EmailChannel(BaseChannel):
    """Email notification channel using async SMTP.

    Sends email notifications via SMTP. Requires aiosmtplib library
    and valid SMTP server credentials to be configured.

    The channel generates both plain text and HTML versions of
    the email for maximum compatibility.
    """

    def __init__(self) -> None:
        """Initialize the email channel."""
        super().__init__()
        self._config: dict[str, Any] | None = None
        self._init_error: str | None = None

    @property
    def channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.EMAIL

    def _load_config(self) -> dict[str, Any] | None:
        """Load SMTP configuration from environment.

        Returns:
            Configuration dictionary or None if not configured.
        """
        if self._config is not None:
            return self._config

        if self._init_error:
            return None

        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = os.environ.get("SMTP_PORT", "587")
        smtp_username = os.environ.get("SMTP_USERNAME")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        smtp_from_email = os.environ.get("SMTP_FROM_EMAIL")

        if not all([smtp_host, smtp_username, smtp_password, smtp_from_email]):
            self._init_error = "SMTP configuration incomplete"
            self.logger.warning(
                "Email notifications disabled: SMTP_HOST, SMTP_USERNAME, "
                "SMTP_PASSWORD, or SMTP_FROM_EMAIL not set"
            )
            return None

        self._config = {
            "host": smtp_host,
            "port": int(smtp_port),
            "username": smtp_username,
            "password": smtp_password,
            "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
            "from_email": smtp_from_email,
            "from_name": os.environ.get("SMTP_FROM_NAME", "EduSynapseOS"),
        }

        self.logger.info("Email channel configured with host %s", smtp_host)
        return self._config

    async def send(self, payload: NotificationPayload) -> ChannelResult:
        """Send email notification via SMTP.

        Args:
            payload: The notification payload.

        Returns:
            ChannelResult with delivery status.
        """
        config = self._load_config()
        if not config:
            return self.create_skipped_result(
                self._init_error or "Email channel not configured"
            )

        if not payload.recipient_email:
            return self.create_skipped_result("No recipient email address")

        try:
            import aiosmtplib

            # Build email message
            message = self._build_email_message(payload, config)

            # Send via SMTP
            await aiosmtplib.send(
                message,
                hostname=config["host"],
                port=config["port"],
                username=config["username"],
                password=config["password"],
                start_tls=config["use_tls"],
            )

            self.logger.info(
                "Email sent to %s: %s",
                payload.recipient_email,
                payload.title,
            )

            return self.create_success_result(
                message_id=message["Message-ID"],
                metadata={"recipient": payload.recipient_email},
            )

        except ImportError:
            return self.create_skipped_result("aiosmtplib library not installed")

        except Exception as e:
            self.logger.error(
                "Failed to send email to %s: %s",
                payload.recipient_email,
                str(e),
                exc_info=True,
            )
            return self.create_failure_result(
                f"SMTP error: {str(e)}",
                metadata={"recipient": payload.recipient_email},
            )

    def _build_email_message(
        self,
        payload: NotificationPayload,
        config: dict[str, Any],
    ) -> MIMEMultipart:
        """Build MIME email message.

        Args:
            payload: Notification payload.
            config: SMTP configuration.

        Returns:
            MIMEMultipart message ready to send.
        """
        message = MIMEMultipart("alternative")

        # Headers
        from_addr = f"{config['from_name']} <{config['from_email']}>"
        message["From"] = from_addr
        message["To"] = payload.recipient_email
        message["Subject"] = payload.title

        # Plain text version
        text_content = self._build_plain_text(payload)
        message.attach(MIMEText(text_content, "plain", "utf-8"))

        # HTML version
        html_content = self._build_html(payload)
        message.attach(MIMEText(html_content, "html", "utf-8"))

        return message

    def _build_plain_text(self, payload: NotificationPayload) -> str:
        """Build plain text email content.

        Args:
            payload: Notification payload.

        Returns:
            Plain text email body.
        """
        lines = [
            payload.title,
            "=" * len(payload.title),
            "",
            payload.message,
            "",
        ]

        if payload.student_name:
            lines.append(f"Student: {payload.student_name}")
            lines.append("")

        if payload.action_url:
            action_text = payload.action_label or "View Details"
            lines.append(f"{action_text}: {payload.action_url}")
            lines.append("")

        lines.extend([
            "---",
            "This notification was sent by EduSynapseOS.",
            "To manage your notification preferences, visit your account settings.",
        ])

        return "\n".join(lines)

    def _build_html(self, payload: NotificationPayload) -> str:
        """Build HTML email content.

        Args:
            payload: Notification payload.

        Returns:
            HTML email body.
        """
        # Escape HTML special characters
        def escape(text: str) -> str:
            return (
                text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        title = escape(payload.title)
        message = escape(payload.message).replace("\n", "<br>")
        student_name = escape(payload.student_name) if payload.student_name else None

        # Build action button if URL provided
        action_button = ""
        if payload.action_url:
            action_label = escape(payload.action_label or "View Details")
            action_button = f"""
            <div style="margin: 24px 0;">
                <a href="{payload.action_url}"
                   style="background-color: #4F46E5; color: white;
                          padding: 12px 24px; text-decoration: none;
                          border-radius: 6px; font-weight: 500;">
                    {action_label}
                </a>
            </div>
            """

        # Build student info if available
        student_info = ""
        if student_name:
            student_info = f"""
            <p style="color: #6B7280; margin: 16px 0 0 0;">
                <strong>Student:</strong> {student_name}
            </p>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
             'Helvetica Neue', Arial, sans-serif; line-height: 1.6;
             color: #1F2937; margin: 0; padding: 0; background-color: #F3F4F6;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: white; border-radius: 8px;
                    padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">

            <!-- Header -->
            <div style="border-bottom: 1px solid #E5E7EB; padding-bottom: 16px;
                        margin-bottom: 24px;">
                <h1 style="color: #4F46E5; font-size: 24px; margin: 0;">
                    {title}
                </h1>
            </div>

            <!-- Content -->
            <div style="font-size: 16px; color: #374151;">
                <p style="margin: 0 0 16px 0;">{message}</p>
                {student_info}
            </div>

            <!-- Action Button -->
            {action_button}

            <!-- Footer -->
            <div style="border-top: 1px solid #E5E7EB; padding-top: 16px;
                        margin-top: 24px; font-size: 12px; color: #9CA3AF;">
                <p style="margin: 0;">
                    This notification was sent by EduSynapseOS.<br>
                    To manage your notification preferences, visit your account settings.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return html.strip()
