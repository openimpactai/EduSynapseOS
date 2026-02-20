# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Push notification channel using Firebase Cloud Messaging.

This channel sends push notifications to mobile devices and browsers
using the FCM HTTP v1 API. It requires valid Firebase service account
credentials to be configured.

Configuration (via environment variables):
- FIREBASE_CREDENTIALS_PATH: Path to service account JSON file
- FIREBASE_PROJECT_ID: Firebase project ID
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.infrastructure.notifications.channels.base import (
    BaseChannel,
    ChannelResult,
    ChannelType,
    NotificationPayload,
)

logger = logging.getLogger(__name__)

# FCM HTTP v1 API endpoint template
FCM_API_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"


class PushChannel(BaseChannel):
    """Push notification channel using Firebase Cloud Messaging.

    Sends push notifications via FCM HTTP v1 API. Requires:
    - google-auth library for OAuth2 token generation
    - httpx library for async HTTP requests
    - Valid Firebase service account credentials

    Push tokens are expected in the payload.push_tokens field as:
    [{"platform": "android|ios|web", "token": "device_token"}]
    """

    # FCM priority mapping
    PRIORITY_MAP = {
        "low": "normal",
        "normal": "high",
        "high": "high",
    }

    def __init__(self) -> None:
        """Initialize the push channel."""
        super().__init__()
        self._credentials = None
        self._project_id: str | None = None
        self._initialized = False
        self._init_error: str | None = None

    @property
    def channel_type(self) -> ChannelType:
        """Return the channel type."""
        return ChannelType.PUSH

    async def _ensure_initialized(self) -> bool:
        """Ensure Firebase credentials are loaded.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if self._init_error:
            return False

        try:
            # Import here to avoid startup failures if not configured
            from google.oauth2 import service_account

            credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
            project_id = os.environ.get("FIREBASE_PROJECT_ID")

            if not credentials_path or not project_id:
                self._init_error = "Firebase credentials not configured"
                self.logger.warning(
                    "Push notifications disabled: FIREBASE_CREDENTIALS_PATH or "
                    "FIREBASE_PROJECT_ID not set"
                )
                return False

            if not os.path.exists(credentials_path):
                self._init_error = f"Credentials file not found: {credentials_path}"
                self.logger.error(self._init_error)
                return False

            # Load service account credentials with FCM scope
            self._credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/firebase.messaging"],
            )
            self._project_id = project_id
            self._initialized = True

            self.logger.info("FCM push channel initialized for project %s", project_id)
            return True

        except ImportError:
            self._init_error = "google-auth library not installed"
            self.logger.warning(
                "Push notifications disabled: google-auth library not installed"
            )
            return False

        except Exception as e:
            self._init_error = f"Failed to initialize: {str(e)}"
            self.logger.error(self._init_error, exc_info=True)
            return False

    async def _get_access_token(self) -> str | None:
        """Get OAuth2 access token for FCM API.

        Returns:
            Access token string or None if failed.
        """
        if not self._credentials:
            return None

        try:
            # Refresh token if needed (runs in thread pool for sync operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._credentials.refresh,
                self._get_auth_request(),
            )
            return self._credentials.token

        except Exception as e:
            self.logger.error("Failed to get FCM access token: %s", str(e))
            return None

    def _get_auth_request(self) -> Any:
        """Get auth request object for token refresh.

        Returns:
            google.auth.transport.requests.Request instance.
        """
        from google.auth.transport.requests import Request
        return Request()

    async def send(self, payload: NotificationPayload) -> ChannelResult:
        """Send push notification via FCM.

        Args:
            payload: The notification payload.

        Returns:
            ChannelResult with delivery status.
        """
        # Check initialization
        if not await self._ensure_initialized():
            return self.create_skipped_result(
                self._init_error or "Push channel not configured"
            )

        # Check for push tokens
        if not payload.push_tokens:
            return self.create_skipped_result("No push tokens available")

        # Get access token
        access_token = await self._get_access_token()
        if not access_token:
            return self.create_failure_result("Failed to obtain access token")

        # Send to each token
        results: list[dict[str, Any]] = []
        success_count = 0
        failure_count = 0

        for token_info in payload.push_tokens:
            token = token_info.get("token")
            platform = token_info.get("platform", "android")

            if not token:
                continue

            result = await self._send_to_token(
                token=token,
                platform=platform,
                payload=payload,
                access_token=access_token,
            )
            results.append(result)

            if result.get("success"):
                success_count += 1
            else:
                failure_count += 1

        # Determine overall status
        if success_count == 0 and failure_count > 0:
            return self.create_failure_result(
                f"All {failure_count} push notifications failed",
                metadata={"results": results},
            )

        return self.create_success_result(
            message_id=results[0].get("message_id") if results else None,
            metadata={
                "success_count": success_count,
                "failure_count": failure_count,
                "results": results,
            },
        )

    async def _send_to_token(
        self,
        token: str,
        platform: str,
        payload: NotificationPayload,
        access_token: str,
    ) -> dict[str, Any]:
        """Send notification to a single device token.

        Args:
            token: Device push token.
            platform: Platform (android, ios, web).
            payload: Notification payload.
            access_token: OAuth2 access token.

        Returns:
            Result dictionary with success status.
        """
        try:
            import httpx

            # Build FCM message
            fcm_message = self._build_fcm_message(token, platform, payload)

            # Send request
            url = FCM_API_URL.format(project_id=self._project_id)
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"message": fcm_message},
                )

            if response.status_code == 200:
                result_data = response.json()
                message_id = result_data.get("name", "").split("/")[-1]
                self.logger.debug(
                    "Push sent successfully to %s...: %s",
                    token[:20],
                    message_id,
                )
                return {
                    "success": True,
                    "token": token[:20] + "...",
                    "platform": platform,
                    "message_id": message_id,
                }
            else:
                error_msg = response.text
                self.logger.warning(
                    "FCM request failed (%d): %s",
                    response.status_code,
                    error_msg,
                )
                return {
                    "success": False,
                    "token": token[:20] + "...",
                    "platform": platform,
                    "error": error_msg,
                    "status_code": response.status_code,
                }

        except ImportError:
            return {
                "success": False,
                "token": token[:20] + "...",
                "error": "httpx library not installed",
            }

        except Exception as e:
            self.logger.error("Failed to send push to token: %s", str(e))
            return {
                "success": False,
                "token": token[:20] + "...",
                "platform": platform,
                "error": str(e),
            }

    def _build_fcm_message(
        self,
        token: str,
        platform: str,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Build FCM message structure.

        Args:
            token: Device token.
            platform: Platform type.
            payload: Notification payload.

        Returns:
            FCM message dictionary.
        """
        # Base notification content
        notification = {
            "title": payload.title,
            "body": payload.message,
        }

        # Build data payload
        data = {
            "notification_type": payload.notification_type,
            "click_action": payload.action_url or "",
        }
        if payload.student_id:
            data["student_id"] = str(payload.student_id)
        if payload.alert_id:
            data["alert_id"] = payload.alert_id

        # Base message structure
        message: dict[str, Any] = {
            "token": token,
            "notification": notification,
            "data": data,
        }

        # Platform-specific configuration
        priority = self.PRIORITY_MAP.get(payload.priority, "high")

        if platform == "android":
            message["android"] = {
                "priority": priority,
                "notification": {
                    "click_action": "FLUTTER_NOTIFICATION_CLICK",
                    "channel_id": "edusynapse_alerts",
                },
            }
        elif platform == "ios":
            message["apns"] = {
                "headers": {
                    "apns-priority": "10" if priority == "high" else "5",
                },
                "payload": {
                    "aps": {
                        "sound": "default",
                        "badge": 1,
                    },
                },
            }
        elif platform == "web":
            message["webpush"] = {
                "headers": {
                    "Urgency": priority,
                },
                "notification": {
                    "icon": "/icons/notification-icon.png",
                },
                "fcm_options": {
                    "link": payload.action_url or "/",
                },
            }

        return message
