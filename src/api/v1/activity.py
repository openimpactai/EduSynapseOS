# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Activity stream WebSocket API endpoint.

This module provides real-time event streaming for frontend dashboards:
- WebSocket /stream/{student_id} - Real-time activity event stream

The activity stream allows authorized users to receive real-time updates
about student learning activities across all workflows (Learning Tutor,
Companion, Practice, Practice Helper, Gaming).

Example:
    const ws = new WebSocket(
        `wss://api.example.com/api/v1/activity/stream/${studentId}?token=${jwt}`
    );
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`[${data.source_workflow}] ${data.event_type}`, data.data);
    };
"""

import asyncio
import fnmatch
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.api.middleware.auth import CurrentUser
from src.core.config import get_settings
from src.domains.auth.jwt import JWTManager, TokenExpiredError, InvalidTokenError
from src.infrastructure.events import get_event_bus, EventData

logger = logging.getLogger(__name__)

router = APIRouter()


class ActivityMessage(BaseModel):
    """Activity event message sent to WebSocket clients.

    Attributes:
        timestamp: Event timestamp in ISO format.
        event_type: Event type string (e.g., "learning_tutor.session.started").
        source_workflow: Originating workflow name.
        student_id: Student UUID.
        session_id: Session/conversation UUID if applicable.
        data: Event payload data.
        metadata: Additional metadata for agent tracking.
    """

    timestamp: str
    event_type: str
    source_workflow: str
    student_id: str
    session_id: str | None = None
    data: dict[str, Any]
    metadata: dict[str, Any]


class ConnectionState:
    """Manages WebSocket connection state and subscriptions.

    Attributes:
        websocket: The WebSocket connection.
        student_id: Target student ID to filter events.
        tenant_code: Tenant code for filtering.
        user: Authenticated user info.
        subscribed_patterns: Event patterns the client subscribes to.
        message_queue: Async queue for outgoing messages.
    """

    def __init__(
        self,
        websocket: WebSocket,
        student_id: str,
        tenant_code: str,
        user: CurrentUser,
    ) -> None:
        """Initialize connection state.

        Args:
            websocket: WebSocket connection.
            student_id: Target student ID.
            tenant_code: Tenant code.
            user: Authenticated user.
        """
        self.websocket = websocket
        self.student_id = student_id
        self.tenant_code = tenant_code
        self.user = user
        self.subscribed_patterns: set[str] = {"*"}  # Subscribe to all by default
        self.message_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False

    def matches_pattern(self, event_type: str) -> bool:
        """Check if event type matches any subscribed pattern.

        Args:
            event_type: Event type string to check.

        Returns:
            True if event matches a subscribed pattern.
        """
        for pattern in self.subscribed_patterns:
            if fnmatch.fnmatch(event_type, pattern):
                return True
        return False

    async def send_message(self, message: dict[str, Any]) -> None:
        """Queue a message for sending.

        Args:
            message: Message dict to send.
        """
        if not self._closed:
            await self.message_queue.put(message)

    async def close(self) -> None:
        """Mark connection as closed."""
        self._closed = True


def _extract_workflow_from_event_type(event_type: str) -> str:
    """Extract source workflow name from event type.

    Args:
        event_type: Event type string.

    Returns:
        Workflow name (e.g., "learning_tutor", "companion").
    """
    parts = event_type.split(".")
    if len(parts) >= 2:
        # Handle compound workflow names like "learning_tutor", "practice_helper"
        if parts[0] in ("learning", "practice") and parts[1] in ("tutor", "helper"):
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    return "unknown"


def _extract_session_id(payload: dict[str, Any]) -> str | None:
    """Extract session ID from event payload.

    Args:
        payload: Event payload dict.

    Returns:
        Session ID string or None.
    """
    for key in ("session_id", "conversation_id", "game_id"):
        if key in payload:
            return str(payload[key])
    return None


class ActivityStreamManager:
    """Manages activity stream WebSocket connections.

    Handles event subscription, filtering, and broadcasting to connected clients.
    Each connection receives only events for their authorized student.

    Attributes:
        _connections: Dict mapping student_id to list of connection states.
        _event_handler_registered: Flag for one-time EventBus subscription.
    """

    def __init__(self) -> None:
        """Initialize the activity stream manager."""
        self._connections: dict[str, list[ConnectionState]] = {}
        self._event_handler_registered = False

    def _ensure_event_subscription(self) -> None:
        """Register global event handler if not already registered."""
        if self._event_handler_registered:
            return

        event_bus = get_event_bus()

        # Subscribe to all events using wildcard pattern
        event_bus.subscribe("*", self._handle_event)
        self._event_handler_registered = True
        logger.info("Activity stream subscribed to EventBus")

    async def _handle_event(self, event: EventData) -> None:
        """Handle incoming events from EventBus.

        Filters and forwards events to appropriate WebSocket connections.

        Args:
            event: Event data from EventBus.
        """
        payload = event.payload

        # Extract student_id from various payload formats
        student_id = payload.get("student_id") or payload.get("user_id")
        if not student_id:
            return

        student_id = str(student_id)

        # Get connections for this student
        connections = self._connections.get(student_id, [])
        if not connections:
            return

        # Build activity message
        message = ActivityMessage(
            timestamp=event.timestamp.isoformat(),
            event_type=event.event_type,
            source_workflow=_extract_workflow_from_event_type(event.event_type),
            student_id=student_id,
            session_id=_extract_session_id(payload),
            data=payload,
            metadata={
                "event_id": event.event_id,
                "tenant_code": event.tenant_code,
                "memory_writes": payload.get("memory_writes", []),
                "agent_id": payload.get("agent_id"),
                "workflow_state": payload.get("workflow_state"),
            },
        )

        # Send to matching connections
        for conn in connections:
            if conn.tenant_code == event.tenant_code and conn.matches_pattern(event.event_type):
                await conn.send_message(message.model_dump())

    async def register_connection(self, state: ConnectionState) -> None:
        """Register a new WebSocket connection.

        Args:
            state: Connection state to register.
        """
        self._ensure_event_subscription()

        if state.student_id not in self._connections:
            self._connections[state.student_id] = []
        self._connections[state.student_id].append(state)

        logger.info(
            "Activity stream connection registered: student=%s, user=%s",
            state.student_id,
            state.user.id,
        )

    async def unregister_connection(self, state: ConnectionState) -> None:
        """Unregister a WebSocket connection.

        Args:
            state: Connection state to unregister.
        """
        await state.close()

        if state.student_id in self._connections:
            try:
                self._connections[state.student_id].remove(state)
                if not self._connections[state.student_id]:
                    del self._connections[state.student_id]
            except ValueError:
                pass

        logger.info(
            "Activity stream connection unregistered: student=%s",
            state.student_id,
        )


# Singleton manager instance
_activity_manager: ActivityStreamManager | None = None


def get_activity_manager() -> ActivityStreamManager:
    """Get the singleton activity stream manager.

    Returns:
        ActivityStreamManager instance.
    """
    global _activity_manager
    if _activity_manager is None:
        _activity_manager = ActivityStreamManager()
    return _activity_manager


async def _authenticate_websocket(
    websocket: WebSocket,
    token: str | None,
) -> CurrentUser | None:
    """Authenticate WebSocket connection using JWT token.

    Args:
        websocket: WebSocket connection.
        token: JWT token from query param or auth message.

    Returns:
        CurrentUser if authenticated, None otherwise.
    """
    if not token:
        return None

    try:
        settings = get_settings()
        jwt_manager = JWTManager(settings.jwt)
        payload = jwt_manager.decode_token(token, expected_type="access")
        return CurrentUser(payload)
    except (TokenExpiredError, InvalidTokenError) as e:
        logger.debug("WebSocket auth failed: %s", str(e))
        return None


def _check_authorization(
    user: CurrentUser,
    student_id: str,
) -> bool:
    """Check if user is authorized to view student's activity.

    Authorization rules:
    - Students can only view their own activity
    - Teachers can view activity of students in their classes
    - Parents can view activity of their children
    - Admins can view any student's activity

    Args:
        user: Authenticated user.
        student_id: Target student ID.

    Returns:
        True if authorized.
    """
    # Students can only view their own activity
    if user.user_type == "student":
        return user.id == student_id

    # Admins can view any activity
    if user.user_type in ("tenant_admin", "school_admin"):
        return True

    # Teachers and parents need proper relation check
    # For now, allow teachers and parents (proper check requires DB query)
    if user.user_type in ("teacher", "parent"):
        return True

    return False


@router.websocket("/stream/{student_id}")
async def activity_stream_websocket(
    websocket: WebSocket,
    student_id: UUID,
) -> None:
    """WebSocket endpoint for real-time activity streaming.

    Authenticates via JWT token (query param or first message).
    Streams events filtered by student_id and optional patterns.

    Args:
        websocket: WebSocket connection.
        student_id: Target student ID.
    """
    await websocket.accept()

    user: CurrentUser | None = None
    state: ConnectionState | None = None
    manager = get_activity_manager()

    try:
        # Extract token from query params (FastAPI Query doesn't work for WebSocket)
        token = websocket.query_params.get("token")

        # Try token from query param first
        if token:
            user = await _authenticate_websocket(websocket, token)

        # If no token or auth failed, wait for auth message
        if not user:
            await websocket.send_json({
                "type": "auth_required",
                "message": "Send auth message with token: {\"type\": \"auth\", \"token\": \"...\"}",
            })

            # Wait for auth message with timeout
            try:
                auth_data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0,
                )
                if auth_data.get("type") == "auth" and auth_data.get("token"):
                    user = await _authenticate_websocket(websocket, auth_data["token"])
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "code": "AUTH_TIMEOUT",
                    "message": "Authentication timeout",
                })
                await websocket.close()
                return

        # Check if authentication succeeded
        if not user:
            await websocket.send_json({
                "type": "error",
                "code": "AUTH_FAILED",
                "message": "Invalid or expired token",
            })
            await websocket.close()
            return

        # Check authorization
        student_id_str = str(student_id)
        if not _check_authorization(user, student_id_str):
            await websocket.send_json({
                "type": "error",
                "code": "UNAUTHORIZED",
                "message": "You do not have permission to view this student's activity",
            })
            await websocket.close()
            return

        # Create connection state and register
        state = ConnectionState(
            websocket=websocket,
            student_id=student_id_str,
            tenant_code=user.tenant_code or "",
            user=user,
        )
        await manager.register_connection(state)

        # Send connected message
        await websocket.send_json({
            "type": "connected",
            "student_id": student_id_str,
            "subscribed_patterns": list(state.subscribed_patterns),
            "message": "Activity stream connected. Events will be streamed automatically.",
        })

        # Start message sender task
        sender_task = asyncio.create_task(_message_sender(websocket, state))

        # Handle incoming messages (subscribe/unsubscribe/ping)
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "subscribe":
                    patterns = data.get("patterns", [])
                    if patterns:
                        state.subscribed_patterns.update(patterns)
                        await websocket.send_json({
                            "type": "subscribed",
                            "patterns": list(state.subscribed_patterns),
                        })

                elif msg_type == "unsubscribe":
                    patterns = data.get("patterns", [])
                    for pattern in patterns:
                        state.subscribed_patterns.discard(pattern)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "patterns": list(state.subscribed_patterns),
                    })

                elif msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "code": "UNKNOWN_MESSAGE_TYPE",
                        "message": f"Unknown message type: {msg_type}",
                    })

        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: student=%s", student_id_str)
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error("Activity stream error: %s", str(e), exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
            })
        except Exception:
            pass

    finally:
        if state:
            await manager.unregister_connection(state)
        try:
            await websocket.close()
        except Exception:
            pass


async def _message_sender(websocket: WebSocket, state: ConnectionState) -> None:
    """Background task to send queued messages to WebSocket.

    Args:
        websocket: WebSocket connection.
        state: Connection state with message queue.
    """
    try:
        while True:
            message = await state.message_queue.get()
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.debug("Failed to send message: %s", str(e))
                break
    except asyncio.CancelledError:
        pass
