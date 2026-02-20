# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tenant middleware for multi-tenant background processing.

Propagates tenant context through Dramatiq message processing,
ensuring each task runs in the correct tenant context.
"""

import contextvars
import logging
from typing import Any

import dramatiq
from dramatiq import Message, Middleware

logger = logging.getLogger(__name__)


# Context variable for tenant code
_tenant_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_code", default=None
)


def get_current_tenant() -> str | None:
    """Get the current tenant code from context.

    Returns:
        Current tenant code or None.
    """
    return _tenant_context.get()


def set_current_tenant(tenant_code: str | None) -> contextvars.Token[str | None]:
    """Set the current tenant code in context.

    Args:
        tenant_code: Tenant code to set.

    Returns:
        Context token for resetting.
    """
    return _tenant_context.set(tenant_code)


class TenantMiddleware(Middleware):
    """Middleware for propagating tenant context in Dramatiq tasks.

    Automatically includes tenant_code in message options when sending,
    and restores tenant context when processing tasks.

    Usage:
        # In task definition
        @dramatiq.actor
        def my_task(tenant_code: str, ...):
            # tenant_code is automatically set in context
            current = get_current_tenant()  # Returns tenant_code

        # Calling the task
        my_task.send("test_school", ...)
    """

    TENANT_KEY = "tenant_code"

    def before_enqueue(
        self,
        broker: dramatiq.Broker,
        message: Message,
        delay: int | None,
    ) -> None:
        """Add tenant code to message before enqueueing.

        Args:
            broker: The broker instance.
            message: The message being enqueued.
            delay: Optional delay in milliseconds.
        """
        tenant_code = get_current_tenant()

        if tenant_code and self.TENANT_KEY not in message.options:
            message.options[self.TENANT_KEY] = tenant_code
            logger.debug(
                "Added tenant context to message: %s (tenant: %s)",
                message.message_id,
                tenant_code,
            )

    def before_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
    ) -> None:
        """Restore tenant context before processing message.

        Args:
            broker: The broker instance.
            message: The message being processed.
        """
        tenant_code = message.options.get(self.TENANT_KEY)

        if tenant_code:
            set_current_tenant(tenant_code)
            logger.debug(
                "Restored tenant context: %s (message: %s)",
                tenant_code,
                message.message_id,
            )

    def after_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
        *,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        """Clear tenant context after processing message.

        Args:
            broker: The broker instance.
            message: The processed message.
            result: The result of processing.
            exception: Any exception that occurred.
        """
        set_current_tenant(None)

    def after_skip_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
    ) -> None:
        """Clear tenant context after skipping message.

        Args:
            broker: The broker instance.
            message: The skipped message.
        """
        set_current_tenant(None)
