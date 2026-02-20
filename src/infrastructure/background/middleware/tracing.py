# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""OpenTelemetry tracing middleware for Dramatiq.

Provides distributed tracing for background tasks, enabling
visibility into task execution across services.
"""

import logging
from typing import Any

import dramatiq
from dramatiq import Message, Middleware

logger = logging.getLogger(__name__)

# OpenTelemetry imports (optional - graceful degradation if not installed)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
    tracer = trace.get_tracer(__name__)
    propagator = TraceContextTextMapPropagator()
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None  # type: ignore[assignment]
    propagator = None  # type: ignore[assignment]


class TracingMiddleware(Middleware):
    """Middleware that adds OpenTelemetry tracing to Dramatiq actors.

    Creates spans for each message processing, propagating trace context
    from the enqueuer to the worker.

    Features:
    - Automatic span creation for each task
    - Context propagation across services
    - Error recording with stack traces
    - Custom attributes for tenant, actor, queue

    Usage:
        from src.infrastructure.background.middleware.tracing import TracingMiddleware

        broker.add_middleware(TracingMiddleware())

    Note:
        Requires OpenTelemetry SDK to be configured. If not available,
        middleware operates as no-op.
    """

    TRACE_CONTEXT_KEY = "trace_context"
    SPAN_KEY = "_otel_span"

    def __init__(self, service_name: str = "edusynapse-worker") -> None:
        """Initialize tracing middleware.

        Args:
            service_name: Name of the service for spans.
        """
        self.service_name = service_name

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available. Tracing middleware will be no-op. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

    def before_enqueue(
        self,
        broker: dramatiq.Broker,
        message: Message,
        delay: int | None,
    ) -> None:
        """Inject trace context into message before enqueueing.

        Args:
            broker: Dramatiq broker.
            message: Message being enqueued.
            delay: Optional delay in milliseconds.
        """
        if not OTEL_AVAILABLE or propagator is None:
            return

        # Get current trace context and inject into carrier
        carrier: dict[str, str] = {}
        propagator.inject(carrier)

        if carrier:
            message.options[self.TRACE_CONTEXT_KEY] = carrier

    def before_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
    ) -> None:
        """Start a span before processing message.

        Args:
            broker: Dramatiq broker.
            message: Message being processed.
        """
        if not OTEL_AVAILABLE or tracer is None or propagator is None:
            return

        # Extract trace context from message
        carrier = message.options.get(self.TRACE_CONTEXT_KEY, {})
        context = propagator.extract(carrier) if carrier else None

        # Get actor and queue info
        actor_name = message.actor_name
        queue_name = message.queue_name or "default"

        # Start span
        span: Span = tracer.start_span(
            name=f"dramatiq.process.{actor_name}",
            kind=SpanKind.CONSUMER,
            context=context,
        )

        # Add semantic attributes
        span.set_attribute("messaging.system", "dramatiq")
        span.set_attribute("messaging.destination", queue_name)
        span.set_attribute("messaging.operation", "process")
        span.set_attribute("dramatiq.actor", actor_name)
        span.set_attribute("dramatiq.message_id", message.message_id)
        span.set_attribute("service.name", self.service_name)

        # Add tenant context if available
        tenant = message.options.get("tenant_code")
        if tenant:
            span.set_attribute("tenant.code", tenant)

        # Store span in message options for after_process_message
        message.options[self.SPAN_KEY] = span

    def after_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
        *,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        """End the span after processing message.

        Args:
            broker: Dramatiq broker.
            message: Message that was processed.
            result: Result of processing (if successful).
            exception: Exception raised (if failed).
        """
        if not OTEL_AVAILABLE:
            return

        span: Span | None = message.options.pop(self.SPAN_KEY, None)
        if span is None:
            return

        try:
            if exception:
                span.set_status(Status(StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end()

    def after_skip_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
    ) -> None:
        """Handle skipped messages.

        Args:
            broker: Dramatiq broker.
            message: Message that was skipped.
        """
        if not OTEL_AVAILABLE:
            return

        span: Span | None = message.options.pop(self.SPAN_KEY, None)
        if span:
            span.set_attribute("dramatiq.skipped", True)
            span.set_status(Status(StatusCode.OK, "Message skipped"))
            span.end()
