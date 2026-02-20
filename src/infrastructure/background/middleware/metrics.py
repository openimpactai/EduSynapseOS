# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prometheus metrics middleware for Dramatiq.

Provides metrics collection for background task monitoring,
including counters, histograms, and gauges.
"""

import logging
import time
from typing import Any

import dramatiq
from dramatiq import Message, Middleware

logger = logging.getLogger(__name__)

# Prometheus imports (optional - graceful degradation if not installed)
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
    Histogram = None  # type: ignore[assignment, misc]
    REGISTRY = None  # type: ignore[assignment]

# Diagnostic actor names for specific metrics
DIAGNOSTIC_ACTORS = {
    "run_diagnostic_scan",
    "run_batch_diagnostic_scans",
    "assess_risk_score",
    "generate_diagnostic_report",
    "check_diagnostic_thresholds",
}


class MetricsMiddleware(Middleware):
    """Middleware that collects Prometheus metrics for Dramatiq actors.

    Metrics Collected:
    - dramatiq_messages_total: Total messages by actor, queue, status
    - dramatiq_message_duration_seconds: Processing time histogram
    - dramatiq_messages_in_flight: Currently processing messages gauge
    - dramatiq_messages_failed_total: Failed messages counter
    - diagnostic_scans_total: Diagnostic scan counter
    - diagnostic_scan_duration_seconds: Diagnostic scan duration histogram
    - diagnostic_indicators_detected_total: Detected indicators counter

    Usage:
        from src.infrastructure.background.middleware.metrics import MetricsMiddleware

        broker.add_middleware(MetricsMiddleware())

    Note:
        Requires prometheus_client. If not available, middleware operates as no-op.
    """

    START_TIME_KEY = "_metrics_start_time"

    def __init__(
        self,
        registry: Any = None,
        namespace: str = "edusynapse",
    ) -> None:
        """Initialize metrics middleware.

        Args:
            registry: Prometheus registry (default: global REGISTRY).
            namespace: Metrics namespace prefix.
        """
        self._initialized = False

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Prometheus client not available. Metrics middleware will be no-op. "
                "Install with: pip install prometheus-client"
            )
            return

        self.registry = registry or REGISTRY
        self.namespace = namespace

        # General Dramatiq metrics
        self.messages_total = Counter(
            f"{namespace}_dramatiq_messages_total",
            "Total Dramatiq messages processed",
            ["actor", "queue", "status"],
            registry=self.registry,
        )

        self.message_duration = Histogram(
            f"{namespace}_dramatiq_message_duration_seconds",
            "Dramatiq message processing duration",
            ["actor", "queue"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry,
        )

        self.messages_in_flight = Gauge(
            f"{namespace}_dramatiq_messages_in_flight",
            "Dramatiq messages currently being processed",
            ["actor", "queue"],
            registry=self.registry,
        )

        self.messages_failed = Counter(
            f"{namespace}_dramatiq_messages_failed_total",
            "Total failed Dramatiq messages",
            ["actor", "queue", "exception_type"],
            registry=self.registry,
        )

        # Diagnostic-specific metrics
        self.diagnostic_scans_total = Counter(
            f"{namespace}_diagnostic_scans_total",
            "Total diagnostic scans",
            ["scan_type", "tenant"],
            registry=self.registry,
        )

        self.diagnostic_scan_duration = Histogram(
            f"{namespace}_diagnostic_scan_duration_seconds",
            "Diagnostic scan duration",
            ["scan_type"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
            registry=self.registry,
        )

        self.diagnostic_indicators_detected = Counter(
            f"{namespace}_diagnostic_indicators_detected_total",
            "Total diagnostic indicators detected",
            ["indicator_type", "threshold_level"],
            registry=self.registry,
        )

        self._initialized = True
        logger.debug("Metrics middleware initialized with namespace: %s", namespace)

    def before_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
    ) -> None:
        """Record message start time and increment in-flight gauge.

        Args:
            broker: Dramatiq broker.
            message: Message being processed.
        """
        if not self._initialized:
            return

        actor_name = message.actor_name
        queue_name = message.queue_name or "default"

        # Store start time
        message.options[self.START_TIME_KEY] = time.perf_counter()

        # Increment in-flight gauge
        self.messages_in_flight.labels(
            actor=actor_name,
            queue=queue_name,
        ).inc()

    def after_process_message(
        self,
        broker: dramatiq.Broker,
        message: Message,
        *,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        """Record message completion metrics.

        Args:
            broker: Dramatiq broker.
            message: Message that was processed.
            result: Result of processing (if successful).
            exception: Exception raised (if failed).
        """
        if not self._initialized:
            return

        actor_name = message.actor_name
        queue_name = message.queue_name or "default"
        start_time = message.options.pop(self.START_TIME_KEY, None)

        # Decrement in-flight gauge
        self.messages_in_flight.labels(
            actor=actor_name,
            queue=queue_name,
        ).dec()

        # Record duration
        duration = 0.0
        if start_time is not None:
            duration = time.perf_counter() - start_time
            self.message_duration.labels(
                actor=actor_name,
                queue=queue_name,
            ).observe(duration)

        # Record status
        if exception:
            status = "failed"
            exception_type = type(exception).__name__
            self.messages_failed.labels(
                actor=actor_name,
                queue=queue_name,
                exception_type=exception_type,
            ).inc()
        else:
            status = "success"

        self.messages_total.labels(
            actor=actor_name,
            queue=queue_name,
            status=status,
        ).inc()

        # Record diagnostic-specific metrics
        if actor_name in DIAGNOSTIC_ACTORS and not exception:
            self._record_diagnostic_metrics(message, actor_name, duration)

    def _record_diagnostic_metrics(
        self,
        message: Message,
        actor_name: str,
        duration: float,
    ) -> None:
        """Record diagnostic-specific metrics.

        Args:
            message: The processed message.
            actor_name: Name of the diagnostic actor.
            duration: Processing duration in seconds.
        """
        tenant = message.options.get("tenant_code") or "unknown"

        # Determine scan type from actor name
        if actor_name == "run_diagnostic_scan":
            scan_type = "single"
        elif actor_name == "run_batch_diagnostic_scans":
            scan_type = "batch"
        elif actor_name == "check_diagnostic_thresholds":
            scan_type = "threshold_check"
        else:
            scan_type = actor_name

        self.diagnostic_scans_total.labels(
            scan_type=scan_type,
            tenant=tenant,
        ).inc()

        if duration > 0:
            self.diagnostic_scan_duration.labels(
                scan_type=scan_type,
            ).observe(duration)

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
        if not self._initialized:
            return

        actor_name = message.actor_name
        queue_name = message.queue_name or "default"

        # Decrement in-flight if we tracked it
        if self.START_TIME_KEY in message.options:
            message.options.pop(self.START_TIME_KEY, None)
            self.messages_in_flight.labels(
                actor=actor_name,
                queue=queue_name,
            ).dec()

        self.messages_total.labels(
            actor=actor_name,
            queue=queue_name,
            status="skipped",
        ).inc()

    def record_indicator_detected(
        self,
        indicator_type: str,
        threshold_level: str,
    ) -> None:
        """Record a detected indicator.

        Call this from diagnostic actors when indicators are found.

        Args:
            indicator_type: Type of indicator (e.g., "dyslexia").
            threshold_level: Threshold level (e.g., "concern", "alert").
        """
        if not self._initialized:
            return

        self.diagnostic_indicators_detected.labels(
            indicator_type=indicator_type,
            threshold_level=threshold_level,
        ).inc()


# Singleton instance for use in actors
_metrics_middleware: MetricsMiddleware | None = None


def get_metrics_middleware() -> MetricsMiddleware | None:
    """Get the metrics middleware instance.

    Returns:
        MetricsMiddleware instance or None if not set.
    """
    return _metrics_middleware


def set_metrics_middleware(middleware: MetricsMiddleware) -> None:
    """Set the global metrics middleware instance.

    Args:
        middleware: MetricsMiddleware instance to set.
    """
    global _metrics_middleware
    _metrics_middleware = middleware
