# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dramatiq broker configuration for EduSynapseOS.

This module provides production-ready background task processing with:
- Redis broker for message persistence and durability
- Result backend for task results
- Custom middleware for tenant context propagation

Example:
    from src.infrastructure.background.broker import setup_dramatiq, get_broker

    # Setup at application startup
    broker = setup_dramatiq()

    # Get broker for manual operations
    broker = get_broker()
"""

import logging
import os
from typing import Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.stub import StubBroker
from dramatiq.results import Results
from dramatiq.results.backends.redis import RedisBackend

from src.core.config import get_settings
from src.infrastructure.background.middleware import (
    OTEL_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    MetricsMiddleware,
    TenantMiddleware,
    TracingMiddleware,
    get_current_tenant,
    set_current_tenant,
    set_metrics_middleware,
)

logger = logging.getLogger(__name__)


class Queues:
    """Queue name constants for task routing."""

    DEFAULT = "default"
    ANALYTICS = "analytics"
    MEMORY = "memory"
    REVIEW = "review"
    DIAGNOSTICS = "diagnostics"
    CURRICULUM = "curriculum"
    HIGH_PRIORITY = "high_priority"
    LOW_PRIORITY = "low_priority"
    LEARNING_ANALYSIS = "learning_analysis"  # For session analysis and pattern detection
    REPORTS = "reports"  # For report generation tasks


class Priority:
    """Task priority levels (lower number = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    DIAGNOSTIC = 2
    NORMAL = 3
    ANALYSIS = 4  # For background analysis tasks
    LOW = 5


class BrokerManager:
    """Manages Dramatiq broker lifecycle.

    Handles broker initialization, middleware setup, and shutdown.

    Attributes:
        _broker: The Dramatiq broker instance.
        _results_backend: The results backend instance.
        _initialized: Whether the broker has been initialized.
    """

    def __init__(self) -> None:
        """Initialize broker manager."""
        self._broker: dramatiq.Broker | None = None
        self._results_backend: RedisBackend | None = None
        self._initialized = False

    @property
    def broker(self) -> dramatiq.Broker:
        """Get the configured broker.

        Returns:
            The Dramatiq broker instance.

        Raises:
            RuntimeError: If broker not initialized.
        """
        if self._broker is None:
            raise RuntimeError("Broker not initialized. Call setup() first.")
        return self._broker

    @property
    def is_initialized(self) -> bool:
        """Check if broker is initialized."""
        return self._initialized

    def setup(self) -> dramatiq.Broker:
        """Setup and configure the Dramatiq broker.

        Returns:
            Configured broker instance.
        """
        if self._initialized:
            return self._broker  # type: ignore

        logger.info("Setting up Dramatiq broker...")

        # Check for test mode
        use_stub = os.getenv("DRAMATIQ_TEST_MODE", "false").lower() == "true"

        if use_stub:
            self._broker = StubBroker()
            self._broker.emit_after("process_boot")
            logger.info("Using StubBroker for testing")
        else:
            settings = get_settings()
            redis_url = settings.redis.url

            # Setup results backend
            self._results_backend = RedisBackend(url=redis_url)

            # Create Redis broker
            self._broker = RedisBroker(url=redis_url)

            # Add middleware
            self._setup_middleware()

            logger.info("Redis broker initialized (url: %s)", redis_url.split("@")[-1])

        # Set as global broker
        dramatiq.set_broker(self._broker)
        self._initialized = True

        return self._broker

    def _setup_middleware(self) -> None:
        """Setup broker middleware.

        Adds the following middleware in order:
        1. Results middleware (for task results)
        2. TenantMiddleware (tenant context propagation)
        3. TracingMiddleware (OpenTelemetry tracing, optional)
        4. MetricsMiddleware (Prometheus metrics, optional)
        """
        if self._broker is None:
            return

        # Add results middleware if backend is available
        if self._results_backend:
            self._broker.add_middleware(Results(backend=self._results_backend))

        # Add tenant context middleware
        self._broker.add_middleware(TenantMiddleware())

        # Add OpenTelemetry tracing middleware (optional)
        if OTEL_AVAILABLE:
            self._broker.add_middleware(TracingMiddleware())
            logger.info("OpenTelemetry tracing middleware enabled")

        # Add Prometheus metrics middleware (optional)
        if PROMETHEUS_AVAILABLE:
            metrics_middleware = MetricsMiddleware()
            self._broker.add_middleware(metrics_middleware)
            set_metrics_middleware(metrics_middleware)
            logger.info("Prometheus metrics middleware enabled")

        logger.debug("Middleware configured")

    def shutdown(self) -> None:
        """Shutdown the broker."""
        if self._broker is not None:
            self._broker.close()
            self._broker = None
            self._initialized = False
            logger.info("Broker shutdown complete")

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Queue statistics dictionary.
        """
        if not self._initialized or self._broker is None:
            return {"status": "not_initialized"}

        if isinstance(self._broker, RedisBroker):
            stats: dict[str, Any] = {"broker_type": "redis"}
            try:
                import redis

                settings = get_settings()
                client = redis.from_url(settings.redis.url)

                # Get queue lengths for known queues
                queues = [
                    Queues.DEFAULT,
                    Queues.ANALYTICS,
                    Queues.MEMORY,
                    Queues.REVIEW,
                    Queues.DIAGNOSTICS,
                    Queues.CURRICULUM,
                ]
                queue_lengths = {}
                for queue in queues:
                    queue_name = f"dramatiq:{queue}"
                    length = client.llen(queue_name)
                    queue_lengths[queue] = length

                stats["queues"] = queue_lengths
                stats["status"] = "healthy"
            except Exception as e:
                stats["status"] = "error"
                stats["error"] = str(e)

            return stats

        return {"broker_type": "stub", "status": "healthy"}


# Singleton instance
_broker_manager: BrokerManager | None = None


def get_broker_manager() -> BrokerManager:
    """Get the singleton broker manager.

    Returns:
        BrokerManager instance.
    """
    global _broker_manager
    if _broker_manager is None:
        _broker_manager = BrokerManager()
    return _broker_manager


def setup_dramatiq() -> dramatiq.Broker:
    """Setup Dramatiq with configuration from settings.

    This should be called once at application startup.

    Returns:
        Configured broker.
    """
    manager = get_broker_manager()
    return manager.setup()


def get_broker() -> dramatiq.Broker:
    """Get the current Dramatiq broker.

    Returns:
        Broker instance.

    Raises:
        RuntimeError: If broker not initialized.
    """
    return get_broker_manager().broker


def shutdown_dramatiq() -> None:
    """Shutdown the Dramatiq broker.

    Should be called at application shutdown.
    """
    global _broker_manager
    if _broker_manager is not None:
        _broker_manager.shutdown()
        _broker_manager = None
