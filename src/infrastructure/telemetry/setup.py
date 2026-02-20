# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""OpenTelemetry setup for EduSynapseOS.

Configures OpenTelemetry SDK for distributed tracing across
the application, including API and background workers.
"""

import logging
import os

logger = logging.getLogger(__name__)


def setup_telemetry(
    service_name: str = "edusynapse-api",
    otlp_endpoint: str | None = None,
) -> bool:
    """Setup OpenTelemetry tracing.

    Initializes the OpenTelemetry SDK with OTLP exporter for
    distributed tracing. If no endpoint is configured, operates
    as no-op.

    Args:
        service_name: Name of the service for trace attribution.
        otlp_endpoint: OTLP collector endpoint. If None, reads from
            OTEL_EXPORTER_OTLP_ENDPOINT environment variable.

    Returns:
        True if setup succeeded, False otherwise.

    Example:
        # In app startup
        setup_telemetry(
            service_name="edusynapse-api",
            otlp_endpoint="http://localhost:4317",
        )
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry SDK not available, skipping telemetry setup. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return False

    # Get endpoint from env if not provided
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not otlp_endpoint:
        logger.info(
            "No OTLP endpoint configured (OTEL_EXPORTER_OTLP_ENDPOINT not set). "
            "Telemetry will use default noop tracer."
        )
        return False

    try:
        # Create resource with service name
        resource = Resource.create({
            SERVICE_NAME: service_name,
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # Use insecure for local development
        )

        # Add span processor with batching
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        logger.info(
            "OpenTelemetry configured: service=%s, endpoint=%s",
            service_name,
            otlp_endpoint,
        )
        return True

    except Exception as e:
        logger.error("Failed to setup OpenTelemetry: %s", e)
        return False


def setup_fastapi_instrumentation() -> bool:
    """Setup OpenTelemetry instrumentation for FastAPI.

    Automatically instruments FastAPI to create spans for
    incoming HTTP requests.

    Returns:
        True if setup succeeded, False otherwise.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor().instrument()
        logger.info("FastAPI instrumentation enabled")
        return True
    except ImportError:
        logger.warning(
            "FastAPI instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-fastapi"
        )
        return False
    except Exception as e:
        logger.error("Failed to setup FastAPI instrumentation: %s", e)
        return False


def setup_sqlalchemy_instrumentation() -> bool:
    """Setup OpenTelemetry instrumentation for SQLAlchemy.

    Automatically instruments SQLAlchemy to create spans for
    database operations.

    Returns:
        True if setup succeeded, False otherwise.
    """
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumentation enabled")
        return True
    except ImportError:
        logger.warning(
            "SQLAlchemy instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-sqlalchemy"
        )
        return False
    except Exception as e:
        logger.error("Failed to setup SQLAlchemy instrumentation: %s", e)
        return False


def setup_redis_instrumentation() -> bool:
    """Setup OpenTelemetry instrumentation for Redis.

    Automatically instruments Redis to create spans for
    cache operations.

    Returns:
        True if setup succeeded, False otherwise.
    """
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")
        return True
    except ImportError:
        logger.warning(
            "Redis instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-redis"
        )
        return False
    except Exception as e:
        logger.error("Failed to setup Redis instrumentation: %s", e)
        return False


def setup_all_instrumentation() -> None:
    """Setup all available OpenTelemetry instrumentations.

    Attempts to setup instrumentation for all supported libraries.
    Failures are logged but do not raise exceptions.
    """
    setup_fastapi_instrumentation()
    setup_sqlalchemy_instrumentation()
    setup_redis_instrumentation()
