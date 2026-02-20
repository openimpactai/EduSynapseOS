# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Background processing middleware for EduSynapseOS.

This module provides custom Dramatiq middleware for:
- Tenant context propagation
- OpenTelemetry tracing (optional)
- Prometheus metrics (optional)
"""

from src.infrastructure.background.middleware.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsMiddleware,
    get_metrics_middleware,
    set_metrics_middleware,
)
from src.infrastructure.background.middleware.tenant import (
    TenantMiddleware,
    get_current_tenant,
    set_current_tenant,
)
from src.infrastructure.background.middleware.tracing import (
    OTEL_AVAILABLE,
    TracingMiddleware,
)

__all__ = [
    # Tenant
    "TenantMiddleware",
    "get_current_tenant",
    "set_current_tenant",
    # Tracing
    "TracingMiddleware",
    "OTEL_AVAILABLE",
    # Metrics
    "MetricsMiddleware",
    "PROMETHEUS_AVAILABLE",
    "get_metrics_middleware",
    "set_metrics_middleware",
]
