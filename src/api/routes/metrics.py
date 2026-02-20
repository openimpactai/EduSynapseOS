# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prometheus metrics endpoint.

Exposes application metrics in Prometheus format for scraping.
"""

import logging

from fastapi import APIRouter, Response

logger = logging.getLogger(__name__)

router = APIRouter()

# Prometheus imports (optional - graceful degradation)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Get application metrics in Prometheus format.",
    responses={
        200: {
            "description": "Prometheus metrics",
            "content": {"text/plain": {}},
        },
        503: {
            "description": "Metrics not available",
        },
    },
)
async def get_metrics() -> Response:
    """Get Prometheus metrics.

    Returns metrics collected by the application including:
    - Dramatiq message processing metrics
    - Diagnostic scan metrics
    - Indicator detection metrics

    Returns:
        Response with Prometheus format metrics.
    """
    if not METRICS_AVAILABLE:
        return Response(
            content="# Prometheus client not available\n",
            media_type="text/plain",
            status_code=503,
        )

    try:
        metrics = generate_latest(REGISTRY)
        return Response(
            content=metrics,
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error("Failed to generate metrics: %s", e)
        return Response(
            content=f"# Error generating metrics: {e}\n",
            media_type="text/plain",
            status_code=500,
        )
