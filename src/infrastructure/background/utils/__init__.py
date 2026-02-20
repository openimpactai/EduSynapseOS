# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Background processing utilities for EduSynapseOS.

This package contains utilities for background task processing:
- rate_limiter: Redis-based rate limiting for task dispatching
"""

from src.infrastructure.background.utils.rate_limiter import (
    TaskRateLimiter,
    diagnostic_scan_limiter,
    threshold_check_limiter,
)

__all__ = [
    "TaskRateLimiter",
    "diagnostic_scan_limiter",
    "threshold_check_limiter",
]
