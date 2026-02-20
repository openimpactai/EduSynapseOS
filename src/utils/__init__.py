# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utility functions and helpers for EduSynapseOS.

This package contains cross-cutting utilities:
- logging: Structured logging with structlog
- datetime: Timezone-aware datetime operations
"""

from src.utils.datetime import (
    days_ago,
    days_from_now,
    ensure_utc,
    format_iso,
    hours_ago,
    is_expired,
    minutes_from_now,
    now,
    parse_iso,
    seconds_to_human,
    time_since,
    time_until,
    utc_from_timestamp,
    utc_now,
    utc_today_end,
    utc_today_start,
)
from src.utils.logging import bind_context, clear_context, get_logger, setup_logging

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "bind_context",
    "clear_context",
    # Datetime
    "utc_now",
    "now",
    "utc_from_timestamp",
    "ensure_utc",
    "utc_today_start",
    "utc_today_end",
    "days_ago",
    "hours_ago",
    "minutes_from_now",
    "days_from_now",
    "is_expired",
    "time_until",
    "time_since",
    "format_iso",
    "parse_iso",
    "seconds_to_human",
]
