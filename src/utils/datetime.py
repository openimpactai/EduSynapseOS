# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""DateTime utilities for EduSynapseOS.

This module provides standardized datetime operations to ensure consistency
across the entire codebase. All datetime operations should use these utilities.

Design Decisions:
-----------------
1. All timestamps are stored in UTC (PostgreSQL TIMESTAMPTZ)
2. All Python datetimes are timezone-aware (with timezone.utc)
3. This ensures no naive/aware datetime mixing errors

Usage:
------
    from src.utils.datetime import utc_now

    # For current time
    now = utc_now()

    # For SQLAlchemy model defaults
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # For Pydantic model defaults
    timestamp: datetime = Field(default_factory=utc_now)
"""

from datetime import datetime, timedelta, timezone


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime.

    Returns:
        Timezone-aware datetime representing current UTC time.

    Example:
        >>> now = utc_now()
        >>> now.tzinfo
        datetime.timezone.utc
    """
    return datetime.now(timezone.utc)


def utc_from_timestamp(timestamp: float) -> datetime:
    """Create a timezone-aware UTC datetime from a Unix timestamp.

    Args:
        timestamp: Unix timestamp (seconds since epoch).

    Returns:
        Timezone-aware UTC datetime.

    Example:
        >>> dt = utc_from_timestamp(1703145600)
        >>> dt.tzinfo
        datetime.timezone.utc
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware UTC.

    Args:
        dt: A datetime object (naive or aware) or None.

    Returns:
        Timezone-aware UTC datetime or None.

    Note:
        - If dt is None, returns None
        - If dt is naive, assumes UTC and adds tzinfo
        - If dt is aware, converts to UTC
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=timezone.utc)

    # Already aware - convert to UTC
    return dt.astimezone(timezone.utc)


def utc_today_start() -> datetime:
    """Get the start of today in UTC (midnight).

    Returns:
        Timezone-aware datetime for today at 00:00:00 UTC.
    """
    now = utc_now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def utc_today_end() -> datetime:
    """Get the end of today in UTC (23:59:59.999999).

    Returns:
        Timezone-aware datetime for today at 23:59:59.999999 UTC.
    """
    now = utc_now()
    return now.replace(hour=23, minute=59, second=59, microsecond=999999)


def days_ago(days: int) -> datetime:
    """Get a datetime N days ago from now.

    Args:
        days: Number of days to go back.

    Returns:
        Timezone-aware UTC datetime.
    """
    return utc_now() - timedelta(days=days)


def hours_ago(hours: int) -> datetime:
    """Get a datetime N hours ago from now.

    Args:
        hours: Number of hours to go back.

    Returns:
        Timezone-aware UTC datetime.
    """
    return utc_now() - timedelta(hours=hours)


def minutes_from_now(minutes: int) -> datetime:
    """Get a datetime N minutes from now.

    Args:
        minutes: Number of minutes to add.

    Returns:
        Timezone-aware UTC datetime.
    """
    return utc_now() + timedelta(minutes=minutes)


def days_from_now(days: int) -> datetime:
    """Get a datetime N days from now.

    Args:
        days: Number of days to add.

    Returns:
        Timezone-aware UTC datetime.
    """
    return utc_now() + timedelta(days=days)


def is_expired(expiry: datetime | None) -> bool:
    """Check if a datetime has passed (is expired).

    Args:
        expiry: The expiry datetime to check.

    Returns:
        True if expired or expiry is None, False otherwise.
    """
    if expiry is None:
        return True

    expiry_utc = ensure_utc(expiry)
    return utc_now() > expiry_utc


def time_until(target: datetime) -> timedelta:
    """Calculate time remaining until a target datetime.

    Args:
        target: The target datetime.

    Returns:
        Timedelta until target (negative if target is in the past).
    """
    target_utc = ensure_utc(target)
    return target_utc - utc_now()


def time_since(start: datetime) -> timedelta:
    """Calculate time elapsed since a start datetime.

    Args:
        start: The start datetime.

    Returns:
        Timedelta since start (negative if start is in the future).
    """
    start_utc = ensure_utc(start)
    return utc_now() - start_utc


def format_iso(dt: datetime | None) -> str | None:
    """Format a datetime as ISO 8601 string.

    Args:
        dt: Datetime to format.

    Returns:
        ISO 8601 formatted string or None.
    """
    if dt is None:
        return None

    dt_utc = ensure_utc(dt)
    return dt_utc.isoformat()


def parse_iso(iso_string: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string.

    Args:
        iso_string: ISO 8601 formatted string.

    Returns:
        Timezone-aware UTC datetime or None.
    """
    if iso_string is None:
        return None

    dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    return ensure_utc(dt)


def seconds_to_human(seconds: int) -> str:
    """Convert seconds to human-readable duration.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string like "2h 30m" or "45m 10s".
    """
    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


# Aliases for convenience
now = utc_now
