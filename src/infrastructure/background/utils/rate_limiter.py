# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rate limiting utilities for background task dispatching.

Provides simple Redis-based rate limiting to prevent excessive task
dispatching for the same entity. This is different from API rate
limiting (slowapi) - this controls background task frequency.

Use Cases:
- Limit diagnostic scans per student to 1/hour
- Limit threshold checks per student to 2/15min

Example:
    from src.infrastructure.background.utils import diagnostic_scan_limiter

    if diagnostic_scan_limiter.allow(f"{tenant_code}:{student_id}"):
        run_diagnostic_scan.send(...)
    else:
        logger.debug("Rate limited, skipping scan")
"""

import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class TaskRateLimiter:
    """Rate limiter for background task dispatching.

    Uses Redis to track and limit task frequency per entity.
    Designed for controlling how often background tasks are
    dispatched for the same entity (e.g., student).

    This differs from API rate limiting:
    - API limiter: Controls HTTP request frequency
    - TaskRateLimiter: Controls background task dispatch frequency

    Attributes:
        key_prefix: Redis key prefix for this limiter.
        window_seconds: Time window in seconds.
        max_requests: Maximum requests allowed in window.

    Example:
        limiter = TaskRateLimiter(
            key_prefix="diagnostic_scan",
            window=timedelta(hours=1),
            max_requests=1,
        )

        entity = f"{tenant_code}:{student_id}"

        if limiter.allow(entity):
            run_diagnostic_scan.send(...)
        else:
            logger.debug("Rate limited: %s", entity)
    """

    def __init__(
        self,
        key_prefix: str,
        window: timedelta = timedelta(hours=1),
        max_requests: int = 1,
    ) -> None:
        """Initialize rate limiter.

        Args:
            key_prefix: Redis key prefix for this limiter.
            window: Time window for rate limiting.
            max_requests: Maximum requests allowed in window.
        """
        self.key_prefix = key_prefix
        self.window_seconds = int(window.total_seconds())
        self.max_requests = max_requests
        self._redis = None

    def _get_redis(self):
        """Get Redis connection lazily.

        Uses sync redis client because:
        - Single INCR operation is fast (<1ms)
        - Event handlers may call this in async context
        - Brief sync call is acceptable for rate limiting

        Returns:
            Redis client or None if unavailable.
        """
        if self._redis is None:
            try:
                import redis

                from src.core.config import get_settings

                settings = get_settings()
                self._redis = redis.Redis.from_url(
                    settings.redis.url,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning("Redis not available for rate limiting: %s", e)
                return None
        return self._redis

    def _get_key(self, entity_id: str) -> str:
        """Generate Redis key for entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            Full Redis key with prefix.
        """
        return f"ratelimit:{self.key_prefix}:{entity_id}"

    def allow(self, entity_id: str) -> bool:
        """Check if request is allowed for entity.

        Uses Redis INCR with expiry for atomic rate limiting.
        Fails open (allows) if Redis is unavailable.

        Args:
            entity_id: Entity identifier (e.g., "tenant:student_id").

        Returns:
            True if request is allowed, False if rate limited.
        """
        redis_client = self._get_redis()
        if redis_client is None:
            # If Redis unavailable, allow all (fail open)
            return True

        key = self._get_key(entity_id)

        try:
            # Increment counter atomically
            current = redis_client.incr(key)

            # Set expiry on first request
            if current == 1:
                redis_client.expire(key, self.window_seconds)

            if current > self.max_requests:
                logger.debug(
                    "Rate limited: %s (count: %d, max: %d)",
                    entity_id,
                    current,
                    self.max_requests,
                )
                return False

            return True

        except Exception as e:
            logger.warning("Rate limit check failed: %s", e)
            return True  # Fail open

    def remaining(self, entity_id: str) -> int:
        """Get remaining requests for entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            Remaining request count in current window.
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return self.max_requests

        key = self._get_key(entity_id)

        try:
            current = redis_client.get(key)
            if current is None:
                return self.max_requests
            return max(0, self.max_requests - int(current))
        except Exception:
            return self.max_requests

    def reset(self, entity_id: str) -> None:
        """Reset rate limit for entity.

        Useful for admin operations or testing.

        Args:
            entity_id: Entity identifier.
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return

        key = self._get_key(entity_id)

        try:
            redis_client.delete(key)
            logger.debug("Rate limit reset for: %s", entity_id)
        except Exception as e:
            logger.warning("Rate limit reset failed: %s", e)


# Pre-configured limiters for diagnostic system
diagnostic_scan_limiter = TaskRateLimiter(
    key_prefix="diagnostic_scan",
    window=timedelta(hours=1),
    max_requests=1,
)

threshold_check_limiter = TaskRateLimiter(
    key_prefix="threshold_check",
    window=timedelta(minutes=15),
    max_requests=2,
)
