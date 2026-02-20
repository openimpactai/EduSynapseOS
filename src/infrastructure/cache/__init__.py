# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cache infrastructure using Redis.

This package provides Redis client for caching and message brokering.
Tenant isolation is achieved via key prefixes: tenant:{tenant_code}:*

Example:
    from src.infrastructure.cache import RedisClient, init_redis, get_redis

    # Initialize at application startup
    await init_redis(settings)

    # Get the Redis client
    redis = get_redis()

    # Use with tenant isolation
    await redis.set_with_tenant("acme", "session:123", session_data)
    data = await redis.get_with_tenant("acme", "session:123")

    # Cleanup at shutdown
    await close_redis()
"""

from src.infrastructure.cache.redis_client import (
    RedisClient,
    RedisError,
    close_redis,
    get_redis,
    init_redis,
)

__all__ = [
    "RedisClient",
    "RedisError",
    "close_redis",
    "get_redis",
    "init_redis",
]
