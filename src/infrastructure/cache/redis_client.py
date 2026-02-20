# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Redis client for caching and message brokering.

This module provides an async Redis client wrapper with tenant isolation
support. All tenant-specific keys are automatically prefixed with
tenant:{tenant_code}: to ensure data isolation.

Example:
    from src.infrastructure.cache import init_redis, get_redis

    # Initialize at startup
    await init_redis(settings)

    # Use the client
    redis = get_redis()
    await redis.set("global_key", "value")
    await redis.set_with_tenant("acme", "user:123", user_data)
"""

import json
from typing import TYPE_CHECKING, Any, Optional, Union

from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError as BaseRedisError

if TYPE_CHECKING:
    from src.core.config.settings import Settings

# Module-level state
_redis_client: Optional["RedisClient"] = None


class RedisError(Exception):
    """Exception raised for Redis operation failures.

    Attributes:
        message: Human-readable error description.
        original_error: The underlying Redis error.
    """

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        """Initialize the Redis error.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.original_error:
            return f"{self.message}: {self.original_error}"
        return self.message


class RedisClient:
    """Async Redis client with tenant isolation support.

    This client wraps the redis-py async client and provides:
    - Connection pooling
    - Tenant-isolated key prefixing
    - JSON serialization/deserialization
    - Common cache operations

    Attributes:
        pool: The Redis connection pool.

    Example:
        client = RedisClient(settings)
        await client.connect()

        # Global operations
        await client.set("key", "value")
        value = await client.get("key")

        # Tenant-isolated operations
        await client.set_with_tenant("acme", "session:123", data)
        data = await client.get_with_tenant("acme", "session:123")

        await client.close()
    """

    TENANT_KEY_PREFIX = "tenant"

    def __init__(self, settings: "Settings") -> None:
        """Initialize the Redis client.

        Args:
            settings: Application settings containing Redis configuration.
        """
        self._settings = settings
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None

    async def connect(self) -> None:
        """Create the Redis connection pool.

        Raises:
            RedisError: If connection fails.
        """
        try:
            self._pool = ConnectionPool.from_url(
                self._settings.redis.url,
                max_connections=self._settings.redis.max_connections,
                decode_responses=True,
            )
            self._redis = Redis(connection_pool=self._pool)

            # Verify connection
            await self._redis.ping()
        except BaseRedisError as e:
            raise RedisError("Failed to connect to Redis", e) from e

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

    def _ensure_connected(self) -> Redis:
        """Ensure the client is connected.

        Returns:
            The Redis client instance.

        Raises:
            RedisError: If not connected.
        """
        if self._redis is None:
            raise RedisError("Redis client not connected. Call connect() first.")
        return self._redis

    def _tenant_key(self, tenant_code: str, key: str) -> str:
        """Build a tenant-prefixed key.

        Args:
            tenant_code: The tenant code.
            key: The original key.

        Returns:
            Key prefixed with tenant:{tenant_code}:
        """
        return f"{self.TENANT_KEY_PREFIX}:{tenant_code}:{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize a value to JSON string.

        Args:
            value: The value to serialize.

        Returns:
            JSON string representation.
        """
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _deserialize(self, value: Optional[str]) -> Any:
        """Deserialize a JSON string to Python object.

        Args:
            value: The JSON string to deserialize.

        Returns:
            Python object or None if value is None.
        """
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # ========== Global operations ==========

    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None,
    ) -> None:
        """Set a key-value pair.

        Args:
            key: The key.
            value: The value (will be JSON serialized if not a string).
            expire_seconds: Optional expiration time in seconds.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            serialized = self._serialize(value)
            await redis.set(key, serialized, ex=expire_seconds)
        except BaseRedisError as e:
            raise RedisError(f"Failed to set key: {key}", e) from e

    async def get(self, key: str) -> Any:
        """Get a value by key.

        Args:
            key: The key.

        Returns:
            The deserialized value or None if not found.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            value = await redis.get(key)
            return self._deserialize(value)
        except BaseRedisError as e:
            raise RedisError(f"Failed to get key: {key}", e) from e

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            result = await redis.delete(key)
            return result > 0
        except BaseRedisError as e:
            raise RedisError(f"Failed to delete key: {key}", e) from e

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            result = await redis.exists(key)
            return result > 0
        except BaseRedisError as e:
            raise RedisError(f"Failed to check key existence: {key}", e) from e

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key.

        Args:
            key: The key.
            seconds: Expiration time in seconds.

        Returns:
            True if the timeout was set, False if key doesn't exist.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            return await redis.expire(key, seconds)
        except BaseRedisError as e:
            raise RedisError(f"Failed to set expiration on key: {key}", e) from e

    async def ttl(self, key: str) -> int:
        """Get the time-to-live for a key.

        Args:
            key: The key.

        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            return await redis.ttl(key)
        except BaseRedisError as e:
            raise RedisError(f"Failed to get TTL for key: {key}", e) from e

    # ========== Tenant-isolated operations ==========

    async def set_with_tenant(
        self,
        tenant_code: str,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None,
    ) -> None:
        """Set a tenant-isolated key-value pair.

        The key is automatically prefixed with tenant:{tenant_code}:

        Args:
            tenant_code: The tenant code.
            key: The key.
            value: The value.
            expire_seconds: Optional expiration time in seconds.

        Raises:
            RedisError: If the operation fails.
        """
        full_key = self._tenant_key(tenant_code, key)
        await self.set(full_key, value, expire_seconds)

    async def get_with_tenant(self, tenant_code: str, key: str) -> Any:
        """Get a tenant-isolated value by key.

        Args:
            tenant_code: The tenant code.
            key: The key.

        Returns:
            The deserialized value or None if not found.

        Raises:
            RedisError: If the operation fails.
        """
        full_key = self._tenant_key(tenant_code, key)
        return await self.get(full_key)

    async def delete_with_tenant(self, tenant_code: str, key: str) -> bool:
        """Delete a tenant-isolated key.

        Args:
            tenant_code: The tenant code.
            key: The key.

        Returns:
            True if the key was deleted, False if it didn't exist.

        Raises:
            RedisError: If the operation fails.
        """
        full_key = self._tenant_key(tenant_code, key)
        return await self.delete(full_key)

    async def exists_with_tenant(self, tenant_code: str, key: str) -> bool:
        """Check if a tenant-isolated key exists.

        Args:
            tenant_code: The tenant code.
            key: The key.

        Returns:
            True if the key exists, False otherwise.

        Raises:
            RedisError: If the operation fails.
        """
        full_key = self._tenant_key(tenant_code, key)
        return await self.exists(full_key)

    async def delete_tenant_keys(self, tenant_code: str, pattern: str = "*") -> int:
        """Delete all keys matching a pattern for a tenant.

        Args:
            tenant_code: The tenant code.
            pattern: Key pattern to match (default: all keys).

        Returns:
            Number of keys deleted.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        full_pattern = self._tenant_key(tenant_code, pattern)

        try:
            keys = []
            async for key in redis.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                return await redis.delete(*keys)
            return 0
        except BaseRedisError as e:
            raise RedisError(
                f"Failed to delete tenant keys: {tenant_code}/{pattern}", e
            ) from e

    # ========== Pub/Sub operations ==========

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel.

        Args:
            channel: The channel name.
            message: The message to publish.

        Returns:
            Number of subscribers that received the message.

        Raises:
            RedisError: If the operation fails.
        """
        redis = self._ensure_connected()
        try:
            return await redis.publish(channel, message)
        except BaseRedisError as e:
            raise RedisError(f"Failed to publish to channel: {channel}", e) from e

    async def publish_with_tenant(
        self, tenant_code: str, channel: str, message: str
    ) -> int:
        """Publish a message to a tenant-isolated channel.

        Args:
            tenant_code: The tenant code.
            channel: The channel name.
            message: The message to publish.

        Returns:
            Number of subscribers that received the message.

        Raises:
            RedisError: If the operation fails.
        """
        full_channel = self._tenant_key(tenant_code, channel)
        return await self.publish(full_channel, message)

    # ========== Health check ==========

    async def ping(self) -> bool:
        """Check if Redis is reachable.

        Returns:
            True if Redis responds to ping, False otherwise.
        """
        try:
            redis = self._ensure_connected()
            await redis.ping()
            return True
        except (RedisError, BaseRedisError):
            return False


# ========== Module-level functions ==========


async def init_redis(settings: "Settings") -> None:
    """Initialize the global Redis client.

    This should be called once at application startup.

    Args:
        settings: Application settings containing Redis configuration.

    Raises:
        RedisError: If connection fails.
    """
    global _redis_client

    _redis_client = RedisClient(settings)
    await _redis_client.connect()


async def close_redis() -> None:
    """Close the global Redis client.

    This should be called at application shutdown.
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


def get_redis() -> RedisClient:
    """Get the global Redis client.

    Returns:
        The RedisClient instance.

    Raises:
        RedisError: If Redis has not been initialized.
    """
    if _redis_client is None:
        raise RedisError("Redis not initialized. Call init_redis() first.")
    return _redis_client
