# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Redis cache connection.

These tests require a running Redis instance.
Run with: pytest tests/integration/test_redis_connection.py -v

Prerequisites:
    - Redis running at localhost:6379
"""

import pytest

from src.core.config.settings import Settings, clear_settings_cache
from src.infrastructure.cache.redis_client import (
    RedisClient,
    RedisError,
    close_redis,
    get_redis,
    init_redis,
)


@pytest.fixture
def settings() -> Settings:
    """Provide fresh settings for each test."""
    clear_settings_cache()
    return Settings()


@pytest.fixture
async def initialized_redis(settings: Settings) -> None:
    """Initialize and cleanup the Redis connection."""
    await init_redis(settings)
    yield
    await close_redis()


@pytest.fixture
async def redis_client(settings: Settings) -> RedisClient:
    """Provide a Redis client for testing."""
    client = RedisClient(settings)
    await client.connect()
    yield client
    await client.close()


@pytest.mark.integration
class TestRedisInitialization:
    """Tests for Redis initialization."""

    async def test_init_creates_client(self, settings: Settings) -> None:
        """Test that initialization creates the Redis client."""
        await init_redis(settings)

        try:
            client = get_redis()
            assert client is not None
        finally:
            await close_redis()

    async def test_close_clears_state(self, settings: Settings) -> None:
        """Test that close clears the module state."""
        await init_redis(settings)
        await close_redis()

        with pytest.raises(RedisError) as exc_info:
            get_redis()

        assert "not initialized" in str(exc_info.value)


@pytest.mark.integration
class TestRedisOperations:
    """Tests for Redis operations."""

    async def test_set_and_get(self, redis_client: RedisClient) -> None:
        """Test basic set and get operations."""
        await redis_client.set("test:key", "test_value")
        value = await redis_client.get("test:key")

        assert value == "test_value"

        # Cleanup
        await redis_client.delete("test:key")

    async def test_set_and_get_json(self, redis_client: RedisClient) -> None:
        """Test set and get with JSON serialization."""
        data = {"name": "test", "count": 42, "active": True}
        await redis_client.set("test:json", data)
        value = await redis_client.get("test:json")

        assert value == data

        # Cleanup
        await redis_client.delete("test:json")

    async def test_set_with_expiration(self, redis_client: RedisClient) -> None:
        """Test set with expiration."""
        await redis_client.set("test:expire", "value", expire_seconds=3600)

        ttl = await redis_client.ttl("test:expire")
        assert ttl > 0
        assert ttl <= 3600

        # Cleanup
        await redis_client.delete("test:expire")

    async def test_delete(self, redis_client: RedisClient) -> None:
        """Test delete operation."""
        await redis_client.set("test:delete", "value")
        assert await redis_client.exists("test:delete") is True

        result = await redis_client.delete("test:delete")
        assert result is True
        assert await redis_client.exists("test:delete") is False

    async def test_delete_nonexistent(self, redis_client: RedisClient) -> None:
        """Test delete of non-existent key."""
        result = await redis_client.delete("test:nonexistent:12345")
        assert result is False

    async def test_exists(self, redis_client: RedisClient) -> None:
        """Test exists operation."""
        assert await redis_client.exists("test:exists") is False

        await redis_client.set("test:exists", "value")
        assert await redis_client.exists("test:exists") is True

        # Cleanup
        await redis_client.delete("test:exists")

    async def test_expire(self, redis_client: RedisClient) -> None:
        """Test setting expiration on existing key."""
        await redis_client.set("test:expire2", "value")

        result = await redis_client.expire("test:expire2", 3600)
        assert result is True

        ttl = await redis_client.ttl("test:expire2")
        assert ttl > 0

        # Cleanup
        await redis_client.delete("test:expire2")

    async def test_ping(self, redis_client: RedisClient) -> None:
        """Test ping health check."""
        result = await redis_client.ping()
        assert result is True


@pytest.mark.integration
class TestRedisPubSub:
    """Tests for Redis Pub/Sub operations."""

    async def test_publish(self, redis_client: RedisClient) -> None:
        """Test publishing a message to a channel."""
        # Publish returns the number of subscribers (0 if none)
        result = await redis_client.publish("test:channel", "test_message")
        assert isinstance(result, int)
        assert result >= 0

    async def test_publish_with_tenant(self, redis_client: RedisClient) -> None:
        """Test publishing to a tenant-isolated channel."""
        result = await redis_client.publish_with_tenant(
            "acme", "notifications", "hello"
        )
        assert isinstance(result, int)
        assert result >= 0


@pytest.mark.integration
class TestRedisTenantIsolation:
    """Tests for tenant-isolated operations."""

    async def test_set_with_tenant(self, redis_client: RedisClient) -> None:
        """Test tenant-isolated set operation."""
        await redis_client.set_with_tenant("acme", "session:123", {"user": "test"})
        value = await redis_client.get_with_tenant("acme", "session:123")

        assert value == {"user": "test"}

        # Cleanup
        await redis_client.delete_with_tenant("acme", "session:123")

    async def test_tenant_isolation(self, redis_client: RedisClient) -> None:
        """Test that tenants are isolated from each other."""
        await redis_client.set_with_tenant("tenant1", "key", "value1")
        await redis_client.set_with_tenant("tenant2", "key", "value2")

        value1 = await redis_client.get_with_tenant("tenant1", "key")
        value2 = await redis_client.get_with_tenant("tenant2", "key")

        assert value1 == "value1"
        assert value2 == "value2"

        # Cleanup
        await redis_client.delete_with_tenant("tenant1", "key")
        await redis_client.delete_with_tenant("tenant2", "key")

    async def test_delete_tenant_keys(self, redis_client: RedisClient) -> None:
        """Test deleting all keys for a tenant."""
        await redis_client.set_with_tenant("cleanup_test", "key1", "value1")
        await redis_client.set_with_tenant("cleanup_test", "key2", "value2")
        await redis_client.set_with_tenant("cleanup_test", "key3", "value3")

        deleted = await redis_client.delete_tenant_keys("cleanup_test")
        assert deleted == 3

        # Verify all keys are deleted
        assert await redis_client.exists_with_tenant("cleanup_test", "key1") is False
        assert await redis_client.exists_with_tenant("cleanup_test", "key2") is False
        assert await redis_client.exists_with_tenant("cleanup_test", "key3") is False


@pytest.mark.integration
class TestRedisErrors:
    """Tests for Redis error handling."""

    async def test_get_without_connect_raises_error(
        self, settings: Settings
    ) -> None:
        """Test that operations without connect raise error."""
        client = RedisClient(settings)

        with pytest.raises(RedisError) as exc_info:
            await client.get("test")

        assert "not connected" in str(exc_info.value)

    async def test_get_redis_without_init_raises_error(self) -> None:
        """Test that get_redis without init raises error."""
        await close_redis()

        with pytest.raises(RedisError) as exc_info:
            get_redis()

        assert "not initialized" in str(exc_info.value)
