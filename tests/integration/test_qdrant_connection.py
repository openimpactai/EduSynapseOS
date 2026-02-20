# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Qdrant vector database connection.

These tests require a running Qdrant instance.
Run with: pytest tests/integration/test_qdrant_connection.py -v

Prerequisites:
    - Qdrant running at localhost:6333
"""

import uuid

import pytest

from src.core.config.settings import Settings, clear_settings_cache
from src.infrastructure.vectors.qdrant_client import (
    QdrantError,
    QdrantVectorClient,
    SearchResult,
    close_qdrant,
    get_qdrant,
    init_qdrant,
)


@pytest.fixture
def settings() -> Settings:
    """Provide fresh settings for each test."""
    clear_settings_cache()
    return Settings()


@pytest.fixture
async def initialized_qdrant(settings: Settings) -> None:
    """Initialize and cleanup the Qdrant connection."""
    await init_qdrant(settings)
    yield
    await close_qdrant()


@pytest.fixture
async def qdrant_client(settings: Settings) -> QdrantVectorClient:
    """Provide a Qdrant client for testing."""
    client = QdrantVectorClient(settings)
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
def test_collection_name() -> str:
    """Generate a unique test collection name."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_vector() -> list[float]:
    """Provide a test vector with 384 dimensions."""
    return [0.1] * 384


@pytest.mark.integration
class TestQdrantInitialization:
    """Tests for Qdrant initialization."""

    async def test_init_creates_client(self, settings: Settings) -> None:
        """Test that initialization creates the Qdrant client."""
        await init_qdrant(settings)

        try:
            client = get_qdrant()
            assert client is not None
        finally:
            await close_qdrant()

    async def test_close_clears_state(self, settings: Settings) -> None:
        """Test that close clears the module state."""
        await init_qdrant(settings)
        await close_qdrant()

        with pytest.raises(QdrantError) as exc_info:
            get_qdrant()

        assert "not initialized" in str(exc_info.value)

    async def test_ping(self, qdrant_client: QdrantVectorClient) -> None:
        """Test ping health check."""
        result = await qdrant_client.ping()
        assert result is True


@pytest.mark.integration
class TestQdrantCollectionManagement:
    """Tests for Qdrant collection operations."""

    async def test_create_collection(
        self, qdrant_client: QdrantVectorClient, test_collection_name: str
    ) -> None:
        """Test creating a collection."""
        try:
            await qdrant_client.create_collection(
                test_collection_name, vector_size=384, distance="Cosine"
            )

            exists = await qdrant_client.collection_exists(test_collection_name)
            assert exists is True
        finally:
            await qdrant_client.delete_collection(test_collection_name)

    async def test_delete_collection(
        self, qdrant_client: QdrantVectorClient, test_collection_name: str
    ) -> None:
        """Test deleting a collection."""
        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        result = await qdrant_client.delete_collection(test_collection_name)
        assert result is True

        exists = await qdrant_client.collection_exists(test_collection_name)
        assert exists is False

    async def test_delete_nonexistent_collection(
        self, qdrant_client: QdrantVectorClient
    ) -> None:
        """Test deleting a non-existent collection."""
        result = await qdrant_client.delete_collection(
            f"nonexistent_{uuid.uuid4().hex[:8]}"
        )
        assert result is False

    async def test_collection_exists(
        self, qdrant_client: QdrantVectorClient, test_collection_name: str
    ) -> None:
        """Test checking if collection exists."""
        exists_before = await qdrant_client.collection_exists(test_collection_name)
        assert exists_before is False

        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        try:
            exists_after = await qdrant_client.collection_exists(test_collection_name)
            assert exists_after is True
        finally:
            await qdrant_client.delete_collection(test_collection_name)


@pytest.mark.integration
class TestQdrantVectorOperations:
    """Tests for Qdrant vector operations."""

    async def test_upsert_and_search(
        self,
        qdrant_client: QdrantVectorClient,
        test_collection_name: str,
        test_vector: list[float],
    ) -> None:
        """Test upserting and searching vectors."""
        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        try:
            # Upsert a point
            points = [
                {
                    "id": "test-point-1",
                    "vector": test_vector,
                    "payload": {"topic": "math", "content": "test content"},
                }
            ]
            await qdrant_client.upsert(test_collection_name, points)

            # Search
            results = await qdrant_client.search(
                test_collection_name, query_vector=test_vector, limit=5
            )

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].id == "test-point-1"
            assert results[0].payload["topic"] == "math"
            assert results[0].score > 0.9  # Same vector should have high score
        finally:
            await qdrant_client.delete_collection(test_collection_name)

    async def test_upsert_multiple_points(
        self,
        qdrant_client: QdrantVectorClient,
        test_collection_name: str,
    ) -> None:
        """Test upserting multiple points."""
        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        try:
            points = [
                {
                    "id": f"point-{i}",
                    "vector": [0.1 * (i + 1)] * 384,
                    "payload": {"index": i},
                }
                for i in range(5)
            ]
            await qdrant_client.upsert(test_collection_name, points)

            # Search with first vector
            results = await qdrant_client.search(
                test_collection_name, query_vector=[0.1] * 384, limit=5
            )

            assert len(results) == 5
        finally:
            await qdrant_client.delete_collection(test_collection_name)

    async def test_search_with_filter(
        self,
        qdrant_client: QdrantVectorClient,
        test_collection_name: str,
        test_vector: list[float],
    ) -> None:
        """Test searching with filter conditions."""
        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        try:
            points = [
                {
                    "id": "point-math",
                    "vector": test_vector,
                    "payload": {"subject": "math"},
                },
                {
                    "id": "point-science",
                    "vector": test_vector,
                    "payload": {"subject": "science"},
                },
            ]
            await qdrant_client.upsert(test_collection_name, points)

            # Search with filter
            results = await qdrant_client.search(
                test_collection_name,
                query_vector=test_vector,
                filter_conditions={"subject": "math"},
            )

            assert len(results) == 1
            assert results[0].id == "point-math"
        finally:
            await qdrant_client.delete_collection(test_collection_name)

    async def test_delete_points(
        self,
        qdrant_client: QdrantVectorClient,
        test_collection_name: str,
        test_vector: list[float],
    ) -> None:
        """Test deleting points."""
        await qdrant_client.create_collection(test_collection_name, vector_size=384)

        try:
            points = [{"id": "point-to-delete", "vector": test_vector, "payload": {}}]
            await qdrant_client.upsert(test_collection_name, points)

            # Delete point
            await qdrant_client.delete_points(test_collection_name, ["point-to-delete"])

            # Verify deletion
            results = await qdrant_client.search(
                test_collection_name, query_vector=test_vector, limit=5
            )
            assert len(results) == 0
        finally:
            await qdrant_client.delete_collection(test_collection_name)


@pytest.mark.integration
class TestQdrantTenantIsolation:
    """Tests for tenant-isolated operations."""

    async def test_create_tenant_collection(
        self, qdrant_client: QdrantVectorClient
    ) -> None:
        """Test creating a tenant-isolated collection."""
        tenant_code = f"test_{uuid.uuid4().hex[:8]}"
        collection = "episodic_memories"

        try:
            await qdrant_client.create_tenant_collection(
                tenant_code, collection, vector_size=384
            )

            full_name = f"tenant_{tenant_code}_{collection}"
            exists = await qdrant_client.collection_exists(full_name)
            assert exists is True
        finally:
            await qdrant_client.delete_tenant_collection(tenant_code, collection)

    async def test_tenant_isolation(
        self, qdrant_client: QdrantVectorClient, test_vector: list[float]
    ) -> None:
        """Test that tenants are isolated from each other."""
        tenant1 = f"tenant1_{uuid.uuid4().hex[:8]}"
        tenant2 = f"tenant2_{uuid.uuid4().hex[:8]}"
        collection = "memories"

        try:
            # Create collections for both tenants
            await qdrant_client.create_tenant_collection(
                tenant1, collection, vector_size=384
            )
            await qdrant_client.create_tenant_collection(
                tenant2, collection, vector_size=384
            )

            # Insert data for tenant1
            points1 = [
                {"id": "point-1", "vector": test_vector, "payload": {"tenant": 1}}
            ]
            await qdrant_client.upsert_with_tenant(tenant1, collection, points1)

            # Insert data for tenant2
            points2 = [
                {"id": "point-1", "vector": test_vector, "payload": {"tenant": 2}}
            ]
            await qdrant_client.upsert_with_tenant(tenant2, collection, points2)

            # Search in tenant1 should only return tenant1's data
            results1 = await qdrant_client.search_with_tenant(
                tenant1, collection, test_vector
            )
            assert len(results1) == 1
            assert results1[0].payload["tenant"] == 1

            # Search in tenant2 should only return tenant2's data
            results2 = await qdrant_client.search_with_tenant(
                tenant2, collection, test_vector
            )
            assert len(results2) == 1
            assert results2[0].payload["tenant"] == 2
        finally:
            await qdrant_client.delete_tenant_collection(tenant1, collection)
            await qdrant_client.delete_tenant_collection(tenant2, collection)


@pytest.mark.integration
class TestQdrantErrors:
    """Tests for Qdrant error handling."""

    async def test_get_without_connect_raises_error(
        self, settings: Settings
    ) -> None:
        """Test that operations without connect raise error."""
        client = QdrantVectorClient(settings)

        with pytest.raises(QdrantError) as exc_info:
            await client.ping()

        assert "not connected" in str(exc_info.value)

    async def test_get_qdrant_without_init_raises_error(self) -> None:
        """Test that get_qdrant without init raises error."""
        await close_qdrant()

        with pytest.raises(QdrantError) as exc_info:
            get_qdrant()

        assert "not initialized" in str(exc_info.value)
