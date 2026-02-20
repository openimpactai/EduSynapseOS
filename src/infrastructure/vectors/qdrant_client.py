# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Qdrant vector database client for similarity search.

This module provides an async Qdrant client wrapper with tenant isolation
support. Tenant-specific collections are prefixed with tenant_{tenant_code}_
while shared collections have no prefix.

Collections:
- curriculum_knowledge: Shared curriculum content for RAG
- episodic_memories: Tenant-isolated student learning events
- student_interests: Tenant-isolated interest-based personalization
- successful_explanations: Tenant-shared successful teaching approaches

Example:
    from src.infrastructure.vectors import init_qdrant, get_qdrant

    # Initialize at startup
    await init_qdrant(settings)

    # Use the client
    qdrant = get_qdrant()

    # Search in shared collection
    results = await qdrant.search(
        "curriculum_knowledge",
        query_vector=embedding,
        limit=5
    )

    # Search in tenant-isolated collection
    results = await qdrant.search_with_tenant(
        "acme",
        "episodic_memories",
        query_vector=embedding,
        limit=5
    )
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

if TYPE_CHECKING:
    from src.core.config.settings import Settings

# Module-level state
_qdrant_client: Optional["QdrantVectorClient"] = None


class QdrantError(Exception):
    """Exception raised for Qdrant operation failures.

    Attributes:
        message: Human-readable error description.
        original_error: The underlying Qdrant error.
    """

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        """Initialize the Qdrant error.

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


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Attributes:
        id: Point ID in Qdrant.
        score: Similarity score.
        payload: Associated metadata.
    """

    id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorClient:
    """Async Qdrant client with tenant isolation support.

    This client wraps the qdrant-client async API and provides:
    - Tenant-isolated collection naming
    - Simplified search interface
    - Collection management

    Collections can be:
    - Shared: Used by all tenants (e.g., curriculum_knowledge)
    - Tenant-isolated: Prefixed with tenant_{code}_ (e.g., tenant_acme_episodic_memories)

    Attributes:
        settings: Application settings containing Qdrant configuration.

    Example:
        client = QdrantVectorClient(settings)
        await client.connect()

        # Shared collection
        results = await client.search("curriculum_knowledge", vector, limit=5)

        # Tenant-isolated collection
        results = await client.search_with_tenant("acme", "memories", vector)

        await client.close()
    """

    def __init__(self, settings: "Settings") -> None:
        """Initialize the Qdrant client.

        Args:
            settings: Application settings containing Qdrant configuration.
        """
        self._settings = settings
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        """Create the Qdrant client connection.

        Raises:
            QdrantError: If connection fails.
        """
        qdrant_settings = self._settings.qdrant
        api_key = (
            qdrant_settings.api_key.get_secret_value()
            if qdrant_settings.api_key
            else None
        )

        try:
            self._client = AsyncQdrantClient(
                host=qdrant_settings.host,
                port=qdrant_settings.http_port,
                grpc_port=qdrant_settings.grpc_port,
                api_key=api_key,
                prefer_grpc=qdrant_settings.prefer_grpc,
                timeout=qdrant_settings.timeout,
            )

            # Verify connection
            await self._client.get_collections()
        except Exception as e:
            raise QdrantError("Failed to connect to Qdrant", e) from e

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ensure_connected(self) -> AsyncQdrantClient:
        """Ensure the client is connected.

        Returns:
            The Qdrant client instance.

        Raises:
            QdrantError: If not connected.
        """
        if self._client is None:
            raise QdrantError("Qdrant client not connected. Call connect() first.")
        return self._client

    def _tenant_collection_name(self, tenant_code: str, collection: str) -> str:
        """Build a tenant-prefixed collection name.

        Args:
            tenant_code: The tenant code.
            collection: The base collection name.

        Returns:
            Collection name prefixed with tenant_{tenant_code}_
        """
        return f"tenant_{tenant_code}_{collection}"

    # ========== Collection management ==========

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        on_disk: bool = True,
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimension of the vectors.
            distance: Distance metric (Cosine, Euclid, Dot).
            on_disk: Whether to store vectors on disk.

        Raises:
            QdrantError: If collection creation fails.
        """
        client = self._ensure_connected()

        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }

        try:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, models.Distance.COSINE),
                    on_disk=on_disk,
                ),
            )
        except UnexpectedResponse as e:
            raise QdrantError(f"Failed to create collection: {collection_name}", e) from e

    async def create_tenant_collection(
        self,
        tenant_code: str,
        collection: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Create a tenant-isolated collection.

        Args:
            tenant_code: The tenant code.
            collection: Base collection name.
            vector_size: Dimension of the vectors.
            distance: Distance metric.

        Raises:
            QdrantError: If collection creation fails.
        """
        collection_name = self._tenant_collection_name(tenant_code, collection)
        await self.create_collection(collection_name, vector_size, distance)

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            True if collection was deleted, False if it didn't exist.

        Raises:
            QdrantError: If deletion fails.
        """
        client = self._ensure_connected()

        try:
            result = await client.delete_collection(collection_name)
            return result
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                return False
            raise QdrantError(f"Failed to delete collection: {collection_name}", e) from e

    async def delete_tenant_collection(self, tenant_code: str, collection: str) -> bool:
        """Delete a tenant-isolated collection.

        Args:
            tenant_code: The tenant code.
            collection: Base collection name.

        Returns:
            True if collection was deleted, False if it didn't exist.

        Raises:
            QdrantError: If deletion fails.
        """
        collection_name = self._tenant_collection_name(tenant_code, collection)
        return await self.delete_collection(collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection.

        Returns:
            True if the collection exists, False otherwise.

        Raises:
            QdrantError: If the check fails.
        """
        client = self._ensure_connected()

        try:
            result = await client.collection_exists(collection_name)
            return result
        except UnexpectedResponse as e:
            raise QdrantError(f"Failed to check collection: {collection_name}", e) from e

    # ========== Vector operations ==========

    async def upsert(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> None:
        """Upsert points into a collection.

        Args:
            collection_name: Name of the collection.
            points: List of points with id, vector, and payload.
                Each point should have: {"id": str, "vector": list[float], "payload": dict}

        Raises:
            QdrantError: If upsert fails.
        """
        client = self._ensure_connected()

        try:
            qdrant_points = [
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]

            await client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
            )
        except UnexpectedResponse as e:
            raise QdrantError(f"Failed to upsert points to: {collection_name}", e) from e

    async def upsert_with_tenant(
        self,
        tenant_code: str,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        """Upsert points into a tenant-isolated collection.

        Args:
            tenant_code: The tenant code.
            collection: Base collection name.
            points: List of points with id, vector, and payload.

        Raises:
            QdrantError: If upsert fails.
        """
        collection_name = self._tenant_collection_name(tenant_code, collection)
        await self.upsert(collection_name, points)

    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> None:
        """Delete points from a collection.

        Args:
            collection_name: Name of the collection.
            point_ids: List of point IDs to delete.

        Raises:
            QdrantError: If deletion fails.
        """
        client = self._ensure_connected()

        try:
            await client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
        except UnexpectedResponse as e:
            raise QdrantError(
                f"Failed to delete points from: {collection_name}", e
            ) from e

    # ========== Search operations ==========

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors in a collection.

        Uses the query_points API (qdrant-client >= 1.16).

        Args:
            collection_name: Name of the collection.
            query_vector: The query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional filter conditions.

        Returns:
            List of SearchResult objects.

        Raises:
            QdrantError: If search fails.
        """
        client = self._ensure_connected()

        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
            query_filter = models.Filter(must=must_conditions)

        try:
            response = await client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            return [
                SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                )
                for point in response.points
            ]
        except UnexpectedResponse as e:
            raise QdrantError(f"Failed to search in: {collection_name}", e) from e

    async def search_with_tenant(
        self,
        tenant_code: str,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search in a tenant-isolated collection.

        Args:
            tenant_code: The tenant code.
            collection: Base collection name.
            query_vector: The query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional filter conditions.

        Returns:
            List of SearchResult objects.

        Raises:
            QdrantError: If search fails.
        """
        collection_name = self._tenant_collection_name(tenant_code, collection)
        return await self.search(
            collection_name,
            query_vector,
            limit,
            score_threshold,
            filter_conditions,
        )

    # ========== Health check ==========

    async def ping(self) -> bool:
        """Check if Qdrant is reachable.

        Returns:
            True if Qdrant responds, False otherwise.
        """
        try:
            client = self._ensure_connected()
            await client.get_collections()
            return True
        except (QdrantError, Exception):
            return False


# ========== Module-level functions ==========


async def init_qdrant(settings: "Settings") -> None:
    """Initialize the global Qdrant client.

    This should be called once at application startup.

    Args:
        settings: Application settings containing Qdrant configuration.

    Raises:
        QdrantError: If connection fails.
    """
    global _qdrant_client

    _qdrant_client = QdrantVectorClient(settings)
    await _qdrant_client.connect()


async def close_qdrant() -> None:
    """Close the global Qdrant client.

    This should be called at application shutdown.
    """
    global _qdrant_client

    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


def get_qdrant() -> QdrantVectorClient:
    """Get the global Qdrant client.

    Returns:
        The QdrantVectorClient instance.

    Raises:
        QdrantError: If Qdrant has not been initialized.
    """
    if _qdrant_client is None:
        raise QdrantError("Qdrant not initialized. Call init_qdrant() first.")
    return _qdrant_client
