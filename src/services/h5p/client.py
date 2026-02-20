# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""H5P API Client for Creatiq integration.

This module provides an async HTTP client for interacting with the H5P
server (Creatiq) to create, update, retrieve, and manage H5P content.

The client handles:
- Content creation and updates
- Content retrieval and preview URLs
- Media uploads
- xAPI event webhooks

Example:
    client = H5PClient(
        api_url="https://h5p.edusynapse.com",
        api_key="your-api-key"
    )

    # Create content
    content_id = await client.create_content(
        library="H5P.MultiChoice 1.16",
        params={...},
        metadata={"title": "Quiz 1"}
    )

    # Get preview URL
    preview_url = client.get_preview_url(content_id)
"""

import json
import logging
from typing import Any

import aiohttp

from src.services.h5p.exceptions import (
    H5PAPIError,
    H5PContentNotFoundError,
    H5PValidationError,
)

logger = logging.getLogger(__name__)


class H5PClient:
    """Async HTTP client for H5P API (Creatiq).

    Provides methods to interact with the H5P server for content
    creation, management, and retrieval.

    Attributes:
        api_url: Base URL of the H5P API server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: int = 30,
    ):
        """Initialize the H5P client.

        Args:
            api_url: Base URL of the H5P API server.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def create_content(
        self,
        library: str,
        params: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        tenant_code: str | None = None,
    ) -> str:
        """Create new H5P content.

        Args:
            library: H5P library identifier (e.g., "H5P.MultiChoice 1.16").
            params: H5P content parameters.
            metadata: Optional content metadata (title, license, etc.).
            tenant_code: Optional tenant code for multi-tenant setup.

        Returns:
            Content ID of the created content.

        Raises:
            H5PAPIError: If the API returns an error.
            H5PValidationError: If content validation fails.
        """
        payload = {
            "library": library,
            "params": json.dumps(params),
            "metadata": metadata or {},
        }

        if tenant_code:
            payload["tenantCode"] = tenant_code

        logger.debug(
            "Creating H5P content: library=%s, tenant=%s",
            library,
            tenant_code,
        )

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/h5p/content",
                    json=payload,
                    headers=self._get_headers(),
                ) as response:
                    response_data = await response.json()

                    if response.status == 201 or response.status == 200:
                        # H5P server wraps response in {success, data: {contentId, ...}}
                        data = response_data.get("data", response_data)
                        content_id = data.get("contentId") or data.get("id") or response_data.get("contentId")
                        logger.info(
                            "Created H5P content: id=%s, library=%s",
                            content_id,
                            library,
                        )
                        return str(content_id)

                    if response.status == 400:
                        raise H5PValidationError(
                            message=response_data.get("message", "Validation failed"),
                            content_type=library,
                            validation_errors=response_data.get("errors", []),
                        )

                    raise H5PAPIError(
                        message=response_data.get("message", "Failed to create content"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    async def update_content(
        self,
        content_id: str,
        library: str,
        params: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update existing H5P content.

        Args:
            content_id: ID of the content to update.
            library: H5P library identifier.
            params: Updated H5P content parameters.
            metadata: Optional updated metadata.

        Returns:
            True if update was successful.

        Raises:
            H5PAPIError: If the API returns an error.
            H5PContentNotFoundError: If content does not exist.
        """
        payload = {
            "library": library,
            "params": json.dumps(params),
        }

        if metadata:
            payload["metadata"] = metadata

        logger.debug("Updating H5P content: id=%s", content_id)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.patch(
                    f"{self.api_url}/h5p/content/{content_id}",
                    json=payload,
                    headers=self._get_headers(),
                ) as response:
                    if response.status == 200:
                        logger.info("Updated H5P content: id=%s", content_id)
                        return True

                    response_data = await response.json()

                    if response.status == 404:
                        raise H5PContentNotFoundError(
                            message="Content not found",
                            content_id=content_id,
                        )

                    raise H5PAPIError(
                        message=response_data.get("message", "Failed to update content"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    async def get_content(
        self,
        content_id: str,
    ) -> dict[str, Any]:
        """Get H5P content by ID.

        Args:
            content_id: ID of the content to retrieve.

        Returns:
            Dictionary with content data including params and metadata.

        Raises:
            H5PAPIError: If the API returns an error.
            H5PContentNotFoundError: If content does not exist.
        """
        logger.debug("Getting H5P content: id=%s", content_id)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    f"{self.api_url}/h5p/content/{content_id}",
                    headers=self._get_headers(),
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        return response_data

                    if response.status == 404:
                        raise H5PContentNotFoundError(
                            message="Content not found",
                            content_id=content_id,
                        )

                    raise H5PAPIError(
                        message=response_data.get("message", "Failed to get content"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    async def delete_content(
        self,
        content_id: str,
    ) -> bool:
        """Delete H5P content.

        Args:
            content_id: ID of the content to delete.

        Returns:
            True if deletion was successful.

        Raises:
            H5PAPIError: If the API returns an error.
            H5PContentNotFoundError: If content does not exist.
        """
        logger.debug("Deleting H5P content: id=%s", content_id)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.delete(
                    f"{self.api_url}/h5p/content/{content_id}",
                    headers=self._get_headers(),
                ) as response:
                    if response.status == 200 or response.status == 204:
                        logger.info("Deleted H5P content: id=%s", content_id)
                        return True

                    response_data = await response.json()

                    if response.status == 404:
                        raise H5PContentNotFoundError(
                            message="Content not found",
                            content_id=content_id,
                        )

                    raise H5PAPIError(
                        message=response_data.get("message", "Failed to delete content"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    async def upload_media(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        tenant_code: str | None = None,
    ) -> dict[str, str]:
        """Upload media file to H5P server.

        Args:
            file_data: Binary file data.
            filename: Original filename.
            content_type: MIME type of the file.
            tenant_code: Optional tenant code.

        Returns:
            Dictionary with path and URL of uploaded file.

        Raises:
            H5PAPIError: If upload fails.
        """
        logger.debug("Uploading media: filename=%s, type=%s", filename, content_type)

        try:
            import base64

            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }

            payload = {
                "filename": filename,
                "data": base64.b64encode(file_data).decode("utf-8"),
                "contentType": content_type,
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/h5p/media/upload",
                    json=payload,
                    headers=headers,
                ) as response:
                    response_data = await response.json()

                    if response.status in (200, 201):
                        data = response_data.get("data", response_data)
                        url = data.get("url", "")
                        logger.info(
                            "Uploaded media: filename=%s, url=%s",
                            filename,
                            url,
                        )
                        return {
                            "path": url,
                            "url": url,
                            "mime": content_type,
                        }

                    raise H5PAPIError(
                        message=response_data.get("error", "Failed to upload media"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    async def attach_media(
        self,
        content_id: str,
        filenames: list[str],
        media_type: str = "images",
    ) -> dict[str, Any]:
        """Attach uploaded temp files to a content directory.

        Copies files from temp/uploads/ to content/{id}/{media_type}/.

        Args:
            content_id: H5P content ID.
            filenames: List of filenames to attach.
            media_type: Subfolder type: 'images', 'videos', or 'audios'.

        Returns:
            Dictionary with attached file info.
        """
        logger.debug("Attaching %s to content %s: %s", media_type, content_id, filenames)

        try:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/h5p/media/attach",
                    json={"contentId": content_id, "files": filenames, "mediaType": media_type},
                    headers=headers,
                ) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        data = response_data.get("data", response_data)
                        logger.info(
                            "Attached %d images to content %s",
                            data.get("count", 0),
                            content_id,
                        )
                        return data
                    logger.warning(
                        "Failed to attach media: %s", response_data
                    )
                    return {}

        except aiohttp.ClientError as e:
            logger.error("H5P attach media error: %s", str(e))
            return {}

    async def get_libraries(self) -> list[dict[str, Any]]:
        """Get list of installed H5P libraries.

        Returns:
            List of library information dictionaries.

        Raises:
            H5PAPIError: If the API returns an error.
        """
        logger.debug("Getting H5P libraries")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    f"{self.api_url}/h5p/libraries",
                    headers=self._get_headers(),
                ) as response:
                    if response.status == 200:
                        return await response.json()

                    response_data = await response.json()
                    raise H5PAPIError(
                        message=response_data.get("message", "Failed to get libraries"),
                        status_code=response.status,
                        response_body=json.dumps(response_data),
                    )

        except aiohttp.ClientError as e:
            logger.error("H5P API connection error: %s", str(e))
            raise H5PAPIError(
                message=f"Failed to connect to H5P API: {str(e)}",
                details={"error_type": type(e).__name__},
            ) from e

    def get_preview_url(self, content_id: str) -> str:
        """Get preview URL for H5P content.

        Args:
            content_id: ID of the content.

        Returns:
            Full preview URL.
        """
        return f"{self.api_url}/h5p/play/{content_id}"

    def get_edit_url(self, content_id: str) -> str:
        """Get edit URL for H5P content.

        Args:
            content_id: ID of the content.

        Returns:
            Full edit URL.
        """
        return f"{self.api_url}/h5p/edit/{content_id}"

    def get_embed_code(
        self,
        content_id: str,
        width: str = "100%",
        height: str = "400",
    ) -> str:
        """Get embed code for H5P content.

        Args:
            content_id: ID of the content.
            width: Width of the iframe.
            height: Height of the iframe.

        Returns:
            HTML iframe embed code.
        """
        embed_url = f"{self.api_url}/h5p/embed/{content_id}"
        return (
            f'<iframe src="{embed_url}" '
            f'width="{width}" height="{height}" '
            f'frameborder="0" allowfullscreen="allowfullscreen"></iframe>'
        )
