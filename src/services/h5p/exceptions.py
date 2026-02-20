# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Custom exceptions for H5P service.

This module defines the exception hierarchy for H5P operations:
- H5PError: Base exception for all H5P-related errors
- H5PAPIError: Errors from the H5P API (Creatiq)
- H5PValidationError: Content validation errors
- H5PContentNotFoundError: Content not found
- H5PConversionError: AI content to H5P conversion errors
"""


class H5PError(Exception):
    """Base exception for all H5P-related errors.

    All H5P service exceptions inherit from this base class,
    allowing for broad exception catching when needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict | None = None):
        """Initialize H5P error.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation with details if available."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class H5PAPIError(H5PError):
    """Error from H5P API (Creatiq).

    Raised when the H5P API returns an error response or
    is unreachable.

    Attributes:
        status_code: HTTP status code from API response.
        response_body: Raw response body if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        details: dict | None = None,
    ):
        """Initialize H5P API error.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code from API response.
            response_body: Raw response body if available.
            details: Optional dictionary with additional error context.
        """
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation with status code."""
        base = f"{self.message}"
        if self.status_code:
            base = f"[{self.status_code}] {base}"
        if self.details:
            base = f"{base} - Details: {self.details}"
        return base


class H5PValidationError(H5PError):
    """Content validation error against H5P schema.

    Raised when generated content does not match the required
    H5P schema for a content type.

    Attributes:
        content_type: The H5P content type being validated.
        validation_errors: List of specific validation errors.
    """

    def __init__(
        self,
        message: str,
        content_type: str | None = None,
        validation_errors: list[str] | None = None,
        details: dict | None = None,
    ):
        """Initialize H5P validation error.

        Args:
            message: Human-readable error description.
            content_type: The H5P content type being validated.
            validation_errors: List of specific validation errors.
            details: Optional dictionary with additional error context.
        """
        self.content_type = content_type
        self.validation_errors = validation_errors or []
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation with validation errors."""
        base = self.message
        if self.content_type:
            base = f"[{self.content_type}] {base}"
        if self.validation_errors:
            errors = "; ".join(self.validation_errors)
            base = f"{base} - Errors: {errors}"
        return base


class H5PContentNotFoundError(H5PError):
    """H5P content not found.

    Raised when requested H5P content does not exist.

    Attributes:
        content_id: The ID of the content that was not found.
    """

    def __init__(
        self,
        message: str,
        content_id: str | None = None,
        details: dict | None = None,
    ):
        """Initialize H5P content not found error.

        Args:
            message: Human-readable error description.
            content_id: The ID of the content that was not found.
            details: Optional dictionary with additional error context.
        """
        self.content_id = content_id
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation with content ID."""
        base = self.message
        if self.content_id:
            base = f"{base} (content_id: {self.content_id})"
        return base


class H5PConversionError(H5PError):
    """AI content to H5P params conversion error.

    Raised when converting AI-generated content to H5P params format fails.

    Attributes:
        content_type: The target H5P content type.
        source_format: Description of the source format.
    """

    def __init__(
        self,
        message: str,
        content_type: str | None = None,
        source_format: str | None = None,
        details: dict | None = None,
    ):
        """Initialize H5P conversion error.

        Args:
            message: Human-readable error description.
            content_type: The target H5P content type.
            source_format: Description of the source format.
            details: Optional dictionary with additional error context.
        """
        self.content_type = content_type
        self.source_format = source_format
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return string representation with conversion context."""
        base = self.message
        if self.content_type:
            base = f"[{self.content_type}] {base}"
        if self.source_format:
            base = f"{base} (from: {self.source_format})"
        return base
