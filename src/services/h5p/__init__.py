# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""H5P Content Management Service.

This package provides services for H5P content creation, storage, and management.
It includes:
- H5PClient: Async HTTP client for H5P API (Creatiq integration)
- ContentStorageService: Draft save/load and content management
- H5P Converters: Transform AI-generated content to H5P params format

Usage:
    from src.services.h5p import H5PClient, ContentStorageService
    from src.services.h5p.converters import ConverterRegistry

    # Create H5P content
    client = H5PClient(api_url="...", api_key="...")
    content_id = await client.create_content(library="H5P.MultiChoice 1.16", params={...})

    # Convert AI content to H5P params
    registry = ConverterRegistry()
    converter = registry.get("multiple-choice")
    h5p_params = converter.convert(ai_content, language="en")
"""

from src.services.h5p.client import H5PClient
from src.services.h5p.converters import BaseH5PConverter, ConverterRegistry
from src.services.h5p.exceptions import (
    H5PAPIError,
    H5PContentNotFoundError,
    H5PConversionError,
    H5PError,
    H5PValidationError,
)
from src.services.h5p.schema_loader import H5PSchemaLoader, get_ai_prompt_schema, get_schema
from src.services.h5p.storage import ContentDraft, ContentStorageService

__all__ = [
    "H5PClient",
    "ContentStorageService",
    "ContentDraft",
    "BaseH5PConverter",
    "ConverterRegistry",
    "H5PSchemaLoader",
    "get_schema",
    "get_ai_prompt_schema",
    "H5PError",
    "H5PAPIError",
    "H5PValidationError",
    "H5PContentNotFoundError",
    "H5PConversionError",
]
