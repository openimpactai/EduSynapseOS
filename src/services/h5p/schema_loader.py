# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""H5P Schema Loader.

This module loads H5P content type schemas from config/h5p-schemas/ directory.
It provides a central source of truth for AI input formats, H5P params formats,
and conversion rules.

The schema loader eliminates code duplication by providing schemas from config
files instead of hardcoding them in workflow and tool code.

Usage:
    from src.services.h5p.schema_loader import H5PSchemaLoader

    loader = H5PSchemaLoader()

    # Get schema for a content type
    schema = loader.get_schema("multiple-choice")

    # Get AI input format description for LLM prompts
    prompt_text = loader.get_ai_prompt_schema("multiple-choice")

    # List all available content types
    types = loader.list_content_types()
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base path for schema files
CONFIG_BASE_PATH = Path(__file__).parent.parent.parent.parent / "config" / "h5p-schemas"


class H5PSchemaLoader:
    """Loads and caches H5P content type schemas from config files.

    This class provides:
    - Schema loading from config/h5p-schemas/ JSON files
    - Caching for performance
    - AI prompt generation from schemas
    - Content type listing and categorization

    The schemas are the single source of truth for content type formats,
    eliminating duplication in workflow and tool code.
    """

    _instance: "H5PSchemaLoader | None" = None
    _schemas: dict[str, dict[str, Any]]
    _content_types: dict[str, dict[str, Any]]
    _initialized: bool

    def __new__(cls) -> "H5PSchemaLoader":
        """Singleton pattern for schema loader."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize schema loader."""
        if self._initialized:
            return

        self._schemas = {}
        self._content_types = {}
        self._load_all_schemas()
        self._initialized = True

    def _load_all_schemas(self) -> None:
        """Load all schema files from config directory."""
        if not CONFIG_BASE_PATH.exists():
            logger.warning("H5P schema config directory not found: %s", CONFIG_BASE_PATH)
            return

        # Load master content-types.json first
        content_types_file = CONFIG_BASE_PATH / "content-types.json"
        if content_types_file.exists():
            try:
                with open(content_types_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._content_types = data.get("content_types", {})
                    logger.debug("Loaded master content types: %d types", len(self._content_types))
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load content-types.json: %s", e)

        # Load individual schema files from category directories
        categories = ["assessment", "vocabulary", "learning", "game", "media"]

        for category in categories:
            category_path = CONFIG_BASE_PATH / category
            if not category_path.exists():
                continue

            for schema_file in category_path.glob("*.json"):
                content_type = schema_file.stem  # e.g., "multiple-choice"
                try:
                    with open(schema_file, "r", encoding="utf-8") as f:
                        schema = json.load(f)
                        self._schemas[content_type] = schema
                        logger.debug("Loaded schema: %s", content_type)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error("Failed to load schema %s: %s", schema_file, e)

        logger.info("H5PSchemaLoader initialized with %d schemas", len(self._schemas))

    def get_schema(self, content_type: str) -> dict[str, Any] | None:
        """Get complete schema for a content type.

        Args:
            content_type: Content type identifier (e.g., "multiple-choice").

        Returns:
            Complete schema dict or None if not found.
        """
        # Normalize content type (replace underscores with hyphens)
        normalized = content_type.replace("_", "-").lower()
        return self._schemas.get(normalized)

    def get_ai_input_format(self, content_type: str) -> dict[str, Any] | None:
        """Get AI input format for a content type.

        Args:
            content_type: Content type identifier.

        Returns:
            AI input format dict or None if not found.
        """
        schema = self.get_schema(content_type)
        if schema:
            return schema.get("ai_input_format")
        return None

    def get_h5p_params_format(self, content_type: str) -> dict[str, Any] | None:
        """Get H5P params format for a content type.

        Args:
            content_type: Content type identifier.

        Returns:
            H5P params format dict or None if not found.
        """
        schema = self.get_schema(content_type)
        if schema:
            return schema.get("h5p_params_format")
        return None

    def get_library(self, content_type: str) -> str | None:
        """Get H5P library identifier for a content type.

        Args:
            content_type: Content type identifier.

        Returns:
            Library string (e.g., "H5P.MultiChoice 1.16") or None.
        """
        schema = self.get_schema(content_type)
        if schema:
            return schema.get("library")
        return None

    def get_ai_prompt_schema(self, content_type: str) -> str:
        """Get formatted schema documentation for LLM prompts.

        This method generates a human-readable schema description
        that can be included in generator prompts to ensure the LLM
        produces correctly formatted content.

        Args:
            content_type: Content type identifier.

        Returns:
            Formatted schema string for LLM prompt, or default message if not found.
        """
        schema = self.get_schema(content_type)

        if not schema:
            return f"""Content Type: {content_type}
Return a valid JSON object with:
- "title": Content title
- Type-specific fields based on the content type

Ensure the JSON is well-formed and follows H5P conventions."""

        ai_format = schema.get("ai_input_format", {})

        # Build prompt text
        prompt_parts = [f"REQUIRED JSON SCHEMA for {content_type}:"]

        # Add schema structure
        if "schema" in ai_format:
            prompt_parts.append(json.dumps(ai_format["schema"], indent=2))

        # Add example
        if "example" in ai_format:
            prompt_parts.append("\nExample:")
            prompt_parts.append(json.dumps(ai_format["example"], indent=2))

        # Add description
        if "description" in ai_format:
            prompt_parts.append(f"\nNote: {ai_format['description']}")

        # Add conversion notes if available
        conversion_notes = schema.get("conversion_notes", {})
        if conversion_notes:
            notes = []
            for key, value in conversion_notes.items():
                if isinstance(value, str):
                    notes.append(f"- {value}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        notes.append(f"- {k}: {v}")
            if notes:
                prompt_parts.append("\nImportant:")
                prompt_parts.extend(notes)

        return "\n".join(prompt_parts)

    def get_tool_schema(self, content_type: str) -> dict[str, Any] | None:
        """Get schema in format suitable for tool responses.

        Args:
            content_type: Content type identifier.

        Returns:
            Schema dict formatted for tool response, or None.
        """
        schema = self.get_schema(content_type)
        if not schema:
            return None

        return {
            "content_type": content_type,
            "library": schema.get("library"),
            "name": schema.get("name"),
            "category": schema.get("category"),
            "description": schema.get("description"),
            "ai_input_format": schema.get("ai_input_format"),
            "h5p_params_format": schema.get("h5p_params_format"),
            "conversion_notes": schema.get("conversion_notes"),
            "requires_media": schema.get("requires_media", False),
            "ai_support": schema.get("ai_support", "full"),
        }

    def list_content_types(self) -> list[str]:
        """List all available content types.

        Returns:
            List of content type identifiers.
        """
        return list(self._schemas.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List content types in a category.

        Args:
            category: Category name (assessment, vocabulary, learning, game, media).

        Returns:
            List of content type identifiers in the category.
        """
        result = []
        for content_type, schema in self._schemas.items():
            if schema.get("category") == category:
                result.append(content_type)
        return result

    def list_by_ai_support(self, support_level: str) -> list[str]:
        """List content types by AI support level.

        Args:
            support_level: Support level ("full" or "partial").

        Returns:
            List of content type identifiers with the support level.
        """
        result = []
        for content_type, schema in self._schemas.items():
            if schema.get("ai_support", "full") == support_level:
                result.append(content_type)
        return result

    def get_content_type_info(self, content_type: str) -> dict[str, Any] | None:
        """Get basic info about a content type from master list.

        Args:
            content_type: Content type identifier.

        Returns:
            Content type info dict or None.
        """
        normalized = content_type.replace("_", "-").lower()
        return self._content_types.get(normalized)

    def get_all_content_types_info(self) -> dict[str, dict[str, Any]]:
        """Get info about all content types from master list.

        Returns:
            Dict mapping content type to info dict.
        """
        return self._content_types.copy()

    def has_schema(self, content_type: str) -> bool:
        """Check if schema exists for a content type.

        Args:
            content_type: Content type identifier.

        Returns:
            True if schema exists.
        """
        normalized = content_type.replace("_", "-").lower()
        return normalized in self._schemas

    def supports_media(self, content_type: str) -> bool:
        """Check if content type supports or requires media/images.

        Args:
            content_type: Content type identifier.

        Returns:
            True if content type can include media.
        """
        # First check individual schema
        schema = self.get_schema(content_type)
        if schema:
            if schema.get("requires_media", False):
                return True
            if schema.get("supports_media", False):
                return True

        # Fall back to master content types list
        ct_info = self.get_content_type_info(content_type)
        if ct_info:
            if ct_info.get("requires_media", False):
                return True
            if ct_info.get("supports_media", False):
                return True

        return False

    def requires_media(self, content_type: str) -> bool:
        """Check if content type requires media (cannot work without it).

        Args:
            content_type: Content type identifier.

        Returns:
            True if content type requires media to function.
        """
        schema = self.get_schema(content_type)
        if schema and schema.get("requires_media", False):
            return True

        ct_info = self.get_content_type_info(content_type)
        if ct_info and ct_info.get("requires_media", False):
            return True

        return False

    def get_all_content_types(self) -> dict[str, dict[str, Any]]:
        """Get all content types with their info.

        Alias for get_all_content_types_info for convenience.

        Returns:
            Dict mapping content type to info dict.
        """
        return self.get_all_content_types_info()


# Convenience function for quick schema access
@lru_cache(maxsize=64)
def get_schema(content_type: str) -> dict[str, Any] | None:
    """Get schema for a content type.

    Convenience function that uses the singleton loader.
    Results are cached for performance.

    Args:
        content_type: Content type identifier.

    Returns:
        Complete schema dict or None.
    """
    return H5PSchemaLoader().get_schema(content_type)


@lru_cache(maxsize=64)
def get_ai_prompt_schema(content_type: str) -> str:
    """Get AI prompt schema for a content type.

    Convenience function that uses the singleton loader.
    Results are cached for performance.

    Args:
        content_type: Content type identifier.

    Returns:
        Formatted schema string for LLM prompt.
    """
    return H5PSchemaLoader().get_ai_prompt_schema(content_type)
