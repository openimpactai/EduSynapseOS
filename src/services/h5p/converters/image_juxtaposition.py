# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Image Juxtaposition H5P Converter.

Converts AI-generated before/after comparison data to H5P.ImageJuxtaposition format.
Requires images to be uploaded or generated separately.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ImageJuxtapositionConverter(BaseH5PConverter):
    """Converter for H5P.ImageJuxtaposition content type.

    AI Input Format:
        {
            "title": "Deforestation Impact",
            "before": {
                "description": "A lush green Amazon rainforest",
                "label": "1990",
                "alt": "Amazon rainforest in 1990"
            },
            "after": {
                "description": "Same area with cleared land",
                "label": "2020",
                "alt": "Amazon rainforest in 2020"
            },
            "startingPosition": 50
        }

    Note: This content type requires media. Image descriptions are used
    for AI image generation, and must be replaced with actual image paths.
    """

    @property
    def content_type(self) -> str:
        return "image-juxtaposition"

    @property
    def library(self) -> str:
        return "H5P.ImageJuxtaposition 1.4"

    @property
    def category(self) -> str:
        return "media"

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    @property
    def bloom_levels(self) -> list[str]:
        return ["understand", "analyze"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "before" not in ai_content:
            raise H5PValidationError(
                message="Missing 'before' field",
                content_type=self.content_type,
            )

        if "after" not in ai_content:
            raise H5PValidationError(
                message="Missing 'after' field",
                content_type=self.content_type,
            )

        before = ai_content.get("before", {})
        after = ai_content.get("after", {})

        if not before.get("description") and not before.get("label"):
            raise H5PValidationError(
                message="'before' must have 'description' or 'label'",
                content_type=self.content_type,
            )

        if not after.get("description") and not after.get("label"):
            raise H5PValidationError(
                message="'after' must have 'description' or 'label'",
                content_type=self.content_type,
            )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P ImageJuxtaposition format."""
        title = ai_content.get("title", "Image Comparison")
        before = ai_content.get("before", {})
        after = ai_content.get("after", {})
        starting_position = ai_content.get("startingPosition", 50)

        # Extract before image data
        before_description = before.get("description", "")
        before_label = before.get("label", "Before")
        before_alt = before.get("alt", before_description[:100] if before_description else before_label)

        # Extract after image data
        after_description = after.get("description", "")
        after_label = after.get("label", "After")
        after_alt = after.get("alt", after_description[:100] if after_description else after_label)

        h5p_params = {
            "title": title,
            "imageBefore": {
                "path": "",  # Must be set by caller after image upload
                "alt": before_alt,
            },
            "imageBefore_label": before_label,
            "imageAfter": {
                "path": "",  # Must be set by caller after image upload
                "alt": after_alt,
            },
            "imageAfter_label": after_label,
            "startingPosition": int(starting_position),
        }

        return h5p_params
