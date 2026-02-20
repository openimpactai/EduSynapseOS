# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Collage H5P Converter.

Converts AI-generated collage data to H5P.Collage format.
Requires images to be uploaded or generated separately.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class CollageConverter(BaseH5PConverter):
    """Converter for H5P.Collage content type.

    AI Input Format:
        {
            "layout": "1-2",
            "images": [
                {
                    "description": "Panoramic view of Grand Canyon",
                    "alt": "Grand Canyon panorama"
                }
            ]
        }

    Layout patterns:
        - "1-1": Two equal images side by side
        - "1-2": One large on left, two stacked on right
        - "2-1": Two stacked on left, one large on right
        - "1-1-1": Three equal images in a row
        - "2-2": Four images in 2x2 grid

    Note: This content type requires media. Image descriptions are used
    for AI image generation, and must be replaced with actual image paths.
    """

    LAYOUT_IMAGE_COUNTS = {
        "1-1": 2,
        "1-2": 3,
        "2-1": 3,
        "1-1-1": 3,
        "2-2": 4,
    }

    @property
    def content_type(self) -> str:
        return "collage"

    @property
    def library(self) -> str:
        return "H5P.Collage 0.3"

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
        return ["understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        layout = ai_content.get("layout", "1-2")
        if layout not in self.LAYOUT_IMAGE_COUNTS:
            raise H5PValidationError(
                message=f"Invalid layout: {layout}. Valid: {list(self.LAYOUT_IMAGE_COUNTS.keys())}",
                content_type=self.content_type,
            )

        if "images" not in ai_content:
            raise H5PValidationError(
                message="Missing 'images' field",
                content_type=self.content_type,
            )

        images = ai_content.get("images", [])
        expected_count = self.LAYOUT_IMAGE_COUNTS[layout]

        if len(images) < expected_count:
            raise H5PValidationError(
                message=f"Layout '{layout}' requires {expected_count} images, got {len(images)}",
                content_type=self.content_type,
            )

        for i, image in enumerate(images[:expected_count]):
            if not image.get("description") and not image.get("alt"):
                raise H5PValidationError(
                    message=f"Image {i+1} must have 'description' or 'alt'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Collage format."""
        layout = ai_content.get("layout", "1-2")
        images = ai_content.get("images", [])
        expected_count = self.LAYOUT_IMAGE_COUNTS.get(layout, 3)

        # Build clips in H5P format
        clips = []
        for i, image in enumerate(images[:expected_count]):
            description = image.get("description", "")
            alt = image.get("alt", description[:100] if description else f"Image {i + 1}")

            clip = {
                "image": {
                    "path": "",  # Must be set by caller after image upload
                    "alt": alt,
                },
            }
            clips.append(clip)

        h5p_params = {
            "collage": {
                "template": layout,
                "clips": clips,
                "spacing": 0,
                "frame": False,
            },
        }

        return h5p_params
