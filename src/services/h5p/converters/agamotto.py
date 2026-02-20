# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Agamotto H5P Converter.

Converts AI-generated image sequence data to H5P.Agamotto format.
Requires images to be uploaded or generated separately.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class AgamottoConverter(BaseH5PConverter):
    """Converter for H5P.Agamotto (Image Sequence) content type.

    AI Input Format:
        {
            "title": "Plant Growth Stages",
            "images": [
                {
                    "description": "A seed just planted in soil",
                    "label": "Week 1",
                    "alt": "Seed germination",
                    "text": "The seed absorbs water and begins to germinate."
                }
            ]
        }

    Note: This content type requires media. Image descriptions are used
    for AI image generation, and must be replaced with actual image paths.
    """

    @property
    def content_type(self) -> str:
        return "agamotto"

    @property
    def library(self) -> str:
        return "H5P.Agamotto 1.6"

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

        if "images" not in ai_content:
            raise H5PValidationError(
                message="Missing 'images' field",
                content_type=self.content_type,
            )

        images = ai_content.get("images", [])
        if len(images) < 2:
            raise H5PValidationError(
                message="At least 2 images are required for a sequence",
                content_type=self.content_type,
            )

        for i, image in enumerate(images):
            if "description" not in image and "label" not in image:
                raise H5PValidationError(
                    message=f"Image {i+1} must have 'description' or 'label'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Agamotto format."""
        title = ai_content.get("title", "Image Sequence")
        images = ai_content.get("images", [])

        # Build items in H5P format
        items = []
        for i, image in enumerate(images):
            description = image.get("description", "")
            label = image.get("label", f"Step {i + 1}")
            alt = image.get("alt", description[:100] if description else label)
            text = image.get("text", "")

            item = {
                "image": {
                    "path": "",  # Must be set by caller after image upload
                    "alt": alt,
                },
                "labelText": label,
            }

            if text:
                item["description"] = self.wrap_html(text)

            items.append(item)

        h5p_params = {
            "items": items,
            "behaviour": {
                "snap": True,
                "ticks": True,
            },
        }

        return h5p_params
