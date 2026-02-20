# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Image Sequencing H5P Converter.

Converts AI-generated image sequencing content to H5P.ImageSequencing format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ImageSequencingConverter(BaseH5PConverter):
    """Converter for H5P.ImageSequencing content type.

    AI Input Format:
        {
            "title": "Life Cycle of a Butterfly",
            "description": "Arrange the images in the correct order",
            "images": [
                {"url": "egg.jpg", "alt": "Egg", "description": "Stage 1: Egg"},
                {"url": "caterpillar.jpg", "alt": "Caterpillar", "description": "Stage 2: Larva"},
                {"url": "chrysalis.jpg", "alt": "Chrysalis", "description": "Stage 3: Pupa"},
                {"url": "butterfly.jpg", "alt": "Butterfly", "description": "Stage 4: Adult"}
            ]
        }

    Images are provided in correct order, H5P shuffles them for the user.
    """

    @property
    def content_type(self) -> str:
        return "image-sequencing"

    @property
    def library(self) -> str:
        return "H5P.ImageSequencing 1.1"

    @property
    def category(self) -> str:
        return "game"

    @property
    def bloom_levels(self) -> list[str]:
        return ["analyze"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

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
                message="At least 2 images are required",
                content_type=self.content_type,
            )

        for i, img in enumerate(images):
            if "url" not in img:
                raise H5PValidationError(
                    message=f"Image {i+1} missing 'url'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P ImageSequencing format."""
        images = ai_content.get("images", [])
        description = ai_content.get("description", self.get_l10n(language).get("defaultDescription", "Arrange the images in the correct order"))

        if not images:
            raise H5PValidationError(
                message="No images provided",
                content_type=self.content_type,
            )

        # Convert images to H5P format (in correct order)
        h5p_sequence = []
        for img in images:
            h5p_image = {
                "image": {
                    "path": img.get("url", ""),
                    "alt": img.get("alt", "Image"),
                },
            }

            if img.get("description"):
                h5p_image["imageDescription"] = img["description"]

            h5p_sequence.append(h5p_image)

        h5p_params = {
            "taskDescription": f"<p>{description}</p>",
            "sequenceImages": h5p_sequence,
            "behaviour": self.get_default_behavior(),
            "l10n": self.get_l10n(language).get("l10n", {}),
            "a11y": self.get_l10n(language).get("a11y", {}),
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for ImageSequencing."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "showSummary": True,
        }

