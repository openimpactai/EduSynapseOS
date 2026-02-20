# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Image Pairing H5P Converter.

Converts AI-generated image pairing content to H5P.ImagePair format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ImagePairingConverter(BaseH5PConverter):
    """Converter for H5P.ImagePair content type.

    AI Input Format:
        {
            "title": "Animal Habitats",
            "description": "Match each animal with its habitat",
            "pairs": [
                {
                    "image1": {"url": "fish.jpg", "alt": "Fish"},
                    "image2": {"url": "ocean.jpg", "alt": "Ocean"}
                },
                {
                    "image1": {"url": "camel.jpg", "alt": "Camel"},
                    "image2": {"url": "desert.jpg", "alt": "Desert"}
                }
            ]
        }

    Two images that should be matched together.
    """

    @property
    def content_type(self) -> str:
        return "image-pairing"

    @property
    def library(self) -> str:
        return "H5P.ImagePair 1.4"

    @property
    def category(self) -> str:
        return "game"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember"]

    @property
    def ai_support(self) -> str:
        return "partial"

    @property
    def requires_media(self) -> bool:
        return True

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "pairs" not in ai_content:
            raise H5PValidationError(
                message="Missing 'pairs' field",
                content_type=self.content_type,
            )

        pairs = ai_content.get("pairs", [])
        if len(pairs) < 2:
            raise H5PValidationError(
                message="At least 2 pairs are required",
                content_type=self.content_type,
            )

        for i, pair in enumerate(pairs):
            if "image1" not in pair or "image2" not in pair:
                raise H5PValidationError(
                    message=f"Pair {i+1} missing 'image1' or 'image2'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P ImagePair format."""
        pairs = ai_content.get("pairs", [])
        description = ai_content.get("description", "")

        if not pairs:
            raise H5PValidationError(
                message="No pairs provided",
                content_type=self.content_type,
            )

        # Convert pairs to H5P format
        h5p_cards = []
        for pair in pairs:
            image1 = pair.get("image1", {})
            image2 = pair.get("image2", {})

            h5p_card = {
                "image": {
                    "path": image1.get("url", ""),
                    "alt": image1.get("alt", "Image 1"),
                },
                "match": {
                    "path": image2.get("url", ""),
                    "alt": image2.get("alt", "Image 2"),
                },
            }
            h5p_cards.append(h5p_card)

        h5p_params = {
            "cards": h5p_cards,
            "behaviour": self.get_default_behavior(),
            "l10n": self.get_l10n(language),
        }

        if description:
            h5p_params["taskDescription"] = f"<p>{description}</p>"

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for ImagePair."""
        return {
            "enableRetry": True,
        }

