# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Image Hotspots H5P Converter.

Converts AI-generated hotspot data to H5P.ImageHotspots format.
Requires an image to be uploaded or generated separately.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class ImageHotspotsConverter(BaseH5PConverter):
    """Converter for H5P.ImageHotspots content type.

    AI Input Format:
        {
            "image_description": "A detailed diagram of an animal cell",
            "hotspots": [
                {
                    "label": "Nucleus",
                    "description": "The control center of the cell",
                    "position": {"x": 50, "y": 50}
                }
            ]
        }

    Note: This content type requires media. The image_description is used
    for AI image generation, and must be replaced with an actual image path.
    """

    @property
    def content_type(self) -> str:
        return "image-hotspots"

    @property
    def library(self) -> str:
        return "H5P.ImageHotspots 1.10"

    @property
    def category(self) -> str:
        return "learning"

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

        if "hotspots" not in ai_content:
            raise H5PValidationError(
                message="Missing 'hotspots' field",
                content_type=self.content_type,
            )

        hotspots = ai_content.get("hotspots", [])
        if not hotspots:
            raise H5PValidationError(
                message="At least one hotspot is required",
                content_type=self.content_type,
            )

        for i, hotspot in enumerate(hotspots):
            if "label" not in hotspot:
                raise H5PValidationError(
                    message=f"Hotspot {i+1} missing 'label'",
                    content_type=self.content_type,
                )
            if "description" not in hotspot:
                raise H5PValidationError(
                    message=f"Hotspot {i+1} missing 'description'",
                    content_type=self.content_type,
                )
            if "position" not in hotspot:
                raise H5PValidationError(
                    message=f"Hotspot {i+1} missing 'position'",
                    content_type=self.content_type,
                )

            pos = hotspot["position"]
            if "x" not in pos or "y" not in pos:
                raise H5PValidationError(
                    message=f"Hotspot {i+1} position must have 'x' and 'y'",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P ImageHotspots format."""
        hotspots = ai_content.get("hotspots", [])
        image_description = ai_content.get("image_description", "")

        # Build hotspots in H5P format
        h5p_hotspots = []
        for hotspot in hotspots:
            position = hotspot.get("position", {})
            label = hotspot.get("label", "")
            description = hotspot.get("description", "")

            h5p_hotspot = {
                "position": {
                    "x": float(position.get("x", 50)),
                    "y": float(position.get("y", 50)),
                },
                "header": label,
                "content": [
                    {
                        "library": "H5P.Text 1.1",
                        "params": {
                            "text": self.wrap_html(
                                f"<strong>{label}</strong><br/>{description}",
                                "p",
                            ),
                        },
                    }
                ],
            }
            h5p_hotspots.append(h5p_hotspot)

        h5p_params = {
            "image": {
                "path": "",  # Must be set by caller after image upload
                "alt": image_description,
                "width": 800,
                "height": 600,
            },
            "hotspots": h5p_hotspots,
            "iconType": "icon",
            "icon": "plus",
            "color": "#981d99",
        }

        return h5p_params
