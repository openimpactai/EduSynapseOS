# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Accordion H5P Converter.

Converts AI-generated FAQ/accordion content to H5P.Accordion format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class AccordionConverter(BaseH5PConverter):
    """Converter for H5P.Accordion content type.

    AI Input Format:
        {
            "title": "Frequently Asked Questions",
            "panels": [
                {
                    "title": "What is photosynthesis?",
                    "content": "Photosynthesis is the process by which plants convert sunlight into energy..."
                },
                {
                    "title": "Why are leaves green?",
                    "content": "Leaves are green because they contain chlorophyll..."
                }
            ],
            "settings": {
                "expandAll": false,
                "collapseAll": true
            }
        }

    Content panels that expand/collapse when clicked.
    """

    @property
    def content_type(self) -> str:
        return "accordion"

    @property
    def library(self) -> str:
        return "H5P.Accordion 1.0"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember", "understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "panels" not in ai_content:
            raise H5PValidationError(
                message="Missing 'panels' field",
                content_type=self.content_type,
            )

        panels = ai_content.get("panels", [])
        if not panels:
            raise H5PValidationError(
                message="At least one panel is required",
                content_type=self.content_type,
            )

        for i, panel in enumerate(panels):
            if "title" not in panel:
                raise H5PValidationError(
                    message=f"Panel {i+1} missing 'title' field",
                    content_type=self.content_type,
                )
            if "content" not in panel:
                raise H5PValidationError(
                    message=f"Panel {i+1} missing 'content' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Accordion format."""
        panels = ai_content.get("panels", [])
        settings = ai_content.get("settings", {})

        if not panels:
            raise H5PValidationError(
                message="No panels provided",
                content_type=self.content_type,
            )

        # Convert panels to H5P format
        h5p_panels = []
        for panel in panels:
            title = panel.get("title", "")
            content = panel.get("content", "")

            # Wrap content in HTML if not already
            if not content.startswith("<"):
                content = f"<p>{content}</p>"

            h5p_panels.append({
                "title": title,
                "content": {
                    "params": {
                        "text": content,
                    },
                    "library": "H5P.AdvancedText 1.1",
                    "subContentId": self._generate_subcontent_id(),
                },
            })

        h5p_params = {
            "panels": h5p_panels,
            "hTag": "h3",
        }

        # Add optional settings
        if settings.get("expandAll"):
            h5p_params["expandAll"] = True
        if settings.get("collapseAll"):
            h5p_params["collapseAll"] = True

        return h5p_params

    def _generate_subcontent_id(self) -> str:
        """Generate a unique subcontent ID."""
        import uuid
        return str(uuid.uuid4())
