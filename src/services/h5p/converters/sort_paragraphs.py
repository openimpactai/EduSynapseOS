# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sort Paragraphs H5P Converter.

Converts AI-generated sort paragraphs content to H5P.SortParagraphs format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class SortParagraphsConverter(BaseH5PConverter):
    """Converter for H5P.SortParagraphs content type.

    AI Input Format:
        {
            "title": "Order the Steps",
            "description": "Put the steps of the water cycle in order",
            "paragraphs": [
                "Water evaporates from oceans, lakes, and rivers.",
                "Water vapor rises into the atmosphere.",
                "Water vapor cools and condenses into clouds.",
                "Precipitation falls as rain, snow, or hail.",
                "Water collects in bodies of water and the cycle repeats."
            ]
        }

    Paragraphs are provided in correct order, H5P shuffles them for the user.
    """

    @property
    def content_type(self) -> str:
        return "sort-paragraphs"

    @property
    def library(self) -> str:
        return "H5P.SortParagraphs 0.11"

    @property
    def category(self) -> str:
        return "learning"

    @property
    def bloom_levels(self) -> list[str]:
        return ["analyze"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "paragraphs" not in ai_content:
            raise H5PValidationError(
                message="Missing 'paragraphs' field",
                content_type=self.content_type,
            )

        paragraphs = ai_content.get("paragraphs", [])
        if len(paragraphs) < 2:
            raise H5PValidationError(
                message="At least 2 paragraphs are required",
                content_type=self.content_type,
            )

        for i, p in enumerate(paragraphs):
            if not isinstance(p, str) or not p.strip():
                raise H5PValidationError(
                    message=f"Paragraph {i+1} must be a non-empty string",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P SortParagraphs format."""
        paragraphs = ai_content.get("paragraphs", [])
        description = ai_content.get("description", self.get_l10n(language).get("defaultDescription", "Put the paragraphs in the correct order"))

        if not paragraphs:
            raise H5PValidationError(
                message="No paragraphs provided",
                content_type=self.content_type,
            )

        # H5P expects paragraphs in correct order (it will shuffle for display)
        h5p_paragraphs = []
        for p in paragraphs:
            h5p_paragraphs.append({
                "text": f"<p>{p}</p>",
            })

        h5p_params = {
            "taskDescription": f"<p>{description}</p>",
            "paragraphs": h5p_paragraphs,
            "behaviour": self.get_default_behavior(),
            "l10n": self.get_l10n(language).get("l10n", {}),
            "a11y": self.get_l10n(language).get("a11y", {}),
            "overallFeedback": self.get_overall_feedback(language),
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for SortParagraphs."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "scoringMode": "positions",  # positions or transitions
            "applyPenalties": False,
            "duplicatesInterchangeable": True,
        }

