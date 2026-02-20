# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Crossword H5P Converter.

Converts AI-generated crossword puzzles to H5P.Crossword format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class CrosswordConverter(BaseH5PConverter):
    """Converter for H5P.Crossword content type.

    AI Input Format:
        {
            "title": "Science Crossword",
            "description": "Find the science vocabulary words",
            "words": [
                {
                    "word": "PHOTOSYNTHESIS",
                    "clue": "Process plants use to make food from sunlight"
                },
                {
                    "word": "CHLOROPHYLL",
                    "clue": "Green pigment that captures light energy"
                }
            ],
            "settings": {
                "poolSize": 10,
                "backgroundColor": "#173354"
            }
        }

    Words are placed automatically by H5P based on letter intersections.
    """

    @property
    def content_type(self) -> str:
        return "crossword"

    @property
    def library(self) -> str:
        return "H5P.Crossword 0.5"

    @property
    def category(self) -> str:
        return "vocabulary"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember", "apply"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "words" not in ai_content:
            raise H5PValidationError(
                message="Missing 'words' field",
                content_type=self.content_type,
            )

        words = ai_content.get("words", [])
        if len(words) < 2:
            raise H5PValidationError(
                message="At least 2 words are required for a crossword",
                content_type=self.content_type,
            )

        for i, w in enumerate(words):
            if "word" not in w:
                raise H5PValidationError(
                    message=f"Word {i+1} missing 'word' field",
                    content_type=self.content_type,
                )
            if "clue" not in w:
                raise H5PValidationError(
                    message=f"Word {i+1} missing 'clue' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Crossword format."""
        words = ai_content.get("words", [])
        description = ai_content.get("description", "")
        settings = ai_content.get("settings", {})

        if not words:
            raise H5PValidationError(
                message="No words provided",
                content_type=self.content_type,
            )

        # Convert words to H5P format
        h5p_words = []
        for word in words:
            word_text = word.get("word", "").upper().strip()
            clue = word.get("clue", "")
            extra_clue = word.get("extraClue", "")

            h5p_word = {
                "answer": word_text,
                "clue": clue,
            }

            if extra_clue:
                h5p_word["extraClue"] = extra_clue

            h5p_words.append(h5p_word)

        h5p_params = {
            "words": h5p_words,
            "behaviour": self.get_default_behavior(settings),
            "theme": self._get_theme_settings(settings),
            "l10n": self.get_l10n(language).get("l10n", {}),
            "a11y": self.get_l10n(language).get("a11y", {}),
            "overallFeedback": self.get_overall_feedback(language),
        }

        if description:
            h5p_params["taskDescription"] = f"<p>{description}</p>"

        return h5p_params

    def get_default_behavior(self, settings: dict[str, Any] = None) -> dict[str, Any]:
        """Get default behavior for Crossword."""
        settings = settings or {}
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableInstantFeedback": False,
            "scoreWords": True,
            "applyPenalties": False,
            "poolSize": settings.get("poolSize", 0),  # 0 = use all words
            "randomize": settings.get("randomize", True),
        }

    def _get_theme_settings(self, settings: dict[str, Any] = None) -> dict[str, Any]:
        """Get theme settings."""
        settings = settings or {}
        return {
            "backgroundColor": settings.get("backgroundColor", "#173354"),
            "gridColor": settings.get("gridColor", "#000000"),
            "cellBackgroundColor": settings.get("cellBackgroundColor", "#ffffff"),
            "cellColor": settings.get("cellColor", "#000000"),
            "clueIdColor": settings.get("clueIdColor", "#606060"),
            "cellHighlightColor": settings.get("cellHighlightColor", "#CCEEFF"),
        }

