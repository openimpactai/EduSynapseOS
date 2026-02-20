# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Word Search H5P Converter.

Converts AI-generated word search puzzles to H5P.FindTheWords format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class WordSearchConverter(BaseH5PConverter):
    """Converter for H5P.FindTheWords content type.

    AI Input Format:
        {
            "title": "Plant Vocabulary",
            "description": "Find all the plant-related words",
            "words": ["photosynthesis", "chlorophyll", "glucose", "oxygen", "carbon"],
            "settings": {
                "gridSize": 10,
                "fillBlanks": true,
                "showVocabulary": true
            }
        }

    Words are placed in a grid and user finds them by clicking/dragging.
    """

    @property
    def content_type(self) -> str:
        return "word-search"

    @property
    def library(self) -> str:
        return "H5P.FindTheWords 1.5"

    @property
    def category(self) -> str:
        return "vocabulary"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember"]

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
                message="At least 2 words are required for word search",
                content_type=self.content_type,
            )

        for i, word in enumerate(words):
            if isinstance(word, str):
                if len(word.strip()) < 2:
                    raise H5PValidationError(
                        message=f"Word {i+1} must be at least 2 characters",
                        content_type=self.content_type,
                    )
            elif isinstance(word, dict):
                if "word" not in word:
                    raise H5PValidationError(
                        message=f"Word {i+1} missing 'word' field",
                        content_type=self.content_type,
                    )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P FindTheWords format."""
        words = ai_content.get("words", [])
        description = ai_content.get("description", "")
        settings = ai_content.get("settings", {})

        if not words:
            raise H5PValidationError(
                message="No words provided",
                content_type=self.content_type,
            )

        # Convert words to H5P format
        # Words can be strings or objects with word/hint
        h5p_vocabulary = []
        for word in words:
            if isinstance(word, str):
                h5p_vocabulary.append({
                    "word": word.upper().strip(),
                })
            elif isinstance(word, dict):
                word_text = word.get("word", "").upper().strip()
                hint = word.get("hint", "")
                h5p_word = {"word": word_text}
                if hint:
                    h5p_word["hint"] = hint
                h5p_vocabulary.append(h5p_word)

        h5p_params = {
            "taskDescription": f"<p>{description}</p>" if description else "",
            "wordList": h5p_vocabulary,
            "behaviour": self.get_default_behavior(settings),
            "l10n": self.get_l10n(language),
            "overallFeedback": self.get_overall_feedback(language),
        }

        return h5p_params

    def get_default_behavior(self, settings: dict[str, Any] = None) -> dict[str, Any]:
        """Get default behavior for FindTheWords."""
        settings = settings or {}
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableShowSolution": True,
            "showVocabulary": settings.get("showVocabulary", True),
            "gridSize": settings.get("gridSize", 10),
            "fillBlanks": settings.get("fillBlanks", True),
            "allowOrientations": settings.get("orientations", [
                "horizontal",
                "vertical",
                "diagonal",
            ]),
        }

