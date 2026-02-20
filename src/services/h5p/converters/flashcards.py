# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flashcards H5P Converter.

Converts AI-generated flashcard content to H5P.Flashcards format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class FlashcardsConverter(BaseH5PConverter):
    """Converter for H5P.Flashcards content type.

    AI Input Format (from config ai_input_format):
        [
            {"front": "Term", "back": "Definition", "tip": "optional hint"},
            ...
        ]

    Also accepts legacy format:
        {"cards": [{"term": "...", "definition": "...", "example": "..."}]}
    """

    @property
    def content_type(self) -> str:
        return "flashcards"

    @property
    def library(self) -> str:
        return "H5P.Flashcards 1.7"

    @property
    def category(self) -> str:
        return "vocabulary"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        cards = ai_content.get("cards", [])
        if not cards:
            raise H5PValidationError(
                message="Missing or empty 'cards' field",
                content_type=self.content_type,
            )

        for i, card in enumerate(cards):
            # Accept both front/back and term/definition
            has_front = "front" in card or "term" in card
            has_back = "back" in card or "definition" in card
            if not has_front:
                raise H5PValidationError(
                    message=f"Card {i+1} missing 'front' (or 'term') field",
                    content_type=self.content_type,
                )
            if not has_back:
                raise H5PValidationError(
                    message=f"Card {i+1} missing 'back' (or 'definition') field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Flashcards format."""
        cards_input = ai_content.get("cards", [])
        description = ai_content.get("description", ai_content.get("title", ""))

        if not cards_input:
            raise H5PValidationError(
                message="No cards provided",
                content_type=self.content_type,
            )

        # Build cards array
        cards = []
        for card in cards_input:
            # Accept both front/back and term/definition
            term = card.get("front", card.get("term", ""))
            definition = card.get("back", card.get("definition", ""))
            example = card.get("example", "")
            tip_text = card.get("tip", "")

            # Combine definition and example if present
            if example:
                answer = f"{definition}\n\n<em>Example: {example}</em>"
            else:
                answer = definition

            h5p_card: dict[str, Any] = {
                "text": term,
                "answer": answer,
            }

            # Image (only if provided)
            image_url = card.get("image_url", "")
            if image_url:
                h5p_card["image"] = {
                    "path": image_url,
                    "alt": card.get("imageAltText", term),
                }
                h5p_card["imageAltText"] = card.get("imageAltText", term)

            # Tip
            if tip_text:
                h5p_card["tip"] = tip_text

            cards.append(h5p_card)

        l10n = self.get_l10n(language)

        h5p_params = {
            "description": self.wrap_html(description) if description else "",
            "cards": cards,
            "progressText": l10n["progressText"],
            "next": l10n["next"],
            "previous": l10n["previous"],
            "checkAnswerText": l10n["checkAnswerText"],
            "showSolutionsRequiresInput": True,
            "defaultAnswerText": l10n["defaultAnswerText"],
            "correctAnswerText": l10n["correctAnswerText"],
            "incorrectAnswerText": l10n["incorrectAnswerText"],
            "showSolutionText": l10n["showSolutionText"],
            "results": l10n["results"],
            "ofCorrect": l10n["ofCorrect"],
            "showResults": l10n["showResults"],
            "answerShortText": l10n["answerShortText"],
            "retry": l10n["retry"],
            "caseSensitive": False,
            "randomCards": False,
            # Accessibility
            "cardAnnouncement": l10n["cardAnnouncement"],
            "correctAnswerAnnouncement": l10n["correctAnswerAnnouncement"],
            "pageAnnouncement": l10n["pageAnnouncement"],
        }

        return h5p_params

