# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dialog Cards H5P Converter.

Converts AI-generated dialog card content to H5P.Dialogcards format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class DialogCardsConverter(BaseH5PConverter):
    """Converter for H5P.Dialogcards content type.

    AI Input Format:
        {
            "title": "Spanish Phrases",
            "dialogs": [
                {
                    "front": "How do you say 'hello' in Spanish?",
                    "back": "Hola",
                    "tips": ["It's one of the most common greetings"]
                }
            ]
        }
    """

    @property
    def content_type(self) -> str:
        return "dialog-cards"

    @property
    def library(self) -> str:
        return "H5P.Dialogcards 1.9"

    @property
    def category(self) -> str:
        return "vocabulary"

    @property
    def bloom_levels(self) -> list[str]:
        return ["remember", "understand"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "dialogs" not in ai_content:
            raise H5PValidationError(
                message="Missing 'dialogs' field",
                content_type=self.content_type,
            )

        for i, dialog in enumerate(ai_content.get("dialogs", [])):
            if "front" not in dialog:
                raise H5PValidationError(
                    message=f"Dialog {i+1} missing 'front' field",
                    content_type=self.content_type,
                )
            if "back" not in dialog:
                raise H5PValidationError(
                    message=f"Dialog {i+1} missing 'back' field",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Dialogcards format."""
        dialogs_input = ai_content.get("dialogs", [])
        title = ai_content.get("title", "")

        if not dialogs_input:
            raise H5PValidationError(
                message="No dialogs provided",
                content_type=self.content_type,
            )

        # Build dialogs array
        dialogs = []
        for dialog in dialogs_input:
            front = dialog.get("front", "")
            back = dialog.get("back", "")
            tips = dialog.get("tips", [])

            h5p_dialog = {
                "text": self.wrap_html(front),
                "answer": self.wrap_html(back),
                "tips": tips if tips else [],
                "image": {},
                "audio": {},
            }

            dialogs.append(h5p_dialog)

        l10n = self.get_l10n(language)

        h5p_params = {
            "dialogs": dialogs,
            "title": self.wrap_html(title),
            "mode": "normal",
            "behaviour": self.get_default_behavior(),
            "answer": l10n["answer"],
            "next": l10n["next"],
            "prev": l10n["prev"],
            "retry": l10n["retry"],
            "correctAnswer": l10n["correctAnswer"],
            "incorrectAnswer": l10n["incorrectAnswer"],
            "round": l10n["round"],
            "cardsLeft": l10n["cardsLeft"],
            "nextRound": l10n["nextRound"],
            "startOver": l10n["startOver"],
            "showSummary": l10n["showSummary"],
            "summary": l10n["summary"],
            "summaryCardsRight": l10n["summaryRight"],
            "summaryCardsWrong": l10n["summaryWrong"],
            "summaryCardsNotShown": l10n["summaryNotShown"],
            "summaryOverallScore": l10n["summaryOverall"],
            "summaryCardsCompleted": l10n["summaryCompleted"],
            "summaryCompletedRounds": l10n["summaryRounds"],
            "summaryAllDone": l10n["summaryAllDone"],
            "progressText": l10n["progressText"],
            "cardFrontLabel": l10n["cardFront"],
            "cardBackLabel": l10n["cardBack"],
            "tipButtonLabel": l10n["tipButton"],
            "audioNotSupported": l10n["audioNotSupported"],
            "confirmStartingOver": {
                "header": l10n["confirmHeader"],
                "body": l10n["confirmBody"],
                "cancelLabel": l10n["cancelLabel"],
                "confirmLabel": l10n["confirmConfirmLabel"],
            },
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for Dialogcards."""
        return {
            "enableRetry": True,
            "disableBackwardsNavigation": False,
            "scaleTextNotCard": False,
            "randomCards": False,
            "maxProficiency": 5,
            "quickProgression": False,
        }

