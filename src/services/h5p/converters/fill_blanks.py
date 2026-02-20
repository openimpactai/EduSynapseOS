# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fill in the Blanks H5P Converter.

Converts AI-generated fill-in-the-blank exercises to H5P.Blanks format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class FillBlanksConverter(BaseH5PConverter):
    """Converter for H5P.Blanks content type.

    AI Input Format:
        {
            "title": "Exercise Title",
            "exercises": [
                {
                    "text": "Plants use *sunlight* and *water* to grow.",
                    "blanks": ["sunlight", "water"],
                    "hint": "Think about what plants need"
                }
            ]
        }

    The * markers in text indicate blank positions.
    Alternatives can be specified with / syntax: *word/alternative*
    """

    @property
    def content_type(self) -> str:
        return "fill-blanks"

    @property
    def library(self) -> str:
        return "H5P.Blanks 1.14"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["apply"]

    def validate_ai_content(self, ai_content: dict[str, Any]) -> bool:
        """Validate AI content has required fields."""
        super().validate_ai_content(ai_content)

        if "exercises" not in ai_content:
            raise H5PValidationError(
                message="Missing 'exercises' field",
                content_type=self.content_type,
            )

        for i, e in enumerate(ai_content.get("exercises", [])):
            if "text" not in e:
                raise H5PValidationError(
                    message=f"Exercise {i+1} missing 'text' field",
                    content_type=self.content_type,
                )
            # Verify text contains blanks (marked with *)
            if "*" not in e.get("text", ""):
                raise H5PValidationError(
                    message=f"Exercise {i+1} text must contain *blank* markers",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P Blanks format."""
        exercises = ai_content.get("exercises", [])

        if not exercises:
            raise H5PValidationError(
                message="No exercises provided",
                content_type=self.content_type,
            )

        # Build questions list — H5P.Blanks semantics:
        #   "text" = task description (instruction)
        #   "questions" = list of text blocks with *blank* markers
        questions_list = []
        for exercise in exercises:
            text = exercise.get("text", "")
            questions_list.append(self.wrap_html(text))

        l10n = self.get_l10n(language)

        h5p_params = {
            "text": l10n["taskDescription"],
            "questions": questions_list,
            "overallFeedback": self.get_overall_feedback(language),
            "behaviour": self.get_default_behavior(),
            "showSolutions": l10n["showSolutions"],
            "tryAgain": l10n["tryAgain"],
            "checkAnswer": l10n["checkAnswer"],
            "submitAnswer": l10n["submitAnswer"],
            "notFilledOut": l10n["notFilledOut"],
            "answerIsCorrect": l10n["correct"],
            "answerIsWrong": l10n["wrong"],
            "answeredCorrectly": l10n["answeredCorrectly"],
            "answeredIncorrectly": l10n["answeredIncorrectly"],
            "solutionLabel": l10n["solutionLabel"],
            "inputLabel": l10n["inputLabel"],
            "inputHasTipLabel": l10n["inputHasTip"],
            "tipLabel": l10n["tipLabel"],
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for Blanks."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "caseSensitive": False,
            "showSolutionsRequiresInput": True,
            "autoCheck": False,
            "acceptSpellingErrors": False,
            "separateLines": False,
        }

    def convert_single_params(
        self,
        params: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert a single exercise's params to H5P format.

        Used by QuestionSetConverter to delegate conversion.
        Accepts: {text, ...}
        """
        return self.convert({"exercises": [params]}, language)
