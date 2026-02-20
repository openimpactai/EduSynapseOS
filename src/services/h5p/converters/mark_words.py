# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mark the Words H5P Converter.

Converts AI-generated mark-the-words exercises to H5P.MarkTheWords format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class MarkWordsConverter(BaseH5PConverter):
    """Converter for H5P.MarkTheWords content type.

    AI Input Format:
        {
            "title": "Exercise Title",
            "instruction": "Click on all the verbs",
            "exercises": [
                {
                    "text": "The cat *runs* and *jumps* over the fence.",
                    "markables": ["runs", "jumps"],
                    "wordType": "verbs"
                }
            ]
        }

    The * markers in text indicate words that should be marked.
    """

    @property
    def content_type(self) -> str:
        return "mark-words"

    @property
    def library(self) -> str:
        return "H5P.MarkTheWords 1.11"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["analyze"]

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
            # Verify text contains markable words (marked with *)
            if "*" not in e.get("text", ""):
                raise H5PValidationError(
                    message=f"Exercise {i+1} text must contain *word* markers",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P MarkTheWords format."""
        exercises = ai_content.get("exercises", [])
        l10n = self.get_l10n(language)
        instruction = ai_content.get("instruction", l10n["defaultInstruction"])

        if not exercises:
            raise H5PValidationError(
                message="No exercises provided",
                content_type=self.content_type,
            )

        # Combine all exercises into one text block
        combined_text_parts = []

        for exercise in exercises:
            text = exercise.get("text", "")
            combined_text_parts.append(self.wrap_html(text))

        combined_text = "\n".join(combined_text_parts)

        h5p_params = {
            "taskDescription": f"<p>{instruction}</p>",
            "textField": combined_text,
            "overallFeedback": self.get_overall_feedback(language),
            "behaviour": self.get_default_behavior(),
            "checkAnswerButton": l10n["checkAnswer"],
            "tryAgainButton": l10n["tryAgain"],
            "showSolutionButton": l10n["showSolution"],
            "correctAnswer": l10n["correct"],
            "incorrectAnswer": l10n["wrong"],
            "missedAnswer": l10n["missed"],
            "displaySolutionDescription": l10n["solutionDescription"],
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for MarkTheWords."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "showScorePoints": True,
        }

    def convert_single_params(
        self,
        params: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert a single exercise's params to H5P format.

        Used by QuestionSetConverter to delegate conversion.
        Accepts: {text, instruction, ...}
        """
        ai_content: dict[str, Any] = {"exercises": [params]}
        if "instruction" in params:
            ai_content["instruction"] = params["instruction"]
        return self.convert(ai_content, language)
