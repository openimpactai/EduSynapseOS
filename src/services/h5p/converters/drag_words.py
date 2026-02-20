# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Drag the Words H5P Converter.

Converts AI-generated drag-the-words exercises to H5P.DragText format.
"""

from typing import Any

from src.services.h5p.converters.base import BaseH5PConverter
from src.services.h5p.exceptions import H5PValidationError


class DragWordsConverter(BaseH5PConverter):
    """Converter for H5P.DragText content type.

    AI Input Format:
        {
            "title": "Exercise Title",
            "instruction": "Drag the words into correct positions",
            "exercises": [
                {
                    "text": "The *nucleus* contains *DNA* and controls the cell.",
                    "draggables": ["nucleus", "DNA"]
                }
            ]
        }

    The * markers indicate draggable word positions.
    """

    @property
    def content_type(self) -> str:
        return "drag-words"

    @property
    def library(self) -> str:
        return "H5P.DragText 1.10"

    @property
    def category(self) -> str:
        return "assessment"

    @property
    def bloom_levels(self) -> list[str]:
        return ["apply", "analyze"]

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
            if "*" not in e.get("text", ""):
                raise H5PValidationError(
                    message=f"Exercise {i+1} text must contain *draggable* markers",
                    content_type=self.content_type,
                )

        return True

    def convert(
        self,
        ai_content: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        """Convert AI content to H5P DragText format."""
        exercises = ai_content.get("exercises", [])
        l10n = self.get_l10n(language)
        instruction = ai_content.get("instruction", l10n["defaultInstruction"])

        if not exercises:
            raise H5PValidationError(
                message="No exercises provided",
                content_type=self.content_type,
            )

        # Combine exercises
        combined_text_parts = []
        for exercise in exercises:
            text = exercise.get("text", "")
            combined_text_parts.append(self.wrap_html(text))

        combined_text = "\n".join(combined_text_parts)

        h5p_params = {
            "taskDescription": self.wrap_html(instruction),
            "textField": combined_text,
            "overallFeedback": self.get_overall_feedback(language),
            "behaviour": self.get_default_behavior(),
            "checkAnswer": l10n["checkAnswer"],
            "submitAnswer": l10n["submitAnswer"],
            "tryAgain": l10n["tryAgain"],
            "showSolution": l10n["showSolution"],
            "dropZoneIndex": l10n["dropZoneLabel"],
            "empty": l10n["emptyLabel"],
            "contains": l10n["containsLabel"],
            "ariaDraggableIndex": l10n["ariaDraggableLabel"],
            "tipLabel": l10n["tipLabel"],
            "correctText": l10n["correct"],
            "incorrectText": l10n["incorrect"],
            "resetDropTitle": l10n["resetDropTitle"],
            "resetDropDescription": l10n["resetDropDescription"],
            "grabbed": l10n["grabbedLabel"],
            "cancelledDragging": l10n["cancelledDraggingLabel"],
            "correctAnswer": l10n["correctAnswerLabel"],
            "feedbackHeader": l10n["feedbackHeader"],
            "scoreBarLabel": l10n["scoreBarLabel"],
        }

        return h5p_params

    def get_default_behavior(self) -> dict[str, Any]:
        """Get default behavior for DragText."""
        return {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "enableCheckButton": True,
            "instantFeedback": False,
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
